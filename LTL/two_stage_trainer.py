"""
Two-Stage Training Framework for Long-Tailed Learning

Overview (Pseudo Code):
Stage 1 - Representation Learning:
1. Train full ResPointNet++ end-to-end on original long-tailed data
2. Use standard cross-entropy or the strong CB loss
3. Learn rich geometric features from all available data
4. Save trained model

Stage 2 - Classifier Re-training:
1. Load Stage 1 model
2. Freeze encoder layers (backbone)
3. Create class-balanced training data
4. Fine-tune only decoder/classifier layers
5. Adjust decision boundaries without corrupting representations

This framework decouples representation learning from classification.
"""

import torch
import torch.nn as nn
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from models.losses.cb_loss import create_cb_loss
from LTL.cb_sampler import create_balanced_dataloader


class TwoStageTrainer:
    """
    Two-Stage Training Framework for ResPointNet-LT.
    
    Implements the decoupled training strategy where representation learning
    is separated from classifier training to handle class imbalance.
    """
    
    def __init__(self, config, model, criterion, optimizer, scheduler, 
                 train_loader, val_loader, logger):
        """
        Initialize two-stage trainer.
        
        Args:
            config: Training configuration
            model: ResPointNet++ model
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            train_loader: Training data loader
            val_loader: Validation data loader
            logger: Logger object
        """
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        
        # Two-stage specific configurations
        self.stage1_epochs = getattr(config, 'stage1_epochs', config.epochs)
        self.stage2_epochs = getattr(config, 'stage2_epochs', config.epochs // 4)
        # FIX #1: Dramatically reduce Stage 2 LR for stability (0.05 = 1/20 of base LR instead of 1/5)
        self.stage2_lr = getattr(config, 'stage2_lr', config.base_learning_rate * 0.05)
        self.use_cb_loss_stage1 = getattr(config, 'use_cb_loss_stage1', False)
        self.use_cb_loss_stage2 = getattr(config, 'use_cb_loss_stage2', False)  # Issue #3: Changed default from True to False (CE loss)
        self.freeze_encoder = getattr(config, 'freeze_encoder', True)
        # FIX #2: Configurable warmup epochs (default to longer warmup)
        self.stage2_warmup_epochs = getattr(config, 'stage2_warmup_epochs', None)  # Will be set dynamically if None
        
        # Stage tracking
        self.current_stage = 1
        self.stage1_completed = False
        
        # Create stage-specific directories
        self.stage1_dir = os.path.join(config.log_dir, 'stage1')
        self.stage2_dir = os.path.join(config.log_dir, 'stage2')
        self.plots_dir = os.path.join(config.log_dir, 'plots')
        os.makedirs(self.stage1_dir, exist_ok=True)
        os.makedirs(self.stage2_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Training metrics tracking
        self.stage1_metrics = {
            'train_loss': [],
            'val_loss': [],
            'miou': [],
            'val_epochs': [],
            'per_class_iou': [],
            'epochs': [],
            'time_per_epoch': [],
            'learning_rates': []
        }
        self.stage2_metrics = {
            'train_loss': [],
            'val_loss': [],
            'miou': [],
            'val_epochs': [],
            'per_class_iou': [],
            'epochs': [],
            'time_per_epoch': [],
            'learning_rates': []
        }
        
        # Training time tracking
        self.stage1_start_time = None
        self.stage1_end_time = None
        self.stage2_start_time = None
        self.stage2_end_time = None
        
        # Checkpoint source tracking
        self.checkpoint_source = None
        
        # Check if Stage 1 was already completed (for resuming)
        stage1_best_path = os.path.join(self.stage1_dir, 'stage1_best.pth')
        if os.path.exists(stage1_best_path):
            self.stage1_completed = True
            self.checkpoint_source = stage1_best_path
            self.logger.info(f"Stage 1 checkpoint found at {stage1_best_path}, will skip Stage 1")
    
    def train_stage1(self, validate_func):
        """
        Stage 1: Representation Learning
        
        Train the full network end-to-end on the original long-tailed dataset.
        Focus on learning rich geometric representations.
        """
        self.stage1_start_time = datetime.now()
        self.logger.info("=" * 50)
        self.logger.info("Starting Stage 1: Representation Learning")
        self.logger.info(f"Start Time: {self.stage1_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Epochs: {self.stage1_epochs}")
        self.logger.info(f"Use CB Loss: {self.use_cb_loss_stage1}")
        self.logger.info("=" * 50)
        
        # Setup Stage 1 criterion
        if self.use_cb_loss_stage1:
            stage1_criterion = create_cb_loss(
                self.train_loader.dataset, 
                self.config.num_classes,
                beta=getattr(self.config, 'cb_beta', 0.9999),
                gamma=getattr(self.config, 'cb_gamma', 2.0),
                alpha=getattr(self.config, 'cb_alpha', 1.0)
            )
            stage1_criterion.cuda()
            self.logger.info("Using Class-Balanced Loss for Stage 1")
        else:
            stage1_criterion = self.criterion
            self.logger.info("Using standard MaskedCrossEntropy for Stage 1")
        
        # Setup tensorboard for Stage 1
        if dist.get_rank() == 0:
            stage1_writer = SummaryWriter(log_dir=self.stage1_dir)
        else:
            stage1_writer = None
        
        # Training loop for Stage 1
        best_miou = 0.0
        for epoch in range(1, self.stage1_epochs + 1):
            self.train_loader.sampler.set_epoch(epoch)
            self.val_loader.sampler.set_epoch(epoch)
            self.train_loader.dataset.epoch = epoch - 1
            
            tic = time.time()
            
            # Train one epoch
            loss = self._train_epoch(epoch, stage1_criterion, stage='stage1')
            epoch_time = time.time() - tic
            
            self.logger.info(f'Stage1 Epoch {epoch}/{self.stage1_epochs}, '
                           f'Time {epoch_time:.2f}s, '
                           f'LR {self.optimizer.param_groups[0]["lr"]:.5f}, '
                           f'Loss {loss:.4f}')
            
            # Track metrics
            self.stage1_metrics['epochs'].append(epoch)
            self.stage1_metrics['train_loss'].append(loss)
            self.stage1_metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.stage1_metrics['time_per_epoch'].append(epoch_time)
            
            # Validation
            if epoch % self.config.val_freq == 0:
                miou = validate_func(f'S1-E{epoch}', self.val_loader, self.model, 
                                   stage1_criterion, num_votes=2)
                self.stage1_metrics['miou'].append(miou)
                self.stage1_metrics['val_epochs'].append(epoch)
                
                # Save best model
                if miou > best_miou:
                    best_miou = miou
                    if dist.get_rank() == 0:
                        self._save_checkpoint(epoch, 'stage1_best.pth')
            
            # Save checkpoint
            if dist.get_rank() == 0 and epoch % self.config.save_freq == 0:
                self._save_checkpoint(epoch, f'stage1_epoch_{epoch}.pth')
            
            # Tensorboard logging
            if stage1_writer is not None:
                stage1_writer.add_scalar('train_loss', loss, epoch)
                stage1_writer.add_scalar('learning_rate', 
                                       self.optimizer.param_groups[0]['lr'], epoch)
                if len(self.stage1_metrics['miou']) > 0:
                    stage1_writer.add_scalar('miou', self.stage1_metrics['miou'][-1], epoch)
        
        # Save final Stage 1 model
        if dist.get_rank() == 0:
            self._save_checkpoint(self.stage1_epochs, 'stage1_final.pth')
            if stage1_writer is not None:
                stage1_writer.close()
        
        self.stage1_completed = True
        self.stage1_end_time = datetime.now()
        duration = (self.stage1_end_time - self.stage1_start_time).total_seconds() / 3600
        self.logger.info(f"Stage 1 completed. Best mIoU: {best_miou:.3%}")
        self.logger.info(f"Stage 1 End Time: {self.stage1_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Stage 1 Duration: {duration:.2f} hours")
    
    def train_stage2(self, validate_func):
        """
        Stage 2: Classifier Re-training
        
        Freeze encoder and fine-tune classifier on class-balanced data.
        """
        if not self.stage1_completed:
            raise RuntimeError("Stage 1 must be completed before Stage 2")
        
        self.stage2_start_time = datetime.now()
        self.logger.info("=" * 50)
        self.logger.info("Starting Stage 2: Classifier Re-training")
        self.logger.info(f"Start Time: {self.stage2_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Epochs: {self.stage2_epochs}")
        self.logger.info(f"Freeze Encoder: {self.freeze_encoder}")
        self.logger.info(f"Use CB Loss: {self.use_cb_loss_stage2}")
        self.logger.info("=" * 50)
        
        # Load best Stage 1 model
        stage1_best_path = os.path.join(self.stage1_dir, 'stage1_best.pth')
        if os.path.exists(stage1_best_path):
            self._load_checkpoint(stage1_best_path)
            self.checkpoint_source = stage1_best_path
            self.logger.info(f"✓ Loaded Stage 1 best model from: {stage1_best_path}")
            
            # CRITICAL: Validate the loaded checkpoint to establish true baseline
            self.logger.info("Validating loaded Stage 1 checkpoint on validation set...")
            stage1_baseline = validate_func('S1-Loaded', self.val_loader, self.model, 
                                          self.criterion, num_votes=2)
            self.logger.info(f"✓ Stage 1 loaded checkpoint baseline mIoU: {stage1_baseline:.3%}")
            self.stage1_metrics['miou'].append(stage1_baseline)
            self.stage1_metrics['val_epochs'].append(0)  # Mark as pre-Stage-2 validation
        else:
            self.logger.warning("Stage 1 best model not found, using current model")
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            self._freeze_encoder()
            self.logger.info("Encoder layers frozen")
        
        # Setup Stage 2 criterion
        # Issue #3: Both CE and CB loss are supported, but CE is default
        # Rationale: With balanced sampling, CE loss is generally better since
        # the data imbalance is already addressed by the sampler
        if self.use_cb_loss_stage2:
            stage2_criterion = create_cb_loss(
                self.train_loader.dataset, 
                self.config.num_classes,
                beta=getattr(self.config, 'cb_beta', 0.9999),
                gamma=getattr(self.config, 'cb_gamma', 2.0),
                alpha=getattr(self.config, 'cb_alpha', 1.0)
            )
            stage2_criterion.cuda()
            self.logger.info("Using Class-Balanced Loss for Stage 2 (with balanced sampling)")
            self.logger.info("  → Experimental: CB loss + balanced sampling may have complementary effects")
        else:
            stage2_criterion = self.criterion
            self.logger.info("Using standard MaskedCrossEntropy for Stage 2 (DEFAULT with balanced sampling)")
            self.logger.info("  → Recommended: CE loss provides cleaner optimization when sampling handles class balance")
        
        # Create class-balanced dataloader for Stage 2
        # This is the CORE of decoupled training - using balanced sampling for classifier fine-tuning
        try:
            # Check if we're in distributed mode
            is_distributed = dist.is_available() and dist.is_initialized()
            
            if is_distributed:
                self.logger.info(f"Creating balanced dataloader for distributed training (world_size={dist.get_world_size()}, rank={dist.get_rank()})")
            
            # FIX #3: Add temperature scaling to smooth the sampling distribution
            # Temperature > 1.0 makes distribution less peaked (more conservative)
            # Default 2.0 provides good balance between helping tail classes and maintaining head class performance
            sampling_temperature = getattr(self.config, 'sampling_temperature', 2.0)
            
            balanced_train_loader = create_balanced_dataloader(
                self.train_loader.dataset,
                self.config.batch_size,
                num_workers=self.config.num_workers,
                sampler_type='point',
                distributed=is_distributed,
                logger=self.logger,
                temperature=sampling_temperature
            )
            self.logger.info("✓ Created class-balanced dataloader for Stage 2 - implementing decoupled training theory!")
            self.logger.info(f"  → Sampling temperature: {sampling_temperature:.1f} (higher = more conservative)")
            self.logger.info("  → Spheres with tail classes will be preferentially sampled")
            
            # Log sampler statistics (only on rank 0 to avoid duplicate logs)
            if not is_distributed or dist.get_rank() == 0:
                if hasattr(balanced_train_loader.sampler, 'get_epoch_stats'):
                    # Will be populated after first epoch
                    pass
        except Exception as e:
            self.logger.error(f"Failed to create balanced dataloader: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.warning("Falling back to standard dataloader - this reduces effectiveness!")
            balanced_train_loader = self.train_loader
        
        # Setup new optimizer for Stage 2 with lower learning rate
        stage2_optimizer = self._create_stage2_optimizer()
        
        # Setup tensorboard for Stage 2
        if dist.get_rank() == 0:
            stage2_writer = SummaryWriter(log_dir=self.stage2_dir)
        else:
            stage2_writer = None
        
        # Training loop for Stage 2
        # Initialize best_miou with Stage 1 baseline if available
        if len(self.stage1_metrics['miou']) > 0:
            best_miou = max(self.stage1_metrics['miou'])
            self.logger.info(f"Starting Stage 2 with Stage 1 baseline: {best_miou:.3%}")
        else:
            best_miou = 0.0
        self.current_stage = 2
        
        # FIX #2: Define longer warmup_epochs for stability
        # Default to 10 epochs or 50% of Stage 2 training (whichever is smaller)
        if self.stage2_warmup_epochs is None:
            warmup_epochs = min(10, max(5, self.stage2_epochs // 2))
        else:
            warmup_epochs = self.stage2_warmup_epochs
        self.logger.info(f"Stage 2 warmup: {warmup_epochs} epochs (out of {self.stage2_epochs} total)")
        
        for epoch in range(1, self.stage2_epochs + 1):
            balanced_train_loader.sampler.set_epoch(epoch)
            self.val_loader.sampler.set_epoch(epoch)
            balanced_train_loader.dataset.epoch = epoch - 1
            
            tic = time.time()
            
            # Train one epoch with balanced data
            loss = self._train_epoch_stage2(epoch, balanced_train_loader, 
                                          stage2_criterion, stage2_optimizer)
            epoch_time = time.time() - tic
            
            self.logger.info(f'Stage2 Epoch {epoch}/{self.stage2_epochs}, '
                           f'Time {epoch_time:.2f}s, '
                           f'LR {stage2_optimizer.param_groups[0]["lr"]:.5f}, '
                           f'Loss {loss:.4f}')
            
            # Track metrics
            self.stage2_metrics['epochs'].append(epoch)
            self.stage2_metrics['train_loss'].append(loss)
            self.stage2_metrics['learning_rates'].append(stage2_optimizer.param_groups[0]['lr'])
            self.stage2_metrics['time_per_epoch'].append(epoch_time)
            
            # Log class-balanced sampler statistics (only on rank 0, and only every 5 epochs to reduce log spam)
            if (epoch == 1 or epoch % 5 == 0) and (not dist.is_initialized() or dist.get_rank() == 0):
                if hasattr(balanced_train_loader.sampler, 'get_epoch_stats'):
                    self.logger.info("\n" + "="*70)
                    self.logger.info(f"Stage 2 Epoch {epoch} - Class-Balanced Sampling Statistics")
                    self.logger.info("="*70)
                    stats = balanced_train_loader.sampler.get_epoch_stats()
                    self.logger.info(stats)
                    self.logger.info("="*70 + "\n")
            
            # Validation
            if epoch % self.config.val_freq == 0:
                miou = validate_func(f'S2-E{epoch}', self.val_loader, self.model, 
                                   stage2_criterion, num_votes=2)
                self.stage2_metrics['miou'].append(miou)
                self.stage2_metrics['val_epochs'].append(epoch)
                
                # Check for catastrophic performance collapse
                stage1_baseline = self.stage1_metrics['miou'][0] if len(self.stage1_metrics['miou']) > 0 else 0.0
                if stage1_baseline > 0.1 and epoch > warmup_epochs and miou < (stage1_baseline - 0.05):
                    self.logger.warning(f"⚠️  Performance dropping! Stage 1: {stage1_baseline:.3%}, Current: {miou:.3%}")
                    self.logger.warning(f"⚠️  Drop of {(stage1_baseline - miou):.3%} detected. This suggests training instability.")
                    if miou < (stage1_baseline - 0.10):
                        self.logger.error(f"❌ CRITICAL: mIoU dropped >10% below Stage 1 baseline!")
                        self.logger.error(f"❌ Stopping early to prevent further degradation.")
                        self.logger.error(f"❌ Consider: 1) Lower LR, 2) Longer warmup, 3) Check checkpoint loading")
                        # Save emergency checkpoint
                        if dist.get_rank() == 0:
                            self._save_checkpoint(epoch, 'stage2_emergency_stop.pth', stage2_optimizer)
                        break
                
                # Save best model
                if miou > best_miou:
                    best_miou = miou
                    self.logger.info(f"✓ New best mIoU: {best_miou:.3%} (epoch {epoch})")
                    if dist.get_rank() == 0:
                        self._save_checkpoint(epoch, 'stage2_best.pth', stage2_optimizer)
                elif miou < best_miou:
                    drop = best_miou - miou
                    self.logger.info(f"Current mIoU: {miou:.3%} (best: {best_miou:.3%}, diff: -{drop:.3%})")
            
            # Save checkpoint
            if dist.get_rank() == 0 and epoch % self.config.save_freq == 0:
                self._save_checkpoint(epoch, f'stage2_epoch_{epoch}.pth', stage2_optimizer)
            
            # Tensorboard logging
            if stage2_writer is not None:
                stage2_writer.add_scalar('train_loss', loss, epoch)
                stage2_writer.add_scalar('learning_rate', 
                                       stage2_optimizer.param_groups[0]['lr'], epoch)
                if len(self.stage2_metrics['miou']) > 0:
                    stage2_writer.add_scalar('miou', self.stage2_metrics['miou'][-1], epoch)
        
        # Save final Stage 2 model
        if dist.get_rank() == 0:
            self._save_checkpoint(self.stage2_epochs, 'stage2_final.pth', stage2_optimizer)
            if stage2_writer is not None:
                stage2_writer.close()
        
        self.stage2_end_time = datetime.now()
        duration = (self.stage2_end_time - self.stage2_start_time).total_seconds() / 3600
        self.logger.info(f"Stage 2 completed. Best mIoU: {best_miou:.3%}")
        self.logger.info(f"Stage 2 End Time: {self.stage2_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Stage 2 Duration: {duration:.2f} hours")
        
        # Generate plots and reports
        self.logger.info("\n" + "="*70)
        self.logger.info("Generating Training Visualizations and Reports...")
        self.logger.info("="*70)
        self._plot_training_curves()
        self._generate_training_report()
        self.logger.info("="*70)
        self.logger.info(f"✓ All visualizations saved to: {self.plots_dir}")
        self.logger.info(f"✓ Training report saved to: {os.path.join(self.config.log_dir, 'training_report.txt')}")
        self.logger.info("="*70 + "\n")
        
        # Unfreeze model for final validation
        if self.freeze_encoder:
            self._unfreeze_encoder()
    
    def _train_epoch(self, epoch, criterion, stage='stage1'):
        """Train one epoch for Stage 1."""
        from utils.util import AverageMeter
        
        self.model.train()
        loss_meter = AverageMeter()
        
        for idx, (points, mask, features, points_labels, cloud_label, input_inds) in enumerate(self.train_loader):
            points = points.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            features = features.cuda(non_blocking=True)
            points_labels = points_labels.cuda(non_blocking=True)
            
            pred = self.model(points, mask, features)
            loss = criterion(pred, points_labels, mask)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()
            self.scheduler.step()
            
            loss_meter.update(loss.item(), points.size(0))
            
            if idx % self.config.print_freq == 0:
                self.logger.info(f'{stage.upper()} Train: [{epoch}][{idx}/{len(self.train_loader)}]\t'
                               f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})')
        
        return loss_meter.avg
    
    def _train_epoch_stage2(self, epoch, train_loader, criterion, optimizer):
        """
        Train one epoch for Stage 2 with balanced data.
        
        Issue #4: Using SAME augmentation as Stage 1 (removed enhanced augmentation)
        Rationale: Enhanced augmentation was too aggressive for short training (20 epochs)
        and may create features too different from what the frozen encoder learned.
        
        CRITICAL FIX: Add warmup and more aggressive gradient clipping to prevent instability.
        """
        from utils.util import AverageMeter
        import datasets.data_utils as d_utils
        
        self.model.train()
        loss_meter = AverageMeter()
        
        # 🔧 FIX: Warmup strategy updated - can be disabled for ultra-low LR training
        # With LR as low as 0.00001, warmup may be unnecessary and can actually harm training
        if self.stage2_warmup_epochs is None:
            # Auto-determine warmup based on LR magnitude
            if self.stage2_lr <= 0.00002:  # Very low LR
                warmup_epochs = 0  # No warmup needed
                self.logger.info("Auto-disabled warmup due to ultra-low LR (<=0.00002)")
            else:
                warmup_epochs = min(5, self.stage2_epochs // 4)  # Short warmup
        else:
            warmup_epochs = self.stage2_warmup_epochs
        
        base_lr = self.stage2_lr  # Use full stage2_lr directly
        
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            # Conservative warmup: start at 50% and ramp up to 100%
            warmup_factor = 0.5 + (0.5 * epoch / warmup_epochs)
            current_lr = base_lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            self.logger.info(f"Epoch {epoch}/{self.stage2_epochs}: Warmup LR = {current_lr:.6f} (factor={warmup_factor:.3f})")
        else:
            # Use full base_lr (no warmup or warmup complete)
            current_lr = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            if epoch == 1 or (warmup_epochs > 0 and epoch == warmup_epochs + 1):
                status = "no warmup" if warmup_epochs == 0 else "warmup complete"
                self.logger.info(f"Epoch {epoch}/{self.stage2_epochs}: Full LR = {base_lr:.6f} ({status})")
        
        for idx, (points, mask, features, points_labels, cloud_label, input_inds) in enumerate(train_loader):
            points = points.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            features = features.cuda(non_blocking=True)
            points_labels = points_labels.cuda(non_blocking=True)
            
            # No additional augmentation - dataset handles it with same config as Stage 1
            
            pred = self.model(points, mask, features)
            loss = criterion(pred, points_labels, mask)
            
            optimizer.zero_grad()
            loss.backward()
            
            # CRITICAL: More aggressive gradient clipping to prevent instability
            # Reduced from 10 to 1.0 for Stage 2 fine-tuning
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            
            loss_meter.update(loss.item(), points.size(0))
            
            if idx % self.config.print_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f'STAGE2 Train: [{epoch}][{idx}/{len(train_loader)}]\t'
                               f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f}) LR {current_lr:.6f}')
        
        return loss_meter.avg
    
    # Issue #4: Removed _get_stage2_augmentation() and _apply_stage2_augmentation()
    # No need for enhanced augmentation methods - using same augmentation as Stage 1
    
    def _freeze_encoder(self):
        """Freeze encoder/backbone layers."""
        # Freeze backbone parameters
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
            
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
            self.logger.info("Backbone parameters frozen")
            
            # Issue #6: Verify what was actually frozen
            self._verify_frozen_parameters()
        else:
            self.logger.warning("No 'backbone' attribute found in model")
    
    def _unfreeze_encoder(self):
        """Unfreeze encoder/backbone layers."""
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
            
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = True
            self.logger.info("Backbone parameters unfrozen")
    
    def _verify_frozen_parameters(self):
        """
        Issue #6: Verify which parameters are trainable.
        
        This ensures that freezing worked correctly and logs the parameter counts.
        """
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        trainable_layers = []
        frozen_layers = []
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
                trainable_layers.append(f"{name} ({param_count:,})")
            else:
                frozen_params += param_count
                frozen_layers.append(name)
        
        self.logger.info("="*70)
        self.logger.info("Parameter Freeze Verification")
        self.logger.info("="*70)
        self.logger.info(f"Total parameters:     {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        self.logger.info(f"Frozen parameters:    {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
        self.logger.info("\nTrainable layers (decoder):")
        for layer in trainable_layers:
            self.logger.info(f"  ✓ {layer}")
        self.logger.info(f"\nFrozen layers (encoder): {len(frozen_layers)} layers frozen")
        self.logger.info("="*70)
    
    def _create_stage2_optimizer(self):
        """Create optimizer for Stage 2 with lower learning rate and stronger regularization."""
        # Only optimize parameters that require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 🔧 FIX: Use stage2_lr directly without additional reduction
        # The ultra-low LR (0.00001) is already configured correctly
        effective_lr = self.stage2_lr
        
        # 🔧 FIX #5: Support separate weight_decay for Stage 2
        # Strong regularization prevents overfitting to resampled tail examples
        stage2_weight_decay = getattr(self.config, 'stage2_weight_decay', self.config.weight_decay)
        
        self.logger.info(f"Creating Stage 2 optimizer:")
        self.logger.info(f"  → Learning rate: {effective_lr:.6f}")
        self.logger.info(f"  → Weight decay: {stage2_weight_decay:.6f} (Stage 1: {self.config.weight_decay:.6f})")
        self.logger.info(f"  → LR ratio: {effective_lr / self.config.base_learning_rate:.4f} of Stage 1 base LR")
        
        if self.config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(trainable_params,
                                      lr=effective_lr,
                                      momentum=self.config.momentum,
                                      weight_decay=stage2_weight_decay)
        elif self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(trainable_params,
                                       lr=effective_lr,
                                       weight_decay=stage2_weight_decay)
        elif self.config.optimizer == 'adamW':
            optimizer = torch.optim.AdamW(trainable_params,
                                        lr=effective_lr,
                                        weight_decay=stage2_weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.config.optimizer} not supported")
        
        return optimizer
    
    def _save_checkpoint(self, epoch, filename, optimizer=None):
        """Save model checkpoint."""
        if optimizer is None:
            optimizer = self.optimizer
            
        state = {
            'config': self.config,
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'stage': self.current_stage
        }
        
        save_path = os.path.join(self.stage1_dir if self.current_stage == 1 else self.stage2_dir, filename)
        torch.save(state, save_path)
        self.logger.info(f"Checkpoint saved: {save_path}")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle both distributed data parallel (DDP) and non-distributed (non-DDP) model state dicts
        model_state_dict = checkpoint['model']
        
        # If current model is DDP but checkpoint doesn't have 'module.' prefix
        if hasattr(self.model, 'module') and not any(key.startswith('module.') for key in model_state_dict.keys()):
            # Add 'module.' prefix to checkpoint keys
            model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
        
        # If current model is not DDP but checkpoint has 'module.' prefix
        elif not hasattr(self.model, 'module') and any(key.startswith('module.') for key in model_state_dict.keys()):
            # Remove 'module.' prefix from checkpoint keys
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        
        self.model.load_state_dict(model_state_dict)
        self.logger.info(f"Model loaded from {checkpoint_path}")
    
    def _plot_training_curves(self):
        """Generate and save comprehensive training visualization plots."""
        if dist.get_rank() != 0:
            return  # Only plot on rank 0
        
        self.logger.info("Generating training visualization plots...")
        
        # Get class names for labels
        class_names = []
        if hasattr(self.train_loader.dataset, 'label_to_names'):
            class_names = [self.train_loader.dataset.label_to_names.get(i, f"Class {i}") 
                          for i in range(self.config.num_classes)]
        else:
            class_names = [f"Class {i}" for i in range(self.config.num_classes)]
        
        # 1. Loss Curves (Train + Val)
        self._plot_loss_curves()
        
        # 2. mIoU Curves
        self._plot_miou_curves()
        
        # 3. Per-Class IoU Heatmap
        self._plot_per_class_iou_heatmap(class_names)
        
        # 4. Learning Rate Schedule
        self._plot_learning_rate_schedule()
        
        # 5. Training Time Analysis
        self._plot_training_time_analysis()
        
        # 6. Combined Summary Plot
        self._plot_combined_summary(class_names)
        
        self.logger.info(f"All plots saved to: {self.plots_dir}")
    
    def _plot_loss_curves(self):
        """Plot training and validation loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Stage 1 Loss
        if len(self.stage1_metrics['train_loss']) > 0:
            ax = axes[0]
            epochs = self.stage1_metrics['epochs']
            ax.plot(epochs, self.stage1_metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if len(self.stage1_metrics['val_loss']) > 0:
                ax.plot(epochs, self.stage1_metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Stage 1: Representation Learning', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'Stage 1 Skipped\n(Using Pretrained Model)', 
                        ha='center', va='center', fontsize=12, transform=axes[0].transAxes)
            axes[0].set_title('Stage 1: Representation Learning', fontsize=14, fontweight='bold')
        
        # Stage 2 Loss
        if len(self.stage2_metrics['train_loss']) > 0:
            ax = axes[1]
            epochs = self.stage2_metrics['epochs']
            ax.plot(epochs, self.stage2_metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if len(self.stage2_metrics['val_loss']) > 0:
                ax.plot(epochs, self.stage2_metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Stage 2: Classifier Re-training', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_miou_curves(self):
        """Plot mIoU progression curves."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        all_epochs = []
        all_mious = []
        stage_boundaries = []
        
        # Stage 1 mIoU
        if len(self.stage1_metrics['miou']) > 0:
            epochs = self.stage1_metrics['val_epochs']
            mious = [m * 100 for m in self.stage1_metrics['miou']]  # Convert to percentage
            ax.plot(epochs, mious, 'b-o', label='Stage 1', linewidth=2, markersize=4)
            all_epochs.extend(epochs)
            all_mious.extend(mious)
            if len(epochs) > 0:
                stage_boundaries.append(max(epochs))
        
        # Stage 2 mIoU
        if len(self.stage2_metrics['miou']) > 0:
            # Offset epochs for Stage 2
            offset = stage_boundaries[-1] if stage_boundaries else 0
            epochs = [e + offset for e in self.stage2_metrics['val_epochs']]
            mious = [m * 100 for m in self.stage2_metrics['miou']]
            ax.plot(epochs, mious, 'r-s', label='Stage 2', linewidth=2, markersize=4)
            all_epochs.extend(epochs)
            all_mious.extend(mious)
        
        # Add stage boundary line
        if stage_boundaries:
            ax.axvline(x=stage_boundaries[0], color='green', linestyle='--', 
                      linewidth=2, label='Stage Transition')
        
        ax.set_xlabel('Training Progress (Epoch)', fontsize=12)
        ax.set_ylabel('mIoU (%)', fontsize=12)
        ax.set_title('mIoU Progression During Two-Stage Training', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add best mIoU annotations
        if all_mious:
            best_miou = max(all_mious)
            best_epoch = all_epochs[all_mious.index(best_miou)]
            ax.annotate(f'Best: {best_miou:.2f}%', 
                       xy=(best_epoch, best_miou), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'miou_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_iou_heatmap(self, class_names):
        """Plot per-class IoU as heatmap over time."""
        # Combine Stage 1 and Stage 2 per-class IoU
        all_per_class_iou = []
        epoch_labels = []
        
        if len(self.stage1_metrics['per_class_iou']) > 0:
            for i, epoch in enumerate(self.stage1_metrics['epochs']):
                if i < len(self.stage1_metrics['per_class_iou']):
                    all_per_class_iou.append(self.stage1_metrics['per_class_iou'][i])
                    epoch_labels.append(f'S1-E{epoch}')
        
        if len(self.stage2_metrics['per_class_iou']) > 0:
            for i, epoch in enumerate(self.stage2_metrics['epochs']):
                if i < len(self.stage2_metrics['per_class_iou']):
                    all_per_class_iou.append(self.stage2_metrics['per_class_iou'][i])
                    epoch_labels.append(f'S2-E{epoch}')
        
        if not all_per_class_iou:
            return
        
        # Convert to numpy array and to percentage
        iou_matrix = np.array(all_per_class_iou).T * 100  # Shape: (num_classes, num_epochs)
        
        fig, ax = plt.subplots(figsize=(max(12, len(epoch_labels) * 0.5), 8))
        im = ax.imshow(iou_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
        
        # Set ticks
        ax.set_xticks(np.arange(len(epoch_labels)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(epoch_labels, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('IoU (%)', rotation=270, labelpad=20, fontsize=12)
        
        ax.set_title('Per-Class IoU Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Progress', fontsize=12)
        ax.set_ylabel('Class', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'per_class_iou_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_rate_schedule(self):
        """Plot learning rate schedule."""
        fig, ax = plt.subplots(figsize=(12, 5))
        
        all_epochs = []
        all_lrs = []
        stage_boundaries = []
        
        # Stage 1 LR
        if len(self.stage1_metrics['learning_rates']) > 0:
            epochs = self.stage1_metrics['epochs']
            lrs = self.stage1_metrics['learning_rates']
            ax.plot(epochs, lrs, 'b-', label='Stage 1', linewidth=2)
            all_epochs.extend(epochs)
            all_lrs.extend(lrs)
            if len(epochs) > 0:
                stage_boundaries.append(max(epochs))
        
        # Stage 2 LR
        if len(self.stage2_metrics['learning_rates']) > 0:
            offset = stage_boundaries[-1] if stage_boundaries else 0
            epochs = [e + offset for e in self.stage2_metrics['epochs']]
            lrs = self.stage2_metrics['learning_rates']
            ax.plot(epochs, lrs, 'r-', label='Stage 2', linewidth=2)
            all_epochs.extend(epochs)
            all_lrs.extend(lrs)
        
        # Add stage boundary
        if stage_boundaries:
            ax.axvline(x=stage_boundaries[0], color='green', linestyle='--', 
                      linewidth=2, label='Stage Transition')
        
        ax.set_xlabel('Training Progress (Epoch)', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_yscale('log')
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'learning_rate_schedule.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_training_time_analysis(self):
        """Plot training time analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Time per epoch
        ax = axes[0]
        all_times = []
        all_labels = []
        
        if len(self.stage1_metrics['time_per_epoch']) > 0:
            s1_times = self.stage1_metrics['time_per_epoch']
            all_times.extend(s1_times)
            all_labels.extend([f'S1-E{e}' for e in self.stage1_metrics['epochs'][:len(s1_times)]])
        
        if len(self.stage2_metrics['time_per_epoch']) > 0:
            s2_times = self.stage2_metrics['time_per_epoch']
            all_times.extend(s2_times)
            all_labels.extend([f'S2-E{e}' for e in self.stage2_metrics['epochs'][:len(s2_times)]])
        
        if all_times:
            x = np.arange(len(all_times))
            colors = ['blue'] * len(self.stage1_metrics.get('time_per_epoch', [])) + \
                    ['red'] * len(self.stage2_metrics.get('time_per_epoch', []))
            ax.bar(x, all_times, color=colors, alpha=0.6)
            ax.axhline(y=np.mean(all_times), color='green', linestyle='--', 
                      label=f'Mean: {np.mean(all_times):.1f}s', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Time (seconds)', fontsize=12)
            ax.set_title('Time Per Epoch', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # Total time breakdown
        ax = axes[1]
        stage_times = []
        stage_labels = []
        stage_colors = []
        
        if self.stage1_start_time and self.stage1_end_time:
            s1_time = (self.stage1_end_time - self.stage1_start_time).total_seconds() / 3600
            stage_times.append(s1_time)
            stage_labels.append(f'Stage 1\n{s1_time:.2f}h')
            stage_colors.append('blue')
        
        if self.stage2_start_time and self.stage2_end_time:
            s2_time = (self.stage2_end_time - self.stage2_start_time).total_seconds() / 3600
            stage_times.append(s2_time)
            stage_labels.append(f'Stage 2\n{s2_time:.2f}h')
            stage_colors.append('red')
        
        if stage_times:
            ax.pie(stage_times, labels=stage_labels, colors=stage_colors, autopct='%1.1f%%',
                  startangle=90, textprops={'fontsize': 11})
            ax.set_title(f'Total Training Time: {sum(stage_times):.2f}h', 
                        fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_time_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_summary(self, class_names):
        """Create a combined summary plot with key metrics."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Loss curves (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        if len(self.stage1_metrics['train_loss']) > 0 or len(self.stage2_metrics['train_loss']) > 0:
            all_epochs = []
            all_train_loss = []
            all_val_loss = []
            
            if len(self.stage1_metrics['train_loss']) > 0:
                all_epochs.extend(self.stage1_metrics['epochs'])
                all_train_loss.extend(self.stage1_metrics['train_loss'])
                all_val_loss.extend(self.stage1_metrics.get('val_loss', [0] * len(self.stage1_metrics['epochs'])))
            
            if len(self.stage2_metrics['train_loss']) > 0:
                offset = max(all_epochs) if all_epochs else 0
                all_epochs.extend([e + offset for e in self.stage2_metrics['epochs']])
                all_train_loss.extend(self.stage2_metrics['train_loss'])
                all_val_loss.extend(self.stage2_metrics.get('val_loss', [0] * len(self.stage2_metrics['epochs'])))
            
            ax1.plot(all_epochs, all_train_loss, 'b-', label='Train Loss', linewidth=2)
            if any(all_val_loss):
                ax1.plot(all_epochs, all_val_loss, 'r-', label='Val Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. mIoU curve (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        all_epochs = []
        all_mious = []
        if len(self.stage1_metrics['miou']) > 0:
            all_epochs.extend(self.stage1_metrics['val_epochs'])
            all_mious.extend([m * 100 for m in self.stage1_metrics['miou']])
        if len(self.stage2_metrics['miou']) > 0:
            offset = max(all_epochs) if all_epochs else 0
            all_epochs.extend([e + offset for e in self.stage2_metrics['val_epochs']])
            all_mious.extend([m * 100 for m in self.stage2_metrics['miou']])
        if all_mious:
            ax2.plot(all_epochs, all_mious, 'g-o', linewidth=2, markersize=3)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('mIoU (%)')
            ax2.set_title('mIoU Progression', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            # Annotate best
            best_miou = max(all_mious)
            ax2.text(0.05, 0.95, f'Best: {best_miou:.2f}%', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 3. Final per-class IoU bar chart (middle row)
        ax3 = fig.add_subplot(gs[1, :])
        # Fix: Check if stage2 has per_class_iou, fallback to stage1 if not
        if self.stage2_metrics['per_class_iou'] and len(self.stage2_metrics['per_class_iou']) > 0:
            final_iou = np.array(self.stage2_metrics['per_class_iou'][-1]) * 100
            title_stage = 'Stage 2'
        elif self.stage1_metrics['per_class_iou'] and len(self.stage1_metrics['per_class_iou']) > 0:
            final_iou = np.array(self.stage1_metrics['per_class_iou'][-1]) * 100
            title_stage = 'Stage 1'
        else:
            final_iou = None
        
        if final_iou is not None:
            colors = ['green' if iou > 50 else 'orange' if iou > 20 else 'red' for iou in final_iou]
            bars = ax3.bar(range(len(final_iou)), final_iou, color=colors, alpha=0.7)
            ax3.set_xticks(range(len(class_names)))
            ax3.set_xticklabels(class_names, rotation=45, ha='right')
            ax3.set_ylabel('IoU (%)')
            ax3.set_title(f'Final Per-Class IoU ({title_stage})', fontweight='bold')
            ax3.axhline(y=np.mean(final_iou), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(final_iou):.2f}%', linewidth=2)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No per-class IoU data available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Final Per-Class IoU', fontweight='bold')
        
        # 4. Training statistics (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis('off')
        stats_text = "Training Statistics\n" + "="*30 + "\n\n"
        
        if self.checkpoint_source:
            stats_text += f"Checkpoint Source:\n  {os.path.basename(self.checkpoint_source)}\n\n"
        
        if self.stage1_start_time and self.stage1_end_time:
            s1_hours = (self.stage1_end_time - self.stage1_start_time).total_seconds() / 3600
            stats_text += f"Stage 1 Time: {s1_hours:.2f}h\n"
        
        if self.stage2_start_time and self.stage2_end_time:
            s2_hours = (self.stage2_end_time - self.stage2_start_time).total_seconds() / 3600
            stats_text += f"Stage 2 Time: {s2_hours:.2f}h\n"
        
        if self.stage1_start_time and self.stage2_end_time:
            total_hours = (self.stage2_end_time - self.stage1_start_time).total_seconds() / 3600
            stats_text += f"Total Time: {total_hours:.2f}h\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        # 5. Model parameters (bottom middle)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        # This will be filled from verification logs
        
        # 6. Stage comparison (bottom right)
        ax6 = fig.add_subplot(gs[2, 2])
        # Fix: Handle case where only one stage has metrics
        if len(self.stage1_metrics['miou']) > 0 or len(self.stage2_metrics['miou']) > 0:
            stages = []
            values = []
            colors_bars = []
            
            if len(self.stage1_metrics['miou']) > 0:
                stage1_best = max(self.stage1_metrics['miou']) * 100
                stages.append('Stage 1\n(Representation)')
                values.append(stage1_best)
                colors_bars.append('blue')
            
            if len(self.stage2_metrics['miou']) > 0:
                stage2_best = max(self.stage2_metrics['miou']) * 100
                stages.append('Stage 2\n(Classifier)')
                values.append(stage2_best)
                
                # Determine color based on improvement
                if len(self.stage1_metrics['miou']) > 0:
                    improvement = stage2_best - stage1_best
                    colors_bars.append('red' if improvement > 0 else 'orange')
                else:
                    colors_bars.append('red')
            
            if len(values) > 0:
                bars = ax6.bar(stages, values, color=colors_bars, alpha=0.7)
                ax6.set_ylabel('Best mIoU (%)')
                ax6.set_title('Stage Comparison', fontweight='bold')
                ax6.set_ylim([0, 100])
                
                for bar, val in zip(bars, values):
                    ax6.text(bar.get_x() + bar.get_width()/2., val + 2,
                            f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
                
                # Add improvement arrow only if both stages exist
                if len(values) == 2 and len(self.stage1_metrics['miou']) > 0 and len(self.stage2_metrics['miou']) > 0:
                    stage1_best = max(self.stage1_metrics['miou']) * 100
                    stage2_best = max(self.stage2_metrics['miou']) * 100
                    improvement = stage2_best - stage1_best
                    ax6.annotate('', xy=(1, stage2_best), xytext=(0, stage1_best),
                               arrowprops=dict(arrowstyle='->', lw=2, color='green' if improvement > 0 else 'red'))
                    ax6.text(0.5, (stage1_best + stage2_best) / 2, 
                            f'{improvement:+.2f}%',
                            ha='center', va='center', fontweight='bold', fontsize=11,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.suptitle('Two-Stage Training Summary', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(os.path.join(self.plots_dir, 'training_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_training_report(self):
        """Generate a text report of training statistics."""
        if dist.get_rank() != 0:
            return
        
        report_path = os.path.join(self.config.log_dir, 'training_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TWO-STAGE TRAINING REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Timestamp
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Checkpoint source
            if self.checkpoint_source:
                f.write(f"Checkpoint Source: {self.checkpoint_source}\n\n")
            
            # Stage 1 Summary
            f.write("-" * 80 + "\n")
            f.write("STAGE 1: Representation Learning\n")
            f.write("-" * 80 + "\n")
            if len(self.stage1_metrics['miou']) > 0:
                f.write(f"Epochs: {len(self.stage1_metrics['epochs'])}\n")
                f.write(f"Best mIoU: {max(self.stage1_metrics['miou']) * 100:.2f}%\n")
                # Fix: Check if train_loss list has data before accessing
                if len(self.stage1_metrics['train_loss']) > 0:
                    f.write(f"Final Train Loss: {self.stage1_metrics['train_loss'][-1]:.4f}\n")
                if self.stage1_start_time and self.stage1_end_time:
                    duration = self.stage1_end_time - self.stage1_start_time
                    f.write(f"Training Time: {duration.total_seconds() / 3600:.2f} hours\n")
            else:
                f.write("Stage 1 was skipped (using pretrained model)\n")
                if self.checkpoint_source:
                    f.write(f"  Loaded from: {os.path.basename(self.checkpoint_source)}\n")
            f.write("\n")
            
            # Stage 2 Summary
            f.write("-" * 80 + "\n")
            f.write("STAGE 2: Classifier Re-training\n")
            f.write("-" * 80 + "\n")
            if len(self.stage2_metrics['miou']) > 0:
                f.write(f"Epochs: {len(self.stage2_metrics['epochs'])}\n")
                f.write(f"Best mIoU: {max(self.stage2_metrics['miou']) * 100:.2f}%\n")
                # Fix: Check if train_loss list has data before accessing
                if len(self.stage2_metrics['train_loss']) > 0:
                    f.write(f"Final Train Loss: {self.stage2_metrics['train_loss'][-1]:.4f}\n")
                if self.stage2_start_time and self.stage2_end_time:
                    duration = self.stage2_end_time - self.stage2_start_time
                    f.write(f"Training Time: {duration.total_seconds() / 3600:.2f} hours\n")
                
                # Per-class IoU
                if self.stage2_metrics['per_class_iou']:
                    f.write("\nFinal Per-Class IoU:\n")
                    final_iou = self.stage2_metrics['per_class_iou'][-1]
                    class_names = [self.train_loader.dataset.label_to_names.get(i, f"Class {i}") 
                                  for i in range(len(final_iou))]
                    for i, (name, iou) in enumerate(zip(class_names, final_iou)):
                        f.write(f"  {i:2d}. {name:20s}: {iou * 100:6.2f}%\n")
                    f.write(f"\nMean IoU: {np.mean(final_iou) * 100:.2f}%\n")
            else:
                f.write("Stage 2 was not executed or completed\n")
            f.write("\n")
            
            # Overall Summary
            f.write("=" * 80 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("=" * 80 + "\n")
            if self.stage1_start_time and self.stage2_end_time:
                total_duration = self.stage2_end_time - self.stage1_start_time
                f.write(f"Total Training Time: {total_duration.total_seconds() / 3600:.2f} hours\n")
            
            if len(self.stage1_metrics['miou']) > 0 and len(self.stage2_metrics['miou']) > 0:
                stage1_best = max(self.stage1_metrics['miou']) * 100
                stage2_best = max(self.stage2_metrics['miou']) * 100
                improvement = stage2_best - stage1_best
                f.write(f"Stage 1 Best mIoU: {stage1_best:.2f}%\n")
                f.write(f"Stage 2 Best mIoU: {stage2_best:.2f}%\n")
                f.write(f"Improvement: {improvement:+.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        self.logger.info(f"Training report saved to: {report_path}")


# Integration function for existing training script
def create_two_stage_trainer(config, model, criterion, optimizer, scheduler, 
                           train_loader, val_loader, logger):
    """
    Factory function to create two-stage trainer.
    
    Args:
        config: Training configuration
        model: ResPointNet++ model
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        logger: Logger object
        
    Returns:
        TwoStageTrainer: Initialized trainer
    """
    return TwoStageTrainer(config, model, criterion, optimizer, scheduler,
                         train_loader, val_loader, logger)