"""
Two-Stage Training Framework for Long-Tailed Learning

Overview (Pseudo Code):
Stage 1 - Representation Learning:
1. Train full ResPointNet++ end-to-end on original long-tailed data
2. Use standard cross-entropy or CB loss
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
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from .cb_loss import create_cb_loss
from .cb_sampler import create_balanced_dataloader


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
        self.stage2_lr = getattr(config, 'stage2_lr', config.base_learning_rate * 0.1)
        self.use_cb_loss_stage1 = getattr(config, 'use_cb_loss_stage1', False)
        self.use_cb_loss_stage2 = getattr(config, 'use_cb_loss_stage2', True)
        self.freeze_encoder = getattr(config, 'freeze_encoder', True)
        
        # Stage tracking
        self.current_stage = 1
        self.stage1_completed = False
        
        # Create stage-specific directories
        self.stage1_dir = os.path.join(config.log_dir, 'stage1')
        self.stage2_dir = os.path.join(config.log_dir, 'stage2')
        os.makedirs(self.stage1_dir, exist_ok=True)
        os.makedirs(self.stage2_dir, exist_ok=True)
    
    def train_stage1(self, validate_func):
        """
        Stage 1: Representation Learning
        
        Train the full network end-to-end on the original long-tailed dataset.
        Focus on learning rich geometric representations.
        """
        self.logger.info("=" * 50)
        self.logger.info("Starting Stage 1: Representation Learning")
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
            
            self.logger.info(f'Stage1 Epoch {epoch}/{self.stage1_epochs}, '
                           f'Time {time.time() - tic:.2f}s, '
                           f'LR {self.optimizer.param_groups[0]["lr"]:.5f}, '
                           f'Loss {loss:.4f}')
            
            # Validation
            if epoch % self.config.val_freq == 0:
                miou = validate_func(f'S1-E{epoch}', self.val_loader, self.model, 
                                   stage1_criterion, num_votes=2)
                
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
        
        # Save final Stage 1 model
        if dist.get_rank() == 0:
            self._save_checkpoint(self.stage1_epochs, 'stage1_final.pth')
            if stage1_writer is not None:
                stage1_writer.close()
        
        self.stage1_completed = True
        self.logger.info(f"Stage 1 completed. Best mIoU: {best_miou:.3%}")
    
    def train_stage2(self, validate_func):
        """
        Stage 2: Classifier Re-training
        
        Freeze encoder and fine-tune classifier on class-balanced data.
        """
        if not self.stage1_completed:
            raise RuntimeError("Stage 1 must be completed before Stage 2")
        
        self.logger.info("=" * 50)
        self.logger.info("Starting Stage 2: Classifier Re-training")
        self.logger.info(f"Epochs: {self.stage2_epochs}")
        self.logger.info(f"Freeze Encoder: {self.freeze_encoder}")
        self.logger.info(f"Use CB Loss: {self.use_cb_loss_stage2}")
        self.logger.info("=" * 50)
        
        # Load best Stage 1 model
        stage1_best_path = os.path.join(self.stage1_dir, 'stage1_best.pth')
        if os.path.exists(stage1_best_path):
            self._load_checkpoint(stage1_best_path)
            self.logger.info(f"Loaded Stage 1 best model from {stage1_best_path}")
        else:
            self.logger.warning("Stage 1 best model not found, using current model")
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            self._freeze_encoder()
            self.logger.info("Encoder layers frozen")
        
        # Setup Stage 2 criterion
        if self.use_cb_loss_stage2:
            stage2_criterion = create_cb_loss(
                self.train_loader.dataset, 
                self.config.num_classes,
                beta=getattr(self.config, 'cb_beta', 0.9999),
                gamma=getattr(self.config, 'cb_gamma', 2.0),
                alpha=getattr(self.config, 'cb_alpha', 1.0)
            )
            stage2_criterion.cuda()
            self.logger.info("Using Class-Balanced Loss for Stage 2")
        else:
            stage2_criterion = self.criterion
            self.logger.info("Using standard MaskedCrossEntropy for Stage 2")
        
        # Create class-balanced dataloader for Stage 2
        balanced_train_loader = create_balanced_dataloader(
            self.train_loader.dataset,
            self.config.batch_size,
            num_workers=self.config.num_workers,
            sampler_type='point'
        )
        self.logger.info("Created class-balanced dataloader for Stage 2")
        
        # Setup new optimizer for Stage 2 with lower learning rate
        stage2_optimizer = self._create_stage2_optimizer()
        
        # Setup tensorboard for Stage 2
        if dist.get_rank() == 0:
            stage2_writer = SummaryWriter(log_dir=self.stage2_dir)
        else:
            stage2_writer = None
        
        # Training loop for Stage 2
        best_miou = 0.0
        self.current_stage = 2
        
        for epoch in range(1, self.stage2_epochs + 1):
            balanced_train_loader.sampler.set_epoch(epoch)
            self.val_loader.sampler.set_epoch(epoch)
            balanced_train_loader.dataset.epoch = epoch - 1
            
            tic = time.time()
            
            # Train one epoch with balanced data
            loss = self._train_epoch_stage2(epoch, balanced_train_loader, 
                                          stage2_criterion, stage2_optimizer)
            
            self.logger.info(f'Stage2 Epoch {epoch}/{self.stage2_epochs}, '
                           f'Time {time.time() - tic:.2f}s, '
                           f'LR {stage2_optimizer.param_groups[0]["lr"]:.5f}, '
                           f'Loss {loss:.4f}')
            
            # Validation
            if epoch % self.config.val_freq == 0:
                miou = validate_func(f'S2-E{epoch}', self.val_loader, self.model, 
                                   stage2_criterion, num_votes=2)
                
                # Save best model
                if miou > best_miou:
                    best_miou = miou
                    if dist.get_rank() == 0:
                        self._save_checkpoint(epoch, 'stage2_best.pth', stage2_optimizer)
            
            # Save checkpoint
            if dist.get_rank() == 0 and epoch % self.config.save_freq == 0:
                self._save_checkpoint(epoch, f'stage2_epoch_{epoch}.pth', stage2_optimizer)
            
            # Tensorboard logging
            if stage2_writer is not None:
                stage2_writer.add_scalar('train_loss', loss, epoch)
                stage2_writer.add_scalar('learning_rate', 
                                       stage2_optimizer.param_groups[0]['lr'], epoch)
        
        # Save final Stage 2 model
        if dist.get_rank() == 0:
            self._save_checkpoint(self.stage2_epochs, 'stage2_final.pth', stage2_optimizer)
            if stage2_writer is not None:
                stage2_writer.close()
        
        self.logger.info(f"Stage 2 completed. Best mIoU: {best_miou:.3%}")
        
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
        """Train one epoch for Stage 2 with balanced data."""
        from utils.util import AverageMeter
        
        self.model.train()
        loss_meter = AverageMeter()
        
        for idx, (points, mask, features, points_labels, cloud_label, input_inds) in enumerate(train_loader):
            points = points.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            features = features.cuda(non_blocking=True)
            points_labels = points_labels.cuda(non_blocking=True)
            
            pred = self.model(points, mask, features)
            loss = criterion(pred, points_labels, mask)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            optimizer.step()
            
            loss_meter.update(loss.item(), points.size(0))
            
            if idx % self.config.print_freq == 0:
                self.logger.info(f'STAGE2 Train: [{epoch}][{idx}/{len(train_loader)}]\t'
                               f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})')
        
        return loss_meter.avg
    
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
    
    def _create_stage2_optimizer(self):
        """Create optimizer for Stage 2 with lower learning rate."""
        # Only optimize parameters that require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(trainable_params,
                                      lr=self.stage2_lr,
                                      momentum=self.config.momentum,
                                      weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(trainable_params,
                                       lr=self.stage2_lr,
                                       weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamW':
            optimizer = torch.optim.AdamW(trainable_params,
                                        lr=self.stage2_lr,
                                        weight_decay=self.config.weight_decay)
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
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        
        self.logger.info(f"Model loaded from {checkpoint_path}")


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
```

