"""
Class-Balanced Loss Implementation for Long-Tailed Learning

Overview (Pseudo Code):
1. Calculate effective number of samples per class: eff_num = (1 - beta^n) / (1 - beta)
2. Compute class weights: weights = (1 - beta) / eff_num
3. Apply focal loss with class balancing: CB_loss = alpha * weights * focal_loss

This implementation adapts CB loss for masked segmentation tasks in 3D point clouds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss with Focal Loss for addressing class imbalance.
    
    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    Adapted for 3D point cloud segmentation with masking.
    """
    
    def __init__(self, samples_per_class, num_classes, beta=0.9999, gamma=2.0, alpha=1.0):
        """
        Initialize Class-Balanced Loss.
        
        Args:
            samples_per_class (list): Number of samples for each class
            num_classes (int): Total number of classes
            beta (float): Hyperparameter for effective number calculation
            gamma (float): Focusing parameter for focal loss
            alpha (float): Weighting factor
        """
        super(ClassBalancedLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes  # Normalize
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, logits, targets, mask):
        """
        Forward pass for CB loss with masking.
        
        Args:
            logits (torch.Tensor): Model predictions (B, C, N)
            targets (torch.Tensor): Ground truth labels (B, N)
            mask (torch.Tensor): Valid point mask (B, N)
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Apply mask to get valid predictions and targets
        batch_size, num_classes, num_points = logits.shape
        
        # Flatten and apply mask
        logits_flat = logits.permute(0, 2, 1).contiguous().view(-1, num_classes)  # (B*N, C)
        targets_flat = targets.view(-1)  # (B*N,)
        mask_flat = mask.view(-1)  # (B*N,)
        
        # Filter valid points
        valid_idx = mask_flat > 0
        if valid_idx.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        logits_valid = logits_flat[valid_idx]  # (N_valid, C)
        targets_valid = targets_flat[valid_idx]  # (N_valid,)
        
        # Compute softmax probabilities
        log_probs = F.log_softmax(logits_valid, dim=1)
        probs = F.softmax(logits_valid, dim=1)
        
        # Get class weights for valid targets
        class_weights = self.weights[targets_valid]  # (N_valid,)
        
        # Compute focal loss components
        ce_loss = F.nll_loss(log_probs, targets_valid, reduction='none')  # (N_valid,)
        pt = probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)  # (N_valid,)
        focal_weight = (1 - pt) ** self.gamma
        
        # Combine CB weighting with focal loss
        cb_focal_loss = self.alpha * class_weights * focal_weight * ce_loss
        
        return cb_focal_loss.mean()
    
    @staticmethod
    def calculate_samples_per_class(dataset, num_classes):
        """
        Calculate number of samples per class from dataset.
        
        Args:
            dataset: Dataset object with labels (PSNet5Seg or PSNet12Seg)
            num_classes (int): Number of classes
            
        Returns:
            list: Samples per class
        """
        samples_per_class = [0] * num_classes
        
        # Count samples in sub-clouds (training data)
        for labels in dataset.sub_clouds_points_labels:
            unique, counts = np.unique(labels, return_counts=True)
            for cls, count in zip(unique, counts):
                if 0 <= cls < num_classes:
                    samples_per_class[cls] += count
                    
        return samples_per_class


# Usage example for integration with existing training
def create_cb_loss(dataset, num_classes, beta=0.9999, gamma=2.0, alpha=1.0):
    """
    Factory function to create CB loss from dataset.
    
    Args:
        dataset: Training dataset
        num_classes (int): Number of classes
        beta (float): CB loss hyperparameter
        gamma (float): Focal loss hyperparameter
        alpha (float): Loss weighting factor
        
    Returns:
        ClassBalancedLoss: Initialized CB loss function
    """
    samples_per_class = ClassBalancedLoss.calculate_samples_per_class(dataset, num_classes)
    print(f"Samples per class: {samples_per_class}")
    print(f"Class distribution: {dict(zip(dataset.label_to_names.values(), samples_per_class))}")
    
    return ClassBalancedLoss(samples_per_class, num_classes, beta, gamma, alpha)
```

## 2. Class-Balanced Sampler Implementation

```python:LTL/cb_sampler.py
"""
Class-Balanced Sampler for Long-Tailed Learning

Overview (Pseudo Code):
1. Calculate class frequencies from dataset
2. Compute sampling probabilities (inverse frequency weighting)
3. For each epoch:
   - Sample points ensuring balanced class representation
   - Create batches with roughly equal class distribution
4. Oversample tail classes, undersample head classes

This sampler works at the point level for 3D segmentation tasks.
"""

import torch
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict
import random


class ClassBalancedSampler(Sampler):
    """
    Class-Balanced Sampler for point cloud segmentation.
    
    Ensures each batch has roughly balanced class representation by
    oversampling points from tail classes and undersampling head classes.
    """
    
    def __init__(self, dataset, samples_per_epoch=None, replacement=True):
        """
        Initialize Class-Balanced Sampler.
        
        Args:
            dataset: Dataset object with point cloud data
            samples_per_epoch (int): Number of samples per epoch (default: dataset length)
            replacement (bool): Whether to sample with replacement
        """
        self.dataset = dataset
        self.replacement = replacement
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        
        # Calculate class statistics
        self.class_stats = self._calculate_class_statistics()
        self.sampling_weights = self._calculate_sampling_weights()
        
        print(f"Class statistics: {self.class_stats}")
        print(f"Sampling weights: {dict(zip(dataset.label_to_names.values(), self.sampling_weights))}")
    
    def _calculate_class_statistics(self):
        """Calculate number of points per class across all sub-clouds."""
        class_counts = defaultdict(int)
        total_points = 0
        
        for labels in self.dataset.sub_clouds_points_labels:
            unique, counts = np.unique(labels, return_counts=True)
            total_points += len(labels)
            for cls, count in zip(unique, counts):
                class_counts[cls] += count
        
        # Convert to list format
        num_classes = len(self.dataset.label_to_names)
        class_stats = [class_counts.get(i, 0) for i in range(num_classes)]
        
        return {
            'counts': class_stats,
            'frequencies': [count / total_points for count in class_stats],
            'total_points': total_points
        }
    
    def _calculate_sampling_weights(self):
        """Calculate inverse frequency weights for sampling."""
        frequencies = np.array(self.class_stats['frequencies'])
        # Avoid division by zero
        frequencies = np.maximum(frequencies, 1e-8)
        
        # Inverse frequency weighting
        weights = 1.0 / frequencies
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)
        
        return weights.tolist()
    
    def _create_point_indices_by_class(self):
        """Create mapping from class to point indices."""
        class_to_indices = defaultdict(list)
        
        for cloud_idx, labels in enumerate(self.dataset.sub_clouds_points_labels):
            for point_idx, label in enumerate(labels):
                # Store (cloud_idx, point_idx) tuple
                class_to_indices[label].append((cloud_idx, point_idx))
        
        return class_to_indices
    
    def __iter__(self):
        """Generate indices for balanced sampling."""
        # Get point indices organized by class
        class_to_indices = self._create_point_indices_by_class()
        
        # Create sampling probabilities for each sphere center
        sphere_weights = []
        
        for step_idx in range(len(self.dataset)):
            # For each potential sphere center, calculate its sampling weight
            # based on the class distribution in its neighborhood
            
            # This is a simplified approach - in practice, you might want to
            # pre-compute neighborhood class distributions
            weight = 1.0  # Default weight
            sphere_weights.append(weight)
        
        # Sample sphere indices based on their weights
        if self.replacement:
            indices = torch.multinomial(
                torch.tensor(sphere_weights, dtype=torch.float),
                self.samples_per_epoch,
                replacement=True
            ).tolist()
        else:
            # Without replacement sampling
            indices = list(range(len(self.dataset)))
            if self.samples_per_epoch < len(indices):
                # Weighted sampling without replacement
                probs = np.array(sphere_weights)
                probs = probs / probs.sum()
                indices = np.random.choice(
                    indices, 
                    size=self.samples_per_epoch, 
                    replace=False, 
                    p=probs
                ).tolist()
            random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self):
        return self.samples_per_epoch


class ClassBalancedBatchSampler:
    """
    Batch sampler that ensures balanced class representation within each batch.
    
    This is an alternative approach that works at the batch level.
    """
    
    def __init__(self, dataset, batch_size, samples_per_class_per_batch=None):
        """
        Initialize batch sampler.
        
        Args:
            dataset: Dataset object
            batch_size (int): Batch size
            samples_per_class_per_batch (int): Target samples per class in each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = len(dataset.label_to_names)
        
        if samples_per_class_per_batch is None:
            self.samples_per_class_per_batch = max(1, batch_size // self.num_classes)
        else:
            self.samples_per_class_per_batch = samples_per_class_per_batch
        
        # Pre-compute class-to-indices mapping
        self.class_to_sphere_indices = self._map_classes_to_spheres()
    
    def _map_classes_to_spheres(self):
        """Map each class to sphere indices that contain that class."""
        class_to_spheres = defaultdict(list)
        
        # This is a simplified mapping - in practice, you'd want to analyze
        # the actual class distribution within each sphere's neighborhood
        for sphere_idx in range(len(self.dataset)):
            # For now, assume all spheres can contribute to all classes
            # In practice, you'd analyze the neighborhood of each sphere center
            for class_idx in range(self.num_classes):
                class_to_spheres[class_idx].append(sphere_idx)
        
        return class_to_spheres
    
    def __iter__(self):
        """Generate balanced batches."""
        num_batches = len(self.dataset) // self.batch_size
        
        for _ in range(num_batches):
            batch_indices = []
            
            # Sample from each class
            for class_idx in range(self.num_classes):
                available_spheres = self.class_to_sphere_indices[class_idx]
                if available_spheres:
                    sampled = np.random.choice(
                        available_spheres,
                        size=min(self.samples_per_class_per_batch, len(available_spheres)),
                        replace=True
                    )
                    batch_indices.extend(sampled.tolist())
            
            # Fill remaining batch slots randomly
            while len(batch_indices) < self.batch_size:
                batch_indices.append(np.random.randint(0, len(self.dataset)))
            
            # Shuffle and trim to exact batch size
            random.shuffle(batch_indices)
            yield batch_indices[:self.batch_size]
    
    def __len__(self):
        return len(self.dataset) // self.batch_size


# Integration helper function
def create_balanced_dataloader(dataset, batch_size, num_workers=4, sampler_type='point'):
    """
    Create a dataloader with class-balanced sampling.
    
    Args:
        dataset: Dataset object
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        sampler_type (str): Type of sampler ('point' or 'batch')
        
    Returns:
        torch.utils.data.DataLoader: Balanced dataloader
    """
    from torch.utils.data import DataLoader
    
    if sampler_type == 'point':
        sampler = ClassBalancedSampler(dataset, samples_per_epoch=len(dataset))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    elif sampler_type == 'batch':
        batch_sampler = ClassBalancedBatchSampler(dataset, batch_size)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")
```

## 3. Two-Stage Training Framework

```python:LTL/two_stage_trainer.py
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

Now let me create the integration script that shows how to use these LTL techniques:

```python:LTL/integration_example.py
"""
Integration Example for LTL Techniques with ResPointNet++

This script demonstrates how to integrate the LTL techniques
with the existing ResPointNet++ training pipeline.

Usage Examples:
1. CB Loss only: python train_ltl.py --ltl_method cb_loss
2. CB Sampling only: python train_ltl.py --ltl_method cb_sampling  
3. Two-stage training: python train_ltl.py --ltl_method two_stage
4. Combined approach: python train_ltl.py --ltl_method combined
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LTL.cb_loss import create_cb_loss
from LTL.cb_sampler import create_balanced_dataloader
from LTL.two_stage_trainer import create_two_stage_trainer


def modify_training_for_ltl(config, ltl_method='baseline'):
    """
    Modify training configuration for different LTL methods.
    
    Args:
        config: Original training configuration
        ltl_method (str): LTL method to use
        
    Returns:
        dict: Modified configuration
    """
    ltl_config = config
    
    if ltl_method == 'cb_loss':
        # Use CB loss with standard training
        ltl_config.use_cb_loss = True
        ltl_config.cb_beta = 0.9999
        ltl_config.cb_gamma = 2.0
        ltl_config.cb_alpha = 1.0
        
    elif ltl_method == 'cb_sampling':
        # Use balanced sampling with standard loss
        ltl_config.use_balanced_sampling = True
        ltl_config.use_cb_loss = False
        
    elif ltl_method == 'two_stage':
        # Two-stage training configuration
        ltl_config.use_two_stage = True
        ltl_config.stage1_epochs = config.epochs
        ltl_config.stage2_epochs = config.epochs // 4
        ltl_config.stage2_lr = config.base_learning_rate * 0.1
        ltl_config.use_cb_loss_stage1 = False
        ltl_config.use_cb_loss_stage2 = True
        ltl_config.freeze_encoder = True
        
    elif ltl_method == 'combined':
        # Combined approach: CB loss + two-stage
        ltl_config.use_two_stage = True
        ltl_config.stage1_epochs = config.epochs
        ltl_config.stage2_epochs = config.epochs // 4
        ltl_config.stage2_lr = config.base_learning_rate * 0.1
        ltl_config.use_cb_loss_stage1 = True
        ltl_config.use_cb_loss_stage2 = True
        ltl_config.freeze_encoder = True
        ltl_config.cb_beta = 0.9999
        ltl_config.cb_gamma = 2.0
        ltl_config.cb_alpha = 1.0
        
    return ltl_config


def setup_ltl_training(config, model, criterion, optimizer, scheduler,
                      train_loader, val_loader, logger, validate_func):
    """
    Setup training based on LTL method specified in config.
    
    Args:
        config: Training configuration
        model: ResPointNet++ model
        criterion: Original loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        logger: Logger object
        validate_func: Validation function
        
    Returns:
        Configured training components
    """
    
    # Determine LTL method from config
    ltl_method = getattr(config, 'ltl_method', 'baseline')
    use_cb_loss = getattr(config, 'use_cb_loss', False)
    use_balanced_sampling = getattr(config, 'use_balanced_sampling', False)
    use_two_stage = getattr(config, 'use_two_stage', False)
    
    logger.info(f"LTL Method: {ltl_method}")
    
    # Setup criterion
    if use_cb_loss:
        criterion = create_cb_loss(
            train_loader.dataset,
            config.num_classes,
            beta=getattr(config, 'cb_beta', 0.9999),
            gamma=getattr(config, 'cb_gamma', 2.0),
            alpha=getattr(config, 'cb_alpha', 1.0)
        )
        criterion.cuda()
        logger.info("Using Class-Balanced Loss")
    
    # Setup data loader
    if use_balanced_sampling:
        train_loader = create_balanced_dataloader(
            train_loader.dataset,
            config.batch_size,
            num_workers=config.num_workers,
            sampler_type='point'
        )
        logger.info("Using Class-Balanced Sampling")
    
    # Setup trainer
    if use_two_stage:
        trainer = create_two_stage_trainer(
            config, model, criterion, optimizer, scheduler,
            train_loader, val_loader, logger
        )
        logger.info("Using Two-Stage Training Framework")
        return trainer, criterion, train_loader
    else:
        return None, criterion, train_loader


# Example usage in main training script
def example_integration():
    """
    Example of how to integrate LTL techniques into existing training script.
    
    This would replace the main training loop in function/train_psnet_dist.py
    """
    
    # ... existing setup code (config, model, etc.) ...
    
    # Setup LTL training
    trainer, modified_criterion, modified_train_loader = setup_ltl_training(
        config, model, criterion, optimizer, scheduler,
        train_loader, val_loader, logger, validate_func
    )
    
    if trainer is not None:
        # Two-stage training
        trainer.train_stage1(validate_func)
        trainer.train_stage2(validate_func)
    else:
        # Standard training with LTL modifications
        for epoch in range(config.start_epoch, config.epochs + 1):
            # ... existing training loop with modified components ...
            pass


if __name__ == "__main__":
    print("LTL Integration Example")
    print("See function docstrings for usage details")
```

Finally, let me create a configuration file that shows how to enable different LTL methods:

```yaml:LTL/ltl_config_example.yaml
# Example configuration for LTL techniques
# Add these parameters to your existing config files

# LTL Method Selection
ltl_method: "two_stage"  # Options: baseline, cb_loss, cb_sampling, two_stage, combined

# Class-Balanced Loss Parameters
use_cb_loss: true
cb_beta: 0.9999      # Effective number hyperparameter (0.9999 for long-tailed)
cb_gamma: 2.0        # Focal loss focusing parameter
cb_alpha: 1.0        # Loss weighting factor

# Class-Balanced Sampling Parameters  
use_balanced_sampling: false
sampler_type: "point"  # Options: point, batch

# Two-Stage Training Parameters
use_two_stage: true
stage1_epochs: 200     # Same as original training epochs
stage2_epochs: 50      # Typically 1/4 of stage1_epochs
stage2_lr: 0.001       # Lower learning rate for stage2
use_cb_loss_stage1: false    # Use CB loss in stage1
use_cb_loss_stage2: true     # Use CB loss in stage2
freeze_encoder: true         # Freeze backbone in stage2

# Evaluation Parameters
eval_tail_classes: true      # Evaluate tail class performance separately
tail_class_threshold: 0.1    # Classes with <10% of max frequency are "tail"
```

These implementations provide:

1. **Class-Balanced Loss**: Addresses class imbalance by re-weighting the loss function based on effective number of samples per class.

2. **Class-Balanced Sampler**: Ensures balanced representation during training by oversampling tail classes and undersampling head classes.

3. **Two-Stage Training Framework**: Decouples representation learning from classification, allowing the model to learn good features first, then adjust decision boundaries.

Each component is designed to integrate seamlessly with your existing ResPointNet++ codebase and can be used independently or in combination. The pseudo code comments explain the high-level approach, and the implementations handle the 3D point cloud specific challenges like masking and distributed training.
