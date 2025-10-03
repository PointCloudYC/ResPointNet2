"""
Class-Balanced Loss Implementation for Long-Tailed Learning

Overview (Pseudo Code):
1. Calculate effective number of **point** samples per class: eff_num = (1 - beta^n) / (1 - beta); when n is large(head class), eff_num plateaus (diminishing returns) while when n is small(tail class), eff_num approximates n (linear growth).
2. Compute class weights: weights = (1 - beta) / eff_num
3. Apply focal loss with class balancing: CB_loss = alpha * weights * focal_loss

Note: This implementation correctly follows the CVPR 2019 paper formula.

This implementation adapts CB loss for masked segmentation tasks in 3D point clouds.
justification:
- Point-Level Imbalance: Even if you have equal numbers of pipe and valve objects, pipes are typically larger → more points per pipe
- Training Dynamics: The loss function operates on individual points, so point-level statistics matter
- Real Impact: A class with 100x fewer points will be overwhelmed during training
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
        # Correct formula: effective_num = (1 - β^n) / (1 - β)
        effective_num = (1.0 - np.power(beta, samples_per_class)) / (1.0 - beta)
        weights = (1.0 - beta) / np.array(effective_num)
        # critical: normalize the weights to sum to num_classes
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
        Calculate number of point samples per class from dataset.
        
        Args:
            dataset: Dataset object with labels (PSNet5Seg or PSNet12Seg)
            num_classes (int): Number of classes
            
        Returns:
            list: Point samples per class
        """
        samples_per_class = [0] * num_classes
        
        # Count point samples in sub-clouds (training data)
        for labels in dataset.sub_clouds_points_labels:
            unique, counts = np.unique(labels, return_counts=True) # example: unique = [0, 1, 2, 3], counts = [100, 200, 300, 400]
            for cls, count in zip(unique, counts):
                if 0 <= cls < num_classes:
                    samples_per_class[cls] += count # count the number of points not #objects
                    
        return samples_per_class


# Factory function for easy integration
def create_cb_loss(dataset, num_classes, beta=0.9999, gamma=2.0, alpha=1.0):
    """
    Factory function to create CB loss from dataset.
    
    Args:
        dataset: Training dataset (PSNet5Seg or PSNet12Seg)
        num_classes (int): Number of classes
        beta (float): CB loss hyperparameter
        gamma (float): Focal loss hyperparameter
        alpha (float): Loss weighting factor
        
    Returns:
        ClassBalancedLoss: Initialized CB loss function
    """
    samples_per_class = ClassBalancedLoss.calculate_samples_per_class(dataset, num_classes)
    print(f"Samples per class: {samples_per_class}")
    if hasattr(dataset, 'label_to_names'):
        print(f"Class distribution: {dict(zip(dataset.label_to_names.values(), samples_per_class))}")
    
    return ClassBalancedLoss(samples_per_class, num_classes, beta, gamma, alpha)