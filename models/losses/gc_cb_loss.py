"""
Geometric Context-Aware Class-Balanced Loss Implementation

This loss modulates the standard CB loss by the entropy of neighbors' predictions,
targeting ambiguous regions at class boundaries where geometrically sparse objects
are particularly vulnerable to misclassification.

Key Innovation:
- Combines class frequency balancing with contextual hardness mining
- Focuses on boundary regions with inconsistent neighborhood predictions
- Geometry-aware hard example mining (different from Focal Loss which only looks at point confidence)

Mathematical Formulation:
    L_GC-CB = (1-β)/(1-β^N_c) * (1 + λ*H(p_neighbors)) * focal_weight * CE(p,y)
    
Where:
    - N_c: number of points in class c
    - H(p_neighbors): entropy of predicted class probabilities for k nearest neighbors
    - λ: weight for contextual hardness term
    - β, γ, α: hyperparameters from CB loss and Focal loss

Reference: 
    Building on "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    Extended with geometric context awareness for 3D point cloud segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
try:
    from .cb_loss import ClassBalancedLoss
except:
    from cb_loss import ClassBalancedLoss


class GeometricContextAwareCBLoss(nn.Module):
    """
    Geometric Context-Aware Class-Balanced Loss for 3D point cloud segmentation.
    
    Modulates CB loss by neighborhood prediction entropy to focus on ambiguous boundary regions.
    """
    
    def __init__(
        self, 
        samples_per_class: List[int], 
        num_classes: int, 
        beta: float = 0.9999, 
        gamma: float = 2.0, 
        alpha: float = 1.0,
        context_k: int = 16,
        context_weight: float = 1.0,
        entropy_temp: float = 1.0
    ):
        """
        Initialize Geometric Context-Aware CB Loss.
        
        Args:
            samples_per_class: Number of point samples for each class
            num_classes: Total number of classes
            beta: Hyperparameter for effective number calculation (CB loss)
            gamma: Focusing parameter for focal loss
            alpha: Overall weighting factor
            context_k: Number of nearest neighbors for context computation
            context_weight: Weight for contextual hardness term (λ in formula)
            entropy_temp: Temperature for entropy calculation (higher = smoother)
        """
        super(GeometricContextAwareCBLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.context_k = context_k
        self.context_weight = context_weight
        self.entropy_temp = entropy_temp
        
        # Calculate CB weights (class frequency balancing)
        effective_num = 1.0 - np.power(beta, samples_per_class)
        cb_weights = (1.0 - beta) / np.array(effective_num)
        cb_weights = cb_weights / np.sum(cb_weights) * num_classes  # Normalize
        
        self.register_buffer('cb_weights', torch.tensor(cb_weights, dtype=torch.float32))
    
    def compute_neighborhood_entropy(
        self,
        points: torch.Tensor,
        probs: torch.Tensor,
        mask: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """
        Compute prediction entropy for k-nearest neighbors of each point.
        
        Args:
            points: Point coordinates (B, N, 3)
            probs: Predicted probabilities (B, N, C)
            mask: Valid point mask (B, N)
            k: Number of nearest neighbors
            
        Returns:
            Neighborhood entropy for each point (B, N)
        """
        batch_size, num_points, _ = points.shape
        device = points.device
        
        # Initialize entropy tensor
        neighborhood_entropy = torch.zeros(batch_size, num_points, device=device)
        
        for b in range(batch_size):
            # Get valid points for this batch item
            valid_mask = mask[b] > 0
            if valid_mask.sum() == 0:
                continue
                
            valid_points = points[b, valid_mask]  # (N_valid, 3)
            valid_probs = probs[b, valid_mask]  # (N_valid, C)
            
            # Compute pairwise distances
            # dist[i,j] = ||point_i - point_j||^2
            diff = valid_points.unsqueeze(1) - valid_points.unsqueeze(0)  # (N_valid, N_valid, 3)
            dist = torch.sum(diff ** 2, dim=2)  # (N_valid, N_valid)
            
            # Get k nearest neighbors (excluding self)
            # Add large value to diagonal to exclude self
            dist = dist + torch.eye(dist.shape[0], device=device) * 1e10
            _, neighbor_indices = torch.topk(dist, k, largest=False, dim=1)  # (N_valid, k)
            
            # Gather neighbor probabilities
            neighbor_probs = valid_probs[neighbor_indices]  # (N_valid, k, C)
            
            # Compute entropy for each point's neighborhood
            # Entropy = -sum(p * log(p)) where p is the mean probability across neighbors
            mean_neighbor_probs = neighbor_probs.mean(dim=1)  # (N_valid, C)
            
            # Apply temperature scaling
            mean_neighbor_probs = mean_neighbor_probs / self.entropy_temp
            mean_neighbor_probs = F.softmax(mean_neighbor_probs, dim=1)
            
            # Calculate entropy (add epsilon to avoid log(0))
            eps = 1e-8
            entropy = -torch.sum(mean_neighbor_probs * torch.log(mean_neighbor_probs + eps), dim=1)  # (N_valid,)
            
            # Normalize entropy to [0, 1] range
            # Maximum entropy for uniform distribution: log(num_classes)
            max_entropy = np.log(self.num_classes)
            entropy = entropy / max_entropy
            
            # Place entropy values back into full tensor
            neighborhood_entropy[b, valid_mask] = entropy
        
        return neighborhood_entropy
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        mask: torch.Tensor,
        points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Geometric Context-Aware CB loss with masking.
        
        Args:
            logits: Model predictions (B, C, N)
            targets: Ground truth labels (B, N)
            mask: Valid point mask (B, N)
            points: Point coordinates (B, N, 3) - required for context computation
            
        Returns:
            Computed loss value
        """
        batch_size, num_classes, num_points = logits.shape
        
        # Compute probabilities for context calculation
        probs = F.softmax(logits, dim=1)  # (B, C, N)
        probs = probs.permute(0, 2, 1).contiguous()  # (B, N, C)
        
        # Compute neighborhood entropy if points are provided
        if points is not None and self.context_weight > 0:
            # Ensure points are in correct shape (B, N, 3)
            if points.shape[1] == 3:  # (B, 3, N) -> (B, N, 3)
                points = points.permute(0, 2, 1).contiguous()
            
            neighborhood_entropy = self.compute_neighborhood_entropy(
                points, probs, mask, self.context_k
            )  # (B, N)
        else:
            # If no points provided or context disabled, set entropy to 0
            neighborhood_entropy = torch.zeros(batch_size, num_points, device=logits.device)
        
        # Flatten and apply mask
        logits_flat = logits.permute(0, 2, 1).contiguous().view(-1, num_classes)  # (B*N, C)
        targets_flat = targets.view(-1)  # (B*N,)
        mask_flat = mask.view(-1)  # (B*N,)
        entropy_flat = neighborhood_entropy.view(-1)  # (B*N,)
        
        # Filter valid points
        valid_idx = mask_flat > 0
        if valid_idx.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        logits_valid = logits_flat[valid_idx]  # (N_valid, C)
        targets_valid = targets_flat[valid_idx]  # (N_valid,)
        entropy_valid = entropy_flat[valid_idx]  # (N_valid,)
        
        # Compute softmax probabilities
        log_probs = F.log_softmax(logits_valid, dim=1)
        probs_valid = F.softmax(logits_valid, dim=1)
        
        # Get CB weights for valid targets
        cb_class_weights = self.cb_weights[targets_valid]  # (N_valid,)
        
        # Compute contextual hardness: (1 + λ*H(neighbors))
        contextual_hardness = 1.0 + self.context_weight * entropy_valid  # (N_valid,)
        
        # Compute focal loss components
        ce_loss = F.nll_loss(log_probs, targets_valid, reduction='none')  # (N_valid,)
        pt = probs_valid.gather(1, targets_valid.unsqueeze(1)).squeeze(1)  # (N_valid,)
        focal_weight = (1 - pt) ** self.gamma
        
        # Combine: α × cb_weights × contextual_hardness × focal_weight × ce_loss
        gc_cb_focal_loss = self.alpha * cb_class_weights * contextual_hardness * focal_weight * ce_loss
        
        return gc_cb_focal_loss.mean()


# Factory function for easy integration
def create_gc_cb_loss(
    dataset,
    num_classes: int,
    beta: float = 0.9999,
    gamma: float = 2.0,
    alpha: float = 1.0,
    context_k: int = 16,
    context_weight: float = 1.0,
    entropy_temp: float = 1.0
) -> GeometricContextAwareCBLoss:
    """
    Factory function to create Geometric Context-Aware CB loss from dataset.
    
    Args:
        dataset: Training dataset (PSNet5Seg or PSNet12Seg)
        num_classes: Number of classes
        beta: CB loss hyperparameter
        gamma: Focal loss hyperparameter
        alpha: Loss weighting factor
        context_k: Number of neighbors for context computation
        context_weight: Weight for contextual hardness (0=disable, 1=full)
        entropy_temp: Temperature for entropy calculation
        
    Returns:
        GeometricContextAwareCBLoss: Initialized GC-CB loss function
    """
    # Calculate samples per class
    samples_per_class = ClassBalancedLoss.calculate_samples_per_class(dataset, num_classes)
    
    # Print statistics
    print("\n=== Geometric Context-Aware CB Loss Initialization ===")
    print(f"\nClass Statistics:")
    print(f"{'Class':<15} {'Points':<12} {'CB Weight':<12}")
    print("-" * 45)
    
    # Calculate preview weights for display
    effective_num = 1.0 - np.power(beta, samples_per_class)
    cb_weights = (1.0 - beta) / np.array(effective_num)
    cb_weights = cb_weights / np.sum(cb_weights) * num_classes
    
    for cls in range(num_classes):
        class_name = dataset.label_to_names.get(cls, f"Class_{cls}") if hasattr(dataset, 'label_to_names') else f"Class_{cls}"
        print(f"{class_name:<15} {samples_per_class[cls]:<12} {cb_weights[cls]:<12.4f}")
    
    print(f"\nContext Parameters:")
    print(f"  k (neighbors): {context_k}")
    print(f"  Context Weight (λ): {context_weight}")
    print(f"  Entropy Temperature: {entropy_temp}")
    print("=" * 45 + "\n")
    
    return GeometricContextAwareCBLoss(
        samples_per_class=samples_per_class,
        num_classes=num_classes,
        beta=beta,
        gamma=gamma,
        alpha=alpha,
        context_k=context_k,
        context_weight=context_weight,
        entropy_temp=entropy_temp
    )
