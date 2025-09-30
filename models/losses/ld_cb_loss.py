"""
Local Density-Aware Class-Balanced Loss Implementation

This loss modulates the standard CB loss by the average local point density for each class,
addressing the "geometric sparsity" challenge where tail-class points (e.g., thin valve handles)
exist in lower-density local neighborhoods than head-class points (e.g., large flat walls).

Key Innovation:
- Combines class frequency balancing (CB Loss) with geometric sparsity awareness
- No instance labels required - works directly with semantic segmentation datasets
- Modulates loss by average local density: rare AND sparse classes get higher weights

Mathematical Formulation:
    L_LD-CB = (1-β)/(1-β^N_c) * 1/(1+log(d̄_c)) * focal_weight * CE(p,y)
    
Where:
    - N_c: number of points in class c
    - d̄_c: average local density for class c (points within radius r)
    - β, γ, α: hyperparameters from CB loss and Focal loss

Reference: 
    Building on "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    Extended with geometric density awareness for 3D point cloud segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
try:
    from .cb_loss import ClassBalancedLoss
except:
    from cb_loss import ClassBalancedLoss


class LocalDensityAwareCBLoss(nn.Module):
    """
    Local Density-Aware Class-Balanced Loss for 3D point cloud segmentation.
    
    Modulates CB loss by average local point density to handle geometric sparsity.
    """
    
    def __init__(
        self, 
        samples_per_class: List[int], 
        density_per_class: List[float],
        num_classes: int, 
        beta: float = 0.9999, 
        gamma: float = 2.0, 
        alpha: float = 1.0,
        density_weight: float = 1.0
    ):
        """
        Initialize Local Density-Aware CB Loss.
        
        Args:
            samples_per_class: Number of point samples for each class
            density_per_class: Average local density for each class
            num_classes: Total number of classes
            beta: Hyperparameter for effective number calculation (CB loss)
            gamma: Focusing parameter for focal loss
            alpha: Overall weighting factor
            density_weight: Weight for density modulation term (default 1.0)
        """
        super(LocalDensityAwareCBLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.density_weight = density_weight
        
        # Calculate CB weights (class frequency balancing)
        effective_num = 1.0 - np.power(beta, samples_per_class)
        cb_weights = (1.0 - beta) / np.array(effective_num)
        cb_weights = cb_weights / np.sum(cb_weights) * num_classes  # Normalize
        
        # Calculate density modulation weights
        # Use 1/(1+log(density)) to down-weight dense classes
        # Add small epsilon to avoid log(0)
        density_array = np.array(density_per_class) + 1e-6
        density_modulation = 1.0 / (1.0 + np.log(density_array))
        
        # Normalize density modulation to have mean = 1
        density_modulation = density_modulation / np.mean(density_modulation)
        
        # Combine CB weights with density modulation
        combined_weights = cb_weights * np.power(density_modulation, self.density_weight)
        combined_weights = combined_weights / np.sum(combined_weights) * num_classes  # Re-normalize
        
        self.register_buffer('cb_weights', torch.tensor(cb_weights, dtype=torch.float32))
        self.register_buffer('density_modulation', torch.tensor(density_modulation, dtype=torch.float32))
        self.register_buffer('combined_weights', torch.tensor(combined_weights, dtype=torch.float32))
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for Local Density-Aware CB loss with masking.
        
        Args:
            logits: Model predictions (B, C, N)
            targets: Ground truth labels (B, N)
            mask: Valid point mask (B, N)
            
        Returns:
            Computed loss value
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
        
        # Get combined weights for valid targets
        class_weights = self.combined_weights[targets_valid]  # (N_valid,)
        
        # Compute focal loss components
        ce_loss = F.nll_loss(log_probs, targets_valid, reduction='none')  # (N_valid,)
        pt = probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)  # (N_valid,)
        focal_weight = (1 - pt) ** self.gamma
        
        # Combine: α × combined_weights × focal_weight × ce_loss
        ld_cb_focal_loss = self.alpha * class_weights * focal_weight * ce_loss
        
        return ld_cb_focal_loss.mean()
    
    @staticmethod
    def calculate_local_density_per_class(
        dataset,
        num_classes: int,
        radius: float = 0.5,
        max_points_per_class: int = 10000
    ) -> Tuple[List[float], dict]:
        """
        Calculate average local density for each class from dataset.
        
        Local density for a point = number of neighbors within radius r.
        Average local density for class c = mean density of all points in class c.
        
        Args:
            dataset: Dataset object with point clouds and labels
            num_classes: Number of classes
            radius: Search radius for local density calculation
            max_points_per_class: Max points to sample per class (for efficiency)
            
        Returns:
            Tuple of (density_per_class list, statistics dict)
        """
        from sklearn.neighbors import NearestNeighbors
        
        density_per_class = [0.0] * num_classes
        count_per_class = [0] * num_classes
        
        print(f"Computing local density (radius={radius}) for each class...")
        
        # Process each sub-cloud in the dataset
        for cloud_idx, (points, labels) in enumerate(zip(
            dataset.sub_clouds_points, 
            dataset.sub_clouds_points_labels
        )):
            if cloud_idx % 10 == 0:
                print(f"  Processing cloud {cloud_idx}/{len(dataset.sub_clouds_points)}")
            
            # Build KD-tree for this cloud
            nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree', n_jobs=-1)
            nbrs.fit(points)
            
            # Get neighbors within radius for all points
            # radius_neighbors returns list of arrays (one per point)
            neighbors_list = nbrs.radius_neighbors(points, return_distance=False)
            
            # Calculate density for each point (number of neighbors)
            point_densities = np.array([len(neighbors) - 1 for neighbors in neighbors_list])  # -1 to exclude self
            
            # Accumulate density statistics per class
            for cls in range(num_classes):
                mask = (labels == cls)
                if mask.sum() > 0:
                    class_points_density = point_densities[mask]
                    
                    # Sample if too many points (for efficiency)
                    if len(class_points_density) > max_points_per_class:
                        sampled_indices = np.random.choice(
                            len(class_points_density), 
                            max_points_per_class, 
                            replace=False
                        )
                        class_points_density = class_points_density[sampled_indices]
                    
                    density_per_class[cls] += class_points_density.sum()
                    count_per_class[cls] += len(class_points_density)
        
        # Calculate average density per class
        avg_density_per_class = [
            density_per_class[cls] / count_per_class[cls] if count_per_class[cls] > 0 else 1.0
            for cls in range(num_classes)
        ]
        
        # Compile statistics
        stats = {
            'avg_density_per_class': avg_density_per_class,
            'total_points_per_class': count_per_class,
            'radius': radius,
            'density_range': (min(avg_density_per_class), max(avg_density_per_class)),
            'density_std': float(np.std(avg_density_per_class))
        }
        
        return avg_density_per_class, stats


# Factory function for easy integration
def create_ld_cb_loss(
    dataset,
    num_classes: int,
    beta: float = 0.9999,
    gamma: float = 2.0,
    alpha: float = 1.0,
    density_radius: float = 0.5,
    density_weight: float = 1.0,
    cache_density: bool = True
) -> LocalDensityAwareCBLoss:
    """
    Factory function to create Local Density-Aware CB loss from dataset.
    
    Args:
        dataset: Training dataset (PSNet5Seg or PSNet12Seg)
        num_classes: Number of classes
        beta: CB loss hyperparameter
        gamma: Focal loss hyperparameter
        alpha: Loss weighting factor
        density_radius: Radius for local density calculation
        density_weight: Weight for density modulation (0=disable, 1=full)
        cache_density: Whether to cache density calculations
        
    Returns:
        LocalDensityAwareCBLoss: Initialized LD-CB loss function
    """
    # Calculate samples per class (from original CB loss)
    samples_per_class = ClassBalancedLoss.calculate_samples_per_class(dataset, num_classes)
    
    # Calculate local density per class
    print("\n=== Local Density-Aware CB Loss Initialization ===")
    density_per_class, density_stats = LocalDensityAwareCBLoss.calculate_local_density_per_class(
        dataset, num_classes, radius=density_radius
    )
    
    # Print statistics
    print(f"\nClass Statistics:")
    print(f"{'Class':<15} {'Points':<12} {'Avg Density':<15} {'CB Weight':<12} {'Density Mod':<15}")
    print("-" * 75)
    
    # Calculate preview weights for display
    effective_num = 1.0 - np.power(beta, samples_per_class)
    cb_weights = (1.0 - beta) / np.array(effective_num)
    cb_weights = cb_weights / np.sum(cb_weights) * num_classes
    
    density_array = np.array(density_per_class) + 1e-6
    density_modulation = 1.0 / (1.0 + np.log(density_array))
    density_modulation = density_modulation / np.mean(density_modulation)
    
    for cls in range(num_classes):
        class_name = dataset.label_to_names.get(cls, f"Class_{cls}") if hasattr(dataset, 'label_to_names') else f"Class_{cls}"
        print(f"{class_name:<15} {samples_per_class[cls]:<12} {density_per_class[cls]:<15.2f} "
              f"{cb_weights[cls]:<12.4f} {density_modulation[cls]:<15.4f}")
    
    print(f"\nDensity Statistics:")
    print(f"  Radius: {density_stats['radius']}")
    print(f"  Range: {density_stats['density_range'][0]:.2f} - {density_stats['density_range'][1]:.2f}")
    print(f"  Std Dev: {density_stats['density_std']:.2f}")
    print(f"  Density Weight: {density_weight}")
    print("=" * 75 + "\n")
    
    return LocalDensityAwareCBLoss(
        samples_per_class=samples_per_class,
        density_per_class=density_per_class,
        num_classes=num_classes,
        beta=beta,
        gamma=gamma,
        alpha=alpha,
        density_weight=density_weight
    )
