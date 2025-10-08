"""
Combined Local Density and Geometric Context-Aware Class-Balanced Loss

This loss function integrates three key ideas to tackle the "dual crisis" in
long-tailed 3D point cloud segmentation:
1.  Class-Balanced (CB) Loss: Addresses statistical rarity by re-weighting
    classes based on their "effective number of samples".
2.  Local Density-Aware (LD) Modulation: Addresses geometric sparsity by
    up-weighting classes that have, on average, a lower local point density.
3.  Geometric Context-Aware (GC) Modulation: Addresses boundary ambiguity by
    up-weighting individual points that lie in neighborhoods with inconsistent
    (high-entropy) predictions.

The final loss is a multiplicative combination of these three factors,
allowing it to simultaneously combat both statistical and geometric challenges.

Mathematical Formulation:
    L = w_cb * w_ld * w_gc * w_focal * CE(p,y)
    
Where:
    - w_cb: Class-Balanced weight, based on effective number of samples.
    - w_ld: Class-level local density weight.
    - w_gc: Point-level geometric context weight (neighborhood entropy).
    - w_focal: Point-level focal weight (prediction confidence).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List

from .ld_cb_loss import LocalDensityAwareCBLoss
from .gc_cb_loss import GeometricContextAwareCBLoss
from .cb_loss import ClassBalancedLoss


class CombinedClassBalancedLoss(GeometricContextAwareCBLoss):
    """
    Combined Loss: Integrates Local Density and Geometric Context awareness
    with the Class-Balanced Loss framework.
    """
    
    def __init__(
        self, 
        samples_per_class: List[int], 
        density_per_class: List[float],
        num_classes: int, 
        beta: float = 0.9999, 
        gamma: float = 2.0, 
        alpha: float = 1.0,
        density_weight: float = 1.0,
        context_k: int = 16,
        context_weight: float = 1.0,
        entropy_temp: float = 1.0
    ):
        """
        Initialize Combined Class-Balanced Loss.
        
        Args:
            samples_per_class: Number of point samples for each class.
            density_per_class: Average local density for each class.
            num_classes: Total number of classes.
            beta: Hyperparameter for effective number calculation (CB loss).
            gamma: Focusing parameter for focal loss.
            alpha: Overall weighting factor.
            density_weight: Weight for density modulation term.
            context_k: Number of nearest neighbors for context computation.
            context_weight: Weight for contextual hardness term (λ in formula).
            entropy_temp: Temperature for entropy calculation.
        """
        # Initialize GC-CB which sets up most parameters and CB weights
        super().__init__(
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            beta=beta,
            gamma=gamma,
            alpha=alpha,
            context_k=context_k,
            context_weight=context_weight,
            entropy_temp=entropy_temp
        )
        
        # Now, incorporate the Local Density component
        self.density_weight = density_weight
        
        # Calculate density modulation weights (from LD-CB Loss)
        density_array = np.array(density_per_class) + 1e-6
        density_modulation = 1.0 / (1.0 + np.log(density_array))
        density_modulation = density_modulation / np.mean(density_modulation) # Normalize
        
        # Combine base CB weights with density modulation
        # self.cb_weights is already registered as a buffer in the parent class
        cb_weights_np = self.cb_weights.cpu().numpy()
        combined_weights = cb_weights_np * np.power(density_modulation, self.density_weight)
        combined_weights = combined_weights / np.sum(combined_weights) * num_classes  # Re-normalize
        
        # Overwrite the cb_weights buffer with the new combined weights
        self.register_buffer('cb_weights', torch.tensor(combined_weights, dtype=torch.float32))

# Factory function for easy integration
def create_combined_cb_loss(
    dataset,
    num_classes: int,
    beta: float = 0.9999,
    gamma: float = 2.0,
    alpha: float = 1.0,
    density_radius: float = 0.5,
    density_weight: float = 1.0,
    context_k: int = 16,
    context_weight: float = 1.0,
    entropy_temp: float = 1.0
) -> CombinedClassBalancedLoss:
    """
    Factory function to create the Combined CB loss from a dataset.
    """
    # 1. Calculate samples per class
    samples_per_class = ClassBalancedLoss.calculate_samples_per_class(dataset, num_classes)
    
    # 2. Calculate local density per class
    print("\n=== Combined (LD+GC) CB Loss Initialization ===")
    density_per_class, density_stats = LocalDensityAwareCBLoss.calculate_local_density_per_class(
        dataset, num_classes, radius=density_radius
    )
    
    # Print statistics (similar to ld_cb_loss factory)
    print(f"\nClass and Density Statistics:")
    print(f"{'Class':<15} {'Points':<12} {'Avg Density':<15}")
    print("-" * 45)
    for cls in range(num_classes):
        class_name = dataset.label_to_names.get(cls, f"Class_{cls}") if hasattr(dataset, 'label_to_names') else f"Class_{cls}"
        print(f"{class_name:<15} {samples_per_class[cls]:<12} {density_per_class[cls]:<15.2f}")
    
    print(f"\nContext Parameters:")
    print(f"  k (neighbors): {context_k}")
    print(f"  Context Weight (λ): {context_weight}")
    print("=" * 45 + "\n")
    
    return CombinedClassBalancedLoss(
        samples_per_class=samples_per_class,
        density_per_class=density_per_class,
        num_classes=num_classes,
        beta=beta,
        gamma=gamma,
        alpha=alpha,
        density_weight=density_weight,
        context_k=context_k,
        context_weight=context_weight,
        entropy_temp=entropy_temp
    )
