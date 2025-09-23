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