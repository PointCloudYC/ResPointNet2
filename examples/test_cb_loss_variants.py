"""
Test script for CB loss variants

This script demonstrates how to use the different CB loss variants
and provides basic sanity checks.
"""

import sys
import os
import numpy as np
import torch

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from models.losses import ClassBalancedLoss, LocalDensityAwareCBLoss, GeometricContextAwareCBLoss


def test_standard_cb_loss():
    """Test standard CB loss"""
    print("\n" + "="*70)
    print("Testing Standard CB Loss")
    print("="*70)
    
    # Simulate class distribution
    num_classes = 5
    samples_per_class = [10000, 5000, 1000, 500, 100]  # Imbalanced
    
    # Create loss
    loss_fn = ClassBalancedLoss(
        samples_per_class=samples_per_class,
        num_classes=num_classes,
        beta=0.9999,
        gamma=2.0,
        alpha=1.0
    )
    
    print(f"Class weights: {loss_fn.weights}")
    
    # Create dummy data
    batch_size = 2
    num_points = 1000
    
    logits = torch.randn(batch_size, num_classes, num_points)
    targets = torch.randint(0, num_classes, (batch_size, num_points))
    mask = torch.ones(batch_size, num_points)
    
    # Compute loss
    loss = loss_fn(logits, targets, mask)
    print(f"Loss value: {loss.item():.4f}")
    
    # Test backward
    loss.backward()
    print("✓ Backward pass successful")
    
    return True


def test_ld_cb_loss():
    """Test Local Density-Aware CB loss"""
    print("\n" + "="*70)
    print("Testing Local Density-Aware CB Loss")
    print("="*70)
    
    # Simulate class distribution and density
    num_classes = 5
    samples_per_class = [10000, 5000, 1000, 500, 100]
    density_per_class = [50.0, 30.0, 15.0, 10.0, 5.0]  # Head classes are denser
    
    # Create loss
    loss_fn = LocalDensityAwareCBLoss(
        samples_per_class=samples_per_class,
        density_per_class=density_per_class,
        num_classes=num_classes,
        beta=0.9999,
        gamma=2.0,
        alpha=1.0,
        density_weight=1.0
    )
    
    print(f"CB weights: {loss_fn.cb_weights}")
    print(f"Density modulation: {loss_fn.density_modulation}")
    print(f"Combined weights: {loss_fn.combined_weights}")
    
    # Create dummy data
    batch_size = 2
    num_points = 1000
    
    logits = torch.randn(batch_size, num_classes, num_points)
    targets = torch.randint(0, num_classes, (batch_size, num_points))
    mask = torch.ones(batch_size, num_points)
    
    # Compute loss
    loss = loss_fn(logits, targets, mask)
    print(f"Loss value: {loss.item():.4f}")
    
    # Test backward
    loss.backward()
    print("✓ Backward pass successful")
    
    # Verify density modulation effect
    # Sparse classes should have higher combined weights
    assert loss_fn.combined_weights[-1] > loss_fn.combined_weights[0], \
        "Sparse tail class should have higher weight than dense head class"
    print("✓ Density modulation working correctly")
    
    return True


def test_gc_cb_loss():
    """Test Geometric Context-Aware CB loss"""
    print("\n" + "="*70)
    print("Testing Geometric Context-Aware CB Loss")
    print("="*70)
    
    # Simulate class distribution
    num_classes = 5
    samples_per_class = [10000, 5000, 1000, 500, 100]
    
    # Create loss
    loss_fn = GeometricContextAwareCBLoss(
        samples_per_class=samples_per_class,
        num_classes=num_classes,
        beta=0.9999,
        gamma=2.0,
        alpha=1.0,
        context_k=16,
        context_weight=1.0,
        entropy_temp=1.0
    )
    
    print(f"CB weights: {loss_fn.cb_weights}")
    print(f"Context k: {loss_fn.context_k}")
    print(f"Context weight: {loss_fn.context_weight}")
    
    # Create dummy data
    batch_size = 2
    num_points = 100  # Smaller for efficiency in testing
    
    logits = torch.randn(batch_size, num_classes, num_points)
    targets = torch.randint(0, num_classes, (batch_size, num_points))
    mask = torch.ones(batch_size, num_points)
    points = torch.randn(batch_size, num_points, 3)  # Random point cloud
    
    # Compute loss without points (should work with zero context)
    loss_no_context = loss_fn(logits, targets, mask, points=None)
    print(f"Loss value (no context): {loss_no_context.item():.4f}")
    
    # Compute loss with points (should include context)
    loss_with_context = loss_fn(logits, targets, mask, points=points)
    print(f"Loss value (with context): {loss_with_context.item():.4f}")
    
    # Test backward
    loss_with_context.backward()
    print("✓ Backward pass successful")
    
    return True


def test_loss_comparison():
    """Compare all three loss variants"""
    print("\n" + "="*70)
    print("Comparing All CB Loss Variants")
    print("="*70)
    
    # Setup
    num_classes = 5
    samples_per_class = [10000, 5000, 1000, 500, 100]
    density_per_class = [50.0, 30.0, 15.0, 10.0, 5.0]
    
    batch_size = 2
    num_points = 100
    
    logits = torch.randn(batch_size, num_classes, num_points)
    targets = torch.randint(0, num_classes, (batch_size, num_points))
    mask = torch.ones(batch_size, num_points)
    points = torch.randn(batch_size, num_points, 3)
    
    # Create all losses
    cb_loss = ClassBalancedLoss(samples_per_class, num_classes)
    ld_cb_loss = LocalDensityAwareCBLoss(samples_per_class, density_per_class, num_classes)
    gc_cb_loss = GeometricContextAwareCBLoss(samples_per_class, num_classes)
    
    # Compute losses
    loss_cb = cb_loss(logits, targets, mask).item()
    loss_ld = ld_cb_loss(logits, targets, mask).item()
    loss_gc = gc_cb_loss(logits, targets, mask, points).item()
    
    print(f"Standard CB Loss:        {loss_cb:.4f}")
    print(f"LD-CB Loss:              {loss_ld:.4f}")
    print(f"GC-CB Loss:              {loss_gc:.4f}")
    
    # All should be positive
    assert loss_cb > 0 and loss_ld > 0 and loss_gc > 0, "All losses should be positive"
    print("✓ All losses computed successfully")
    
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*70)
    print("Testing Edge Cases")
    print("="*70)
    
    num_classes = 5
    samples_per_class = [10000, 5000, 1000, 500, 100]
    density_per_class = [50.0, 30.0, 15.0, 10.0, 5.0]
    
    loss_fn = LocalDensityAwareCBLoss(samples_per_class, density_per_class, num_classes)
    
    batch_size = 2
    num_points = 100
    
    # Test 1: Empty mask (no valid points)
    print("\nTest 1: Empty mask")
    logits = torch.randn(batch_size, num_classes, num_points)
    targets = torch.randint(0, num_classes, (batch_size, num_points))
    mask = torch.zeros(batch_size, num_points)  # All masked out
    
    loss = loss_fn(logits, targets, mask)
    print(f"Loss with empty mask: {loss.item():.4f}")
    assert loss.item() == 0.0, "Loss should be 0 for empty mask"
    print("✓ Empty mask handled correctly")
    
    # Test 2: Single class targets
    print("\nTest 2: Single class targets")
    mask = torch.ones(batch_size, num_points)
    targets = torch.zeros(batch_size, num_points, dtype=torch.long)  # All class 0
    
    loss = loss_fn(logits, targets, mask)
    print(f"Loss with single class: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("✓ Single class handled correctly")
    
    # Test 3: Partial mask
    print("\nTest 3: Partial mask")
    mask = torch.rand(batch_size, num_points) > 0.5  # Random mask
    targets = torch.randint(0, num_classes, (batch_size, num_points))
    
    loss = loss_fn(logits, targets, mask.float())
    print(f"Loss with partial mask: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("✓ Partial mask handled correctly")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("CB Loss Variants Test Suite")
    print("="*70)
    
    tests = [
        ("Standard CB Loss", test_standard_cb_loss),
        ("Local Density-Aware CB Loss", test_ld_cb_loss),
        ("Geometric Context-Aware CB Loss", test_gc_cb_loss),
        ("Loss Comparison", test_loss_comparison),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = []
    for test_name, test_fn in tests:
        try:
            result = test_fn()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with error:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
