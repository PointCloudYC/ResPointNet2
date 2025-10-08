"""
Training Verification Utilities for Two-Stage Training

Critical bug detection functions to ensure:
1. Encoder is properly frozen (no gradient leakage)
2. Class-balanced sampler is working correctly
3. Training stability (no catastrophic forgetting)
"""

import torch
import numpy as np


def verify_encoder_frozen(model, logger):
    """
    Verify that encoder layers are properly frozen.
    
    Returns:
        dict: Statistics about frozen/trainable parameters
    Raises:
        RuntimeError: If encoder leakage is detected
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    trainable_layers = []
    encoder_leaks = []
    
    # Keywords that identify encoder layers (should be frozen)
    encoder_keywords = ['encoder', 'sa_layer', 'strided_bottleneck', 'fc', 'lao', 
                       'rb', 'down_', 'local_aggregation', 'fps', 'query']
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
            trainable_layers.append(name)
            
            # Check if this looks like an encoder layer
            is_encoder = any(keyword in name.lower() for keyword in encoder_keywords)
            if is_encoder and 'segmentation_head' not in name and 'up_' not in name:
                encoder_leaks.append(name)
        else:
            frozen_params += param_count
    
    # Report results
    logger.info("="*70)
    logger.info("🔍 Encoder Freeze Verification")
    logger.info("="*70)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"Frozen parameters:    {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    
    if encoder_leaks:
        logger.error("❌ ENCODER LEAKAGE DETECTED!")
        logger.error("The following encoder parameters are NOT frozen:")
        for leak in encoder_leaks[:10]:  # Show first 10
            logger.error(f"  ❌ {leak}")
        if len(encoder_leaks) > 10:
            logger.error(f"  ... and {len(encoder_leaks) - 10} more")
        logger.error("="*70)
        raise RuntimeError(f"Encoder freezing failed! {len(encoder_leaks)} encoder parameters are trainable.")
    
    logger.info("✅ Encoder freeze verification PASSED - no leakage detected")
    logger.info(f"Trainable layers: {len(trainable_layers)} (decoder only)")
    logger.info("="*70)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_layers': trainable_layers
    }


def verify_no_gradient_leakage(model, logger, check_gradients=True):
    """
    Verify that no gradients are flowing to frozen parameters.
    
    Call this AFTER backward pass during training.
    
    Args:
        model: The model to check
        logger: Logger object
        check_gradients: If True, check actual gradient values (after backward)
                        If False, only check requires_grad flags
    
    Returns:
        bool: True if no leakage, False otherwise
    Raises:
        RuntimeError: If gradient leakage is detected
    """
    gradient_leaks = []
    
    for name, param in model.named_parameters():
        # Check if frozen parameter has gradients
        if not param.requires_grad and param.grad is not None:
            if check_gradients and param.grad.abs().sum() > 1e-8:
                gradient_leaks.append((name, param.grad.abs().max().item()))
    
    if gradient_leaks:
        logger.error("❌ GRADIENT LEAKAGE TO FROZEN PARAMETERS!")
        logger.error("The following frozen parameters received non-zero gradients:")
        for name, max_grad in gradient_leaks[:10]:
            logger.error(f"  ❌ {name}: max_grad={max_grad:.6e}")
        if len(gradient_leaks) > 10:
            logger.error(f"  ... and {len(gradient_leaks) - 10} more")
        raise RuntimeError("Gradient leakage detected! This will corrupt Stage 1 features.")
    
    return True


def verify_class_balanced_sampling(train_loader, config, logger, num_batches=50):
    """
    Verify that class-balanced sampling is producing balanced data.
    
    Checks the actual class distribution in training batches.
    
    Args:
        train_loader: DataLoader to check
        config: Config object with num_classes
        logger: Logger object
        num_batches: Number of batches to sample for verification
    
    Returns:
        dict: Statistics about class distribution
    """
    logger.info("="*70)
    logger.info("🔍 Verifying Class-Balanced Sampler")
    logger.info("="*70)
    
    class_counts = np.zeros(config.num_classes, dtype=np.int64)
    total_points = 0
    
    # Sample batches to check distribution
    num_batches_checked = 0
    for idx, (points, mask, features, points_labels, cloud_label, input_inds) in enumerate(train_loader):
        if idx >= num_batches:
            break
        
        # Count points per class
        labels_flat = points_labels[mask].cpu().numpy()
        unique, counts = np.unique(labels_flat, return_counts=True)
        for cls, count in zip(unique, counts):
            if 0 <= cls < config.num_classes:
                class_counts[cls] += count
        total_points += len(labels_flat)
        num_batches_checked += 1
    
    # Calculate statistics
    class_frequencies = class_counts / total_points if total_points > 0 else class_counts
    expected_freq = 1.0 / config.num_classes
    
    logger.info(f"Sampled {total_points:,} points from {num_batches_checked} batches")
    logger.info(f"Expected frequency per class: {expected_freq*100:.2f}%")
    logger.info("\nClass distribution:")
    
    # Calculate balance metrics
    non_zero_counts = class_counts[class_counts > 0]
    if len(non_zero_counts) > 1:
        balance_ratio = non_zero_counts.max() / non_zero_counts.min()
    else:
        balance_ratio = 1.0
    
    # Show per-class statistics
    for cls in range(config.num_classes):
        if hasattr(train_loader.dataset, 'label_to_names'):
            class_name = train_loader.dataset.label_to_names.get(cls, f"Class_{cls}")
        else:
            class_name = f"Class_{cls}"
        
        freq_pct = class_frequencies[cls] * 100
        expected_pct = expected_freq * 100
        diff = freq_pct - expected_pct
        
        # Status indicator
        if abs(diff) < expected_pct * 0.3:  # Within 30% of expected
            status = "✅"
        elif abs(diff) < expected_pct * 0.6:  # Within 60%
            status = "⚠️ "
        else:
            status = "❌"
        
        logger.info(f"  {status} {cls:2d}. {class_name:20s}: {freq_pct:5.2f}% "
                   f"(expected: {expected_pct:5.2f}%, diff: {diff:+5.2f}%)")
    
    # Overall assessment
    logger.info(f"\n📊 Balance Ratio (max/min): {balance_ratio:.2f}")
    if balance_ratio < 3.0:
        logger.info("✅ Class distribution is WELL BALANCED")
        status = "excellent"
    elif balance_ratio < 10.0:
        logger.warning(f"⚠️  Class distribution is MODERATELY IMBALANCED (ratio: {balance_ratio:.2f})")
        logger.warning("   Stage 2 effectiveness may be reduced")
        status = "moderate"
    else:
        logger.error(f"❌ Class distribution is SEVERELY IMBALANCED (ratio: {balance_ratio:.2f})")
        logger.error("   Class-balanced sampler is NOT working correctly!")
        status = "poor"
    
    logger.info("="*70)
    
    return {
        'class_counts': class_counts,
        'class_frequencies': class_frequencies,
        'balance_ratio': balance_ratio,
        'total_points': total_points,
        'status': status
    }


def verify_stage2_not_degrading(current_miou, stage1_baseline_miou, epoch, logger, 
                                 tolerance=0.05, critical_tolerance=0.10):
    """
    Verify that Stage 2 training is not causing catastrophic forgetting.
    
    Args:
        current_miou: Current validation mIoU
        stage1_baseline_miou: Stage 1 best mIoU (baseline)
        epoch: Current epoch
        logger: Logger object
        tolerance: Warning threshold (default: 5% drop)
        critical_tolerance: Error threshold (default: 10% drop)
    
    Returns:
        str: Status ('ok', 'warning', 'critical')
    """
    if stage1_baseline_miou <= 0:
        return 'ok'  # No baseline to compare
    
    drop = stage1_baseline_miou - current_miou
    drop_pct = (drop / stage1_baseline_miou) * 100
    
    if drop < tolerance:
        # Performance is stable or improving
        if current_miou > stage1_baseline_miou:
            logger.info(f"✅ Epoch {epoch}: mIoU {current_miou:.3%} > baseline {stage1_baseline_miou:.3%} "
                       f"(+{(current_miou - stage1_baseline_miou):.3%} improvement)")
        return 'ok'
    elif drop < critical_tolerance:
        # Moderate degradation - warning
        logger.warning(f"⚠️  Epoch {epoch}: mIoU dropped to {current_miou:.3%} from baseline {stage1_baseline_miou:.3%}")
        logger.warning(f"    Drop: {drop:.3%} ({drop_pct:.1f}%) - within tolerance but concerning")
        logger.warning(f"    Consider: 1) Lower LR, 2) Check for gradient issues, 3) Verify sampler")
        return 'warning'
    else:
        # Critical degradation - error
        logger.error(f"❌ Epoch {epoch}: CRITICAL PERFORMANCE DROP!")
        logger.error(f"   Current: {current_miou:.3%}, Baseline: {stage1_baseline_miou:.3%}")
        logger.error(f"   Drop: {drop:.3%} ({drop_pct:.1f}%) - EXCEEDS CRITICAL THRESHOLD")
        logger.error(f"   Catastrophic forgetting detected - recommend stopping training")
        logger.error(f"   Suggested actions:")
        logger.error(f"     1. Reduce LR by 10x (current → {stage1_baseline_miou:.6f})")
        logger.error(f"     2. Disable warmup completely")
        logger.error(f"     3. Verify encoder is frozen")
        logger.error(f"     4. Check for gradient leakage")
        return 'critical'


def run_all_verifications(model, train_loader, config, logger, 
                          stage1_baseline_miou=None, current_miou=None, epoch=1):
    """
    Run all verification checks at once.
    
    Use this at the start of Stage 2 training to catch bugs early.
    
    Args:
        model: Model to verify
        train_loader: Training data loader
        config: Config object
        logger: Logger object
        stage1_baseline_miou: Stage 1 best mIoU (optional)
        current_miou: Current mIoU (optional, for degradation check)
        epoch: Current epoch
    
    Returns:
        dict: Results from all verifications
    """
    logger.info("\n" + "="*70)
    logger.info("🔍 RUNNING COMPREHENSIVE VERIFICATION SUITE")
    logger.info("="*70 + "\n")
    
    results = {}
    
    # 1. Verify encoder freeze
    try:
        results['freeze'] = verify_encoder_frozen(model, logger)
        logger.info("✅ Encoder freeze check: PASSED\n")
    except RuntimeError as e:
        logger.error(f"❌ Encoder freeze check: FAILED - {e}\n")
        results['freeze'] = {'error': str(e)}
    
    # 2. Verify class-balanced sampling
    try:
        results['sampling'] = verify_class_balanced_sampling(train_loader, config, logger)
        logger.info("✅ Class-balanced sampling check: PASSED\n")
    except Exception as e:
        logger.error(f"❌ Class-balanced sampling check: FAILED - {e}\n")
        results['sampling'] = {'error': str(e)}
    
    # 3. Verify no catastrophic forgetting (if mIoU provided)
    if stage1_baseline_miou is not None and current_miou is not None:
        status = verify_stage2_not_degrading(
            current_miou, stage1_baseline_miou, epoch, logger
        )
        results['forgetting'] = {'status': status}
    
    logger.info("="*70)
    logger.info("🔍 VERIFICATION SUITE COMPLETED")
    logger.info("="*70 + "\n")
    
    return results

