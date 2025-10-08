#!/usr/bin/env python3
"""
Two-Stage Training Results Analyzer

Analyzes the log file from two-stage training and produces:
1. Performance comparison (Stage 1 vs Stage 2)
2. Per-class IoU improvements
3. Training stability metrics
4. Verification status

Usage:
    python scripts/analyze_two_stage_results.py <log_dir>
    
Example:
    python scripts/analyze_two_stage_results.py log/psnet12/two_stage_FIXED_lr1e5_noWarmup_20251008/
"""

import sys
import os
import re
import numpy as np
from pathlib import Path


def parse_miou_from_log(log_path, stage=None):
    """Extract mIoU values from log file."""
    mious = []
    epochs = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match patterns like "E10 V1 * mIoU 52.345%"
            if stage == 1:
                match = re.search(r'ES1-E(\d+)\s+V\d+\s+\*\s+mIoU\s+([\d.]+)%', line)
            elif stage == 2:
                match = re.search(r'ES2-E(\d+)\s+V\d+\s+\*\s+mIoU\s+([\d.]+)%', line)
            else:
                match = re.search(r'E(\d+)\s+V\d+\s+\*\s+mIoU\s+([\d.]+)%', line)
            
            if match:
                epoch = int(match.group(1))
                miou = float(match.group(2))
                epochs.append(epoch)
                mious.append(miou)
    
    return epochs, mious


def parse_per_class_iou(log_path, stage=None):
    """Extract per-class IoU from log file."""
    per_class_ious = {}
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match patterns like "ES2-E10 V1  * msIoU [0.913 0.415 ...]"
            if stage == 1:
                match = re.search(r'ES1-E(\d+)\s+V\d+\s+\*\s+msIoU\s+\[([\d.\s]+)\]', line)
            elif stage == 2:
                match = re.search(r'ES2-E(\d+)\s+V\d+\s+\*\s+msIoU\s+\[([\d.\s]+)\]', line)
            else:
                match = re.search(r'E(\d+)\s+V\d+\s+\*\s+msIoU\s+\[([\d.\s]+)\]', line)
            
            if match:
                epoch = int(match.group(1))
                iou_str = match.group(2)
                ious = [float(x) for x in iou_str.split()]
                per_class_ious[epoch] = ious
    
    return per_class_ious


def check_verification_status(log_path):
    """Check verification status from log."""
    status = {
        'encoder_freeze': 'Unknown',
        'gradient_leakage': 'Unknown',
        'class_balance': 'Unknown',
        'catastrophic_forgetting': 'Unknown'
    }
    
    with open(log_path, 'r') as f:
        content = f.read()
        
        # Check encoder freeze
        if 'Encoder freeze verification PASSED' in content:
            status['encoder_freeze'] = 'PASSED ✅'
        elif 'ENCODER LEAKAGE DETECTED' in content:
            status['encoder_freeze'] = 'FAILED ❌'
        
        # Check gradient leakage
        if 'GRADIENT LEAKAGE' in content:
            status['gradient_leakage'] = 'DETECTED ❌'
        else:
            status['gradient_leakage'] = 'None ✅'
        
        # Check class balance
        if 'WELL BALANCED' in content:
            status['class_balance'] = 'Good ✅'
        elif 'MODERATELY IMBALANCED' in content:
            status['class_balance'] = 'Moderate ⚠️'
        elif 'SEVERELY IMBALANCED' in content:
            status['class_balance'] = 'Poor ❌'
        
        # Check catastrophic forgetting
        forgetting_count = content.count('Performance dropping')
        if forgetting_count == 0:
            status['catastrophic_forgetting'] = 'None ✅'
        elif forgetting_count < 3:
            status['catastrophic_forgetting'] = f'Mild ({forgetting_count} warnings) ⚠️'
        else:
            status['catastrophic_forgetting'] = f'Severe ({forgetting_count} warnings) ❌'
    
    return status


def analyze_results(log_dir):
    """Main analysis function."""
    log_dir = Path(log_dir)
    log_path = log_dir / 'log.txt'
    
    if not log_path.exists():
        print(f"❌ Error: Log file not found at {log_path}")
        return 1
    
    print("=" * 80)
    print("TWO-STAGE TRAINING RESULTS ANALYSIS")
    print("=" * 80)
    print(f"Analyzing: {log_dir}")
    print()
    
    # Parse Stage 1 and Stage 2 mIoU
    s1_epochs, s1_mious = parse_miou_from_log(log_path, stage=1)
    s2_epochs, s2_mious = parse_miou_from_log(log_path, stage=2)
    
    # Performance comparison
    print("-" * 80)
    print("PERFORMANCE COMPARISON")
    print("-" * 80)
    
    if s1_mious:
        s1_best = max(s1_mious)
        s1_best_epoch = s1_epochs[s1_mious.index(s1_best)]
        print(f"Stage 1 Best mIoU: {s1_best:.3f}% (Epoch {s1_best_epoch})")
    else:
        s1_best = None
        print("Stage 1: No data (skipped or checkpoint loaded)")
    
    if s2_mious:
        s2_best = max(s2_mious)
        s2_best_epoch = s2_epochs[s2_mious.index(s2_best)]
        s2_final = s2_mious[-1]
        s2_final_epoch = s2_epochs[-1]
        print(f"Stage 2 Best mIoU: {s2_best:.3f}% (Epoch {s2_best_epoch})")
        print(f"Stage 2 Final mIoU: {s2_final:.3f}% (Epoch {s2_final_epoch})")
        
        if s1_best is not None:
            improvement_best = s2_best - s1_best
            improvement_final = s2_final - s1_best
            print(f"\nImprovement (Best):  {improvement_best:+.3f}%")
            print(f"Improvement (Final): {improvement_final:+.3f}%")
            
            if improvement_best > 0:
                print(f"✅ Stage 2 improved over Stage 1")
            elif improvement_best > -0.5:
                print(f"⚠️  Stage 2 maintained Stage 1 performance (within 0.5%)")
            else:
                print(f"❌ Stage 2 degraded from Stage 1 (catastrophic forgetting)")
    else:
        print("Stage 2: No data found")
    
    # Training stability
    print()
    print("-" * 80)
    print("TRAINING STABILITY")
    print("-" * 80)
    
    if s2_mious and len(s2_mious) > 1:
        # Check if performance is stable or improving
        stable_epochs = 0
        improving_epochs = 0
        degrading_epochs = 0
        
        for i in range(1, len(s2_mious)):
            diff = s2_mious[i] - s2_mious[i-1]
            if abs(diff) < 0.5:
                stable_epochs += 1
            elif diff > 0:
                improving_epochs += 1
            else:
                degrading_epochs += 1
        
        total = len(s2_mious) - 1
        print(f"Stable epochs:    {stable_epochs}/{total} ({stable_epochs/total*100:.1f}%)")
        print(f"Improving epochs: {improving_epochs}/{total} ({improving_epochs/total*100:.1f}%)")
        print(f"Degrading epochs: {degrading_epochs}/{total} ({degrading_epochs/total*100:.1f}%)")
        
        # Calculate variance
        variance = np.var(s2_mious)
        print(f"\nVariance: {variance:.4f}")
        if variance < 0.5:
            print("✅ Training is stable (low variance)")
        elif variance < 2.0:
            print("⚠️  Training is moderately stable")
        else:
            print("❌ Training is unstable (high variance)")
    
    # Per-class IoU analysis
    print()
    print("-" * 80)
    print("PER-CLASS IoU ANALYSIS")
    print("-" * 80)
    
    s1_per_class = parse_per_class_iou(log_path, stage=1)
    s2_per_class = parse_per_class_iou(log_path, stage=2)
    
    if s1_per_class and s2_per_class:
        # Get best epoch IoUs
        s1_best_iou = s1_per_class[max(s1_per_class.keys())]
        s2_best_iou = s2_per_class[max(s2_per_class.keys())]
        
        if len(s1_best_iou) == len(s2_best_iou):
            class_names = ['duct', 'elbow', 'flange', 'ibeam', 'pipe', 'pump',
                          'rectangularbeam', 'reducer', 'strainer', 'tank', 'tee', 'valve']
            
            print(f"{'Class':<20} {'Stage1':>8} {'Stage2':>8} {'Change':>8}")
            print("-" * 50)
            
            for i, name in enumerate(class_names[:len(s1_best_iou)]):
                s1 = s1_best_iou[i] * 100
                s2 = s2_best_iou[i] * 100
                diff = s2 - s1
                status = "✅" if diff > 0 else "⚠️" if diff > -1 else "❌"
                print(f"{name:<20} {s1:7.2f}% {s2:7.2f}% {diff:+7.2f}% {status}")
    
    # Verification status
    print()
    print("-" * 80)
    print("VERIFICATION STATUS")
    print("-" * 80)
    
    status = check_verification_status(log_path)
    print(f"Encoder Freeze:         {status['encoder_freeze']}")
    print(f"Gradient Leakage:       {status['gradient_leakage']}")
    print(f"Class Balance:          {status['class_balance']}")
    print(f"Catastrophic Forgetting: {status['catastrophic_forgetting']}")
    
    # Overall assessment
    print()
    print("=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    
    if s2_mious and s1_best is not None:
        improvement = s2_best - s1_best
        
        all_passed = all('✅' in v for v in status.values())
        
        if improvement > 1.0 and all_passed:
            print("✅ EXCELLENT: Stage 2 significantly improved over Stage 1")
            print("   All verification checks passed")
        elif improvement > 0.0 and all_passed:
            print("✅ GOOD: Stage 2 improved over Stage 1")
            print("   All verification checks passed")
        elif improvement > -0.5:
            print("⚠️  ACCEPTABLE: Stage 2 maintained Stage 1 performance")
            print("   Consider tuning hyperparameters for better results")
        else:
            print("❌ POOR: Stage 2 degraded from Stage 1")
            print("   Review verification failures and adjust configuration")
    
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_two_stage_results.py <log_dir>")
        print("Example: python scripts/analyze_two_stage_results.py log/psnet12/two_stage_FIXED_*/")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    sys.exit(analyze_results(log_dir))
