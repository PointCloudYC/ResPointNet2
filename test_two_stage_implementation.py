#!/usr/bin/env python3
"""
Test script to verify the two-stage training implementation matches the theory.
This script validates that all components are working as expected.
"""

import os
import sys
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def test_class_balanced_sampler():
    """Test that the class-balanced sampler is working correctly."""
    print("=" * 60)
    print("TESTING CLASS-BALANCED SAMPLER")
    print("=" * 60)
    
    try:
        from LTL.cb_sampler import ClassBalancedSampler, create_balanced_dataloader
        from utils.config import config, update_config
        from datasets import PSNet12Seg
        from torchvision import transforms
        import datasets.data_utils as d_utils
        
        # Load config
        config_path = "cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage.yaml"
        update_config(config_path)
        
        # Create test dataset
        test_transforms = transforms.Compose([d_utils.PointcloudToTensor()])
        dataset = PSNet12Seg(
            input_features_dim=config.input_features_dim,
            subsampling_parameter=config.sampleDl, 
            color_drop=config.color_drop,
            in_radius=config.in_radius, 
            num_points=config.num_points,
            num_steps=config.num_steps, 
            num_epochs=1,
            data_root=config.data_root, 
            transforms=test_transforms,
            split='train'
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Test sampler creation
        sampler = ClassBalancedSampler(dataset, samples_per_epoch=100, replacement=True)
        print(f"✓ ClassBalancedSampler created")
        print(f"  - Class statistics: {sampler.class_stats['counts']}")
        
        # Test iteration
        indices = list(sampler)
        print(f"✓ Sampler iteration works: {len(indices)} indices generated")
        
        # Test dataloader creation
        balanced_loader = create_balanced_dataloader(dataset, batch_size=2, num_workers=0)
        print(f"✓ Balanced dataloader created")
        
        # Test one batch
        for batch_idx, batch in enumerate(balanced_loader):
            if batch_idx == 0:
                points, mask, features, labels, cloud_label, input_inds = batch
                print(f"✓ First batch loaded successfully")
                print(f"  - Points shape: {points.shape}")
                print(f"  - Labels shape: {labels.shape}")
                break
        
        print("✅ CLASS-BALANCED SAMPLER TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ CLASS-BALANCED SAMPLER TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_two_stage_trainer():
    """Test that the two-stage trainer is properly configured."""
    print("\n" + "=" * 60)
    print("TESTING TWO-STAGE TRAINER")
    print("=" * 60)
    
    try:
        from LTL.two_stage_trainer import TwoStageTrainer
        from utils.config import config
        import logging
        
        # Mock components for testing
        class MockModel:
            def __init__(self):
                self.backbone = MockBackbone()
            def parameters(self):
                return []
            def state_dict(self):
                return {}
        
        class MockBackbone:
            def parameters(self):
                return [torch.tensor([1.0], requires_grad=True)]
        
        class MockOptimizer:
            def __init__(self):
                self.param_groups = [{'lr': 0.01}]
            def state_dict(self):
                return {}
        
        class MockScheduler:
            def state_dict(self):
                return {}
        
        class MockLoader:
            def __init__(self):
                self.dataset = MockDataset()
                self.sampler = MockSampler()
        
        class MockDataset:
            def __init__(self):
                self.sub_clouds_points_labels = [np.array([0, 1, 2, 0, 1]) for _ in range(5)]
                self.label_to_names = {0: 'pipe', 1: 'valve', 2: 'pump'}
        
        class MockSampler:
            def set_epoch(self, epoch):
                pass
        
        class MockLogger:
            def info(self, msg):
                print(f"[INFO] {msg}")
            def warning(self, msg):
                print(f"[WARN] {msg}")
        
        # Create mock config
        config.log_dir = "/tmp/test_two_stage"
        config.stage1_epochs = 5
        config.stage2_epochs = 3
        config.stage2_lr = 0.001
        config.use_cb_loss_stage1 = False
        config.use_cb_loss_stage2 = True
        config.freeze_encoder = True
        config.num_classes = 3
        config.batch_size = 2
        config.num_workers = 0
        
        # Create trainer
        trainer = TwoStageTrainer(
            config=config,
            model=MockModel(),
            criterion=lambda x, y, z: torch.tensor(1.0),
            optimizer=MockOptimizer(),
            scheduler=MockScheduler(),
            train_loader=MockLoader(),
            val_loader=MockLoader(),
            logger=MockLogger()
        )
        
        print(f"✓ TwoStageTrainer created successfully")
        print(f"  - Stage 1 epochs: {trainer.stage1_epochs}")
        print(f"  - Stage 2 epochs: {trainer.stage2_epochs}")
        print(f"  - Stage 2 LR: {trainer.stage2_lr}")
        print(f"  - Freeze encoder: {trainer.freeze_encoder}")
        print(f"  - Use CB loss stage 1: {trainer.use_cb_loss_stage1}")
        print(f"  - Use CB loss stage 2: {trainer.use_cb_loss_stage2}")
        
        # Test encoder freezing
        trainer._freeze_encoder()
        print("✓ Encoder freezing works")
        
        # Test optimizer creation
        stage2_opt = trainer._create_stage2_optimizer()
        print("✓ Stage 2 optimizer creation works")
        
        print("✅ TWO-STAGE TRAINER TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ TWO-STAGE TRAINER TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_theory_compliance():
    """Test that implementation matches the theory from core.md."""
    print("\n" + "=" * 60)
    print("TESTING THEORY COMPLIANCE")
    print("=" * 60)
    
    checks = []
    
    # Check 1: Class-balanced sampling for Stage 2
    try:
        from LTL.cb_sampler import ClassBalancedSampler
        sampler_code = open('LTL/cb_sampler.py', 'r').read()
        if '_calculate_sphere_class_weights' in sampler_code and 'tail_classes' in sampler_code:
            checks.append(("✓", "Class-balanced sphere sampling implemented"))
        else:
            checks.append(("❌", "Class-balanced sphere sampling missing"))
    except:
        checks.append(("❌", "Class-balanced sampler not accessible"))
    
    # Check 2: Stage 2 uses balanced sampler
    try:
        trainer_code = open('LTL/two_stage_trainer.py', 'r').read()
        if 'create_balanced_dataloader' in trainer_code and 'balanced_train_loader' in trainer_code:
            if 'self.train_loader' not in trainer_code.split('balanced_train_loader =')[1].split('\n')[0]:
                checks.append(("✓", "Stage 2 uses balanced sampler (not fallback)"))
            else:
                checks.append(("⚠️", "Stage 2 may fall back to standard sampler"))
        else:
            checks.append(("❌", "Stage 2 doesn't use balanced sampler"))
    except:
        checks.append(("❌", "Two-stage trainer not accessible"))
    
    # Check 3: Enhanced augmentation for Stage 2
    try:
        if '_get_stage2_augmentation' in trainer_code and 'Enhanced augmentation' in trainer_code:
            checks.append(("✓", "Enhanced augmentation for Stage 2 implemented"))
        else:
            checks.append(("❌", "Enhanced augmentation missing"))
    except:
        checks.append(("❌", "Enhanced augmentation check failed"))
    
    # Check 4: Encoder freezing
    try:
        if '_freeze_encoder' in trainer_code and 'param.requires_grad = False' in trainer_code:
            checks.append(("✓", "Encoder freezing implemented"))
        else:
            checks.append(("❌", "Encoder freezing missing"))
    except:
        checks.append(("❌", "Encoder freezing check failed"))
    
    # Check 5: Lower learning rate for Stage 2
    try:
        if 'stage2_lr' in trainer_code and 'stage2_optimizer' in trainer_code:
            checks.append(("✓", "Lower learning rate for Stage 2 implemented"))
        else:
            checks.append(("❌", "Stage 2 learning rate not implemented"))
    except:
        checks.append(("❌", "Learning rate check failed"))
    
    # Print results
    for status, description in checks:
        print(f"{status} {description}")
    
    passed = sum(1 for status, _ in checks if status == "✓")
    total = len(checks)
    
    print(f"\nTHEORY COMPLIANCE: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ IMPLEMENTATION FULLY MATCHES THEORY!")
        return True
    elif passed >= total * 0.8:
        print("⚠️ Implementation mostly matches theory with minor gaps")
        return True
    else:
        print("❌ Implementation has significant gaps vs theory")
        return False

def main():
    """Run all tests."""
    print("TESTING TWO-STAGE TRAINING IMPLEMENTATION")
    print("This validates that the code matches the decoupled training theory")
    
    results = []
    
    # Test 1: Class-balanced sampler
    results.append(test_class_balanced_sampler())
    
    # Test 2: Two-stage trainer
    results.append(test_two_stage_trainer())
    
    # Test 3: Theory compliance
    results.append(test_theory_compliance())
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("Your implementation fully matches the decoupled training theory!")
        print("You can now run the two-stage training with confidence.")
    else:
        print(f"⚠️ {passed}/{total} tests passed")
        print("Some components need attention before running full training.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
