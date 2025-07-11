#!/usr/bin/env python3
"""
Test script for enhanced camouflage scoring model
This script verifies that all enhanced components work correctly
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_components():
    """Test individual enhanced components"""
    print("=== Testing Enhanced Components ===")
    
    try:
        from models.enhanced_components import (
            UnsupervisedSemanticDiscovery, 
            AdaptiveReceptiveFieldModule,
            DomainAdapter,
            StreamlinedLoss
        )
        print("✓ Successfully imported enhanced components")
    except ImportError as e:
        print(f"✗ Failed to import enhanced components: {e}")
        return False
    
    # Test semantic discovery
    try:
        semantic_module = UnsupervisedSemanticDiscovery(in_channels=128)
        features = torch.randn(2, 128, 32, 32)
        mask = torch.randn(2, 1, 32, 32)
        
        enhanced_features, semantic_weights, part_attention = semantic_module(features, mask)
        print(f"✓ Semantic Discovery: {enhanced_features.shape}, {semantic_weights.shape}, {part_attention.shape}")
    except Exception as e:
        print(f"✗ Semantic Discovery failed: {e}")
        return False
    
    # Test adaptive RF module
    try:
        rf_module = AdaptiveReceptiveFieldModule(channels=128)
        input1 = torch.randn(2, 128, 32, 32)
        input2 = torch.randn(2, 128, 32, 32)
        input3 = torch.randn(2, 128, 32, 32)
        
        output = rf_module(input1, input2, input3)
        print(f"✓ Adaptive RF Module: {output.shape}")
    except Exception as e:
        print(f"✗ Adaptive RF Module failed: {e}")
        return False
    
    # Test domain adapter
    try:
        domain_adapter = DomainAdapter(channels=128)
        features = torch.randn(2, 128, 32, 32)
        
        adapted_features, domain_logits = domain_adapter(features, training_mode=True)
        print(f"✓ Domain Adapter: {adapted_features.shape}, {domain_logits.shape if domain_logits is not None else 'None'}")
    except Exception as e:
        print(f"✗ Domain Adapter failed: {e}")
        return False
    
    # Test streamlined loss
    try:
        criterion = StreamlinedLoss()
        
        # Create mock outputs
        outputs = {
            'pred': torch.randn(2, 10, 352, 352),
            'target_pred': torch.randn(2, 10),
            'fixation': torch.randn(2, 1, 352, 352),
            'pvt_features': [torch.randn(2, 64, 88, 88), torch.randn(2, 128, 44, 44)],
            'part_attention': torch.randn(2, 6, 32, 32),
            'domain_logits': torch.randn(2, 2)
        }
        targets = torch.randint(0, 10, (2,)).float()
        images = torch.randn(2, 3, 352, 352)
        masks = torch.randn(2, 1, 352, 352)
        domain_labels = torch.randint(0, 2, (2,))
        
        loss, loss_dict = criterion(outputs, targets, images, masks, domain_labels)
        print(f"✓ Streamlined Loss: {loss.item():.4f}")
        print(f"  Loss components: {loss_dict}")
    except Exception as e:
        print(f"✗ Streamlined Loss failed: {e}")
        return False
    
    return True

def test_enhanced_decoder():
    """Test enhanced decoder"""
    print("\n=== Testing Enhanced Decoder ===")
    
    try:
        from models.enhanced_decoder import EnhancedDecoder
        print("✓ Successfully imported enhanced decoder")
    except ImportError as e:
        print(f"✗ Failed to import enhanced decoder: {e}")
        return False
    
    try:
        decoder = EnhancedDecoder(channels=128)
        
        # Create mock inputs
        E4 = torch.randn(2, 512, 11, 11)
        E3 = torch.randn(2, 320, 22, 22)
        E2 = torch.randn(2, 128, 44, 44)
        E1 = torch.randn(2, 64, 88, 88)
        mask = torch.randn(2, 1, 352, 352)
        
        pred, target_pred, fixation, part_attention = decoder(E4, E3, E2, E1, mask)
        
        print(f"✓ Enhanced Decoder outputs:")
        print(f"  pred: {pred.shape}")
        print(f"  target_pred: {target_pred.shape}")
        print(f"  fixation: {fixation.shape}")
        print(f"  part_attention: {part_attention.shape}")
        
    except Exception as e:
        print(f"✗ Enhanced Decoder failed: {e}")
        return False
    
    return True

def test_enhanced_model():
    """Test the complete enhanced model"""
    print("\n=== Testing Enhanced Degree Model ===")
    
    try:
        from models.CamoEval_enhanced import EnhancedDegreeNet
        print("✓ Successfully imported Enhanced Degree Model (EnhancedDegreeNet)")
    except ImportError as e:
        print(f"✗ Failed to import Enhanced Degree Model: {e}")
        return False
    
    try:
        # Test enhanced model
        model = EnhancedDegreeNet(channel=128)
        
        # Create mock inputs
        images = torch.randn(2, 3, 352, 352)
        masks = torch.randn(2, 1, 352, 352)
        
        # Test enhanced forward pass (training mode)
        outputs = model.forward_enhanced(images, masks, training_mode=True)
        print(f"✓ Enhanced Degree Model (training mode):")
        print(f"  pred: {outputs['pred'].shape}")
        print(f"  target_pred: {outputs['target_pred'].shape}")
        print(f"  fixation: {outputs['fixation'].shape}")
        print(f"  part_attention: {outputs['part_attention'].shape}")
        print(f"  domain_logits: {outputs['domain_logits'].shape if outputs['domain_logits'] is not None else 'None'}")
        
        # Test enhanced forward pass (inference mode)
        pred, target_pred, fixation = model.forward_enhanced(images, masks, training_mode=False)
        print(f"✓ Enhanced Degree Model (inference mode):")
        print(f"  pred: {pred.shape}")
        print(f"  target_pred: {target_pred.shape}")
        print(f"  fixation: {fixation.shape}")
        
    except Exception as e:
        print(f"✗ Enhanced Degree Model failed: {e}")
        return False
    
    try:
        # Test EnhancedDegreeNet (now enhanced by default)
        main_model = EnhancedDegreeNet(channel=128)
        pred, target_pred, fixation = main_model(images, masks)
        print(f"✓ EnhancedDegreeNet (Enhanced by Default):")
        print(f"  pred: {pred.shape}")
        print(f"  target_pred: {target_pred.shape}")  
        print(f"  fixation: {fixation.shape}")
        
    except Exception as e:
        print(f"✗ EnhancedDegreeNet (Enhanced by Default) failed: {e}")
        return False
    
    return True

def test_training_compatibility():
    """Test compatibility with existing training pipeline"""
    print("\n=== Testing Training Compatibility ===")
    
    try:
        from models.CamoEval_enhanced import EnhancedDegreeNet
        from models.enhanced_components import StreamlinedLoss
        
        # Test with Enhanced Degree Model (default)
        model = EnhancedDegreeNet(channel=128)
        criterion = StreamlinedLoss()
        
        # Mock training data
        images = torch.randn(4, 3, 352, 352)
        masks = torch.randn(4, 1, 352, 352)
        targets = torch.randint(1, 11, (4,)).float() - 1  # Convert to 0-9 range
        
        # Forward pass
        if hasattr(model, 'forward_enhanced'):
            outputs = model.forward_enhanced(images, masks, training_mode=True)
            loss, loss_dict = criterion(outputs, targets, images, masks)
        else:
            outputs = model(images, masks)
            # Fallback to simple MSE loss
            loss = F.mse_loss(outputs[1], targets)
            loss_dict = {'main_loss': loss.item()}
        
        print(f"✓ Training compatibility test passed:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Loss components: {loss_dict}")
        
        # Test backward pass
        loss.backward()
        print("✓ Backward pass successful")
        
    except Exception as e:
        print(f"✗ Training compatibility test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced Degree Model (Default)")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    tests = [
        test_enhanced_components,
        test_enhanced_decoder,
        test_enhanced_model,
        test_training_compatibility
    ]
    
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Enhanced Degree Model is ready to use.")
        print("\nNext steps:")
        print("1. Run with Enhanced Degree Model (default): python MyTrain_Enhanced.py")
        print("2. Disable enhanced features if needed: python MyTrain_Enhanced.py --disable_enhanced")
        print("3. Adjust loss weights with: --alpha_perceptual 0.1 --alpha_contrast 0.05")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 