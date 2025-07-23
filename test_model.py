#!/usr/bin/env python3
"""Test script to verify model components work without dimension errors"""

import torch
import sys

def test_model_components():
    try:
        from models.CamoEval_enhanced_dofixre9 import EnhancedDegreeNet as Network
        from models.enhanced_components_dofixre9 import StreamlinedLoss
        print('✓ Model imports successful')
        
        # Test model creation
        model = Network(channel=128)
        print('✓ Model creation successful')
        
        # Test criterion creation  
        criterion = StreamlinedLoss(alpha_perceptual=1.0, alpha_domain=50.0)
        print('✓ Criterion creation successful')

        # Test forward pass with dummy data
        dummy_images = torch.randn(2, 3, 352, 352)
        dummy_mask = torch.randn(2, 1, 352, 352)

        model.eval()
        with torch.no_grad():
            try:
                outputs = model.forward_enhanced(dummy_images, dummy_mask, training_mode=False)
                print('✓ Forward pass successful')
                print(f'  Output keys: {list(outputs.keys()) if isinstance(outputs, dict) else "tuple"}')
                
                if isinstance(outputs, dict):
                    pred = outputs.get('pred')
                    target_pred = outputs.get('target_pred')
                    fixation = outputs.get('fixation')
                    
                    if pred is not None:
                        print(f'  pred shape: {pred.shape}')
                    if target_pred is not None:
                        print(f'  target_pred shape: {target_pred.shape}')
                    if fixation is not None:
                        print(f'  fixation shape: {fixation.shape}')
                        
            except Exception as e:
                print(f'✗ Forward pass failed: {e}')
                import traceback
                traceback.print_exc()
                return False

        print('✓ All components loaded and tested successfully!')
        return True

    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_model_components()
    if success:
        print('\n🎉 All tests passed! No dimension errors detected.')
    else:
        print('\n❌ Tests failed. Please check the error messages above.')
        sys.exit(1)
