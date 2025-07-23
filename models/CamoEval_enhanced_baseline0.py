import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Any
from models.pvtv2 import pvt_v2_b2
from models.enhanced_decoder_baseline0 import EnhancedDecoder


class EnhancedDegreeNet(nn.Module):
    """Enhanced Degree Estimation Network - Default enhanced model"""

    def __init__(self, channel=128):
        super().__init__()

        # Backbone (single backbone as requested)
        self.backbone = pvt_v2_b2()

        # Load pretrained weights
        path = './pre_trained/pvt_v2_b2.pth'
        try:
            save_model = torch.load(path, map_location='cpu')
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
            print(f"Successfully loaded pretrained weights from {path}")
        except FileNotFoundError:
            print(f"Warning: Pretrained weights not found at {path}. Training from scratch.")


        # Enhanced decoder
        self.decoder = EnhancedDecoder(channel)

    def forward(self, x, mask):
        # Enhanced model by default, return original format for compatibility
        outputs = self.forward_enhanced(x, mask, training_mode=True)

        # Return in original format: (pred, target_pred, fixation)
        # Type checking: outputs is guaranteed to be dict when training_mode=True
        assert isinstance(outputs, dict), "Expected dict output from forward_enhanced in training mode"
        return outputs['pred'], outputs['target_pred'], outputs['fixation']

    def forward_enhanced(self, x, mask, training_mode=True):
        """Enhanced forward pass with all outputs"""
        # Extract backbone features
        pvt_features = self.backbone(x)
        fb1, fb2, fb3, fb4 = pvt_features


        # Process through enhanced decoder
        pred, target_pred = self.decoder(
            fb4, fb3, fb2, fb1, mask
        )

        # Return comprehensive outputs for enhanced training
        if training_mode:
            return_dict = {
                'pred': pred,
                'target_pred': target_pred,
                'pvt_features': pvt_features
            }
            return return_dict
        else:
            # Inference mode - return simplified outputs
            return pred, target_pred


# Backward compatibility alias
UEDGNet = EnhancedDegreeNet

if __name__ == '__main__':
    # Test the EnhancedDegreeNet (Enhanced Degree Model)
    net = EnhancedDegreeNet(channel=128).eval()
    inputs = torch.randn(1, 3, 352, 352)
    mask = torch.randn(1, 1, 352, 352)

    # Test enhanced forward pass (training mode)
    outputs = net.forward_enhanced(inputs, mask, training_mode=True)
    print("EnhancedDegreeNet enhanced outputs:")
    print(f"pred shape: {outputs['pred'].shape}")
    print(f"target_pred shape: {outputs['target_pred'].shape}")

    # Test enhanced forward pass (inference mode)
    pred, target_pred = net.forward_enhanced(inputs, mask, training_mode=False)
    print(f"\nInference mode outputs:")
    print(f"pred shape: {pred.shape}")
    print(f"target_pred shape: {target_pred.shape}")

    # Test standard forward (backward compatibility)
    print(f"\nTesting standard forward (backward compatibility):")
    pred, target_pred = net(inputs, mask)
    print(f"EnhancedDegreeNet pred shape: {pred.shape}")
    print(f"EnhancedDegreeNet target_pred shape: {target_pred.shape}")
