import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Any
from models.pvtv2 import pvt_v2_b2
from models.enhanced_decoder_ablation_receptive124 import EnhancedDecoder
from models.enhanced_components_ablation_receptive124 import DomainAdapter


class EnhancedDegreeNet(nn.Module):
    """Enhanced Degree Estimation Network with improved module integration"""

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

        # Domain adaptation module
        self.domain_adapter = DomainAdapter(channel)

        # Enhanced decoder with harmonization
        self.decoder = EnhancedDecoder(channel)

        print("Initialized EnhancedDegreeNet with improved DomainAdapter/AdaptiveReceptiveField integration")

    def forward(self, x, mask):
        # Enhanced model by default, return original format for compatibility
        outputs = self.forward_enhanced(x, mask, training_mode=True)

        # Return in original format: (pred, target_pred, fixation)
        # Type checking: outputs is guaranteed to be dict when training_mode=True
        assert isinstance(outputs, dict), "Expected dict output from forward_enhanced in training mode"
        return outputs['pred'], outputs['target_pred'], outputs['fixation']

    def forward_enhanced(self, x, mask, training_mode=True):
        """Enhanced forward pass with improved module integration"""
        # Extract backbone features
        pvt_features = self.backbone(x)
        fb1, fb2, fb3, fb4 = pvt_features

        # Apply domain adaptation - keeping gradients for better convergence
        # Remove .detach() to allow gradient flow while using harmonization to reduce conflicts
        fb4_adapted, fb3_adapted, fb2_adapted, domain_logits = self.domain_adapter(
            fb4, fb3, fb2, training_mode  # Remove .detach() - harmonizer handles conflicts
        )

        # Pass original and adapted features to the decoder with harmonization
        pred, target_pred, fixation = self.decoder(
            originals={'fb4': fb4, 'fb3': fb3, 'fb2': fb2, 'fb1': fb1},
            adapted={'fb4': fb4_adapted, 'fb3': fb3_adapted, 'fb2': fb2_adapted},
            mask=mask
        )

        # Return comprehensive outputs for enhanced training
        if training_mode:
            return_dict = {
                'pred': pred,
                'target_pred': target_pred,
                'fixation': fixation,
                'pvt_features': pvt_features,
                'domain_logits': domain_logits
            }
            return return_dict
        else:
            # Inference mode - return simplified outputs
            return pred, target_pred, fixation

    def get_harmonization_weights(self):
        """Get current harmonization weights for analysis"""
        weights = {}
        for name, module in self.decoder.named_modules():
            if 'harmonizer' in name and hasattr(module, 'fusion_weight'):
                weights[name] = torch.sigmoid(module.fusion_weight).item()
        return weights

# Backward compatibility alias
UEDGNet = EnhancedDegreeNet

if __name__ == '__main__':
    # Test the improved EnhancedDegreeNet
    net = EnhancedDegreeNet(channel=128).eval()
    inputs = torch.randn(1, 3, 352, 352)
    mask = torch.randn(1, 1, 352, 352)

    print("Testing improved EnhancedDegreeNet with harmonization...")

    # Test enhanced forward pass (training mode)
    outputs = net.forward_enhanced(inputs, mask, training_mode=True)
    print("Enhanced outputs (training mode):")
    print(f"pred shape: {outputs['pred'].shape}")
    print(f"target_pred shape: {outputs['target_pred'].shape}")
    print(f"fixation shape: {outputs['fixation'].shape}")
    if outputs['domain_logits'] is not None:
        print(f"domain_logits shape: {outputs['domain_logits'].shape}")

    # Check harmonization weights
    harmony_weights = net.get_harmonization_weights()
    print(f"\nHarmonization weights: {harmony_weights}")

    # Test enhanced forward pass (inference mode)
    pred, target_pred, fixation = net.forward_enhanced(inputs, mask, training_mode=False)
    print(f"\nInference mode outputs:")
    print(f"pred shape: {pred.shape}")
    print(f"target_pred shape: {target_pred.shape}")
    print(f"fixation shape: {fixation.shape}")

    # Test standard forward (backward compatibility)
    print(f"\nTesting standard forward (backward compatibility):")
    pred, target_pred, fixation = net(inputs, mask)
    print(f"pred shape: {pred.shape}")
    print(f"target_pred shape: {target_pred.shape}")
    print(f"fixation shape: {fixation.shape}")

    print("\nImproved model test completed successfully!")