import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Any
from models.pvtv2 import pvt_v2_b2
from models.enhanced_decoder_dofixre9 import EnhancedDecoder
from models.enhanced_components_dofixre9 import DomainAdapter, AdaptiveReceptiveFieldModule


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

        # Domain adaptation
        self.domain_adapter = DomainAdapter(channel)
        # Adaptive Receptive Field Module (ARFM)
        self.arfm = AdaptiveReceptiveFieldModule(channel)
        # Gating parameter for feature fusion (learnable scalar)
        self.gate_param = nn.Parameter(torch.tensor(0.1))
        # LayerNorm for fused features
        self.ln_fb4 = nn.LayerNorm([channel, 22, 22])  # assuming fb4 spatial size, adjust as needed
        self.ln_fb3 = nn.LayerNorm([channel, 44, 44])  # adjust as needed
        self.ln_fb2 = nn.LayerNorm([channel, 88, 88])  # adjust as needed
        # Enhanced decoder
        self.decoder = EnhancedDecoder(channel)
        # Projections to ensure all ARFM inputs have 'channel' channels
        self.proj_fb4 = nn.Conv2d(512, channel, 1)
        self.proj_fb3 = nn.Conv2d(320, channel, 1)
        self.proj_fb2 = nn.Conv2d(128, channel, 1)
        self.proj_fb1 = nn.Conv2d(64, channel, 1)

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

        # Apply domain adaptation on *detached* features to avoid gradient conflict with backbone
        fb4_adapted, fb3_adapted, fb2_adapted, domain_logits = self.domain_adapter(
            fb4.detach(), fb3.detach(), fb2.detach(), training_mode
        )

        # Project all features to 'channel' channels
        fb4_proj = self.proj_fb4(fb4_adapted.detach())
        fb3_proj = self.proj_fb3(fb3_adapted.detach())
        fb2_proj = self.proj_fb2(fb2_adapted.detach())
        fb1_proj = self.proj_fb1(fb1.detach())

        # --- Ensure all ARFM inputs have the same spatial size ---
        # For fb4_arfm, upsample fb3_proj and fb2_proj to fb4_proj's size
        target_size_fb4 = fb4_proj.shape[2:]
        fb3_proj_up4 = F.interpolate(fb3_proj, size=target_size_fb4, mode='bilinear', align_corners=False)
        fb2_proj_up4 = F.interpolate(fb2_proj, size=target_size_fb4, mode='bilinear', align_corners=False)
        fb4_arfm = self.arfm(fb4_proj, fb3_proj_up4, fb2_proj_up4)

        # For fb3_arfm, upsample fb2_proj and fb1_proj to fb3_proj's size
        target_size_fb3 = fb3_proj.shape[2:]
        fb2_proj_up3 = F.interpolate(fb2_proj, size=target_size_fb3, mode='bilinear', align_corners=False)
        fb1_proj_up3 = F.interpolate(fb1_proj, size=target_size_fb3, mode='bilinear', align_corners=False)
        fb3_arfm = self.arfm(fb3_proj, fb2_proj_up3, fb1_proj_up3)

        # For fb2_arfm, upsample fb1_proj twice to fb2_proj's size
        target_size_fb2 = fb2_proj.shape[2:]
        fb1_proj_up2a = F.interpolate(fb1_proj, size=target_size_fb2, mode='bilinear', align_corners=False)
        fb1_proj_up2b = F.interpolate(fb1_proj, size=target_size_fb2, mode='bilinear', align_corners=False)
        fb2_arfm = self.arfm(fb2_proj, fb1_proj_up2a, fb1_proj_up2b)

        # Gated fusion of adapted and ARFM features
        alpha = torch.sigmoid(self.gate_param)
        def norm_fused(feat, fused):
            shape = fused.shape
            ln = nn.LayerNorm(shape[1:], device=fused.device, dtype=fused.dtype)
            return ln(feat)
        fb4_fused = norm_fused(alpha * fb4_proj + (1 - alpha) * fb4_arfm, fb4_arfm)
        fb3_fused = norm_fused(alpha * fb3_proj + (1 - alpha) * fb3_arfm, fb3_arfm)
        fb2_fused = norm_fused(alpha * fb2_proj + (1 - alpha) * fb2_arfm, fb2_arfm)

        # Pass original backbone features as originals, domain-adapted as adapted
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
    print(f"fixation shape: {outputs['fixation'].shape}")
    if outputs['domain_logits'] is not None:
        print(f"domain_logits shape: {outputs['domain_logits'].shape}")

    # Test enhanced forward pass (inference mode)
    pred, target_pred, fixation = net.forward_enhanced(inputs, mask, training_mode=False)
    print(f"\nInference mode outputs:")
    print(f"pred shape: {pred.shape}")
    print(f"target_pred shape: {target_pred.shape}")
    print(f"fixation shape: {fixation.shape}")

    # Test standard forward (backward compatibility)
    print(f"\nTesting standard forward (backward compatibility):")
    pred, target_pred, fixation = net(inputs, mask)
    print(f"EnhancedDegreeNet pred shape: {pred.shape}")
    print(f"EnhancedDegreeNet target_pred shape: {target_pred.shape}")
    print(f"EnhancedDegreeNet fixation shape: {fixation.shape}")