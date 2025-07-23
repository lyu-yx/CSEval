import torch
import torch.nn as nn
import torch.nn.functional as F
from models.camoformer_decoder_codsod_mask_fixation import MSA_module
from models.enhanced_components_dofixre9 import AdaptiveReceptiveFieldModule


class FeatureFusion(nn.Module):
    """Residual-gated fusion: original + γ · adapted (γ learnable, shared across channels)."""
    def __init__(self, channels):
        super().__init__()
        # Single scalar gate; initialise small so original features dominate early training
        self.gamma = nn.Parameter(torch.tensor(0.1))

        # Separate normalisation for stability
        self.norm_orig = nn.InstanceNorm2d(channels, affine=True)
        self.norm_adapt = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, original_feature, adapted_feature):
        # Normalise both tensors
        orig_n = self.norm_orig(original_feature)
        adapt_n = self.norm_adapt(adapted_feature)

        # Sigmoid to keep gate in [0,1]
        gamma = torch.sigmoid(self.gamma)

        # Residual-gated fusion
        return orig_n + gamma * adapt_n


class EnhancedDecoder(nn.Module):
    """Enhanced decoder with semantic-aware fixation and adaptive receptive fields"""

    def __init__(self, channels):
        super().__init__()

        # --- Main Path Channel Adaptation ---
        self.side_conv1 = nn.Conv2d(512, channels, 3, padding=1)
        self.side_conv2 = nn.Conv2d(320, channels, 3, padding=1)
        self.side_conv3 = nn.Conv2d(128, channels, 3, padding=1)
        self.side_conv4 = nn.Conv2d(64, channels, 3, padding=1)

        # --- Receptive Field Path Channel Adaptation (separate weights) ---
        self.rf_side_conv1 = nn.Conv2d(512, channels, 3, padding=1)
        self.rf_side_conv2 = nn.Conv2d(320, channels, 3, padding=1)
        self.rf_side_conv3 = nn.Conv2d(128, channels, 3, padding=1)

        # Feature fusion modules moved inside the decoder
        self.fusion_fb4 = FeatureFusion(512)
        self.fusion_fb3 = FeatureFusion(320)
        self.fusion_fb2 = FeatureFusion(128)

        self.adaptive_rf_module = AdaptiveReceptiveFieldModule(channels)

        # Original fusion layers
        self.fuse1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # MSA modules (keeping original attention)
        self.MSA5 = MSA_module(dim=channels)
        self.MSA4 = MSA_module(dim=channels)
        self.MSA3 = MSA_module(dim=channels)
        self.MSA2 = MSA_module(dim=channels)

        # Output layers
        self.output_conv1 = nn.Conv2d(channels, 1, 1)

        # Enhanced semantic-aware fixation prediction
        self.semantic_fixation_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, originals, adapted, mask):
        # 1. Get original and adapted features
        fb4_orig, fb3_orig, fb2_orig, fb1_orig = originals['fb4'], originals['fb3'], originals['fb2'], originals['fb1']
        fb4_adapt, fb3_adapt, fb2_adapt = adapted['fb4'], adapted['fb3'], adapted['fb2']

        # 2. Fuse them
        # Lightweight alignment (GroupNorm) on adapted features before fusion
        fb4_adapt = self.fusion_fb4.norm_adapt(fb4_adapt)
        fb3_adapt = self.fusion_fb3.norm_adapt(fb3_adapt)
        fb2_adapt = self.fusion_fb2.norm_adapt(fb2_adapt)

        E4 = self.fusion_fb4(fb4_orig, fb4_adapt)
        E3 = self.fusion_fb3(fb3_orig, fb3_adapt)
        E2 = self.fusion_fb2(fb2_orig, fb2_adapt)
        E1 = fb1_orig  # fb1 is not adapted, so no fusion

        # 3. Perform channel adaptation ONCE on the fused features
        E4_c = self.side_conv1(E4)
        E3_c = self.side_conv2(E3)
        E2_c = self.side_conv3(E2)
        E1_c = self.side_conv4(E1)

        # 4. Run Adaptive RF Module on ORIGINAL features (after channel adaptation)
        fb4_orig_c = self.rf_side_conv1(fb4_orig)
        fb3_orig_c = self.rf_side_conv2(fb3_orig)
        fb2_orig_c = self.rf_side_conv3(fb2_orig)

        # Align sizes for adaptive RF module
        if fb4_orig_c.size()[2:] != fb3_orig_c.size()[2:]:
            fb4_orig_c = F.interpolate(fb4_orig_c, size=fb3_orig_c.size()[2:], mode='bilinear', align_corners=True)
        if fb2_orig_c.size()[2:] != fb3_orig_c.size()[2:]:
            fb2_orig_c = F.interpolate(fb2_orig_c, size=fb3_orig_c.size()[2:], mode='bilinear', align_corners=True)

        # Enhanced context modeling with adaptive receptive fields on original features
        E5 = self.adaptive_rf_module(fb4_orig_c, fb3_orig_c, fb2_orig_c)

        # Use semantic features for further processing, resizing E5 to match each feature map
        E4_fused = torch.cat((E4_c, F.interpolate(E5, size=E4_c.size()[2:], mode='bilinear', align_corners=True)), 1)
        E3_fused = torch.cat((E3_c, F.interpolate(E5, size=E3_c.size()[2:], mode='bilinear', align_corners=True)), 1)
        E2_fused = torch.cat((E2_c, F.interpolate(E5, size=E2_c.size()[2:], mode='bilinear', align_corners=True)), 1)

        E4_fused = F.relu(self.fuse1(E4_fused), inplace=True)
        E3_fused = F.relu(self.fuse2(E3_fused), inplace=True)
        E2_fused = F.relu(self.fuse3(E2_fused), inplace=True)

        # MSA processing (using enhanced features)
        # Resize E5 to match the smallest feature map E4_fused for the first MSA stage
        E5_for_msa5 = F.interpolate(E5, size=E4_fused.size()[2:], mode='bilinear', align_corners=True)
        P5 = F.interpolate(mask, size=E4_fused.size()[2:], mode='bilinear', align_corners=True)

        D4 = self.MSA5(E5_for_msa5, E4_fused, P5)
        D4 = F.interpolate(D4, size=E3_fused.size()[2:], mode='bilinear', align_corners=True)
        P4 = F.interpolate(mask, size=D4.size()[2:], mode='bilinear', align_corners=True)

        D3 = self.MSA4(D4, E3_fused, P4)
        D3 = F.interpolate(D3, size=E2_fused.size()[2:], mode='bilinear', align_corners=True)
        P3 = F.interpolate(mask, size=D3.size()[2:], mode='bilinear', align_corners=True)

        D2 = self.MSA3(D3, E2_fused, P3)
        D2 = F.interpolate(D2, size=E1_c.size()[2:], mode='bilinear', align_corners=True)
        P2 = F.interpolate(mask, size=D2.size()[2:], mode='bilinear', align_corners=True)

        D1 = self.MSA2(D2, E1_c, P2)

        # Final processing
        D1 = F.interpolate(D1, size=mask.shape[2:], mode='bilinear', align_corners=True)

        # Generate enhanced fixation with semantic weighting
        base_fixation = self.semantic_fixation_conv(D1)
        fixation = base_fixation * mask

        # Generate prediction
        pred = self.output_conv1(D1)

        # Compute target prediction with semantic-aware fixation weighting
        # mask_expanded = mask.expand(-1, 10, -1, -1)
        # fixation_expanded = fixation.expand(-1, 10, -1, -1)

        masked_pred = pred * mask * fixation
        masked_sum = masked_pred.sum(dim=[2, 3])
        fixation_area = (mask * fixation).sum(dim=[2, 3]).clamp(min=1e-6)
        target_pred = masked_sum / fixation_area
        target_pred = target_pred.squeeze(dim=1)  # 

        return pred, target_pred, fixation