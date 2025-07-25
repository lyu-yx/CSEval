import torch
import torch.nn as nn
import torch.nn.functional as F
from models.camoformer_decoder_codsod_mask_fixation import MSA_module
from models.enhanced_components_ablation_receptive12 import AdaptiveReceptiveFieldModule, AdaptiveFeatureHarmonizer


class FeatureFusion(nn.Module):
    """Residual-gated fusion: original + 纬 路 adapted (纬 learnable, shared across channels)."""
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

        # --- Domain Path Channel Adaptation ---
        self.domain_path_conv1 = nn.Conv2d(512, channels, 3, padding=1)
        self.domain_path_conv2 = nn.Conv2d(320, channels, 3, padding=1)
        self.domain_path_conv3 = nn.Conv2d(128, channels, 3, padding=1)
        self.domain_path_conv4 = nn.Conv2d(64, channels, 3, padding=1)

        # --- Receptive Field Path Channel Adaptation (separate weights) ---
        self.rf_path_conv1 = nn.Conv2d(512, channels, 3, padding=1)
        self.rf_path_conv2 = nn.Conv2d(320, channels, 3, padding=1)
        self.rf_path_conv3 = nn.Conv2d(128, channels, 3, padding=1)

        # Feature fusion modules for original approach
        self.fusion_fb4 = FeatureFusion(512)
        self.fusion_fb3 = FeatureFusion(320)
        self.fusion_fb2 = FeatureFusion(128)

        # Adaptive Receptive Field Module
        self.adaptive_rf_module = AdaptiveReceptiveFieldModule(channels)

        # NEW: Feature harmonizers to resolve interference
        self.harmonizer_level4 = AdaptiveFeatureHarmonizer(channels)
        self.harmonizer_level3 = AdaptiveFeatureHarmonizer(channels)
        self.harmonizer_level2 = AdaptiveFeatureHarmonizer(channels)

        # Original fusion layers (now for harmonized features)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),  # Use GroupNorm for consistency
            nn.ReLU(inplace=True)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True)
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
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, originals, adapted, mask):
        # 1. Get original and adapted features
        fb4_orig, fb3_orig, fb2_orig, fb1_orig = originals['fb4'], originals['fb3'], originals['fb2'], originals['fb1']
        fb4_adapt, fb3_adapt, fb2_adapt = adapted['fb4'], adapted['fb3'], adapted['fb2']

        # 2. Process domain-adapted features through domain path
        domain_fb4 = self.domain_path_conv1(fb4_adapt)
        domain_fb3 = self.domain_path_conv2(fb3_adapt)
        domain_fb2 = self.domain_path_conv3(fb2_adapt)
        fb1_processed = self.domain_path_conv4(fb1_orig)  # fb1 is not adapted

        # 3. Process original features through RF path
        rf_fb4 = self.rf_path_conv1(fb4_orig)
        rf_fb3 = self.rf_path_conv2(fb3_orig)
        rf_fb2 = self.rf_path_conv3(fb2_orig)

        # 4. Align sizes for adaptive RF module
        if rf_fb4.size()[2:] != rf_fb3.size()[2:]:
            rf_fb4 = F.interpolate(rf_fb4, size=rf_fb3.size()[2:], mode='bilinear', align_corners=True)
        if rf_fb2.size()[2:] != rf_fb3.size()[2:]:
            rf_fb2 = F.interpolate(rf_fb2, size=rf_fb3.size()[2:], mode='bilinear', align_corners=True)

        # 5. Generate enhanced context with adaptive receptive fields
        rf_context = self.adaptive_rf_module(rf_fb4, rf_fb3, rf_fb2)

        # 6. Harmonize domain and RF features at each level to reduce interference
        # Resize RF context to match domain feature scales
        rf_context_for_4 = F.interpolate(rf_context, size=domain_fb4.size()[2:], mode='bilinear', align_corners=True)
        rf_context_for_3 = F.interpolate(rf_context, size=domain_fb3.size()[2:], mode='bilinear', align_corners=True)
        rf_context_for_2 = F.interpolate(rf_context, size=domain_fb2.size()[2:], mode='bilinear', align_corners=True)

        # Apply harmonization to resolve feature conflicts
        E4_harmonized = self.harmonizer_level4(domain_fb4, rf_context_for_4)
        E3_harmonized = self.harmonizer_level3(domain_fb3, rf_context_for_3)
        E2_harmonized = self.harmonizer_level2(domain_fb2, rf_context_for_2)

        # 7. Final fusion with RF context
        E4_fused = torch.cat((E4_harmonized, rf_context_for_4), 1)
        E3_fused = torch.cat((E3_harmonized, rf_context_for_3), 1)
        E2_fused = torch.cat((E2_harmonized, rf_context_for_2), 1)

        E4_fused = self.fuse1(E4_fused)
        E3_fused = self.fuse2(E3_fused)
        E2_fused = self.fuse3(E2_fused)

        # 8. MSA processing (using harmonized features)
        # Resize RF context to match the smallest feature map E4_fused for the first MSA stage
        rf_context_for_msa5 = F.interpolate(rf_context, size=E4_fused.size()[2:], mode='bilinear', align_corners=True)
        P5 = F.interpolate(mask, size=E4_fused.size()[2:], mode='bilinear', align_corners=True)

        D4 = self.MSA5(rf_context_for_msa5, E4_fused, P5)
        D4 = F.interpolate(D4, size=E3_fused.size()[2:], mode='bilinear', align_corners=True)
        P4 = F.interpolate(mask, size=D4.size()[2:], mode='bilinear', align_corners=True)

        D3 = self.MSA4(D4, E3_fused, P4)
        D3 = F.interpolate(D3, size=E2_fused.size()[2:], mode='bilinear', align_corners=True)
        P3 = F.interpolate(mask, size=D3.size()[2:], mode='bilinear', align_corners=True)

        D2 = self.MSA3(D3, E2_fused, P3)
        D2 = F.interpolate(D2, size=fb1_processed.size()[2:], mode='bilinear', align_corners=True)
        P2 = F.interpolate(mask, size=D2.size()[2:], mode='bilinear', align_corners=True)

        D1 = self.MSA2(D2, fb1_processed, P2)

        # Final processing
        D1 = F.interpolate(D1, size=mask.shape[2:], mode='bilinear', align_corners=True)

        # Generate enhanced fixation with semantic weighting
        base_fixation = self.semantic_fixation_conv(D1)
        fixation = base_fixation * mask

        # Generate prediction
        pred = self.output_conv1(D1)

        # Compute target prediction with semantic-aware fixation weighting
        masked_pred = pred * mask * fixation
        masked_sum = masked_pred.sum(dim=[2, 3])
        fixation_area = (mask * fixation).sum(dim=[2, 3]).clamp(min=1e-6)
        target_pred = masked_sum / fixation_area
        target_pred = target_pred.squeeze(dim=1)

        return pred, target_pred, fixation