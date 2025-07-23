import torch
import torch.nn as nn
import torch.nn.functional as F
from models.camoformer_decoder_codsod_mask_fixation import MSA_module
from models.enhanced_components_dofixre3 import AdaptiveReceptiveFieldModule


class EnhancedDecoder(nn.Module):
    """Enhanced decoder with semantic-aware fixation and adaptive receptive fields"""

    def __init__(self, channels):
        super().__init__()

        # Original features processing (for AdaptiveReceptiveField) - 使用Conv+BN+ReLU
        self.side_conv1_orig = nn.Sequential(
            nn.Conv2d(512, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.side_conv2_orig = nn.Sequential(
            nn.Conv2d(320, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.side_conv3_orig = nn.Sequential(
            nn.Conv2d(128, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Adapted features processing (for fusion) - 独立的卷积层
        self.side_conv1_adapted = nn.Sequential(
            nn.Conv2d(512, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.side_conv2_adapted = nn.Sequential(
            nn.Conv2d(320, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.side_conv3_adapted = nn.Sequential(
            nn.Conv2d(128, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # E1 processing (unchanged, no adaptation needed)
        self.side_conv4 = nn.Sequential(
            nn.Conv2d(64, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.adaptive_rf_module = AdaptiveReceptiveFieldModule(channels)

        # Fusion layers for combining adapted features with semantic features
        self.fuse1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
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

    def forward(self, E4_orig, E3_orig, E2_orig, E1, E4_adapted, E3_adapted, E2_adapted, mask):
        # Process original features (for AdaptiveReceptiveField)
        E4_orig_processed = self.side_conv1_orig(E4_orig)
        E3_orig_processed = self.side_conv2_orig(E3_orig)
        E2_orig_processed = self.side_conv3_orig(E2_orig)
        E1_processed = self.side_conv4(E1)

        # Process adapted features (for fusion)
        E4_adapted_processed = self.side_conv1_adapted(E4_adapted)
        E3_adapted_processed = self.side_conv2_adapted(E3_adapted)
        E2_adapted_processed = self.side_conv3_adapted(E2_adapted)

        # Align sizes for adaptive RF module (only for original features)
        if E4_orig_processed.size()[2:] != E3_orig_processed.size()[2:]:
            E4_orig_processed = F.interpolate(E4_orig_processed, size=E3_orig_processed.size()[2:], mode='bilinear', align_corners=True)
        if E2_orig_processed.size()[2:] != E3_orig_processed.size()[2:]:
            E2_orig_processed = F.interpolate(E2_orig_processed, size=E3_orig_processed.size()[2:], mode='bilinear', align_corners=True)

        # Enhanced context modeling with adaptive receptive fields (using pure original features)
        E5 = self.adaptive_rf_module(E4_orig_processed, E3_orig_processed, E2_orig_processed)

        # Prepare E5 for fusion with adapted features at their natural scales
        E5_for_E4 = F.interpolate(E5, size=E4_adapted_processed.size()[2:], mode='bilinear', align_corners=True)
        E5_for_E3 = F.interpolate(E5, size=E3_adapted_processed.size()[2:], mode='bilinear', align_corners=True)
        E5_for_E2 = F.interpolate(E5, size=E2_adapted_processed.size()[2:], mode='bilinear', align_corners=True)

        # Fusion: combine adapted features with semantic features
        E4_fused = torch.cat((E4_adapted_processed, E5_for_E4), 1)
        E3_fused = torch.cat((E3_adapted_processed, E5_for_E3), 1)
        E2_fused = torch.cat((E2_adapted_processed, E5_for_E2), 1)

        E4_fused = self.fuse1(E4_fused)
        E3_fused = self.fuse2(E3_fused)
        E2_fused = self.fuse3(E2_fused)

        # Progressive MSA fusion: 从语义到细节的渐进融合
        # Start with E5 (highest semantic level)
        D5 = E5
        
        # Progressive integration: E4 queries semantic context from E5
        D5_for_E4 = F.interpolate(D5, size=E4_fused.size()[2:], mode='bilinear', align_corners=True)
        P4 = F.interpolate(mask, size=E4_fused.size()[2:], mode='bilinear', align_corners=True)
        D4 = self.MSA5(E4_fused, D5_for_E4, P4)  # E4 as query, E5 as key/value
        
        # E3 queries enhanced context from D4
        D4_for_E3 = F.interpolate(D4, size=E3_fused.size()[2:], mode='bilinear', align_corners=True)
        P3 = F.interpolate(mask, size=E3_fused.size()[2:], mode='bilinear', align_corners=True)
        D3 = self.MSA4(E3_fused, D4_for_E3, P3)  # E3 as query, D4 as key/value
        
        # E2 queries enhanced context from D3
        D3_for_E2 = F.interpolate(D3, size=E2_fused.size()[2:], mode='bilinear', align_corners=True)
        P2 = F.interpolate(mask, size=E2_fused.size()[2:], mode='bilinear', align_corners=True)
        D2 = self.MSA3(E2_fused, D3_for_E2, P2)  # E2 as query, D3 as key/value
        
        # Final integration: E1 queries enhanced context from D2
        D2_for_E1 = F.interpolate(D2, size=E1_processed.size()[2:], mode='bilinear', align_corners=True)
        P1 = F.interpolate(mask, size=E1_processed.size()[2:], mode='bilinear', align_corners=True)
        D1 = self.MSA2(E1_processed, D2_for_E1, P1)  # E1 as query, D2 as key/value

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
        target_pred = target_pred.squeeze(dim=1)  # ѹ����1ά��C=1��ά�ȣ�

        return pred, target_pred, fixation