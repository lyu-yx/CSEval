import torch
import torch.nn as nn
import torch.nn.functional as F
from models.camoformer_decoder_codsod_mask_fixation import MSA_module
from models.enhanced_components_dofixre6 import AdaptiveReceptiveFieldModule, DomainAdapter


class EnhancedDecoder(nn.Module):
    """Enhanced decoder with semantic-aware fixation and adaptive receptive fields"""

    def __init__(self, channels):
        super().__init__()

        # Channel adaptation layers (matching the original side_convs)
        self.side_conv1 = nn.Conv2d(512, channels, 3, padding=1)
        self.side_conv2 = nn.Conv2d(320, channels, 3, padding=1)
        self.side_conv3 = nn.Conv2d(128, channels, 3, padding=1)
        self.side_conv4 = nn.Conv2d(64, channels, 3, padding=1)

        self.adaptive_rf_module4 = AdaptiveReceptiveFieldModule(channels)
        self.adaptive_rf_module3 = AdaptiveReceptiveFieldModule(channels)
        self.adaptive_rf_module2 = AdaptiveReceptiveFieldModule(channels)


        self.domain_adapter = DomainAdapter(channels)

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

    def forward(self, E4, E3, E2, E1, mask):
        # Process through side convs (channel adaptation)
        E4 = self.side_conv1(E4)
        E3 = self.side_conv2(E3)
        E2 = self.side_conv3(E2)
        E1 = self.side_conv4(E1)

        E4_ad, E3_ad, E2_ad, domain_logits = self.domain_adapter(E4, E3, E2)

        # Enhanced context modeling with adaptive receptive fields
        E4_rf = self.adaptive_rf_module4(E4_ad)
        E3_rf = self.adaptive_rf_module3(E3_ad)
        E2_rf = self.adaptive_rf_module2(E2_ad)

        # Use semantic features for further processing

        E4 = torch.cat((E4, E4_rf), 1)
        E3 = torch.cat((E3, E3_rf), 1)
        E2 = torch.cat((E2, E2_rf), 1)

        E4 = F.relu(self.fuse1(E4), inplace=True)
        E3 = F.relu(self.fuse2(E3), inplace=True)
        E2 = F.relu(self.fuse3(E2), inplace=True)

        # MSA processing (using enhanced features)

        P4 = F.interpolate(mask, size=E4.size()[2:], mode='bilinear', align_corners=True)
        E4 = F.interpolate(E3, size=E3.size()[2:], mode='bilinear', align_corners=True)

        D3 = self.MSA4(E4, E3, P4)
        D3 = F.interpolate(D3, size=E2.size()[2:], mode='bilinear', align_corners=True)
        P3 = F.interpolate(mask, size=D3.size()[2:], mode='bilinear', align_corners=True)

        D2 = self.MSA3(D3, E2, P3)
        D2 = F.interpolate(D2, size=E1.size()[2:], mode='bilinear', align_corners=True)
        P2 = F.interpolate(mask, size=D2.size()[2:], mode='bilinear', align_corners=True)

        D1 = self.MSA2(D2, E1, P2)

        # Final processing
        D1 = F.interpolate(D1, size=mask.shape[2:], mode='bilinear', align_corners=True)

        # Generate enhanced fixation with semantic weighting
        base_fixation = self.semantic_fixation_conv(D1)
        fixation = base_fixation * mask

        # Generate prediction
        pred = self.output_conv1(D1)

        masked_pred = pred * mask * fixation
        masked_sum = masked_pred.sum(dim=[2, 3])
        fixation_area = (mask * fixation).sum(dim=[2, 3]).clamp(min=1e-6)
        target_pred = masked_sum / fixation_area
        target_pred = target_pred.squeeze(dim=1)

        return pred, target_pred, fixation, domain_logits