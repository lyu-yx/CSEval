import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UnsupervisedSemanticDiscovery(nn.Module):
    """Discover semantic parts without labels using clustering in feature space"""

    def __init__(self, in_channels=128, num_semantic_parts=6, alpha_proto=0.5):
        super().__init__()
        self.num_parts = num_semantic_parts
        self.proto_weight = alpha_proto
        self.conv_weight = 1 - alpha_proto

        # Part discovery through learnable prototypes
        self.part_prototypes = nn.Parameter(torch.randn(num_semantic_parts, in_channels))
        self.part_attention = nn.Conv2d(in_channels, num_semantic_parts, 1)

        # Learnable weight for combining conv and prototype features
        self.combine_weight = nn.Parameter(
            torch.tensor([self.proto_weight, self.conv_weight]))  # [conv_weight, proto_weight]

        self.part_refine = nn.Sequential(
            nn.Conv2d(in_channels + num_semantic_parts, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Modulation parameters (inspired by RegionAwareFusion)
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        B, C, H, W = features.shape

        # Method 1: Convolution-based part attention
        conv_logits = self.part_attention(features)  # [B, num_parts, H, W]

        # Method 2: Prototype-based similarity
        features_flat = features.view(B, C, -1)  # [B, C, H*W]
        prototypes = self.part_prototypes  # [num_parts, C]

        # Normalize features and prototypes for cosine similarity
        features_norm = F.normalize(features_flat, dim=1)  # [B, C, H*W]
        prototypes_norm = F.normalize(prototypes, dim=1)  # [num_parts, C]

        proto_logits = torch.einsum('bch,nc->bnh', features_norm, prototypes_norm)  # [B, num_parts, H*W]
        proto_logits = proto_logits.view(B, self.num_parts, H, W)

        # Combine two methods with learnable weights
        combine_weights = F.softmax(self.combine_weight, dim=0)
        conv_weight, proto_weight = combine_weights[0], combine_weights[1]

        # Weighted combination
        part_logits = conv_weight * conv_logits + proto_weight * proto_logits

        part_attention = F.softmax(part_logits, dim=1)  # [B, num_parts, H, W]

        # Create region modulation mask
        # We'll use the part with maximum attention at each location
        part_mask = torch.argmax(part_attention, dim=1)  # [B, H, W]
        part_mask_onehot = F.one_hot(part_mask, self.num_parts).permute(0, 3, 1, 2).float()  # [B, num_parts, H, W]

        # Select attention weights based on dominant part
        selected_attention = torch.sum(part_attention * part_mask_onehot, dim=1, keepdim=True)  # [B, 1, H, W]

        # Modulate features
        modulated_features = features * (1 + selected_attention * self.gamma) + self.beta

        # Final refinement
        enhanced_features = self.refinement(modulated_features)

        return enhanced_features, part_attention


class PVTPerceptualLoss(nn.Module):
    """Use PVT backbone features for perceptual comparison (no additional backbone)"""

    def __init__(self):
        super().__init__()

    def forward(self, pvt_features, object_mask, background_mask):
        """
        Compare PVT features of object vs background regions
        pvt_features: list of [fb1, fb2, fb3, fb4] from PVT backbone
        """
        perceptual_loss = 0

        for feat in pvt_features:
            # Resize masks to match feature size
            H, W = feat.shape[2:]
            obj_mask = F.interpolate(object_mask, size=(H, W), mode='bilinear', align_corners=True)
            bg_mask = F.interpolate(background_mask, size=(H, W), mode='bilinear', align_corners=True)

            # Extract object and background features
            obj_feat = feat * obj_mask
            bg_feat = feat * bg_mask

            # Compute feature statistics in masked regions
            obj_mean = obj_feat.sum(dim=[2, 3]) / (obj_mask.sum(dim=[2, 3]) + 1e-8)
            bg_mean = bg_feat.sum(dim=[2, 3]) / (bg_mask.sum(dim=[2, 3]) + 1e-8)

            # Perceptual similarity loss (camouflaged objects should have similar features to background)
            perceptual_loss += F.mse_loss(obj_mean, bg_mean)

        return perceptual_loss


class StreamlinedLoss(nn.Module):
    """Simplified loss with only essential components"""

    def __init__(self, alpha_perceptual=0.1,alpha_semantic=0.1):
        super().__init__()
        self.alpha_perceptual = alpha_perceptual
        self.alpha_semantic = alpha_semantic

        self.pvt_perceptual_loss = PVTPerceptualLoss()

    def create_background_mask(self, mask, margin=10):
        """Create background mask around object for perceptual comparison"""
        # Dilate mask to get nearby background region
        kernel = torch.ones(1, 1, margin * 2 + 1, margin * 2 + 1).to(mask.device)
        dilated_mask = F.conv2d(mask, kernel, padding=margin) > 0
        background_mask = dilated_mask.float() - mask
        return background_mask.clamp(0, 1)

    def forward(self, model_outputs, targets, images, masks, domain_labels=None):
        # Handle both old and new model output formats
        if isinstance(model_outputs, dict):
            pred = model_outputs['pred']
            target_pred = model_outputs['target_pred']
            fixation = model_outputs['fixation']
            pvt_features = model_outputs.get('pvt_features', None)
            part_attention = model_outputs.get('part_attention', None)
        else:
            pred = model_outputs['pred']
            target_pred = model_outputs['target_pred']
            target_pred = model_outputs['target_pred']
            fixation = model_outputs['fixation']
            pvt_features = model_outputs.get('pvt_features', None)
            part_attention = model_outputs.get('part_attention', None)

        total_loss = 0
        loss_dict = {}

        # 1. Main regression loss
        main_loss = F.mse_loss(target_pred, targets)
        total_loss += main_loss
        loss_dict['main_loss'] = main_loss.item()

        # 2. PVT-based perceptual loss (only if pvt_features available)
        if self.alpha_perceptual > 0 and pvt_features is not None:
            background_mask = self.create_background_mask(masks)
            perceptual_loss = self.pvt_perceptual_loss(pvt_features, masks, background_mask)
            total_loss += self.alpha_perceptual * perceptual_loss
            loss_dict['perceptual_loss'] = perceptual_loss.item()
        else:
            loss_dict['perceptual_loss'] = 0


        # 5. Semantic consistency loss (smooth part transitions)
        if self.alpha_semantic > 0 and part_attention is not None:
            # Encourage smooth transitions between semantic parts
            part_smoothness = 0
            if part_attention.size(2) > 1:
                part_smoothness += F.mse_loss(part_attention[:, :, :-1, :],
                                              part_attention[:, :, 1:, :])
            if part_attention.size(3) > 1:
                part_smoothness += F.mse_loss(part_attention[:, :, :, :-1],
                                              part_attention[:, :, :, 1:])
            total_loss += self.alpha_semantic * part_smoothness
            loss_dict['semantic_loss'] = part_smoothness.item()
        else:
            loss_dict['semantic_loss'] = 0

        return total_loss, loss_dict