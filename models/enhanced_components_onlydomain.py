import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DomainAdapter(nn.Module):
    """Lightweight domain adaptation module - shared weights with small adapters"""

    def __init__(self, channels, adapter_dim=16):
        super().__init__()
        # Small adapter networks for domain-specific adjustments
        self.cod_adapter4 = nn.Sequential(
            nn.Conv2d(channels * 4, adapter_dim*4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_dim*4, channels * 4, 1),
            nn.Sigmoid()  # Multiplicative adaptation
        )

        self.sod_adapter4 = nn.Sequential(
            nn.Conv2d(channels * 4, adapter_dim*4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_dim*4, channels * 4, 1),
            nn.Sigmoid()  # Multiplicative adaptation
        )

        self.cod_adapter3 = nn.Sequential(
            nn.Conv2d(320, adapter_dim*2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_dim*2, 320, 1),
            nn.Sigmoid()  # Multiplicative adaptation
        )

        self.sod_adapter3 = nn.Sequential(
            nn.Conv2d(320, adapter_dim*2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_dim*2, 320, 1),
            nn.Sigmoid()  # Multiplicative adaptation
        )

        self.cod_adapter2 = nn.Sequential(
            nn.Conv2d(channels, adapter_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_dim, channels, 1),
            nn.Sigmoid()  # Multiplicative adaptation
        )

        self.sod_adapter2 = nn.Sequential(
            nn.Conv2d(channels, adapter_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_dim, channels, 1),
            nn.Sigmoid()  # Multiplicative adaptation
        )

        # Domain detection (for training only, not used in inference)
        self.domain_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),  # COD vs SOD
            nn.Flatten()
        )

    def forward(self, features4, features3, features2, training_mode=True):
        # Domain detection (only for training loss, not used in forward)
        domain_logits = self.domain_detector(features4) if training_mode else None

        # Apply both adapters and learn to weight them
        cod_adapted4 = features4 * self.cod_adapter4(features4)
        sod_adapted4 = features4 * self.sod_adapter4(features4)
        cod_adapted3 = features3 * self.cod_adapter3(features3)
        sod_adapted3 = features3 * self.sod_adapter3(features3)
        cod_adapted2 = features2 * self.cod_adapter2(features2)
        sod_adapted2 = features2 * self.sod_adapter2(features2)

        # Automatic domain weighting based on features
        domain_weights = F.softmax(domain_logits, dim=1) if training_mode else None

        if domain_weights is not None:
            # Weighted combination during training
            cod_weight = domain_weights[:, 0:1, None, None]
            sod_weight = domain_weights[:, 1:2, None, None]
            adapted_features4 = cod_weight * cod_adapted4 + sod_weight * sod_adapted4
            adapted_features3 = cod_weight * cod_adapted3 + sod_weight * sod_adapted3
            adapted_features2 = cod_weight * cod_adapted2 + sod_weight * sod_adapted2
        else:
            # Simple average during inference to avoid error accumulation
            adapted_features4 = 0.5 * (cod_adapted4 + sod_adapted4)
            adapted_features3 = 0.5 * (cod_adapted3 + sod_adapted3)
            adapted_features2 = 0.5 * (cod_adapted2 + sod_adapted2)


        return adapted_features4, adapted_features3, adapted_features2, domain_logits

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

    def __init__(self, alpha_perceptual=0.1,
                 alpha_domain=0.1):
        super().__init__()
        self.alpha_perceptual = alpha_perceptual
        self.alpha_domain = alpha_domain

        self.pvt_perceptual_loss = PVTPerceptualLoss()
        self.domain_loss = nn.CrossEntropyLoss()

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
            domain_logits = model_outputs.get('domain_logits', None)
        else:
            pred = model_outputs['pred']
            target_pred = model_outputs['target_pred']
            fixation = model_outputs['fixation']
            pvt_features = model_outputs.get('pvt_features', None)
            domain_logits = model_outputs.get('domain_logits', None)

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

        # 4. Domain adversarial loss (only for adapter training, not used in forward)
        if self.alpha_domain > 0 and domain_labels is not None and domain_logits is not None:
            domain_loss = self.domain_loss(domain_logits, domain_labels)
            total_loss += self.alpha_domain * domain_loss
            loss_dict['domain_loss'] = domain_loss.item()
        else:
            loss_dict['domain_loss'] = 0

        return total_loss, loss_dict