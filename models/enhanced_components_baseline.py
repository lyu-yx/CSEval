import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
                 alpha_domain=0.1, alpha_semantic=0.1):
        super().__init__()
        self.alpha_perceptual = alpha_perceptual
        self.alpha_domain = alpha_domain
        self.alpha_semantic = alpha_semantic

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
        else:
            pred = model_outputs['pred']
            target_pred = model_outputs['target_pred']
            fixation = model_outputs['fixation']
            pvt_features = model_outputs.get('pvt_features', None)

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


        return total_loss, loss_dict