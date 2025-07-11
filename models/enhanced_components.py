import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UnsupervisedSemanticDiscovery(nn.Module):
    """Discover semantic parts without labels using clustering in feature space"""
    def __init__(self, in_channels=128, num_semantic_parts=6):
        super().__init__()
        self.num_parts = num_semantic_parts
        
        # Part discovery through learnable prototypes
        self.part_prototypes = nn.Parameter(torch.randn(num_semantic_parts, in_channels))
        self.part_attention = nn.Conv2d(in_channels, num_semantic_parts, 1)
        self.part_refine = nn.Sequential(
            nn.Conv2d(in_channels + num_semantic_parts, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Semantic importance weighting (learned without supervision)
        self.semantic_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_semantic_parts, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, mask):
        B, C, H, W = features.shape
        
        # Compute part attention maps
        part_logits = self.part_attention(features)  # [B, num_parts, H, W]
        part_attention = F.softmax(part_logits, dim=1)
        
        # Apply mask to focus only on object regions
        mask_expanded = mask.expand(-1, self.num_parts, -1, -1)
        part_attention = part_attention * mask_expanded
        
        # Compute semantic importance weights
        semantic_weights = self.semantic_importance(features)  # [B, num_parts, 1, 1]
        
        # Create semantic-aware fixation weighting
        semantic_fixation_weights = part_attention * semantic_weights
        semantic_fixation_weights = semantic_fixation_weights.sum(dim=1, keepdim=True)
        
        # Refine features with semantic information
        enhanced_features = self.part_refine(torch.cat([features, part_attention], dim=1))
        
        return enhanced_features, semantic_fixation_weights, part_attention

class AdaptiveReceptiveFieldModule(nn.Module):
    """Replace fixed Conv_Block with adaptive receptive fields"""
    def __init__(self, channels):
        super().__init__()
        # Multi-scale dilated convolutions with learnable rates
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(channels * 3, channels, 3, padding=1, dilation=1),
            nn.Conv2d(channels * 3, channels, 3, padding=2, dilation=2), 
            nn.Conv2d(channels * 3, channels, 3, padding=4, dilation=4),
            nn.Conv2d(channels * 3, channels, 3, padding=8, dilation=8)
        ])
        
        # Adaptive weighting for different scales
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, 4, 1),
            nn.Softmax(dim=1)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 5, padding=2),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, input1, input2, input3):
        fused_input = torch.cat((input1, input2, input3), 1)
        
        # Apply multi-scale dilated convolutions
        scale_features = []
        for conv in self.dilated_convs:
            scale_features.append(conv(fused_input))
        
        # Adaptive scale weighting
        scale_weights = self.scale_attention(fused_input)
        weighted_features = sum(feat * weight for feat, weight in 
                              zip(scale_features, scale_weights.split(1, dim=1)))
        
        return self.fusion(weighted_features)

class DomainAdapter(nn.Module):
    """Lightweight domain adaptation module - shared weights with small adapters"""
    def __init__(self, channels, adapter_dim=16):
        super().__init__()
        # Small adapter networks for domain-specific adjustments
        self.cod_adapter = nn.Sequential(
            nn.Conv2d(channels, adapter_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_dim, channels, 1),
            nn.Sigmoid()  # Multiplicative adaptation
        )
        
        self.sod_adapter = nn.Sequential(
            nn.Conv2d(channels, adapter_dim, 1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(adapter_dim, channels, 1),
            nn.Sigmoid()  # Multiplicative adaptation
        )
        
        # Domain detection (for training only, not used in inference)
        self.domain_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),  # COD vs SOD
            nn.Flatten()
        )
        
    def forward(self, features, training_mode=True):
        # Domain detection (only for training loss, not used in forward)
        domain_logits = self.domain_detector(features) if training_mode else None
        
        # Apply both adapters and learn to weight them
        cod_adapted = features * self.cod_adapter(features)
        sod_adapted = features * self.sod_adapter(features)
        
        # Automatic domain weighting based on features
        domain_weights = F.softmax(self.domain_detector(features), dim=1) if training_mode else None
        
        if domain_weights is not None:
            # Weighted combination during training
            cod_weight = domain_weights[:, 0:1, None, None]
            sod_weight = domain_weights[:, 1:2, None, None]
            adapted_features = cod_weight * cod_adapted + sod_weight * sod_adapted
        else:
            # Simple average during inference to avoid error accumulation
            adapted_features = 0.5 * (cod_adapted + sod_adapted)
            
        return adapted_features, domain_logits

class ContrastLoss(nn.Module):
    """Measure local contrast around object boundaries"""
    def __init__(self):
        super().__init__()
        # Sobel operators for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def forward(self, image, mask):
        # Convert to grayscale
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        gray = gray.unsqueeze(1)
        
        # Compute gradients
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Focus on object boundary regions
        boundary_mask = F.conv2d(mask, torch.ones(1, 1, 3, 3).to(mask.device), padding=1)
        boundary_mask = (boundary_mask > 0) & (boundary_mask < 9)  # Boundary detection
        
        # Compute contrast in boundary regions
        boundary_contrast = gradient_magnitude * boundary_mask.float()
        return boundary_contrast.mean()

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
    def __init__(self, alpha_perceptual=0.1, alpha_contrast=0.05, 
                 alpha_domain=0.1, alpha_semantic=0.1):
        super().__init__()
        self.alpha_perceptual = alpha_perceptual
        self.alpha_contrast = alpha_contrast  
        self.alpha_domain = alpha_domain
        self.alpha_semantic = alpha_semantic
        
        self.pvt_perceptual_loss = PVTPerceptualLoss()
        self.contrast_loss = ContrastLoss()
        self.domain_loss = nn.CrossEntropyLoss()
        
    def create_background_mask(self, mask, margin=10):
        """Create background mask around object for perceptual comparison"""
        # Dilate mask to get nearby background region
        kernel = torch.ones(1, 1, margin*2+1, margin*2+1).to(mask.device)
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
            domain_logits = model_outputs.get('domain_logits', None)
        else:
            # Original format: (pred, target_pred, fixation)
            pred, target_pred, fixation = model_outputs
            pvt_features = None
            part_attention = None
            domain_logits = None
        
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
            
        # 3. Contrast loss for boundary analysis
        if self.alpha_contrast > 0:
            contrast_loss = self.contrast_loss(images, masks)
            total_loss += self.alpha_contrast * contrast_loss
            loss_dict['contrast_loss'] = contrast_loss.item()
        else:
            loss_dict['contrast_loss'] = 0
            
        # 4. Domain adversarial loss (only for adapter training, not used in forward)
        if self.alpha_domain > 0 and domain_labels is not None and domain_logits is not None:
            domain_loss = self.domain_loss(domain_logits, domain_labels)
            total_loss += self.alpha_domain * domain_loss
            loss_dict['domain_loss'] = domain_loss.item()
        else:
            loss_dict['domain_loss'] = 0
            
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