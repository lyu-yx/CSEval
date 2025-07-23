#!/usr/bin/env python3
"""
Training script for Enhanced Degree Estimation Network with improved module integration
This version resolves interference between DomainAdapter and AdaptiveReceptiveFieldModule
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from datetime import datetime
from torch.autograd import Variable
from models.dataset_sem import get_loader
from models.CamoEval_enhanced_dofixre8 import EnhancedDegreeNet
from models.enhanced_components_dofixre8 import StreamlinedLoss
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def structure_loss(pred, mask):
    """Compute structure loss"""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def create_domain_labels(batch_size, device, cod_ratio=0.5):
    """Create domain labels for training (0=COD, 1=SOD)"""
    # Randomly assign domains for mixed training
    cod_count = int(batch_size * cod_ratio)
    sod_count = batch_size - cod_count
    
    labels = torch.cat([
        torch.zeros(cod_count, dtype=torch.long),
        torch.ones(sod_count, dtype=torch.long)
    ]).to(device)
    
    # Shuffle to randomize order
    perm = torch.randperm(batch_size)
    return labels[perm]

def validate_model(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, gts, masks in val_loader:
            images = Variable(images).to(device)
            gts = Variable(gts).to(device)
            masks = Variable(masks).to(device)
            
            # Forward pass
            outputs = model.forward_enhanced(images, masks, training_mode=True)
            
            # Create domain labels for validation
            domain_labels = create_domain_labels(images.size(0), device)
            
            # Compute loss
            total_loss, loss_dict = criterion(outputs, gts, images, masks, domain_labels)
            val_loss += total_loss.item()
            num_batches += 1
            
            if num_batches >= 50:  # Limit validation batches
                break
    
    model.train()
    return val_loss / num_batches

def train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                num_epochs=100, save_path='./snapshots_improved/', print_freq=10):
    """Train the improved model with harmonization"""
    
    logger.info("Starting training with improved module integration...")
    logger.info(f"Model: Enhanced Degree Network with Harmonization")
    logger.info(f"Device: {device}")
    logger.info(f"Number of epochs: {num_epochs}")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Track harmonization weights evolution
        harmony_weights = model.get_harmonization_weights() 
        
        for i, (images, gts, masks) in enumerate(train_loader):
            # Prepare data
            images = Variable(images).to(device)
            gts = Variable(gts).to(device)
            masks = Variable(masks).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model.forward_enhanced(images, masks, training_mode=True)
            
            # Create domain labels (mix of COD and SOD)
            domain_labels = create_domain_labels(images.size(0), device)
            
            # Compute comprehensive loss
            total_loss, loss_dict = criterion(outputs, gts, images, masks, domain_labels)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Track loss
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Print progress
            if (i + 1) % print_freq == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                          f'Loss: {total_loss.item():.4f}, '
                          f'Main: {loss_dict["main_loss"]:.4f}, '
                          f'Perceptual: {loss_dict["perceptual_loss"]:.4f}, '
                          f'Domain: {loss_dict["domain_loss"]:.4f}')
        
        # Average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f'Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_epoch_loss:.4f}')
        
        # Log harmonization weights
        current_harmony = model.get_harmonization_weights()
        logger.info(f'Harmonization weights: {current_harmony}')
        
        # Validation
        if val_loader is not None and (epoch + 1) % 5 == 0:
            val_loss = validate_model(model, val_loader, criterion, device)
            logger.info(f'Validation Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 
                          os.path.join(save_path, f'best_model_epoch_{epoch+1}.pth'))
                logger.info(f'Best model saved at epoch {epoch+1}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'harmonization_weights': current_harmony
            }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
            logger.info(f'Checkpoint saved at epoch {epoch+1}')
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pth'))
    logger.info('Training completed!')

def main():
    # Configuration
    config = {
        'batchsize': 4,
        'trainsize': 352,
        'clip': 0.5,
        'decay_rate': 0.1,
        'decay_epoch': 50,
        'epoch': 100,
        'lr': 1e-4,
        'train_root': './Dataset/TrainingSet/',
        'val_root': './Dataset/TestingSet/',
        'save_path': './snapshots_improved_harmonized/',
        'channel': 128
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create improved model
    model = EnhancedDegreeNet(channel=config['channel']).to(device)
    
    # Setup optimizer with different learning rates for different components
    backbone_params = []
    decoder_params = []
    harmonizer_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'harmonizer' in name:
            harmonizer_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': config['lr'] * 0.1},     # Lower LR for backbone
        {'params': decoder_params, 'lr': config['lr']},            # Standard LR for decoder
        {'params': harmonizer_params, 'lr': config['lr'] * 2.0}    # Higher LR for harmonizers
    ], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    # Setup loss
    criterion = StreamlinedLoss(
        alpha_perceptual=0.1,  # Perceptual loss weight
        alpha_domain=0.1       # Domain loss weight
    )
    
    # Setup data loaders
    train_loader = get_loader(
        image_root=config['train_root'] + 'Imgs/',
        gt_root=config['train_root'] + 'GT/',
        batchsize=config['batchsize'],
        trainsize=config['trainsize']
    )
    
    val_loader = get_loader(
        image_root=config['val_root'] + 'Imgs/',
        gt_root=config['val_root'] + 'GT/',
        batchsize=config['batchsize'],
        trainsize=config['trainsize']
    )
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['decay_epoch'], 
        gamma=config['decay_rate']
    )
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Start training
    try:
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=config['epoch'],
            save_path=config['save_path']
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save current state
        torch.save(model.state_dict(), 
                  os.path.join(config['save_path'], 'interrupted_model.pth'))
        logger.info("Model saved before exit")

if __name__ == '__main__':
    main() 