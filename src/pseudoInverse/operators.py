import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class SuperResolutionPseudoinverseOperator:
    """
    Pseudoinverse operator for super-resolution tasks.
    Handles both average pooling and bicubic downsampling.
    """
    def __init__(self, scale_factor=4, mode='pool'):
        """
        Initialize the super-resolution pseudoinverse operator.
        
        Args:
            scale_factor: The scaling factor for super-resolution (default: 4x)
            mode: Downsampling mode - 'pool' for average pooling or 'bicubic' for bicubic interpolation
        """
        self.scale_factor = scale_factor
        self.mode = mode
        
        # For average pooling, create a fixed kernel
        if mode == 'pool':
            kernel_size = scale_factor
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)
            
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        Forward operation h(x): high-resolution to low-resolution.
        
        Args:
            x: High-resolution image tensor [B, C, H, W]
            
        Returns:
            Low-resolution image tensor [B, C, H/scale, W/scale]
        """
        if self.mode == 'pool':
            # Average pooling downsampling
            return self.pool(x)
        elif self.mode == 'bicubic':
            # Bicubic downsampling
            B, C, H, W = x.shape
            new_H, new_W = H // self.scale_factor, W // self.scale_factor
            return F.interpolate(x, size=(new_H, new_W), mode='bicubic', align_corners=False)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def pseudoinverse(self, y):
        """
        Pseudoinverse operation h†(y): low-resolution to high-resolution.
        
        Args:
            y: Low-resolution image tensor [B, C, H/scale, W/scale]
            
        Returns:
            High-resolution image tensor [B, C, H, W]
        """
        if self.mode == 'pool':
            # For average pooling, the pseudoinverse is nearest neighbor upsampling
            # followed by scaling to maintain energy
            B, C, H, W = y.shape
            new_H, new_W = H * self.scale_factor, W * self.scale_factor
            
            # Nearest upsampling maintains the average value across each block
            upsampled = F.interpolate(y, size=(new_H, new_W), mode='nearest')
            
            # No need to scale for average pooling pseudoinverse as the energy is preserved
            return upsampled
            
        elif self.mode == 'bicubic':
            # For bicubic, the pseudoinverse is bicubic upsampling
            B, C, H, W = y.shape
            new_H, new_W = H * self.scale_factor, W * self.scale_factor
            
            # Use bicubic interpolation as the pseudoinverse
            return F.interpolate(y, size=(new_H, new_W), mode='bicubic', align_corners=False)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")