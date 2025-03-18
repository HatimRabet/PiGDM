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



class RotationOperator:
    """
    Rotation operator for image transformations.
    Allows both forward (rotation) and pseudoinverse (inverse rotation) operations.
    """
    def __init__(self, theta):
        """
        Initialize the rotation operator.
        
        Args:
            theta: Rotation angle in degrees.
        """
        self.theta = theta
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        Forward operation R(x): rotate image by theta degrees.
        
        Args:
            x: Image tensor [B, C, H, W]
        
        Returns:
            Rotated image tensor [B, C, H, W]
        """
        theta_rad = np.radians(self.theta)
        
        # Create affine transformation matrix
        affine_matrix = torch.tensor([
            [np.cos(theta_rad), -np.sin(theta_rad), 0],
            [np.sin(theta_rad), np.cos(theta_rad), 0]
        ], dtype=torch.float32, device=x.device)
        
        # Expand for batch processing
        batch_size = x.shape[0]
        affine_matrix = affine_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Generate grid and apply transformation
        grid = F.affine_grid(affine_matrix, x.size(), align_corners=False)
        rotated = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        return rotated
    
    def pseudoinverse(self, y):
        """
        Pseudoinverse operation R†(y): rotate image by -theta degrees.
        
        Args:
            y: Rotated image tensor [B, C, H, W]
        
        Returns:
            Original image tensor [B, C, H, W]
        """
        inverse_theta = -self.theta
        theta_rad = np.radians(inverse_theta)
        
        # Create inverse affine transformation matrix
        affine_matrix = torch.tensor([
            [np.cos(theta_rad), -np.sin(theta_rad), 0],
            [np.sin(theta_rad), np.cos(theta_rad), 0]
        ], dtype=torch.float32, device=y.device)
        
        # Expand for batch processing
        batch_size = y.shape[0]
        affine_matrix = affine_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Generate grid and apply inverse transformation
        grid = F.affine_grid(affine_matrix, y.size(), align_corners=False)
        restored = F.grid_sample(y, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        return restored
    


class IdentityOperator:
    """
    Rotation operator for image transformations.
    Allows both forward (rotation) and pseudoinverse (inverse rotation) operations.
    """
    def __init__(self):
        self.name = "identity"
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        return x
    
    def pseudoinverse(self, y):
        return y