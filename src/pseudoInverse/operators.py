import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from pseudoInverse.utils import compute_svd, apply_matrix

from pseudoInverse.utils import apply_matrix, compute_svd

class SuperResolutionOperator:
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
        return x.clone()
    
    def pseudoinverse(self, y):
        return y.clone()
    
    
class GrayscaleOperator:
    """
    Grayscale operator for image transformations.
    Converts RGB images to grayscale in the forward pass and reconstructs 
    RGB images in the pseudoinverse by duplicating the grayscale values across channels.
    
    Assumes input tensor x has shape (batch, channels, height, width)
    and expects 3 channels in forward.
    """
    def __init__(self):
        self.name = "grayscale"

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Convert RGB image (3 channels) to grayscale.
        x: Tensor with shape (batch, 3, height, width)
        """
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {x.shape[1]} channels.")
        
        # Apply standard luminance conversion weights for grayscale
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device).view(1, 3, 1, 1)
        grayscale = (x * weights).sum(dim=1, keepdim=True)  # shape: (batch, 1, height, width)
        return grayscale

    def pseudoinverse(self, y):
        """
        Reconstruct RGB image from grayscale by duplicating the grayscale channel.
        y: Tensor with shape (batch, 1, height, width)
        """
        if y.shape[1] != 1:
            raise ValueError(f"Expected 1 channel (grayscale), got {y.shape[1]} channels.")
        
        rgb_reconstructed = y.repeat(1, 3, 1, 1)  # shape: (batch, 3, height, width)
        return rgb_reconstructed
    
    
class BlurOperator:
    """
    Pseudoinverse operator for blur tasks.
    Uses SVD to compute the forward blur operation and its pseudoinverse.
    """
    def __init__(self, kernel_size, sigma, img_dim=256, device="cuda"):
        """
        Initialize the blur pseudoinverse operator.
        
        Args:
            kernel: The 1D kernel to use for blurring
            img_dim: The dimension of the image (default: 256)
            device: The device to use for computation (default: "cuda")
        """
        self.kernel = self.create_gaussian_kernel(kernel_size, sigma, device)
        self.img_dim = img_dim
        self.device = device
        
        # Precompute SVD for efficiency
        self.U_small, self.singulars_small, self.V_small = compute_svd(self.kernel, img_dim, device)
        
        # Precompute singulars inverse for pseudoinverse operation
        nonzero_idx = torch.nonzero(self.singulars_small, as_tuple=True)[0]

        self.singulars_inv = torch.zeros_like(self.singulars_small)
        self.singulars_inv[nonzero_idx] = 1 / self.singulars_small[nonzero_idx]
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        Forward operation h(x): apply blur to the input image.
        
        Args:
            x: Input image tensor
            
        Returns:
            Blurred image tensor
        """
        temp = apply_matrix(self.V_small.T, x, self.img_dim)
        temp = self.singulars_small.view(1, 1, self.img_dim, 1) * temp
        return apply_matrix(self.U_small, temp, self.img_dim)
    
    def pseudoinverse(self, y):
        """
        Pseudoinverse operation h†(y): attempt to recover the original image from a blurred one.
        
        Args:
            y: Blurred image tensor
            
        Returns:
            Deblurred image tensor approximation
        """
        temp = apply_matrix(self.U_small.T, y, self.img_dim)
        temp = self.singulars_inv.view(1, 1, self.img_dim, 1) * temp
        return apply_matrix(self.V_small, temp, self.img_dim)
    

    def create_gaussian_kernel(self, size, sigma, device="cuda"):
        """
        Create a 1D Gaussian kernel.
        
        Args:
            size: Size of the kernel (odd number recommended)
            sigma: Standard deviation of the Gaussian
            device: Device to create the kernel on
        
        Returns:
            1D Gaussian kernel tensor
        """
        coords = torch.arange(size, device=device) - (size - 1) / 2
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()  # Normalize to sum to 1
        return kernel
    
    
class GaussianBlurOperator:
    """
    Gaussian blur operator for image transformations.
    Applies Gaussian blur in the forward pass.
    Uses Wiener deconvolution as the pseudoinverse to approximate deblurring.
    
    Assumes input tensor x has shape (batch, channels, height, width).
    """
    
    def __init__(self, kernel_size=5, sigma=1.0, wiener_k=0.01):
        self.name = "gaussian_blur"
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.wiener_k = wiener_k  # Wiener regularization parameter
        self.kernel = None  # Lazy initialization on device
        
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Applies Gaussian blur to input tensor x.
        """
        if self.kernel is None or self.kernel.device != x.device:
            self.kernel = self.create_gaussian_kernel(x.device, x.dtype)

        # Apply convolution per channel
        channels = x.shape[1]
        blurred = F.conv2d(x, self.kernel.expand(channels, 1, -1, -1), 
                           padding=self.kernel_size // 2, groups=channels)
        return blurred

    def pseudoinverse(self, y):
        """
        Applies Wiener deconvolution to approximate the inverse of Gaussian blur.
        
        Args:
            y: blurred input tensor (batch, channels, height, width)
        Returns:
            Deconvolved (sharpened) tensor
        """
        return self.wiener_deconvolution(y, self.kernel, self.wiener_k)

    def create_gaussian_kernel(self, device, dtype):
        """
        Creates a 2D Gaussian kernel tensor of shape (1, 1, kernel_size, kernel_size).
        """
        k = self.kernel_size
        sigma = self.sigma

        # Create 1D kernel
        x = torch.arange(-k // 2 + 1., k // 2 + 1., device=device, dtype=dtype)
        x = x.view(1, -1)
        gaussian_1d = torch.exp(-0.5 * (x / sigma)**2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()

        # Create 2D kernel via outer product
        gaussian_2d = gaussian_1d.T @ gaussian_1d
        gaussian_2d = gaussian_2d / gaussian_2d.sum()

        kernel = gaussian_2d.view(1, 1, k, k)
        return kernel

    def wiener_deconvolution(self, y, kernel, K=0.01):
        """
        Performs Wiener deconvolution on input tensor y using blur kernel.
        
        Args:
            y: Blurry input image (batch, channels, height, width)
            kernel: Blur kernel (1, 1, kH, kW)
            K: Regularization constant (noise-to-signal ratio)
        Returns:
            Deconvolved image tensor (same shape as y)
        """
        batch_size, channels, height, width = y.shape
        
        # Pad kernel to image size
        pad_h = height - kernel.shape[2]
        pad_w = width - kernel.shape[3]
        pad = (0, pad_w, 0, pad_h)
        kernel_padded = F.pad(kernel, pad)

        # FFT of kernel and image
        H = torch.fft.fft2(kernel_padded, dim=(-2, -1))
        H_conj = torch.conj(H)
        H_abs2 = (H.real ** 2 + H.imag ** 2)
        
        Y = torch.fft.fft2(y, dim=(-2, -1))

        # Wiener filter calculation
        wiener_filter = H_conj / (H_abs2 + K)

        # Apply filter and inverse FFT
        X = wiener_filter * Y
        x_reconstructed = torch.fft.ifft2(X, dim=(-2, -1)).real

        return x_reconstructed

    
class InpaintingOperator:
    """
    Pseudoinverse operator for inpainting tasks.
    Applies a binary mask to an image, zeroing out masked regions.
    """
    def __init__(self, img_shape, ratio_mask, block_size, random):
        """
        Initialize the inpainting pseudoinverse operator.
        
        Args:
            mask: Binary mask tensor where 1 indicates pixels to keep and 0 indicates pixels to remove
                 Shape should match the input images [1, 1, H, W] or [1, C, H, W]
        """
        self.height, self.width = img_shape[0], img_shape[1]
        if random:
            self.mask = self.create_random_mask(self.height, self.width, ratio_mask)
        else:
            self.mask = self.create_block_mask(self.height, self.width, block_size)
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        Forward operation h(x): apply mask to the input image.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Masked image tensor with same shape as input
        """
        return x * self.mask
    
    def pseudoinverse(self, y):
        """
        Pseudoinverse operation h†(y): for inpainting, this is simply the identity function
        since we cannot recover the masked out information.
        
        Args:
            y: Masked image tensor [B, C, H, W]
            
        Returns:
            Same tensor as input (no recovery of masked pixels is possible)
        """
        return y

    def create_random_mask(self, height=256, width=256, mask_ratio=0.5, device='cuda'):
        """
        Create a random binary mask.
        
        Args:
            height: Height of the mask
            width: Width of the mask
            mask_ratio: Ratio of pixels to keep (1-mask_ratio will be masked out)
            device: Device to create mask on
        
        Returns:
            Binary mask tensor [1, 1, H, W]
        """
        mask = torch.rand(1, 1, height, width, device=device) > (1 - mask_ratio)
        return mask.float()


    def create_block_mask(self, height=256, width=256, block_size=100, device='cuda'):
        """
        Create a mask with a block hole in the center.
        
        Args:
            height: Height of the mask
            width: Width of the mask
            block_size: Size of the square hole
            device: Device to create mask on
        
        Returns:
            Binary mask tensor [1, 1, H, W]
        """
        mask = torch.ones(1, 1, height, width, device=device)
        center_h, center_w = height // 2, width // 2
        half_block = block_size // 2
        
        mask[:, :, 
            center_h-half_block:center_h+half_block, 
            center_w-half_block:center_w+half_block] = 0
        
        return mask