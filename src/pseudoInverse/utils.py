
import torch
import torch.nn.functional as F

# Diffusion Model Wrapper
class DiffusionModel:
    """
    Example wrapper class for the diffusion model that predicts epsilon.
    This should be replaced with your actual diffusion model implementation.
    """
    def __init__(self, model):
        self.model = model
        self.betas = model.betas
        self.alphas = model.alphas
        self.alphas_cumprod = model.alphas_cumprod
        self.imgshape = model.imgshape
        self.num_diffusion_timesteps = model.num_diffusion_timesteps
        
    def __call__(self, x, t):
        """
        Predict noise epsilon given noisy input x at timestep t.
        """
        eps = self.model(x, t)
        x0_pred = self.model.predict_xstart_from_eps(x, eps=eps, t=t)
        return eps, x0_pred
    
    
def compute_svd(kernel, img_dim, device):
    H_small = torch.zeros(img_dim, img_dim, device=device)
    kernel_size = kernel.shape[0]

    for i in range(img_dim):
        for j in range(i - kernel_size // 2, i + kernel_size // 2 + 1):
            if 0 <= j < img_dim:
                H_small[i, j] = kernel[j - i + kernel_size // 2]

    U_small, singulars_small, V_small = torch.svd(H_small, some=False)
    singulars_small[singulars_small < 3e-2] = 0

    return U_small, singulars_small, V_small


def apply_matrix(M, vec, img_dim):
    batch_size = vec.shape[0]
    vec = vec.view(batch_size * 3, img_dim, img_dim)
    vec = torch.matmul(M, vec)
    vec = torch.matmul(vec, M.T)
    return vec.view(batch_size, 3, img_dim, img_dim)


def wiener_deconvolution(y, kernel, K=0.01):
    """
    Perform Wiener deconvolution on input tensor y using blur kernel.
    
    Args:
        y: Blurry input image (batch, channels, height, width)
        kernel: Blur kernel (height, width)
        K: Regularization constant (noise-to-signal ratio)
        
    Returns:
        Deconvolved image tensor (same shape as y)
    """
    # Pad kernel to image size
    _, _, h, w = y.shape
    pad_h = h - kernel.shape[0]
    pad_w = w - kernel.shape[1]
    kernel_padded = F.pad(kernel, (0, pad_w, 0, pad_h))
    
    # Fourier transforms
    Y = torch.fft.fft2(y)
    H = torch.fft.fft2(kernel_padded)
    
    # Wiener filter
    H_conj = torch.conj(H)
    H_abs2 = H.abs() ** 2
    wiener_filter = H_conj / (H_abs2 + K)
    
    # Apply filter and inverse FFT
    X = wiener_filter * Y
    x_reconstructed = torch.fft.ifft2(X).real
    return x_reconstructed