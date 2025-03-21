
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
    
    

def noiseless_guidance(
    x,                          # Current x sample (requires_grad = True)
    x_hat_t,                    # Denoised prediction x_hat_t
    y,                          # Measurement
    measurement_operator        # measurement operator
):
    # Calculate h(x_hat_t)
    h_x_hat = measurement_operator(x_hat_t)
    
    # Calculate h†(y) - h†(h(x_hat_t))
    h_pseudoinv_y = measurement_operator.pseudoinverse(y)
    h_pseudoinv_h_x_hat = measurement_operator.pseudoinverse(h_x_hat)
    
    # Calculate guidance term
    diff = h_pseudoinv_y - h_pseudoinv_h_x_hat
    mat_x = (diff.detach() * x_hat_t).sum()      
    g = torch.autograd.grad(mat_x, x, retain_graph=False)[0].detach()
    return g

    
def matrix_based_noisy_guidance(
    x,                          # Current x sample (requires_grad = True)
    x_hat_t,                    # Denoised prediction x_hat_t
    y,                          # Measurement
    H,                          # H operator matrix
    sigma_y,                    # Measurement noise std
    r_t_2,                      # r_t squared
):
    HH_T = H @ H.T
    noise_term = (sigma_y**2 / r_t_2) * torch.eye(HH_T.shape[0], device=HH_T.device, dtype=HH_T.dtype)
    inv_matrix = torch.linalg.solve(HH_T + noise_term, torch.eye(HH_T.shape[0], device=HH_T.device, dtype=HH_T.dtype))
    
    mat_x = ((y - H @ x_hat_t).detach() * ((inv_matrix @ H) @ x_hat_t)).sum()
    g = torch.autograd.grad(mat_x, x, retain_graph=False)[0].detach()
    return g

    
def operator_based_noisy_guidance(
    x,                          # Current x sample (requires_grad = True)
    x_hat_t,                    # Denoised prediction x_hat_t
    y,                          # Measurement
    measurement_operator,       # h(x): forward operator
    sigma_y,                    # Measurement noise std
    r_t_2,                      # r_t squared
    cg_tol=1e-5,                # Convergence tolerance for conjugate gradients
    cg_max_iter=25              # Maximum iterations for conjugate gradients
):


    # Compute residual: y - h(x_hat_t)
    residual = y - measurement_operator(x_hat_t)

    # Define linear operator A = HH_T + sigma_y^2 / r_t^2 * I
    def A_fn(v):
        H_T_v = measurement_operator.pseudoinverse(v)  # Back to RGB
        H_H_T_v = measurement_operator(H_T_v)          # Back to grayscale (forward op)
        return H_H_T_v + (sigma_y**2 / r_t_2) * v

    # Solve A z = residual using conjugate gradients
    z, _ = conjugate_gradients(A_fn, residual, max_iter=cg_max_iter, tol=cg_tol)
    # print("z shape", z.shape)
    # print("h(x_hat_t) shape",measurement_operator(x_hat_t).shape)
    # Compute gradient (VJP): Jᵗ(z)
    # Autograd way: sum(h(x_hat_t) * z) then differentiate w.r.t. x_hat_t
    mat_x = (measurement_operator(x_hat_t) * z.detach()).sum()
    # print("mat_x shape", z.shape)
    g = torch.autograd.grad(mat_x, x, retain_graph=False)[0].detach()

    return g


def conjugate_gradients(A_fn, b, max_iter=25, tol=1e-5):
    """
    Solves A x = b using the conjugate gradients method.
    A_fn: function implementing matrix-vector product A(v)
    b: right-hand side vector
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.sum(r * r)

    for i in range(max_iter):
        Ap = A_fn(p)
        alpha = rs_old / (torch.sum(p * Ap) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, i    
    
    
    
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
