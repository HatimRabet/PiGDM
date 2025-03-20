import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from pseudoInverse.operators import SuperResolutionPseudoinverseOperator, RotationOperator, IdentityOperator, GrayscaleOperator, BlurPseudoinverseOperator, InpaintingPseudoinverseOperator, GaussianBlurOperator
from pseudoInverse.utils import DiffusionModel

from ddpm.model import DDPM
from ddpm.utils import pilimg_to_tensor, save_pilimg


class PiGDM:
    def __init__(self, model, measurement_operator, measurement_matrix=None, eta=1, guidance_factor=0.01, device='cuda'):
        self.model = model
        self.measurement_operator = measurement_operator
        self.H = measurement_matrix
        self.eta = eta
        self.guidance_factor = guidance_factor
        self.device = device

    def initialize_xt(self, x0_estimate, alphas_cumprod_t):
        noise = torch.randn_like(x0_estimate)
        return torch.sqrt(alphas_cumprod_t) * x0_estimate + torch.sqrt(1-alphas_cumprod_t) * noise
        

    def sample(
        self,
        y,                          
        num_steps=500,           
        sigma_y=None,               
        noiseless=True,   
        optimized=True,          
        seed=None                   
    ):
        """
        Implementation of the PiGDM (Pseudoinverse Guided Diffusion Model) sampling algorithm.
        
        Args:
            y: The observation/measurement
            eps_model: A function that predicts noise epsilon given x and timestep t
            timesteps: Sequence of timesteps for sampling (should be in descending order)
            eta: Controls the stochasticity (0 for deterministic DDIM, 1 for stochastic like DDPM)
            measurement_operator: For noiseless case, function h(x) and its pseudoinverse h†
            measurement_matrix: For noisy case, matrix H
            sigma_y: Measurement noise standard deviation (for noisy case)
            noiseless: Whether to use noiseless or noisy formulation
            seed: Random seed for reproducibility
        
        Returns:
            The sampled image/data
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize x from standard Gaussian
        y = y.to(self.device)
        shape = self.model.imgshape
        
        x0_estimate = self.measurement_operator.pseudoinverse(y)
        
        T = self.model.num_diffusion_timesteps
        timesteps = np.arange(T)

        betas = torch.tensor(self.model.betas)
        alphas = torch.tensor(self.model.alphas)
        alphas_cumprod = torch.tensor(self.model.alphas_cumprod)
        
        # Prepare for sampling loop
        alphas_cumprod_t = alphas_cumprod[num_steps - 1]
            
        x = self.initialize_xt(x0_estimate, alphas_cumprod_t)
        
        # Main sampling loop 
        for i in tqdm(range(num_steps-1, -1, -1)):
            t = timesteps[i]
            # s = timesteps[i-1]
            
            x.requires_grad_(True)
        
            # Predict the noise using the model
            epsilon_theta, x_hat_t = self.model(x, t) 
                    
            # Predict the one-step denoised result (DDPM version)
            z = torch.randn_like(x, device=self.device)
            mut = (x - betas[t] * epsilon_theta / np.sqrt(1 - alphas_cumprod[t])) / np.sqrt(alphas[t])
            

            
            # Calculate the guidance term g
            if noiseless:
                if self.measurement_operator is None:
                    raise ValueError("measurement_operator must be provided for noiseless case")
                # # Calculate h(x_hat_t)
                # h_x_hat = self.measurement_operator(x_hat_t)
                
                # # Calculate h†(y) - h†(h(x_hat_t))
                # h_pseudoinv_y = self.measurement_operator.pseudoinverse(y)
                # h_pseudoinv_h_x_hat = self.measurement_operator.pseudoinverse(h_x_hat)
                
                # # Calculate guidance term
                # diff = h_pseudoinv_y - h_pseudoinv_h_x_hat
                # mat_x = (diff.detach() * x_hat_t).sum()      
                # g = torch.autograd.grad(mat_x, x, retain_graph=False)[0].detach()
                g = noiseless_guidance(
                    x = x,
                    x_hat_t=x_hat_t,
                    y = y,
                    measurement_operator=self.measurement_operator
                )
                
            else:
                # if self.H is None:
                #     raise ValueError("measurement_matrix must be provided for noisy case")
                # sigma_t = betas[t] ** 0.5
                # r_t = ((sigma_t ** 2) / (1 + sigma_t ** 2)) ** 0.5

                # HH_T = self.H @ self.H.T
                # noise_term = (sigma_y**2 / r_t**2) * torch.eye(HH_T.shape[0], device=HH_T.device, dtype=HH_T.dtype)
                # inv_matrix = torch.linalg.solve(HH_T + noise_term, torch.eye(HH_T.shape[0], device=HH_T.device, dtype=HH_T.dtype))
                
                # mat_x = ((y - self.H @ x_hat_t).detach() * ((inv_matrix @ self.H) @ x_hat_t)).sum()
                # g = torch.autograd.grad(mat_x, x, retain_graph=False)[0].detach()
                r_t_2 = (1 - alphas_cumprod[t]) / alphas_cumprod[t]
                
                if optimized :
                    g = operator_based_noisy_guidance(
                        x=x,
                        x_hat_t=x_hat_t,
                        y=y,
                        measurement_operator=self.measurement_operator,
                        sigma_y=sigma_y,
                        r_t_2=r_t_2,
                    )
                else :
                    #exact implementation from the paper
                    if self.H is None:
                        raise ValueError("measurement_matrix must be provided for noisy case")
                    
                    g = matrix_based_noisy_guidance(
                        x=x,
                        x_hat_t=x_hat_t,
                        y=y,
                        H = self.H,
                        sigma_y=sigma_y,
                        r_t_2=r_t_2,
                    )

            
            x = x.detach()
            # PiGDM update (last part of the algorithm)
            x = (mut + np.sqrt(betas[t]) * z + torch.sqrt(alphas_cumprod[t]) * self.guidance_factor * g).detach()
            
        return x
    
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


if __name__ == "__main__":
    # CONFIGURATION
    guidance_factor = 0.001
    num_steps = 1000
    
    image_name = "00003"
    
    image_path = f"ddpm/diffusion-posterior-sampling/data/samples/{image_name}.png"

    noiseless = False
    sigma_y = 1
    # Load image with PIL
    pil_img = Image.open(image_path).convert('RGB')

    tensor_img = pilimg_to_tensor(pil_img)
    # measurement_operator = SuperResolutionPseudoinverseOperator(mode="bicubic")
    # measurement_operator = BlurPseudoinverseOperator(kernel_size=31, sigma=10)
    measurement_operator = GaussianBlurOperator(kernel_size=11, sigma=10, wiener_k=0.05)
    # measurement_operator = IdentityOperator()
    # measurement_operator = RotationOperator(45)
    # measurement_operator = GrayscaleOperator()

    low_res_img = measurement_operator(tensor_img)
    if not noiseless :
        low_res_img += sigma_y * torch.randn_like(low_res_img)
    
    low_res_img_show = measurement_operator.pseudoinverse(low_res_img)
 
    # Intialize Model
    ddpm = DDPM()
    model = DiffusionModel(model=ddpm) 

    # Initialize sampler
    pidgm_sampler = PiGDM(model, measurement_operator, guidance_factor=guidance_factor)

    # Result
    high_res_img = pidgm_sampler.sample(low_res_img, num_steps, sigma_y, noiseless)
    out = torch.cat((low_res_img, high_res_img, tensor_img), dim = 2)    
    print(f"Min: {high_res_img.min().item()}, Max: {high_res_img.max().item()}")
    # out = torch.cat((low_res_img, low_res_img_show, tensor_img), dim = 2)


    

    # Save Image
    save_pilimg(out, f"DDPM_{image_name}.png")