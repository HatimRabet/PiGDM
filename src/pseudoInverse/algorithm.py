import numpy as np
import torch
from tqdm import tqdm
from PIL import Image


import torch.nn.functional as F
import torchvision.transforms as T

from pseudoInverse.operators import SuperResolutionPseudoinverseOperator
from ddpm.model import DDPM
from ddpm.utils import pilimg_to_tensor, save_pilimg
from tqdm import tqdm

def stats_tensor(x, name):
    print(f"{name} Min: {x.min().item()}, Max: {x.max().item()}")

def pigdm_sampling(
    y,                      # Measurement/observation
    eps_model,              # Epsilon prediction model
    # timesteps,              # Sequence of timesteps {vi}
    eta=0.0,                # Controls the stochasticity (0 for DDIM, 1 for DDPM)
    measurement_operator=None,  # Function h(x) for noiseless case
    measurement_matrix=None,    # Matrix H for noisy case
    sigma_y=None,           # Measurement noise std for noisy case
    noiseless=True,         # Whether to use noiseless or noisy formulation
    seed=None,
    superResolution=True               # Random seed for reproducibility
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
    available_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = next(eps_model.parameters()).device if hasattr(eps_model, 'parameters') else available_device
    
    y = y.to(device)
    # shape = y.shape # measurement_operator.forward(torch.zeros(1, device=device)).shape if measurement_operator is not None else 
    shape = eps_model.model.imgshape
    if superResolution:
        N = 500
        num_iterations = 100
        step = N // num_iterations

        x =  pseudoinverse_operator(y, measurement_operator)
        save_pilimg(x, "image_upsampled.jpg")
        x += (eps_model.model.betas[N] ** 0.5) * torch.randn_like(x)
    
    # N_timeseps = eps_model.model.num_diffusion_timesteps
    timesteps = np.arange(N, -1, -step) 
    # timesteps = np.arange(N+1)

    
    # Main sampling loop 
    for i in tqdm(range(num_iterations)):
        t = timesteps[i]
        s = timesteps[i+1]
        
        x.requires_grad_(True)
        
        # Get alpha_t as per VP-SDE
        sigma_t = eps_model.model.betas[t]**0.5
        sigma_s = eps_model.model.betas[s]**0.5
        
        # print("sigma_t", sigma_t.item())
        # print("sigma_s", sigma_s.item())
        alpha_t = torch.tensor(1 / (1 + sigma_t**2))
        alpha_s = torch.tensor(1 / (1 + sigma_s**2))
        
        # Predict the noise using the model
        epsilon_theta = eps_model(x, t) # same as model.get_eps_from_model
                
        # Predict the one-step denoised result
        x_hat_t = (x - torch.sqrt(1 - alpha_t) * epsilon_theta) / torch.sqrt(alpha_t)
        
        # Calculate DDIM coefficients
        c1 = eta * torch.sqrt((1 - alpha_t/alpha_s) * (1 - alpha_s) / (1 - alpha_t))
        c2 = torch.sqrt(1 - alpha_s - c1**2)
        
        # Calculate the guidance term g
        if noiseless:
            if measurement_operator is None:
                raise ValueError("measurement_operator must be provided for noiseless case")
            # Calculate h(x_hat_t)
            h_x_hat = measurement_operator(x_hat_t)
            
            # Calculate h†(y) - h†(h(x_hat_t))
            h_pseudoinv_y = pseudoinverse_operator(y, measurement_operator)
            h_pseudoinv_h_x_hat = pseudoinverse_operator(h_x_hat, measurement_operator)
   
            # Calculate gradient of x_hat_t with respect to x
            # x.requires_grad_(True)
            # x_hat_t_temp = (x - torch.sqrt(1 - alpha_t) * eps_model(x, t)) / torch.sqrt(alpha_t)
            # grad = torch.autograd.grad(x_hat_t_temp, x, 
            #                           grad_outputs=torch.ones_like(x_hat_t_temp))[0]
            # # grad = torch.autograd.grad(x_hat_t_temp, x)[0]
            # x = x.detach()
            # x.requires_grad_(False)
            
            # Calculate guidance term
            diff = h_pseudoinv_y - h_pseudoinv_h_x_hat
            
            # stats_tensor(diff, "diff")
            
            mat_x = (diff.detach() * x_hat_t).sum()
            
            # stats_tensor(mat_x, "mat_x")
            
            # print("diff", diff.transpose(2,3).shape)
            # print("grad", grad.shape)
            # g = torch.matmul(diff.transpose(2,3), grad)
            g = torch.autograd.grad(mat_x, x)[0] 
            # print(f"||g|| = {g.norm().item()}")
            # stats_tensor(g, "g")
            
            
        else:
            if measurement_matrix is None or sigma_y is None:
                raise ValueError("measurement_matrix and sigma_y must be provided for noisy case")
            
            # Calculate predicted measurement
            H = measurement_matrix
            r_t = torch.sqrt(sigma_t ** 2 / (sigma_t ** 2 + 1))  # from the paper see page 16
            
            # Calculate derivative of x_hat_t with respect to x_t
            x.requires_grad_(True)
            x_hat_t_temp = (x - torch.sqrt(1 - alpha_t) * eps_model(x, t)) / torch.sqrt(alpha_t)
            # dx_hat_dx = torch.autograd.grad(x_hat_t_temp, x, 
            #                                grad_outputs=torch.ones_like(x_hat_t_temp))[0]
            dx_hat_dx = torch.autograd.grad(x_hat_t_temp, x)[0]
            x.requires_grad_(False)
            
            # Calculate the inverse matrix term (HHᵀ + σ²y/r²t * I)⁻¹
            HHT = torch.matmul(H, H.transpose(0, 1))
            reg_term = (sigma_y / r_t)**2 * torch.eye(HHT.shape[0], device=device)
            inv_term = torch.inverse(HHT + reg_term)
            
            # Calculate residual y - H*x_hat_t
            residual = y - torch.matmul(H, x_hat_t)
            
            # Calculate guidance term
            g_term = torch.matmul(torch.matmul(torch.matmul(residual.transpose(0, 1), inv_term), H), dx_hat_dx)
            g = g_term.view_as(x)
        
        # Sample Gaussian noise for the stochastic part
        epsilon = torch.randn_like(x)
        
        x = x.detach()
        # PiGDM update (last part of the algorithm)
        
        # print("c1", c1.item())
        # print("c2", c2.item())
        # print("alpha_s", alpha_s.item())
        # print("alpha_t", alpha_t.item())
        
        # stats_tensor(x_hat_t, "x_hat_t")
        # stats_tensor(epsilon_theta, "epsilon_theta")
        # stats_tensor(torch.sqrt(alpha_s) * x_hat_t, "x_hat term")
        # stats_tensor(c1 * epsilon, "epsilon term")
        # stats_tensor(c2 * epsilon_theta, "eps_theta term")
        # stats_tensor(torch.sqrt(alpha_t) * g, "g_term")
        
        x = torch.sqrt(alpha_s) * x_hat_t + c1 * epsilon + c2 * epsilon_theta + torch.sqrt(alpha_t) * g
        
        # stats_tensor(x, "x")
        
    
    return eps_model.model.predict_xstart_from_eps(x, epsilon_theta, t)


## I should modify this don't forget
##########################################################################################
def get_noise_schedule(t):
    """
    Helper function to compute noise level sigma_t at timestep t.
    """
    return t  # Linear schedule

#########################################################################################

def pseudoinverse_operator(x, operator):
    """
    Compute the pseudoinverse operation h†(x).
    """
    return operator.pseudoinverse(x)


class DiffusionModel:
    """
    Example wrapper class for the diffusion model that predicts epsilon.
    This should be replaced with your actual diffusion model implementation.
    """
    def __init__(self, model):
        self.model = model
        
    def __call__(self, x, t):
        """
        Predict noise epsilon given noisy input x at timestep t.
        """
        return self.model(x, t)

# Example usage:
def example_usage(y):
    measurement_operator = SuperResolutionPseudoinverseOperator(mode="bicubic")
    # Create or load your epsilon prediction model
    ddpm = DDPM()
    eps_model = DiffusionModel(model=ddpm)  
    
    # Define timesteps (in descending order)
    # timesteps = torch.linspace(1.0, 0.0, 1000)
    
    # Run PiGDM sampling
    result = pigdm_sampling(
        y=y,
        eps_model=eps_model,
        # timesteps=timesteps,
        eta=1,
        measurement_operator=measurement_operator,
        noiseless=True
    )
    
    return result



if __name__ == "__main__":
    # CONFIGURATION
    image_path = "ddpm/diffusion-posterior-sampling/data/samples/00003.png"
    scale_factor = 4  # Example scaling (update to match self.scale_factor in your class)

    # Load image with PIL
    pil_img = Image.open(image_path).convert('RGB')
    
    

    tensor_img = pilimg_to_tensor(pil_img)
    measurement_operator = SuperResolutionPseudoinverseOperator(mode="bicubic")
    low_res_img = measurement_operator(tensor_img)

    high_res_img = example_usage(low_res_img)
    
    # print(high_res_img)
    print(f"Min: {high_res_img.min().item()}, Max: {high_res_img.max().item()}")
    
    save_pilimg(high_res_img, "image_2.jpg")

