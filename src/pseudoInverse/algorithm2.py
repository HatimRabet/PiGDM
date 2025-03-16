import numpy as np
import torch
from tqdm import tqdm
from PIL import Image


import torch.nn.functional as F
import torchvision.transforms as T

from pseudoInverse.operators import SuperResolutionPseudoinverseOperator
from ddpm.model import DDPM
from ddpm.utils import pilimg_to_tensor, save_pilimg

def stats_tensor(x, name):
    print(f"{name} Min: {x.min().item()}, Max: {x.max().item()}")
    
    
def initialize_xt(x0_estimate, alphas_cumprod_t):
    noise = torch.randn_like(x0_estimate)
    
    return torch.sqrt(alphas_cumprod_t) * x0_estimate + torch.sqrt(1-alphas_cumprod_t) * noise

def pigdm_sampling(
    y,                          # Measurement/observation
    eps_model,                  # Epsilon prediction model
    num_steps = 100,           # number of steps
    guidance_factor = 0.005,    # Guidance coefficient
    measurement_operator=None,  # Function h(x) for noiseless case
    measurement_matrix=None,    # Matrix H for noisy case
    sigma_y=None,               # Measurement noise std for noisy case
    noiseless=True,             # Whether to use noiseless or noisy formulation
    seed=None                   # Random seed for reproducibility
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
    shape = eps_model.imgshape
    
    x0_estimate = pseudoinverse_operator(y, measurement_operator)
    
    N_timeseps = eps_model.num_diffusion_timesteps
    timesteps = np.arange(N_timeseps)

    betas = torch.tensor(eps_model.betas)
    alphas = torch.tensor(eps_model.alphas)
    alphas_cumprod = torch.tensor(eps_model.alphas_cumprod)
    
    # Prepare for sampling loop
    N = num_steps - 1
    
    alphas_cumprod_t = alphas_cumprod[N]
        
    x = initialize_xt(x0_estimate, alphas_cumprod_t)
    
    # Main sampling loop 
    for i in tqdm(range(N, -1, -1)):
        t = timesteps[i]
        # s = timesteps[i-1]
        
        x.requires_grad_(True)
    
        # Predict the noise using the model
        epsilon_theta, x_hat_t = eps_model(x, t) # same as model.get_eps_from_model
                
        # Predict the one-step denoised result
                
        z = torch.randn_like(x, device=device)
                
        mut = (
            x - betas[t] * epsilon_theta / (np.sqrt(1 - alphas_cumprod[t]))
        ) / np.sqrt(alphas[t])
        
        # Calculate the guidance term g
        if noiseless:
            if measurement_operator is None:
                raise ValueError("measurement_operator must be provided for noiseless case")
            # Calculate h(x_hat_t)
            h_x_hat = measurement_operator(x_hat_t)
            
            # Calculate h†(y) - h†(h(x_hat_t))
            h_pseudoinv_y = pseudoinverse_operator(y, measurement_operator)
            h_pseudoinv_h_x_hat = pseudoinverse_operator(h_x_hat, measurement_operator)
            
            # Calculate guidance term
            diff = h_pseudoinv_y - h_pseudoinv_h_x_hat
            mat_x = (diff.detach() * x_hat_t).sum()      
            g = torch.autograd.grad(mat_x, x, retain_graph=False)[0].detach()
        
        x = x.detach()
        # PiGDM update (last part of the algorithm)
        # print("sqrt alpha cumprod", torch.sqrt(alphas_cumprod[t]).item())
        x = (mut + np.sqrt(betas[t]) * z + torch.sqrt(alphas_cumprod[t]) * guidance_factor * g).detach()
        
    return x


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

# Example usage:
def example_usage(y, num_steps, guidance_factor):
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
        num_steps=num_steps,
        guidance_factor = guidance_factor,
        measurement_operator=measurement_operator,
        noiseless=True
    )
    
    return result



if __name__ == "__main__":
    # CONFIGURATION
    
    guidance_factor = 0.01
    num_steps = 1000
    
    image_path = "ddpm/diffusion-posterior-sampling/data/samples/00003.png"
    scale_factor = 4  # Example scaling (update to match self.scale_factor in your class)

    # Load image with PIL
    pil_img = Image.open(image_path).convert('RGB')
    
    

    tensor_img = pilimg_to_tensor(pil_img)
    measurement_operator = SuperResolutionPseudoinverseOperator(mode="bicubic")
    low_res_img = measurement_operator(tensor_img)
    
    low_res_img_show = pseudoinverse_operator(low_res_img, measurement_operator)

    high_res_img = example_usage(low_res_img, num_steps, guidance_factor)
    out = torch.cat((low_res_img_show, high_res_img, tensor_img), dim = 2)
    
    # print(high_res_img)
    print(f"Min: {high_res_img.min().item()}, Max: {high_res_img.max().item()}")
    
    save_pilimg(out, "image_3.jpg")

