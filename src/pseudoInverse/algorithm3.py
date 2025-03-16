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
    eta=0.0,                # Controls stochasticity (0=DDIM, 1=DDPM)
    measurement_operator=None,  # h(x) for noiseless case
    measurement_matrix=None,    # H for noisy case
    sigma_y=None,           # Measurement noise std
    noiseless=True,         # Noiseless or noisy formulation
    seed=None,
    superResolution=True    # Super-resolution task
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    device = next(eps_model.parameters()).device if hasattr(eps_model, 'parameters') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = y.to(device)
    shape = eps_model.model.imgshape  # Expected output shape
    
    if superResolution:
        N = 200  # Starting timestep (e.g., 500/1000)
        num_iterations = 100
        step = N // num_iterations

        # Initialize x with pseudoinverse and add noise
        x = pseudoinverse_operator(y, measurement_operator)
        x += np.sqrt(1 - eps_model.model.alphas_cumprod[N]) * torch.randn_like(x)
    
    # Generate timestep sequence
    timesteps = np.arange(N, -1, -step)  # Adjust to match model's total steps

    # Main sampling loop
    for i in tqdm(range(num_iterations)):
        t = timesteps[i]
        if i == num_iterations - 1:
            s = 0  # Last step
        else:
            s = timesteps[i + 1]

        x.requires_grad_(True)

        # Retrieve alpha_bar_t and alpha_bar_s from DDPM
        alpha_bar_t = eps_model.model.alphas_cumprod[t]
        alpha_bar_s = eps_model.model.alphas_cumprod[s]

        # Compute alpha_t and alpha_s as per VP-SDE (1 / (1 + sigma_t^2))
        sigma_t_sq = 1 - alpha_bar_t
        sigma_s_sq = 1 - alpha_bar_s
        alpha_t = 1 / (1 + sigma_t_sq)
        alpha_s = 1 / (1 + sigma_s_sq)

        # Predict noise (epsilon_theta)
        epsilon_theta = eps_model(x, t)

        # Compute x_hat_t (denoised estimate)
        x_hat_t = (x - np.sqrt(1 - alpha_t) * epsilon_theta) / np.sqrt(alpha_t)

        # Compute DDIM coefficients c1 and c2
        c1 = eta * np.sqrt((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t))
        c2 = np.sqrt(1 - alpha_s - c1**2)

        # Compute guidance term g
        if noiseless:
            h_x_hat = measurement_operator(x_hat_t)
            h_pinv_y = pseudoinverse_operator(y, measurement_operator)
            h_pinv_h_x_hat = pseudoinverse_operator(h_x_hat, measurement_operator)
            diff = h_pinv_y - h_pinv_h_x_hat
            
            # Compute gradient via autograd
            mat_x = (diff.detach() * x_hat_t).sum()
            g = torch.autograd.grad(mat_x, x, retain_graph=True)[0]
        else:
            # Noisy case (not shown here)
            pass

        # Stochastic noise term
        epsilon = torch.randn_like(x)

        # Update x
        x = (
            np.sqrt(alpha_s) * x_hat_t 
            + c1 * epsilon 
            + c2 * epsilon_theta 
            + np.sqrt(alpha_t) * g
        ).detach()

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

