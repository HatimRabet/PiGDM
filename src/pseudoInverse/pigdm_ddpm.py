import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from pseudoInverse.operators import SuperResolutionOperator, RotationOperator, IdentityOperator, GrayscaleOperator, BlurOperator, InpaintingOperator
from pseudoInverse.utils import DiffusionModel, noiseless_guidance, matrix_based_noisy_guidance, operator_based_noisy_guidance

from ddpm.model import DDPM
from ddpm.utils import pilimg_to_tensor, save_pilimg


class PiGDM_DDPM:
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
                g = noiseless_guidance(
                    x = x,
                    x_hat_t=x_hat_t,
                    y = y,
                    measurement_operator=self.measurement_operator
                )
                
            else:
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
            x = (mut + np.sqrt(betas[t]) * z + self.eta * torch.sqrt(alphas_cumprod[t]) * self.guidance_factor * g).detach()
            
        return x
    


if __name__ == "__main__":
    # CONFIGURATION
    guidance_factor = 0.01
    num_steps = 1000
    
    image_name = "00877"
    
    image_path = f"dataset/samples/{image_name}.png"

    noiseless = True
    sigma_y = 0.05
    # Load image with PIL
    pil_img = Image.open(image_path).convert('RGB')

    tensor_img = pilimg_to_tensor(pil_img)
    measurement_operator = SuperResolutionOperator(mode="bicubic")

    low_res_img = measurement_operator(tensor_img)
    if not noiseless :
        low_res_img += sigma_y * torch.randn_like(low_res_img)
    
    low_res_img_show = measurement_operator.pseudoinverse(low_res_img)
 
    # Intialize Model
    ddpm = DDPM()
    model = DiffusionModel(model=ddpm) 

    # Initialize sampler
    pidgm_sampler = PiGDM_DDPM(model, measurement_operator, guidance_factor=guidance_factor)

    # Result
    high_res_img = pidgm_sampler.sample(low_res_img, num_steps, sigma_y, noiseless)
    out = torch.cat((low_res_img_show, high_res_img, tensor_img), dim = 2)    
    
    print(f"Min: {high_res_img.min().item()}, Max: {high_res_img.max().item()}")

    # Save Image
    save_pilimg(out, f"DDPM_{image_name}.png")
