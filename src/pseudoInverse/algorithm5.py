import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from pseudoInverse.operators import SuperResolutionPseudoinverseOperator, RotationOperator, IdentityOperator
from pseudoInverse.utils import DiffusionModel

from ddpm.model import DDPM
from ddpm.utils import pilimg_to_tensor, save_pilimg


class PiGDM:
    def __init__(self, model, measurement_operator, measurement_matrix=None, eta=1, guidance_factor=0.005, device='cuda'):
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
        num_steps = 100,           
        sigma_y=None,               
        noiseless=True,             
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
                # Calculate h(x_hat_t)
                h_x_hat = self.measurement_operator(x_hat_t)
                
                # Calculate h†(y) - h†(h(x_hat_t))
                h_pseudoinv_y = self.measurement_operator.pseudoinverse(y)
                h_pseudoinv_h_x_hat = self.measurement_operator.pseudoinverse(h_x_hat)
                
                # Calculate guidance term
                diff = h_pseudoinv_y - h_pseudoinv_h_x_hat
                mat_x = (diff.detach() * x_hat_t).sum()      
                g = torch.autograd.grad(mat_x, x, retain_graph=False)[0].detach()
            else:
                if self.H is None:
                    raise ValueError("measurement_matrix must be provided for noisy case")
                sigma_t = betas[t] ** 0.5
                r_t = ((sigma_t ** 2) / (1 + sigma_t ** 2)) ** 0.5

                HH_T = self.H @ self.H.T
                noise_term = (sigma_y**2 / r_t**2) * torch.eye(HH_T.shape[0], device=HH_T.device, dtype=HH_T.dtype)
                inv_matrix = torch.linalg.solve(HH_T + noise_term, torch.eye(HH_T.shape[0], device=HH_T.device, dtype=HH_T.dtype))
                
                mat_x = ((y - self.H @ x_hat_t).detach() * ((inv_matrix @ self.H) @ x_hat_t)).sum()
                g = torch.autograd.grad(mat_x, x, retain_graph=False)[0].detach()

            
            x = x.detach()
            # PiGDM update (last part of the algorithm)
            x = (mut + np.sqrt(betas[t]) * z + torch.sqrt(alphas_cumprod[t]) * self.guidance_factor * g).detach()
            
        return x


if __name__ == "__main__":
    # CONFIGURATION
    guidance_factor = 0.01
    num_steps = 1000
    
    image_path = "ddpm/diffusion-posterior-sampling/data/samples/00015.png"
    scale_factor = 4  

    noiseless = False
    sigma_y = 0.1

    # Load image with PIL
    pil_img = Image.open(image_path).convert('RGB')

    tensor_img = pilimg_to_tensor(pil_img)
    # measurement_operator = SuperResolutionPseudoinverseOperator(mode="bicubic")
    measurement_operator = IdentityOperator()
    # measurement_operator = RotationOperator(45)
    measurement_matrix = torch.eye(256).to('cuda')
    low_res_img = measurement_operator(tensor_img)
    
    low_res_img_show = measurement_operator.pseudoinverse(low_res_img)
 
    # Intialize Model
    ddpm = DDPM()
    model = DiffusionModel(model=ddpm) 

    # Initialize sampler
    pidgm_sampler = PiGDM(model, measurement_operator, measurement_matrix, guidance_factor=guidance_factor)

    # Result
    high_res_img = pidgm_sampler.sample(low_res_img, num_steps, sigma_y, noiseless)
    out = torch.cat((low_res_img_show, high_res_img, tensor_img), dim = 2)

    print(f"Min: {high_res_img.min().item()}, Max: {high_res_img.max().item()}")

    # Save Image
    save_pilimg(out, "image_15.jpg")