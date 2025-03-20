import torch
from tqdm import tqdm
import numpy as np
from PIL import Image

from pseudoInverse.operators import SuperResolutionPseudoinverseOperator, RotationOperator, IdentityOperator, GrayscaleOperator
from pseudoInverse.utils import DiffusionModel

from ddpm.utils import pilimg_to_tensor, save_pilimg
from ddpm.model import DDPM

class PiGDM:
    def __init__(self, model, measurement_operator, measurement_matrix=None, eta=1, guidance_factor=0.01, device='cuda'):
        self.model = model
        self.measurement_operator = measurement_operator
        self.H = measurement_matrix
        self.eta = eta
        self.guidance_factor = guidance_factor
        self.device = device

    def initialize(self, y, t):
        alpha_t = self.model.alphas_cumprod[t]
        x0 = self.measurement_operator.pseudoinverse(y).to(self.device)
        n = x0.size()
        t = torch.ones(n).to(x0.device).long() * t
        noise = torch.randn_like(x0)
        return np.sqrt(alpha_t) * x0 + np.sqrt((1 - alpha_t)) * noise

    def sample(self, 
                y,                        
                num_steps = 100,           
                sigma_y=None,               
                noiseless=True,             
                seed=None 
                ):
        
        device = 'cuda'
        T = self.model.num_diffusion_timesteps
        timesteps = torch.linspace(T // 2, 0, num_steps+1).long().to(self.device)

        xt = self.initialize(y, timesteps[0])

        for i in tqdm(range(num_steps)):
            t = timesteps[i]
            s = timesteps[i + 1]

            xt = xt.clone().requires_grad_(True)
            
            alpha_t = self.model.alphas_cumprod[t]
            alpha_s = self.model.alphas_cumprod[s]

            c1 = np.sqrt(((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t))) * self.eta
            c2 = np.sqrt((1 - alpha_s - c1**2))

            et, x0_pred = self.model(xt, t)
            if noiseless:
                if self.measurement_operator is None:
                    raise ValueError("measurement_operator must be provided for noiseless case")
                H = self.measurement_operator
                diff = (H.pseudoinverse(y.to(device)) - H.pseudoinverse(H(x0_pred.to(device)))).reshape(x0_pred.to(device).size(0), -1)
                mat_x = (diff.detach() * x0_pred.reshape(x0_pred.size(0), -1)).sum()
                g = torch.autograd.grad(mat_x, xt, retain_graph=True)[0].detach()
            else:
                if self.H is None:
                    raise ValueError("measurement_matrix must be provided for noisy case")
                H = self.H
                sigma_t = self.model.betas[t] ** 0.5
                r_t = ((sigma_t ** 2) / (1 + sigma_t ** 2)) ** 0.5

                HH_T = self.H @ self.H.T
                noise_term = (sigma_y**2 / r_t**2) * torch.eye(HH_T.shape[0], device=HH_T.device, dtype=HH_T.dtype)
                inv_matrix = torch.linalg.solve(HH_T + noise_term, torch.eye(HH_T.shape[0], device=HH_T.device, dtype=HH_T.dtype))
                
                mat_x = ((y - self.H @ x0_pred).detach() * ((inv_matrix @ self.H) @ x0_pred)).sum()
                g = torch.autograd.grad(mat_x, xt, retain_graph=False)[0].detach()
                

            # coeff = np.sqrt(alpha_s) * np.sqrt(alpha_t) * self.guidance_factor
            coeff = np.sqrt(alpha_t) * self.guidance_factor

            noise = torch.randn_like(xt)
            xt = (np.sqrt(alpha_s) * x0_pred 
                  + c1 * noise 
                  + c2 * et 
                  + coeff * g).detach()

        return xt



if __name__ == "__main__":
    # CONFIGURATION
    image_name = "00015"
    image_path = f"ddpm/diffusion-posterior-sampling/data/samples/{image_name}.png"
    scale_factor = 4  

    noiseless = False
    sigma_y = 0.1

    # Load image with PIL
    pil_img = Image.open(image_path).convert('RGB')
    tensor_img = pilimg_to_tensor(pil_img)

    # Setup measurement operator
    # measurement_operator = SuperResolutionPseudoinverseOperator(mode="bicubic", scale_factor=scale_factor)
    # measurement_operator = RotationOperator(45)
    # measurement_operator = IdentityOperator()
    # measurement_operator = RotationOperator(45)
    measurement_operator = GrayscaleOperator()
    measurement_matrix = torch.eye(256).to('cuda')


    # Create low-resolution image (measurement)
    low_res_img = measurement_operator(tensor_img)

    ddpm = DDPM()  

    # Wrap the model
    model = DiffusionModel(ddpm)

    # Instantiate the PGDM sampler
    pgdm_sampler = PiGDM(
        model=model,
        measurement_operator=measurement_operator,
        measurement_matrix=measurement_matrix,
        eta=1, 
        guidance_factor=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Perform super-resolution
    high_res_reconstructed = pgdm_sampler.sample(
        y=low_res_img,
        num_steps=500,
        sigma_y=sigma_y,
        noiseless=noiseless
    )

    # Save the high-resolution reconstructed image
    save_pilimg(high_res_reconstructed, f"DDIM_{image_name}.png")

    # Print the min and max pixel values
    print(f"Min: {high_res_reconstructed.min().item():.4f}, Max: {high_res_reconstructed.max().item():.4f}")
