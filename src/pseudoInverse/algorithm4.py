import torch
from tqdm import tqdm
import numpy as np
from ddpm.utils import pilimg_to_tensor, save_pilimg

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from pseudoInverse.operators import SuperResolutionPseudoinverseOperator
from ddpm.model import DDPM


class PiGDM:
    def __init__(self, model, measurement_operator, eta=1, grad_term_weight=0.01, device='cuda'):
        self.model = model
        self.measurement_operator = measurement_operator
        self.eta = eta
        self.grad_term_weight = grad_term_weight
        self.device = device

    def initialize(self, y, t):
        alpha_t = self.model.alphas_cumprod[t]
        x0 = self.measurement_operator.pseudoinverse(y).to(self.device)
        n = x0.size()
        t = torch.ones(n).to(x0.device).long() * t
        noise = torch.randn_like(x0)
        return np.sqrt(alpha_t) * x0 + np.sqrt((1 - alpha_t)) * noise

    def sample(self, y, num_steps=100):
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

            H = self.measurement_operator
            diff = (H.pseudoinverse(y.to(device)) - H.pseudoinverse(H(x0_pred.to(device)))).reshape(x0_pred.to(device).size(0), -1)
            mat_x = (diff.detach() * x0_pred.reshape(x0_pred.size(0), -1)).sum()
            grad_term = torch.autograd.grad(mat_x, xt, retain_graph=True)[0].detach()

            # coeff = np.sqrt(alpha_s) * np.sqrt(alpha_t) * self.grad_term_weight
            coeff = np.sqrt(alpha_t) * self.grad_term_weight


            noise = torch.randn_like(xt)
            xt = (np.sqrt(alpha_s) * x0_pred 
                  + c1 * noise 
                  + c2 * et 
                  + coeff * grad_term).detach()

        return xt


class DiffusionWrapper:
    def __init__(self, model):
        self.model = model
        self.alphas_cumprod = model.alphas_cumprod
        self.imgshape = model.imgshape
        self.num_diffusion_timesteps = model.num_diffusion_timesteps

    def __call__(self, x, t):
        eps = self.model(x, t)
        x0_pred = self.model.predict_xstart_from_eps(x, eps=eps, t=t)
        return eps, x0_pred


if __name__ == "__main__":
    # CONFIGURATION
    image_path = "ddpm/diffusion-posterior-sampling/data/samples/00003.png"
    scale_factor = 4  

    # Load image with PIL
    pil_img = Image.open(image_path).convert('RGB')
    tensor_img = pilimg_to_tensor(pil_img)

    # Setup measurement operator
    measurement_operator = SuperResolutionPseudoinverseOperator(mode="bicubic", scale_factor=scale_factor)

    # Create low-resolution image (measurement)
    low_res_img = measurement_operator(tensor_img)

    ddpm = DDPM()  

    # Wrap the model
    wrapped_model = DiffusionWrapper(ddpm)

    # Instantiate the PGDM sampler
    pgdm_sampler = PiGDM(
        model=wrapped_model,
        measurement_operator=measurement_operator,
        eta=1, 
        grad_term_weight=0.05,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Perform super-resolution
    high_res_reconstructed = pgdm_sampler.sample(
        y=low_res_img,
        num_steps=100
    )

    # Save the high-resolution reconstructed image
    save_pilimg(high_res_reconstructed[-1], "reconstructed_image.png")

    # Print the min and max pixel values
    print(f"Min: {high_res_reconstructed.min().item():.4f}, Max: {high_res_reconstructed.max().item():.4f}")