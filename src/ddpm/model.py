import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from ddpm.guided_diffusion.unet import create_model
from ddpm.utils import display_as_pilimg


# CONFIG
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_config = {
    "image_size": 256,
    "num_channels": 128,
    "num_res_blocks": 1,
    "channel_mult": "",
    "learn_sigma": True,
    "class_cond": False,
    "use_checkpoint": False,
    "attention_resolutions": 16,
    "num_heads": 4,
    "num_head_channels": 64,
    "num_heads_upsample": -1,
    "use_scale_shift_norm": True,
    "dropout": 0.0,
    "resblock_updown": True,
    "use_fp16": False,
    "use_new_attention_order": False,
    "model_path": "ffhq_10m.pt",
}

model = create_model(**model_config)
model = model.to(device)
model.eval()


class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM) class for sampling and posterior inference.
    """

    def __init__(self, model=model):
        self.num_diffusion_timesteps = 1000
        self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
        beta_start = 0.0001
        beta_end = 0.02
        self.betas = np.linspace(
            beta_start, beta_end, self.num_diffusion_timesteps, dtype=np.float64
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.model = model
        self.imgshape = (1, 3, 256, 256)
        
    def __call__(self, x, t):
        return self.get_eps_from_model(x, t)

    def get_eps_from_model(self, x, t):
        model_output = self.model(x, torch.tensor(t, device=device).unsqueeze(0))
        model_output = model_output[:, :3, :, :]
        return model_output

    def predict_xstart_from_eps(self, x, eps, t):
        x_start = (
            np.sqrt(1.0 / self.alphas_cumprod[t]) * x
            - np.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
        )
        x_start = x_start.clamp(-1.0, 1.0)
        return x_start

    def sample(self, show_steps=True): # probably wouldnt work without notebook
        with torch.no_grad():
            x = torch.randn(self.imgshape, device=device)
            for i, t in enumerate(self.reversed_time_steps):
                eps = self.get_eps_from_model(x, t)
                xhat = self.predict_xstart_from_eps(x, eps, t)
                z = torch.randn_like(x, device=device)
                mut = (
                    x - self.betas[t] * eps / (np.sqrt(1 - self.alphas_cumprod[t]))
                ) / np.sqrt(self.alphas[t])
                x = mut + np.sqrt(self.betas[t]) * z
                
                print(f"Min: {x.min().item()}, Max: {x.max().item()}")

                if i == 0 or t % 100 == 0 or t == 0:
                    print("Iteration:", i, "; Discrete time:", t)
                    if show_steps:
                        pilimg = display_as_pilimg(torch.cat((x, xhat), dim=2))
        return x

    def posterior_sampling(
        self, linear_operator, y, x_true=None, show_steps=True, vis_y=None
    ):
        if vis_y is None:
            vis_y = y

        x = torch.randn(self.imgshape, device=device)
        x.requires_grad = True

        if vis_y.shape != x.shape:
            vis_y = F.interpolate(
                vis_y, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            print("vis_y shape", vis_y.shape)

        for i, t in enumerate(self.reversed_time_steps):
            eps = self.get_eps_from_model(x, t)
            xhat = self.predict_xstart_from_eps(x, eps, t)

            loss = torch.sum((linear_operator(xhat) - y) ** 2)
            grad_x = torch.autograd.grad(outputs=loss, inputs=x)[0]
            zeta_t = 0.1 / torch.sqrt(loss)

            mut = (
                x - self.betas[t] * eps / np.sqrt(1 - self.alphas_cumprod[t])
            ) / np.sqrt(self.alphas[t])
            z = torch.randn_like(x)

            x = mut + np.sqrt(self.betas[t]) * z - zeta_t * grad_x

            if i == 0 or t % 100 == 0 or t == 0:
                print("Iteration:", i, "; Discrete time:", t)
                if show_steps:
                    if x_true is not None:
                        pilimg = display_as_pilimg(
                            torch.cat((x, xhat, vis_y, x_true), dim=3)
                        )
                    else:
                        pilimg = display_as_pilimg(torch.cat((x, xhat, vis_y), dim=3))
        return x

    def ode_euler_sampling(
        self,
        nb_times_integration=1000,
        noise_seed=None,
        show_steps=True,
        show_interval=100,
    ):
        x = torch.randn(self.imgshape, device=device)
        T = 1000
        delta = 1000 // nb_times_integration

        with torch.no_grad():
            for i in range(nb_times_integration):
                t = T - i * delta - 1
                eps = self.get_eps_from_model(x, t)
                s_theta = -eps / np.sqrt(1 - self.alphas_cumprod[t])
                xhat = self.predict_xstart_from_eps(x, eps, t)
                x = x + self.betas[t] * (x + s_theta) * delta / 2

                if show_steps and t % show_interval == 0:
                    pilimg = display_as_pilimg(torch.cat((x, xhat), dim=3))
        return x
