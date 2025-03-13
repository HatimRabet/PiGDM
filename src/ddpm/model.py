import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from guided_diffusion.unet import create_model



# CONFIG
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_config = {'image_size': 256,
                'num_channels': 128,
                'num_res_blocks': 1,
                'channel_mult': '',
                'learn_sigma': True,
                'class_cond': False,
                'use_checkpoint': False,
                'attention_resolutions': 16,
                'num_heads': 4,
                'num_head_channels': 64,
                'num_heads_upsample': -1,
                'use_scale_shift_norm': True,
                'dropout': 0.0,
                'resblock_updown': True,
                'use_fp16': False,
                'use_new_attention_order': False,
                'model_path': 'ffhq_10m.pt'}

model = create_model(**model_config)
model = model.to(device)
# use in eval mode:
model.eval();




class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM) class for sampling and posterior inference.

    This class provides methods for:
      - Getting the predicted noise (epsilon) from the model.
      - Predicting the original image (x_start) from the noisy image.
      - Sampling from the diffusion model.
      - Performing posterior sampling with a linear operator constraint.
      - Sampling using an Euler ODE integration approach.

    Attributes:
        num_diffusion_timesteps (int): Total number of diffusion steps.
        reversed_time_steps (np.ndarray): Array of timesteps in reverse order.
        betas (np.ndarray): Linear schedule of beta values.
        alphas (np.ndarray): Complementary values (1 - beta) for each timestep.
        alphas_cumprod (np.ndarray): Cumulative product of alphas.
        alphas_cumprod_prev (np.ndarray): Cumulative product of alphas for the previous timestep.
        model: The denoising model used for epsilon prediction.
        imgshape (tuple): Shape of the image tensor.
    """

  def __init__(self, model=model):
    """
    Initialize the DDPM instance.

    Args:
        model: A PyTorch model used for noise prediction. It should be compatible with the guided diffusion framework.
    """
    self.num_diffusion_timesteps = 1000
    self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
    beta_start = 0.0001
    beta_end = 0.02
    self.betas = np.linspace(beta_start, beta_end, self.num_diffusion_timesteps,
                              dtype=np.float64)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
    self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
    self.model = model
    self.imgshape = (1,3,256,256)


  def get_eps_from_model(self, x, t):
    """
    Predict the noise epsilon from the model at a given timestep.

    The model is expected to output a tensor where the first three channels represent
    the estimated noise (epsilon). Any additional channels (e.g. learned variances)
    are discarded in this implementation.

    Args:
        x (torch.Tensor): The current noisy image tensor.
        t (int): The current timestep.

    Returns:
        torch.Tensor: The predicted noise tensor (epsilon) with the same spatial dimensions as x.
    """
    model_output = self.model(x, torch.tensor(t, device=device).unsqueeze(0))
    model_output = model_output[:,:3,:,:]
    return(model_output)

  def predict_xstart_from_eps(self, x, eps, t):
    """
    Predict the original image x_start from the noisy image x and the predicted noise epsilon.

    Uses the following relationship:
        x_start = sqrt(1 / alpha_cumprod[t]) * x - sqrt(1 / alpha_cumprod[t] - 1) * eps

    The result is clamped to the range [-1, 1].

    Args:
        x (torch.Tensor): The current noisy image tensor.
        eps (torch.Tensor): The predicted noise tensor.
        t (int): The current timestep.

    Returns:
        torch.Tensor: The predicted original image tensor (x_start), clamped between -1 and 1.
    """
    x_start = (
        np.sqrt(1.0 / self.alphas_cumprod[t])* x
        - np.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
    )
    x_start = x_start.clamp(-1.,1.)
    return(x_start)

  def sample(self, show_steps=True):
    """
    Generate a sample image by iteratively reversing the diffusion process.

    Starting from a random noise tensor, this method iterates backwards over the diffusion timesteps,
    predicting the noise, estimating the original image, and updating the sample using a stochastic step.
    Optionally, intermediate steps are displayed.

    Args:
        show_steps (bool): If True, displays intermediate steps using the display_as_pilimg function.

    Returns:
        torch.Tensor: The final sampled image tensor.
    """
    with torch.no_grad():  # avoid backprop wrt model parameters
      x = torch.randn(self.imgshape,device=device)  # initialize x_t for t=T
      for i, t in enumerate(self.reversed_time_steps):

          # TODO
          eps = self.get_eps_from_model(x, t)
          xhat = self.predict_xstart_from_eps(x, eps, t)
          z = torch.randn_like(x, device = device)
          mut = (x - self.betas[t] * eps / (np.sqrt(1 - self.alphas_cumprod[t]))) / np.sqrt(self.alphas[t])
          x = mut + np.sqrt(self.betas[t]) * z

          if i==0 or t%100==0 or t==0:
            print('Iteration:', i, '; Discrete time:', t)
            if show_steps:
                # Display xt and xhat
                pilimg = display_as_pilimg(torch.cat((x, xhat), dim=2))

    return(x)

  def posterior_sampling(self, linear_operator, y, x_true=None, show_steps=True, vis_y=None):
    """
    Perform posterior sampling with a linear operator constraint.

    This method applies an iterative update on the noisy image tensor x using the gradient
    of a loss function defined between the transformed predicted image (xhat) and the observation y.
    An optional ground truth (x_true) and visualization image (vis_y) may be provided for comparison.

    Args:
        linear_operator (callable): A function that applies the forward model (linear operator) on an image tensor.
        y (torch.Tensor): The observed data tensor.
        x_true (torch.Tensor, optional): Ground truth image tensor for visualization. Defaults to None.
        show_steps (bool): If True, displays intermediate steps using display_as_pilimg. Defaults to True.
        vis_y (torch.Tensor, optional): A tensor for visualizing the observation. If None, y is used.

    Returns:
        torch.Tensor: The final image tensor after posterior sampling.
    """
    # visualization image for the observation y:
    if vis_y is None:
      vis_y = y

    # initialize xt for t=T
    x = torch.randn(self.imgshape,device=device)
    x.requires_grad = True

    # Make vis_y have the same size as x!
    if vis_y.shape != x.shape:
      # Interpolation
      vis_y = F.interpolate(vis_y, size=x.shape[-2:], mode='bilinear', align_corners=False)
      print("vis_y shape", vis_y.shape)

    # TODO
    for i, t in enumerate(self.reversed_time_steps):
      # Predict xhat
      eps = self.get_eps_from_model(x, t)
      xhat = self.predict_xstart_from_eps(x, eps, t)

      loss = torch.sum((linear_operator(xhat) - y)**2)

      # Compute the gradient of the loss w.r.t x
      grad_x = torch.autograd.grad(outputs = loss, inputs = x)[0]
      zeta_t = 0.1 / torch.sqrt(loss)

      # Update x
      mut = (x - self.betas[t] * eps / np.sqrt(1 - self.alphas_cumprod[t])) / np.sqrt(self.alphas[t])
      z = torch.randn_like(x)

      x = mut + np.sqrt(self.betas[t]) * z - zeta_t * grad_x

      if i==0 or t%100==0 or t==0:
            print('Iteration:', i, '; Discrete time:', t)
            if show_steps:
                # Display xt and xhat and vis_y and x_true
                if x_true is not None:
                  pilimg = display_as_pilimg(torch.cat((x, xhat, vis_y, x_true), dim=3))
                else:
                  pilimg = display_as_pilimg(torch.cat((x, xhat, vis_y), dim=3))

    return(x)

  def ode_euler_sampling(self, nb_times_integration=1000, noise_seed = None, show_steps=True, show_interval = 100):
    """
    Sample using Euler's method to solve an ODE approximation of the reverse diffusion process.

    This method uses a discretized Euler integration scheme to update the image tensor over a given number
    of integration steps. Intermediate results are optionally displayed.

    Args:
        nb_times_integration (int): Number of integration steps. Defaults to 1000.
        noise_seed (optional): An optional seed for noise generation. (Note: not used in the current implementation.)
        show_steps (bool): If True, displays intermediate results. Defaults to True.
        show_interval (int): Interval of timesteps at which to show the intermediate images. Defaults to 100.

    Returns:
        torch.Tensor: The final image tensor after ODE Euler sampling.
    """
    # Initialize xt for t = T
    x = torch.randn(self.imgshape,device=device)
    T = 1000
    delta = 1000 // nb_times_integration

    # number of sub-steps
    with torch.no_grad():
      for i in range(nb_times_integration):
        # Integration for nb_times integration
        t = T - i * delta - 1
        eps = self.get_eps_from_model(x, t)
        s_theta = - eps / np.sqrt(1 - self.alphas_cumprod[t])
        xhat = self.predict_xstart_from_eps(x, eps, t)
        x = x + self.betas[t] * (x + s_theta) * delta / 2

        if show_steps and t%show_interval == 0:
          pilimg = display_as_pilimg(torch.cat((x, xhat), dim=3))
    return x


