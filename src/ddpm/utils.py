import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm


def pilimg_to_tensor(pil_img):
    """
    Convert a PIL image to a normalized PyTorch tensor.

    This function converts an input PIL image to a tensor and scales its pixel values from [0, 1]
    to [-1, 1]. It then adds a batch dimension and moves the tensor to a predefined device.

    Parameters:
        pil_img (PIL.Image.Image): The input image in PIL format.

    Returns:
        torch.Tensor: A tensor of shape [1, C, H, W] with pixel values in the range [-1, 1].

    Note:
        The variable `device` must be defined globally before calling this function.
    """
    t = torchvision.transforms.ToTensor()(pil_img)
    t = 2 * t - 1  # Scale from [0,1] to [-1,1]
    t = t.unsqueeze(0)
    t = t.to(device)
    return t


def display_as_pilimg(t):
    """
    Convert a tensor to a PIL image, display it, and return the image.

    This function rescales an input tensor from [-1, 1] back to [0, 1], moves it to CPU,
    removes any singleton dimensions, clamps the values to ensure they are within [0, 1],
    and converts it into a PIL image. The image is then displayed.

    Parameters:
        t (torch.Tensor): A tensor of shape [1, C, H, W] with values in the range [-1, 1].

    Returns:
        PIL.Image.Image: The converted PIL image.
    """
    t = 0.5 + 0.5 * t.to('cpu')
    t = t.squeeze()
    t = t.clamp(0., 1.)
    pil_img = torchvision.transforms.ToPILImage()(t)
    display(pil_img)
    return pil_img


def visualize_denoiser(ddpm, img_pil):
    """
    Visualize the denoising process of a diffusion model on a given image.

    This function demonstrates the iterative denoising process of a denoising diffusion 
    probabilistic model (ddpm) applied to an input image. It converts the image to a tensor,
    and for each diffusion timestep, it:
      - Adds Gaussian noise to the image tensor.
      - Uses the model to predict the noise component.
      - Computes the predicted original image from the noisy tensor.
      - Calculates the Peak Signal-to-Noise Ratio (PSNR) for both the noisy and the denoised images.
      - Displays a concatenated image (noisy, denoised, and original) every 100 iterations.

    Parameters:
        ddpm: An instance of a denoising diffusion probabilistic model which must provide:
              - num_diffusion_timesteps (int): Total number of diffusion steps.
              - alphas (list or tensor): Alpha values for each timestep.
              - betas (list or tensor): Beta values for each timestep.
              - get_eps_from_model(xt, t): Method to predict the noise given the current tensor and timestep.
              - predict_xstart_from_eps(xt, epst, t): Method to estimate the original image from the noisy tensor and predicted noise.
        img_pil (PIL.Image.Image): The input image in PIL format.

    Returns:
        None
    """
    x0 = pilimg_to_tensor(img_pil)
    psnr_noisy = []
    psnr_denoised = []

    def mypsnr(x, y):
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR) between two tensors.

        Parameters:
            x (torch.Tensor): First image tensor.
            y (torch.Tensor): Second image tensor.

        Returns:
            float: The PSNR value in decibels.
        """
        error = torch.mean((x - y) ** 2).item()
        psnr = 10 * np.log10(2 ** 2 / error)
        return psnr

    xt = x0.clone()  # Initialize xt with the original image tensor
    for t in range(ddpm.num_diffusion_timesteps):
        with torch.no_grad():  # Avoid computing gradients with respect to model parameters
            # Sample Gaussian noise (zt) and perform a forward diffusion step.
            zt = torch.randn_like(xt)  # Preserves the device of xt
            xt = np.sqrt(ddpm.alphas[t]) * xt + np.sqrt(ddpm.betas[t]) * zt

            # Compute the predicted noise and estimate the original image (xhat)
            epst = ddpm.get_eps_from_model(xt, t)
            xhat = ddpm.predict_xstart_from_eps(xt, epst, t)

            # Calculate and record PSNR for noisy and denoised images.
            psnr_noisy.append(mypsnr(xt, x0))
            psnr_denoised.append(mypsnr(xhat, x0))

            # Every 100 iterations, display the current state of the images.
            if (t + 1) % 100 == 0:
                print('Iteration:', t + 1)
                # Concatenate images along the width (dim=3) for side-by-side comparison.
                pilimg = display_as_pilimg(torch.cat((xt, xhat, x0), dim=3))

