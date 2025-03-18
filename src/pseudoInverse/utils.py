# Diffusion Model Wrapper
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