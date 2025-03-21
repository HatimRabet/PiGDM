import os

from PiGDM.src.pseudoInverse.pigdm_ddim import PiGDM_DDIM, DiffusionModel
from pseudoInverse.operators import SuperResolutionOperator

from ddpm.model import DDPM

from experiments.utils import evaluate_model, ImageDataset, plot_results

from torch.utils.data import DataLoader, Subset

if __name__ == "__main__":
    dataset_path = "ffhq256-1k-validation"
    dataset_size = len(os.listdir(dataset_path))
    
    print(f"The Dataset FFHQ Contains : {dataset_size} Images\n")

    dataset = ImageDataset(dataset_path)

    subset_size = 10
    indices = list(range(subset_size)) 
    subset = Subset(dataset, indices)  
    test_loader_all = DataLoader(subset, batch_size=2, shuffle=False)

    # Create DataLoader
    measurement_operator = SuperResolutionOperator(mode="bicubic")

    # Intialize Model
    ddpm = DDPM()
    model = DiffusionModel(model=ddpm) 

    guidance_factor = 0.05
    num_steps = 100

    # Initialize sampler
    pigdm_sampler = PiGDM_DDIM(model, measurement_operator, guidance_factor=guidance_factor)

    results = evaluate_model(pigdm_sampler, test_loader_all, num_steps, sigma_y=None, noiseless=True, seed=None, device="cuda")
    plot_results(results)