from seaborn import reset_defaults
from pseudoInverse.algorithm5 import PiGDM, DiffusionModel
from pseudoInverse.operators import SuperResolutionPseudoinverseOperator
from ddpm.model import DDPM
from experiments.utils import evaluate_model, ImageDataset, plot_results
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import os

if __name__ == "__main__":
    dataset_path = "ffhq256-1k-validation"
    dataset_size = len(os.listdir(dataset_path))
    
    print(f"The Dataset FFHQ Contains : {dataset_size} Images\n")

    dataset = ImageDataset(dataset_path)

    # Create DataLoader
    test_loader_all = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    subset_size = 5
    indices = list(range(subset_size))  
    test_loader = Subset(test_loader_all, indices)

    measurement_operator = SuperResolutionPseudoinverseOperator(mode="bicubic")

    # Intialize Model
    ddpm = DDPM()
    model = DiffusionModel(model=ddpm) 

    guidance_factor = 0.01
    num_steps = 1000

    # Initialize sampler
    pigdm_sampler = PiGDM(model, measurement_operator, guidance_factor=guidance_factor)

    results = evaluate_model(pigdm_sampler, test_loader, num_steps, sigma_y=None, noiseless=True, seed=None, device="cuda")
    plot_results(results)


