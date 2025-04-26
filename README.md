# Interpreting-Variational-Autoencoders
# Pythae: A Unified Framework for Variational Autoencoders

Pythae is a comprehensive Python library for implementing, training, and benchmarking various Variational Autoencoder (VAE) models. This library aims to provide a unified and straightforward API for working with different VAE architectures.

## Installation

```bash
# Install via pip
pip install pythae

# Or clone and install the repository
git clone https://github.com/clementchadebec/benchmark_VAE.git
cd benchmark_VAE
pip install -e .

# Install additional dependencies
pip install torch torchvision matplotlib numpy
```

## Basic Usage

### Training a Simple VAE

```python
import torch
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from pythae.pipelines import TrainingPipeline

# 1. Define the model configuration
model_config = VAEConfig(
    input_dim=(1, 28, 28),  # Input dimensions (channels, height, width)
    latent_dim=10           # Size of latent space
)

# 2. Create the model
model = VAE(model_config)

# 3. Set up training configuration
training_config = BaseTrainerConfig(
    num_epochs=20,          # Number of training epochs
    learning_rate=1e-3,     # Learning rate
    output_dir="my_vae_output"  # Directory to save results
)

# 4. Create and run the training pipeline
pipeline = TrainingPipeline(
    training_config=training_config,
    model=model
)

# 5. Train the model
pipeline(
    train_data=train_dataset,  # Your training data
    eval_data=test_dataset     # Your evaluation data
)
```

### Loading and Using a Trained Model

```python
# Load a trained model
trained_model = VAE.load_from_folder("my_vae_output/VAE_training_2023-XX-XX_00-00-00/final_model")

# Generate reconstructions
with torch.no_grad():
    reconstructions = trained_model.reconstruct(test_samples)
    
# Visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for i in range(8):
    # Original
    plt.subplot(2, 8, i+1)
    plt.imshow(test_samples[i].squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    
    # Reconstructed
    plt.subplot(2, 8, i+9)
    plt.imshow(reconstructions[i].squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    
plt.tight_layout()
plt.show()
```

## Supported Models

Pythae implements numerous VAE variants:

- `VAE`: Standard Variational Autoencoder
- `BetaVAE`: VAE with β parameter for disentanglement
- `IWAE`: Importance Weighted Autoencoder
- `VAMP`: VAE with Vamprior
- `VAE_IAF`: VAE with Inverse Autoregressive Flows
- `VAE_LinNF`: VAE with Linear Normalizing Flows
- `FactorVAE`: VAE for disentangled representations
- `BetaTCVAE`: β-Total Correlation VAE
- `INFOVAE`: Information Maximizing VAE
- `WAE`: Wasserstein Autoencoder
- `VQVAE`: Vector Quantized VAE
- `MSSSIM-VAE`: VAE with MS-SSIM perceptual loss
- `VAEGAN`: VAE with adversarial training
- `RAE`: Regularized Autoencoder

## Generating New Samples

```python
from pythae.samplers import NormalSampler, NormalSamplerConfig, GaussianMixtureSampler

# 1. Set up a sampler
sampler_config = NormalSamplerConfig(n_samples=64)
sampler = NormalSampler(model=trained_model, sampler_config=sampler_config)

# 2. Generate samples
samples = sampler.sample(num_samples=16)

# 3. Visualize samples
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(samples[i].squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.suptitle("Generated Samples")
plt.tight_layout()
plt.show()

# Alternative: Use GMM sampler for better quality
gmm_sampler = GaussianMixtureSampler(model=trained_model, n_components=10)
gmm_samples = gmm_sampler.sample(num_samples=16)
```

## Exploring the Latent Space

### Latent Space Interpolation

```python
# Interpolate between two points in latent space
z1 = torch.randn(1, trained_model.latent_dim).to(device)
z2 = torch.randn(1, trained_model.latent_dim).to(device)

steps = 10
interpolations = []

for alpha in np.linspace(0, 1, steps):
    z_interp = z1 * (1 - alpha) + z2 * alpha
    interp_output = trained_model.decoder(z_interp)
    interpolations.append(interp_output.reconstruction)

# Visualize interpolation
plt.figure(figsize=(15, 3))
for i in range(steps):
    plt.subplot(1, steps, i+1)
    plt.imshow(interpolations[i].squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
```

### Analyzing Latent Dimensions

```python
# Analyze the effect of each latent dimension
z_base = torch.zeros(1, trained_model.latent_dim).to(device)
variations = [-3, -2, -1, 0, 1, 2, 3]

for dim in range(trained_model.latent_dim):
    plt.figure(figsize=(14, 2))
    
    for i, val in enumerate(variations):
        z = z_base.clone()
        z[0, dim] = val
        
        output = trained_model.decoder(z)
        img = output.reconstruction
        
        plt.subplot(1, len(variations), i+1)
        plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"z_{dim}={val}")
        plt.axis('off')
        
    plt.suptitle(f"Effect of latent dimension {dim}")
    plt.tight_layout()
    plt.show()
```

## Benchmarking VAE Models

Pythae allows comprehensive benchmarking of different models:

```python
# Train and compare models with different parameters
beta_values = [0.5, 1.0, 4.0, 10.0]
results = {}

for beta in beta_values:
    # Create and train Beta-VAE with specific beta
    config = BetaVAEConfig(
        input_dim=(1, 28, 28),
        latent_dim=10,
        beta=beta
    )
    model = BetaVAE(config)
    
    # Train model
    # ...
    
    # Evaluate reconstruction loss
    loss = compute_reconstruction_loss(model, test_dataset)
    results[f"beta_{beta}"] = loss

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(beta_values, [results[f"beta_{beta}"] for beta in beta_values], 'o-')
plt.xlabel('Beta Value')
plt.ylabel('Reconstruction Loss')
plt.title('Effect of Beta Value on Reconstruction Loss')
plt.grid(True)
plt.show()
```

## Advanced Features

### Custom Encoder/Decoder Architectures

```python
import torch.nn as nn
from pythae.models.base.base_utils import ModelOutput

# Define custom encoder
class CustomEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_dim), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)
        )
        
    def forward(self, x):
        h = self.net(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        return ModelOutput(embedding=mu, log_covariance=log_var)

# Use custom encoder with VAE
model = VAE(
    model_config=model_config,
    encoder=CustomEncoder(input_dim=(1, 28, 28), latent_dim=10)
)
```

### Temperature-Based Sampling

```python
# Control sample variability with temperature
temperature = 0.8  # Values < 1 make samples more conservative
z = torch.randn(16, trained_model.latent_dim).to(device) * temperature
samples = trained_model.decoder(z)

plt.figure(figsize=(8, 4))
for i in range(16):
    plt.subplot(2, 8, i+1)
    plt.imshow(samples[i].squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.suptitle(f"Temperature Sampling (T={temperature})")
plt.tight_layout()
plt.show()
```

## Integration with Experiment Tracking

```python
# Using wandb for experiment tracking
from pythae.trainers import BaseTrainerConfig

training_config = BaseTrainerConfig(
    num_epochs=20,
    learning_rate=1e-3,
    output_dir="vae_experiments",
    use_wandb=True,
    wandb_project="vae-benchmark"
)
```

## Resources

- [Documentation](https://pythae.readthedocs.io/)
- [GitHub Repository](https://github.com/clementchadebec/benchmark_VAE)
- [Paper: Pythae: Unifying Generative Autoencoders in Python](https://arxiv.org/abs/2203.00867)

## Citation

If you use Pythae in your research, please cite:

```bibtex
@inproceedings{chadebec2022pythae,
  title={Pythae: Unifying Generative Autoencoders in Python - A Benchmarking Use Case},
  author={Chadebec, Cl{\'e}ment and Vincent, Louis J. and Allassonni{\`e}re, St{\'e}phanie},
  booktitle={36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks},
  year={2022}
}
```
