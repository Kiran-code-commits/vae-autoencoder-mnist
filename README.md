# vae-autoencoder-mnist
Variational Autoencoder and Bottleneck Autoencoder implemented in PyTorch on MNIST. Compares reconstruction quality and latent space structure.
# Autoencoder & Variational Autoencoder on MNIST

Implementation and comparison of a Bottleneck Autoencoder (AE) and 
Variational Autoencoder (VAE) in PyTorch on the MNIST dataset.

## Overview
Two unsupervised generative architectures trained for 20 epochs using 
Adam (lr=0.001), batch size 128, on normalized MNIST inputs.

## Autoencoder (AE)
Symmetric 3-layer architecture: 784→512→256→d→256→512→784  
Evaluated across four bottleneck dimensions with MSE reconstruction error:

| Bottleneck Dim | Test MSE |
|----------------|----------|
| 100            | 0.004221 |
| 50             | 0.004611 |
| 30             | 0.005163 |
| 20             | 0.006909 |

Compression threshold identified below dim=30 — complex digits (8, 5) 
show visible blurring at dim=20.

## Variational Autoencoder (VAE)
Encoder produces μ and log σ² heads; latent variable sampled via the 
reparameterization trick. Compared 2D vs 10D latent spaces:

| Model  | Total Loss | BCE   | KL    |
|--------|------------|-------|-------|
| 2D VAE | 139.59     | 132.86| 6.73  |
| 10D VAE| 100.66     | 81.78 | 18.88 |

10D latent space achieves 38% lower BCE and well-separated digit clusters 
(via t-SNE), at the cost of severe uniform sampling holes due to the 
curse of dimensionality.

## Key Finding
Standard normal sampling z~N(0,I) is the only correct generation strategy 
for VAEs regardless of latent dimensionality — uniform sampling over 
[-3,3]^d exposes latent holes that worsen exponentially with dimension.

## Tech Stack
Python, PyTorch, NumPy, Matplotlib
