# Resonance

Wireless Channel Denoising using Deep Convolutional Autoencoders

---

## Overview
**Resonance** is a proof‑of‑concept project that demonstrates the feasibility of using deep learning for denoising wireless channel matrices.  
Accurate Channel State Information (CSI) is critical for 5G/IoT systems, but raw measurements are corrupted by noise, hardware imperfections, and multipath interference.  
This project leverages a **Convolutional Autoencoder (CAE)** to learn noise distributions directly from data and reconstruct cleaner channel estimates.

---

## Problem Statement
- **Noisy channel matrices** degrade wireless system performance.  
- Classical methods (Least Squares, Wiener, Kalman filters) assume Gaussian noise and fail at low SNR.  
- Pilot signals consume bandwidth and are not adaptive.  
- Deep learning offers a **data‑driven, adaptive, real‑time** alternative.

---

## Dataset
- Source: **DeepMIMO** synthetic dataset (ray‑tracing based).  
- Initial file: `channels_o1_60.npy` → 51 samples.  
- Augmentation strategies (physics‑motivated):  
  - Gaussian noise injection (120 samples)  
  - Amplitude scaling (120 samples)  
  - Phase perturbation (150 samples)  
  - Spatial flips (153 samples)  
- **Final dataset size**: 594 samples (11.7× growth).  
- Input shape: `64 × 64 × 2` (Amplitude + Phase maps).

---

## Preprocessing & Feature Extraction
- **Normalization**: Per‑sample unit norm (L₂ = 1).  
- **Resizing**: Pad/truncate to `64 × 64`.  
- **Feature decomposition**:  
  - Amplitude: \( A = |H| \)  
  - Phase: \( \phi = \angle H \)  
- Output tensor: `(64, 64, 2)`.

---

## Model Architecture
**Convolutional Autoencoder (CAE)**

- **Encoder**  
  - Conv2D → MaxPooling → Conv2D → MaxPooling  
  - Compresses input to latent space `(16 × 16 × 64)`.

- **Latent Space**  
  - Compact representation of multipath structure.  
  - Noise discarded during compression.

- **Decoder**  
  - Conv2DTranspose → Conv2DTranspose → Conv2D Output  
  - Reconstructs denoised amplitude + phase maps.

- **Training Setup**  
  - Optimizer: Adam (lr = 1e‑3)  
  - Loss: Mean Squared Error (MSE)  
  - Epochs: 50  
  - Batch size: 16  
  - Hardware: GPU (CUDA)

---

## Results
- **Reconstruction MSE**: ~0.17  
- **Training time**: < 5 minutes (50 epochs)  
- **Observations**:  
  - Amplitude maps reconstructed smoothly.  
  - Phase maps show higher residual error (due to 2π discontinuities).  
  - Multipath structure preserved.  
  - Stable convergence (~30 epochs).

---

## Contributions
- Physics‑motivated augmentation (not arbitrary).  
- Amplitude + Phase decomposition instead of raw complex matrices.  
- Compact latent representation that implicitly captures multipath.  
- Real‑time inference potential (single forward pass).

---

## Roadmap (Next Steps)
- Patch extraction from 15 GB combined dataset → 50,000+ samples.  
- Evaluate **SNR gain**, **variance reduction**, and **energy preservation**.  
- Architecture refinements: deeper encoder, skip connections, attention modules.  
- Alternative loss functions: SSIM, perceptual losses.  
- Benchmark against classical baselines (Wiener, Kalman filters).

---

## Status
- ✔️ Preprocessing pipeline complete  
- ✔️ Feature extraction implemented  
- ✔️ Augmentation applied (51 → 594 samples)  
- ✔️ Convolutional Autoencoder trained (MSE ~0.17)  
- ⏳ Dataset scaling & advanced metrics pending  
- ⏳ Architecture refinements planned  

---

## License
This project is for academic and research purposes.  
Feel free to fork and experiment, but cite appropriately if used in publications.

---
