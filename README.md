# Resonance

Physics-Informed ConvNeXt Architecture for 6G Massive MIMO Channel Denoising

---

## Overview
**Resonance** is a state-of-the-art deep learning architecture designed to denoise Massive MIMO Channel State Information (CSI) for 5G and 6G networks. 
While traditional AI models act as "black boxes" that can hallucinate physically impossible signals, Resonance utilizes a **Physics-Informed Neural Network (PINN)** loss function to guarantee that its high-speed denoised outputs strictly adhere to real-world radio frequency physics. 

---

## Problem Statement
- **Noisy channel matrices** severely degrade wireless system capacity and beamforming accuracy.
- Classical methods (Least Squares, Wiener filters) assume linear Gaussian noise and fail in complex, low-SNR multipath environments.
- Standard AI models (Vision Transformers, basic CNNs) treat RF data like square photographs, leading to spatial geometry erasure and the generation of mathematically impossible radio waves.
- Processing raw Phase data introduces complex $2\pi$ wrap-around discontinuities that break standard error calculations.

---

## The Resonance Solution & Data Pipeline
Resonance discards artificial zero-padding and processes data in its native 3GPP hardware geometry using a high-performance HDF5 streaming pipeline.

- **Source:** DeepMIMO synthetic ray-tracing dataset.
- **Hardware Grid:** Native `128 x 256` rectangular arrays (128 Antennas, 256 Subcarriers).
- **Data Representation:** Cartesian Decomposition. Signals are split into Real ($I$) and Imaginary ($Q$) grids `(128, 256, 2)`, completely eliminating phase wrap-around errors.
- **Generalization:** Zero-shot multi-scenario training. The pipeline streams continuous chunks of both indoor (e.g., I2_28b) and outdoor (e.g., O1_60) environments to prevent geographic overfitting.
- **Dynamic SNR:** Noise variance is dynamically injected per-batch during training to simulate real-time user mobility and varying distances from the cell tower.

---

## Model Architecture
**Physics-Informed ConvNeXt U-Net (2D PINN)**

- **ConvNeXt Backbone:** Replaces legacy convolutional layers with modern ConvNeXt blocks (large 7x7 depthwise convolutions, inverted bottlenecks, and LayerNorm). This provides the massive analytical capability of a Vision Transformer with the lightweight VRAM footprint and speed of a CNN.
- **U-Net Geometry:** Encoder-Decoder structure with skip connections to preserve high-resolution spatial details while compressing noise out of the latent space.
- **Physics-Informed Loss Function:** The model is constrained by a custom dual-objective loss function:
  1. **Accuracy:** Normalized Mean Square Error (NMSE) against the ground truth.
  2. **Physics (Spatial Correlation):** A penalty applied to adjacent antenna variance, forcing the network to respect the physical spacing and correlation of the Massive MIMO array.

---

## Telecom Evaluation Metrics
Instead of relying solely on computer science metrics (Loss/MSE), Resonance translates its performance directly into telecommunications business value:
- **Effective SINR Gain (dB):** Measuring the exact increase in Signal-to-Interference-plus-Noise Ratio post-inference.
- **Spectral Efficiency (bps/Hz):** Calculating the theoretical throughput increase using the Shannon Capacity formula.
- **Bit Error Rate (BER):** Evaluating the reliability of the reconstructed signal through a simulated 64-QAM modulation scheme.

---

## Contributions & Unique Differentiators
- **"Safe AI" Guarantee:** The PINN loss function acts as a mathematical guardrail, ensuring RF engineers can trust the model's physical viability.
- **Rectangular Native Processing:** No square-image padding. The model learns the exact dimensions of the 128-antenna array.
- **RAM-Safe HDF5 Streaming:** Custom `tf.data.Dataset` generators allow for training on 50,000+ massive user grids without exceeding local system memory or GPU limits.
- **Mixed Precision:** Built to utilize RTX Tensor Cores (`mixed_float16`) for hyper-accelerated training.

---

## Status
- [x] Cartesian I/Q data generation and HDF5 chunking complete.
- [x] Dynamic noise injection and hardware-optimized data pipeline established.
- [x] Physics-Informed (NMSE + Spatial Correlation) custom loss function integrated.
- [x] ConvNeXt U-Net architecture built and compiled.
- [x] Telecom metrics translation engine (SINR, Capacity, BER) operational.
- [ ] Large-scale (50k+ samples) multi-environment training execution.
- [ ] Edge-device quantization for real-time SDR deployment.

---

## License
This project is for academic and research purposes. 
Feel free to fork and experiment, but cite appropriately if used in publications.