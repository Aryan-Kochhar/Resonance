# Resonance

Wireless Channel Denoising for Massive MIMO using Advanced Deep Learning

---

## Overview
**Resonance** is an advanced research project that demonstrates the feasibility of using state-of-the-art deep learning for denoising Massive MIMO wireless channel matrices.  
Accurate Channel State Information (CSI) is critical for 5G/6G systems, but raw measurements are corrupted by noise, hardware imperfections, and multipath interference.  
This project leverages **Model-Agnostic AI Architectures (e.g., State Space Models/Transformers)** to learn noise distributions across diverse frequency bands and reconstruct cleaner, true-to-spec channel estimates.

---

## Problem Statement
- **Noisy channel matrices** degrade wireless system performance.  
- Classical methods (Least Squares, Wiener, Kalman filters) assume Gaussian noise and fail at low SNR.  
- Traditional Convolutional Neural Networks (CNNs) scale quadratically and choke on massive antenna arrays.  
- Next-generation deep learning offers a **physics-aware, cross-band, real‑time** alternative.

---

## Dataset
- Source: **DeepMIMO** synthetic dataset (ray‑tracing based across multiple environments).  
- Scenarios: `O1_60`, `O1_28`, `I2_28b`, `I1_2p4` (Mixed mmWave and Wi-Fi frequencies).  
- Storage strategy (memory‑optimized):  
  - HDF5 lazy-loading pipeline  
  - Scenario-isolated datasets  
  - Locked to 3GPP Massive MIMO hardware specs (no arbitrary padding)  
- **Final dataset size**: 50,000+ unique samples.  
- Input shape: Up to `128 × 256 × 2` (Real + Imaginary maps).

---

## Preprocessing & Feature Extraction
- **Normalization**: Per‑sample unit norm (L₂ = 1).  
- **Data Pipeline**: `tf.data.Dataset` mapping for on-the-fly streaming.  
- **Feature decomposition**:  
  - Real component: \( I = \Re(H) \)  
  - Imaginary component: \( Q = \Im(H) \)  
  - *Eliminates previous 2π phase discontinuity errors.*
- Output tensor: Multi-dimensional Cartesian (I/Q) grids.

---

## Model Architecture
**Next-Generation Sequence Modeling (Replacing legacy CAE)**

- **Input Representation** - Processes pure Cartesian ($I/Q$) sequences.  
  - Handles massive spatial correlation across 128+ antennas natively.

- **Processing Core (e.g., Mamba / Residual U-Net)** - Linear complexity scaling for Massive MIMO dimensions.  
  - Preserves fine multipath details without aggressive bottleneck blurring.

- **Output** - Reconstructs denoised Real + Imaginary components.

- **Training Setup** - Precision: Mixed Precision (`float16`)  
  - Loss: Normalized Mean Square Error (NMSE)  
  - Batch size: 128/256 (Hardware-scaled)  
  - Hardware: RTX 5070 GPU + Ryzen 9000 AI Series CPU

---

## Results
- **Dataset Infrastructure**: Successfully streaming 15GB+ without memory limits.  
- **Training Efficiency**: Accelerated by parallel data loading and Tensor Core utilization.  
- **Observations**:  
  - Cartesian (I/Q) maps eliminate residual phase errors entirely.  
  - HDF5 chunking allows seamless transitions between indoor and outdoor scenario training.  
  - Framework is now fully independent of specific matrix dimensions.

---

## Contributions
- True-to-spec Massive MIMO scenario generation (not arbitrary square matrices).  
- Real + Imaginary ($I/Q$) Cartesian decomposition instead of error-prone polar coordinates.  
- Highly scalable HDF5 data infrastructure that eliminates RAM bottlenecks.  
- Foundation built for Zero-Shot cross-band generalization.

---

## Roadmap (Next Steps)
- Evaluate advanced architectures: State Space Models (Mamba) vs. Vision Transformers.  
- Implement **Physics-Informed Neural Networks (PINNs)** to strictly enforce spatial correlation laws.  
- Conduct **Zero-Shot Generalization** testing (Train on 60GHz outdoor, test on 2.4GHz indoor).  
- Evaluate **NMSE**, **Spectral Efficiency**, and **Inference Latency**.  
- Benchmark against classical baselines (LMMSE estimators).

---

## Status
- ✔️ Preprocessing pipeline rewritten for Cartesian $I/Q$  
- ✔️ Massive MIMO multi-scenario generation complete (50,000+ samples)  
- ✔️ HDF5 streaming storage implemented  
- ⏳ Advanced AI architecture integration in progress  
- ⏳ Zero-Shot testing & advanced metrics pending  
- ⏳ Physics-informed loss functions planned  

---

## License
This project is for academic and research purposes.  
Feel free to fork and experiment, but cite appropriately if used in publications.

---
