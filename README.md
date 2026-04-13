# Resonance
### AI-Powered Massive MIMO Channel Estimation for 5G/6G

A deep learning pipeline that replaces classical channel estimators in 5G Massive MIMO systems using a physics-informed ConvNeXt U-Net with attention mechanisms.

---

## Results

| SNR | Corrupted Input | Resonance AI | Gain |
|-----|----------------|--------------|------|
| 0 dB | 0.0 dB NMSE | −23.9 dB NMSE | **+23.9 dB** |
| 10 dB | −10.0 dB NMSE | −24.0 dB NMSE | **+14.0 dB** |
| 20 dB | −20.0 dB NMSE | −24.0 dB NMSE | **+4.0 dB** |

**BER at SNR=10 dB:** 0.163 (corrupted) → 0.003 (AI) — **98.2% reduction**

---

## What This Is

Classical 5G base stations use Least Squares (LS) estimation to recover the Channel State Information (CSI) matrix from pilot signals. This works reasonably at high SNR but degrades badly in low-SNR conditions (urban mmWave, cell edges, high mobility).

Resonance trains a neural network to take a noise-corrupted channel observation and reconstruct the clean CSI matrix — learning the physical structure of the channel (antenna correlations, multipath delay patterns) from ray-tracing data.

**Scenario:** O1_28 — outdoor street environment at 28 GHz  
**Array:** 128-antenna Uniform Planar Array (16×8) × 256 OFDM subcarriers  
**Input/Output:** `(128, 256, 2)` I/Q tensor — real and imaginary components of the complex channel matrix

---

## Architecture

```
Input (128, 256, 2)
        │
    ConvNeXt Stem ──────────────────────────────────────────┐
        │                                                    │ enc1 (32ch)
    Downsample + ConvNeXt ──────────────────────────────────┤
        │                                                    │ enc2 (64ch)
    Downsample + ConvNeXt ──────────────────────────────────┤
        │                                                    │ enc3 (128ch)
    Downsample + ConvNeXt                                    │
        │                                                    │
    ── BOTTLENECK (256ch) ──                                 │
    ConvNeXt × 2 + CBAM Channel Attention                   │
        │                                                    │
    Upsample + Attention Gate ◄─────────────────────────────┘ (enc3)
        │
    Upsample + Attention Gate ◄─────────────────────────────── (enc2)
        │
    Upsample + Attention Gate ◄─────────────────────────────── (enc1)
        │
    Final Upsample (16ch)
        │
    Output Conv (128, 256, 2)
```

**Key components:**
- **ConvNeXt blocks** — 7×7 depthwise convolutions capture long-range antenna correlations (analogous to Vision Transformer attention span)
- **Attention Gates** — decoder selectively focuses on encoder features; suppresses noise-dominated regions of the channel
- **CBAM Channel Attention** — bottleneck learns which feature maps correspond to dominant propagation modes
- **Linear output activation** — I/Q values are unbounded real numbers, not sigmoid-bounded

---

## Loss Function

Four-component physics-informed loss:

```
L = NMSE                          # reconstruction accuracy
  + λ_phys × physics_penalty      # adjacent-antenna spatial correlation
  + λ_spec × spectral_loss        # FFT domain accuracy (impulse response)
  + λ_mag  × magnitude_loss       # channel amplitude accuracy
```

The physics penalty enforces the spatial correlation structure of Uniform Planar Arrays — adjacent antennas must have smoothly varying channels, which is a hard physical constraint derived from the array steering vector.

Default weights: `λ_phys=0.10`, `λ_spec=0.10`, `λ_mag=0.05`

---

## Project Structure

```
Resonance/
├── dgen_o1_28.py       # Step 1: Generate channels from raw .mat ray-tracing files
├── preprocess_2.py     # Step 2: Clean, normalize, split into train/val/test
├── model.py            # Model architecture + loss function + metrics
├── train.py            # Training loop with cosine annealing
├── eval.py             # Evaluation: heatmaps, NMSE/BER curves, summary poster
│
├── logs/
│   ├── training_log.csv
│   ├── train/          # TensorBoard train logs
│   └── validation/     # TensorBoard val logs
│
├── weights/
│   └── resonance_best.weights.h5
│
└── visualizations/
    ├── summary_poster.png       ← main result figure
    ├── heatmaps_sample0.png
    ├── heatmaps_sample1.png
    ├── nmse_vs_snr.png
    ├── ber_vs_snr.png
    ├── spectral_sample0.png
    └── training_curves.png
```

---

## Setup

```bash
pip install tensorflow numpy scipy matplotlib scikit-learn h5py
```

Tested on Python 3.10, TensorFlow 2.11, Windows with CUDA GPU.

---

## Usage

### Step 1 — Generate channel data from raw DeepMIMO .mat files
```bash
# Edit SCENARIO_FOLDER and OUTPUT_NPY paths in dgen_o1_28.py first
python dgen_o1_28.py
```
Reads `power`, `delay`, `phase`, `aod_az`, `aod_el` .mat files from the O1_28 DeepMIMO v4 scenario and constructs complex MIMO channel matrices via UPA steering vectors + OFDM phase shifts.

### Step 2 — Preprocess
```bash
# Edit INPUT_NPY and OUTPUT_DIR paths in preprocess_2.py first
python preprocess_2.py
```
Removes NaN/zero-energy samples, applies per-sample L2 normalization, splits into train/val/test.

### Step 3 — Train
```bash
# Edit SPLITS_DIR path in train.py first
python train.py
```
Trains for up to 60 epochs with cosine annealing LR schedule and early stopping. Best weights saved to `weights/resonance_best.weights.h5`.

### Step 4 — Evaluate
```bash
# Edit SPLITS_DIR and WEIGHTS_PATH in eval.py first
python eval.py
```
Runs full evaluation across SNR range −10 to +30 dB, generates all visualizations, prints performance table.

---

## Data

Uses the **DeepMIMO O1_28** ray-tracing scenario — an outdoor street environment at 28 GHz (mmWave 5G band) generated with Remcom Wireless InSite.

- **25,000 channel samples** (5 user rows × 5,000 users per row)
- **Train / Val / Test:** 12,453 / 1,099 / 1,099 samples
- Raw .mat files not included in this repo (13 GB). Download from [deepmimo.net](https://deepmimo.net)

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Batch size | 4 |
| Epochs | 60 (early stopping) |
| Optimizer | Adam |
| LR schedule | Cosine annealing with 200-step warmup |
| LR range | 1e-3 → 1e-6 |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 (global norm) |
| Mixed precision | Disabled |
| Augmentation | On-the-fly AWGN, random std 0.02–0.05 |

---

## References

- A. Alkhateeb, "DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications," ITA 2019
- Z. Liu et al., "A ConvNet for the 2020s" (ConvNeXt), CVPR 2022
- O. Oktay et al., "Attention U-Net," MIDL 2018
- S. Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018