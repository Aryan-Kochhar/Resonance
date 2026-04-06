"""
Preprocess and augment a combined channel dataset:
- Pads/truncates each channel to target_shape
- L2 normalizes per sample
- Extracts I/Q Cartesian components (NOT amplitude/phase)
- Augments samples with synthetic noise and transformations
"""

from sklearn.model_selection import train_test_split
import numpy as np

# ─────────────────────────────────────────────
# STEP 1: Preprocessing
# ─────────────────────────────────────────────

def preprocess_channels(channels, target_shape=(64, 64)):
    """
    Preprocess heterogeneous channel matrices:
    - Ensures 2D complex shape
    - Pads/truncates to target_shape
    - Applies per-sample L2 normalization on the complex matrix
    """
    processed = []
    for ch in channels:
        arr = np.array(ch)

        # Force into 2D
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.squeeze()
            if arr.ndim > 2:
                arr = arr[..., 0]

        # Truncate if larger than target
        arr = arr[:target_shape[0], :target_shape[1]]

        # Pad if smaller than target
        pad_rows = max(0, target_shape[0] - arr.shape[0])
        pad_cols = max(0, target_shape[1] - arr.shape[1])
        arr = np.pad(arr, ((0, pad_rows), (0, pad_cols)), mode='constant')

        # Per-sample L2 normalization on the complex matrix
        # Done here before I/Q split to preserve the phase relationship
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm

        processed.append(arr)

    return np.array(processed)  # shape: (N, 64, 64), dtype: complex


# ─────────────────────────────────────────────
# STEP 2: Load & preprocess
# ─────────────────────────────────────────────

combined = np.load('channels_o1_60.npy', allow_pickle=True)

# Unwrap 0-D object array if needed
if combined.ndim == 0:
    combined = combined.item()

print("Type after unwrapping:", type(combined))

preprocessed = preprocess_channels(combined, target_shape=(64, 64))
print("Preprocessed shape:", preprocessed.shape)

np.save('channels_o1_60_preprocessed.npy', preprocessed)
print("Saved preprocessed channels.")

# ─────────────────────────────────────────────
# STEP 3: I/Q Cartesian Feature Extraction
# ─────────────────────────────────────────────
# WHY I/Q instead of amplitude/phase:
# Phase values wrap at ±π, causing discontinuities (2π jumps).
# Neural networks struggle to learn across these discontinuities.
# Cartesian I/Q is smooth and continuous — no wrap artifacts.

print("=" * 70)
print("Feature Extraction: Cartesian I/Q Decomposition")
print("=" * 70)

I = np.real(preprocessed).astype("float32")  # Real component
Q = np.imag(preprocessed).astype("float32")  # Imaginary component

print("I (Real) shape:", I.shape)
print("Q (Imag) shape:", Q.shape)

# Stack into (N, 64, 64, 2) — channel-last format for Conv2D
features = np.stack([I, Q], axis=-1)
print("Feature tensor shape:", features.shape)

np.save('channels_o1_60_features.npy', features)
print("Saved features.")

# ─────────────────────────────────────────────
# STEP 4: Summary statistics (optional logging)
# ─────────────────────────────────────────────

magnitude = np.sqrt(I**2 + Q**2)  # Computed from I/Q, not np.abs of complex
stats = {
    "mean":     np.mean(magnitude, axis=(1, 2)),
    "variance": np.var(magnitude, axis=(1, 2)),
    "energy":   np.sum(magnitude**2, axis=(1, 2))
}
print(f"Mean energy across dataset: {stats['energy'].mean():.4f}")

# ─────────────────────────────────────────────
# STEP 5: Augmentation
# ─────────────────────────────────────────────

def augment_channels(features, num_augments=5, noise_level=0.05):
    """
    Augment channel features via:
    - Additive Gaussian noise (simulates real-world SNR variation)
    - Random amplitude scaling (simulates path loss variation)
    - Random axis flip (spatial symmetry augmentation)

    NOTE: Noise is applied to I/Q directly — this is physically meaningful
    since AWGN in wireless systems adds to both real and imaginary components.
    """
    augmented = []
    for sample in features:
        augmented.append(sample)  # keep original
        for _ in range(num_augments):
            # Additive white Gaussian noise on I/Q
            noisy = sample + np.random.normal(0, noise_level, sample.shape).astype("float32")
            # Random amplitude scaling (±10%)
            scaled = noisy * np.random.uniform(0.9, 1.1)
            # Random spatial flip along antenna or subcarrier axis
            flipped = np.flip(scaled, axis=np.random.choice([0, 1]))
            augmented.append(flipped.astype("float32"))

    return np.array(augmented)


augmented_features = augment_channels(features, num_augments=10, noise_level=0.05)
print(f"\nOriginal dataset size : {features.shape[0]}")
print(f"Augmented dataset size: {augmented_features.shape[0]}")

# ─────────────────────────────────────────────
# STEP 6: Train / Val / Test split
# ─────────────────────────────────────────────

X_train, X_temp = train_test_split(augmented_features, test_size=0.3, random_state=42)
X_val, X_test   = train_test_split(X_temp, test_size=0.5, random_state=42)

print(f"\nSplit sizes — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

np.save('X_train.npy', X_train)
np.save('X_val.npy',   X_val)
np.save('X_test.npy',  X_test)
print("Saved train/val/test splits.")