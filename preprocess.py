"""
Preprocess and augment a combined channel dataset:
- Pads/truncates each channel to target_shape
- Normalizes values
- Extracts amplitude and phase
- Augments samples with synthetic noise and transformations
"""

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

def preprocess_channels(channels, target_shape=(64, 64), normalize=True):
    """
    Preprocess heterogeneous channel matrices:
    - Ensures 2D shape
    - Pads/truncates to target_shape
    - Normalizes values
    """
    processed = []

    for ch in channels:
        arr = np.array(ch)

        # Force into 2D (if it's 1D, reshape; if >2D, take first slice)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.squeeze()
            if arr.ndim > 2:     # still >2D, take first 2D slice
                arr = arr[..., 0]

        # Truncate if larger
        arr = arr[:target_shape[0], :target_shape[1]]

        # Pad if smaller
        pad_rows = max(0, target_shape[0] - arr.shape[0])
        pad_cols = max(0, target_shape[1] - arr.shape[1])
        arr = np.pad(arr, ((0, pad_rows), (0, pad_cols)), mode='constant')

        # Normalize
        if normalize:
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm

        processed.append(arr)

    return np.array(processed)


# Load and preprocess combined dataset
combined = np.load('channels_o1_60.npy', allow_pickle=True)

# If it's a 0-D array containing an object, extract it
if combined.ndim == 0:
    combined = combined.item()

print("Type after unwrapping:", type(combined))
preprocessed_combined = preprocess_channels(combined, target_shape=(64, 64))

print("Preprocessed combined dataset shape:", preprocessed_combined.shape)
np.save('channels_o1_60_preprocessed.npy', preprocessed_combined)

print("="*90)
print("Feature Extraction part")
# print(tf.config.list_physical_devices('GPU'))
# print(tf.test.is_built_with_cuda())
# print(tf.sysconfig.get_build_info())
print("="*90)

# Extract amplitude and phase
amplitude = np.abs(preprocessed_combined).astype("float32")
phase = np.angle(preprocessed_combined).astype("float32")
print("Amplitude shape:", amplitude.shape)
print("Phase shape:", phase.shape)

features = np.stack([amplitude, phase], axis=-1)  # shape: (N, 64, 64, 2)
np.save('channels_o1_60_features.npy', features)

# Feature engineering (mean, variance, energy)
stats = {
    "mean": np.mean(amplitude, axis=(1,2)),
    "variance": np.var(amplitude, axis=(1,2)),
    "energy": np.sum(amplitude**2, axis=(1,2))
}


def augment_channels(features, num_augments=5, noise_level=0.05):
    """
    Augment channel features by adding synthetic noise and transformations.
    """
    augmented = []
    for sample in features:
        augmented.append(sample)  # keep original
        for _ in range(num_augments):
            noisy = sample + np.random.normal(0, noise_level, sample.shape)
            scaled = noisy * np.random.uniform(0.9, 1.1)
            flipped = np.flip(scaled, axis=np.random.choice([0,1]))
            augmented.append(flipped)
    return np.array(augmented)


# Augment dataset
augmented_features = augment_channels(features, num_augments=10, noise_level=0.05)
print("Original dataset size:", features.shape[0])
print("Augmented dataset size:", augmented_features.shape[0])

print("="*90)

# Flatten augmented dataset for ML models
X = augmented_features.reshape(augmented_features.shape[0], -1)

# Train/val/test split
X_train, X_temp = train_test_split(X, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

# Load model code
with open("model_heavy.py", encoding="utf-8") as f:
    code = f.read()
exec(code)
