"""
Resonance - Evaluation & Visualization Engine
Loads trained weights, performs inference, and generates visual heatmaps.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from data_pipeline import build_resonance_dataset
from model import build_resonance_model

# ==========================================
# 1. INITIALIZATION & WEIGHT LOADING
# ==========================================
WEIGHTS_PATH = os.path.join('weights', 'resonance_best_weights.h5')

print("Initializing Evaluation Engine...")

# Load the data pipeline to grab a test batch
dataset, input_shape = build_resonance_dataset(batch_size=8)
print(f"Dataset loaded. Expected input shape: {input_shape}")

# Build the model architecture
model = build_resonance_model(input_shape=input_shape)

# Load the trained weights
if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print("Successfully loaded trained weights.")
else:
    raise FileNotFoundError(f"Could not find weights at {WEIGHTS_PATH}. Train the model first.")

# ==========================================
# 2. INFERENCE
# ==========================================
print("Running inference on a test batch...")

# Grab a single batch of data (Noisy X, Clean Y)
for x_batch, y_batch in dataset.take(1):
    noisy_input = x_batch.numpy()
    ground_truth = y_batch.numpy()
    break

# Generate the AI predictions
predictions = model.predict(noisy_input)

# Calculate standard NMSE for the batch to print out
mse = np.mean(np.square(ground_truth - predictions))
signal_energy = np.mean(np.square(ground_truth))
batch_nmse = mse / (signal_energy + 1e-8)
# Convert to decibels (standard telecom metric format)
batch_nmse_db = 10 * np.log10(batch_nmse)

print(f"Batch Inference Complete. Raw NMSE: {batch_nmse_db:.2f} dB")

# ==========================================
# 3. VISUALIZATION (HEATMAPS)
# ==========================================
def plot_channel_comparisons(noisy, truth, pred, sample_idx=0, channel=0):
    """
    Generates a 1x3 Matplotlib figure comparing the matrices.
    channel=0 corresponds to Real (I), channel=1 corresponds to Imaginary (Q).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Extract the specific 2D grids (128 Antennas x 256 Subcarriers)
    grid_noisy = noisy[sample_idx, :, :, channel]
    grid_truth = truth[sample_idx, :, :, channel]
    grid_pred = pred[sample_idx, :, :, channel]
    
    # Determine color limits to keep scales consistent across the three plots
    vmin = np.min(grid_truth)
    vmax = np.max(grid_truth)
    
    # Plot 1: Corrupted Input
    im0 = axes[0].imshow(grid_noisy, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Corrupted Input (Noisy)', fontsize=14)
    axes[0].set_xlabel('Subcarriers')
    axes[0].set_ylabel('Antennas')
    fig.colorbar(im0, ax=axes[0])
    
    # Plot 2: Ground Truth
    im1 = axes[1].imshow(grid_truth, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Ground Truth (Clean)', fontsize=14)
    axes[1].set_xlabel('Subcarriers')
    fig.colorbar(im1, ax=axes[1])
    
    # Plot 3: AI Prediction
    im2 = axes[2].imshow(grid_pred, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title('Resonance Output (Denoised)', fontsize=14)
    axes[2].set_xlabel('Subcarriers')
    fig.colorbar(im2, ax=axes[2])
    
    channel_name = "Real (I)" if channel == 0 else "Imaginary (Q)"
    plt.suptitle(f"Massive MIMO Channel State Information - {channel_name} Component", fontsize=16)
    plt.tight_layout()
    
    # Save the figure to disk
    save_dir = 'visualizations'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'inference_sample_{sample_idx}_ch{channel}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Plot the first sample in the batch, looking at the Real (I) channel
    plot_channel_comparisons(noisy_input, ground_truth, predictions, sample_idx=0, channel=0)
    
    # Plot the same sample, looking at the Imaginary (Q) channel
    plot_channel_comparisons(noisy_input, ground_truth, predictions, sample_idx=0, channel=1)