import tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np
import h5py
import time
import os

# ==========================================
# 1. HARDWARE OPTIMIZATION
# ==========================================
print("Initializing Hardware & Mixed Precision...")

# Enable GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" GPU Ready: {gpus[0].name}")
    except RuntimeError as e:
        print(e)

# Engage Tensor Cores for 2x speedup
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed Precision Enabled: {policy.compute_dtype}")

# ==========================================
# 2. THE MULTI-SCENARIO HDF5 STREAMER
# ==========================================
H5_FILE_PATH = 'resonance_massive_mimo_data.h5'

def dynamic_noise_generator(batch_size=64, min_noise_std=0.05, max_noise_std=0.2):
    def generator():
        with h5py.File(H5_FILE_PATH, 'r') as hf:
            scenarios = list(hf.keys())
            
            # Shuffle the order of environments (e.g., Indoor vs Outdoor) every epoch
            np.random.shuffle(scenarios)
            
            for scenario in scenarios:
                dset = hf[scenario]
                num_samples = dset.shape[0]
                
                # Yield data in RAM-safe contiguous chunks
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    
                    # Y is the clean Ground Truth (Shape: Batch, 128, 256, 2)
                    y_clean = dset[start_idx:end_idx]
                    
                    # Randomize the noise level for this specific batch
                    # This simulates a user moving further away from the cell tower
                    current_noise_std = np.random.uniform(min_noise_std, max_noise_std)
                    
                    # Generate the physical noise and add it to the clean signal
                    noise = np.random.normal(loc=0.0, scale=current_noise_std, size=y_clean.shape)
                    x_noisy = y_clean + noise
                    
                    yield (x_noisy, y_clean)
                    
    return generator

# ==========================================
# 3. HIGH-PERFORMANCE TF.DATASET
# ==========================================
def build_resonance_dataset(batch_size=64):
    """Wraps the Python generator in a C++ optimized TensorFlow Dataset."""
    
    # Peek at the file to get exact hardware dimensions dynamically
    with h5py.File(H5_FILE_PATH, 'r') as hf:
        first_scenario = list(hf.keys())[0]
        sample_shape = hf[first_scenario].shape[1:] # e.g., (128, 256, 2)
    
    print(f"Hardware Grid Detected: {sample_shape}")
    
    dataset = tf.data.Dataset.from_generator(
        dynamic_noise_generator(batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, *sample_shape), dtype=tf.float32), # X Noisy
            tf.TensorSpec(shape=(None, *sample_shape), dtype=tf.float32)  # Y Clean
        )
    )
    
    # Let the CPU pre-fetch the next batch while the GPU trains the current one
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, sample_shape

# ==========================================
# 4. PIPELINE TEST
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 64
    
    # Initialize the pipeline
    train_dataset, input_shape = build_resonance_dataset(batch_size=BATCH_SIZE)
    
    print("\n Commencing Pipeline Test...")
    start_time = time.time()
    
    # Pull 3 batches to verify the pre-fetching and noise variance
    for i, (x_batch, y_batch) in enumerate(train_dataset.take(3)):
        print(f"Batch {i+1} loaded. X shape: {x_batch.shape} | Y shape: {y_batch.shape}")
        
    print(f" Pipeline test completed in {time.time() - start_time:.3f} seconds.")