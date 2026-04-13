"""
Resonance - Telecom Metrics Engine
Translates AI denoising performance into 5G/6G physical layer metrics.
"""

import tensorflow as tf
import numpy as np
import os
from scipy.special import erfc

from data_pipeline import build_resonance_dataset
from model import build_resonance_model

# ==========================================
# 1. INITIALIZATION
# ==========================================
WEIGHTS_PATH = os.path.join('weights', 'resonance_best_weights.h5')

print("Initializing Telecom Metrics Engine...")

dataset, input_shape = build_resonance_dataset(batch_size=16)
model = build_resonance_model(input_shape=input_shape)

if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print("Successfully loaded trained weights.")
else:
    raise FileNotFoundError(f"Could not find weights at {WEIGHTS_PATH}.")

# ==========================================
# 2. TELECOM MATH UTILITIES
# ==========================================
def cartesian_to_complex(tensor):
    """Converts the (..., 2) I/Q grid back into standard complex numbers."""
    # Tensor shape is expected to be (Batch, Antennas, Subcarriers, 2)
    real_part = tensor[..., 0]
    imag_part = tensor[..., 1]
    return real_part + 1j * imag_part

def q_function(x):
    """Standard Q-function used in digital communications."""
    return 0.5 * erfc(x / np.sqrt(2.0))

def calculate_theoretical_ber_64qam(sinr_linear):
    """
    Calculates the theoretical Bit Error Rate (BER) for a 64-QAM signal.
    Formula based on standard M-QAM modulation over AWGN.
    """
    # For 64-QAM, M = 64
    sinr_db = 10 * np.log10(sinr_linear + 1e-10)
    
    # Cap SINR at a reasonable level to prevent math domain errors in erfc
    sinr_linear_capped = np.clip(sinr_linear, a_min=1e-5, a_max=10000)
    
    # Standard 64-QAM BER approximation
    argument = np.sqrt((3.0 * sinr_linear_capped) / 63.0)
    ber = (4.0 / 6.0) * q_function(argument)
    return np.mean(ber)

# ==========================================
# 3. PERFORMANCE EVALUATION
# ==========================================
print("Running full batch inference for metric translation...")

for x_batch, y_batch in dataset.take(1):
    noisy_input = x_batch.numpy()
    ground_truth = y_batch.numpy()
    break

predictions = model.predict(noisy_input)

# Convert all data back to the complex domain for RF math
H_true = cartesian_to_complex(ground_truth)
H_noisy = cartesian_to_complex(noisy_input)
H_pred = cartesian_to_complex(predictions)

# ------------------------------------------
# Metric A: Effective SINR (Signal-to-Interference-plus-Noise Ratio)
# ------------------------------------------
# Signal Power
signal_power = np.mean(np.abs(H_true)**2, axis=(1, 2))

# Noise Power (Before AI)
raw_noise_power = np.mean(np.abs(H_true - H_noisy)**2, axis=(1, 2))
raw_sinr = signal_power / (raw_noise_power + 1e-10)

# Residual Error Power (After AI) -> The AI's mistake acts as the new noise floor
residual_error_power = np.mean(np.abs(H_true - H_pred)**2, axis=(1, 2))
ai_sinr = signal_power / (residual_error_power + 1e-10)

# Convert to dB
raw_sinr_db = np.mean(10 * np.log10(raw_sinr))
ai_sinr_db = np.mean(10 * np.log10(ai_sinr))
sinr_gain = ai_sinr_db - raw_sinr_db

# ------------------------------------------
# Metric B: Spectral Efficiency (Shannon Capacity)
# ------------------------------------------
# Capacity = log2(1 + SINR) measured in bits/second/Hertz
raw_capacity = np.mean(np.log2(1 + raw_sinr))
ai_capacity = np.mean(np.log2(1 + ai_sinr))
capacity_gain_percent = ((ai_capacity - raw_capacity) / raw_capacity) * 100

# ------------------------------------------
# Metric C: 64-QAM Bit Error Rate (BER)
# ------------------------------------------
raw_ber = calculate_theoretical_ber_64qam(raw_sinr)
ai_ber = calculate_theoretical_ber_64qam(ai_sinr)

# ==========================================
# 4. EXECUTIVE SUMMARY PRINTOUT
# ==========================================
print("\n" + "="*50)
print("   RESONANCE: TELECOM PERFORMANCE REPORT")
print("="*50)
print(f"Network Environment: 128x256 Massive MIMO Array")
print(f"Modulation Scheme:   64-QAM")
print("-" * 50)

print("1. SIGNAL QUALITY (Effective SINR)")
print(f"   Raw Corrupted Signal: {raw_sinr_db:>8.2f} dB")
print(f"   AI Denoised Signal:   {ai_sinr_db:>8.2f} dB")
print(f"   Net System Gain:      {sinr_gain:>8.2f} dB")

print("\n2. SPEED (Spectral Efficiency)")
print(f"   Raw Capacity:         {raw_capacity:>8.2f} bps/Hz")
print(f"   AI Enhanced Capacity: {ai_capacity:>8.2f} bps/Hz")
print(f"   Throughput Increase:  {capacity_gain_percent:>8.1f} %")

print("\n3. RELIABILITY (Bit Error Rate)")
print(f"   Raw BER:              {raw_ber:>8.2e}")
print(f"   AI Enhanced BER:      {ai_ber:>8.2e}")
print("="*50)