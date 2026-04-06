"""
Residual Convolutional Autoencoder for Massive MIMO Channel Denoising

Improvements over baseline:
- Residual blocks with skip connections (preserves multipath detail)
- BatchNormalization (stable training on noisy channel data)
- Dropout (prevents overfitting on synthetic DeepMIMO data)
- Deeper bottleneck: 64 -> 128 -> 256 filters
- Linear output activation (I/Q values are NOT bounded to [0,1])
- NMSE loss (standard metric for channel estimation tasks)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# GPU Check
# ─────────────────────────────────────────────

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU detected:", gpus)
    # Allow memory growth to avoid OOM on large channel matrices
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️  No GPU detected, training on CPU.")

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────

X_train = np.load('X_train.npy')
X_val   = np.load('X_val.npy')

input_shape = X_train.shape[1:]  # (64, 64, 2)
print(f"Input shape: {input_shape}")
print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")

# ─────────────────────────────────────────────
# Building Block: Residual Block
# ─────────────────────────────────────────────

def residual_block(x, filters, dropout_rate=0.1):
    """
    Residual block: two Conv2D layers with a skip connection.

    WHY RESIDUAL:
    In deep networks, gradients vanish and fine spatial details get lost.
    Skip connections let the network learn residual corrections on top
    of the identity — critical for preserving multipath channel structure.
    """
    shortcut = x

    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Match shortcut dimensions if filter count changed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    return x

# ─────────────────────────────────────────────
# Model Architecture: Residual CAE
# ─────────────────────────────────────────────
#
#  Input (64,64,2)
#    │
#  ResBlock(64) → MaxPool → (32,32,64)
#    │
#  ResBlock(128) → MaxPool → (16,16,128)
#    │
#  ResBlock(256) [Bottleneck] → (16,16,256)
#    │
#  ResBlock(128) → UpSample → (32,32,128)
#    │
#  ResBlock(64) → UpSample → (64,64,64)
#    │
#  Conv2D(2, linear) → Output (64,64,2)
#

inp = layers.Input(shape=input_shape)

# ── Encoder ──
x = residual_block(inp, 64)
x = layers.MaxPooling2D((2, 2), padding='same')(x)      # (32, 32, 64)

x = residual_block(x, 128)
x = layers.MaxPooling2D((2, 2), padding='same')(x)      # (16, 16, 128)

# ── Bottleneck ──
x = residual_block(x, 256)                              # (16, 16, 256)

# ── Decoder ──
x = residual_block(x, 128)
x = layers.UpSampling2D((2, 2))(x)                     # (32, 32, 128)

x = residual_block(x, 64)
x = layers.UpSampling2D((2, 2))(x)                     # (64, 64, 64)

# Output: linear activation — I/Q values are real-valued, not bounded
decoded = layers.Conv2D(2, (3, 3), activation='linear', padding='same')(x)

autoencoder = models.Model(inp, decoded, name="Residual_CAE")
autoencoder.summary()

# ─────────────────────────────────────────────
# Loss: Normalized MSE (NMSE)
# ─────────────────────────────────────────────
#
# WHY NMSE instead of plain MSE:
# MSE treats all samples equally regardless of signal power.
# NMSE normalizes error relative to signal power — which is the
# standard metric in wireless channel estimation literature.
# Lower NMSE = better reconstruction relative to channel strength.

def nmse_loss(y_true, y_pred):
    error        = y_true - y_pred
    nmse_per_sample = (
        tf.reduce_sum(tf.square(error),  axis=[1, 2, 3]) /
        tf.reduce_sum(tf.square(y_true), axis=[1, 2, 3])
    )
    return tf.reduce_mean(nmse_per_sample)

# ─────────────────────────────────────────────
# Compile
# ─────────────────────────────────────────────

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=nmse_loss,
    metrics=[tf.keras.metrics.MeanSquaredError(name='mse')]  # tracked alongside NMSE
)

# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

callbacks = [
    # Reduce LR if val_loss plateaus for 5 epochs
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ),
    # Stop early if no improvement after 10 epochs
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    ),
    # Save best checkpoint
    tf.keras.callbacks.ModelCheckpoint(
        'best_residual_cae.keras', monitor='val_loss', save_best_only=True, verbose=1
    )
]

# ─────────────────────────────────────────────
# Training Loop with tqdm
# ─────────────────────────────────────────────

def train_with_tqdm(model, X_train, X_val, epochs=50, batch_size=32):
    history = {"loss": [], "val_loss": [], "mse": [], "val_mse": []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        progbar = tqdm(total=1, desc="Training", unit="epoch")

        hist = model.fit(
            X_train, X_train,
            epochs=1,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=0
        )

        progbar.update(1)
        progbar.close()

        history["loss"].append(hist.history["loss"][0])
        history["val_loss"].append(hist.history["val_loss"][0])
        history["mse"].append(hist.history["mse"][0])
        history["val_mse"].append(hist.history["val_mse"][0])

        print(
            f"NMSE: {history['loss'][-1]:.6f} | "
            f"Val NMSE: {history['val_loss'][-1]:.6f} | "
            f"MSE: {history['mse'][-1]:.6f} | "
            f"Val MSE: {history['val_mse'][-1]:.6f}"
        )

        # Early stopping already handled by callback, but check here too
        if hasattr(model, 'stop_training') and model.stop_training:
            print("Early stopping triggered.")
            break

    return history


history = train_with_tqdm(autoencoder, X_train, X_val, epochs=50, batch_size=32)

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

X_test = np.load('X_test.npy')
denoised = autoencoder.predict(X_test)

# NMSE on test set
error = X_test - denoised
nmse_test = np.mean(
    np.sum(error**2, axis=(1,2,3)) /
    np.sum(X_test**2, axis=(1,2,3))
)
print(f"\n✅ Test NMSE: {nmse_test:.6f}")
print(f"✅ Test NMSE (dB): {10 * np.log10(nmse_test):.2f} dB")

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

idx = 0
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Residual CAE — Channel Reconstruction", fontsize=14)

axes[0, 0].imshow(X_test[idx, :, :, 0], cmap='viridis')
axes[0, 0].set_title("Original — I (Real)")

axes[0, 1].imshow(denoised[idx, :, :, 0], cmap='viridis')
axes[0, 1].set_title("Denoised — I (Real)")

axes[1, 0].imshow(X_test[idx, :, :, 1], cmap='plasma')
axes[1, 0].set_title("Original — Q (Imag)")

axes[1, 1].imshow(denoised[idx, :, :, 1], cmap='plasma')
axes[1, 1].set_title("Denoised — Q (Imag)")

plt.tight_layout()
plt.savefig("reconstruction_comparison.png", dpi=150)
plt.show()

# Loss curves
plt.figure(figsize=(8, 4))
plt.plot(history["loss"],     label="Train NMSE")
plt.plot(history["val_loss"], label="Val NMSE")
plt.xlabel("Epoch")
plt.ylabel("NMSE")
plt.title("Training Curves")
plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()