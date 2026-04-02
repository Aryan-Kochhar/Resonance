import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU detected:", gpus)
else:
    print("⚠️ No GPU detected, training will run on CPU.")

# Input shape matches your preprocessed features: (64, 64, 2)
input_shape = (64, 64, 2)
features = np.load("channels_o1_60_features.npy")

print("Features shape:", features.shape)

X_train, X_val = train_test_split(features, test_size=0.2, random_state=42)

# Encoder
inp = layers.Input(shape=input_shape)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2,2), padding='same')(x)

# Decoder
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(2, (3,3), activation='sigmoid', padding='same')(x)

# Build autoencoder
autoencoder = models.Model(inp, decoded)

# Compile with MSE loss (good for denoising)
autoencoder.compile(optimizer='adam', loss='mse')

# ✅ Simplified training loop: use model.fit directly with tqdm per epoch
def train_with_tqdm(model, X_train, X_val, epochs=20, batch_size=16):
    history = {"loss": [], "val_loss": []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        progbar = tqdm(total=1, desc="Training", unit="epoch")  # show per-epoch progress

        hist = model.fit(
            X_train, X_train,
            epochs=1,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            verbose=0
        )

        progbar.update(1)
        progbar.close()

        history["loss"].append(hist.history["loss"][0])
        history["val_loss"].append(hist.history["val_loss"][0])

        print(f"Loss: {history['loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f}")

    return history

# Train autoencoder with tqdm visualization
history = train_with_tqdm(autoencoder, X_train, X_val, epochs=20, batch_size=16)

# Reconstruct denoised channels
denoised = autoencoder.predict(features)

print("Original shape:", features.shape)
print("Denoised shape:", denoised.shape)

mse = np.mean((features - denoised)**2)
print("Reconstruction MSE:", mse)

idx = 0
plt.subplot(1,2,1)
plt.imshow(features[idx,:,:,0], cmap='viridis')
plt.title("Original amplitude")

plt.subplot(1,2,2)
plt.imshow(denoised[idx,:,:,0], cmap='viridis')
plt.title("Denoised amplitude")
plt.show()
