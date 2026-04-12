"""
Resonance - Master Execution Engine
Imports the data pipeline and model, configures hardware, and runs the training loop.
"""

import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import os

# Import your custom modules
from data_pipeline import build_resonance_dataset
from model import build_resonance_model, pinn_loss

# ==========================================
# 1. HARDWARE CONFIGURATION
# ==========================================
print("Initializing RTX Hardware Setup...")

# Enable Memory Growth to prevent VRAM allocation crashes
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Registered: {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU Setup Error: {e}")

# Enable Mixed Precision to utilize RTX Tensor Cores
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed Precision Active. Compute dtype: {policy.compute_dtype}")

# ==========================================
# 2. HYPERPARAMETERS & DIRECTORIES
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PHYSICS_WEIGHT = 0.15

# Create weights directory if it doesn't exist
WEIGHTS_DIR = 'weights'
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)
    
LOGS_DIR = 'logs'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# ==========================================
# 3. INITIALIZE PIPELINE & MODEL
# ==========================================
print("\nLoading Data Pipeline...")
# The shape returned will be (128, 256, 2) based on your hardware specs
train_dataset, input_shape = build_resonance_dataset(batch_size=BATCH_SIZE)
print(f"Pipeline Active. Expected Input Shape: {input_shape}")

print("\nConstructing ConvNeXt PINN Model...")
model = build_resonance_model(input_shape=input_shape)

# ==========================================
# 4. COMPILATION
# ==========================================
# AdamW is superior to standard Adam for Transformer/ConvNeXt architectures
optimizer = optimizers.AdamW(
    learning_rate=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)

model.compile(
    optimizer=optimizer,
    loss=pinn_loss(lambda_phys=PHYSICS_WEIGHT)
)

print("\nModel Compiled Successfully.")

# ==========================================
# 5. TRAINING CALLBACKS
# ==========================================
callbacks_list = [
    # Saves the model only when it achieves a new best loss score
    callbacks.ModelCheckpoint(
        filepath=os.path.join(WEIGHTS_DIR, 'resonance_best_weights.h5'),
        save_best_only=True,
        save_weights_only=True,
        monitor='loss',
        verbose=1
    ),
    # If the model stops learning for 3 epochs, cut the learning rate in half
    callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    # Stop training entirely if it flatlines for 8 epochs to save compute time
    callbacks.EarlyStopping(
        monitor='loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    # Allows you to visualize training curves using TensorBoard
    callbacks.TensorBoard(
        log_dir=LOGS_DIR,
        histogram_freq=1
    )
]

# ==========================================
# 6. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    print("\nCommencing Training Phase...")
    
    # We estimate steps per epoch based on 50,000 total users / Batch Size
    # Adjust this if your DeepMIMO generation script outputs a different number of users
    ESTIMATED_STEPS = 1500 
    
    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            steps_per_epoch=ESTIMATED_STEPS,
            callbacks=callbacks_list
        )
        print("\nTraining Complete! Best weights are securely saved in the 'weights' folder.")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Existing weights in 'weights/' remain intact.")