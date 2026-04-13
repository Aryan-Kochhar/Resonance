"""
Resonance - Model Architecture & Physics Engine
Defines the ConvNeXt U-Net and the Physics-Informed Loss Function.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. PHYSICS-INFORMED LOSS FUNCTION
# ==========================================
def pinn_loss(lambda_phys=0.15):
    """
    Combines Normalized Mean Square Error (NMSE) with Spatial Correlation.
    Forces the network to respect the physical relationship between adjacent antennas.
    
    Args:
        lambda_phys (float): The weight of the physics penalty.
    """
    def loss(y_true, y_pred):
        # 1. NMSE (Accuracy metric)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        signal_energy = tf.reduce_mean(tf.square(y_true))
        nmse = mse / (signal_energy + 1e-8)
        
        # 2. Adjacent Antenna Spatial Correlation Penalty (Physics metric)
        # Axis 1 corresponds to the 128 Antennas in the (Batch, Antennas, Subcarriers, Channels) grid
        true_adjacent_diff = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        pred_adjacent_diff = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        
        physics_penalty = tf.reduce_mean(tf.square(true_adjacent_diff - pred_adjacent_diff))
        
        return nmse + (lambda_phys * physics_penalty)
        
    return loss

# ==========================================
# 2. CONVNEXT ARCHITECTURE BLOCKS
# ==========================================
def convnext_block(x, dim):
    """
    A modern ConvNeXt Block.
    Processes 2D data using large receptive fields while maintaining low VRAM usage.
    """
    shortcut = x
    
    # 1. Depthwise Convolution (7x7 kernel mimics Transformer attention span)
    x = layers.DepthwiseConv2D(kernel_size=7, padding='same', use_bias=False)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # 2. Pointwise Expansion (Inverted Bottleneck, expands feature space by 4x)
    x = layers.Conv2D(filters=dim * 4, kernel_size=1)(x)
    x = layers.Activation('gelu')(x)
    
    # 3. Pointwise Projection (Compresses feature space back to original dimension)
    x = layers.Conv2D(filters=dim, kernel_size=1)(x)
    
    # 4. Residual Skip Connection
    x = layers.Add()([shortcut, x])
    return x

# ==========================================
# 3. MAIN U-NET BUILDER
# ==========================================
def build_resonance_model(input_shape=(128, 256, 2)):
    """
    Constructs the full U-Net architecture using ConvNeXt blocks.
    
    Args:
        input_shape (tuple): The hardware grid shape (Antennas, Subcarriers, I/Q).
    Returns:
        tf.keras.Model: The uncompiled Resonance model.
    """
    inputs = layers.Input(shape=input_shape)
    
    # --- ENCODER (Feature Extraction) ---
    # Stem: Initial downsampling to reduce compute load
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    enc1 = convnext_block(x, 64)
    
    # Downsample Stage 1
    x = layers.Conv2D(128, kernel_size=2, strides=2, padding='same')(enc1)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    enc2 = convnext_block(x, 128)
    
    # Downsample Stage 2
    x = layers.Conv2D(256, kernel_size=2, strides=2, padding='same')(enc2)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    enc3 = convnext_block(x, 256)
    
    # --- BOTTLENECK (Deepest Features) ---
    x = layers.Conv2D(512, kernel_size=2, strides=2, padding='same')(enc3)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = convnext_block(x, 512)
    x = convnext_block(x, 512)
    
    # --- DECODER (Signal Reconstruction) ---
    # Upsample Stage 1
    x = layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Concatenate()([x, enc3])
    x = convnext_block(x, 256)
    
    # Upsample Stage 2
    x = layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Concatenate()([x, enc2])
    x = convnext_block(x, 128)
    
    # Upsample Stage 3
    x = layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Concatenate()([x, enc1])
    x = convnext_block(x, 64)
    
    # Final Upsample back to original resolution (128x256)
    x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same')(x)
    
    # Output Layer: Restores the 2 channels (Real/Imaginary I/Q data)
    # Using 'linear' activation because Cartesian I/Q values can be negative or positive
    outputs = layers.Conv2D(2, kernel_size=1, activation='linear', dtype='float32')(x)
    
    model = models.Model(inputs, outputs, name="Resonance_ConvNeXt_PINN")
    return model

if __name__ == "__main__":
    # Quick sanity check to ensure the graph compiles and shapes align
    test_model = build_resonance_model()
    test_model.summary()