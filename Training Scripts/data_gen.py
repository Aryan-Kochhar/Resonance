import DeepMIMO
import numpy as np
import h5py
import os

# ==========================================
# 1. CONFIGURATION & HARDWARE SPECS
# ==========================================
# Update this to the folder containing your downloaded DeepMIMO scenarios
DEEPMIMO_DATASET_FOLDER = './DeepMIMO_Dataset'
OUTPUT_H5_FILE = 'resonance_massive_mimo_data.h5'

# Define the hardware grid (The 2D Physical Reality)
NUM_ANTENNAS = 128
NUM_SUBCARRIERS = 256
BATCH_CHUNK_SIZE = 500  # How many users to process before writing to disk

# ==========================================
# 2. DEEPMIMO SCENARIO SETUP
# ==========================================
def setup_deepmimo_params(scenario_name):
    """Configures the DeepMIMO environment to match our hardware specs."""
    parameters = DeepMIMO.default_parameters()
    
    parameters['dataset_folder'] = DEEPMIMO_DATASET_FOLDER
    parameters['scenario'] = scenario_name
    
    parameters['active_BS'] = np.array([1])
    
    parameters['bs_antenna']['shape'] = np.array([8, 16, 1]) 
    
    parameters['OFDM']['subcarriers'] = NUM_SUBCARRIERS
    parameters['OFDM']['bandwidth'] = 100e6  # 100 MHz
    
    parameters['user_row_first'] = 1
    parameters['user_row_last'] = 1000 
    return parameters

# ==========================================
# 3. HDF5 STREAMING ENGINE
# ==========================================
def generate_and_save_scenario(scenario_name, h5_file):
    """Generates the data and writes it to disk in RAM-safe chunks."""
    print(f"\nInitializing Scenario: {scenario_name}")
    
    # Generate the raw complex data from DeepMIMO
    parameters = setup_deepmimo_params(scenario_name)
    dataset = DeepMIMO.generate_data(parameters)
    
    # The channel matrix is located in dataset[0]['user']['channel']
    # Shape is typically: (Users, Antennas, Subcarriers)
    raw_channels = dataset[0]['user']['channel']
    total_users = raw_channels.shape[0]
    
    print(f"Total simulated users found: {total_users}")
    print(f"Applying Cartesian Split and HDF5 Chunking...")
    
    # Create a dataset inside the HDF5 file specifically for this scenario
    # Shape: (Total Users, 128 Antennas, 256 Subcarriers, 2 Channels for I/Q)
    target_shape = (total_users, NUM_ANTENNAS, NUM_SUBCARRIERS, 2)
    
    h5_dataset = h5_file.create_dataset(
        name=scenario_name,
        shape=target_shape,
        dtype=np.float32,
        chunks=(64, NUM_ANTENNAS, NUM_SUBCARRIERS, 2) # Optimize chunk reading for batch size 64
    )
    
    # Process and write in RAM-safe chunks
    for start_idx in range(0, total_users, BATCH_CHUNK_SIZE):
        end_idx = min(start_idx + BATCH_CHUNK_SIZE, total_users)
        
        # Extract chunk and squeeze out unnecessary single dimensions
        complex_chunk = np.squeeze(raw_channels[start_idx:end_idx])
        
        # Mathematical Innovation: Early Cartesian Decomposition
        # Split complex numbers (a + bj) into a 2-channel array [Real, Imaginary]
        real_part = np.real(complex_chunk)
        imag_part = np.imag(complex_chunk)
        
        # Stack them along the last axis to create the final (Batch, 128, 256, 2) shape
        cartesian_chunk = np.stack((real_part, imag_part), axis=-1)
        
        # Stream directly to the hard drive
        h5_dataset[start_idx:end_idx] = cartesian_chunk
        
        print(f"  -> Written users {start_idx} to {end_idx-1} to disk.")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    scenarios_to_run = ['O1_60', 'O1_28b', 'I2_60', 'I2_28b'] 
    
    # Open the HDF5 file in write mode
    with h5py.File(OUTPUT_H5_FILE, 'w') as h5f:
        for scenario in scenarios_to_run:
            try:
                generate_and_save_scenario(scenario, h5f)
            except Exception as e:
                print(f"Failed to process scenario {scenario}: {e}")
                
    print("\nData Generation Complete.")
    print(f"File saved successfully as: {OUTPUT_H5_FILE}")