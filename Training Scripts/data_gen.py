# import time
# import numpy as np
# from deepmimo.datasets import generate

# # for i, ch in enumerate(dataset['channels'][:5]):
# #     print(i, type(ch), getattr(ch, "shape", None))



# def timed_generation(scen_name, label):
#     print(f"Starting {label} dataset...")
#     start = time.time()
#     dataset = generate(scen_name)
#     end = time.time()
#     print(f"{label} dataset finished in {end-start:.2f} seconds")

#     # Try stacking if possible
#     try:
#         channels = np.stack(dataset['channels'], axis=0)
#         print(f"{label} channels shape: {channels.shape}")
#     except Exception:
#         print(f"{label} channels could not be stacked, saving as list")
#         channels = dataset['channels']  # keep as list

#     return channels


# # Indoor and outdoor datasets with defaults
# dataset_indoor = timed_generation('i2_28b', "Indoor (i2_28b)")
# dataset_outdoor = timed_generation('o1_60', "Outdoor (o1_60)")

# # Inspect the first few channel entries
# for i, ch in enumerate(dataset_outdoor['channels'][:10]):  # look at first 10
#     if isinstance(ch, np.ndarray):
#         print(f"Index {i}: ndarray with shape {ch.shape}")
#     else:
#         print(f"Index {i}: {type(ch)}")

# # Save them

# np.save('channels_i2_28b.npy', dataset_indoor)
# np.save('channels_o1_60.npy', dataset_outdoor)

# # Combine them
# channels_combined = np.concatenate([dataset_indoor, dataset_outdoor], axis=0)
# np.save('channels_combined.npy', channels_combined)
# print("Combined channels shape:", channels_combined.shape)


import time
import numpy as np
from deepmimo.datasets import generate

# Indoor dataset
print("Starting Indoor (i2_28b) dataset...")
start = time.time()
dataset_indoor = generate('i2_28b')
end = time.time()
print(f"Indoor dataset finished in {end-start:.2f} seconds")

channels_indoor = np.array(dataset_indoor['channels'])
print("Indoor channels shape:", channels_indoor.shape)

# Outdoor dataset
print("Starting Outdoor (o1_60) dataset...")
start = time.time()
dataset_outdoor = generate('o1_60')
end = time.time()
print(f"Outdoor dataset finished in {end-start:.2f} seconds")

channels_outdoor = np.array(dataset_outdoor['channels'], dtype=object)
print("Outdoor channels length:", len(channels_outdoor))

# Save individual datasets
np.save('channels_i2_28b.npy', channels_indoor)
np.save('channels_o1_60.npy', channels_outdoor, allow_pickle=True)

# Save combined as a dictionary (robust way)
channels_combined = {
    "indoor": channels_indoor,
    "outdoor": channels_outdoor
}
np.save('channels_combined.npy', channels_combined, allow_pickle=True)

print("Combined dataset saved as dictionary with keys 'indoor' and 'outdoor'")
