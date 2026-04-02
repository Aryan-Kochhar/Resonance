import numpy as np

indoor = np.load('channels_i2_28b.npy')
outdoor = np.load('channels_o1_60.npy', allow_pickle=True)
combined = np.load('channels_combined.npy', allow_pickle=True).item()

print("Indoor shape:", indoor.shape)
print("Outdoor length:", len(outdoor))
print("Combined keys:", combined.keys())
