# Load preprocessed dataset
outdoor_pre = np.load('channels_o1_60_preprocessed.npy')

# Extract amplitude and phase
amplitude = np.abs(outdoor_pre)
phase = np.angle(outdoor_pre)

print("Amplitude shape:", amplitude.shape)
print("Phase shape:", phase.shape)
