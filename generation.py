# generate_data.py (пример для создания своих тестов)
import numpy as np

def generate_activity_sample(activity_type):
    t = np.linspace(0, 2, 100)  # 2 seconds at 50 Hz
    
    if activity_type == 0:  # walking
        x = 8 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz step frequency
        y = 3 * np.sin(2 * np.pi * 2.0 * t)
        z = 64 + 15 * np.sin(2 * np.pi * 0.5 * t)
    elif activity_type == 1:  # running
        x = 15 * np.sin(2 * np.pi * 2.5 * t)
        y = 5 * np.sin(2 * np.pi * 3.0 * t)
        z = 64 + 25 * np.sin(2 * np.pi * 1.2 * t)
    elif activity_type == 2:  # jumping
        x = 5 * np.sin(2 * np.pi * 0.8 * t)
        y = 5 * np.sin(2 * np.pi * 0.8 * t + np.pi/4)
        z = 64 + 40 * np.abs(np.sin(2 * np.pi * 1.5 * t)) - 20
    elif activity_type == 3:  # sitting
        x = 2 * np.random.randn(100)
        y = 2 * np.random.randn(100)
        z = 64 + 3 * np.random.randn(100)
    else:  # standing
        x = 1 * np.random.randn(100)
        y = 1 * np.random.randn(100)
        z = 64 + 2 * np.random.randn(100)
    
    signal = np.column_stack([x, y, z])
    
    # Add noise
    signal += np.random.normal(0, 8, signal.shape)  # Gaussian noise
    
    # Spike noise (3% of points)
    spike_mask = np.random.binomial(1, 0.03, 100).astype(bool)
    signal[spike_mask] = np.random.choice([-127, 127], size=(spike_mask.sum(), 3))
    
    # Calibration drift
    drift = 10 * np.sin(2 * np.pi * 0.1 * t)[:, None]
    signal += drift
    
    # Clip to sensor range [-128, 127] and convert to int8
    signal = np.clip(signal, -128, 127).astype(np.int8)
    return signal

# Generate dataset
np.random.seed(42)
N = 1000
samples = []
labels = []

for i in range(N):
    act = np.random.randint(0, 5)
    labels.append(act)
    samples.append(generate_activity_sample(act))

# Save to files
with open("input.txt", "w") as f:
    f.write(f"{N}\n\n")
    for i, sample in enumerate(samples):
        for row in sample:
            f.write(f"{row[0]} {row[1]} {row[2]}\n")
        if i < N - 1:
            f.write("\n")

with open("labels.txt", "w") as f:
    f.write(" ".join(map(str, labels)))