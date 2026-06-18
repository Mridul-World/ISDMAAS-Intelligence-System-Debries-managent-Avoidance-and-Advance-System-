import torch
import numpy as np

baseline = torch.load("baseline_val.pt")
target = torch.load("../phase4/y_val.pt")

print("Baseline first sample:")
print(baseline[0])

print("\nTarget first sample:")
print(target[0])

pos_err = np.linalg.norm(
    baseline[0][:3].numpy()
    -
    target[0][:3].numpy()
)

print("\nPosition error:")
print(pos_err)