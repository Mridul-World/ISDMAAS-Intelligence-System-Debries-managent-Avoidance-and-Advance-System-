import torch

baseline = torch.load(
    "baseline_val.pt"
)

target = torch.load(
"../phase4/y_val.pt")

pos_err = (
    ((baseline[:, :3] - target[:, :3]) ** 2)
    .sum(dim=1)
    .sqrt()
    .mean()
)

vel_err = (
    ((baseline[:, 3:] - target[:, 3:]) ** 2)
    .sum(dim=1)
    .sqrt()
    .mean()
)

print("Mean Position Error (km):")
print(pos_err.item())

print("\nMean Velocity Error (km/s):")
print(vel_err.item())