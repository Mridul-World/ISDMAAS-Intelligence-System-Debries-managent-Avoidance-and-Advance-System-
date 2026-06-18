# phase5/debug_predictions.py

import torch
import numpy as np

truth = torch.load("../phase4/y_val.pt")

print("\nTruth shape:")
print(truth.shape)

print("\nFirst 5 truth samples:")
print(truth[:5])

print("\nMin:")
print(torch.min(truth))

print("\nMax:")
print(torch.max(truth))