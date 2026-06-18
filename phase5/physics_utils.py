import torch
import numpy as np

MU = 398600.4418

def orbital_energy(r, v):
    r_norm = torch.norm(r, dim=-1)
    v_norm = torch.norm(v, dim=-1)

    return 0.5 * v_norm**2 - MU / r_norm


def angular_momentum(r, v):
    return torch.cross(r, v, dim=-1)


def semi_major_axis(r, v):

    E = orbital_energy(r, v)

    return -MU / (2 * E)


def orbital_period(r, v):

    a = semi_major_axis(r, v)

    return 2 * np.pi * torch.sqrt(
        torch.abs(a**3) / MU
    )