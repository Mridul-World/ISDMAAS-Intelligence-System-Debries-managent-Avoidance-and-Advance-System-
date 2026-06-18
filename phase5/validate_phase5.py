import torch
import numpy as np

from physics_utils import (
    orbital_energy,
    angular_momentum,
    semi_major_axis,
    orbital_period,
)

def compute_metrics(pred, true):

    pred = pred.cpu()
    true = true.cpu()

    # Position error
    pos_error = torch.mean(
        torch.norm(
            pred[:, 0:3] - true[:, 0:3],
            dim=-1
        )
    ).item()

    # Energy
    E_pred = orbital_energy(
        pred[:,0:3],
        pred[:,3:6]
    )

    E_true = orbital_energy(
        true[:,0:3],
        true[:,3:6]
    )

    energy_error = torch.mean(
        torch.abs(E_pred - E_true)
    ).item()

    # Momentum
    h_pred = angular_momentum(
        pred[:,0:3],
        pred[:,3:6]
    )

    h_true = angular_momentum(
        true[:,0:3],
        true[:,3:6]
    )

    momentum_error = torch.mean(
        torch.norm(
            h_pred - h_true,
            dim=-1
        )
    ).item()

    # SMA
    sma_pred = semi_major_axis(
        pred[:,0:3],
        pred[:,3:6]
    )

    sma_true = semi_major_axis(
        true[:,0:3],
        true[:,3:6]
    )

    sma_error = torch.mean(
        torch.abs(
            sma_pred - sma_true
        )
    ).item()

    # Period
    T_pred = orbital_period(
        pred[:,0:3],
        pred[:,3:6]
    )

    T_true = orbital_period(
        true[:,0:3],
        true[:,3:6]
    )

    period_error = torch.mean(
        torch.abs(
            T_pred - T_true
        )
    ).item()

    return {
        "position_error_km": round(pos_error,4),
        "energy_error": round(energy_error,4),
        "momentum_error": round(momentum_error,4),
        "sma_error_km": round(sma_error,4),
        "period_error_sec": round(period_error,4),
    }