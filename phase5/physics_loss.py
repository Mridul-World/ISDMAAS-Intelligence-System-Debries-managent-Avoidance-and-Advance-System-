import torch
import torch.nn.functional as F

from physics_utils import (
    orbital_energy,
    angular_momentum,
    semi_major_axis,
    orbital_period
)

def physics_loss(
    pred,
    target,
    w_energy=0.05,
    w_momentum=0.05,
    w_sma=0.02,
    w_period=0.01,
):

    # =====================================================
    # MSE
    # =====================================================

    mse = F.mse_loss(pred, target)

    # =====================================================
    # SPLIT STATE VECTORS
    # =====================================================

    r_pred = pred[:, 0:3]
    v_pred = pred[:, 3:6]

    r_true = target[:, 0:3]
    v_true = target[:, 3:6]

    # =====================================================
    # ENERGY
    # =====================================================

    E_pred = orbital_energy(
        r_pred,
        v_pred
    )

    E_true = orbital_energy(
        r_true,
        v_true
    )

    energy_loss = torch.mean(
        torch.abs(
            E_pred - E_true
        )
    ) / 100.0

    # =====================================================
    # ANGULAR MOMENTUM
    # =====================================================

    h_pred = angular_momentum(
        r_pred,
        v_pred
    )

    h_true = angular_momentum(
        r_true,
        v_true
    )

    momentum_loss = torch.mean(
        torch.norm(
            h_pred - h_true,
            dim=-1
        )
    ) / 10000.0

    # =====================================================
    # SEMI-MAJOR AXIS
    # =====================================================

    sma_pred = semi_major_axis(
        r_pred,
        v_pred
    )

    sma_true = semi_major_axis(
        r_true,
        v_true
    )

    sma_loss = torch.mean(
        torch.abs(
            sma_pred - sma_true
        )
    ) / 1000.0

    # =====================================================
    # ORBITAL PERIOD
    # =====================================================

    period_pred = orbital_period(
        r_pred,
        v_pred
    )

    period_true = orbital_period(
        r_true,
        v_true
    )

    period_loss = torch.mean(
        torch.abs(
            period_pred - period_true
        )
    ) / 1000.0

    # =====================================================
    # TOTAL LOSS
    # =====================================================

    total = (
        mse
        + w_energy * energy_loss
        + w_momentum * momentum_loss
        + w_sma * sma_loss
        + w_period * period_loss
    )

    return total