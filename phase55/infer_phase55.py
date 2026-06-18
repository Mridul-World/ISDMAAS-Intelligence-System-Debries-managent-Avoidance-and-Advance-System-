"""
Phase 5.5 — inference on validation tensors (sanity tool).
    python infer_phase55.py [index]
For operational future prediction use infer_future.py.
"""
import sys
import numpy as np
import torch
from train_phase55_rtn import RTNTransformer, rtn_to_eci, CFG as TCFG


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    ld = lambda f: torch.load(f, weights_only=False).float()
    Xva, bva, yva = ld("X_val.pt"), ld("baseline_val.pt"), ld("y_val.pt")
    norm = torch.load(f"{TCFG['MODEL_DIR']}/residual_norm_phase55.pt", weights_only=False)
    rm, rs = norm["rm"], norm["rs"]
    net = RTNTransformer(Xva.shape[2], TCFG["D_MODEL"], TCFG["NHEAD"],
                         TCFG["NUM_LAYERS"], TCFG["DIM_FF"], TCFG["DROPOUT"])
    net.load_state_dict(torch.load(f"{TCFG['MODEL_DIR']}/rtn_physics_ml_phase55.pt",
                                   map_location="cpu"))
    net.eval()
    x, b, y = Xva[idx:idx+1], bva[idx:idx+1], yva[idx:idx+1]
    with torch.no_grad():
        pred_rtn = net(x) * rs + rm
        corrected = b.clone()
        corrected[:, :3] = b[:, :3] - rtn_to_eci(pred_rtn, b)
    e_sgp4 = float(np.linalg.norm((b[0, :3] - y[0, :3]).numpy()))
    e_ml = float(np.linalg.norm((corrected[0, :3] - y[0, :3]).numpy()))
    print("pred RTN residual [R,T,N] km:", pred_rtn[0].numpy().round(4).tolist())
    print(f"SGP4 error      : {e_sgp4:8.3f} km")
    print(f"SGP4 + ML error : {e_ml:8.3f} km")
    print(f"improvement     : {100*(1 - e_ml/max(e_sgp4,1e-9)):8.1f} %")


if __name__ == "__main__":
    main()
