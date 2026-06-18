"""
=============================================================
  SSA PHASE 3 — INTERACTIVE ORBIT VISUALIZER
  Reads REAL data from Phase 2 outputs
=============================================================

What this renders
─────────────────
  1.  3D Earth sphere with lat/lon grid
  2.  Real satellite orbits from your TLE dataset
  3.  Real debris orbits
  4.  Real conjunction/collision point from Phase 2 alerts
  5.  Time animation slider
  6.  Before vs After maneuver comparison

Inputs (from Phase 2)
─────────────────────
  • SSA_PROPAGATION_DATASET.csv   — your TLE dataset
  • conjunction_alerts_phase2.csv — Phase 2 output

Outputs
───────
  • orbit_visualizer.html     — main 3D interactive explorer
  • maneuver_comparison.html  — before/after avoidance maneuver
  • orbit_summary.png         — static Matplotlib summary figure

Requirements
────────────
  pip install plotly pandas numpy sgp4 scipy matplotlib
=============================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════
#  !! CONFIGURE THESE TWO PATHS !!
# ══════════════════════════════════════════════════════

TLE_DATA_PATH   = r"C:\Work_place\projects\isdmaas\phase2\data\ssa_data\SSA_PROPAGATION_DATASET.csv"
ALERTS_CSV_PATH = r"C:\Work_place\projects\isdmaas\phase2\data\conjunction_alerts_phase2.csv"

# Output folder — files are saved next to this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Visualization settings ───────────────────────────
ORBIT_HOURS    = 3      # hours of orbit arc to draw
TIME_STEP_MIN  = 2      # propagation step in minutes
N_SHOW_OBJECTS = 20     # max background objects to draw (keeps it fast)

EARTH_RADIUS_KM = 6371.0

RISK_COLORS = {
    "CRITICAL": "#FF0044",
    "HIGH":     "#FF8800",
    "ELEVATED": "#FFD700",
    "NOMINAL":  "#44FF88",
}


# ══════════════════════════════════════════════════════
#  STEP 1 — LOAD PHASE 2 DATA
# ══════════════════════════════════════════════════════

def load_tle_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] TLE dataset not found: {path}")
        sys.exit(1)
    needed = ["NORAD_ID", "TLE_LINE1", "TLE_LINE2", "OBJECT_TYPE"]
    df = pd.read_csv(path, usecols=needed, low_memory=False,
                     dtype={"TLE_LINE1": str, "TLE_LINE2": str, "NORAD_ID": str})
    df = df.dropna(subset=["TLE_LINE1", "TLE_LINE2"])
    df = df[df["TLE_LINE1"].str.strip().str.len() == 69]
    df = df[df["TLE_LINE2"].str.strip().str.len() == 69]
    df = df.drop_duplicates(subset=["NORAD_ID"]).reset_index(drop=True)
    print(f"[LOAD]  TLE dataset: {len(df):,} usable objects")
    return df


def load_alerts(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] Alerts CSV not found: {path}")
        print("        Run ssa_pipeline_phase2.py first to generate it.")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[LOAD]  Alerts CSV: {len(df):,} conjunction alerts")
    for risk in ["CRITICAL", "HIGH", "ELEVATED", "NOMINAL"]:
        n = len(df[df["RISK"] == risk])
        if n:
            print(f"         {risk:<10}: {n:>4}")
    return df


def pick_top_conjunction(alerts: pd.DataFrame) -> dict:
    """Pick highest-risk, smallest miss distance conjunction."""
    priority = {"CRITICAL": 0, "HIGH": 1, "ELEVATED": 2, "NOMINAL": 3}
    df = alerts.copy()
    df["_p"] = df["RISK"].map(priority).fillna(4)
    df = df.sort_values(["_p", "MISS_DIST_KM"]).reset_index(drop=True)
    row = df.iloc[0]
    conj = {
        "satA":      str(row["SAT_A"]),
        "satB":      str(row["SAT_B"]),
        "typeA":     str(row.get("TYPE_A", "UNKNOWN")),
        "typeB":     str(row.get("TYPE_B", "UNKNOWN")),
        "miss_dist": float(row["MISS_DIST_KM"]),
        "pc":        float(row["Pc"]),
        "risk":      str(row["RISK"]),
        "tca_str":   str(row.get("TCA_UTC", "N/A")),
        "tca_step":  int(row.get("TCA_STEP", 20)),
    }
    print(f"\n[CONJ]  Top conjunction selected:")
    print(f"         Sat A : {conj['satA']} ({conj['typeA']})")
    print(f"         Sat B : {conj['satB']} ({conj['typeB']})")
    print(f"         Miss  : {conj['miss_dist']:.3f} km")
    print(f"         Pc    : {conj['pc']:.6f}")
    print(f"         Risk  : {conj['risk']}")
    print(f"         TCA   : {conj['tca_str']}")
    return conj


# ══════════════════════════════════════════════════════
#  STEP 2 — PROPAGATION
# ══════════════════════════════════════════════════════

def propagate_orbit(line1: str, line2: str,
                    hours: float, step_min: int,
                    epoch: datetime) -> np.ndarray:
    try:
        sat = Satrec.twoline2rv(line1.strip(), line2.strip())
    except Exception:
        return None
    positions = []
    n_steps = int(hours * 60 / step_min) + 1
    for i in range(n_steps):
        t = epoch + timedelta(minutes=i * step_min)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
        e, r, _ = sat.sgp4(jd, fr)
        if e != 0:
            return None
        positions.append(r)
    return np.array(positions)


def get_tle_for(norad_id: str, tle_df: pd.DataFrame):
    row = tle_df[tle_df["NORAD_ID"].astype(str).str.strip() == str(norad_id).strip()]
    if row.empty:
        return None, None, None
    r = row.iloc[0]
    return r["TLE_LINE1"], r["TLE_LINE2"], str(r.get("OBJECT_TYPE", "UNKNOWN"))


# ══════════════════════════════════════════════════════
#  EARTH + GRID HELPERS
# ══════════════════════════════════════════════════════

def make_earth_sphere(n: int = 80) -> go.Surface:
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones(n), np.cos(v))
    lon = np.outer(np.linspace(-180, 180, n), np.ones(n))
    lat = np.outer(np.ones(n), np.linspace(-90, 90, n))
    texture = (
        0.4 * np.sin(np.radians(lon * 3)) * np.cos(np.radians(lat * 2)) +
        0.3 * np.cos(np.radians(lon * 5)) * np.sin(np.radians(lat * 4)) +
        0.3 * np.sin(np.radians(lat * 6))
    )
    return go.Surface(
        x=x, y=y, z=z, surfacecolor=texture,
        colorscale=[
            [0.0, "#0a1f4b"], [0.3, "#1a4a8a"], [0.5, "#2d6a4f"],
            [0.7, "#4a8c3f"], [0.85, "#8B7355"], [1.0, "#FFFFFF"],
        ],
        showscale=False, opacity=0.92,
        lighting=dict(ambient=0.7, diffuse=0.8, specular=0.2),
        lightposition=dict(x=2000, y=3000, z=5000),
        hoverinfo="skip", name="Earth",
    )


def make_lat_lon_grid() -> list:
    traces = []
    R = EARTH_RADIUS_KM * 1.001
    u = np.linspace(0, 2 * np.pi, 100)
    for lat_deg in range(-60, 91, 30):
        lat = np.radians(lat_deg)
        traces.append(go.Scatter3d(
            x=R * np.cos(lat) * np.cos(u),
            y=R * np.cos(lat) * np.sin(u),
            z=R * np.sin(lat) * np.ones_like(u),
            mode="lines", line=dict(color="rgba(255,255,255,0.10)", width=1),
            hoverinfo="skip", showlegend=False, name="grid",
        ))
    lats = np.linspace(-np.pi / 2, np.pi / 2, 100)
    for lon_deg in range(0, 360, 45):
        lon = np.radians(lon_deg)
        traces.append(go.Scatter3d(
            x=R * np.cos(lats) * np.cos(lon),
            y=R * np.cos(lats) * np.sin(lon),
            z=R * np.sin(lats),
            mode="lines", line=dict(color="rgba(255,255,255,0.10)", width=1),
            hoverinfo="skip", showlegend=False, name="grid",
        ))
    return traces


# ══════════════════════════════════════════════════════
#  OUTPUT 1 — orbit_visualizer.html
# ══════════════════════════════════════════════════════

def build_orbit_visualizer(tle_df, alerts_df, conj):
    print("\n[VIZ-1] Building main 3D orbit visualizer …")
    epoch = datetime.utcnow().replace(microsecond=0)

    l1A, l2A, _ = get_tle_for(conj["satA"], tle_df)
    l1B, l2B, _ = get_tle_for(conj["satB"], tle_df)

    traj_A = propagate_orbit(l1A, l2A, ORBIT_HOURS, TIME_STEP_MIN, epoch) if l1A else None
    traj_B = propagate_orbit(l1B, l2B, ORBIT_HOURS, TIME_STEP_MIN, epoch) if l1B else None

    if traj_A is not None:
        print(f"  ✓ Sat A ({conj['satA']}): {len(traj_A)} pts")
    if traj_B is not None:
        print(f"  ✓ Sat B ({conj['satB']}): {len(traj_B)} pts")

    # Background objects from other alerts
    main_ids = {conj["satA"], conj["satB"]}
    other_ids = []
    for _, row in alerts_df.iterrows():
        for sid in [str(row["SAT_A"]), str(row["SAT_B"])]:
            if sid not in main_ids and sid not in other_ids:
                other_ids.append(sid)
        if len(other_ids) >= N_SHOW_OBJECTS:
            break

    bg_trajs, bg_types = {}, {}
    for sid in other_ids:
        l1, l2, otype = get_tle_for(sid, tle_df)
        if l1:
            t = propagate_orbit(l1, l2, ORBIT_HOURS, TIME_STEP_MIN, epoch)
            if t is not None:
                bg_trajs[sid] = t
                bg_types[sid] = otype
    print(f"  ✓ Background objects: {len(bg_trajs)}")

    tca_step = 0
    if traj_A is not None and traj_B is not None:
        tca_step = min(conj["tca_step"], len(traj_A)-1, len(traj_B)-1)

    risk_color = RISK_COLORS.get(conj["risk"], "#FFFFFF")
    type_color_map = {"PAYLOAD": "#4488FF", "DEBRIS": "#FF6644",
                      "ROCKET BODY": "#CC88FF", "UNKNOWN": "#888888"}

    fig = go.Figure()
    fig.add_trace(make_earth_sphere())
    for g in make_lat_lon_grid():
        fig.add_trace(g)

    # Background orbits
    shown_types = set()
    for sid, traj in bg_trajs.items():
        otype = bg_types.get(sid, "UNKNOWN")
        col = type_color_map.get(otype, "#888888")
        show_leg = otype not in shown_types
        shown_types.add(otype)
        fig.add_trace(go.Scatter3d(
            x=traj[:,0], y=traj[:,1], z=traj[:,2],
            mode="lines", line=dict(color=col, width=1),
            opacity=0.30, name=otype, legendgroup=otype,
            showlegend=show_leg, hoverinfo="skip",
        ))

    # Conjunction orbit A
    if traj_A is not None:
        fig.add_trace(go.Scatter3d(
            x=traj_A[:,0], y=traj_A[:,1], z=traj_A[:,2],
            mode="lines", line=dict(color="#00D4FF", width=3),
            name=f"Sat A: {conj['satA']} ({conj['typeA']})",
            hoverinfo="skip", opacity=0.95,
        ))
        pA = traj_A[tca_step]
        alt_A = np.linalg.norm(pA) - EARTH_RADIUS_KM
        fig.add_trace(go.Scatter3d(
            x=[pA[0]], y=[pA[1]], z=[pA[2]],
            mode="markers+text",
            marker=dict(size=9, color="#00D4FF", line=dict(color="white", width=1)),
            text=[f"Sat A\n{conj['satA']}"],
            textposition="top center",
            textfont=dict(size=9, color="#00D4FF"),
            name="Sat A @ TCA",
            hovertemplate=(
                f"<b>Sat A: {conj['satA']}</b><br>"
                f"Type: {conj['typeA']}<br>"
                f"Alt: {alt_A:.0f} km<extra></extra>"
            ),
        ))

    # Conjunction orbit B
    if traj_B is not None:
        fig.add_trace(go.Scatter3d(
            x=traj_B[:,0], y=traj_B[:,1], z=traj_B[:,2],
            mode="lines", line=dict(color=risk_color, width=3),
            name=f"Sat B: {conj['satB']} ({conj['typeB']})",
            hoverinfo="skip", opacity=0.95,
        ))
        pB = traj_B[tca_step]
        alt_B = np.linalg.norm(pB) - EARTH_RADIUS_KM
        fig.add_trace(go.Scatter3d(
            x=[pB[0]], y=[pB[1]], z=[pB[2]],
            mode="markers+text",
            marker=dict(size=9, color=risk_color, line=dict(color="white", width=1)),
            text=[f"Sat B\n{conj['satB']}"],
            textposition="top center",
            textfont=dict(size=9, color=risk_color),
            name="Sat B @ TCA",
            hovertemplate=(
                f"<b>Sat B: {conj['satB']}</b><br>"
                f"Type: {conj['typeB']}<br>"
                f"Alt: {alt_B:.0f} km<extra></extra>"
            ),
        ))

    # Miss distance line + conjunction diamond
    if traj_A is not None and traj_B is not None:
        pA = traj_A[tca_step]
        pB = traj_B[tca_step]
        mid = (pA + pB) / 2

        fig.add_trace(go.Scatter3d(
            x=[pA[0], pB[0]], y=[pA[1], pB[1]], z=[pA[2], pB[2]],
            mode="lines", line=dict(color=risk_color, width=5, dash="dash"),
            name=f"Miss dist: {conj['miss_dist']:.3f} km",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter3d(
            x=[mid[0]], y=[mid[1]], z=[mid[2]],
            mode="markers",
            marker=dict(size=18, color=risk_color, symbol="diamond",
                        opacity=0.9, line=dict(color="white", width=2)),
            name=f"⚠ {conj['risk']} CONJUNCTION",
            hovertemplate=(
                f"<b>⚠ {conj['risk']} CONJUNCTION</b><br>"
                f"Sat A: {conj['satA']} ({conj['typeA']})<br>"
                f"Sat B: {conj['satB']} ({conj['typeB']})<br>"
                f"Miss Distance: {conj['miss_dist']:.3f} km<br>"
                f"Pc: {conj['pc']:.6f}<br>"
                f"TCA: {conj['tca_str']}<extra></extra>"
            ),
        ))

        # Warning sphere
        u2 = np.linspace(0, 2*np.pi, 25)
        v2 = np.linspace(0, np.pi, 25)
        r_warn = max(conj["miss_dist"] * 30, 80)
        fig.add_trace(go.Surface(
            x=mid[0] + r_warn * np.outer(np.cos(u2), np.sin(v2)),
            y=mid[1] + r_warn * np.outer(np.sin(u2), np.sin(v2)),
            z=mid[2] + r_warn * np.outer(np.ones(25), np.cos(v2)),
            colorscale=[[0,"rgba(255,0,68,0.05)"],[1,"rgba(255,80,0,0.12)"]],
            showscale=False, hoverinfo="skip", name="Warning Zone", opacity=0.20,
        ))

    # Animation
    update_menus, sliders = [], []
    if traj_A is not None and traj_B is not None:
        n_frames = min(len(traj_A), len(traj_B), 25)
        step_idx = np.linspace(0, min(len(traj_A), len(traj_B))-1,
                               n_frames, dtype=int)
        frames = []
        for fi, si in enumerate(step_idx):
            pA_f = traj_A[si]
            pB_f = traj_B[si]
            t_min = si * TIME_STEP_MIN
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(x=[pA_f[0]], y=[pA_f[1]], z=[pA_f[2]],
                                 mode="markers",
                                 marker=dict(size=9, color="#00D4FF",
                                             line=dict(color="white", width=1)),
                                 hovertemplate=f"Sat A @ T+{t_min}min<extra></extra>"),
                    go.Scatter3d(x=[pB_f[0]], y=[pB_f[1]], z=[pB_f[2]],
                                 mode="markers",
                                 marker=dict(size=9, color=risk_color,
                                             line=dict(color="white", width=1)),
                                 hovertemplate=f"Sat B @ T+{t_min}min<extra></extra>"),
                ],
                name=str(fi),
            ))
        fig.frames = frames

        update_menus = [dict(
            type="buttons", showactive=False,
            y=0.05, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶  Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=150, redraw=True),
                                      fromcurrent=True)]),
                dict(label="⏸  Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
            font=dict(color="white", size=12),
            bgcolor="rgba(40,40,80,0.9)",
            bordercolor="rgba(100,100,255,0.5)",
        )]
        sliders = [dict(
            active=0,
            steps=[dict(
                args=[[str(i)], dict(frame=dict(duration=0, redraw=True),
                                     mode="immediate")],
                label=f"T+{step_idx[i]*TIME_STEP_MIN}m",
                method="animate",
            ) for i in range(n_frames)],
            x=0.05, y=0.02, len=0.9,
            currentvalue=dict(font=dict(size=11, color="white"),
                              prefix="Time: ", visible=True, xanchor="center"),
            tickcolor="white", font=dict(color="white", size=9),
            bgcolor="rgba(30,30,60,0.8)",
            bordercolor="rgba(100,100,255,0.4)",
        )]

    fig.update_layout(
        title=dict(
            text=(
                f"🛰  SSA Phase 3 — Real Orbit Visualizer<br>"
                f"<sup>{len(alerts_df):,} alerts · "
                f"Top conjunction: {conj['satA']} ↔ {conj['satB']} · "
                f"Miss: {conj['miss_dist']:.3f} km · Risk: {conj['risk']}</sup>"
            ),
            font=dict(size=17, color="white"), x=0.5, xanchor="center",
        ),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, backgroundcolor="black"),
            yaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, backgroundcolor="black"),
            zaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, backgroundcolor="black"),
            bgcolor="black", aspectmode="cube",
            camera=dict(eye=dict(x=1.8, y=1.0, z=0.9)),
            dragmode="orbit",
        ),
        paper_bgcolor="#0a0a14", plot_bgcolor="#0a0a14",
        legend=dict(
            font=dict(color="white", size=10),
            bgcolor="rgba(20,20,40,0.85)",
            bordercolor="rgba(100,100,255,0.3)",
            borderwidth=1, x=0.01, y=0.99,
        ),
        margin=dict(l=0, r=0, t=90, b=0),
        updatemenus=update_menus,
        sliders=sliders,
        annotations=[dict(
            x=0.5, y=0.97, xref="paper", yref="paper",
            text=(
                f"⚠ {conj['risk']}  |  "
                f"{conj['satA']} ({conj['typeA']}) ↔ "
                f"{conj['satB']} ({conj['typeB']})   "
                f"Miss: {conj['miss_dist']:.3f} km   "
                f"Pc: {conj['pc']:.6f}   "
                f"TCA: {conj['tca_str']}"
            ),
            showarrow=False,
            font=dict(size=11, color=risk_color),
            bgcolor="rgba(60,0,0,0.6)",
            bordercolor=risk_color, borderwidth=1, borderpad=6,
        )],
    )

    out = os.path.join(OUTPUT_DIR, "orbit_visualizer.html")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "scrollZoom": True})
    print(f"  ✓ Saved → {out}")


# ══════════════════════════════════════════════════════
#  OUTPUT 2 — maneuver_comparison.html
# ══════════════════════════════════════════════════════

def build_maneuver_comparison(tle_df, conj):
    print("\n[VIZ-2] Building before/after maneuver comparison …")
    epoch = datetime.utcnow().replace(microsecond=0)

    l1A, l2A, _ = get_tle_for(conj["satA"], tle_df)
    l1B, l2B, _ = get_tle_for(conj["satB"], tle_df)

    if l1A is None or l1B is None:
        print("[WARN]  TLEs missing — skipping maneuver comparison.")
        return

    traj_A = propagate_orbit(l1A, l2A, ORBIT_HOURS, TIME_STEP_MIN, epoch)
    traj_B = propagate_orbit(l1B, l2B, ORBIT_HOURS, TIME_STEP_MIN, epoch)

    if traj_A is None or traj_B is None:
        print("[WARN]  Propagation failed — skipping maneuver comparison.")
        return

    tca_step = min(conj["tca_step"], len(traj_A)-1, len(traj_B)-1)

    # Dodge maneuver: 1.2 m/s retrograde on Sat A at T-45 min
    DELTA_V_KMS   = 0.0012
    maneuver_step = max(0, tca_step - int(45 / TIME_STEP_MIN))
    traj_A_dodge  = traj_A.copy()
    for step in range(maneuver_step, len(traj_A_dodge)):
        drift = (step - maneuver_step) * DELTA_V_KMS * TIME_STEP_MIN * 60
        direction = traj_A_dodge[step] / (np.linalg.norm(traj_A_dodge[step]) + 1e-9)
        traj_A_dodge[step] -= drift * np.array([0.3, 0.3, 0.4]) * direction

    pA_nom  = traj_A[tca_step]
    pA_dod  = traj_A_dodge[tca_step]
    pB      = traj_B[tca_step]
    mid_nom = (pA_nom + pB) / 2
    mid_dod = (pA_dod + pB) / 2

    new_miss = float(np.linalg.norm(pA_dod - pB))
    new_pc   = float(np.exp(-(new_miss**2) / (2.0 * 1.5**2)))
    new_risk = ("CRITICAL" if new_miss < 1.0 and new_pc > 0.5 else
                "HIGH"     if new_miss < 5.0 and new_pc > 0.05 else
                "ELEVATED" if new_miss < 10.0 and new_pc > 0.001 else "NOMINAL")

    rc_nom = RISK_COLORS.get(conj["risk"], "#FF0044")
    rc_dod = RISK_COLORS.get(new_risk, "#44FF88")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type":"scene"},{"type":"scene"}]],
        subplot_titles=[
            f"<b>BEFORE MANEUVER</b> — {conj['risk']}<br>"
            f"Miss: {conj['miss_dist']:.3f} km  |  Pc: {conj['pc']:.4f}",
            f"<b>AFTER MANEUVER</b> (Δv = 1.2 m/s) — {new_risk}<br>"
            f"Miss: {new_miss:.1f} km  |  Pc: {new_pc:.6f}",
        ],
        horizontal_spacing=0.02,
    )

    SCENE = dict(
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,backgroundcolor="black"),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,backgroundcolor="black"),
        zaxis=dict(showgrid=False,zeroline=False,showticklabels=False,backgroundcolor="black"),
        bgcolor="black", aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=0.8, z=0.8)),
    )

    def add_panel(col, traj_a, pA_tca, pB_tca, mid, miss_km, risk_c):
        n = 40
        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(0, np.pi, n)
        fig.add_trace(go.Surface(
            x=EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v)),
            y=EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v)),
            z=EARTH_RADIUS_KM * np.outer(np.ones(n), np.cos(v)),
            colorscale=[[0,"#0a1f4b"],[0.5,"#2d6a4f"],[1,"#FFFFFF"]],
            showscale=False, opacity=0.85, hoverinfo="skip",
        ), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=traj_a[:,0], y=traj_a[:,1], z=traj_a[:,2],
            mode="lines", line=dict(color="#00D4FF", width=2.5),
            name=f"Sat A", hoverinfo="skip",
        ), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=traj_B[:,0], y=traj_B[:,1], z=traj_B[:,2],
            mode="lines", line=dict(color=risk_c, width=2),
            name=f"Sat B", hoverinfo="skip",
        ), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=[pA_tca[0]], y=[pA_tca[1]], z=[pA_tca[2]],
            mode="markers", marker=dict(size=8, color="#00D4FF",
                                        line=dict(color="white", width=1)),
        ), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=[pB_tca[0]], y=[pB_tca[1]], z=[pB_tca[2]],
            mode="markers", marker=dict(size=7, color=risk_c,
                                        line=dict(color="white", width=1)),
        ), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=[pA_tca[0], pB_tca[0]],
            y=[pA_tca[1], pB_tca[1]],
            z=[pA_tca[2], pB_tca[2]],
            mode="lines", line=dict(color=risk_c, width=5, dash="dash"),
            name=f"Miss: {miss_km:.1f} km", hoverinfo="skip",
        ), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=[mid[0]], y=[mid[1]], z=[mid[2]],
            mode="markers",
            marker=dict(size=14, color=risk_c, symbol="diamond",
                        opacity=0.9, line=dict(color="white", width=2)),
            hovertemplate=f"Miss: {miss_km:.1f} km<extra></extra>",
        ), row=1, col=col)

    add_panel(1, traj_A,       pA_nom, pB, mid_nom, conj["miss_dist"], rc_nom)
    add_panel(2, traj_A_dodge, pA_dod, pB, mid_dod, new_miss,          rc_dod)

    fig.update_layout(
        title=dict(
            text=f"🛰  Maneuver Analysis — {conj['satA']} Avoidance Burn",
            font=dict(size=16, color="white"), x=0.5, xanchor="center",
        ),
        paper_bgcolor="#0a0a14",
        scene=SCENE, scene2=SCENE,
        showlegend=False,
        height=650, margin=dict(l=0, r=0, t=100, b=10),
    )

    out = os.path.join(OUTPUT_DIR, "maneuver_comparison.html")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"scrollZoom": True})
    print(f"  ✓ Saved → {out}")
    print(f"  ℹ  Nominal miss  : {conj['miss_dist']:.3f} km  ({conj['risk']})")
    print(f"  ℹ  Post-maneuver : {new_miss:.1f} km  ({new_risk})")


# ══════════════════════════════════════════════════════
#  OUTPUT 3 — orbit_summary.png
# ══════════════════════════════════════════════════════

def build_static_summary(tle_df, alerts_df, conj):
    print("\n[VIZ-3] Building static Matplotlib summary …")
    epoch = datetime.utcnow().replace(microsecond=0)

    l1A, l2A, _ = get_tle_for(conj["satA"], tle_df)
    l1B, l2B, _ = get_tle_for(conj["satB"], tle_df)
    traj_A = propagate_orbit(l1A, l2A, ORBIT_HOURS, TIME_STEP_MIN, epoch) if l1A else None
    traj_B = propagate_orbit(l1B, l2B, ORBIT_HOURS, TIME_STEP_MIN, epoch) if l1B else None

    type_color_map = {"PAYLOAD":"#4488FF","DEBRIS":"#FF6644",
                      "ROCKET BODY":"#CC88FF","UNKNOWN":"#888888"}

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12), facecolor="#0a0a14")
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    # Panel 1: 3D orbit view
    ax1 = fig.add_subplot(gs[0, :2], projection="3d")
    ax1.set_facecolor("#0a0a14")
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    ax1.plot_surface(
        EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v)),
        EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v)),
        EARTH_RADIUS_KM * np.outer(np.ones(40), np.cos(v)),
        color="#1a4a8a", alpha=0.5, linewidth=0,
    )

    # Background orbits
    bg_ids = list(set(
        alerts_df["SAT_A"].astype(str).tolist() +
        alerts_df["SAT_B"].astype(str).tolist()
    ))
    bg_ids = [x for x in bg_ids if x not in {conj["satA"], conj["satB"]}][:8]
    for sid in bg_ids:
        l1, l2, otype = get_tle_for(sid, tle_df)
        if l1:
            t = propagate_orbit(l1, l2, ORBIT_HOURS, TIME_STEP_MIN, epoch)
            if t is not None:
                ax1.plot(t[:,0], t[:,1], t[:,2],
                         color=type_color_map.get(otype, "#888888"),
                         linewidth=0.8, alpha=0.3)

    if traj_A is not None:
        ax1.plot(traj_A[:,0], traj_A[:,1], traj_A[:,2],
                 color="#00D4FF", linewidth=2.5,
                 label=f"Sat A: {conj['satA']}")
    if traj_B is not None:
        ax1.plot(traj_B[:,0], traj_B[:,1], traj_B[:,2],
                 color=RISK_COLORS.get(conj["risk"],"#FF8800"),
                 linewidth=2.5, label=f"Sat B: {conj['satB']}")

    if traj_A is not None and traj_B is not None:
        tca_step = min(conj["tca_step"], len(traj_A)-1, len(traj_B)-1)
        pA = traj_A[tca_step]
        pB = traj_B[tca_step]
        mid = (pA + pB) / 2
        ax1.scatter([mid[0]], [mid[1]], [mid[2]],
                    color="#FF0044", s=250, marker="*", zorder=10,
                    label=f"⚠ Conjunction ({conj['risk']})")

    ax1.set_xlabel("X (km)", color="gray", fontsize=8)
    ax1.set_ylabel("Y (km)", color="gray", fontsize=8)
    ax1.set_zlabel("Z (km)", color="gray", fontsize=8)
    ax1.set_title(f"3D Orbit Overview  |  {len(alerts_df):,} total alerts",
                  color="white", fontsize=11, pad=10)
    ax1.tick_params(colors="gray", labelsize=7)
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.4,
               facecolor="#0a0a14", edgecolor="gray")
    ax1.grid(False)

    # Panel 2: Risk pie
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#0a0a14")
    risk_counts = alerts_df["RISK"].value_counts()
    pie_labels, pie_sizes, pie_colors = [], [], []
    for r in ["CRITICAL","HIGH","ELEVATED","NOMINAL"]:
        if r in risk_counts:
            pie_labels.append(f"{r}\n({risk_counts[r]})")
            pie_sizes.append(risk_counts[r])
            pie_colors.append(RISK_COLORS[r])
    if pie_sizes:
        _, texts, autotexts = ax2.pie(
            pie_sizes, labels=pie_labels, colors=pie_colors,
            autopct="%1.1f%%", startangle=90,
            textprops=dict(color="white", fontsize=8),
            wedgeprops=dict(edgecolor="#0a0a14", linewidth=1.5),
        )
        for at in autotexts:
            at.set_fontsize(8)
    ax2.set_title("Alert Distribution\nby Risk Level",
                  color="white", fontsize=11, pad=10)

    # Panel 3: Altitude vs time
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor("#0d0d1a")
    if traj_A is not None:
        t_axis = np.arange(len(traj_A)) * TIME_STEP_MIN
        ax3.plot(t_axis, np.linalg.norm(traj_A, axis=1) - EARTH_RADIUS_KM,
                 color="#00D4FF", linewidth=2.5,
                 label=f"Sat A: {conj['satA']}")
    if traj_B is not None:
        t_axis_B = np.arange(len(traj_B)) * TIME_STEP_MIN
        ax3.plot(t_axis_B, np.linalg.norm(traj_B, axis=1) - EARTH_RADIUS_KM,
                 color=RISK_COLORS.get(conj["risk"],"#FF8800"),
                 linewidth=2.5, label=f"Sat B: {conj['satB']}")
    if traj_A is not None and traj_B is not None:
        tca_min = min(conj["tca_step"], len(traj_A)-1) * TIME_STEP_MIN
        tca_alt = np.linalg.norm(traj_A[min(conj["tca_step"], len(traj_A)-1)]) - EARTH_RADIUS_KM
        ax3.axvline(x=tca_min, color="#FF0044", linestyle="--",
                    linewidth=1.5, alpha=0.8)
        ax3.annotate(
            f"⚠ TCA\nMiss: {conj['miss_dist']:.3f} km\nPc: {conj['pc']:.4f}",
            xy=(tca_min, tca_alt),
            xytext=(tca_min + 10, tca_alt + 15),
            color="#FF4444", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#FF4444"),
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#300010", edgecolor="#FF4444"),
        )
    ax3.set_xlabel("Time (minutes)", color="gray", fontsize=9)
    ax3.set_ylabel("Altitude (km)", color="gray", fontsize=9)
    ax3.set_title("Orbital Altitude vs Time", color="white", fontsize=11)
    ax3.tick_params(colors="gray")
    ax3.legend(fontsize=8, framealpha=0.4, facecolor="#0a0a14",
               edgecolor="gray", loc="best")
    ax3.grid(True, alpha=0.15, color="gray")
    for sp in ax3.spines.values():
        sp.set_edgecolor("#333355")

    # Panel 4: Object type bar chart
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor("#0a0a14")
    types_A = alerts_df["TYPE_A"].value_counts() if "TYPE_A" in alerts_df.columns else pd.Series(dtype=int)
    types_B = alerts_df["TYPE_B"].value_counts() if "TYPE_B" in alerts_df.columns else pd.Series(dtype=int)
    type_totals = types_A.add(types_B, fill_value=0).sort_values(ascending=True)
    if len(type_totals):
        bar_colors = [type_color_map.get(t, "#888888") for t in type_totals.index]
        bars = ax4.barh(type_totals.index, type_totals.values,
                        color=bar_colors, edgecolor="#0a0a14", height=0.5)
        for bar, val in zip(bars, type_totals.values):
            ax4.text(bar.get_width() + 0.5,
                     bar.get_y() + bar.get_height()/2,
                     f"  {int(val)}", va="center", color="white", fontsize=9)
        ax4.set_xlim(0, type_totals.max() * 1.25)
    ax4.set_xlabel("Appearances in Alerts", color="gray", fontsize=9)
    ax4.set_title("Object Types in Alerts", color="white", fontsize=11)
    ax4.tick_params(colors="gray")
    ax4.grid(True, axis="x", alpha=0.15, color="gray")
    for sp in ax4.spines.values():
        sp.set_edgecolor("#333355")

    fig.suptitle(
        f"SSA Phase 3 — Orbit Visualization Summary  "
        f"({len(alerts_df):,} alerts from Phase 2)",
        color="white", fontsize=14, fontweight="bold", y=0.99,
    )

    out = os.path.join(OUTPUT_DIR, "orbit_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="#0a0a14", edgecolor="none")
    plt.close()
    print(f"  ✓ Saved → {out}")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  SSA PHASE 3 — ORBIT VISUALIZATION  (Real Phase 2 Data)")
    print("=" * 62)

    tle_df    = load_tle_dataset(TLE_DATA_PATH)
    alerts_df = load_alerts(ALERTS_CSV_PATH)
    conj      = pick_top_conjunction(alerts_df)

    build_orbit_visualizer(tle_df, alerts_df, conj)
    build_maneuver_comparison(tle_df, conj)
    build_static_summary(tle_df, alerts_df, conj)

    print("\n" + "=" * 62)
    print("  PHASE 3 COMPLETE — 3 files generated")
    print("=" * 62)
    print(f"\n  📁 Output folder: same folder as this script")
    print("  1. orbit_visualizer.html    — 3D interactive explorer")
    print("  2. maneuver_comparison.html — before/after avoidance")
    print("  3. orbit_summary.png        — static summary chart\n")