"""
Performance baseline for the screening pipeline.

Run directly, not under pytest — it is a measurement, not an assertion:

    python tests/benchmark_screening.py

Catalog sizes above the bundled 15 894 objects are synthesised by cloning real
element sets with perturbed mean anomaly and RAAN. That keeps the propagation
work honest (every object is a distinct orbit that SGP4 must actually integrate)
while letting the scaling behaviour be measured past what the public catalog
provides. Synthesised sizes are labelled as such in the output; nothing here
claims 100 000 real objects were screened.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from sgp4.api import Satrec  # noqa: E402

from isdmaas_core import astrodynamics as astro  # noqa: E402
from isdmaas_core import screening as screening_service  # noqa: E402

CATALOG = APP_DIR / "data" / "catalog_active.tle"
EPOCH = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)


def load_real_catalog():
    lines = [line.rstrip() for line in CATALOG.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    objects = []
    for i in range(0, len(lines) - 2, 3):
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        if not (line1.startswith("1 ") and line2.startswith("2 ")):
            continue
        try:
            objects.append((int(line2[2:7]), name.strip(),
                            Satrec.twoline2rv(line1, line2)))
        except (ValueError, IndexError):
            continue
    return objects


def _checksum(line):
    total = 0
    for character in line[:68]:
        if character.isdigit():
            total += int(character)
        elif character == "-":
            total += 1
    return str(total % 10)


def synthesise(real, target):
    """
    Grow the catalog to `target` objects by perturbing real element sets.

    Mean anomaly and RAAN are varied so each clone is a genuinely different
    orbit — SGP4 has to do the same work per object as for a real one. Cloning
    without perturbation would measure memory bandwidth, not propagation.
    """
    if target <= len(real):
        return real[:target]
    out = list(real)
    source_lines = []
    lines = [line.rstrip() for line in CATALOG.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    for i in range(0, len(lines) - 2, 3):
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            source_lines.append((lines[i + 1], lines[i + 2]))

    index = 0
    norad = 700000
    while len(out) < target:
        line1, line2 = source_lines[index % len(source_lines)]
        index += 1
        raan = (float(line2[17:25]) + (index * 7.3)) % 360.0
        anomaly = (float(line2[43:51]) + (index * 11.7)) % 360.0
        body = (line2[:17] + f"{raan:8.4f}" + line2[25:43]
                + f"{anomaly:8.4f}" + line2[51:68])
        body = body[:68].ljust(68)
        try:
            out.append((norad, f"SYN-{norad}",
                        Satrec.twoline2rv(line1, body + _checksum(body))))
            norad += 1
        except Exception:
            continue
    return out[:target]


def benchmark(catalog, primary, window_h, gate_km, label):
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    result = screening_service.screen_primary(
        primary_sat=primary,
        primary_name="BENCH-PRIMARY",
        catalog=catalog,
        epoch=EPOCH,
        window_s=window_h * 3600.0,
        gate_km=gate_km,
        hbr_km=0.020,
        coarse_step_s=30.0,
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "label": label,
        "objects": len(catalog),
        "window_hours": window_h,
        "gate_km": gate_km,
        "seconds": round(elapsed, 3),
        "peak_mib": round(peak / 1024 / 1024, 1),
        "objects_per_second": int(len(catalog) / elapsed) if elapsed > 0 else 0,
        "after_geometric_filter": result.objects_after_geometric_filter,
        "coarse_candidates": result.coarse_candidates,
        "conjunctions": len(result.conjunctions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[1000, 5000, 15000, 50000, 100000])
    parser.add_argument("--window-hours", type=float, default=24.0)
    parser.add_argument("--gate-km", type=float, default=25.0)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    if not CATALOG.exists():
        raise SystemExit(f"bundled catalog not found at {CATALOG}")

    real = load_real_catalog()
    primary = next(sat for norad, _, sat in real if norad == 39634)
    print(f"real catalog: {len(real)} objects")
    print(f"window {args.window_hours} h, gate {args.gate_km} km, "
          f"coarse step 30 s, epoch {EPOCH.isoformat()}")
    print()
    header = (f"{'objects':>9} {'source':>11} {'seconds':>9} {'obj/s':>10} "
              f"{'peak MiB':>9} {'filtered':>9} {'coarse':>7} {'conj':>5}")
    print(header)
    print("-" * len(header))

    rows = []
    for size in args.sizes:
        catalog = synthesise(real, size)
        source = "real" if size <= len(real) else "synthesised"
        row = benchmark(catalog, primary, args.window_hours, args.gate_km, source)
        rows.append(row)
        print(f"{row['objects']:>9} {source:>11} {row['seconds']:>9.3f} "
              f"{row['objects_per_second']:>10} {row['peak_mib']:>9.1f} "
              f"{row['after_geometric_filter']:>9} "
              f"{row['coarse_candidates']:>7} {row['conjunctions']:>5}")

    # Component-level timings on the real catalog.
    print()
    print("component breakdown (real catalog, one primary):")
    survivors, considered = astro.geometric_filter(primary, real, args.gate_km)
    t0 = time.perf_counter()
    survivors, considered = astro.geometric_filter(primary, real, args.gate_km)
    t_filter = time.perf_counter() - t0

    offsets = astro.coarse_grid(args.window_hours * 3600.0, 30.0)
    t0 = time.perf_counter()
    track, _, ok = astro.propagate_series(primary, EPOCH, offsets)
    t_propagate = time.perf_counter() - t0

    t0 = time.perf_counter()
    candidates = astro.screen_track(track, ok, survivors, EPOCH, offsets, args.gate_km)
    t_screen = time.perf_counter() - t0

    t0 = time.perf_counter()
    by_norad = {n: s for n, _, s in survivors}
    refined = 0
    for candidate in candidates:
        if astro.refine_candidate(primary, by_norad[candidate.norad], EPOCH,
                                  candidate, args.gate_km):
            refined += 1
    t_refine = time.perf_counter() - t0

    print(f"  geometric filter   {t_filter * 1000:8.1f} ms  "
          f"({considered} -> {len(survivors)} objects)")
    print(f"  primary propagate  {t_propagate * 1000:8.1f} ms  "
          f"({len(offsets)} epochs)")
    print(f"  coarse screen      {t_screen * 1000:8.1f} ms  "
          f"({len(survivors)} objects x {len(offsets)} epochs)")
    print(f"  TCA refinement     {t_refine * 1000:8.1f} ms  "
          f"({len(candidates)} candidates -> {refined} refined)")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"rows": rows, "components": {
                "geometric_filter_ms": round(t_filter * 1000, 1),
                "primary_propagate_ms": round(t_propagate * 1000, 1),
                "coarse_screen_ms": round(t_screen * 1000, 1),
                "tca_refinement_ms": round(t_refine * 1000, 1),
            }}, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")


if __name__ == "__main__":
    main()
