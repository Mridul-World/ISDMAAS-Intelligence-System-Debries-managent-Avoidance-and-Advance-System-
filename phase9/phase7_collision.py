"""
Collision probability engine - re-exported from phase11/phase7_collision.py.

This file used to be a private copy of the engine. Six copies existed across
phase7/ phase8/ phase9/ phase10/ phase12/ phase13/, and they drifted: every one
of them still carried the one-sided TCA clamp (a pair that passed at 0.000 km
300 s ago was reported as a comfortable 3.000 km miss), a divide-by-zero on
co-located objects that produced pc=nan -- and `nan > threshold` is False, so
"we could not tell" was reported as "no action required" -- an uncaught
LinAlgError on a zero covariance, and confident Pc values (1.0, 0.039) for
covariances that were non-PSD or asymmetric. phase10/phase8_maneuver.py had also
quietly drifted to PC_SAFE = 1e-7 while every other phase used 1e-6, so the same
conjunction got a 10x stricter safety target depending on which directory you
ran it from.

Rather than copy the corrected engine six more times and wait for it to drift
again, each phase now resolves to the single validated implementation in
phase11/, which is the one covered by tests/scientific/. The public names are
unchanged, so `from phase7_collision import ...` keeps working exactly as before.

tests/scientific/test_engine_is_shared.py fails if a copy reappears.
"""
import importlib.util
import pathlib
import sys

_CANONICAL_DIR = pathlib.Path(__file__).resolve().parent.parent / "phase11"
_CANONICAL = _CANONICAL_DIR / "phase7_collision.py"


def _load(module_name):
    """
    Load phase11/<module_name>.py once, under a private name.

    It is registered in sys.modules before execution so that a canonical module
    importing another canonical module (phase8_maneuver imports phase7_collision)
    resolves to the same objects rather than re-executing the file.
    """
    private = "_isdmaas_canonical_" + module_name
    cached = sys.modules.get(private)
    if cached is not None:
        return cached
    path = _CANONICAL_DIR / (module_name + ".py")
    if not path.is_file():
        raise ImportError(
            "the canonical " + module_name + " is missing from " + str(_CANONICAL_DIR)
            + "; this phase directory re-exports it rather than keeping its own copy"
        )
    spec = importlib.util.spec_from_file_location(private, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[private] = module
    # phase8_maneuver does `from phase7_collision import ...` at import time.
    # Point that name at the canonical engine before executing it, so it cannot
    # pick up a stale copy that happens to be earlier on sys.path.
    if module_name != "phase7_collision":
        sys.modules.setdefault("phase7_collision", _load("phase7_collision"))
    spec.loader.exec_module(module)
    return module


_engine = _load("phase7_collision")

globals().update({
    name: value for name, value in vars(_engine).items()
    if not name.startswith("_")
})

__all__ = [name for name in vars(_engine) if not name.startswith("_")]
