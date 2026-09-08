"""
Regression guard for the duplicated-engine defect.

phase7_collision.py existed as seven separate files and phase8_maneuver.py as
five. The copies drifted, and every copy outside phase11/ was stale. Measured on
the copies as found:

    a pair that passed at 0.000 km 300 s ago  reported as a 3.000 km miss
    co-located objects                        pc = nan, and `nan > threshold` is
                                              False, so "we could not tell"
                                              became "no action required"
    all-zero covariance                       uncaught LinAlgError
    non-PSD covariance                        pc = 1.0, confident and wrong
    asymmetric covariance                     pc = 0.039, silently accepted

phase10/phase8_maneuver.py had also drifted to PC_SAFE = 1e-7 while every other
phase used 1e-6 -- the same conjunction got a 10x stricter safety target
depending on which directory you ran it from.

These tests fail if a private copy reappears, and they check the engine's
behaviour through each phase directory rather than trusting the file contents.
"""
import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
CANONICAL_DIR = REPO / "phase11"

COLLISION_DIRS = ["phase7", "phase8", "phase9", "phase10", "phase12", "phase13"]
MANEUVER_DIRS = ["phase8", "phase9", "phase10", "phase12"]

CASES = [
    ("phase7_collision", COLLISION_DIRS),
    ("phase8_maneuver", MANEUVER_DIRS),
]


def _existing(module, dirs):
    """Only test directories that are actually present in this checkout."""
    return [d for d in dirs if (REPO / d / f"{module}.py").is_file()]


# ============================================== no private copy may reappear
@pytest.mark.parametrize("module,dirs", CASES)
def test_no_phase_directory_carries_its_own_engine(module, dirs):
    """
    Each phase directory must re-export the canonical module, not reimplement it.

    The check is for the marker the shim uses, not for file size or a hash, so
    that ordinary edits to the canonical engine do not trip it.
    """
    offenders = []
    for name in _existing(module, dirs):
        text = (REPO / name / f"{module}.py").read_text(encoding="utf-8")
        if "_CANONICAL_DIR" not in text:
            offenders.append(name)
    assert not offenders, (
        f"{offenders} carry a private copy of {module}.py. Copies drift: every "
        f"one of them was stale when this guard was written. Re-export "
        f"phase11/{module}.py instead."
    )


@pytest.mark.parametrize("module,dirs", CASES)
def test_every_phase_directory_resolves_to_the_canonical_file(module, dirs):
    """Import the module the way its own phase does, and check where it came from."""
    for name in _existing(module, dirs):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(f"""
                import sys
                sys.path.insert(0, {str(REPO / name)!r})
                import {module}
                print({module}._engine.__file__)
            """)],
            capture_output=True, text=True, cwd=str(REPO / name), timeout=120,
        )
        assert result.returncode == 0, f"{name}/{module}.py failed to import:\n{result.stderr}"
        resolved = Path(result.stdout.strip())
        assert resolved == CANONICAL_DIR / f"{module}.py", (
            f"{name}/{module}.py resolved to {resolved}, not the canonical engine"
        )


def test_the_safety_target_is_the_same_in_every_phase():
    """
    PC_SAFE and PC_THRESHOLD must not vary by directory.

    phase10 had drifted to PC_SAFE = 1e-7. A safety target that depends on which
    folder the code was run from is not a safety target.
    """
    values = {}
    for name in _existing("phase8_maneuver", MANEUVER_DIRS) + ["phase11"]:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(f"""
                import sys
                sys.path.insert(0, {str(REPO / name)!r})
                import phase8_maneuver as m
                print(m.PC_SAFE, m.PC_THRESHOLD)
            """)],
            capture_output=True, text=True, cwd=str(REPO / name), timeout=120,
        )
        assert result.returncode == 0, result.stderr
        values[name] = result.stdout.strip()
    assert len(set(values.values())) == 1, f"safety thresholds differ by directory: {values}"


# ================================================ the defects, through a phase dir
@pytest.fixture(scope="module")
def engine():
    """The engine as phase7/ sees it."""
    sys.path.insert(0, str(REPO / "phase7"))
    try:
        spec = importlib.util.spec_from_file_location(
            "_shimmed_phase7_collision", REPO / "phase7" / "phase7_collision.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(REPO / "phase7"))


R1 = np.array([7000.0, 0.0, 0.0])
V1 = np.array([0.0, 7.5, 0.0])
GOOD_COV = np.diag([0.05, 0.05, 0.05]) ** 2


def test_a_closest_approach_in_the_past_is_reported_where_it_happened(engine):
    """
    The one-sided clamp reported the sampled separation whenever the approach had
    already happened. Here the pair passed at 0 km 300 s ago and is now 3 km
    apart; the stale engine called that a 3 km miss.
    """
    r2 = R1 + np.array([0.0, 0.0, 3.0])
    v2 = np.array([0.0, 7.5, 0.010])          # opening: the minimum is behind us
    tca, miss, _, _ = engine.find_tca(R1, V1, r2, v2, 3600.0)
    assert tca == pytest.approx(-300.0, abs=1.0)
    assert miss == pytest.approx(0.0, abs=1e-6)


def test_co_located_objects_do_not_produce_nan(engine):
    """nan > threshold is False, which reads as 'no action required'."""
    result = engine.assess_conjunction(R1, V1, GOOD_COV, R1.copy(), V1.copy(),
                                       GOOD_COV, 0.02)
    assert result["pc"] is None
    assert result["risk_level"] != "NOMINAL"


@pytest.mark.parametrize("bad,label", [
    (np.zeros((3, 3)), "all-zero"),
    (np.diag([2.5e-3, -2.5e-3, 2.5e-3]), "non-PSD"),
    (np.array([[2.5e-3, 9e-4, 0.0], [0.0, 2.5e-3, 0.0], [0.0, 0.0, 2.5e-3]]), "asymmetric"),
])
def test_an_invalid_covariance_yields_no_probability_rather_than_a_wrong_one(engine, bad, label):
    r2 = R1 + np.array([0.0, 0.0, 0.3])
    v2 = np.array([0.0, 7.5, -0.01])
    result = engine.assess_conjunction(R1, V1, bad, r2, v2, GOOD_COV, 0.02)
    assert result["pc"] is None, f"{label} covariance produced a Pc"
    assert result["covariance_valid"] is False
    assert result["risk_level"] == "DATA_INVALID"


# ================================================ the unavailable-Pc primitives
def test_an_unavailable_pc_fails_toward_caution(engine):
    """Unknown must escalate, never be read as zero."""
    assert engine.pc_for_safety(None) == 1.0
    assert engine.pc_for_safety(float("nan")) == 1.0
    assert engine.pc_for_safety(1e-9) == 1e-9

    threshold = 1e-4
    assert engine.pc_for_safety(None) >= threshold        # action required
    assert not engine.pc_for_safety(None) < 1e-6          # never "resolved"


def test_an_unavailable_pc_formats_instead_of_raising(engine):
    assert engine.pc_text(None) == "unavailable"
    assert engine.pc_text(float("nan")) == "unavailable"
    assert engine.pc_text(1.5e-4) == "1.50e-04"
    assert engine.pc_text(1.5e-4, ".3e") == "1.500e-04"
