# Maneuver Validation

**Date:** 2026-08-31 · **Suite:** `phase11/tests/scientific/test_maneuver_validation.py`
**Verdict: PASS** — 37 tests.

---

## 1. What has to be true

A planner that returns a delta-v has proved nothing. The chain that must hold is:

```
CURRENT ORBIT -> BURN -> POST-BURN STATE -> PROPAGATION ->
NEW CONJUNCTION SEARCH -> NEW TCA -> NEW MISS -> NEW Pc -> SAFETY CHECK
```

Each link is validated below, and the first-order model the planner uses is
validated against an independent numerical propagation of the actual burned
orbit.

---

## 2. Reference method

RK4 integration of the two-body equations at 20 000 steps. Independent of the
Clohessy-Wiltshire model under test: no linearization, no rotating frame, no
shared code.

The comparison is made in the correct frame — displacement relative to the
**propagated nominal**, not to the burn point — because the CW response is
expressed in the local orbital frame at the *evaluation* epoch. Comparing against
the burn point instead produces vector errors of 30-200% for a correct
implementation, which is a frame mistake in the test, not a defect in the model.

---

## 3. CW model vs numerical propagation

| dv (m/s) | Orbits | Model (km) | Truth (km) | Error (m) | Error (%) |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.05 | 0.0146 | 0.0146 | 0.00 | 0.018 |
| 0.05 | 0.25 | 0.0999 | 0.0998 | 0.11 | 0.108 |
| 0.05 | 0.50 | 0.4818 | 0.4818 | 0.09 | 0.020 |
| 0.05 | 1.00 | 0.8870 | 0.8863 | 0.77 | 0.087 |
| 0.05 | 3.00 | 2.6611 | 2.6588 | 2.35 | 0.089 |
| 0.50 | 0.05 | 0.1456 | 0.1456 | 0.03 | 0.018 |
| 0.50 | 0.25 | 0.9992 | 0.9985 | 1.10 | 0.110 |
| 0.50 | 0.50 | 4.8180 | 4.8173 | 1.92 | 0.040 |
| 0.50 | 1.00 | 8.8702 | 8.8647 | 7.85 | 0.089 |
| 0.50 | 3.00 | 26.6106 | 26.5940 | 52.73 | 0.198 |
| 2.00 | 0.05 | 0.5825 | 0.5825 | 0.10 | 0.018 |
| 2.00 | 0.25 | 3.9969 | 3.9936 | 4.69 | 0.118 |
| 2.00 | 0.50 | 19.2720 | 19.2640 | 20.59 | 0.107 |
| 2.00 | 1.00 | 35.4808 | 35.4869 | 89.25 | 0.251 |
| 2.00 | 3.00 | 106.4424 | 106.4599 | 801.48 | 0.753 |

**Worst error across the operational envelope: 0.75%**, against a 2% threshold.

### Threshold rationale

CW is a first-order linearization; its error grows as `(dv/v)^2` and with the
number of orbits. A single fixed tolerance would be meaningless for small burns
and wrong for large ones, so the bound is **2% of the displacement magnitude**
over the envelope the planner actually operates in (sub-m/s burns, lead times up
to a few orbits).

Outside that envelope the model is documented as degrading, not asserted
accurate: `test_cw_model_degrades_predictably_outside_its_envelope` pins a 5 m/s
burn over 12 orbits to a *visible but bounded* error, so a future change that
makes the model silently worse still fails.

---

## 4. Burn direction semantics

These four tests exist because the original implementation had the sign
backwards — it displaced the satellite *forwards* for a prograde burn. Because
the search tried both signs the magnitude came out similar, so the defect was
invisible in the delta-v; only the reported **direction** was wrong, and an
operator following it would have burned the wrong way.

| Property | Assertion |
|---|---|
| Prograde burn raises the orbit, lengthens the period | satellite falls **behind** (along-track < 0) |
| Retrograde burn | satellite moves **ahead** (along-track > 0) |
| Radial response at the burn instant | exactly 0 — the satellite does not teleport to its new altitude |
| Radial response half an orbit later | `4 dv / n`, the peak |
| Radial response one orbit later | back to 0 |
| Model is odd in dv | `response(+dv) == -response(-dv)` to 1e-12 |
| Post-burn velocity | carries the delta-v (previously returned unchanged) |
| Post-burn orbit energy | increases; `da = 2 dv / n` to 0.1% |

---

## 5. Planner objective — a defect found and fixed

The planner filtered options to those reaching `PC_SAFE`, then picked minimum
delta-v. That filter is legitimate — every option genuinely solves the problem
by construction and is re-verified by the safety gate — but two things were
missing.

### 5.1 Operationally infeasible lead times were recommended as best

The lead-time floor was **one minute**. For a short-notice conjunction the
planner returned a burn scheduled 2.5 minutes out as its recommendation, ranked
first because it was cheapest. No operations desk can plan, independently verify,
schedule, uplink and confirm an avoidance maneuver in 2.5 minutes; presenting it
as the answer is presenting an unexecutable plan.

`OPERATIONAL_MIN_LEAD_H = 0.5` (30 minutes) now separates *physically possible*
from *operationally executable*. Short-notice options are still computed and
still shown — an operator facing a late detection needs to see what the physics
would allow — but they are marked `operationally_feasible: false` and are never
the recommendation unless nothing else exists, in which case the recommendation
carries an explicit escalation warning.

Measured, 11 minutes to TCA:

```
options: [(0.048 h, 11.663 m/s, infeasible), (0.095 h, 4.867, infeasible),
          (0.142 h, 2.693, infeasible), (0.171 h, 2.020, infeasible)]
recommendation: 2.020 m/s at T-0.171 h, operationally_feasible = False
warning: "No operationally feasible option exists ... Escalate rather than
          treating this as a routine recommendation."
```

### 5.2 Robustness was not surfaced

Two burns that both reach the safety target are not equivalent — one may land
just under the threshold and the other far below it. The recommendation now
reports `pc_safety_margin` (`PC_SAFE / achieved Pc`) so an operator choosing
between similar burns sees robustness, not only propellant.

---

## 6. Post-burn chain

| Test | What it establishes |
|---|---|
| `test_every_option_actually_reaches_the_safety_target` | No option is presented merely because its delta-v is small; each independently reaches `PC_SAFE` |
| `test_earlier_burns_cost_less_propellant` | The trade table is monotonic — when every row showed the same delta-v, that was the constant-radial defect |
| `test_the_burn_actually_improves_the_conjunction` | Pc before and after, same engine: Pc falls and miss distance grows |
| `test_a_burn_in_the_wrong_direction_is_detected_as_worsening` | The safety gate measures the real post-burn geometry, so a sign error cannot pass |
| `test_orbit_safety_rejects_a_burn_that_leaves_the_mission_band` | A 200 m/s burn is rejected on semi-major-axis drift |
| `test_an_unsolvable_conjunction_reports_escalation_not_a_fake_burn` | One minute to TCA at 1 m miss returns escalation, never a delta-v that does not solve the problem |
| `test_zero_delta_v_is_never_presented_as_a_solution` | No option has `dv == 0` |

The post-burn re-screen itself (against the full catalog, using a time-shifted
SGP4 trajectory rather than a linear extrapolation) is exercised end to end in
`tests/e2e/`.

---

## 7. A defect in this suite, recorded

`test_a_burn_in_the_wrong_direction_is_detected_as_worsening` initially failed on
its own premise. It applied a round 0.05 m/s burn against a 2 km separation,
expecting the primary to drift toward the secondary. Over 8 hours that burn
drifts 4.3 km — it overshoots the threat and ends up *further away*
(2.0 -> 2.46 km).

The burn is now sized from the geometry (`dv = separation / (2 * 3 * t)`) so it
covers half the gap. The production code was correct throughout; the test's
arithmetic was not.

---

## 8. Known limitations, stated plainly

- **Only tangential burns are searched.** Radial and cross-track burns are
  modelled correctly by `cw_impulse_response` and are validated here, but
  `solve_min_dv` searches the along-track axis only. That is the right default —
  tangential is far the most propellant-efficient way to change along-track
  position — but a cross-track burn is sometimes preferable for a high-inclination
  crossing geometry, and the planner will not find it.
- **Single-burn plans only.** No multi-burn or return-to-station sequences.
- **No operational constraint model** beyond lead time and propellant: attitude
  constraints, thruster duty cycles, ground-station contact windows and payload
  outage windows are not represented.
- **The two-body reference** used for validation excludes J2 and drag, which is
  appropriate for validating the *linearization* over hours but means these
  numbers are not a statement about absolute propagation accuracy.

---

## 9. Reproducing

```bash
cd phase11
python -m pytest tests/scientific/test_maneuver_validation.py -v
```

Runtime ~40 s, dominated by the RK4 reference propagations.
