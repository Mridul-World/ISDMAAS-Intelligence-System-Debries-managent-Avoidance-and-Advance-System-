"""
cdm_ingest.py  —  CCSDS Conjunction Data Message (CDM) ingestion for ISDMAAS.
================================================================================
WHAT THIS IS (honest scope):
A real parser for the CCSDS 508.0-B-1 Conjunction Data Message standard — the
format that satellite operators receive from the 18th Space Defense Squadron (and
commercial SSA providers) for real conjunctions. It extracts each object's state
and its REAL covariance, and converts that covariance into the same ECI 3x3 form
ISDMAAS's collision engine already uses.

WHY IT MATTERS:
ISDMAAS's internal Pc uses an *assumed* covariance model (fine for screening). When
a real CDM is available, this module lets the engine use the operator's *real*
covariance instead — turning the Pc from a screening estimate into an
operational-grade assessment. This is the bridge to a pilot: the moment an operator
provides CDMs, ISDMAAS uses them directly. No fabricated data — it processes the
actual standard, and runs on whatever real CDMs are supplied.

WHAT IT NEEDS:
Real CDM files (KVN .txt or XML). These come from an operator / 18th SDS in a pilot.
Without real CDMs there is nothing to ingest — by design. This module is the
*capability*; the pilot supplies the *data*.

SUPPORTS:
  - KVN (key-value notation, the common .txt CDM form)
  - XML (CCSDS NDM/XML CDM form)
Extracts: TCA, miss distance, relative speed, both objects' RTN covariance, and
converts to ECI for the engine. Handles the 6x6 RTN covariance (position block).
================================================================================
"""
import re
import numpy as np
import xml.etree.ElementTree as ET


# ----------------------------------------------------------------------------
# Covariance helpers
# ----------------------------------------------------------------------------
def _rtn_to_eci_rotation(r_eci, v_eci):
    """Build the 3x3 rotation from RTN (radial/transverse/normal) to ECI, from
    the object's ECI position and velocity. Matches ISDMAAS's convention."""
    r = np.asarray(r_eci, float)
    v = np.asarray(v_eci, float)
    R = r / np.linalg.norm(r)               # radial
    N = np.cross(r, v)
    N = N / np.linalg.norm(N)               # normal (orbit plane)
    T = np.cross(N, R)                       # transverse (completes RH set)
    # columns are RTN axes expressed in ECI
    return np.column_stack([R, T, N])


def rtn_cov_to_eci(cov_rtn_3x3, r_eci, v_eci):
    """Rotate a 3x3 RTN position covariance into ECI: C_eci = A C_rtn A^T."""
    A = _rtn_to_eci_rotation(r_eci, v_eci)
    return A @ np.asarray(cov_rtn_3x3, float) @ A.T


# ----------------------------------------------------------------------------
# KVN (key-value notation) CDM parsing
# ----------------------------------------------------------------------------
# CDM covariance keys (units in the CDM are m^2 / m^2/s etc.; we convert to km).
_KVN_COV_KEYS = [
    # position block (RTN) — the 6 unique terms of the symmetric 3x3
    "CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N",
]


def _kvn_blocks(text):
    """Split a KVN CDM into the header + the two object segments.
    Object segments begin at 'OBJECT = OBJECT1' / 'OBJECT = OBJECT2'."""
    # normalize line endings
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("COMMENT")]
    header, obj1, obj2 = [], [], []
    cur = header
    for ln in lines:
        m = re.match(r"OBJECT\s*=\s*OBJECT(\d)", ln, re.I)
        if m:
            cur = obj1 if m.group(1) == "1" else obj2
            cur.append(ln)
            continue
        cur.append(ln)
    return header, obj1, obj2


def _kvn_get(block, key):
    """Get the value for KEY = VALUE [UNIT] in a block (first match)."""
    for ln in block:
        m = re.match(rf"{re.escape(key)}\s*=\s*([^\[\s]+)", ln, re.I)
        if m:
            return m.group(1)
    return None


def _kvn_cov_rtn_km2(block):
    """Assemble the 3x3 RTN position covariance (km^2) from the KVN terms.
    CDM covariance position terms are in m^2; convert to km^2 (factor 1e-6)."""
    vals = {}
    for k in _KVN_COV_KEYS:
        v = _kvn_get(block, k)
        if v is None:
            return None
        vals[k] = float(v) * 1e-6   # m^2 -> km^2
    C = np.array([
        [vals["CR_R"], vals["CT_R"], vals["CN_R"]],
        [vals["CT_R"], vals["CT_T"], vals["CN_T"]],
        [vals["CN_R"], vals["CN_T"], vals["CN_N"]],
    ])
    return C


def _kvn_state_km(block):
    """Extract position (km) and velocity (km/s) from a KVN object block.
    CDM state X/Y/Z are in km, X_DOT/.. in km/s (CCSDS default)."""
    def g(*keys):
        for k in keys:
            v = _kvn_get(block, k)
            if v is not None:
                return float(v)
        return None
    x, y, z = g("X"), g("Y"), g("Z")
    vx, vy, vz = g("X_DOT"), g("Y_DOT"), g("Z_DOT")
    if None in (x, y, z, vx, vy, vz):
        return None, None
    return np.array([x, y, z]), np.array([vx, vy, vz])


def parse_cdm_kvn(text):
    """Parse a KVN-format CDM. Returns a dict with both objects' state + ECI
    covariance, plus TCA / miss / relative speed from the header."""
    header, b1, b2 = _kvn_blocks(text)

    def hdr(*keys):
        for k in keys:
            v = _kvn_get(header, k)
            if v is not None:
                return v
        return None

    tca = hdr("TCA")
    miss_m = hdr("MISS_DISTANCE")
    rel_spd = hdr("RELATIVE_SPEED")

    out = {
        "tca": tca,
        "miss_km": (float(miss_m) / 1000.0) if miss_m else None,   # m -> km
        "rel_speed_kms": (float(rel_spd) / 1000.0) if rel_spd else None,
        "objects": [],
    }
    for blk in (b1, b2):
        r, v = _kvn_state_km(blk)
        c_rtn = _kvn_cov_rtn_km2(blk)
        obj = {
            "name": _kvn_get(blk, "OBJECT_NAME"),
            "norad": _kvn_get(blk, "OBJECT_DESIGNATOR") or _kvn_get(blk, "CATALOG_NAME"),
            "r_eci_km": r, "v_eci_kms": v,
            "cov_rtn_km2": c_rtn,
            "cov_eci_km2": (rtn_cov_to_eci(c_rtn, r, v) if (c_rtn is not None and r is not None) else None),
        }
        out["objects"].append(obj)
    return out


# ----------------------------------------------------------------------------
# XML (CCSDS NDM/XML) CDM parsing
# ----------------------------------------------------------------------------
def parse_cdm_xml(text):
    """Parse an XML-format CCSDS CDM. Returns the same dict shape as the KVN parser."""
    # strip namespaces for simpler XPath
    text = re.sub(r'\sxmlns(:\w+)?="[^"]+"', "", text, count=0)
    root = ET.fromstring(text)

    def findtext(node, tag):
        el = node.find(f".//{tag}")
        return el.text.strip() if el is not None and el.text else None

    rel = root.find(".//relativeMetadataData")
    tca = findtext(rel, "TCA") if rel is not None else findtext(root, "TCA")
    miss = findtext(rel, "MISS_DISTANCE") if rel is not None else None
    spd = findtext(rel, "RELATIVE_SPEED") if rel is not None else None

    out = {
        "tca": tca,
        "miss_km": (float(miss) / 1000.0) if miss else None,
        "rel_speed_kms": (float(spd) / 1000.0) if spd else None,
        "objects": [],
    }
    for seg in root.findall(".//segment"):
        # state vector
        def gv(tag):
            v = findtext(seg, tag)
            return float(v) if v is not None else None
        x, y, z = gv("X"), gv("Y"), gv("Z")
        vx, vy, vz = gv("X_DOT"), gv("Y_DOT"), gv("Z_DOT")
        r = np.array([x, y, z]) if None not in (x, y, z) else None
        v = np.array([vx, vy, vz]) if None not in (vx, vy, vz) else None
        # covariance (RTN position block), m^2 -> km^2
        def gc(tag):
            val = findtext(seg, tag)
            return float(val) * 1e-6 if val is not None else None
        terms = {k: gc(k) for k in _KVN_COV_KEYS}
        if None not in terms.values():
            C = np.array([
                [terms["CR_R"], terms["CT_R"], terms["CN_R"]],
                [terms["CT_R"], terms["CT_T"], terms["CN_T"]],
                [terms["CN_R"], terms["CN_T"], terms["CN_N"]],
            ])
        else:
            C = None
        out["objects"].append({
            "name": findtext(seg, "OBJECT_NAME"),
            "norad": findtext(seg, "OBJECT_DESIGNATOR"),
            "r_eci_km": r, "v_eci_kms": v,
            "cov_rtn_km2": C,
            "cov_eci_km2": (rtn_cov_to_eci(C, r, v) if (C is not None and r is not None) else None),
        })
    return out


def parse_cdm(text):
    """Auto-detect KVN vs XML and parse. Returns the unified dict."""
    t = text.lstrip()
    if t.startswith("<"):
        return parse_cdm_xml(text)
    return parse_cdm_kvn(text)


# ----------------------------------------------------------------------------
# Bridge to the ISDMAAS Pc engine
# ----------------------------------------------------------------------------
def assess_from_cdm(text, hbr_km=0.020):
    """
    Parse a real CDM and assess it with ISDMAAS's collision engine using the
    CDM's REAL covariance (not the assumed model). Returns the engine's result
    dict, annotated with 'covariance_source': 'CDM (real)'.
    """
    from phase7_collision import assess_conjunction
    cdm = parse_cdm(text)
    if len(cdm["objects"]) < 2:
        raise ValueError("CDM did not contain two objects with state vectors.")
    a, b = cdm["objects"][0], cdm["objects"][1]
    for o in (a, b):
        if o["r_eci_km"] is None or o["cov_eci_km2"] is None:
            raise ValueError(f"CDM object '{o.get('name')}' missing state or covariance.")
    # CCSDS 508.0-B-1 defines the object state vectors as the states AT TCA, so
    # the geometry is already the conjunction geometry. Re-solving for a TCA here
    # would displace the states away from the epoch the operator's covariance was
    # computed for, pairing a moved position with an unmoved uncertainty.
    res = assess_conjunction(
        a["r_eci_km"], a["v_eci_kms"], a["cov_eci_km2"],
        b["r_eci_km"], b["v_eci_kms"], b["cov_eci_km2"],
        hbr_km, tca_already_refined=True,
    )
    res["covariance_source"] = "CDM (real)"
    res["cdm_tca"] = cdm["tca"]
    res["cdm_miss_km"] = cdm["miss_km"]
    res["primary_name"] = a["name"]
    res["secondary_name"] = b["name"]
    return res


# ----------------------------------------------------------------------------
# Self-test with a synthetic-but-format-correct sample CDM
# ----------------------------------------------------------------------------
_SAMPLE_KVN = """CCSDS_CDM_VERS = 1.0
CREATION_DATE = 2026-06-27T00:00:00.000
ORIGINATOR = 18 SDS
MESSAGE_ID = 000001
TCA = 2026-06-28T14:35:10.000
MISS_DISTANCE = 1090.0
RELATIVE_SPEED = 8200.0
OBJECT = OBJECT1
OBJECT_NAME = EOS-06
OBJECT_DESIGNATOR = 44636
X = 6878.0
Y = 0.0
Z = 0.0
X_DOT = 0.0
Y_DOT = 7.61
Z_DOT = 0.0
CR_R = 250000.0
CT_R = 0.0
CT_T = 1000000.0
CN_R = 0.0
CN_T = 0.0
CN_N = 250000.0
OBJECT = OBJECT2
OBJECT_NAME = COSMOS-2251 DEB
OBJECT_DESIGNATOR = 34427
X = 6878.0
Y = 1.09
Z = 0.0
X_DOT = 0.0
Y_DOT = -7.61
Z_DOT = 0.0
CR_R = 562500.0
CT_R = 0.0
CT_T = 12250000.0
CN_R = 0.0
CN_T = 0.0
CN_N = 562500.0
"""

if __name__ == "__main__":
    print("Parsing a format-correct sample CDM (KVN)...\n")
    cdm = parse_cdm(_SAMPLE_KVN)
    print(f"TCA: {cdm['tca']}")
    print(f"Miss: {cdm['miss_km']} km   Rel speed: {cdm['rel_speed_kms']} km/s\n")
    for o in cdm["objects"]:
        print(f"  {o['name']} (NORAD {o['norad']})")
        if o["cov_rtn_km2"] is not None:
            print(f"    RTN sigma (km): {np.sqrt(np.diag(o['cov_rtn_km2']))}")
    print("\nThis is the CAPABILITY. In a pilot, the operator supplies REAL CDMs and")
    print("ISDMAAS uses their real covariance directly via assess_from_cdm().")
    try:
        print("\nAssessing the sample CDM with the real-covariance engine...")
        res = assess_from_cdm(_SAMPLE_KVN)
        print(f"  Pc (from CDM covariance): {res['pc']:.3e}")
        print(f"  covariance_source: {res['covariance_source']}")
    except Exception as e:
        print(f"  (engine bridge needs phase7_collision present: {e})")
