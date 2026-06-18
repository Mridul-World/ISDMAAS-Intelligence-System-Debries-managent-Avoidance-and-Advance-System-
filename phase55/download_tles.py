"""
Phase 5.5 — download historical TLEs for the whole fleet from Space-Track.

    set SPACETRACK_USER=...   set SPACETRACK_PASS=...
    python download_tles.py

Output: data/tle/tle_<norad>.csv   (epoch, tle1, tle2)
Idempotent: existing files are skipped (delete a file to re-download).
"""
import os, sys, csv, time
import requests
from config import CFG, SATELLITES, tle_csv
from phase55_utils import tle_epoch

BASE = "https://www.space-track.org"


def main():
    user, pw = os.environ.get("SPACETRACK_USER"), os.environ.get("SPACETRACK_PASS")
    if not (user and pw):
        sys.exit("Set SPACETRACK_USER and SPACETRACK_PASS environment variables.")
    os.makedirs(CFG["TLE_DIR"], exist_ok=True)

    s = requests.Session()
    r = s.post(f"{BASE}/ajaxauth/login", data={"identity": user, "password": pw}, timeout=60)
    r.raise_for_status()
    if "Failed" in r.text:
        sys.exit("Space-Track login failed.")

    for norad, info in SATELLITES.items():
        out = tle_csv(norad)
        if os.path.exists(out):
            print(f"skip (exists): {out}"); continue
        q = (f"{BASE}/basicspacedata/query/class/gp_history/"
             f"NORAD_CAT_ID/{norad}/"
             f"EPOCH/{CFG['DATE_START']}--{CFG['DATE_END']}/"
             f"orderby/EPOCH asc/format/json")
        print(f"[{info['name']}] querying Space-Track ...")
        rows = s.get(q, timeout=300).json()
        if not rows:
            print(f"  WARN: no TLEs for {norad}"); continue
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch", "tle1", "tle2"])
            for rec in rows:
                t1, t2 = rec["TLE_LINE1"], rec["TLE_LINE2"]
                w.writerow([tle_epoch(t1).isoformat(), t1, t2])
        print(f"  saved {len(rows)} TLEs -> {out}")
        time.sleep(3)   # Space-Track rate-limit etiquette


if __name__ == "__main__":
    main()
