"""
fix_all.py — one-shot diagnose + fix for the three dashboard issues.
=====================================================================
Run in phase11\:   python fix_all.py

Fixes/diagnoses:
  (A) "Catalog unavailable"  -> builds catalog_cache.json slowly (rate-limit safe)
  (B) click-to-load          -> confirms dashboard.html has loadScreenResult
  (C) telemetry vanishes     -> confirms reload guard is present; checks API is up
Nothing here is destructive. It reports PASS/FAIL per item.
"""

# ---------------------------------------------------------------------------
# ARCHIVED — DO NOT RUN.
#
# This is a one-shot script that rewrites source files in place by string
# substitution. Every fix it applied is now permanent in the code, and the
# patterns it searches for no longer exist. Running it today would either do
# nothing or corrupt a file.
#
# Scripts of this shape already caused one production incident: a run of
# add_login.py appended `install_auth(app)` to phase11_api.py, a later script
# rewrote that file from a stale .bak, and the wiring was silently lost — the
# entire authentication surface returned 404 while the console still showed a
# sign-in dialog. It stayed broken because nothing tested it.
#
# It is kept for the historical record of what was changed and why. If you need
# to change the code, edit the code and add a test.
# ---------------------------------------------------------------------------
import sys as _sys

print(__doc__)
print("ARCHIVED: this patch script is disabled. See the note at the top of "
      "this file, and phase11/README.md for the current layout.")
_sys.exit(1)

import os, sys, json, time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "catalog_cache.json")
DASH = os.path.join(HERE, "dashboard.html")

def line(): print("-" * 60)

# ============================================================ (A) CATALOG
def build_catalog():
    print("\n[A] CATALOG CACHE")
    line()
    try:
        import requests
    except ImportError:
        print("  FAIL: 'requests' not installed. Run: pip install requests")
        return False
    groups = ["active", "cosmos-2251-debris", "iridium-33-debris",
              "fengyun-1c-debris", "cosmos-1408-debris"]
    objs = {}
    ok_any = False
    for g in groups:
        url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={g}&FORMAT=tle"
        try:
            r = requests.get(url, headers={"User-Agent": "ISDMAAS/1.0"}, timeout=90)
            lines = [x.strip() for x in r.text.splitlines() if x.strip()]
            n0 = len(objs)
            for i in range(0, len(lines) - 2, 3):
                l1, l2 = lines[i + 1], lines[i + 2]
                if l1.startswith("1 ") and l2.startswith("2 "):
                    objs[l1[2:7].strip()] = {"name": lines[i], "tle1": l1,
                                             "tle2": l2, "group": g}
            print(f"  {g:22s} OK  (+{len(objs)-n0}, total {len(objs)})")
            ok_any = True
        except Exception as e:
            print(f"  {g:22s} FAILED: {str(e)[:50]}")
        time.sleep(3)   # <-- rate-limit safety (this is what was missing)
    if not ok_any:
        print("\n  FAIL: could not reach CelesTrak at all.")
        print("  -> Check VPN/firewall; try a different network. Test with:")
        print("     python -c \"import requests;print(requests.get('https://celestrak.org').status_code)\"")
        return False
    json.dump({"fetched_utc": datetime.now(timezone.utc).isoformat(),
               "objects": objs}, open(CACHE, "w"))
    print(f"\n  PASS: wrote catalog_cache.json with {len(objs)} objects")
    return True

# ============================================================ (B)+(C) DASHBOARD
def check_dashboard():
    print("\n[B/C] DASHBOARD FILE")
    line()
    if not os.path.exists(DASH):
        print("  FAIL: dashboard.html not found in this folder.")
        return False
    html = open(DASH, encoding="utf-8", errors="ignore").read()
    checks = {
        "(B) click-to-load function": "function loadScreenResult" in html,
        "(B) cards are clickable": "click to load into simulator" in html,
        "(C) reload guard present": "RELOAD GUARD" in html,
        "(C) buttons forced non-submit": 'setAttribute("type","button")' in html,
        "login system present": "function doLogin" in html,
        "screening function present": "function screenMySat" in html,
        "no duplicate screenMySat": html.count("function screenMySat") == 1,
        "no duplicate loadScreenResult": html.count("function loadScreenResult") == 1,
    }
    allok = True
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
        if not v: allok = False
    if not allok:
        print("\n  -> dashboard.html is an OLD version. Replace it with the latest")
        print("     file from the chat, then HARD REFRESH the browser (Ctrl+Shift+R).")
    else:
        print("\n  PASS: dashboard.html has all fixes. If issues persist in the")
        print("        browser, it's CACHE -> open in Incognito (Ctrl+Shift+N).")
    return allok

# ============================================================ API LIVE CHECK
def check_api():
    print("\n[API] BACKEND LIVE CHECK")
    line()
    try:
        import requests
        r = requests.get("http://localhost:8000/health", timeout=5)
        print(f"  PASS: API up (/health -> {r.status_code})")
        # check the screening route exists
        o = requests.get("http://localhost:8000/openapi.json", timeout=5).json()
        paths = list(o.get("paths", {}).keys())
        for p in ["/user/screen/{norad}", "/my/satellites", "/user/assess", "/auth/login"]:
            print(f"  {'PASS' if p in paths else 'FAIL'}  route {p}")
        return True
    except Exception as e:
        print(f"  FAIL: API not reachable at localhost:8000 ({str(e)[:40]})")
        print("  -> Start it:  uvicorn phase11_api:app --port 8000")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ISDMAAS — FIX ALL (catalog + dashboard + API)")
    print("=" * 60)
    a = build_catalog()
    bc = check_dashboard()
    api = check_api()
    print("\n" + "=" * 60)
    print("SUMMARY")
    line()
    print(f"  (A) catalog cache : {'FIXED' if a else 'NEEDS ATTENTION'}")
    print(f"  (B/C) dashboard   : {'OK' if bc else 'REPLACE FILE + HARD REFRESH'}")
    print(f"  API backend       : {'UP' if api else 'START UVICORN'}")
    print("=" * 60)
    print("\nNEXT:")
    if not api:
        print("  1. Start API:  uvicorn phase11_api:app --port 8000")
    if a and bc:
        print("  2. Open the dashboard in INCOGNITO (Ctrl+Shift+N) to bypass cache")
        print("  3. Log in -> Screen -> click a result -> Assess")
    print("  If screening still says 'Catalog unavailable' after this ran PASS,")
    print("  the API process is using an OLD cache in memory -> RESTART uvicorn.")
