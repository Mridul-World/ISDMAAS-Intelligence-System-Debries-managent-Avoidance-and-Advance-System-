"""
fix_celestrak.py  —  one-shot fix for the CelesTrak 403 / timeout.

Run this ONCE in phase11\:   python fix_celestrak.py

It does three things to debris_data.py (making a backup first):
  1. Sets a User-Agent string CelesTrak accepts (the 403 cause).
  2. Raises the request timeout 60 -> 120 s (the 'active' group is large).
  3. Verifies the fix by actually fetching a small debris group live.

Safe: it backs up debris_data.py to debris_data.py.bak before editing, and
only rewrites the UA assignment + timeout numbers. If anything looks wrong it
prints what it changed so you can review.
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

import re, shutil, sys, os

FILE = "debris_data.py"
GOOD_UA = '{"User-Agent": "ISDMAAS/1.0 (research; orbital-safety)"}'

if not os.path.exists(FILE):
    sys.exit(f"ERROR: {FILE} not found. Run this inside phase11\\.")

src = open(FILE, encoding="utf-8").read()
shutil.copy2(FILE, FILE + ".bak")
print(f"backed up {FILE} -> {FILE}.bak")

orig = src
changes = []

# --- 1. Fix the UA assignment (handles several common forms) -------------
# matches:  UA = {...}   /   UA = "..."   /   UA = '...'
ua_pat = re.compile(r'^(\s*UA\s*=\s*)(\{.*?\}|".*?"|\'.*?\')\s*$', re.M)
if ua_pat.search(src):
    old = ua_pat.search(src).group(2)
    src = ua_pat.sub(lambda m: m.group(1) + GOOD_UA, src, count=1)
    changes.append(f"UA: {old}  ->  {GOOD_UA}")
else:
    # no UA variable found — inject one near the top (after imports)
    lines = src.splitlines()
    inject_at = 0
    for i, ln in enumerate(lines[:40]):
        if ln.startswith("import ") or ln.startswith("from "):
            inject_at = i + 1
    lines.insert(inject_at, f"UA = {GOOD_UA}  # added by fix_celestrak.py")
    src = "\n".join(lines)
    changes.append(f"UA variable was missing -> injected {GOOD_UA}")

# --- 2. Raise timeouts 60 -> 120 -----------------------------------------
def bump_timeout(m):
    return m.group(0).replace(m.group(1), "120")
to_pat = re.compile(r'timeout\s*=\s*(\d+)')
n_to = 0
def _bump(m):
    global n_to
    if int(m.group(1)) < 120:
        n_to += 1
        return f"timeout={120}"
    return m.group(0)
src = to_pat.sub(_bump, src)
if n_to:
    changes.append(f"raised {n_to} request timeout(s) to 120 s")

# --- write it back -------------------------------------------------------
if src != orig:
    open(FILE, "w", encoding="utf-8").write(src)
    print("\nApplied changes:")
    for c in changes:
        print("  -", c)
else:
    print("\nNo changes needed (UA and timeouts already look fine).")

# --- 3. Verify with a live fetch -----------------------------------------
print("\nVerifying with a live CelesTrak fetch...")
try:
    import importlib, requests
    # reload the patched module
    sys.modules.pop("debris_data", None)
    import debris_data
    importlib.reload(debris_data)
    r = requests.get(debris_data.CELESTRAK_GP,
                     params={"GROUP": "cosmos-2251-debris", "FORMAT": "JSON"},
                     headers=debris_data.UA, timeout=120)
    print(f"  HTTP {r.status_code}, {len(r.content)} bytes")
    if r.status_code == 200:
        print("  SUCCESS — CelesTrak now accepts the request. SOCRATES/live data will work.")
    elif r.status_code == 403:
        print("  STILL 403 — CelesTrak rejected even the clean UA. This is unusual;")
        print("  paste this output back and we'll try an alternate UA / endpoint.")
    else:
        print(f"  Unexpected status {r.status_code} — paste this output back.")
except Exception as e:
    print(f"  fetch error: {type(e).__name__}: {e}")
    print("  (If this is a timeout, your network may be slow — try again once.)")

print("\nDone. Restart the API (uvicorn) and reload the dashboard.")
print("If anything broke, restore with:  copy debris_data.py.bak debris_data.py")
