"""
Phase 5.5 — download AUX_POEORB precise orbits for the whole fleet from the
Copernicus Data Space Ecosystem, plus daily F10.7/Kp from CelesTrak.

    set CDSE_USER=...   set CDSE_PASS=...      (free: dataspace.copernicus.eu)
    python download_precise_orbits.py

Outputs:
    data/precise_orbit/<PREFIX>/*.EOF          (per satellite)
    data/space_weather_daily.csv
Idempotent: already-downloaded files are skipped. Satellites with no POEORB
products in the CDSE catalogue are reported and skipped.

Fallback for Sentinel-1: pip install sentineleof
"""
import os, sys, io, csv, zipfile
from datetime import timedelta
import requests
from config import CFG, SATELLITES, parse_date

TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
             "protocol/openid-connect/token")
CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products({pid})/$value"
CELESTRAK_SW = "https://celestrak.org/SpaceData/SW-All.csv"


def get_token(user, pw):
    r = requests.post(TOKEN_URL, data={
        "client_id": "cdse-public", "grant_type": "password",
        "username": user, "password": pw}, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def query_products(prefix, d0, d1):
    flt = (f"contains(Name,'AUX_POEORB') and startswith(Name,'{prefix}') and "
           f"ContentDate/Start ge {d0:%Y-%m-%dT00:00:00.000Z} and "
           f"ContentDate/Start le {d1:%Y-%m-%dT00:00:00.000Z}")
    products, url = [], f"{CATALOG}?$filter={flt}&$top=200&$orderby=ContentDate/Start asc"
    while url:
        j = requests.get(url, timeout=120).json()
        products += j.get("value", [])
        url = j.get("@odata.nextLink")
    return products


def fetch(pid, hdr):
    return requests.get(DOWNLOAD.format(pid=pid), headers=hdr,
                        timeout=300, allow_redirects=True)


def download_fleet():
    user, pw = os.environ.get("CDSE_USER"), os.environ.get("CDSE_PASS")
    if not (user and pw):
        sys.exit("Set CDSE_USER and CDSE_PASS environment variables.")
    d0 = parse_date(CFG["DATE_START"]) - timedelta(days=1)
    d1 = parse_date(CFG["DATE_END"]) + timedelta(days=1)
    token = get_token(user, pw); hdr = {"Authorization": f"Bearer {token}"}

    for norad, info in SATELLITES.items():
        out_dir = os.path.join(CFG["POE_DIR"], info["prefix"])
        os.makedirs(out_dir, exist_ok=True)
        prods = query_products(info["prefix"], d0, d1)
        if not prods:
            print(f"[{info['name']}] no AUX_POEORB in CDSE catalogue — SKIPPED "
                  f"(model trains on the satellites that have truth)")
            continue
        print(f"[{info['name']}] {len(prods)} POEORB products")
        for p in prods:
            name = p["Name"]
            eof_name = name if name.upper().endswith(".EOF") else name + ".EOF"
            dest = os.path.join(out_dir, os.path.basename(eof_name))
            if os.path.exists(dest):
                continue
            r = fetch(p["Id"], hdr)
            if r.status_code == 401:                      # token expired mid-run
                token = get_token(user, pw); hdr = {"Authorization": f"Bearer {token}"}
                r = fetch(p["Id"], hdr)
            if r.status_code != 200:
                print(f"  WARN {r.status_code} on {name}"); continue
            data = r.content
            if data[:2] == b"PK":                         # zipped container
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    inner = [n for n in z.namelist() if n.upper().endswith(".EOF")]
                    if not inner:
                        print(f"  WARN no .EOF inside {name}"); continue
                    with z.open(inner[0]) as fh, open(dest, "wb") as outf:
                        outf.write(fh.read())
            else:
                with open(dest, "wb") as outf:
                    outf.write(data)
        n = len([f for f in os.listdir(out_dir) if f.upper().endswith(".EOF")])
        print(f"  {n} EOF files in {out_dir}")


def download_space_weather():
    out = CFG["SW_CSV"]
    if os.path.exists(out):
        print(f"skip (exists): {out}"); return
    print("Downloading CelesTrak space weather ...")
    r = requests.get(CELESTRAK_SW, timeout=120); r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.text)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "f107", "kp"])
        for rec in rows:
            try:
                f107 = float(rec.get("F10.7_OBS") or rec.get("F10.7_ADJ") or "nan")
                kps = [float(rec[f"KP{i}"]) for i in range(1, 9)
                       if rec.get(f"KP{i}") not in (None, "")]
                kp = (sum(kps) / len(kps)) / 10.0 if kps else float("nan")
                w.writerow([rec["DATE"], f107, kp])
            except (ValueError, KeyError):
                continue
    print("saved:", out)


if __name__ == "__main__":
    download_fleet()
    download_space_weather()
