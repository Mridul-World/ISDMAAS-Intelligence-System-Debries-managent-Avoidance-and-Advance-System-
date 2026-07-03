"""
socrates_tca_probe.py  —  shows exactly where TCA lives in the SOCRATES HTML.
Run in phase11\:   python socrates_tca_probe.py
Then paste me the entire output.
"""
import requests, re

url = "https://celestrak.org/SOCRATES/table-socrates.php"
params = {"NAME": ",", "ORDER": "MAXPROB", "MAX": "3"}
html = requests.get(url, params=params,
                    headers={"User-Agent": "ISDMAAS/1.0"}, timeout=60).text

# split into rows, then cells, stripping tags
rows = []
for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)
    clean = [re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").strip() for c in cells]
    rows.append(clean)

# data rows = those containing a date like "2026 Jun 25" OR an ISO datetime
def has_date(cells):
    s = " ".join(cells)
    return bool(re.search(r"\d{4}\s+\w{3}\s+\d{1,2}", s) or
                re.search(r"\d{4}-\d{2}-\d{2}", s))

datarows = [r for r in rows if has_date(r)]

print("=" * 60)
print(f"Total <tr> rows: {len(rows)}")
print(f"Rows containing a date (likely TCA rows): {len(datarows)}")
print("=" * 60)

if not datarows:
    # fall back: show rows that have a 5-6 digit NORAD id
    print("No date-rows found. Showing rows with a NORAD id instead:")
    norad_rows = [r for r in rows if any(re.fullmatch(r"\d{5,6}", c) for c in r)]
    for i, r in enumerate(norad_rows[:6]):
        print(f"\nNORAD-ROW {i} ({len(r)} cells):")
        for j, c in enumerate(r):
            print(f"   [{j}] {c!r}")
else:
    for i, r in enumerate(datarows[:5]):
        print(f"\nDATA-ROW {i} ({len(r)} cells):")
        for j, c in enumerate(r):
            print(f"   [{j}] {c!r}")
print("\n" + "=" * 60)
print("Paste everything above.")
