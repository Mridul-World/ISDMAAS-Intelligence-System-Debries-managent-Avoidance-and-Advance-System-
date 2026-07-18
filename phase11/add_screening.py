"""
add_screening.py — adds "Screen vs full catalog" to the console + wires the API.
Run in phase11\ AFTER add_login.py:   python add_screening.py
"""
import os, shutil, sys

CONSOLE, APIFILE = "dashboard.html", "phase11_api.py"
for f in (CONSOLE, APIFILE, "user_screening.py", "ops_db.py"):
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} missing — run inside phase11\\ after installing the modules.")

shutil.copy2(CONSOLE, CONSOLE + ".bak2")
html = open(CONSOLE, encoding="utf-8").read()
results = []

def patch(name, old, new):
    global html
    n = html.count(old)
    if n != 1:
        results.append((name, f"FAILED (anchor x{n})")); return
    html = html.replace(old, new, 1)
    results.append((name, "OK"))

# S1: button in the My Satellites card
patch("S1 screen button",
"""    <button class="ghost" onclick="addMySat()">Add my satellite</button>""",
"""    <button class="ghost" onclick="addMySat()">Add my satellite</button>
    <button class="cyan" onclick="screenMySat()">&#128225; Screen my satellite vs full catalog</button>
    <div class="muted" style="margin-top:5px;font-size:9.5px">ISDMAAS's own SOCRATES-style
      screen: every close approach in the public catalog for your asset \u00b7 TCA \u00b7 miss \u00b7 Pc.
      Full-catalog scans take up to ~1 min of local computation.</div>""")

# S2: JS — screen + render
patch("S2 screening JS",
"""function selectUserSat(n){document.getElementById("sat").value=n;loadOrbit();}""",
"""function selectUserSat(n){document.getElementById("sat").value=n;loadOrbit();}
async function screenMySat(){
  const mine=Object.keys(userSatSet).filter(n=>userSatSet[n].mine);
  if(!mine.length){alert("Log in and register a satellite first.");return;}
  let norad=document.getElementById("sat").value;
  if(!userSatSet[norad]||!userSatSet[norad].mine)norad=mine[0];
  document.getElementById("telemetry").style.display="flex";
  const out=document.getElementById("result");
  out.innerHTML='<div class="loading"><span class="spin"></span>Screening '+userSatSet[norad].name+
    ' against the full public catalog\\u2026 (local computation, up to ~1 min)</div>';
  try{
    const r=await fetch(API+"/user/screen/"+norad+"?hours=24&max_results=10");
    const d=await r.json();
    if(!r.ok){out.innerHTML='<div class="err">'+(d.detail||"screen failed")+'</div>';return;}
    let h='<div class="kv"><span class="k">Asset</span><span class="v">'+d.primary.name+'</span></div>';
    h+='<div class="kv"><span class="k">Objects screened</span><span class="v">'+d.objects_screened+'</span></div>';
    h+='<div class="kv"><span class="k">Scan time</span><span class="v">'+d.scan_seconds+' s (local)</span></div>';
    if(!d.conjunctions.length){
      h+='<div class="decblock monitor" style="margin-top:8px"><div class="decrow">'+
         '<span class="deck">Result</span><span class="decv monitor-t">CLEAR</span></div>'+
         '<div class="decrow"><span class="deck">Detail</span><span class="decv2">no close approaches '+
         'above reporting gates in the next '+d.window_hours+' h</span></div></div>';
    } else {
      h+='<div class="hd" style="margin:10px 0 6px">Close approaches (next '+d.window_hours+' h)</div>';
      for(const c of d.conjunctions){
        h+='<div class="conj"><div class="cn">'+d.primary.name+' \\u00d7 '+c.secondary_name+
           ' <span class="pill '+c.risk+'" style="font-size:8px">'+c.risk+'</span></div>'+
           '<div class="cm"><span>miss <b>'+c.miss_km+' km</b></span>'+
           '<span>Pc '+Number(c.pc).toExponential(1)+'</span>'+
           '<span>v '+c.rel_speed_kms+' km/s</span></div>'+
           '<div class="cm"><span>TCA in '+c.tca_in_hours+' h</span>'+
           '<span>'+c.group+'</span></div></div>';
      }
    }
    h+='<div class="muted" style="margin-top:8px">Run stored in the operational dataset '+
       '(screening_runs). Pc from the documented TLE-scale covariance model; real CDM '+
       'covariance is used when supplied.</div>';
    out.innerHTML=h;
  }catch(e){out.innerHTML='<div class="err">'+e+'</div>';}
}""")

open(CONSOLE, "w", encoding="utf-8").write(html)

api = open(APIFILE, encoding="utf-8").read()
if "install_screening" not in api:
    api += ("\nfrom user_screening import install_screening\n"
            "install_screening(app)\n")
    open(APIFILE, "w", encoding="utf-8").write(api)
    results.append(("S3 API wiring", "OK (appended)"))
else:
    results.append(("S3 API wiring", "already present"))

print("Patch results:")
ok = True
for n, r in results:
    print(f"  {n}: {r}")
    if "FAILED" in r: ok = False
print("\n" + ("DONE — restart uvicorn, reload console." if ok else
      "Anchors failed — run add_login.py first, or paste this output back."))
