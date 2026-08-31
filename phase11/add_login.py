"""
add_login.py — patches console.html + phase11_api.py for the multi-operator demo.
===================================================================================
Adds exactly the meeting-agreed flow (Jun 30):
  1. LOGIN PAGE: operator login overlay on startup (with demo credentials shown)
  2. MY SATELLITES card: each operator registers their own TLEs
  3. OWNER AUTHORITY: assess the two-satellite conjunction; planning a maneuver
     for a satellite you don't own is refused with a visible 403 SECURITY block
  4. Wires auth_store.py + user_conjunctions.py into phase11_api.py

Run in phase11\ (with console.html, phase11_api.py, auth_store.py,
user_conjunctions.py all present):
    python add_login.py
Backs up both files first. Reports each patch as OK/FAILED.
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

import os, shutil, sys

CONSOLE = "dashboard.html"
APIFILE = "phase11_api.py"

for f in (CONSOLE, APIFILE, "auth_store.py", "user_conjunctions.py"):
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} not found — run inside phase11\\ with all four files present.")

shutil.copy2(CONSOLE, CONSOLE + ".bak")
shutil.copy2(APIFILE, APIFILE + ".bak")
print(f"backed up {CONSOLE} and {APIFILE} (.bak)")

html = open(CONSOLE, encoding="utf-8").read()
patches = []

def patch(name, old, new):
    global html
    n = html.count(old)
    if n != 1:
        patches.append((name, f"FAILED (anchor found {n}x)"))
        return
    html = html.replace(old, new, 1)
    patches.append((name, "OK"))

# ---- P1: CSS for the login overlay + operator badge ----
patch("P1 login CSS",
"""</style>""",
"""  #loginov{position:fixed;inset:0;z-index:100;display:none;align-items:center;justify-content:center;
    background:rgba(4,6,13,.88);backdrop-filter:blur(10px)}
  .loginbox{background:var(--panel);border:1px solid var(--line2);border-radius:14px;
    padding:26px 28px;width:340px}
  .loginbox h2{font-size:16px;margin-bottom:4px}
  .loginbox .sub{font-size:10px;color:var(--dim);letter-spacing:1px;text-transform:uppercase;margin-bottom:16px}
  .democreds{margin-top:14px;background:rgba(57,212,232,.06);border:1px solid rgba(57,212,232,.25);
    border-radius:8px;padding:9px 11px;font-family:var(--mono);font-size:10.5px;color:var(--dim);line-height:1.6}
  .democreds b{color:var(--cyan)}
  #opbadge{font-family:var(--mono);font-size:11px;color:var(--dim)}
</style>""")

# ---- P2: login overlay HTML ----
patch("P2 login overlay",
"""<div class="ov" id="livestat"></div>""",
"""<div class="ov" id="livestat"></div>

<div id="loginov">
  <div class="loginbox">
    <h2><b style="color:var(--accent)">ISDMAAS</b> Operator Login</h2>
    <div class="sub">satellite operators manage their own assets</div>
    <label>Operator username</label>
    <input id="lg_user" placeholder="operator_a" style="margin-bottom:9px">
    <label>Password</label>
    <input id="lg_pass" type="password" placeholder="••••••••">
    <button onclick="doLogin(false)">Log in</button>
    <div class="row"><button class="ghost" onclick="doLogin(true)" style="margin-top:9px">Register new operator</button>
    <button class="ghost" onclick="skipLogin()" style="margin-top:9px">View only (no login)</button></div>
    <div id="lg_msg" class="muted" style="margin-top:9px"></div>
    <div class="democreds">DEMO OPERATORS<br>
      <b>operator_a</b> / alpha123 &nbsp;·&nbsp; owns ALPHASAT-DEMO<br>
      <b>operator_b</b> / bravo123 &nbsp;·&nbsp; owns BRAVOSAT-DEMO<br>
      <span style="color:var(--faint)">Two fictitious satellites on a collision course —
      only each owner can approve their own maneuver.</span></div>
  </div>
</div>""")

# ---- P3: operator badge in the top bar ----
patch("P3 operator badge",
"""<div class="stitem"><span class="led cy"></span><span id="utc">--:--:-- UTC</span></div>""",
"""<div class="stitem" id="opbadge"></div>
    <div class="stitem"><span class="led cy"></span><span id="utc">--:--:-- UTC</span></div>""")

# ---- P4: My Satellites card (before Primary Asset card) ----
patch("P4 my-satellites card",
"""  <div class="card">
    <div class="hd">Primary Asset</div>""",
"""  <div class="card" style="border-color:#39d4e840">
    <div class="hd">&#128100; My Satellites (operator)</div>
    <div id="mysatsbody"><div class="muted">Log in to register and manage your satellites.</div></div>
    <label style="margin-top:11px">Register a satellite (name + TLE)</label>
    <input id="newsat_name" placeholder="MY-SAT-1" style="margin-bottom:6px">
    <input id="newsat_t1" placeholder="1 90001U ..." style="margin-bottom:6px;font-size:10px">
    <input id="newsat_t2" placeholder="2 90001 ..." style="font-size:10px">
    <button class="ghost" onclick="addMySat()">Add my satellite</button>
    <div id="mysat_msg" class="muted" style="margin-top:6px"></div>
  </div>

  <div class="card">
    <div class="hd">Primary Asset</div>""")

# ---- P5: auth + user-pipeline JS ----
patch("P5 auth JS",
"""let tcaSeconds=null,tcaStart=0,tcaRealEpoch=null;""",
"""let tcaSeconds=null,tcaStart=0,tcaRealEpoch=null;

// ===================== OPERATOR LOGIN & OWNED SATELLITES =====================
let opToken=localStorage.getItem("isdmaas_tok")||null, opUser=localStorage.getItem("isdmaas_user")||null;
let userSatSet={};
function authHeaders(){return opToken?{"Authorization":"Bearer "+opToken}:{};}
async function doLogin(reg){
  const u=document.getElementById("lg_user").value.trim(), p=document.getElementById("lg_pass").value;
  const msg=document.getElementById("lg_msg");
  if(!u||!p){msg.textContent="enter username and password";return;}
  try{
    const r=await fetch(API+(reg?"/auth/register":"/auth/login"),{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({username:u,password:p})});
    const d=await r.json();
    if(!r.ok){msg.textContent=d.detail||"failed";return;}
    if(reg){msg.textContent="\\u2713 registered \\u2014 now log in";return;}
    opToken=d.token;opUser=d.username;
    localStorage.setItem("isdmaas_tok",opToken);localStorage.setItem("isdmaas_user",opUser);
    document.getElementById("loginov").style.display="none";
    refreshAuthUI();
  }catch(e){msg.textContent="API offline? "+e;}
}
function skipLogin(){document.getElementById("loginov").style.display="none";refreshAuthUI();}
async function doLogout(){
  try{await fetch(API+"/auth/logout",{method:"POST",headers:authHeaders()});}catch(e){}
  opToken=null;opUser=null;localStorage.removeItem("isdmaas_tok");localStorage.removeItem("isdmaas_user");
  refreshAuthUI();document.getElementById("loginov").style.display="flex";
}
async function refreshAuthUI(){
  const badge=document.getElementById("opbadge");
  const card=document.getElementById("mysatsbody");
  if(!opToken){if(badge)badge.innerHTML='<span style="color:var(--faint)">not logged in</span>';
    if(card)card.innerHTML='<div class="muted">Log in to register and manage your satellites.</div>';return;}
  try{
    const r=await fetch(API+"/my/satellites",{headers:authHeaders()});
    if(r.status===401){opToken=null;opUser=null;refreshAuthUI();return;}
    const d=await r.json();
    badge.innerHTML='OPERATOR <b style="color:var(--cyan)">'+opUser+'</b>'+
      ' <span onclick="doLogout()" style="cursor:pointer;color:var(--red);margin-left:6px">logout</span>';
    userSatSet={};
    const sel=document.getElementById("sat");
    let h="";
    for(const s of d.mine){
      userSatSet[s.norad]={name:s.name,mine:true};
      if(![...sel.options].find(o=>o.value==s.norad)){
        const o=document.createElement("option");o.value=s.norad;
        o.textContent=s.name+" \\u00b7 yours";sel.appendChild(o);}
      h+='<div class="conj" onclick="selectUserSat(\\''+s.norad+'\\')"><div class="cn">'+s.name+
         ' <span class="pill NOMINAL" style="font-size:8px">YOURS</span></div>'+
         '<div class="cm"><span>NORAD '+s.norad+'</span></div></div>';
    }
    for(const s of d.others){
      userSatSet[s.norad]={name:s.name,mine:false,owner:s.owner};
      h+='<div class="conj" style="opacity:.7"><div class="cn">'+s.name+'</div>'+
         '<div class="cm"><span>NORAD '+s.norad+'</span><span>owner: '+s.owner+'</span></div></div>';
    }
    const mine=d.mine.map(s=>s.norad), all=[...d.mine,...d.others].map(s=>s.norad);
    if(mine.length&&all.length>=2){
      const other=all.find(n=>n!==mine[0]);
      h+='<button class="histbtn" onclick="assessUserConj(\\''+mine[0]+'\\',\\''+other+'\\')" style="margin-top:8px">'+
         '\\u26a0 Assess conjunction: '+userSatSet[mine[0]].name+' \\u00d7 '+userSatSet[other].name+'</button>';
    }
    card.innerHTML=h||'<div class="muted">No satellites yet \\u2014 add a TLE below.</div>';
  }catch(e){if(badge)badge.textContent="auth error";}
}
function selectUserSat(n){document.getElementById("sat").value=n;loadOrbit();}
async function addMySat(){
  const name=document.getElementById("newsat_name").value.trim();
  const t1=document.getElementById("newsat_t1").value.trim();
  const t2=document.getElementById("newsat_t2").value.trim();
  const msg=document.getElementById("mysat_msg");
  msg.textContent="registering\\u2026";
  try{
    const r=await fetch(API+"/my/satellites",{method:"POST",
      headers:{...authHeaders(),"Content-Type":"application/json"},
      body:JSON.stringify({name:name,tle1:t1,tle2:t2})});
    const d=await r.json();
    if(!r.ok){msg.textContent=d.detail||"failed";return;}
    msg.textContent="\\u2713 registered NORAD "+d.norad;
    refreshAuthUI();
  }catch(e){msg.textContent=""+e;}
}
async function assessUserConj(pn,sn){
  document.getElementById("telemetry").style.display="flex";
  const out=document.getElementById("result");
  out.innerHTML='<div class="loading"><span class="spin"></span>Real TCA search on operator TLEs\\u2026</div>';
  try{
    const oa=await (await fetch(API+"/user/orbit/"+pn+"?points=240")).json();
    primaryTrack=oa.track_km;primaryOrbit=drawOrbit(primaryTrack,0x3d8bff,primaryOrbit,true);
    if(!primaryDot)primaryDot=mkDot(0x3d8bff,200);
    const ob=await (await fetch(API+"/user/orbit/"+sn+"?points=240")).json();
    threatTrack=ob.track_km;threatOrbit=drawOrbit(threatTrack,0xf4506a,threatOrbit,true);
    if(!threatDot)threatDot=mkDot(0xf4506a,180);
    const rmax=Math.max(...primaryTrack.map(p=>Math.hypot(p[0],p[1],p[2])));camR=rmax*3.4;updateCam();
  }catch(e){}
  try{
    const r=await fetch(API+"/user/assess",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({primary_norad:pn,secondary_norad:sn})});
    const d=await r.json();
    if(!r.ok){out.innerHTML='<div class="err">'+(d.detail||"assess failed")+'</div>';return;}
    let h='<div class="kv"><span class="k">Conjunction</span><span class="v" style="font-size:11px">'+
      d.primary.name+' \\u00d7 '+d.secondary.name+'</span></div>';
    h+='<div class="kv"><span class="k">Owners</span><span class="v" style="font-size:10px">'+
      d.primary.owner+' / '+d.secondary.owner+'</span></div>';
    h+='<div class="kv"><span class="k">TCA</span><span class="v">in '+d.tca_in_hours+' h</span></div>';
    h+='<div class="kv"><span class="k">Miss distance</span><span class="v">'+d.miss_km+' km</span></div>';
    h+='<div class="kv"><span class="k">Rel speed</span><span class="v">'+d.rel_speed_kms+' km/s</span></div>';
    h+='<div class="kv"><span class="k">Collision Pc</span><span class="v">'+sci(d.pc)+' '+pill(d.risk_level)+'</span></div>';
    h+='<div class="row" style="margin-top:10px">'+
       '<button class="ghost" style="margin-top:0;border-color:#34d399;color:#34d399" '+
       'onclick="planUserManeuver(\\''+pn+'\\',\\''+sn+'\\')">Plan: '+d.primary.name+'</button>'+
       '<button class="ghost" style="margin-top:0;border-color:#fbbf24;color:#fbbf24" '+
       'onclick="planUserManeuver(\\''+sn+'\\',\\''+pn+'\\')">Plan: '+d.secondary.name+'</button></div>';
    h+='<div class="muted" style="margin-top:8px">Maneuver authority is restricted to the owning '+
       'operator \\u2014 planning for a satellite you don\\u2019t own is refused (403).</div>';
    out.innerHTML=h;
  }catch(e){out.innerHTML='<div class="err">'+e+'</div>';}
}
async function planUserManeuver(pn,sn){
  const out=document.getElementById("result");
  const prev=out.innerHTML;
  out.innerHTML=prev+'<div class="loading"><span class="spin"></span>Requesting owner-authorized plan\\u2026</div>';
  try{
    const r=await fetch(API+"/user/plan",{method:"POST",
      headers:{...authHeaders(),"Content-Type":"application/json"},
      body:JSON.stringify({primary_norad:pn,secondary_norad:sn})});
    const d=await r.json();
    if(r.status===403||r.status===401){
      out.innerHTML=prev+'<div class="decblock plan" style="margin-top:8px">'+
        '<div class="decrow"><span class="deck">Security</span><span class="decv" style="color:var(--red)">DENIED</span></div>'+
        '<div class="decrow"><span class="deck">Reason</span><span class="decv2">'+(d.detail||"not authorized")+'</span></div></div>';
      return;}
    if(!r.ok){out.innerHTML=prev+'<div class="err">'+(d.detail||"plan failed")+'</div>';return;}
    const rec=d.recommendation;
    let h=prev+'<div class="manbanner" style="margin-top:10px"><div class="manbanner-l">PLAN MANEUVER</div>'+
      '<div class="manbanner-r">OWNER: '+d.authorized_operator+'</div></div>';
    h+='<div class="kv"><span class="k">Burn \\u0394v</span><span class="v big">'+rec.dv_magnitude_ms.toFixed(3)+
       ' <span style="font-size:11px">m/s</span></span></div>';
    h+='<div class="kv"><span class="k">Execute</span><span class="v">T\\u2212'+rec.burn_lead_time_h+' h</span></div>';
    h+='<div class="kv"><span class="k">Propellant</span><span class="v">'+rec.fuel_kg.toFixed(4)+' kg</span></div>';
    h+='<div class="kv"><span class="k">Expected miss</span><span class="v">'+d.assessment.miss_km+
       ' km <span class="arrow">\\u2192</span> '+rec.predicted_new_miss_km+' km</span></div>';
    h+='<div class="kv"><span class="k">Expected Pc</span><span class="v">'+sci(d.assessment.pc)+
       ' <span class="arrow">\\u2192</span> '+sci(rec.predicted_new_pc)+'</span></div>';
    for(const c of d.safety_validation.checks)
      h+='<div class="chk"><span class="mk '+(c.pass?"ok":"no")+'">'+(c.pass?"\\u2713":"\\u2715")+'</span>'+
         c.check.replace(/_/g," ")+'</div>';
    h+='<div class="verdict '+d.verdict+'">'+d.verdict+' \\u00b7 owner-authorized</div>';
    out.innerHTML=h;
  }catch(e){out.innerHTML=prev+'<div class="err">'+e+'</div>';}
}""")

# ---- P6: route user satellites through /user/orbit ----
patch("P6 orbit routing",
"""async function loadOrbit(){
  const norad=document.getElementById("sat").value;
  try{
    const d=await (await fetch(API+"/orbit/"+norad+"?points=240")).json();""",
"""async function loadOrbit(){
  const norad=document.getElementById("sat").value;
  try{
    const url=userSatSet[norad]?"/user/orbit/"+norad+"?points=240":"/orbit/"+norad+"?points=240";
    const d=await (await fetch(API+url)).json();""")

# ---- P7: show login on startup + refresh auth UI ----
patch("P7 init hook",
"""    loadLibrary();
    autoSync();
    loadOrbit();""",
"""    loadLibrary();
    autoSync();
    loadOrbit();
    if(!opToken)document.getElementById("loginov").style.display="flex";
    refreshAuthUI();""")

open(CONSOLE, "w", encoding="utf-8").write(html)

# ---- API wiring ----
api = open(APIFILE, encoding="utf-8").read()
if "install_auth" not in api:
    api += ("\n\n# ---- multi-operator login & user-satellite pipeline (demo) ----\n"
            "from auth_store import install_auth\n"
            "from user_conjunctions import install_user_pipeline\n"
            "install_auth(app)\n"
            "install_user_pipeline(app)\n")
    open(APIFILE, "w", encoding="utf-8").write(api)
    patches.append(("P8 API wiring", "OK (appended)"))
else:
    patches.append(("P8 API wiring", "already present"))

print("\nPatch results:")
ok = True
for name, res in patches:
    print(f"  {name}: {res}")
    if "FAILED" in res: ok = False
print("\n" + ("ALL PATCHES APPLIED — restart uvicorn and reload the console." if ok else
      "SOME PATCHES FAILED — your console.html differs from the expected version. "
      "Paste the failure lines back and I'll re-anchor them."))
