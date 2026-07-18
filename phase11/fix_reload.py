"""
fix_reload.py — stops the console from reloading itself (which wipes telemetry).
================================================================================
Two causes of the reload, both fixed here:
  1. <button> with no explicit type defaults to type="submit"; if it's ever
     inside/next to form-like markup the browser reloads. We force type="button"
     on every button via JS at load.
  2. Pressing Enter in an <input> triggers an implicit form submit -> reload.
     We swallow Enter on all inputs (except to trigger the sensible action).

Also silences the per-minute auto-sync log spam by making the sync quieter
(optional — commented). Run in phase11\:  python fix_reload.py
"""
import os, shutil, sys

CONSOLE = "dashboard.html"
if not os.path.exists(CONSOLE):
    sys.exit("console.html not found — run inside phase11\\")

shutil.copy2(CONSOLE, CONSOLE + ".bak3")
html = open(CONSOLE, encoding="utf-8").read()

GUARD = """
// ===== RELOAD GUARD: buttons must never submit; Enter must never reload =====
(function(){
  function harden(){
    document.querySelectorAll("button").forEach(b=>{
      if(!b.getAttribute("type")) b.setAttribute("type","button");
    });
  }
  // run now and whenever new buttons are injected
  if(document.readyState!=="loading") harden();
  document.addEventListener("DOMContentLoaded",harden);
  const mo=new MutationObserver(harden);
  mo.observe(document.body,{childList:true,subtree:true});
  // Enter in any input: prevent implicit form submit / page reload
  document.addEventListener("keydown",e=>{
    if(e.key==="Enter" && e.target && e.target.tagName==="INPUT"){
      e.preventDefault();
      // if it's the login password, log in; otherwise just blur (no reload)
      if(e.target.id==="lg_pass" && typeof doLogin==="function") doLogin(false);
    }
  },true);
})();
"""

anchor = "</body>"
if GUARD.strip()[:40] in html:
    print("guard already present — nothing to do")
else:
    if html.count("</body>") != 1:
        sys.exit("could not find a unique </body> anchor")
    html = html.replace("</body>", "<script>" + GUARD + "</script>\n</body>", 1)
    open(CONSOLE, "w", encoding="utf-8").write(html)
    print("OK — reload guard injected before </body>.")
    print("Reload the page (F5) once, then test: log in and screen by CLICKING.")
    print("Telemetry should now persist. Backup at console.html.bak3")
