/* ===========================================================================
   ISDMAAS Mission Console
   ===========================================================================
   Two rules govern this file.

   1. NOTHING FROM THE NETWORK IS EVER PARSED AS HTML.
      Every value that reaches the screen — satellite names from CelesTrak,
      operator-supplied asset names, event titles, error messages — goes through
      textContent. The previous console built its markup by concatenating those
      values into innerHTML, including into inline onclick="" attributes, so a
      catalog object named `<img src=x onerror=...>` executed script in the
      operator's session. There is no escaping helper here because there is
      nothing to escape: the DOM builder below cannot produce markup from data.

   2. NO INLINE HANDLERS.
      Interaction is wired with addEventListener and delegated clicks keyed off
      data attributes, which is what lets the page run under a strict
      Content-Security-Policy with no 'unsafe-inline'.

   Also deliberately absent: the previous file's "reload trap", which overrode
   Location.prototype.reload, cancelled beforeunload and swallowed every Enter
   keypress. Those hid a symptom (submit-type buttons) rather than fixing it,
   and they broke the browser's normal behaviour for the operator.
   =========================================================================== */
'use strict';

/* --------------------------------------------------------------- constants */
const EARTH_RADIUS_KM = 6371;
const PC_ACTION_THRESHOLD = 1e-4;
const AUTO_MONITOR_INTERVAL_MS = 120000;
const TOKEN_KEY = 'isdmaas.session';

const RISK_ORDER = ['NOMINAL', 'ELEVATED', 'HIGH', 'CRITICAL', 'UNDETERMINED'];
const COLOR = {
  primary: 0x4d95ff, threat: 0xff6079, post: 0x3ddc9a,
  debris: 0xff6079, rocketBody: 0xffc53d, payload: 0x4d95ff, unknown: 0x93a6c4
};

/**
 * API base URL.
 *
 * Same origin by default, which is how this should be deployed: one origin for
 * the console and the API means no CORS surface and no mixed-content risk. The
 * localhost fallback covers the two-terminal development setup, and `?api=` is
 * an explicit override for pointing a local console at a staging service.
 */
const API = (() => {
  const override = new URLSearchParams(location.search).get('api');
  if (override) {
    try {
      const url = new URL(override);
      if (url.protocol === 'http:' || url.protocol === 'https:') {
        return url.origin;
      }
    } catch { /* fall through to the defaults */ }
  }
  if (location.protocol === 'file:') return 'http://localhost:8000';
  if (location.port && location.port !== '8000') {
    return `${location.protocol}//${location.hostname}:8000`;
  }
  return location.origin;
})();

/* ------------------------------------------------------------ DOM building */
/** Shorthand for document.getElementById. */
const $ = (id) => document.getElementById(id);

/**
 * Build an element. Children that are strings become TEXT NODES, never markup.
 *
 * @param {string} tag
 * @param {object} [props] className, textContent, dataset, attrs, on, style
 * @param {Array} [children] elements or strings
 */
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  if (props.className) node.className = props.className;
  if (props.text !== undefined && props.text !== null) node.textContent = String(props.text);
  if (props.title) node.title = String(props.title);
  if (props.dataset) {
    for (const [key, value] of Object.entries(props.dataset)) {
      node.dataset[key] = String(value);
    }
  }
  if (props.attrs) {
    for (const [key, value] of Object.entries(props.attrs)) {
      node.setAttribute(key, String(value));
    }
  }
  if (props.on) {
    for (const [event, handler] of Object.entries(props.on)) {
      node.addEventListener(event, handler);
    }
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === 'string' || typeof child === 'number'
      ? document.createTextNode(String(child))
      : child);
  }
  return node;
}

/** Replace an element's contents with the given nodes. */
function render(target, ...nodes) {
  target.replaceChildren(...nodes.flat().filter(Boolean));
}

const kv = (key, ...value) => el('div', { className: 'kv' }, [
  el('span', { className: 'k', text: key }),
  el('span', { className: 'v' }, value)
]);

const pill = (risk) => el('span', {
  className: `pill ${RISK_ORDER.includes(risk) ? risk : 'UNDETERMINED'}`,
  text: risk || 'UNDETERMINED'
});

const spinner = (message) => el('div', { className: 'loading' }, [
  el('span', { className: 'spin' }), message
]);

const errorBox = (message) => el('p', { className: 'err', text: message });

/* ------------------------------------------------------------- formatting */
/** Scientific notation, with an honest dash when the value is unavailable. */
function sci(value) {
  if (value === null || value === undefined || !isFinite(value)) return '—';
  if (value === 0) return '0';
  return Number(value).toExponential(2);
}

function num(value, digits = 3) {
  if (value === null || value === undefined || !isFinite(value)) return '—';
  return Number(value).toFixed(digits);
}

function hhmmss(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const h = String(Math.floor(total / 3600)).padStart(2, '0');
  const m = String(Math.floor((total % 3600) / 60)).padStart(2, '0');
  const s = String(total % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

function humanDuration(ms) {
  const hours = Math.floor(ms / 3600000);
  const minutes = Math.floor((ms % 3600000) / 60000);
  if (hours >= 24) return `${Math.floor(hours / 24)}d ${hours % 24}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function parseUtc(raw) {
  if (!raw) return null;
  const direct = new Date(String(raw).trim());
  if (!isNaN(direct)) return direct;
  const match = String(raw).match(/(\d{4})\s+(\w{3})\s+(\d{1,2})\s+(\d{1,2}):(\d{2})/);
  if (!match) return null;
  const months = { Jan: 0, Feb: 1, Mar: 2, Apr: 3, May: 4, Jun: 5,
                   Jul: 6, Aug: 7, Sep: 8, Oct: 9, Nov: 10, Dec: 11 };
  const parsed = new Date(Date.UTC(
    +match[1], months[match[2]] ?? 0, +match[3], +match[4], +match[5]));
  return isNaN(parsed) ? null : parsed;
}

function formatTca(raw) {
  const when = parseUtc(raw);
  if (!when) return { label: String(raw || '—').slice(0, 20), countdown: '' };
  const month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][when.getUTCMonth()];
  const label = `${month} ${String(when.getUTCDate()).padStart(2, '0')} `
    + `${String(when.getUTCHours()).padStart(2, '0')}:`
    + `${String(when.getUTCMinutes()).padStart(2, '0')}Z`;
  const delta = when.getTime() - Date.now();
  return {
    label,
    countdown: delta < 0
      ? `passed ${humanDuration(-delta)} ago`
      : `in ${humanDuration(delta)}`,
    epoch: when.getTime()
  };
}

/* --------------------------------------------------------------- API layer */
const session = {
  token: null,
  username: null,
  expires: null,

  load() {
    try {
      const stored = JSON.parse(localStorage.getItem(TOKEN_KEY) || 'null');
      if (!stored || !stored.token) return;
      if (stored.expires && Date.parse(stored.expires) <= Date.now()) {
        this.clear();
        return;
      }
      Object.assign(this, stored);
    } catch { this.clear(); }
  },

  save(payload) {
    this.token = payload.token;
    this.username = payload.username;
    this.expires = payload.expires_utc || null;
    try {
      localStorage.setItem(TOKEN_KEY, JSON.stringify({
        token: this.token, username: this.username, expires: this.expires
      }));
    } catch { /* private browsing: the session simply does not survive a reload */ }
  },

  clear() {
    this.token = null;
    this.username = null;
    this.expires = null;
    try { localStorage.removeItem(TOKEN_KEY); } catch { /* nothing to do */ }
  },

  headers() {
    return this.token ? { Authorization: `Bearer ${this.token}` } : {};
  }
};

class ApiError extends Error {
  constructor(status, message, code) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

/**
 * Call the API.
 *
 * A 401 clears the stored session and re-opens the sign-in dialog rather than
 * leaving the console in a half-authenticated state where every later action
 * fails with an unexplained error.
 */
async function api(path, options = {}) {
  const { method = 'GET', body, auth = false, timeoutMs = 180000 } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(API + path, {
      method,
      signal: controller.signal,
      headers: {
        ...(body ? { 'Content-Type': 'application/json' } : {}),
        ...(auth ? session.headers() : {})
      },
      body: body ? JSON.stringify(body) : undefined
    });
  } catch (cause) {
    clearTimeout(timer);
    throw new ApiError(0, cause.name === 'AbortError'
      ? 'The request timed out.'
      : `Cannot reach the ISDMAAS service at ${API}.`, 'unreachable');
  }
  clearTimeout(timer);

  let payload = null;
  try { payload = await response.json(); } catch { /* empty or non-JSON body */ }

  if (!response.ok) {
    const message = payload?.error?.message || payload?.detail
      || `Request failed (HTTP ${response.status}).`;
    if (response.status === 401 && auth) {
      session.clear();
      refreshAuthUi();
      openLogin('Your session expired. Please sign in again.');
    }
    throw new ApiError(response.status, message, payload?.error?.code);
  }
  return payload;
}

/* -------------------------------------------------------------- top banner */
let bannerTimer = null;
function showBanner(message, kind = 'error', autoHideMs = 8000) {
  const banner = $('banner');
  banner.textContent = message;
  banner.className = kind === 'info' ? 'info' : '';
  banner.hidden = false;
  clearTimeout(bannerTimer);
  if (autoHideMs) bannerTimer = setTimeout(() => { banner.hidden = true; }, autoHideMs);
}

/* =====================================================================
   3-D SCENE
   ===================================================================== */
let scene, camera, renderer, earth, graticule;
let primaryOrbit, threatOrbit, postOrbit, primaryDot, threatDot, tcaMarker;
let primaryTrack = [], threatTrack = [];
let camTheta = 0.9, camPhi = 1.05, camRadius = 22000, earthSpin = 0;
let dragging = false, dragX = 0, dragY = 0;
let tcaEpochMs = null, tcaSeconds = null, tcaStartedAt = 0;
let raycaster, pointerNdc, pointerX = 0, pointerY = 0;

function initScene() {
  const canvas = $('scene');
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x05070e, 0.0000035);
  camera = new THREE.PerspectiveCamera(42, innerWidth / innerHeight, 80, 400000);
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  buildEarth();
  buildStarfield();

  // Lighting is tuned so the globe reads as a lit sphere with a visible
  // terminator rather than a saturated disc: one dominant key light, a low
  // ambient fill that keeps the night side legible, and a cool rim to separate
  // the limb from the starfield. Summed intensity stays under 1.0 on the lit
  // side, which is what keeps the graticule and the orbit tracks readable
  // against it.
  scene.add(new THREE.AmbientLight(0x38506e, 0.32));
  const sun = new THREE.DirectionalLight(0xfff2e2, 0.85);
  sun.position.set(3, 1.2, 2);
  scene.add(sun);
  const rim = new THREE.DirectionalLight(0x3f7fd0, 0.22);
  rim.position.set(-2, -0.4, -1.2);
  scene.add(rim);

  raycaster = new THREE.Raycaster();
  raycaster.params.Points = { threshold: 120 };
  pointerNdc = new THREE.Vector2(-2, -2);

  updateCamera();
  animate();

  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
  canvas.addEventListener('pointerdown', (event) => {
    dragging = true; dragX = event.clientX; dragY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointerup', (event) => {
    dragging = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', (event) => {
    const rect = canvas.getBoundingClientRect();
    pointerNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointerNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    pointerX = event.clientX; pointerY = event.clientY;
    if (!dragging) return;
    camTheta -= (event.clientX - dragX) * 0.005;
    camPhi = Math.max(0.15, Math.min(2.95, camPhi - (event.clientY - dragY) * 0.005));
    dragX = event.clientX; dragY = event.clientY;
    updateCamera();
  });
  canvas.addEventListener('pointerleave', () => {
    pointerNdc.set(-2, -2);
    hideTooltip();
  });
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    camRadius = Math.max(8800, Math.min(70000, camRadius + event.deltaY * 16));
    updateCamera();
  }, { passive: false });
}

/**
 * A self-contained Earth.
 *
 * The previous console pulled its Earth textures from unpkg. That meant the
 * globe silently degraded to a blank sphere whenever the network was
 * unavailable, and it forced the page's CSP to allow a third-party image
 * origin. A shaded sphere with a latitude/longitude graticule reads more like
 * an operations display than a photograph anyway, and it always renders.
 */
function buildEarth() {
  const material = new THREE.MeshPhongMaterial({
    color: 0x15304f,
    // A faint emissive keeps the night side from going pure black without
    // washing out the lit side; a near-black specular avoids the mirror-ball
    // highlight that a shiny untextured sphere otherwise produces.
    emissive: 0x060f1c, emissiveIntensity: 1.0,
    shininess: 6, specular: 0x0a1626
  });
  earth = new THREE.Mesh(new THREE.SphereGeometry(EARTH_RADIUS_KM, 96, 96), material);
  scene.add(earth);

  const lines = [];
  for (let lat = -75; lat <= 75; lat += 15) {
    const phi = (90 - lat) * Math.PI / 180;
    const radius = EARTH_RADIUS_KM * Math.sin(phi);
    const y = EARTH_RADIUS_KM * Math.cos(phi);
    for (let i = 0; i < 96; i++) {
      const a = (i / 96) * Math.PI * 2, b = ((i + 1) / 96) * Math.PI * 2;
      lines.push(radius * Math.cos(a), y, radius * Math.sin(a),
                 radius * Math.cos(b), y, radius * Math.sin(b));
    }
  }
  for (let lon = 0; lon < 360; lon += 15) {
    const theta = lon * Math.PI / 180;
    for (let i = 0; i < 96; i++) {
      const a = (i / 96) * Math.PI, b = ((i + 1) / 96) * Math.PI;
      lines.push(
        EARTH_RADIUS_KM * Math.sin(a) * Math.cos(theta), EARTH_RADIUS_KM * Math.cos(a),
        EARTH_RADIUS_KM * Math.sin(a) * Math.sin(theta),
        EARTH_RADIUS_KM * Math.sin(b) * Math.cos(theta), EARTH_RADIUS_KM * Math.cos(b),
        EARTH_RADIUS_KM * Math.sin(b) * Math.sin(theta));
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
  graticule = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
    color: 0x5b9bd8, transparent: true, opacity: 0.22, depthWrite: false
  }));
  graticule.scale.setScalar(1.0015);
  scene.add(graticule);

  // Two back-facing shells give the limb an atmospheric falloff. They are thin
  // on purpose: a strong halo swallows the low-altitude orbit tracks, which are
  // the thing an operator is actually looking at.
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(EARTH_RADIUS_KM * 1.02, 64, 64),
    new THREE.MeshBasicMaterial({
      color: 0x3f8ae0, transparent: true, opacity: 0.07,
      side: THREE.BackSide, depthWrite: false })));
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(EARTH_RADIUS_KM * 1.09, 64, 64),
    new THREE.MeshBasicMaterial({
      color: 0x1e57a8, transparent: true, opacity: 0.03,
      side: THREE.BackSide, depthWrite: false })));
}

function buildStarfield() {
  for (const [count, radius, size, color] of
       [[1800, 140000, 260, 0x6a7a98], [600, 90000, 420, 0x9fb0d0]]) {
    const points = [];
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      points.push(radius * Math.sin(phi) * Math.cos(theta),
                  radius * Math.sin(phi) * Math.sin(theta),
                  radius * Math.cos(phi));
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({ color, size })));
  }
}

function updateCamera() {
  camera.position.set(
    camRadius * Math.sin(camPhi) * Math.cos(camTheta),
    camRadius * Math.cos(camPhi),
    camRadius * Math.sin(camPhi) * Math.sin(camTheta));
  camera.lookAt(0, 0, 0);
}

function animate() {
  requestAnimationFrame(animate);
  // A console is often left open on a second monitor or behind other windows.
  // Skipping the work while the document is hidden keeps a background tab from
  // holding the GPU at full tilt for a scene nobody is looking at.
  if (document.hidden) return;
  earthSpin += 0.0006;
  earth.rotation.y = earthSpin;
  graticule.rotation.y = earthSpin;

  const phase = Date.now() * 0.00003;
  if (primaryDot && primaryTrack.length) primaryDot.position.copy(trackPoint(primaryTrack, phase));
  if (threatDot && threatTrack.length) threatDot.position.copy(trackPoint(threatTrack, phase + 0.5));

  if (tcaEpochMs !== null) {
    $('clockval').textContent = hhmmss((tcaEpochMs - Date.now()) / 1000);
  } else if (tcaSeconds !== null) {
    $('clockval').textContent =
      hhmmss(tcaSeconds - (Date.now() - tcaStartedAt) / 1000 * 400);
  }

  propagateLive();
  updateTooltip();
  renderer.render(scene, camera);
}

function trackPoint(track, phase) {
  const index = Math.floor(((phase % 1) + 1) % 1 * track.length) % track.length;
  return new THREE.Vector3(...track[index]);
}

function drawOrbit(track, color, previous, opacity = 0.95) {
  if (previous && previous.parent) {
    scene.remove(previous);
    previous.geometry.dispose();
    previous.material.dispose();
  }
  const points = track.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
  if (points.length) points.push(points[0].clone());
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const line = new THREE.Line(geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  scene.add(line);
  return line;
}

function makeDot(color, size = 190) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(size, 18, 18),
    new THREE.MeshBasicMaterial({ color }));
  mesh.add(new THREE.Mesh(new THREE.SphereGeometry(size * 2.2, 16, 16),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.18 })));
  scene.add(mesh);
  return mesh;
}

function clearScenario() {
  for (const name of ['threatOrbit', 'postOrbit', 'tcaMarker']) {
    const object = { threatOrbit, postOrbit, tcaMarker }[name];
    if (object) {
      scene.remove(object);
      object.geometry?.dispose();
      object.material?.dispose();
    }
  }
  threatOrbit = postOrbit = tcaMarker = null;
  threatTrack = [];
  tcaSeconds = null;
  tcaEpochMs = null;
  $('clock').hidden = true;
}

/** A visually distinct crossing orbit for the threat. Illustrative, not a fit. */
function crossingTrack(track, incDeg, raanDeg) {
  const i = incDeg * Math.PI / 180, omega = raanDeg * Math.PI / 180;
  const ci = Math.cos(i), si = Math.sin(i);
  const co = Math.cos(omega), so = Math.sin(omega);
  return track.map(([x, y, z]) => {
    const y1 = y * ci - z * si;
    const z1 = y * si + z * ci;
    return [x * co - y1 * so, x * so + y1 * co, z1];
  });
}

function markTca(position) {
  if (tcaMarker) scene.remove(tcaMarker);
  tcaMarker = new THREE.Mesh(
    new THREE.RingGeometry(380, 520, 32),
    new THREE.MeshBasicMaterial({
      color: 0xffc53d, side: THREE.DoubleSide, transparent: true, opacity: 0.85 }));
  tcaMarker.position.set(position[0], position[1], position[2]);
  tcaMarker.lookAt(0, 0, 0);
  scene.add(tcaMarker);
}

/* ---------------------------------------------------------- live tracking */
let liveObjects = [];
let liveSimTime = new Date();
let liveLastFrame = performance.now();
let currentLiveGroup = null;

function eciToScene(p) { return new THREE.Vector3(p.x, p.z, -p.y); }

function propagateLive() {
  if (!liveObjects.length) return;
  const now = performance.now();
  const dt = (now - liveLastFrame) / 1000;
  liveLastFrame = now;
  const rate = parseFloat($('livespeed').value) || 60;
  liveSimTime = new Date(liveSimTime.getTime() + dt * rate * 1000);
  for (const object of liveObjects) {
    const state = satellite.propagate(object.satrec, liveSimTime);
    if (state && state.position) object.mesh.position.copy(eciToScene(state.position));
  }
}

async function loadLive() {
  const group = $('group').value;
  const limit = $('livelimit').value;
  const message = $('livemsg');
  message.textContent = 'Fetching catalog…';
  clearLive();
  try {
    const data = await api(
      `/live/catalog?group=${encodeURIComponent(group)}&limit=${encodeURIComponent(limit)}`);
    if (!data.objects?.length) {
      message.textContent = 'No objects cached. Run: python debris_data.py sync';
      return;
    }
    for (const object of data.objects) {
      let satrec;
      try { satrec = satellite.twoline2satrec(object.tle1, object.tle2); } catch { continue; }
      const color = object.type === 'DEBRIS' ? COLOR.debris
        : object.type === 'ROCKET BODY' ? COLOR.rocketBody
        : object.type === 'PAYLOAD' ? COLOR.payload : COLOR.unknown;
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(95, 8, 8),
        new THREE.MeshBasicMaterial({ color }));
      scene.add(mesh);
      liveObjects.push({ satrec, mesh, ...object });
    }
    liveSimTime = new Date();
    liveLastFrame = performance.now();
    currentLiveGroup = group;
    $('liveleg').hidden = false;
    message.textContent = `Tracking ${liveObjects.length} objects.`;
    camRadius = Math.max(camRadius, 18000);
    updateCamera();
  } catch (error) {
    message.textContent = error.message;
  }
}

function clearLive() {
  for (const object of liveObjects) {
    scene.remove(object.mesh);
    object.mesh.geometry.dispose();
    object.mesh.material.dispose();
  }
  liveObjects = [];
  currentLiveGroup = null;
  $('liveleg').hidden = true;
}

function hideTooltip() { $('tooltip').style.display = 'none'; }

function updateTooltip() {
  if (!raycaster) return;
  const targets = liveObjects.map((o) => o.mesh);
  if (!targets.length) { hideTooltip(); return; }
  raycaster.setFromCamera(pointerNdc, camera);
  const hits = raycaster.intersectObjects(targets, false);
  if (!hits.length) { hideTooltip(); return; }

  const object = liveObjects.find((o) => o.mesh === hits[0].object);
  if (!object) { hideTooltip(); return; }
  const typeClass = object.type === 'ROCKET BODY' ? 'RB' : (object.type || 'PAYLOAD');
  const tooltip = $('tooltip');
  render(tooltip,
    el('div', { className: 'tname', text: object.name || 'object' }),
    object.norad ? el('div', { className: 'tmeta', text: `NORAD ${object.norad}` }) : null,
    (object.apogee_km && object.perigee_km)
      ? el('div', { className: 'tmeta',
          text: `alt ${Math.round(object.perigee_km)}–${Math.round(object.apogee_km)} km` })
      : null,
    el('span', { className: `ttype ${typeClass}`, text: object.type || '' }));
  tooltip.style.display = 'block';
  tooltip.style.left = `${pointerX + 14}px`;
  tooltip.style.top = `${pointerY + 14}px`;
}

/* =====================================================================
   AUTHENTICATION
   ===================================================================== */
function openLogin(message) {
  $('loginov').hidden = false;
  if (message) $('lg_msg').textContent = message;
  $('lg_user').focus();
}

function closeLogin() {
  $('loginov').hidden = true;
  $('lg_msg').textContent = '';
  $('lg_pass').value = '';
}

async function doLogin(event) {
  event.preventDefault();
  const username = $('lg_user').value.trim();
  const password = $('lg_pass').value;
  const message = $('lg_msg');
  if (!username || !password) {
    message.textContent = 'Enter a username and password.';
    return;
  }
  const submit = $('lg_submit');
  submit.disabled = true;
  message.textContent = 'Signing in…';
  try {
    session.save(await api('/auth/login', {
      method: 'POST', body: { username, password }
    }));
    closeLogin();
    await refreshAuthUi();
  } catch (error) {
    message.textContent = error.message;
  } finally {
    submit.disabled = false;
  }
}

async function doRegister() {
  const username = $('lg_user').value.trim();
  const password = $('lg_pass').value;
  const message = $('lg_msg');
  if (!username || !password) {
    message.textContent = 'Enter the username and password you want to create.';
    return;
  }
  try {
    await api('/auth/register', { method: 'POST', body: { username, password } });
    message.textContent = 'Registered. You can sign in now.';
  } catch (error) {
    message.textContent = error.message;
  }
}

async function doLogout() {
  try { await api('/auth/logout', { method: 'POST', auth: true }); } catch { /* best effort */ }
  session.clear();
  await refreshAuthUi();
  openLogin();
}

async function refreshAuthUi() {
  const badge = $('opbadge');
  const body = $('mysatsbody');
  if (!session.token) {
    render(badge, el('span', { className: 'muted', text: 'not signed in' }));
    render(body, el('p', { className: 'muted',
      text: 'Sign in to register and manage your assets.' }));
    return;
  }
  render(badge,
    'OPERATOR ', el('b', { text: session.username }),
    el('button', { type: 'button', text: 'sign out', on: { click: doLogout } }));

  try {
    const data = await api('/my/satellites', { auth: true });
    const nodes = [];
    for (const asset of data.mine) {
      nodes.push(el('button', {
        className: 'conj', attrs: { type: 'button' },
        dataset: { action: 'select-asset', norad: asset.norad }
      }, [
        el('div', { className: 'cn' }, [asset.name, pill('NOMINAL')]),
        el('div', { className: 'cm' }, [el('span', { text: `NORAD ${asset.norad}` })])
      ]));
    }
    for (const asset of data.others) {
      nodes.push(el('div', { className: 'conj readonly' }, [
        el('div', { className: 'cn', text: asset.name }),
        el('div', { className: 'cm' }, [
          el('span', { text: `NORAD ${asset.norad}` }),
          el('span', { text: asset.owner })
        ])
      ]));
    }
    if (data.mine.length && data.others.length) {
      nodes.push(el('button', {
        className: 'ghost danger',
        attrs: { type: 'button' },
        dataset: {
          action: 'assess-pair',
          primary: data.mine[0].norad,
          secondary: data.others[0].norad
        },
        text: `Assess conjunction: ${data.mine[0].name} × ${data.others[0].name}`
      }));
    }
    render(body, nodes.length ? nodes
      : el('p', { className: 'muted', text: 'No satellites registered yet.' }));

    const select = $('sat');
    for (const asset of data.mine) {
      if (![...select.options].some((option) => option.value === asset.norad)) {
        select.appendChild(el('option', {
          attrs: { value: asset.norad }, text: `${asset.name} · yours` }));
      }
    }
  } catch (error) {
    render(body, errorBox(error.message));
  }
}

async function loadDemoCredentials() {
  try {
    const info = await api('/auth/demo-info');
    const box = $('democreds');
    render(box,
      el('div', { text: 'DEMO OPERATORS' }),
      ...info.operators.map((operator) => el('div', {}, [
        el('b', { text: operator.username }), ` / ${operator.password} · `,
        operator.satellite
      ])),
      el('span', { className: 'warn', text: info.warning }));
    box.hidden = false;
  } catch {
    // Production deployments return 404 here. That is the correct behaviour:
    // published credentials must not be discoverable from a live service.
    $('democreds').hidden = true;
  }
}

/* =====================================================================
   RENDERING HELPERS FOR ASSESSMENTS
   ===================================================================== */
function conjunctionCard(conjunction, options = {}) {
  const tca = formatTca(conjunction.tca_utc);
  const node = el('button', {
    className: 'conj',
    attrs: { type: 'button' },
    dataset: options.dataset || {},
    title: options.title || ''
  }, [
    el('div', { className: 'cn' }, [
      `${conjunction.primary_name || options.primaryName || 'primary'} × ${conjunction.secondary_name}`,
      conjunction.is_new ? el('span', { className: 'pill NEW', text: 'NEW' }) : null,
      pill(conjunction.risk)
    ]),
    el('div', { className: 'cm' }, [
      el('span', {}, ['miss ', el('b', { text: `${num(conjunction.miss_km)} km` })]),
      el('span', { text: `Pc ${sci(conjunction.pc)}` }),
      el('span', { text: `${num(conjunction.rel_speed_kms, 1)} km/s` })
    ]),
    el('div', { className: 'cm' }, [
      el('span', { text: `TCA ${tca.label}` }),
      el('span', { text: tca.countdown })
    ]),
    conjunction.pc_methods_agree === false
      ? el('div', { className: 'hintline',
          text: 'Pc methods disagree — quadrature reported, see docs' })
      : null,
    options.hint ? el('div', { className: 'hintline', text: options.hint }) : null
  ]);
  if (!options.dataset) node.disabled = true;
  return node;
}

function safetyChecks(checks) {
  return checks.map((check) => el('div', { className: 'chk' }, [
    el('span', { className: `mk ${check.pass ? 'ok' : 'no'}`, text: check.pass ? '✓' : '✕' }),
    el('div', {}, [
      check.check.replace(/_/g, ' '),
      check.reason ? el('div', { className: 'why', text: check.reason }) : null
    ])
  ]));
}

function tradeTable(options, chosenLead) {
  return el('table', { className: 'trade' }, [
    el('thead', {}, el('tr', {}, [
      el('th', { text: 'Burn at' }), el('th', { text: 'Δv m/s' }),
      el('th', { text: 'Propellant kg' }), el('th', { text: 'New miss km' })
    ])),
    el('tbody', {}, options.map((option) => el('tr', {
      className: option.burn_lead_time_h === chosenLead ? 'best' : ''
    }, [
      el('td', { text: `T−${num(option.burn_lead_time_h, 2)} h` }),
      el('td', { text: num(option.dv_ms, 4) }),
      el('td', { text: num(option.fuel_kg, 4) }),
      el('td', { text: num(option.new_miss_km, 2) })
    ])))
  ]);
}

function monitorDecisionBlock(pc, missKm, tcaHours) {
  const margin = pc > 0 ? PC_ACTION_THRESHOLD / pc : Infinity;
  const confidence = (margin > 100 || missKm > 5) ? 'High'
    : margin > 10 ? 'Medium' : 'Low';
  const nextReview = tcaHours < 6 ? '1 hour — TCA approaching'
    : margin < 10 ? '2 hours — margin is thin' : '6 hours';
  return el('div', { className: 'decblock monitor' }, [
    decRow('Decision', el('span', { className: 'decv monitor-t', text: 'MONITOR' })),
    decRow('Reason', el('span', { className: 'decv2',
      text: `Pc ${sci(pc)} is below the ${sci(PC_ACTION_THRESHOLD)} action threshold`
        + (isFinite(margin) ? ` by about ${margin >= 10 ? Math.round(margin) : num(margin, 1)}×` : '') })),
    decRow('Miss distance', el('span', { className: 'decv2', text: `${num(missKm)} km` })),
    decRow('Next review', el('span', { className: 'decv2', text: nextReview })),
    decRow('Confidence', el('span', { className: 'decv2', text: confidence }))
  ]);
}

const decRow = (key, value) => el('div', { className: 'decrow' }, [
  el('span', { className: 'dk', text: key }), value
]);

function showTelemetry(...nodes) {
  $('telemetry').hidden = false;
  render($('result'), ...nodes);
}

/* =====================================================================
   MONITOR
   ===================================================================== */
let autoMonitorTimer = null;

async function runMonitor() {
  const watchlist = $('watchlist').value.trim();
  const status = $('monitorStatus');
  const alerts = $('monitorAlerts');
  const stats = $('monitorStats');
  render(status, spinner('Screening the catalog…'));
  try {
    const query = watchlist ? `?watchlist=${encodeURIComponent(watchlist)}` : '';
    const data = await api(`/monitor${query}`);
    stats.hidden = false;
    $('ms_sats').textContent = data.satellites_monitored ?? '—';
    $('ms_conj').textContent = data.total_conjunctions ?? 0;
    $('ms_new').textContent = data.new_alerts ?? 0;
    $('ms_crit').textContent = data.n_critical ?? 0;
    $('ms_warn').textContent = data.n_warning ?? 0;
    $('ms_time').textContent = data.scan_ms == null ? '—'
      : data.scan_ms < 1000 ? `${data.scan_ms} ms` : `${(data.scan_ms / 1000).toFixed(1)} s`;

    const at = new Date(data.checked_utc || Date.now()).toISOString().slice(11, 19);
    const headline = data.new_alerts > 0 ? `${data.new_alerts} new alert(s)`
      : data.total_conjunctions > 0 ? `${data.total_conjunctions} tracked, none new`
      : 'fleet clear';
    status.textContent = `Last scan ${at}Z · ${headline}`
      + ` · catalog ${num(data.catalog_age_hours, 1)} h old`;

    const conjunctions = data.conjunctions || [];
    if (!conjunctions.length) {
      const nodes = [el('p', { className: 'muted',
        text: 'No conjunctions above the alert thresholds. Fleet is clear.' })];
      if (data.errors?.length) {
        nodes.push(el('p', { className: 'muted small',
          text: `${data.errors.length} satellite(s) skipped: ${data.errors[0].error}` }));
      }
      render(alerts, ...nodes);
      return;
    }
    render(alerts, conjunctions.slice(0, 12).map((conjunction) =>
      conjunctionCard(conjunction, {
        dataset: {
          action: 'load-conjunction',
          primary: conjunction.primary_norad,
          secondary: conjunction.secondary_norad,
          miss: conjunction.miss_km,
          tca: conjunction.tca_h
        },
        hint: 'Load into the simulator'
      })));
  } catch (error) {
    stats.hidden = true;
    render(status, errorBox(error.message));
  }
}

function toggleAutoMonitor() {
  const button = $('btnAutoMonitor');
  if (autoMonitorTimer) {
    clearInterval(autoMonitorTimer);
    autoMonitorTimer = null;
    button.textContent = 'Auto-scan: off';
    button.setAttribute('aria-pressed', 'false');
  } else {
    runMonitor();
    autoMonitorTimer = setInterval(runMonitor, AUTO_MONITOR_INTERVAL_MS);
    button.textContent = 'Auto-scan: on';
    button.setAttribute('aria-pressed', 'true');
  }
}

/* =====================================================================
   OPERATOR ASSETS
   ===================================================================== */
async function addMySatellite() {
  const message = $('mysat_msg');
  if (!session.token) { openLogin('Sign in to register a satellite.'); return; }
  message.textContent = 'Registering…';
  try {
    const data = await api('/my/satellites', {
      method: 'POST', auth: true,
      body: {
        name: $('newsat_name').value.trim(),
        tle1: $('newsat_t1').value.trim(),
        tle2: $('newsat_t2').value.trim()
      }
    });
    message.textContent = `Registered NORAD ${data.norad}.`;
    $('newsat_t1').value = '';
    $('newsat_t2').value = '';
    await refreshAuthUi();
  } catch (error) {
    message.textContent = error.message;
  }
}

async function screenMySatellite() {
  if (!session.token) { openLogin('Sign in to screen your assets.'); return; }
  const norad = $('sat').value;
  showTelemetry(spinner('Screening against the full public catalog…'));
  try {
    const data = await api(
      `/user/screen/${encodeURIComponent(norad)}?hours=24&gate_km=25&max_results=10`,
      { auth: true });
    const nodes = [
      kv('Asset', data.primary.name),
      kv('Objects screened', String(data.objects_screened)),
      kv('After geometric filter', String(data.objects_after_geometric_filter)),
      kv('Scan time', `${num(data.scan_seconds, 2)} s`),
      kv('Catalog age', `${num(data.catalog_age_hours, 1)} h`)
    ];
    if (!data.conjunctions.length) {
      nodes.push(el('div', { className: 'decblock monitor' }, [
        decRow('Result', el('span', { className: 'decv monitor-t', text: 'CLEAR' })),
        decRow('Detail', el('span', { className: 'decv2',
          text: `no close approach inside ${data.gate_km} km over the next ${data.window_hours} h` }))
      ]));
    } else {
      nodes.push(el('h3', { className: 'hd', text: `Close approaches · next ${data.window_hours} h` }));
      for (const conjunction of data.conjunctions) {
        nodes.push(conjunctionCard(
          { ...conjunction, primary_name: data.primary.name },
          {
            dataset: {
              action: 'load-screen-result',
              miss: conjunction.miss_km,
              tca: conjunction.tca_in_hours,
              name: conjunction.secondary_name
            },
            hint: 'Load into the simulator'
          }));
      }
    }
    nodes.push(el('p', { className: 'muted small', text: data.note }));
    showTelemetry(...nodes);
  } catch (error) {
    showTelemetry(errorBox(error.message));
  }
}

async function assessRegisteredPair(primary, secondary) {
  showTelemetry(spinner('Refining time of closest approach…'));
  try {
    for (const [norad, color, isPrimary] of
         [[primary, COLOR.primary, true], [secondary, COLOR.threat, false]]) {
      try {
        const orbit = await api(`/user/orbit/${encodeURIComponent(norad)}?points=240`);
        if (isPrimary) {
          primaryTrack = orbit.track_km;
          primaryOrbit = drawOrbit(primaryTrack, color, primaryOrbit);
          if (!primaryDot) primaryDot = makeDot(COLOR.primary, 200);
          camRadius = Math.max(...primaryTrack.map((p) => Math.hypot(...p))) * 3.4;
          updateCamera();
        } else {
          threatTrack = orbit.track_km;
          threatOrbit = drawOrbit(threatTrack, color, threatOrbit);
          if (!threatDot) threatDot = makeDot(COLOR.threat, 180);
        }
      } catch { /* the assessment is still useful without the 3-D track */ }
    }

    const data = await api('/user/assess', {
      method: 'POST', body: { primary_norad: primary, secondary_norad: secondary }
    });
    const tca = formatTca(data.tca_utc);
    if (tca.epoch) { tcaEpochMs = tca.epoch; $('clock').hidden = false;
      $('clocksub').textContent = 'real conjunction · live countdown'; }

    showTelemetry(
      kv('Conjunction', `${data.primary.name} × ${data.secondary.name}`),
      kv('Time of closest approach', `${tca.label} (${tca.countdown})`),
      kv('Miss distance', `${num(data.miss_km)} km`),
      kv('Relative speed', `${num(data.rel_speed_kms)} km/s`),
      kv('Separation', `${num(data.mahalanobis, 1)} σ`),
      kv('Collision probability', sci(data.pc), ' ', pill(data.risk_level)),
      kv('Covariance', data.covariance),
      el('div', { className: 'row' }, [
        el('button', {
          className: 'ghost success', attrs: { type: 'button' },
          dataset: { action: 'plan-user', primary, secondary },
          text: `Plan: ${data.primary.name}`
        }),
        el('button', {
          className: 'ghost', attrs: { type: 'button' },
          dataset: { action: 'plan-user', primary: secondary, secondary: primary },
          text: `Plan: ${data.secondary.name}`
        })
      ]),
      el('p', { className: 'muted small',
        text: 'Maneuver authority is restricted to the owning operator. Planning a '
          + 'burn for an asset you do not own is refused.' })
    );
  } catch (error) {
    showTelemetry(errorBox(error.message));
  }
}

async function planRegisteredManeuver(primary, secondary) {
  const previous = [...$('result').childNodes];
  showTelemetry(...previous, spinner('Requesting an owner-authorized plan…'));
  try {
    const data = await api('/user/plan', {
      method: 'POST', auth: true,
      body: { primary_norad: primary, secondary_norad: secondary }
    });
    if (data.verdict === 'NO_ACTION') {
      showTelemetry(...previous, el('div', { className: 'decblock monitor' }, [
        decRow('Decision', el('span', { className: 'decv monitor-t', text: 'MONITOR' })),
        decRow('Reason', el('span', { className: 'decv2',
          text: typeof data.recommendation === 'string'
            ? data.recommendation : 'Pc below the action threshold' }))
      ]));
      return;
    }
    const recommendation = data.recommendation;
    showTelemetry(...previous,
      el('div', { className: 'manbanner' }, [
        el('span', { className: 'manbanner-l', text: 'PLAN MANEUVER' }),
        el('span', { className: 'manbanner-r', text: `OWNER: ${data.authorized_operator}` })
      ]),
      kv('Burn Δv', el('span', { className: 'v big' }, [
        num(recommendation.dv_magnitude_ms, 4), el('span', { className: 'unit', text: ' m/s' })])),
      kv('Direction', recommendation.direction),
      kv('Execute at', `T−${num(recommendation.burn_lead_time_h, 2)} h`),
      kv('Propellant', `${num(recommendation.fuel_kg, 4)} kg`),
      kv('Miss distance', `${num(data.assessment.miss_km)} km`,
        el('span', { className: 'arrow', text: '→' }),
        `${num(recommendation.predicted_new_miss_km, 2)} km`),
      kv('Collision probability', sci(data.assessment.pc),
        el('span', { className: 'arrow', text: '→' }), sci(recommendation.predicted_new_pc)),
      el('h3', { className: 'hd', text: 'Lead-time trade' }),
      tradeTable(data.options || [], recommendation.burn_lead_time_h),
      el('h3', { className: 'hd', text: 'Safety validation' }),
      ...safetyChecks(data.safety_validation.checks),
      el('div', { className: `verdict ${data.verdict}`, text: `${data.verdict} · owner-authorized` }),
      el('p', { className: 'muted small', text: data.note })
    );
  } catch (error) {
    const isAuthz = error.status === 401 || error.status === 403;
    showTelemetry(...previous, isAuthz
      ? el('div', { className: 'decblock plan' }, [
          decRow('Security', el('span', { className: 'decv', text: 'DENIED' })),
          decRow('Reason', el('span', { className: 'decv2', text: error.message }))
        ])
      : errorBox(error.message));
  }
}

/* =====================================================================
   PRIMARY ASSET / ORBIT
   ===================================================================== */
async function resolveSatellite() {
  const query = $('query').value.trim();
  const message = $('resolveMsg');
  if (!query) { message.textContent = 'Enter a NORAD id, name, or COSPAR id.'; return; }
  message.textContent = `Resolving “${query}”…`;
  try {
    const data = await api(`/resolve?q=${encodeURIComponent(query)}`);
    if (data.candidates) {
      render(message, `${data.total_matches} matches — be more specific:`,
        ...data.candidates.map((candidate) =>
          el('div', { text: `${candidate.name} (${candidate.norad})` })));
      return;
    }
    const select = $('sat');
    if (![...select.options].some((option) => option.value === String(data.norad))) {
      select.appendChild(el('option', {
        attrs: { value: String(data.norad) }, text: `${data.name} (${data.norad})` }));
    }
    select.value = String(data.norad);
    message.textContent = `${data.name} · NORAD ${data.norad} · matched by ${data.matched_by}`;
    await loadOrbit();
  } catch (error) {
    message.textContent = error.message;
  }
}

async function loadOrbit() {
  const norad = $('sat').value;
  if (!norad) return;
  try {
    const isOperatorAsset = [...$('sat').options]
      .find((option) => option.value === norad)?.textContent.endsWith('· yours');
    const path = isOperatorAsset
      ? `/user/orbit/${encodeURIComponent(norad)}?points=240`
      : `/orbit/${encodeURIComponent(norad)}?points=240`;
    const data = await api(path);
    primaryTrack = data.track_km;
    primaryOrbit = drawOrbit(primaryTrack, COLOR.primary, primaryOrbit);
    if (!primaryDot) primaryDot = makeDot(COLOR.primary, 200);
    clearScenario();
    camRadius = Math.max(...primaryTrack.map((p) => Math.hypot(...p))) * 3.4;
    camPhi = 1.05;
    updateCamera();
  } catch (error) {
    showBanner(error.message);
  }
}

/* =====================================================================
   MANEUVER PLANNING
   ===================================================================== */
async function runPlan() {
  const norad = parseInt($('sat').value, 10);
  if (!Number.isFinite(norad)) { showBanner('Select a primary asset first.'); return; }
  const body = {
    primary_norad: norad,
    synthetic_miss_km: parseFloat($('miss').value),
    tca_hours: parseFloat($('tca').value),
    sat_mass_kg: parseFloat($('mass').value),
    fuel_available_kg: parseFloat($('fuel').value)
  };
  const button = $('btnPlan');
  button.disabled = true;
  showTelemetry(spinner('Planning and validating · re-screening the catalog…'));
  try {
    const data = await api('/maneuver-plan', { method: 'POST', body });
    const pre = data.pre_maneuver;

    if (primaryTrack.length) {
      threatTrack = crossingTrack(primaryTrack, 14, 8);
      threatOrbit = drawOrbit(threatTrack, COLOR.threat, threatOrbit);
      if (!threatDot) threatDot = makeDot(COLOR.threat, 180);
      markTca(primaryTrack[0]);
      tcaSeconds = body.tca_hours * 3600;
      tcaStartedAt = Date.now();
      tcaEpochMs = null;
      $('clock').hidden = false;
      $('clocksub').textContent = 'what-if scenario · accelerated view';
    }

    const header = [
      kv('Primary', data.primary),
      kv('Threat', data.secondary),
      kv('Miss distance', `${num(pre.miss_distance_km)} km`),
      kv('Collision probability', sci(pre.pc), ' ', pill(pre.risk_level)),
      kv('Covariance source', data.covariance_source.replace(/_/g, ' '))
    ];

    if (data.verdict === 'NO_ACTION') {
      if (primaryTrack.length) { tcaSeconds = null; $('clock').hidden = true; }
      showTelemetry(...header,
        monitorDecisionBlock(pre.pc, pre.miss_distance_km, body.tca_hours));
      return;
    }

    const recommendation = data.recommendation;
    const executeAt = new Date(
      Date.now() + (body.tca_hours - recommendation.burn_lead_time_h) * 3600000);
    const allPassed = data.safety_validation.checks.every((check) => check.pass);

    showTelemetry(...header,
      el('div', { className: 'manbanner' }, [
        el('span', { className: 'manbanner-l', text: 'PLAN MANEUVER' }),
        el('span', { className: 'manbanner-r', text: 'READY FOR REVIEW' })
      ]),
      el('div', { className: 'decblock plan' }, [
        decRow('Reason', el('span', { className: 'decv2',
          text: `Pc ${sci(pre.pc)} is at or above the ${sci(data.pc_action_threshold)} action threshold` }))
      ]),
      el('h3', { className: 'hd', text: 'Recommended maneuver' }),
      kv('Burn Δv', el('span', { className: 'v big' }, [
        num(recommendation.dv_magnitude_ms, 4), el('span', { className: 'unit', text: ' m/s' })])),
      kv('Direction', recommendation.direction),
      kv('Execute at', `T−${num(recommendation.burn_lead_time_h, 2)} h`),
      kv('Propellant required', `${num(recommendation.fuel_kg, 4)} kg`),
      kv('Miss distance', `${num(pre.miss_distance_km)} km`,
        el('span', { className: 'arrow', text: '→' }),
        `${num(recommendation.predicted_new_miss_km, 2)} km`),
      kv('Collision probability', sci(pre.pc),
        el('span', { className: 'arrow', text: '→' }), sci(recommendation.predicted_new_pc)),
      el('h3', { className: 'hd', text: 'Lead-time trade' }),
      tradeTable(data.options || [], recommendation.burn_lead_time_h),
      el('h3', { className: 'hd', text: 'Safety validation' }),
      ...safetyChecks(data.safety_validation.checks),
      el('div', { className: 'cmdbox' }, [
        el('div', { className: 'cmdhd', text: '▸ MANEUVER COMMAND (pending approval)' }),
        el('div', { className: 'cmdline' }, [
          'Fire thrusters: ',
          el('b', { text: `${num(recommendation.dv_magnitude_ms, 4)} m/s` }),
          ` ${recommendation.direction}`
        ]),
        el('div', { className: 'cmdline' }, [
          'Execute: ', el('b', { text: `${executeAt.toISOString().slice(0, 16).replace('T', ' ')} UTC` })
        ]),
        el('div', { className: 'cmdline',
          text: `Result: Pc → ${sci(recommendation.predicted_new_pc)}` }),
        el('p', { className: 'cmdnote',
          text: 'Decision-support recommendation. A human operator reviews and '
            + 'executes via the spacecraft’s flight system — ISDMAAS does not '
            + 'command the spacecraft.' })
      ]),
      el('div', { className: 'row' }, [
        el('button', { className: 'ghost success', attrs: { type: 'button' },
          dataset: { action: 'approve' }, text: 'Acknowledge recommendation' }),
        el('button', { className: 'ghost danger', attrs: { type: 'button' },
          dataset: { action: 'reject' }, text: 'Reject' })
      ]),
      el('div', { className: `verdict ${data.verdict}`,
        text: allPassed ? `${data.verdict} · all safety checks passed` : data.verdict })
    );
  } catch (error) {
    showTelemetry(errorBox(error.message));
  } finally {
    button.disabled = false;
  }
}

function loadIntoSimulator(missKm, tcaHours, label) {
  $('miss').value = Math.max(Number(missKm), 0.001).toFixed(3);
  $('tca').value = Math.max(Number(tcaHours), 0.05).toFixed(2);
  const card = $('miss').closest('.card');
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  showBanner(`Loaded ${label} into the simulator — ${num(missKm)} km at T−${num(tcaHours, 1)} h.`,
    'info', 6000);
}

/* =====================================================================
   AUTONOMOUS SEQUENCE
   ===================================================================== */
const SEQUENCE_STEPS = [
  'Load current orbit', 'Screen the catalog', 'Detect conjunction',
  'Compute Pc', 'Select Δv and timing', 'Validate safety', 'Present decision'
];

function buildSequence() {
  render($('seqsteps'), SEQUENCE_STEPS.map((label, index) =>
    el('li', { attrs: { id: `step${index}` } }, [
      el('span', { className: 'n', text: String(index + 1) }),
      el('span', { className: 't', text: label })
    ])));
}

function setStep(index, state) {
  const node = $(`step${index}`);
  if (node) node.className = state || '';
}

function setStage(message, sub, color) {
  let stage = $('stage');
  if (!stage) {
    stage = el('div', { attrs: { id: 'stage' } });
    document.body.appendChild(stage);
  }
  render(stage,
    el('span', { className: 'stagelab', text: 'Autonomous loop' }),
    el('span', { className: 'stagemsg', text: message }),
    sub ? el('span', { className: 'stagesub', text: sub }) : null);
  if (color) stage.querySelector('.stagemsg').style.color = color;
  stage.hidden = false;
}

const hideStage = () => setTimeout(() => { const s = $('stage'); if (s) s.hidden = true; }, 1500);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function runAutonomous() {
  const norad = parseInt($('sat').value, 10);
  if (!Number.isFinite(norad)) { showBanner('Select a primary asset first.'); return; }
  $('seq').hidden = false;
  buildSequence();
  $('btnAuto').disabled = true;
  $('btnPlan').disabled = true;
  showTelemetry(spinner('Autonomous loop running…'));

  try {
    setStep(0, 'active');
    setStage('Loading current orbit…');
    await loadOrbit();
    await sleep(600);
    setStep(0, 'done');

    setStep(1, 'active');
    setStage('Screening the catalog for conjunctions…');
    const request = api('/autonomous', {
      method: 'POST',
      body: {
        primary_norad: norad, mode: 'auto',
        inject_miss_km: parseFloat($('miss').value),
        inject_tca_h: parseFloat($('tca').value),
        sat_mass_kg: parseFloat($('mass').value),
        fuel_available_kg: parseFloat($('fuel').value)
      }
    });
    await sleep(900);
    setStep(1, 'done');
    const data = await request;

    setStep(2, 'active');
    const isReal = data.detection_source === 'real_catalog';
    setStage(`Threat detected — ${data.threat.name}`,
      `${isReal ? 'real catalog object' : 'simulated threat'} · miss ${num(data.threat.miss_km)} km`,
      '#ff6079');
    if (primaryTrack.length) {
      threatTrack = crossingTrack(primaryTrack, 14, 8);
      threatOrbit = drawOrbit(threatTrack, COLOR.threat, threatOrbit);
      if (!threatDot) threatDot = makeDot(COLOR.threat, 180);
      markTca(primaryTrack[0]);
      tcaSeconds = data.threat.tca_h * 3600;
      tcaStartedAt = Date.now();
      tcaEpochMs = null;
      $('clock').hidden = false;
      $('clocksub').textContent = 'projected lead time · accelerated view';
    }
    await sleep(1100);
    setStep(2, data.action_required ? 'alert' : 'done');

    setStep(3, 'active');
    setStage('Computing collision probability…');
    await sleep(700);
    setStep(3, 'done');

    const header = [
      kv('Detection', isReal ? 'Real catalog object' : 'Simulated threat'),
      kv('Covariance', data.covariance_source.replace(/_/g, ' ')),
      kv('Threat', data.threat.name),
      kv('Miss distance', `${num(data.threat.miss_km)} km`),
      kv('Collision probability', sci(data.pre_maneuver.pc), ' ', pill(data.pre_maneuver.risk_level))
    ];

    if (data.verdict === 'NO_ACTION') {
      setStage('Pc below threshold — monitoring only', '', '#3ddc9a');
      [4, 5, 6].forEach((index) => setStep(index, 'done'));
      showTelemetry(...header, monitorDecisionBlock(
        data.pre_maneuver.pc, data.pre_maneuver.miss_distance_km, data.threat.tca_h));
      hideStage();
      return;
    }

    const recommendation = data.recommendation;
    setStep(4, 'active');
    setStage('Selecting Δv, direction and burn time…',
      `${num(recommendation.dv_magnitude_ms, 4)} m/s ${recommendation.direction}`
      + ` · T−${num(recommendation.burn_lead_time_h, 2)} h`);
    await sleep(1200);
    setStep(4, 'done');

    setStep(5, 'active');
    setStage('Validating fuel, orbit and post-burn catalog re-screen…');
    await sleep(1200);
    setStep(5, data.verdict === 'APPROVED' ? 'done' : 'alert');

    setStep(6, 'active');
    if (data.verdict === 'APPROVED') {
      setStage('Recommendation ready for uplink',
        `new miss ${num(recommendation.predicted_new_miss_km, 2)} km`, '#3ddc9a');
      if (primaryOrbit && primaryTrack.length) {
        postOrbit = drawOrbit(
          primaryTrack.map((p) => p.map((c) => c * 1.012)), COLOR.post, postOrbit);
      }
      setStep(6, 'done');
    } else {
      setStage('Held for review — a safety check failed', '', '#ffc53d');
      setStep(6, 'alert');
    }
    hideStage();

    showTelemetry(...header,
      el('h3', { className: 'hd', text: 'Autonomous recommendation' }),
      kv('Burn Δv', el('span', { className: 'v big' }, [
        num(recommendation.dv_magnitude_ms, 4), el('span', { className: 'unit', text: ' m/s' })])),
      kv('Execute at', `T−${num(recommendation.burn_lead_time_h, 2)} h`),
      kv('Propellant', `${num(recommendation.fuel_kg, 4)} kg`),
      kv('Outcome', `${num(recommendation.predicted_new_miss_km, 2)} km`,
        el('span', { className: 'arrow', text: '→' }), sci(recommendation.predicted_new_pc)),
      el('h3', { className: 'hd', text: 'Safety validation' }),
      ...safetyChecks(data.safety_validation.checks),
      el('div', { className: `verdict ${data.verdict}`, text: data.verdict }),
      el('p', { className: 'muted small',
        text: data.autonomous_action === 'RECOMMENDED_FOR_UPLINK'
          ? 'The loop reached this decision without a human step. The decision is '
            + 'still a recommendation: a human operator commands the spacecraft.'
          : 'Held for operator review — at least one safety check did not pass.' })
    );
  } catch (error) {
    showTelemetry(errorBox(error.message));
    hideStage();
  } finally {
    $('btnAuto').disabled = false;
    $('btnPlan').disabled = false;
  }
}

/* =====================================================================
   SOCRATES + LIBRARY
   ===================================================================== */
async function loadSocrates() {
  const list = $('socrList');
  render(list, spinner('Fetching CelesTrak SOCRATES…'));
  let path = `/socrates?order=${encodeURIComponent($('socrOrder').value)}&maxrows=20`;
  if ($('socrScope').value === 'primary' && $('sat').value) {
    path += `&norad=${encodeURIComponent($('sat').value)}`;
  }
  try {
    const data = await api(path);
    if (!data.conjunctions?.length) {
      render(list, el('p', { className: 'muted', text: 'No conjunctions returned.' }));
      return;
    }
    const nodes = data.conjunctions.slice(0, 20).map((row) => {
      const risk = row.max_prob == null ? 'UNDETERMINED'
        : row.max_prob > 1e-4 ? 'CRITICAL' : row.max_prob > 1e-5 ? 'HIGH'
        : row.max_prob > 1e-7 ? 'ELEVATED' : 'NOMINAL';
      const tca = formatTca(row.tca);
      return el('button', {
        className: 'conj', attrs: { type: 'button' },
        dataset: {
          action: 'load-socrates',
          primary: row.p_norad, miss: row.miss_km, tca: row.tca || '',
          name: `${row.p_name} × ${row.s_name}`
        }
      }, [
        el('div', { className: 'cn' }, [`${row.p_name} × ${row.s_name}`, pill(risk)]),
        el('div', { className: 'cm' }, [
          el('span', {}, ['miss ', el('b', { text: `${num(row.miss_km)} km` })]),
          el('span', { text: row.max_prob == null ? 'Pc not published' : `MaxPc ${sci(row.max_prob)}` }),
          el('span', { text: `${num(row.rel_speed_kms, 1)} km/s` })
        ]),
        el('div', { className: 'cm' }, [
          el('span', { text: `TCA ${tca.label}` }), el('span', { text: tca.countdown })
        ]),
        el('div', { className: 'hintline', text: 'Load into the simulator' })
      ]);
    });
    nodes.push(el('p', { className: 'muted small',
      text: `${data.count} conjunctions · parsed as ${data.format}`
        + (data.max_pc_available ? '' : ' · this view does not publish Pc') }));
    render(list, ...nodes);
  } catch (error) {
    render(list, errorBox(error.message));
  }
}

async function loadLibrary() {
  const tree = $('libraryTree');
  try {
    const data = await api('/library');
    const nodes = Object.entries(data.categories).map(([category, items]) =>
      el('details', { className: 'libcat' }, [
        el('summary', { text: `${category} (${items.length})` }),
        el('div', { className: 'libitems' }, items.map((item) =>
          el('button', {
            className: 'libev', attrs: { type: 'button' },
            dataset: { action: 'show-event', key: item.key }
          }, [
            el('span', { className: `qd ${item.data_quality}` }),
            el('span', { text: item.title })
          ])))
      ]));
    if (nodes.length) nodes[0].open = true;
    render(tree, ...nodes);
  } catch (error) {
    render(tree, el('p', { className: 'muted', text: error.message }));
  }
}

async function showEvent(key) {
  const detail = $('eventDetail');
  render(detail, spinner('Loading event…'));
  try {
    const event = await api(`/library/${encodeURIComponent(key)}`);
    const quality = event.data_quality;
    const qualityLabel = {
      quantified: 'FULL REPLAY DATA',
      partial: 'PARTIAL DATA',
      record_only: 'RECORD ONLY — geometry not public'
    }[quality] || quality;

    const nodes = [
      el('span', { className: `evtag ${quality}`, text: qualityLabel }),
      kv('Event', event.title),
      event.date_utc ? kv('Date', event.date_utc.slice(0, 10)) : null,
      kv('Outcome', event.outcome),
      el('p', { className: 'evfact', text: event.public_facts }),
      el('p', { className: 'evsrc', text: `Sources: ${(event.sources || []).join('; ')}` })
    ];

    if (quality === 'quantified' || quality === 'partial') {
      const geometry = event.geometry || {};
      const fields = [
        ['final_pc', 'Documented Pc', sci],
        ['esa_threshold_pc', 'Operator threshold', sci],
        ['socrates_pc', 'Public SOCRATES Pc', sci],
        ['altitude_km', 'Intercept altitude', (v) => `~${v} km`],
        ['altitude_raise_m', 'Documented maneuver', (v) => `+${v} m altitude`],
        ['rel_velocity_kms', 'Relative velocity', (v) => `${v} km/s`],
        ['miss_km', 'Miss distance', (v) => `${v} km`]
      ];
      for (const [field, label, format] of fields) {
        if (geometry[field] != null) nodes.push(kv(label, format(geometry[field])));
        else if (field in geometry) nodes.push(kv(label, 'not public'));
      }
      const mode = event.replay_mode || 'conjunction';
      if (mode === 'conjunction' && event.primary?.norad) {
        nodes.push(el('button', {
          className: 'ghost danger', attrs: { type: 'button' },
          dataset: { action: 'replay-event', key, norad: event.primary.norad },
          text: 'Run ISDMAAS replay'
        }));
      } else if (mode === 'debris_cloud' && event.debris_group) {
        nodes.push(el('button', {
          className: 'ghost', attrs: { type: 'button' },
          dataset: { action: 'replay-debris', group: event.debris_group },
          text: 'Visualize the debris cloud (live)'
        }));
      }
      if (event.replay_note) nodes.push(el('p', { className: 'muted small', text: event.replay_note }));
    } else {
      nodes.push(el('p', { className: 'muted', style: '',
        text: 'A full replay needs per-event conjunction geometry — miss distance, '
          + 'Pc, covariance — from a Conjunction Data Message. Those are '
          + 'operator-restricted and were not publicly released for this event, '
          + 'so it is listed as a documented record without invented numbers.' }));
    }
    render(detail, ...nodes.filter(Boolean));
  } catch (error) {
    render(detail, errorBox(error.message));
  }
}

async function replayEvent(key, norad) {
  showTelemetry(spinner('Running the ISDMAAS replay…'));
  try {
    try {
      const orbit = await api(`/orbit/${encodeURIComponent(norad)}?points=240`);
      primaryTrack = orbit.track_km;
      primaryOrbit = drawOrbit(primaryTrack, COLOR.primary, primaryOrbit);
      if (!primaryDot) primaryDot = makeDot(COLOR.primary, 200);
      camRadius = Math.max(...primaryTrack.map((p) => Math.hypot(...p))) * 3.4;
      updateCamera();
    } catch { /* the objects may no longer be in the active catalog */ }

    const data = await api('/autonomous', {
      method: 'POST',
      body: {
        primary_norad: Number(norad), mode: 'historical', event_key: key,
        sat_mass_kg: 1000, fuel_available_kg: 10
      }
    });

    if (primaryTrack.length) {
      threatTrack = crossingTrack(primaryTrack, 14, 8);
      threatOrbit = drawOrbit(threatTrack, COLOR.threat, threatOrbit);
      if (!threatDot) threatDot = makeDot(COLOR.threat, 180);
      markTca(primaryTrack[0]);
      tcaSeconds = (data.threat.tca_h || 8) * 3600;
      tcaStartedAt = Date.now();
      tcaEpochMs = null;
      $('clock').hidden = false;
      $('clocksub').textContent = 'documented geometry · accelerated view';
    }

    const historical = data.historical || {};
    const nodes = [
      kv('Event', historical.title || ''),
      kv('Outcome', historical.outcome || ''),
      kv('Threat', data.threat.name),
      kv('Documented miss', `${num(data.threat.miss_km)} km`),
      data.pre_maneuver.pc != null
        ? kv('ISDMAAS Pc', sci(data.pre_maneuver.pc), ' ', pill(data.pre_maneuver.risk_level))
        : null
    ];
    if (typeof data.recommendation === 'object') {
      nodes.push(
        el('h3', { className: 'hd', text: 'Recommended avoidance' }),
        kv('Burn Δv', `${num(data.recommendation.dv_magnitude_ms, 4)} m/s`),
        kv('Outcome', `${num(data.recommendation.predicted_new_miss_km, 2)} km`,
          el('span', { className: 'arrow', text: '→' }),
          sci(data.recommendation.predicted_new_pc)));
    }
    nodes.push(
      el('div', { className: `verdict ${data.verdict || 'NO_ACTION'}`, text: data.verdict || '' }),
      el('p', { className: 'muted small',
        text: 'ISDMAAS assessment on documented geometry — not a detection from '
          + 'public element sets.' }));
    showTelemetry(...nodes.filter(Boolean));
  } catch (error) {
    showTelemetry(errorBox(error.message));
  }
}

async function replayDebrisCloud(group) {
  $('group').value = group;
  showTelemetry(spinner('Loading the real debris cloud…'));
  await loadLive();
  showTelemetry(
    kv('Debris group', group),
    kv('Fragments tracked', String(liveObjects.length)),
    el('p', { className: 'muted',
      text: 'These are the real tracked fragments this event created, propagating '
        + 'live now. The parent satellite was destroyed, so there is no orbit to '
        + 'replay — the debris cloud is the lasting signature.' }));
}

/* =====================================================================
   SYNC + BOOTSTRAP
   ===================================================================== */
async function refreshData(force) {
  const icon = $('refreshicon');
  const label = $('synctxt');
  const led = $('syncled');
  icon.classList.add('spinning');
  label.textContent = 'Refreshing…';
  led.className = 'led warn';
  try {
    if (force) await api('/sync-now', { method: 'POST', timeoutMs: 300000 });
    const status = await api('/sync-status');
    if (status.last_sync.catalog) {
      const age = status.catalog_age_hours;
      label.textContent = `Data ${new Date(status.last_sync.catalog)
        .toISOString().slice(11, 16)}Z · ${status.catalog_objects} objects`;
      led.className = age != null && age > 24 ? 'led warn' : 'led on';
      if (age != null && age > 24) {
        showBanner(`The catalog is ${num(age, 1)} hours old. Screening results reflect `
          + 'element sets from that time.', 'info');
      }
    } else {
      label.textContent = 'Data unavailable';
      led.className = 'led fail';
    }
    if (force && currentLiveGroup && liveObjects.length) await loadLive();
  } catch (error) {
    label.textContent = 'Refresh failed';
    led.className = 'led fail';
    showBanner(error.message);
  } finally {
    icon.classList.remove('spinning');
  }
}

/**
 * One delegated click handler for the whole page.
 *
 * Every interactive element built from API data carries a data-action instead of
 * an inline handler, which is what allows the strict CSP and removes the
 * attribute-injection surface entirely.
 */
function handleDelegatedClick(event) {
  const target = event.target.closest('[data-action]');
  if (!target) return;
  const { action } = target.dataset;
  const data = target.dataset;
  switch (action) {
    case 'select-asset':
      $('sat').value = data.norad;
      loadOrbit();
      break;
    case 'assess-pair':
      assessRegisteredPair(data.primary, data.secondary);
      break;
    case 'plan-user':
      planRegisteredManeuver(data.primary, data.secondary);
      break;
    case 'load-conjunction':
    case 'load-screen-result':
      loadIntoSimulator(data.miss, data.tca, data.name || `NORAD ${data.secondary}`);
      break;
    case 'load-socrates': {
      const parsed = parseUtc(data.tca);
      const hours = parsed ? Math.max(0.05, (parsed.getTime() - Date.now()) / 3600000) : 8;
      const select = $('sat');
      if (![...select.options].some((option) => option.value === data.primary)) {
        select.appendChild(el('option', {
          attrs: { value: data.primary }, text: `NORAD ${data.primary}` }));
      }
      select.value = data.primary;
      loadOrbit().then(() => loadIntoSimulator(data.miss, hours, data.name));
      break;
    }
    case 'show-event':
      showEvent(data.key);
      break;
    case 'replay-event':
      replayEvent(data.key, data.norad);
      break;
    case 'replay-debris':
      replayDebrisCloud(data.group);
      break;
    case 'approve':
      target.textContent = 'Acknowledged';
      target.disabled = true;
      $('result').appendChild(el('p', { className: 'muted small',
        text: 'Recommendation acknowledged. In an operational deployment the '
          + 'command would be uplinked by the operator’s flight system at the '
          + 'scheduled time. ISDMAAS’s role ends at the recommendation.' }));
      break;
    case 'reject':
      $('result').appendChild(el('p', { className: 'muted small',
        text: 'Recommendation rejected. ISDMAAS would re-plan with adjusted '
          + 'constraints, or keep monitoring as the geometry updates.' }));
      break;
    default:
      break;
  }
}

async function init() {
  session.load();
  initScene();

  const clock = $('utc');
  const tick = () => { clock.textContent = `${new Date().toISOString().slice(11, 19)} UTC`; };
  tick();
  setInterval(tick, 1000);

  document.addEventListener('click', handleDelegatedClick);
  $('loginForm').addEventListener('submit', doLogin);
  $('lg_register').addEventListener('click', doRegister);
  $('lg_skip').addEventListener('click', closeLogin);
  $('syncstat').addEventListener('click', () => refreshData(true));
  $('btnMonitor').addEventListener('click', runMonitor);
  $('btnAutoMonitor').addEventListener('click', toggleAutoMonitor);
  $('btnAddSat').addEventListener('click', addMySatellite);
  $('btnScreenMine').addEventListener('click', screenMySatellite);
  $('btnResolve').addEventListener('click', resolveSatellite);
  $('btnLoadOrbit').addEventListener('click', loadOrbit);
  $('btnLoadLive').addEventListener('click', loadLive);
  $('btnClearLive').addEventListener('click', clearLive);
  $('btnPlan').addEventListener('click', runPlan);
  $('btnAuto').addEventListener('click', runAutonomous);
  $('btnSocrates').addEventListener('click', loadSocrates);
  $('query').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') { event.preventDefault(); resolveSatellite(); }
  });
  addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !$('loginov').hidden) closeLogin();
  });

  try {
    await api('/health');
    $('apiled').className = 'led on';
    $('apitxt').textContent = 'Link active';
  } catch (error) {
    $('apiled').className = 'led fail';
    $('apitxt').textContent = 'Service offline';
    showBanner(`${error.message} Start it with: uvicorn phase11_api:app --port 8000`);
    return;
  }

  try {
    const satellites = await api('/satellites');
    const select = $('sat');
    for (const [norad, info] of Object.entries(satellites)) {
      select.appendChild(el('option', {
        attrs: { value: norad },
        text: info.state_available ? info.name : `${info.name} · no calibrated state`
      }));
    }
  } catch { /* the asset picker degrades to whatever resolve() adds */ }

  loadLibrary();
  refreshData(false);
  loadOrbit();
  await refreshAuthUi();
  if (!session.token) {
    await loadDemoCredentials();
    openLogin();
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
