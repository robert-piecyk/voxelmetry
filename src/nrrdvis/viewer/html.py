"""Render a :class:`~nrrdvis.viewer.scene.Scene` to a self-contained HTML page.

Geometry travels as base64-encoded binary in a single inline JSON blob and is
decoded into typed arrays in the browser, which is roughly four times smaller
than the JSON number lists Plotly emits and avoids parsing hundreds of
thousands of decimal strings on load.

Only three.js is fetched from a CDN; everything else, including camera
controls, is inlined so the page has one external dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scene import Scene

#: Pinned so a page rendered today still renders identically in a year.
THREE_JS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"


def render(scene: Scene, standalone: bool = True) -> str:
    """Render a scene to HTML.

    Args:
        scene: The scene to render.
        standalone: When True, emit a complete document that opens from disk.
            When False, emit only title, styles and body content, for embedding
            in a host that supplies its own document skeleton.

    Returns:
        The HTML source.
    """
    lo, hi = scene.bounds
    centre = ((lo + hi) / 2.0).tolist()
    radius = float(np.linalg.norm(hi - lo)) / 2.0 or 100.0

    volumes = [m.volume_ml for m in scene.measurements]
    summary = {
        "n_structures": len(scene.measurements),
        "total_volume_ml": float(sum(volumes)),
        "largest_volume_ml": float(max(volumes)) if volumes else 0.0,
        "n_flagged": sum(1 for m in scene.measurements if m.resolution_limited),
    }

    payload = {
        "title": scene.title,
        "subtitle": scene.subtitle,
        "summary": summary,
        "centre": centre,
        "radius": radius,
        "bounds": {"min": lo.tolist(), "max": hi.tolist()},
        "structures": [m.to_payload() for m in scene.meshes],
        "measurements": [m.as_dict() for m in scene.measurements],
        "provenance": scene.provenance,
    }

    body = _TEMPLATE.replace("__SCENE_JSON__", json.dumps(payload, separators=(",", ":")))
    body = body.replace("__THREE_CDN__", THREE_JS_CDN)
    body = body.replace("__TITLE__", _escape(scene.title))

    if not standalone:
        return body
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<style>html,body{margin:0;padding:0}</style>\n'
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )


def write(scene: Scene, path: str | Path, standalone: bool = True) -> Path:
    """Render a scene and write it to ``path``.

    Args:
        scene: The scene to render.
        path: Destination ``.html`` file; parents are created as needed.
        standalone: See :func:`render`.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(scene, standalone=standalone), encoding="utf-8")
    return path


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


_TEMPLATE = r"""<title>__TITLE__</title>
<style>
  :root{
    /* Neutrals carry a slight cool bias toward the steel accent, so the panel
       reads as an instrument surface rather than as undecided grey. --warn is
       deliberately not the accent: it encodes data quality and must not be
       mistaken for an interactive affordance. */
    --bg:#eceef0; --panel:#fafbfc; --ink:#181c20; --muted:#5f6a73;
    --line:#d8dde2; --line-soft:#e6eaee; --accent:#2f6f9f; --accent-ink:#ffffff;
    --warn:#8a6a1f; --warn-bg:#f5edd8;
    --stage-bot:#c7ccd1; --hover:#eef1f4; --focus:#2f6f9f;
  }
  @media (prefers-color-scheme:dark){
    :root:not([data-theme="light"]){
      --bg:#101317; --panel:#181c21; --ink:#e6e9ec; --muted:#98a2ab;
      --line:#2b3138; --line-soft:#232930; --accent:#7fb2dc; --accent-ink:#0e1216;
      --warn:#d9bd72; --warn-bg:#2e2717;
      --stage-bot:#0c0e11; --hover:#1e242a; --focus:#7fb2dc;
    }
  }
  :root[data-theme="dark"]{
    --bg:#101317; --panel:#181c21; --ink:#e6e9ec; --muted:#98a2ab;
    --line:#2b3138; --line-soft:#232930; --accent:#7fb2dc; --accent-ink:#0e1216;
    --warn:#d9bd72; --warn-bg:#2e2717;
    --stage-bot:#0c0e11; --hover:#1e242a; --focus:#7fb2dc;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:400 13px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased;}
  #app{display:grid;grid-template-columns:300px 1fr;height:100vh;height:100dvh;}
  #panel{background:var(--panel);border-right:1px solid var(--line);overflow-y:auto;
    display:flex;flex-direction:column;}
  #stage{position:relative;min-width:0;background:var(--stage-bot);}
  canvas{display:block;width:100%;height:100%}

  /* Type scale: 10 / 12 / 13 / 15. Four steps, nothing between them. */
  .sec{padding:15px 16px;border-bottom:1px solid var(--line)}
  .sec:last-child{border-bottom:none}
  h1{font-size:15px;margin:0;letter-spacing:-.012em;font-weight:650;
    text-wrap:balance;line-height:1.3}
  .sub{color:var(--muted);font-size:12px;margin-top:4px;word-break:break-word}
  h2{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
    margin:0 0 10px;font-weight:650}

  /* Summary reads before the per-structure detail. */
  #sum{margin:0;display:grid;grid-template-columns:1fr 1fr;gap:11px 12px}
  #sum div{min-width:0}
  #sum dt{font-size:10px;text-transform:uppercase;letter-spacing:.07em;
    color:var(--muted);margin-bottom:2px}
  #sum dd{margin:0;font-size:15px;font-weight:600;font-variant-numeric:tabular-nums;
    letter-spacing:-.015em}
  #sum dd small{font-size:11px;font-weight:400;color:var(--muted);margin-left:1px}

  .row{display:flex;align-items:center;gap:9px;border-radius:6px;
    padding:5px 6px;margin:0 -6px;transition:background .12s}
  .row:hover{background:var(--hover)}
  .row label{display:flex;align-items:center;gap:9px;cursor:pointer;flex:1;min-width:0}
  .sw{width:10px;height:10px;border-radius:2.5px;flex:none;
    box-shadow:0 0 0 1px rgba(0,0,0,.22) inset}
  .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .val{color:var(--muted);font-variant-numeric:tabular-nums;font-size:12px;flex:none}
  input[type=checkbox]{accent-color:var(--accent);flex:none;margin:0;cursor:pointer}
  input[type=range]{width:100%;accent-color:var(--accent);margin:3px 0;cursor:pointer}

  .ctl{margin-bottom:12px}
  .ctl:last-child{margin-bottom:0}
  .ctl > span{display:flex;justify-content:space-between;align-items:center;
    font-size:12px;color:var(--muted);margin-bottom:3px;gap:8px}
  .ctl b{font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}

  .btns{display:flex;gap:6px;flex-wrap:wrap}
  button{font:inherit;font-size:12px;padding:5px 10px;border-radius:6px;cursor:pointer;
    border:1px solid var(--line);background:transparent;color:var(--ink);
    transition:background .12s,border-color .12s,color .12s}
  button:hover{background:var(--hover);border-color:var(--muted)}
  button[aria-pressed=true]{background:var(--accent);border-color:var(--accent);
    color:var(--accent-ink)}
  select{font:inherit;font-size:12px;padding:3px 6px;border-radius:6px;
    border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
  :focus-visible{outline:2px solid var(--focus);outline-offset:2px;border-radius:4px}

  table{width:100%;border-collapse:collapse;font-size:12px;
    font-variant-numeric:tabular-nums}
  th{text-align:right;font-weight:650;color:var(--muted);padding:0 0 5px;
    border-bottom:1px solid var(--line);font-size:10px;text-transform:uppercase;
    letter-spacing:.06em;white-space:nowrap}
  th:first-child{text-align:left}
  td{padding:5px 0;border-bottom:1px solid var(--line-soft);text-align:right}
  td:first-child{text-align:left;max-width:104px;overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap}
  tr:last-child td{border-bottom:none}

  /* Data quality reads as a chip, not a footnote glyph. */
  .chip{display:inline-block;margin-left:5px;padding:0 4px;border-radius:3px;
    background:var(--warn-bg);color:var(--warn);font-size:9.5px;font-weight:650;
    letter-spacing:.05em;text-transform:uppercase;vertical-align:1px;cursor:help}
  .note{margin-top:10px;font-size:11px;line-height:1.45;color:var(--muted);
    display:flex;gap:6px;align-items:flex-start}
  .note .chip{margin:0;flex:none;margin-top:1px}

  dl#prov{margin:0;display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:12px}
  dl#prov dt{color:var(--muted)}
  dl#prov dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;
    word-break:break-word}

  #hud{position:absolute;left:14px;bottom:12px;color:var(--muted);font-size:11px;
    pointer-events:none;font-variant-numeric:tabular-nums}
  #scale{position:absolute;right:16px;bottom:12px;color:var(--muted);font-size:11px;
    text-align:center;pointer-events:none;font-variant-numeric:tabular-nums}
  #scalebar{height:3px;background:currentColor;opacity:.7;margin-bottom:3px;border-radius:2px}
  #err{position:absolute;inset:0;display:none;place-items:center;padding:28px;
    text-align:center;color:var(--muted);font-size:13px;line-height:1.5}
  #hint{position:absolute;left:14px;top:12px;color:var(--muted);font-size:11px;
    pointer-events:none;opacity:.85}

  @media (prefers-reduced-motion:reduce){ *{transition-duration:.01ms !important} }
  @media (max-width:760px){
    #app{grid-template-columns:1fr;grid-template-rows:minmax(0,1fr) auto}
    #panel{order:2;border-right:none;border-top:1px solid var(--line);max-height:46vh}
    #stage{order:1}
    #hint{display:none}
  }
</style>

<div id="app">
  <aside id="panel">
    <div class="sec">
      <h1 id="ttl"></h1>
      <div class="sub" id="sub"></div>
    </div>
    <div class="sec">
      <h2>Summary</h2>
      <dl id="sum"></dl>
    </div>
    <div class="sec">
      <h2>Structures</h2>
      <div id="list"></div>
    </div>
    <div class="sec">
      <h2>Display</h2>
      <div class="ctl">
        <span><em style="font-style:normal">Opacity</em><b id="opv">100%</b></span>
        <input type="range" id="op" min="10" max="100" value="100">
      </div>
      <div class="ctl">
        <span>
          <em style="font-style:normal">Clip plane</em>
          <select id="ax">
            <option value="-1">off</option>
            <option value="0">sagittal (X)</option>
            <option value="1">coronal (Y)</option>
            <option value="2">axial (Z)</option>
          </select>
        </span>
        <input type="range" id="clip" min="0" max="100" value="50" disabled>
      </div>
      <div class="btns">
        <button id="reset">Reset view</button>
        <button id="wire" aria-pressed="false">Wireframe</button>
        <button id="spin" aria-pressed="false">Spin</button>
      </div>
    </div>
    <div class="sec">
      <h2>Measurements</h2>
      <table id="tbl"></table>
    </div>
    <div class="sec">
      <h2>Acquisition</h2>
      <dl id="prov"></dl>
    </div>
  </aside>
  <div id="stage">
    <div id="hint">drag to orbit &middot; scroll to zoom &middot; shift-drag to pan</div>
    <div id="hud"></div>
    <div id="scale"><div id="scalebar"></div><span id="scaletx"></span></div>
    <div id="err"></div>
  </div>
</div>

<script src="__THREE_CDN__"></script>
<script>
(function(){
  "use strict";
  var SCENE = __SCENE_JSON__;

  var stage = document.getElementById('stage');
  var errBox = document.getElementById('err');
  function fail(msg){ errBox.textContent = msg; errBox.style.display = 'grid'; }

  if (typeof THREE === 'undefined') {
    fail('The 3-D library could not be loaded. The measurements in the panel are unaffected.');
    buildPanel(); return;
  }

  function b64(str, Type){
    var bin = atob(str), n = bin.length, bytes = new Uint8Array(n);
    for (var i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i);
    return new Type(bytes.buffer);
  }

  // ---- renderer -----------------------------------------------------------
  var renderer = new THREE.WebGLRenderer({antialias:true, alpha:false});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  stage.appendChild(renderer.domElement);

  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100000);

  var css = getComputedStyle(document.documentElement);
  function tone(name, fallback){
    var v = css.getPropertyValue(name).trim();
    return v ? new THREE.Color(v) : new THREE.Color(fallback);
  }
  scene.background = tone('--stage-bot', '#cfcfc9');

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  var key = new THREE.DirectionalLight(0xffffff, 0.75); key.position.set(1, 1, 1);
  var fill = new THREE.DirectionalLight(0xffffff, 0.35); fill.position.set(-1, -0.4, -0.8);
  scene.add(key); scene.add(fill);

  var C = SCENE.centre, R = SCENE.radius || 100;
  var pivot = new THREE.Vector3(C[0], C[1], C[2]);
  var root = new THREE.Group(); scene.add(root);

  // ---- geometry -----------------------------------------------------------
  var planes = [new THREE.Plane(new THREE.Vector3(-1,0,0), 0)];
  renderer.localClippingEnabled = true;

  var items = SCENE.structures.map(function(s){
    var geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(b64(s.vertices, Float32Array), 3));
    geom.setIndex(new THREE.BufferAttribute(b64(s.faces, Uint32Array), 1));
    geom.computeVertexNormals();
    // transparent is left off at full opacity: an always-transparent material
    // is depth-sorted per object, which makes lesions inside an organ flicker
    // in and out depending on camera angle. It is switched on only when the
    // opacity slider actually calls for blending.
    var mat = new THREE.MeshPhongMaterial({
      color: new THREE.Color(s.color), specular: 0x111111, shininess: 18,
      transparent: false, opacity: 1, side: THREE.DoubleSide,
      clippingPlanes: [], clipShadows: true
    });
    var mesh = new THREE.Mesh(geom, mat);
    root.add(mesh);
    return {def:s, mesh:mesh, mat:mat, on:true};
  });

  if (!items.length) fail('This scene contains no surfaces.');

  // ---- camera orbit -------------------------------------------------------
  // Written inline rather than pulled from a second CDN file: it is ~50 lines
  // and removes a dependency that could version-drift away from three.js core.
  var cam = {theta: 0.9, phi: 1.15, dist: R * 3.0, pan: new THREE.Vector3()};
  var home = {theta: cam.theta, phi: cam.phi, dist: cam.dist};
  var vel = {theta:0, phi:0};

  function place(){
    var p = Math.max(0.02, Math.min(Math.PI - 0.02, cam.phi));
    cam.phi = p;
    var x = cam.dist * Math.sin(p) * Math.cos(cam.theta);
    var y = cam.dist * Math.cos(p);
    var z = cam.dist * Math.sin(p) * Math.sin(cam.theta);
    var target = pivot.clone().add(cam.pan);
    camera.position.set(target.x + x, target.y + y, target.z + z);
    camera.up.set(0, 1, 0);
    camera.lookAt(target);
    key.position.copy(camera.position);
  }

  var drag = null;
  function pointer(e){ return e.touches ? e.touches[0] : e; }
  renderer.domElement.addEventListener('pointerdown', function(e){
    drag = {x:e.clientX, y:e.clientY, pan:e.shiftKey || e.button === 1 || e.button === 2};
    renderer.domElement.setPointerCapture(e.pointerId);
  });
  renderer.domElement.addEventListener('pointermove', function(e){
    if (!drag) return;
    var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan){
      var scale = cam.dist / 900;
      var right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
      var up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      cam.pan.add(right.multiplyScalar(-dx * scale)).add(up.multiplyScalar(dy * scale));
    } else {
      vel.theta = -dx * 0.007; vel.phi = -dy * 0.007;
      cam.theta += vel.theta; cam.phi += vel.phi;
    }
    place();
  });
  function endDrag(e){
    if (drag && e.pointerId !== undefined && renderer.domElement.hasPointerCapture(e.pointerId))
      renderer.domElement.releasePointerCapture(e.pointerId);
    drag = null;
  }
  renderer.domElement.addEventListener('pointerup', endDrag);
  renderer.domElement.addEventListener('pointercancel', endDrag);
  renderer.domElement.addEventListener('contextmenu', function(e){ e.preventDefault(); });
  renderer.domElement.addEventListener('wheel', function(e){
    e.preventDefault();
    cam.dist *= Math.exp((e.deltaY > 0 ? 1 : -1) * 0.11);
    cam.dist = Math.max(R * 0.25, Math.min(R * 14, cam.dist));
    place();
  }, {passive:false});

  // ---- panel --------------------------------------------------------------
  function buildPanel(){
    document.getElementById('ttl').textContent = SCENE.title;
    document.getElementById('sub').textContent = SCENE.subtitle;

    var sum = SCENE.summary || {};
    var sumEl = document.getElementById('sum');
    function stat(label, value, unit){
      var wrap = document.createElement('div');
      var dt = document.createElement('dt'); dt.textContent = label;
      var dd = document.createElement('dd'); dd.textContent = value;
      if (unit){
        var u = document.createElement('small'); u.textContent = unit;
        dd.appendChild(u);
      }
      wrap.appendChild(dt); wrap.appendChild(dd); sumEl.appendChild(wrap);
    }
    var totalFaces = SCENE.structures.reduce(function(a, st){ return a + st.n_faces; }, 0);
    stat('Structures', String(sum.n_structures != null ? sum.n_structures : 0));
    stat('Total volume', (sum.total_volume_ml || 0).toFixed(1), ' mL');
    stat('Largest', (sum.largest_volume_ml || 0).toFixed(1), ' mL');
    stat('Triangles', totalFaces.toLocaleString());

    var list = document.getElementById('list');
    SCENE.structures.forEach(function(s, i){
      var vol = s.metadata && s.metadata.volume_ml;
      var row = document.createElement('div'); row.className = 'row';
      var lab = document.createElement('label');
      var cb = document.createElement('input');
      cb.type = 'checkbox'; cb.checked = true; cb.dataset.i = i;
      var sw = document.createElement('span'); sw.className = 'sw';
      sw.style.background = s.color;
      var nm = document.createElement('span'); nm.className = 'nm';
      nm.textContent = s.name;
      lab.appendChild(cb); lab.appendChild(sw); lab.appendChild(nm);
      var val = document.createElement('span'); val.className = 'val';
      val.textContent = (vol != null) ? vol.toFixed(1) + ' mL' : '';
      row.appendChild(lab); row.appendChild(val);
      list.appendChild(row);
      cb.addEventListener('change', function(){
        if (items[i]) { items[i].on = cb.checked; items[i].mesh.visible = cb.checked; }
      });
    });

    var tbl = document.getElementById('tbl');
    var head = '<tr><th>structure</th><th>vol mL</th><th>max &#216; mm</th><th>spher.</th></tr>';
    var rows = SCENE.measurements.map(function(m){
      // Encoded as a chip rather than a footnote glyph, so a reader scanning
      // the column sees which rows to discount without hunting for a legend.
      var flag = m.resolution_limited
        ? ' <span class="chip" title="Spans under 5 voxels on its thinnest axis.' +
          ' Volume and diameter hold; shape figures are indicative only.">thin</span>'
        : '';
      return '<tr><td>' + esc(m.name) + flag + '</td><td>' + m.volume_ml.toFixed(1) +
        '</td><td>' + m.max_diameter_mm.toFixed(1) +
        '</td><td>' + m.sphericity.toFixed(2) + '</td></tr>';
    }).join('');
    tbl.innerHTML = head + (rows || '<tr><td colspan="4">no labels</td></tr>');

    if (sum.n_flagged) {
      var note = document.createElement('div');
      note.className = 'note';
      var chip = document.createElement('span');
      chip.className = 'chip'; chip.textContent = 'thin';
      var text = document.createElement('span');
      text.textContent = sum.n_flagged + ' of ' + sum.n_structures +
        ' structures span under 5 voxels on their thinnest axis.' +
        ' Volume and diameter hold; shape figures are indicative only.';
      note.appendChild(chip); note.appendChild(text);
      tbl.parentNode.appendChild(note);
    }

    var prov = document.getElementById('prov'), html = '';
    Object.keys(SCENE.provenance).forEach(function(k){
      html += '<dt>' + esc(k) + '</dt><dd>' + esc(SCENE.provenance[k]) + '</dd>';
    });
    prov.innerHTML = html;
  }
  function esc(t){
    return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
  buildPanel();

  // ---- controls -----------------------------------------------------------
  var opEl = document.getElementById('op'), opv = document.getElementById('opv');
  opEl.addEventListener('input', function(){
    var v = opEl.value / 100; opv.textContent = opEl.value + '%';
    var blend = v < 1;
    items.forEach(function(it){
      it.mat.opacity = v;
      it.mat.transparent = blend;
      // Skipping the depth write lets structures behind a translucent organ
      // show through instead of being culled by it.
      it.mat.depthWrite = !blend;
      it.mat.needsUpdate = true;
    });
  });

  var axEl = document.getElementById('ax'), clipEl = document.getElementById('clip');
  var AXES = [new THREE.Vector3(-1,0,0), new THREE.Vector3(0,-1,0), new THREE.Vector3(0,0,-1)];
  function applyClip(){
    var a = parseInt(axEl.value, 10);
    clipEl.disabled = (a < 0);
    if (a < 0){ items.forEach(function(it){ it.mat.clippingPlanes = []; }); return; }
    var lo = SCENE.bounds.min[a], hi = SCENE.bounds.max[a];
    var at = lo + (hi - lo) * (clipEl.value / 100);
    // three.js keeps the half-space where normal.p + constant > 0. With
    // normal = -axis that reduces to p[a] < at, so the slider cuts away
    // everything beyond its position along the chosen axis.
    planes[0].set(AXES[a], at);
    items.forEach(function(it){ it.mat.clippingPlanes = planes; });
  }
  axEl.addEventListener('change', applyClip);
  clipEl.addEventListener('input', applyClip);

  var wireBtn = document.getElementById('wire');
  wireBtn.addEventListener('click', function(){
    var on = wireBtn.getAttribute('aria-pressed') !== 'true';
    wireBtn.setAttribute('aria-pressed', String(on));
    items.forEach(function(it){ it.mat.wireframe = on; });
  });

  var spinBtn = document.getElementById('spin'), spinning = false;
  // Spin never starts on its own, so a viewer who asked the OS for reduced
  // motion is never given movement they did not request. If they turn it on
  // deliberately it runs, but gently.
  var reduceMotion = window.matchMedia
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  spinBtn.addEventListener('click', function(){
    spinning = !spinning; spinBtn.setAttribute('aria-pressed', String(spinning));
  });

  document.getElementById('reset').addEventListener('click', function(){
    cam.theta = home.theta; cam.phi = home.phi; cam.dist = home.dist;
    cam.pan.set(0,0,0); place();
  });

  // ---- scale bar and HUD --------------------------------------------------
  var hud = document.getElementById('hud');
  var scaleEl = document.getElementById('scale');
  var barEl = document.getElementById('scalebar');
  var txEl = document.getElementById('scaletx');
  var faces = SCENE.structures.reduce(function(a, s){ return a + s.n_faces; }, 0);

  function updateScale(){
    // Millimetres per pixel at the pivot depth, from the vertical FOV.
    var h = renderer.domElement.clientHeight || 1;
    var mmPerPx = 2 * cam.dist * Math.tan(camera.fov * Math.PI / 360) / h;
    var targetPx = 108, raw = mmPerPx * targetPx;
    var pow = Math.pow(10, Math.floor(Math.log10(raw)));
    var nice = [1, 2, 5, 10].map(function(m){ return m * pow; })
      .reduce(function(best, v){
        return Math.abs(v - raw) < Math.abs(best - raw) ? v : best;
      });
    barEl.style.width = (nice / mmPerPx).toFixed(0) + 'px';
    txEl.textContent = (nice >= 10 ? (nice / 10).toFixed(nice % 10 ? 1 : 0) + ' cm'
                                   : nice.toFixed(0) + ' mm');
    hud.textContent = items.filter(function(i){ return i.on; }).length + ' of ' +
      items.length + ' structures  ·  ' + faces.toLocaleString() + ' triangles';
  }

  // ---- loop ---------------------------------------------------------------
  function resize(){
    var w = stage.clientWidth || 1, h = stage.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    updateScale();
  }
  window.addEventListener('resize', resize);

  function frame(){
    if (spinning){ cam.theta += reduceMotion ? 0.0012 : 0.0035; place(); }
    updateScale();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  place(); resize(); applyClip(); frame();
})();
</script>
"""
