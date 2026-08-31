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

    payload = {
        "title": scene.title,
        "subtitle": scene.subtitle,
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
    --bg:#f4f4f2; --panel:#ffffff; --ink:#1c1c1a; --muted:#6b6b66;
    --line:#dedcd6; --accent:#2f6f9f; --stage-top:#e8e8e4; --stage-bot:#cfcfc9;
    --shadow:0 1px 2px rgba(0,0,0,.06),0 8px 24px rgba(0,0,0,.06);
  }
  @media (prefers-color-scheme:dark){
    :root:not([data-theme="light"]){
      --bg:#16181a; --panel:#1e2124; --ink:#e8e6e1; --muted:#9a9892;
      --line:#31353a; --accent:#7fb2dc; --stage-top:#232629; --stage-bot:#111315;
      --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
    }
  }
  :root[data-theme="dark"]{
    --bg:#16181a; --panel:#1e2124; --ink:#e8e6e1; --muted:#9a9892;
    --line:#31353a; --accent:#7fb2dc; --stage-top:#232629; --stage-bot:#111315;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  #app{display:grid;grid-template-columns:296px 1fr;height:100vh;height:100dvh;}
  #panel{background:var(--panel);border-right:1px solid var(--line);overflow-y:auto;
    display:flex;flex-direction:column;}
  #stage{position:relative;min-width:0;background:var(--stage-bot);}
  canvas{display:block;width:100%;height:100%}
  .sec{padding:14px 16px;border-bottom:1px solid var(--line)}
  .sec:last-child{border-bottom:none}
  h1{font-size:15px;margin:0;letter-spacing:-.01em;font-weight:650}
  .sub{color:var(--muted);font-size:12px;margin-top:3px;word-break:break-word}
  h2{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
    margin:0 0 9px;font-weight:650}
  .row{display:flex;align-items:center;gap:9px;padding:5px 0}
  .row label{display:flex;align-items:center;gap:9px;cursor:pointer;flex:1;min-width:0}
  .sw{width:11px;height:11px;border-radius:3px;flex:none;
    box-shadow:0 0 0 1px rgba(0,0,0,.18) inset}
  .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .val{color:var(--muted);font-variant-numeric:tabular-nums;font-size:12px;flex:none}
  input[type=checkbox]{accent-color:var(--accent);flex:none;margin:0}
  input[type=range]{width:100%;accent-color:var(--accent);margin:2px 0}
  .ctl{margin-bottom:11px}
  .ctl:last-child{margin-bottom:0}
  .ctl > span{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);
    margin-bottom:2px}
  .btns{display:flex;gap:6px;flex-wrap:wrap}
  button{font:inherit;font-size:12px;padding:5px 10px;border-radius:7px;cursor:pointer;
    border:1px solid var(--line);background:transparent;color:var(--ink)}
  button:hover{border-color:var(--accent);color:var(--accent)}
  button[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:#fff}
  select{font:inherit;font-size:12px;padding:4px 7px;border-radius:7px;
    border:1px solid var(--line);background:var(--panel);color:var(--ink)}
  table{width:100%;border-collapse:collapse;font-size:12px;
    font-variant-numeric:tabular-nums}
  th{text-align:right;font-weight:600;color:var(--muted);padding:3px 0;
    border-bottom:1px solid var(--line);font-size:10.5px;text-transform:uppercase;
    letter-spacing:.05em}
  th:first-child{text-align:left}
  td{padding:4px 0;border-bottom:1px solid var(--line);text-align:right}
  td:first-child{text-align:left;max-width:110px;overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap}
  tr:last-child td{border-bottom:none}
  .flag{color:var(--muted);cursor:help}
  dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:12px}
  dt{color:var(--muted)}
  dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;word-break:break-word}
  #hud{position:absolute;left:14px;bottom:12px;color:var(--muted);font-size:11px;
    pointer-events:none;font-variant-numeric:tabular-nums;text-shadow:0 1px 2px var(--stage-bot)}
  #scale{position:absolute;right:16px;bottom:12px;color:var(--muted);font-size:11px;
    text-align:center;pointer-events:none}
  #scalebar{height:3px;background:currentColor;opacity:.65;margin-bottom:3px;border-radius:2px}
  #err{position:absolute;inset:0;display:none;place-items:center;padding:24px;
    text-align:center;color:var(--muted);font-size:13px}
  @media (max-width:760px){
    #app{grid-template-columns:1fr;grid-template-rows:minmax(0,1fr) auto}
    #panel{order:2;border-right:none;border-top:1px solid var(--line);max-height:44vh}
    #stage{order:1}
  }
</style>

<div id="app">
  <aside id="panel">
    <div class="sec">
      <h1 id="ttl"></h1>
      <div class="sub" id="sub"></div>
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
      var flag = m.resolution_limited
        ? ' <span class="flag" title="Spans under 5 voxels on its thinnest axis; shape figures are indicative only.">*</span>'
        : '';
      return '<tr><td>' + esc(m.name) + flag + '</td><td>' + m.volume_ml.toFixed(1) +
        '</td><td>' + m.max_diameter_mm.toFixed(1) +
        '</td><td>' + m.sphericity.toFixed(2) + '</td></tr>';
    }).join('');
    tbl.innerHTML = head + (rows || '<tr><td colspan="4">no labels</td></tr>');

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
    if (spinning){ cam.theta += 0.0035; place(); }
    updateScale();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  place(); resize(); applyClip(); frame();
})();
</script>
"""
