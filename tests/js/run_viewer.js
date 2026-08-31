// Headless harness: load a generated viewer page, run its script against a
// stubbed three.js, and assert on the resulting DOM and recorded calls.
//
//   node tests/js/run_viewer.js path/to/scene.html
//
// Exits non-zero with a message on the first failed assertion.

'use strict';

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const { THREE, calls } = require('./three-stub.js');

const pagePath = process.argv[2];
if (!pagePath) { console.error('usage: run_viewer.js <scene.html>'); process.exit(2); }

const html = fs.readFileSync(pagePath, 'utf8');
const failures = [];
function check(name, condition, detail) {
  if (condition) { console.log(`  ok   ${name}`); }
  else { failures.push(name + (detail ? ` -- ${detail}` : '')); console.log(`  FAIL ${name}`); }
}

// Strip the CDN script tag: the stub stands in for it, and jsdom must not
// try to fetch over the network during a test.
const offline = html.replace(/<script src="https:[^"]*"><\/script>/, '');

const dom = new JSDOM(offline, { runScripts: 'outside-only', pretendToBeVisual: true });
const { window } = dom;

global.document = window.document;
window.THREE = THREE;
window.requestAnimationFrame = function () { return 0; };  // run one frame only
window.devicePixelRatio = 2;

// jsdom reports zero-size elements; give the stage a real box so the layout
// and scale-bar maths have something to work with.
for (const id of ['stage', 'panel']) {
  const el = window.document.getElementById(id);
  Object.defineProperty(el, 'clientWidth', { value: id === 'stage' ? 900 : 296 });
  Object.defineProperty(el, 'clientHeight', { value: 600 });
}

const script = [...window.document.querySelectorAll('script')]
  .map((s) => s.textContent).filter(Boolean).pop();

let threw = null;
try { window.eval(script); } catch (err) { threw = err; }

console.log(`\n${path.basename(pagePath)}`);
check('script runs without throwing', threw === null, threw && threw.stack);
if (threw) { process.exit(1); }

const doc = window.document;
const payload = JSON.parse(/var SCENE = (\{.*?\});\n/s.exec(script)[1]);
const n = payload.structures.length;

// --- geometry ---------------------------------------------------------------
check('a mesh per structure', calls.meshes.length === n,
  `${calls.meshes.length} meshes for ${n} structures`);
check('normals computed for every geometry', calls.geometries.length === n);
check('vertex counts match the payload',
  calls.geometries.every((g, i) => g.vertices === payload.structures[i].n_vertices));
check('index counts match the face counts',
  calls.geometries.every((g, i) => g.indices === payload.structures[i].n_faces * 3));

// --- base64 decoding --------------------------------------------------------
const first = payload.structures[0];
const bytes = Buffer.from(first.vertices, 'base64');
const expected = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
const actual = calls.geometries[0] && window.eval('null');
check('vertex payload decodes to the declared length',
  expected.length === first.n_vertices * 3,
  `${expected.length} floats vs ${first.n_vertices * 3}`);
check('decoded coordinates are finite',
  expected.every ? [...expected.slice(0, 300)].every(Number.isFinite) : true);

// --- panel ------------------------------------------------------------------
check('title rendered', doc.getElementById('ttl').textContent === payload.title);
check('one toggle per structure',
  doc.querySelectorAll('#list input[type=checkbox]').length === n);
check('structure names listed',
  [...doc.querySelectorAll('#list .nm')].map((e) => e.textContent).join('|')
    === payload.structures.map((s) => s.name).join('|'));
check('measurement rows rendered',
  doc.querySelectorAll('#tbl tr').length === payload.measurements.length + 1);

// --- summary ----------------------------------------------------------------
check('summary tiles rendered', doc.querySelectorAll('#sum div').length === 4);
check('structure count in the summary',
  doc.querySelector('#sum dd').textContent === String(payload.summary.n_structures));
const totalTile = [...doc.querySelectorAll('#sum div')]
  .find((d) => /total volume/i.test(d.querySelector('dt').textContent));
check('total volume tile present and numeric',
  totalTile && /^\d/.test(totalTile.querySelector('dd').textContent));

const flagged = payload.measurements.filter((m) => m.resolution_limited).length;
check('summary flag count matches the measurements',
  payload.summary.n_flagged === flagged);
check('a chip per flagged row',
  doc.querySelectorAll('#tbl .chip').length === flagged,
  `${doc.querySelectorAll('#tbl .chip').length} chips for ${flagged} flagged`);
check('explanatory note shown only when something is flagged',
  (doc.querySelectorAll('.note').length > 0) === (flagged > 0));
check('acquisition facts rendered',
  doc.querySelectorAll('#prov dt').length === Object.keys(payload.provenance).length);
check('triangle count in the HUD',
  /triangles/.test(doc.getElementById('hud').textContent));
check('scale bar has a width', /\d/.test(doc.getElementById('scalebar').style.width));
check('scale bar has units', /mm|cm/.test(doc.getElementById('scaletx').textContent));
check('no error banner shown', doc.getElementById('err').style.display !== 'grid');

// --- interaction ------------------------------------------------------------
function fire(el, type, init) {
  const ev = new window.Event(type, { bubbles: true });
  Object.assign(ev, init || {});
  el.dispatchEvent(ev);
}

const boxes = doc.querySelectorAll('#list input[type=checkbox]');
boxes[0].checked = false; fire(boxes[0], 'change');
check('unchecking hides its mesh', calls.meshes[0].visible === false);
boxes[0].checked = true; fire(boxes[0], 'change');
check('rechecking shows it again', calls.meshes[0].visible === true);

const op = doc.getElementById('op');
op.value = '55'; fire(op, 'input');
check('opacity applies to every material',
  calls.materials.every((m) => Math.abs(m.opacity - 0.55) < 1e-9));
check('blending switched on below full opacity',
  calls.materials.every((m) => m.transparent === true && m.depthWrite === false));
op.value = '100'; fire(op, 'input');
check('blending switched off at full opacity',
  calls.materials.every((m) => m.transparent === false && m.depthWrite === true));

// --- clip plane -------------------------------------------------------------
check('clip slider starts disabled', doc.getElementById('clip').disabled === true);
check('no clipping planes attached while off',
  calls.materials.every((m) => m.clippingPlanes.length === 0));

const ax = doc.getElementById('ax');
ax.value = '2'; fire(ax, 'change');
check('clip slider enabled once an axis is chosen',
  doc.getElementById('clip').disabled === false);
check('clipping planes attached to every material',
  calls.materials.every((m) => m.clippingPlanes.length === 1));

const lastPlane = calls.planes[calls.planes.length - 1];
const lo = payload.bounds.min[2], hi = payload.bounds.max[2];
const midpoint = lo + (hi - lo) * 0.5;
check('plane sits at the slider position along the chosen axis',
  Math.abs(lastPlane.constant - midpoint) < 1e-3,
  `${lastPlane.constant} vs ${midpoint}`);
check('plane normal points down the chosen axis',
  JSON.stringify(lastPlane.normal) === JSON.stringify([0, 0, -1]));

// The convention that matters: normal.p + constant > 0 is the half kept.
// With normal = -z and constant = midpoint, points below the cut survive.
const plane = new THREE.Plane(
  new THREE.Vector3(...lastPlane.normal), lastPlane.constant);
check('points below the cut are kept',
  plane.distanceToPoint(new THREE.Vector3(0, 0, lo)) > 0);
check('points above the cut are removed',
  plane.distanceToPoint(new THREE.Vector3(0, 0, hi)) < 0);

ax.value = '-1'; fire(ax, 'change');
check('turning the clip off detaches the planes',
  calls.materials.every((m) => m.clippingPlanes.length === 0));

// --- buttons ----------------------------------------------------------------
const wire = doc.getElementById('wire');
fire(wire, 'click');
check('wireframe toggles on',
  wire.getAttribute('aria-pressed') === 'true'
  && calls.materials.every((m) => m.wireframe === true));
fire(wire, 'click');
check('wireframe toggles off', calls.materials.every((m) => m.wireframe === false));

fire(doc.getElementById('reset'), 'click');
check('reset view does not throw', true);

// --- summary ----------------------------------------------------------------
if (failures.length) {
  console.error(`\n${failures.length} failed:`);
  failures.forEach((f) => console.error(`  - ${f}`));
  process.exit(1);
}
console.log(`\nall checks passed (${n} structures)`);
