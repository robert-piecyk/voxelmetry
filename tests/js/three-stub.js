// Minimal stand-in for the three.js API surface the viewer uses.
//
// jsdom has no WebGL, so the real library cannot run here. Stubbing it means
// the harness exercises OUR logic -- scene assembly, geometry decoding, DOM
// wiring, the clip-plane maths -- and records every call so the test can
// assert on what the viewer asked three.js to do.

'use strict';

const calls = { materials: [], meshes: [], planes: [], geometries: [] };

class Vector3 {
  constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
  set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
  copy(v) { return this.set(v.x, v.y, v.z); }
  clone() { return new Vector3(this.x, this.y, this.z); }
  add(v) { this.x += v.x; this.y += v.y; this.z += v.z; return this; }
  multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; }
  getComponent(i) { return [this.x, this.y, this.z][i]; }
  setFromMatrixColumn() { return this; }
  length() { return Math.hypot(this.x, this.y, this.z); }
}

class Plane {
  constructor(normal = new Vector3(), constant = 0) {
    this.normal = normal; this.constant = constant;
  }
  set(normal, constant) {
    this.normal = normal.clone(); this.constant = constant;
    calls.planes.push({ normal: [normal.x, normal.y, normal.z], constant });
    return this;
  }
  // Sign convention the viewer relies on: a point is kept when this is > 0.
  distanceToPoint(p) {
    return this.normal.x * p.x + this.normal.y * p.y + this.normal.z * p.z + this.constant;
  }
}

class Color { constructor(v) { this.value = v; } }

class BufferAttribute {
  constructor(array, itemSize) { this.array = array; this.itemSize = itemSize; }
}

class BufferGeometry {
  constructor() { this.attributes = {}; this.index = null; }
  setAttribute(name, attr) { this.attributes[name] = attr; }
  setIndex(attr) { this.index = attr; }
  computeVertexNormals() {
    this.normalsComputed = true;
    calls.geometries.push({
      vertices: this.attributes.position ? this.attributes.position.array.length / 3 : 0,
      indices: this.index ? this.index.array.length : 0,
    });
  }
}

class Material {
  constructor(opts = {}) { Object.assign(this, opts); calls.materials.push(this); }
}
class MeshPhongMaterial extends Material {}

class Object3D {
  constructor() { this.children = []; this.position = new Vector3(); this.visible = true; }
  add(child) { this.children.push(child); }
}
class Group extends Object3D {}
class Scene extends Object3D {}
class Mesh extends Object3D {
  constructor(geometry, material) {
    super(); this.geometry = geometry; this.material = material;
    calls.meshes.push(this);
  }
}
class Light extends Object3D { constructor(color, intensity) { super(); this.intensity = intensity; } }

class PerspectiveCamera extends Object3D {
  constructor(fov, aspect, near, far) {
    super(); this.fov = fov; this.aspect = aspect; this.near = near; this.far = far;
    this.up = new Vector3(0, 1, 0); this.matrix = {};
  }
  updateProjectionMatrix() { this.projectionUpdated = true; }
  lookAt(v) { this.lookingAt = v.clone ? v.clone() : v; }
}

class WebGLRenderer {
  constructor(opts) {
    this.options = opts;
    this.localClippingEnabled = false;
    this.renders = 0;
    this.domElement = global.document.createElement('canvas');
    this.domElement.setPointerCapture = function () {};
    this.domElement.releasePointerCapture = function () {};
    this.domElement.hasPointerCapture = function () { return false; };
    Object.defineProperty(this.domElement, 'clientWidth', { value: 900, configurable: true });
    Object.defineProperty(this.domElement, 'clientHeight', { value: 600, configurable: true });
  }
  setPixelRatio(r) { this.pixelRatio = r; }
  setSize(w, h) { this.size = [w, h]; }
  render() { this.renders++; }
}

module.exports = {
  calls,
  THREE: {
    Vector3, Plane, Color, BufferAttribute, BufferGeometry, MeshPhongMaterial,
    Group, Scene, Mesh, PerspectiveCamera, WebGLRenderer,
    AmbientLight: Light, DirectionalLight: Light,
    DoubleSide: 2, FrontSide: 0,
  },
};
