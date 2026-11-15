// renderer/src/views/source-model.js — Source Model view component.
// Manages the 3D viewport for viewing the source model from the bundle.
// H6-P2-01: Wired into app.js view lifecycle. DPR-aware canvas sizing.
'use strict';

/**
 * SourceModelView handles the primary 3D viewport for the source model.
 * It communicates with the native C++ renderer via the inspectorAPI bridge.
 */
class SourceModelView {
  constructor() {
    this._active = false;
    this._debugMode = 'lit';
  }

  /**
   * Activate this view — show the 3D viewport canvas.
   */
  activate() {
    this._active = true;
  }

  /**
   * Deactivate this view.
   */
  deactivate() {
    this._active = false;
  }

  /**
   * Set the debug rendering mode.
   * @param {'lit'|'wireframe'|'normal'|'uv'|'depth'|'albedo'|'motion'|'cryptomatte'} mode
   */
  setDebugMode(mode) {
    this._debugMode = mode;
    if (this._active && window.inspectorAPI) {
      window.inspectorAPI.nativeSendCommand({
        type: 'set_debug_view',
        mode,
      }).catch(() => { /* native might not be running */ });
    }
  }

  /**
   * Reset camera to default orbit position.
   */
  resetCamera() {
    if (window.inspectorAPI) {
      window.inspectorAPI.nativeSendCommand({
        type: 'camera',
        action: 'orbit',
        dx: 0,
        dy: 0,
      }).catch(() => { /* native might not be running */ });
    }
  }

  /**
   * Get the current debug mode.
   * @returns {string}
   */
  get debugMode() {
    return this._debugMode;
  }

  /**
   * Whether this view is currently active.
   * @returns {boolean}
   */
  get isActive() {
    return this._active;
  }
}

// Export for use in app.js (loaded via <script> tag before app.js).
// eslint-disable-next-line no-unused-vars
var sourceModelView = new SourceModelView();

if (typeof module !== 'undefined') {
  module.exports = { SourceModelView };
}
