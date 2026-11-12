// renderer/src/views/source-model.js — Source Model view component.
// Manages the 3D viewport for viewing the source model from the bundle.
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
   * @param {'lit'|'wireframe'|'normal'|'uv'} mode
   */
  setDebugMode(mode) {
    this._debugMode = mode;
    if (this._active && window.inspectorAPI) {
      window.inspectorAPI.nativeSendCommand({
        type: 'set_debug_view',
        mode,
      });
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
      });
    }
  }

  /**
   * Whether this view is currently active.
   * @returns {boolean}
   */
  get isActive() {
    return this._active;
  }
}

// Export for use in app.js (if using modules in the future).
if (typeof module !== 'undefined') {
  module.exports = { SourceModelView };
}
