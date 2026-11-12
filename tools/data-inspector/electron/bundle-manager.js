// electron/bundle-manager.js — Loads and validates InspectionBundle directories.
'use strict';

const fs = require('fs');
const path = require('path');

const SCHEMA_VERSION = 'inspection.v1.0';
const MANIFEST_FILENAME = 'bundle.json';

class BundleManager {
  constructor() {
    this._bundlePath = null;
    this._manifest = null;
    this._inventory = null;
  }

  /**
   * Load and validate an InspectionBundle directory.
   * @param {string} bundlePath - Absolute path to the bundle directory.
   * @returns {{ ok: boolean, error?: string, manifest?: object, inventory?: object }}
   */
  loadBundle(bundlePath) {
    this._bundlePath = null;
    this._manifest = null;
    this._inventory = null;

    if (!bundlePath || !fs.existsSync(bundlePath)) {
      return { ok: false, error: `Bundle directory not found: ${bundlePath}` };
    }

    const stat = fs.statSync(bundlePath);
    if (!stat.isDirectory()) {
      return { ok: false, error: `Not a directory: ${bundlePath}` };
    }

    // Read and parse bundle.json.
    const manifestPath = path.join(bundlePath, MANIFEST_FILENAME);
    if (!fs.existsSync(manifestPath)) {
      return { ok: false, error: `Missing ${MANIFEST_FILENAME} in bundle` };
    }

    let manifest;
    try {
      const raw = fs.readFileSync(manifestPath, 'utf8');
      manifest = JSON.parse(raw);
    } catch (err) {
      return { ok: false, error: `Invalid ${MANIFEST_FILENAME}: ${err.message}` };
    }

    // Validate schema version.
    if (manifest.schema_version !== SCHEMA_VERSION) {
      return {
        ok: false,
        error: `Schema version mismatch: expected '${SCHEMA_VERSION}', ` +
               `got '${manifest.schema_version || 'undefined'}'`,
      };
    }

    // Validate required source_model path.
    const sources = manifest.sources || {};
    const sourceModel = sources.source_model;
    if (!sourceModel) {
      return { ok: false, error: 'Missing required sources.source_model in manifest' };
    }

    // Security: reject path traversal.
    for (const [key, val] of Object.entries(sources)) {
      if (typeof val === 'string') {
        if (val.includes('..') || path.isAbsolute(val)) {
          return { ok: false, error: `Unsafe path in sources.${key}: '${val}'` };
        }
      }
      if (Array.isArray(val)) {
        for (const entry of val) {
          if (typeof entry === 'string' && (entry.includes('..') || path.isAbsolute(entry))) {
            return { ok: false, error: `Unsafe path in sources.${key}: '${entry}'` };
          }
        }
      }
    }

    // Check source model file exists.
    const modelPath = path.join(bundlePath, sourceModel);
    if (!fs.existsSync(modelPath)) {
      return { ok: false, error: `Source model not found: ${sourceModel}` };
    }

    // Build inventory.
    const inventory = this._buildInventory(bundlePath);

    this._bundlePath = bundlePath;
    this._manifest = manifest;
    this._inventory = inventory;

    return { ok: true, manifest, inventory };
  }

  /**
   * Get current bundle info.
   * @returns {{ loaded: boolean, path?: string, manifest?: object, inventory?: object }}
   */
  getBundleInfo() {
    if (!this._bundlePath) {
      return { loaded: false };
    }
    return {
      loaded: true,
      path: this._bundlePath,
      manifest: this._manifest,
      inventory: this._inventory,
    };
  }

  /**
   * Build a file inventory of the bundle.
   * @param {string} bundlePath
   * @returns {{ files: Array<{path: string, size: number}>, totalSize: number }}
   * @private
   */
  _buildInventory(bundlePath) {
    const files = [];
    let totalSize = 0;

    const walk = (dir, prefix) => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        const relPath = prefix ? `${prefix}/${entry.name}` : entry.name;
        if (entry.isDirectory()) {
          walk(fullPath, relPath);
        } else if (entry.isFile()) {
          const size = fs.statSync(fullPath).size;
          files.push({ path: relPath, size });
          totalSize += size;
        }
      }
    };

    walk(bundlePath, '');
    return { files, totalSize, fileCount: files.length };
  }
}

module.exports = { BundleManager, SCHEMA_VERSION, MANIFEST_FILENAME };
