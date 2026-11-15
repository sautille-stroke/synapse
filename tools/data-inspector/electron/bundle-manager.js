// electron/bundle-manager.js — Loads and validates InspectionBundle directories.
// H6-P1-07: Support animation_*/generated_*/fgpp_* multi-instance source keys.
'use strict';

const fs = require('fs');
const path = require('path');

const SCHEMA_VERSION = 'inspection.v1.0';
const ACCEPTED_SCHEMA_VERSIONS = new Set([
  'inspection.v1.0',
  'synapse.bundle.v1.0',
]);
const MANIFEST_FILENAME = 'bundle.json';

// Prefixes for multi-instance source keys.
const MULTI_INSTANCE_PREFIXES = [
  'animation_',
  'generated_',
  'fgpp_',
];

/**
 * Extract relative path from a source entry, which can be either a string
 * or a SourceEntry object ({ path: '...', media_type: '...' }).
 * @param {unknown} entry
 * @returns {string|null}
 */
function extractEntryPath(entry) {
  if (!entry) return null;
  if (typeof entry === 'string') return entry;
  if (typeof entry === 'object') {
    if (typeof entry.path === 'string') return entry.path;
    if (typeof entry.url === 'string') return entry.url;
  }
  return null;
}

/**
 * Extract a display label from a source entry.
 * Returns the path string for display (handles both string and object entries).
 * @param {unknown} entry
 * @returns {string}
 */
function extractEntryLabel(entry) {
  const p = extractEntryPath(entry);
  if (p) return p;
  if (entry && typeof entry === 'object') {
    if (typeof entry.label === 'string') return entry.label;
    if (typeof entry.name === 'string') return entry.name;
  }
  return String(entry || '');
}

/**
 * Check if a path string contains traversal segments or is absolute.
 * @param {unknown} p
 * @returns {boolean}
 */
function isUnsafePath(p) {
  if (typeof p !== 'string') return false;
  return p.includes('..') || path.isAbsolute(p) || /^[a-zA-Z]:[/\\]/.test(p);
}

/**
 * Check if a key matches any of the multi-instance source prefixes.
 * @param {string} key
 * @returns {boolean}
 */
function isMultiInstanceKey(key) {
  return MULTI_INSTANCE_PREFIXES.some(prefix => key.startsWith(prefix));
}

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
    if (!ACCEPTED_SCHEMA_VERSIONS.has(manifest.schema_version)) {
      return {
        ok: false,
        error: `Schema version mismatch: expected one of [${Array.from(ACCEPTED_SCHEMA_VERSIONS).join(', ')}], ` +
               `got '${manifest.schema_version || 'undefined'}'`,
      };
    }

    // Validate required source_model path.
    const sources = manifest.sources || {};
    let sourceModelEntry = null;
    if (Array.isArray(sources)) {
      sourceModelEntry = sources.find(s =>
        s && (s.label === 'source_model' || s.name === 'source_model' || s.modality === 'source_model' ||
              (typeof s.path === 'string' && s.path.includes('model.')))
      ) || sources[0];
    } else if (typeof sources === 'object') {
      sourceModelEntry = sources.source_model || sources.model || sources['source/model'];
    }

    const sourceModelPath = extractEntryPath(sourceModelEntry);
    if (!sourceModelPath) {
      return { ok: false, error: 'Missing required sources.source_model in manifest' };
    }

    // Security: reject path traversal.
    const traversalError = this._checkPathTraversal(sources);
    if (traversalError) {
      return { ok: false, error: traversalError };
    }

    // Check source model file exists.
    const modelPath = path.join(bundlePath, sourceModelPath);
    if (!fs.existsSync(modelPath)) {
      return { ok: false, error: `Source model not found: ${sourceModelPath}` };
    }

    // Build inventory.
    const inventory = this._buildInventory(bundlePath);

    this._bundlePath = bundlePath;
    this._manifest = manifest;
    this._inventory = inventory;

    return { ok: true, manifest, inventory };
  }

  /**
   * Check all source entries for path traversal.
   * Supports multi-instance keys (animation_*, generated_*, fgpp_*).
   * @param {object|Array} sources
   * @returns {string|null} Error message or null if clean.
   * @private
   */
  _checkPathTraversal(sources) {
    if (Array.isArray(sources)) {
      for (let i = 0; i < sources.length; i++) {
        const p = extractEntryPath(sources[i]);
        if (p && isUnsafePath(p)) {
          return `Unsafe path in sources[${i}]: '${p}'`;
        }
      }
    } else if (typeof sources === 'object') {
      for (const [key, val] of Object.entries(sources)) {
        if (typeof val === 'string') {
          if (isUnsafePath(val)) {
            return `Unsafe path in sources.${key}: '${val}'`;
          }
        } else if (val && typeof val === 'object' && !Array.isArray(val)) {
          const p = extractEntryPath(val);
          if (p && isUnsafePath(p)) {
            return `Unsafe path in sources.${key}: '${p}'`;
          }
        } else if (Array.isArray(val)) {
          for (const entry of val) {
            const p = extractEntryPath(entry);
            if (p && isUnsafePath(p)) {
              return `Unsafe path in sources.${key}: '${p}'`;
            }
          }
        }
      }
    }
    return null;
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

module.exports = {
  BundleManager,
  SCHEMA_VERSION,
  ACCEPTED_SCHEMA_VERSIONS,
  MANIFEST_FILENAME,
  MULTI_INSTANCE_PREFIXES,
  extractEntryPath,
  extractEntryLabel,
  isUnsafePath,
  isMultiInstanceKey,
};
