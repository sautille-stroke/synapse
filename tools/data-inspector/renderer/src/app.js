// renderer/src/app.js — Main application logic for the Data Inspector renderer.
// Handles bundle loading, viewport canvas, camera interaction, and view switching.
// Uses safe DOM construction (textContent) for all manifest/file strings.
// H6-P0-10: Unpacks SourceEntry.path (eliminates [object Object] node text).
// H6-P0-10: Decouples top nav tabs from native AOV set_debug_view.
// H6-P0-10: Fixes uncaught Promise rejection on tab/debug-mode click.
// H6-P2-01: Canvas ResizeObserver for DPR adaptation. macOS traffic light padding.
'use strict';

// ─── State ───────────────────────────────────────────────────────
let currentView = 'source';
let currentDebugMode = 'lit';
let bundleLoaded = false;
let nativeRunning = false;

// Canvas and rendering.
const canvas = document.getElementById('viewport-canvas');
const ctx = canvas ? canvas.getContext('2d') : null;
let imageData = null;

// Camera interaction state.
let isDragging = false;
let lastMouse = { x: 0, y: 0 };

// ─── Platform setup ──────────────────────────────────────────────

// H6-P2-01: Apply macOS traffic light offset via CSS custom property.
if (window.inspectorAPI && window.inspectorAPI.platform === 'darwin') {
  document.documentElement.style.setProperty('--traffic-light-inset', '76px');
}

// ─── Helper: extract display label from source entry ─────────────

/**
 * Extract a display-safe string from a source entry.
 * Handles both plain strings and SourceEntry objects ({ path: '...' }).
 * Prevents [object Object] from appearing in the UI.
 * @param {unknown} entry
 * @returns {string}
 */
function extractDisplayLabel(entry) {
  if (!entry) return '';
  if (typeof entry === 'string') return entry;
  if (typeof entry === 'object') {
    if (typeof entry.path === 'string') return entry.path;
    if (typeof entry.url === 'string') return entry.url;
    if (typeof entry.label === 'string') return entry.label;
    if (typeof entry.name === 'string') return entry.name;
  }
  return '';
}

// ─── Bundle loading ──────────────────────────────────────────────

document.getElementById('btn-open-bundle').addEventListener('click', async () => {
  const bundlePath = await window.inspectorAPI.openBundleDialog();
  if (!bundlePath) return;
  await loadBundle(bundlePath);
});

function createPropRow(label, valueText) {
  const row = document.createElement('div');
  row.className = 'prop-row';
  const labelEl = document.createElement('span');
  labelEl.className = 'prop-label';
  labelEl.textContent = String(label);
  const valEl = document.createElement('span');
  valEl.className = 'prop-value';
  valEl.textContent = String(valueText);
  row.appendChild(labelEl);
  row.appendChild(valEl);
  return row;
}

async function loadBundle(bundlePath) {
  updateStatus('Loading...');
  const result = await window.inspectorAPI.loadBundle(bundlePath);

  if (!result.ok) {
    updateStatus(`Error: ${result.error}`);
    return;
  }

  bundleLoaded = true;
  const manifest = result.manifest;
  const inv = result.inventory;

  // Update sidebar using textContent (no innerHTML injection).
  const sidebarContent = document.getElementById('sidebar-content');
  sidebarContent.replaceChildren();
  const bundleNameEl = document.createElement('div');
  bundleNameEl.className = 'bundle-name';
  bundleNameEl.textContent = String(bundlePath).split(/[/\\]/).pop() || '';
  sidebarContent.appendChild(bundleNameEl);

  // Show available view tabs based on manifest sources.
  const sources = manifest.sources || {};

  // H6-P0-10: Support multi-instance keys (animation_*, generated_*, fgpp_*).
  const hasAnimations = !!(sources.animations && sources.animations.length) ||
    Object.keys(sources).some(k => k.startsWith('animation_'));
  const hasVideo = !!(sources.generated_video || sources.fgpp_sequence) ||
    Object.keys(sources).some(k => k.startsWith('generated_') || k.startsWith('fgpp_'));
  const hasComparison = !!sources.reconstructed_model;

  showTab('source', true);
  showTab('animation', hasAnimations);
  showTab('video', hasVideo);
  showTab('comparison', hasComparison);

  // Update bundle info safely via textContent.
  const bundleInfo = document.getElementById('bundle-info');
  bundleInfo.replaceChildren(
    createPropRow('Status', 'Loaded'),
    createPropRow('Schema', manifest.schema_version || '--'),
    createPropRow('Bundle ID', manifest.bundle_id || '--'),
    createPropRow('Files', String(inv.fileCount)),
    createPropRow('Total Size', formatBytes(inv.totalSize)),
  );

  // Build scene tree from manifest sources.
  buildSceneTree(manifest);

  updateStatus(`Bundle loaded: ${inv.fileCount} files`);

  // Activate source view and start native renderer.
  if (typeof sourceModelView !== 'undefined') {
    sourceModelView.activate();
  }
  startNativeRenderer(bundlePath);
}

function showTab(view, visible) {
  const tab = document.querySelector(`.view-tab[data-view="${view}"]`);
  if (tab) tab.style.display = visible ? '' : 'none';
}

function buildSceneTree(manifest) {
  const tree = document.getElementById('scene-tree');
  tree.replaceChildren();

  const sources = manifest.sources || {};
  const addNode = (icon, label) => {
    const node = document.createElement('div');
    node.className = 'tree-node';
    const iconSpan = document.createElement('span');
    iconSpan.className = 'tree-icon';
    iconSpan.textContent = String(icon);
    const labelSpan = document.createElement('span');
    // H6-P0-10: Use extractDisplayLabel to handle SourceEntry objects.
    labelSpan.textContent = extractDisplayLabel(label);
    node.appendChild(iconSpan);
    node.appendChild(labelSpan);
    tree.appendChild(node);
  };

  // Core sources.
  if (sources.source_model) addNode('\u25B6', sources.source_model);
  if (sources.animations) {
    for (const anim of sources.animations) {
      addNode('\u25B7', anim);
    }
  }
  if (sources.fgpp_sequence) addNode('\u25A0', sources.fgpp_sequence);
  if (sources.generated_video) addNode('\u25A0', sources.generated_video);
  if (sources.reconstructed_model) addNode('\u25B6', sources.reconstructed_model);
  if (sources.metrics) addNode('\u25CB', sources.metrics);

  // H6-P0-10: Multi-instance source keys (animation_*, generated_*, fgpp_*).
  for (const [key, val] of Object.entries(sources)) {
    if (key.startsWith('animation_')) {
      addNode('\u25B7', val);
    } else if (key.startsWith('generated_')) {
      addNode('\u25A0', val);
    } else if (key.startsWith('fgpp_')) {
      addNode('\u25A0', val);
    }
  }
}

// ─── Native renderer ─────────────────────────────────────────────

function startNativeRenderer(bundlePath) {
  const viewport = document.getElementById('viewport');
  const dpr = window.devicePixelRatio || 1;
  const cssW = viewport.clientWidth || 1280;
  const cssH = viewport.clientHeight || 720;
  // H6-P2-01: DPR-aware canvas sizing.
  const w = Math.round(cssW * dpr);
  const h = Math.round(cssH * dpr);

  window.inspectorAPI.nativeStart(bundlePath, w, h)
    .catch((err) => {
      console.error('[native start error]', err);
      updateStatus(`Error starting native renderer: ${err.message || err}`);
    });
  nativeRunning = true;

  // Show canvas, hide empty state.
  document.getElementById('viewport-empty').style.display = 'none';
  canvas.style.display = 'block';
  canvas.width = w;
  canvas.height = h;
  // CSS size matches the viewport container.
  canvas.style.width = cssW + 'px';
  canvas.style.height = cssH + 'px';
  document.getElementById('prop-resolution').textContent = `${w}x${h}`;

  const overlay = document.getElementById('viewport-overlay');
  if (overlay) {
    overlay.textContent = 'SHM Diagnostic Rasterizer (--rhi=null)';
  }

  imageData = ctx.createImageData(w, h);
}

// H6-P2-01: Canvas ResizeObserver for DPR adaptation on resize.
if (typeof ResizeObserver !== 'undefined') {
  const viewportEl = document.getElementById('viewport');
  if (viewportEl && canvas) {
    const resizeObserver = new ResizeObserver(() => {
      if (!nativeRunning) return;
      const dpr = window.devicePixelRatio || 1;
      const cssW = viewportEl.clientWidth;
      const cssH = viewportEl.clientHeight;
      const w = Math.round(cssW * dpr);
      const h = Math.round(cssH * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        canvas.style.width = cssW + 'px';
        canvas.style.height = cssH + 'px';
        imageData = ctx.createImageData(w, h);
        document.getElementById('prop-resolution').textContent = `${w}x${h}`;
      }
    });
    resizeObserver.observe(viewportEl);
  }
}

// Listen for frames from the native process.
if (window.inspectorAPI) {
  window.inspectorAPI.onNativeFrame((header, pixels) => {
    if (!canvas || !ctx || !imageData) return;

    // `pixels` is a Buffer/Uint8Array of RGBA data.
    if (pixels && pixels.length === imageData.data.length) {
      imageData.data.set(pixels);
      ctx.putImageData(imageData, 0, 0);
    }

    // Update FPS display.
    if (header && header.fps !== undefined) {
      document.getElementById('prop-fps').textContent =
        `${Math.round(header.fps * 10) / 10}`;
    }
  });

  window.inspectorAPI.onNativeLog((msg) => {
    console.log('[native]', msg);
  });

  window.inspectorAPI.onNativeError((msg) => {
    console.error('[native error]', msg);
    updateStatus(`Native error: ${msg}`);
  });

  window.inspectorAPI.onNativeExit((code) => {
    console.log('[native exit]', code);
    nativeRunning = false;
    if (typeof sourceModelView !== 'undefined') {
      sourceModelView.deactivate();
    }
    updateStatus(`Native process exited (code ${code})`);
  });
}

// ─── View switching (top panel navigation tabs) ──────────────────
// H6-P0-10: DECOUPLED from native AOV set_debug_view. Tabs switch the
// app's conceptual view (source/animation/video/comparison), NOT the
// native renderer's debug visualization mode.

document.querySelectorAll('.view-tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.view-tab').forEach((t) =>
      t.classList.remove('active'));
    tab.classList.add('active');
    currentView = tab.dataset.view;

    // Activate/deactivate SourceModelView based on tab.
    if (typeof sourceModelView !== 'undefined') {
      if (currentView === 'source') {
        sourceModelView.activate();
      } else {
        sourceModelView.deactivate();
      }
    }

    // View switching does NOT send set_debug_view to native.
    // Debug view mode is controlled independently by the .debug-modes buttons.
  });
});

// ─── Debug view mode (native AOV selector) ───────────────────────
// H6-P0-10: Independent from view tabs. Controls the native renderer's
// debug visualization (lit/wireframe/normals/depth/uv/albedo/motion/cryptomatte).

document.querySelectorAll('.debug-modes button').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.debug-modes button').forEach((b) =>
      b.classList.remove('active'));
    btn.classList.add('active');
    currentDebugMode = btn.dataset.mode;
    document.getElementById('prop-debug-view').textContent =
      btn.dataset.mode.charAt(0).toUpperCase() + btn.dataset.mode.slice(1);

    // Update SourceModelView state.
    if (typeof sourceModelView !== 'undefined') {
      sourceModelView.setDebugMode(btn.dataset.mode);
    }

    // H6-P0-10: Guard against uncaught Promise rejection when native is not running.
    if (nativeRunning) {
      window.inspectorAPI.nativeSendCommand({
        type: 'set_debug_view',
        mode: btn.dataset.mode,
      }).catch((err) => {
        console.warn('[debug-mode] Failed to send command:', err.message || err);
      });
    }
  });
});

// ─── Camera interaction (orbit/pan/zoom) ─────────────────────────

if (canvas) {
  canvas.addEventListener('mousedown', (e) => {
    isDragging = true;
    lastMouse = { x: e.clientX, y: e.clientY };
  });

  window.addEventListener('mouseup', () => {
    isDragging = false;
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging || !nativeRunning) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    lastMouse = { x: e.clientX, y: e.clientY };

    const action = e.buttons === 2 ? 'pan' :
                   e.buttons === 4 ? 'zoom' : 'orbit';

    window.inspectorAPI.nativeSendCommand({
      type: 'camera',
      action,
      dx,
      dy,
    }).catch(() => { /* ignore send failures during drag */ });
  });

  canvas.addEventListener('wheel', (e) => {
    if (!nativeRunning) return;
    e.preventDefault();
    window.inspectorAPI.nativeSendCommand({
      type: 'camera',
      action: 'zoom',
      dx: 0,
      dy: e.deltaY,
    }).catch(() => { /* ignore */ });
  });

  // Prevent context menu on right-click.
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
}

// ─── Helpers ─────────────────────────────────────────────────────

function updateStatus(text) {
  const el = document.getElementById('status-text');
  if (el) el.textContent = text;
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}
