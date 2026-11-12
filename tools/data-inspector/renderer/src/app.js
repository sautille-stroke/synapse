// renderer/src/app.js — Main application logic for the Data Inspector renderer.
// Handles bundle loading, viewport canvas, camera interaction, and view switching.
// Uses safe DOM construction (textContent) for all manifest/file strings.
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
  showTab('source', true);
  showTab('animation', !!(sources.animations && sources.animations.length));
  showTab('video', !!(sources.generated_video || sources.fgpp_sequence));
  showTab('comparison', !!sources.reconstructed_model);

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

  // Start native renderer for the source model view.
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
    labelSpan.textContent = String(label);
    node.appendChild(iconSpan);
    node.appendChild(labelSpan);
    tree.appendChild(node);
  };

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
}

// ─── Native renderer ─────────────────────────────────────────────

function startNativeRenderer(bundlePath) {
  const viewport = document.getElementById('viewport');
  const w = viewport.clientWidth || 1280;
  const h = viewport.clientHeight || 720;

  window.inspectorAPI.nativeStart(bundlePath, w, h);
  nativeRunning = true;

  // Show canvas, hide empty state.
  document.getElementById('viewport-empty').style.display = 'none';
  canvas.style.display = 'block';
  canvas.width = w;
  canvas.height = h;
  document.getElementById('prop-resolution').textContent = `${w}x${h}`;

  const overlay = document.getElementById('viewport-overlay');
  if (overlay) {
    overlay.textContent = 'SHM Diagnostic Rasterizer (--rhi=null)';
  }

  imageData = ctx.createImageData(w, h);
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
    updateStatus(`Native process exited (code ${code})`);
  });
}

// ─── View switching ──────────────────────────────────────────────

document.querySelectorAll('.view-tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.view-tab').forEach((t) =>
      t.classList.remove('active'));
    tab.classList.add('active');
    currentView = tab.dataset.view;
  });
});

// ─── Debug view mode ─────────────────────────────────────────────

document.querySelectorAll('.debug-modes button').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.debug-modes button').forEach((b) =>
      b.classList.remove('active'));
    btn.classList.add('active');
    currentDebugMode = btn.dataset.mode;
    document.getElementById('prop-debug-view').textContent =
      btn.dataset.mode.charAt(0).toUpperCase() + btn.dataset.mode.slice(1);

    if (nativeRunning) {
      window.inspectorAPI.nativeSendCommand({
        type: 'set_debug_view',
        mode: btn.dataset.mode,
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
    });
  });

  canvas.addEventListener('wheel', (e) => {
    if (!nativeRunning) return;
    e.preventDefault();
    window.inspectorAPI.nativeSendCommand({
      type: 'camera',
      action: 'zoom',
      dx: 0,
      dy: e.deltaY,
    });
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
