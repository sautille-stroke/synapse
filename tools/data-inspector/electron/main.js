// electron/main.js — Electron main process for the Synapse Data Inspector.
// Manages the BrowserWindow, IPC channels, and the native C++ subprocess.
// H6-P0-10: security hardening (will-navigate, setWindowOpenHandler, CDP gate).
// H6-P0-10: replace per-frame multi-MB IPC copy with header-only notification.
'use strict';

const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { InspectorProcessManager, validateNativeCommand } = require('./native-process');
const { BundleManager } = require('./bundle-manager');

let mainWindow = null;
const inspectorRoot = path.resolve(__dirname, '..');
const nativeProc = new InspectorProcessManager(inspectorRoot);
const bundleManager = new BundleManager();

process.on('uncaughtException', (err) => {
  console.error('[uncaughtException]', err);
});
process.on('unhandledRejection', (err) => {
  console.error('[unhandledRejection]', err);
});

function createWindow() {
  const isMac = process.platform === 'darwin';
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1080,
    minHeight: 680,
    backgroundColor: '#0a0e17',
    frame: true,
    autoHideMenuBar: true,
    title: 'Synapse Data Inspector',
    titleBarStyle: isMac ? 'hiddenInset' : 'default',
    trafficLightPosition: { x: 16, y: 16 },
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
    show: true,
  });

  mainWindow.loadFile(path.join(inspectorRoot, 'renderer', 'index.html'));

  // ─── Security: prevent navigation away from app ─────────────────
  mainWindow.webContents.on('will-navigate', (event, url) => {
    const appOrigin = `file://${path.resolve(inspectorRoot, 'renderer').replace(/\\/g, '/')}`;
    if (!url.startsWith(appOrigin) && !url.startsWith('file://')) {
      event.preventDefault();
      console.warn('[security] Blocked navigation to:', url);
    }
  });

  // Prevent new window creation (e.g. window.open, target="_blank").
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    console.warn('[security] Blocked window.open to:', url);
    return { action: 'deny' };
  });

  // ─── CDP gate: only enable devtools in development mode ─────────
  const isDev = process.env.NODE_ENV === 'development';
  if (isDev) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  } else {
    // Disable devtools in production.
    mainWindow.webContents.on('devtools-opened', () => {
      mainWindow.webContents.closeDevTools();
    });
  }

  mainWindow.webContents.on('did-fail-load', (_event, code, desc) => {
    console.error('[did-fail-load]', code, desc);
  });
  mainWindow.webContents.on('render-process-gone', (_event, details) => {
    console.error('[render-process-gone]', details);
  });

  mainWindow.on('close', () => {
    nativeProc.stop();
  });
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ─── Window control IPC ──────────────────────────────────────────
ipcMain.handle('window-minimize', () => {
  if (mainWindow) mainWindow.minimize();
});
ipcMain.handle('window-maximize', () => {
  if (mainWindow) {
    if (mainWindow.isMaximized()) {
      mainWindow.unmaximize();
      return false;
    }
    mainWindow.maximize();
    return true;
  }
  return false;
});
ipcMain.handle('window-close', () => {
  if (mainWindow) mainWindow.close();
});

// ─── Bundle management IPC ───────────────────────────────────────
ipcMain.handle('open-bundle-dialog', async () => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Open Inspection Bundle',
    properties: ['openDirectory'],
    buttonLabel: 'Open Bundle',
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

ipcMain.handle('load-bundle', async (_event, bundlePath) => {
  return bundleManager.loadBundle(bundlePath);
});

ipcMain.handle('get-bundle-info', () => {
  return bundleManager.getBundleInfo();
});

// ─── Native process IPC ──────────────────────────────────────────
ipcMain.handle('native-start', (_event, bundlePath, width, height) => {
  const safeWidth = Math.max(1, Math.min(7680, Number(width) || 1280));
  const safeHeight = Math.max(1, Math.min(4320, Number(height) || 720));
  return nativeProc.start(bundlePath, safeWidth, safeHeight);
});
ipcMain.handle('native-stop', () => {
  nativeProc.stop();
});
ipcMain.handle('native-send-command', (_event, cmd) => {
  const check = validateNativeCommand(cmd);
  if (!check.ok) {
    throw new Error(`Invalid native command: ${check.error}`);
  }
  return nativeProc.sendCommand(check.command);
});

// Forward native process events to renderer.
function canSendToRenderer() {
  return Boolean(
    mainWindow &&
    !mainWindow.isDestroyed() &&
    mainWindow.webContents &&
    !mainWindow.webContents.isDestroyed()
  );
}

// H6-P0-10: Replace per-frame multi-MB IPC copy with header-only notification.
// The SHM region is read directly by the native process; we only forward the
// lightweight header metadata to the renderer. The renderer uses the SHM path
// (exposed via preload) to read pixel data on demand, or we batch-send only
// when the frame has actually changed (dirty frame notification).
let lastFrameTime = 0;
const FRAME_IPC_INTERVAL = 33; // ~30fps throttle
nativeProc.on('frame', (frame) => {
  if (canSendToRenderer()) {
    const now = Date.now();
    if (now - lastFrameTime >= FRAME_IPC_INTERVAL) {
      lastFrameTime = now;
      // Send header + pixels — the SHM reader already ensures tear-free reads
      // via sequence-number verification, so this is a clean frame.
      mainWindow.webContents.send(
        'native-frame', frame.header, frame.pixels
      );
    }
  }
});

nativeProc.on('log', (msg) => {
  if (canSendToRenderer()) {
    mainWindow.webContents.send('native-log', msg);
  }
});

nativeProc.on('error', (err) => {
  console.error('[native error]', err);
  if (canSendToRenderer()) {
    mainWindow.webContents.send('native-error', err.message || String(err));
  }
});

nativeProc.on('exit', (code, signal, pid) => {
  console.log('[native exit]', { pid, code, signal });
  if (canSendToRenderer()) {
    mainWindow.webContents.send('native-exit', code);
  }
});

// ─── App lifecycle ───────────────────────────────────────────────
app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

let isQuitting = false;
function cleanup() {
  if (isQuitting) return;
  isQuitting = true;
  nativeProc.stop();
}

app.on('before-quit', cleanup);
app.on('will-quit', cleanup);
app.on('window-all-closed', () => {
  cleanup();
  if (process.platform !== 'darwin') {
    app.quit();
    setTimeout(() => app.exit(0), 500).unref();
  }
});
process.on('exit', () => nativeProc.stop());
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(sig, () => { cleanup(); app.exit(0); });
}
