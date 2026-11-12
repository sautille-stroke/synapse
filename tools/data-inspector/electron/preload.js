// electron/preload.js — Secure contextBridge API for the Data Inspector.
// Exposes a controlled set of IPC methods to the renderer process.
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('inspectorAPI', {
  platform: process.platform,

  // Window controls.
  minimize: () => ipcRenderer.invoke('window-minimize'),
  maximize: () => ipcRenderer.invoke('window-maximize'),
  close: () => ipcRenderer.invoke('window-close'),

  // Bundle management.
  openBundleDialog: () => ipcRenderer.invoke('open-bundle-dialog'),
  loadBundle: (path) => ipcRenderer.invoke('load-bundle', path),
  getBundleInfo: () => ipcRenderer.invoke('get-bundle-info'),

  // Native renderer.
  nativeStart: (bundlePath, w, h) =>
    ipcRenderer.invoke('native-start', bundlePath, w, h),
  nativeStop: () => ipcRenderer.invoke('native-stop'),
  nativeSendCommand: (cmd) =>
    ipcRenderer.invoke('native-send-command', cmd),

  // Event listeners (main -> renderer push).
  onNativeFrame: (cb) =>
    ipcRenderer.on('native-frame', (_e, header, pixels) =>
      cb(header, pixels)),
  onNativeLog: (cb) =>
    ipcRenderer.on('native-log', (_e, msg) => cb(msg)),
  onNativeError: (cb) =>
    ipcRenderer.on('native-error', (_e, msg) => cb(msg)),
  onNativeExit: (cb) =>
    ipcRenderer.on('native-exit', (_e, code) => cb(code)),
});
