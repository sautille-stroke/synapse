// electron/preload.js — Secure contextBridge API for the Data Inspector.
// Exposes a controlled set of IPC methods to the renderer process, returning
// unsubscribe disposers for all event listeners.
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

function subscribeChannel(channel, wrapArgs) {
  const listener = (_event, ...args) => {
    wrapArgs(...args);
  };
  ipcRenderer.on(channel, listener);
  return () => {
    ipcRenderer.removeListener(channel, listener);
  };
}

contextBridge.exposeInMainWorld('inspectorAPI', {
  platform: process.platform,

  // Window controls.
  minimize: () => ipcRenderer.invoke('window-minimize'),
  maximize: () => ipcRenderer.invoke('window-maximize'),
  close: () => ipcRenderer.invoke('window-close'),

  // Bundle management.
  openBundleDialog: () => ipcRenderer.invoke('open-bundle-dialog'),
  loadBundle: (bundlePath) => ipcRenderer.invoke('load-bundle', bundlePath),
  getBundleInfo: () => ipcRenderer.invoke('get-bundle-info'),

  // Native renderer.
  nativeStart: (bundlePath, w, h) =>
    ipcRenderer.invoke('native-start', bundlePath, w, h),
  nativeStop: () => ipcRenderer.invoke('native-stop'),
  nativeSendCommand: (cmd) =>
    ipcRenderer.invoke('native-send-command', cmd),

  // Event listeners (main -> renderer push). Returns an unsubscribe function.
  onNativeFrame: (cb) =>
    subscribeChannel('native-frame', (header, pixels) => cb(header, pixels)),
  onNativeLog: (cb) =>
    subscribeChannel('native-log', (msg) => cb(msg)),
  onNativeError: (cb) =>
    subscribeChannel('native-error', (msg) => cb(msg)),
  onNativeExit: (cb) =>
    subscribeChannel('native-exit', (code) => cb(code)),
});
