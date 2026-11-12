// electron/native-process.js — Manages the native C++ inspector subprocess.
// Adapted from fulgur explorer's NativeProcessManager pattern.
'use strict';

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const EventEmitter = require('events');

const DEFAULT_SHM_NAME = '/inspector-frame';

class InspectorProcessManager extends EventEmitter {
  constructor(inspectorRoot) {
    super();
    this._inspectorRoot = inspectorRoot;
    this._nativeRoot = path.join(inspectorRoot, 'native');
    this._shmName = DEFAULT_SHM_NAME;
    this._proc = null;
  }

  /**
   * Resolve the path to the data-inspector-native binary.
   * Looks in bazel-bin relative to the native/ subdirectory.
   * @returns {string|null}
   */
  _resolveBinaryPath() {
    const ext = process.platform === 'win32' ? '.exe' : '';
    const candidates = [
      // Bazel output
      path.join(this._nativeRoot, 'bazel-bin', 'data-inspector-native' + ext),
      // Direct build (CMake or manual)
      path.join(this._nativeRoot, 'build', 'data-inspector-native' + ext),
      // Development fallback
      path.join(this._nativeRoot, 'data-inspector-native' + ext),
    ];
    for (const p of candidates) {
      try {
        if (fs.existsSync(p)) {
          fs.accessSync(p, fs.constants.X_OK);
          return p;
        }
      } catch (_) { /* skip */ }
    }
    return null;
  }

  /**
   * Start the native process for a given bundle.
   * @param {string} bundlePath - Path to the InspectionBundle directory.
   * @param {number} width - Frame width.
   * @param {number} height - Frame height.
   * @returns {boolean} true if the process was started.
   */
  start(bundlePath, width, height) {
    this.stop();

    const binPath = this._resolveBinaryPath();
    if (!binPath) {
      this.emit('error', new Error(
        'Native binary not found. Build it first:\n' +
        '  cd tools/data-inspector/native && bazel build //:data-inspector-native'
      ));
      return false;
    }

    const w = width || 1280;
    const h = height || 720;
    const args = [
      '--headless',
      `--width=${w}`,
      `--height=${h}`,
      `--shm-name=${this._shmName}`,
    ];
    if (bundlePath) {
      args.push(`--bundle=${bundlePath}`);
    }

    this.emit('log', `[inspector] Launching: ${binPath} ${args.join(' ')}`);

    let proc;
    try {
      proc = spawn(binPath, args, {
        cwd: path.dirname(binPath),
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true,
        env: { ...process.env },
      });
      this._proc = proc;
    } catch (err) {
      this.emit('error', err);
      return false;
    }

    proc.stdout.on('data', (chunk) => {
      if (this._proc !== proc) return;
      const lines = chunk.toString().split('\n');
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        // Try to parse JSON events from native process.
        if (trimmed.startsWith('{')) {
          this.emit('native-event', trimmed);
        }
        this.emit('log', trimmed);
      }
    });

    proc.stderr.on('data', (chunk) => {
      if (this._proc !== proc) return;
      const lines = chunk.toString().split('\n');
      for (const line of lines) {
        if (line.trim()) this.emit('log', `[stderr] ${line}`);
      }
    });

    proc.on('error', (err) => {
      if (this._proc === proc) this.emit('error', err);
    });

    proc.on('close', (code, signal) => {
      if (this._proc === proc) this._proc = null;
      if (!proc._stoppedByManager) {
        this.emit('exit', code, signal, proc.pid);
      }
    });

    return true;
  }

  /**
   * Send a JSON command to the native process via stdin.
   * @param {object} cmd - Command object.
   */
  sendCommand(cmd) {
    if (!this._proc || !this._proc.stdin || this._proc.stdin.destroyed) {
      return;
    }
    try {
      this._proc.stdin.write(JSON.stringify(cmd) + '\n');
    } catch (err) {
      console.error('[native-process] sendCommand error:', err.message);
    }
  }

  /**
   * Stop the native process.
   */
  stop() {
    const proc = this._proc;
    this._proc = null;
    if (!proc) return;

    proc._stoppedByManager = true;
    const pid = proc.pid;

    // Send quit command.
    try {
      if (proc.stdin && !proc.stdin.destroyed) {
        proc.stdin.write('{"type":"quit"}\n');
      }
    } catch (_) { /* ignore */ }

    // Destroy stdio streams.
    for (const stream of [proc.stdout, proc.stderr]) {
      try {
        if (stream) { stream.removeAllListeners(); stream.destroy(); }
      } catch (_) { /* ignore */ }
    }
    try {
      if (proc.stdin) { proc.stdin.removeAllListeners(); proc.stdin.end(); proc.stdin.destroy(); }
    } catch (_) { /* ignore */ }

    try { proc.unref(); } catch (_) { /* ignore */ }

    // Kill if still running.
    try {
      if (proc.exitCode === null && !proc.killed) {
        proc.kill('SIGTERM');
      }
    } catch (_) { /* ignore */ }

    // Windows: force kill process tree.
    if (process.platform === 'win32' && pid) {
      try {
        const killer = spawn('taskkill', ['/F', '/T', '/PID', String(pid)], {
          stdio: 'ignore', windowsHide: true, detached: true,
        });
        killer.unref();
      } catch (_) { /* ignore */ }
    }
  }
}

module.exports = { InspectorProcessManager, DEFAULT_SHM_NAME };
