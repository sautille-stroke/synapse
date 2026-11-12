// electron/native-process.js — Manages the native C++ inspector subprocess.
// Includes line-buffered stdout/stderr parsing, strict command schema validation,
// and SHM frame reader integration.
'use strict';

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const readline = require('readline');
const EventEmitter = require('events');
const { ShmFrameReader, DEFAULT_SHM_NAME } = require('./shm-reader');

const ALLOWED_COMMAND_TYPES = new Set([
  'camera',
  'set_debug_view',
  'set_entity_visible',
  'export_screenshot',
  'load_bundle',
  'quit',
]);

const ALLOWED_CAMERA_ACTIONS = new Set(['orbit', 'pan', 'zoom']);
const ALLOWED_DEBUG_MODES = new Set(['lit', 'wireframe', 'normal', 'uv']);

/**
 * Validate a command object before serialising it to the native process stdin.
 * @param {unknown} cmd
 * @returns {{ ok: boolean, error?: string, command?: object }}
 */
function validateNativeCommand(cmd) {
  if (!cmd || typeof cmd !== 'object' || Array.isArray(cmd)) {
    return { ok: false, error: 'Command must be a non-null object' };
  }
  const type = cmd.type;
  if (typeof type !== 'string' || !ALLOWED_COMMAND_TYPES.has(type)) {
    return { ok: false, error: `Disallowed or unknown command type: '${String(type)}'` };
  }

  switch (type) {
    case 'camera': {
      if (!ALLOWED_CAMERA_ACTIONS.has(cmd.action)) {
        return { ok: false, error: `Invalid camera action: '${String(cmd.action)}'` };
      }
      if (typeof cmd.dx !== 'number' || !Number.isFinite(cmd.dx) ||
          typeof cmd.dy !== 'number' || !Number.isFinite(cmd.dy)) {
        return { ok: false, error: 'Camera dx and dy must be finite numbers' };
      }
      return {
        ok: true,
        command: { type: 'camera', action: cmd.action, dx: cmd.dx, dy: cmd.dy },
      };
    }
    case 'set_debug_view': {
      if (!ALLOWED_DEBUG_MODES.has(cmd.mode)) {
        return { ok: false, error: `Invalid debug view mode: '${String(cmd.mode)}'` };
      }
      return { ok: true, command: { type: 'set_debug_view', mode: cmd.mode } };
    }
    case 'set_entity_visible': {
      if (typeof cmd.name !== 'string' || cmd.name.length === 0 || cmd.name.length > 256 ||
          cmd.name.includes('\0') || cmd.name.includes('"')) {
        return { ok: false, error: 'Invalid entity name' };
      }
      if (typeof cmd.visible !== 'boolean') {
        return { ok: false, error: 'Entity visible flag must be a boolean' };
      }
      return {
        ok: true,
        command: { type: 'set_entity_visible', name: cmd.name, visible: cmd.visible },
      };
    }
    case 'export_screenshot':
    case 'load_bundle': {
      if (typeof cmd.path !== 'string' || cmd.path.length === 0 || cmd.path.length > 4096 ||
          cmd.path.includes('\0') || cmd.path.includes('"')) {
        return { ok: false, error: `Invalid path for ${type}` };
      }
      return { ok: true, command: { type, path: cmd.path } };
    }
    case 'quit':
      return { ok: true, command: { type: 'quit' } };
    default:
      return { ok: false, error: `Unsupported command type: '${type}'` };
  }
}

class InspectorProcessManager extends EventEmitter {
  constructor(inspectorRoot, options = {}) {
    super();
    this._inspectorRoot = inspectorRoot;
    this._nativeRoot = path.join(inspectorRoot, 'native');
    this._shmName = options.shmName || `${DEFAULT_SHM_NAME}-${process.pid}`;
    this._proc = null;
    this._stdoutRl = null;
    this._stderrRl = null;
    this._shmReader = new ShmFrameReader(this._shmName);
    this._shmReader.on('frame', (frame) => {
      this.emit('frame', frame);
    });
  }

  /**
   * Resolve the path to the data-inspector-native binary.
   * @returns {string|null}
   */
  _resolveBinaryPath() {
    const ext = process.platform === 'win32' ? '.exe' : '';
    const candidates = [
      path.join(this._nativeRoot, 'bazel-bin', 'data-inspector-native' + ext),
      path.join(this._nativeRoot, 'build', 'data-inspector-native' + ext),
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
   * Process a single line from the native subprocess stdout.
   * Exposed for unit testing of line buffering and JSON event routing.
   * @param {string} rawLine
   */
  _handleStdoutLine(rawLine) {
    const trimmed = String(rawLine).trim();
    if (!trimmed) return;
    if (trimmed.startsWith('{')) {
      try {
        const parsed = JSON.parse(trimmed);
        this.emit('native-event', parsed);
        if (parsed && parsed.type === 'frame_ready') {
          this._shmReader.pollOnce();
        }
      } catch (_) {
        // Non-JSON line starting with '{'; fall through to log.
      }
    }
    this.emit('log', trimmed);
  }

  /**
   * Start the native process for a given bundle.
   * @param {string} bundlePath - Path to the InspectionBundle directory.
   * @param {number} width - Frame width.
   * @param {number} height - Frame height.
   * @param {object} [extraOptions] - Optional extra CLI options (e.g. maxFrames, keepShm).
   * @returns {boolean} true if the process was started.
   */
  start(bundlePath, width, height, extraOptions = {}) {
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
    if (extraOptions.maxFrames > 0) {
      args.push(`--frames=${extraOptions.maxFrames}`);
    }
    if (extraOptions.keepShm) {
      args.push('--keep-shm');
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

    this._shmReader.start(16);

    this._stdoutRl = readline.createInterface({
      input: proc.stdout,
      crlfDelay: Infinity,
    });
    this._stdoutRl.on('line', (line) => {
      if (this._proc !== proc) return;
      this._handleStdoutLine(line);
    });

    this._stderrRl = readline.createInterface({
      input: proc.stderr,
      crlfDelay: Infinity,
    });
    this._stderrRl.on('line', (line) => {
      if (this._proc !== proc) return;
      const trimmed = line.trim();
      if (trimmed) this.emit('log', `[stderr] ${trimmed}`);
    });

    proc.on('error', (err) => {
      if (this._proc === proc) this.emit('error', err);
    });

    proc.on('close', (code, signal) => {
      this._shmReader.pollOnce();
      this._shmReader.close();
      if (this._proc === proc) this._proc = null;
      if (!proc._stoppedByManager) {
        this.emit('exit', code, signal, proc.pid);
      }
    });

    return true;
  }

  /**
   * Validate and send a JSON command to the native process via stdin.
   * @param {object} cmd - Command object.
   * @returns {{ ok: boolean, error?: string }}
   */
  sendCommand(cmd) {
    const validation = validateNativeCommand(cmd);
    if (!validation.ok) {
      return { ok: false, error: validation.error };
    }
    if (!this._proc || !this._proc.stdin || this._proc.stdin.destroyed) {
      return { ok: false, error: 'Native process is not running' };
    }
    try {
      this._proc.stdin.write(JSON.stringify(validation.command) + '\n');
      return { ok: true };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  }

  /**
   * Stop the native process and close SHM reader.
   */
  stop() {
    this._shmReader.close();

    if (this._stdoutRl) {
      try { this._stdoutRl.close(); } catch (_) { /* ignore */ }
      this._stdoutRl = null;
    }
    if (this._stderrRl) {
      try { this._stderrRl.close(); } catch (_) { /* ignore */ }
      this._stderrRl = null;
    }

    const proc = this._proc;
    this._proc = null;
    if (!proc) return;

    proc._stoppedByManager = true;
    const pid = proc.pid;

    try {
      if (proc.stdin && !proc.stdin.destroyed) {
        proc.stdin.write('{"type":"quit"}\n');
      }
    } catch (_) { /* ignore */ }

    for (const stream of [proc.stdout, proc.stderr]) {
      try {
        if (stream) { stream.removeAllListeners(); stream.destroy(); }
      } catch (_) { /* ignore */ }
    }
    try {
      if (proc.stdin) {
        proc.stdin.removeAllListeners();
        proc.stdin.end();
        proc.stdin.destroy();
      }
    } catch (_) { /* ignore */ }

    try { proc.unref(); } catch (_) { /* ignore */ }

    try {
      if (proc.exitCode === null && !proc.killed) {
        proc.kill('SIGTERM');
      }
    } catch (_) { /* ignore */ }

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

module.exports = {
  InspectorProcessManager,
  validateNativeCommand,
  ALLOWED_COMMAND_TYPES,
  DEFAULT_SHM_NAME,
};
