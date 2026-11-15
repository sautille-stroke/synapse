// electron/shm-reader.js — Shared-memory frame reader for the Data Inspector.
// Reads triple-buffered frames written by InspectorShmServer (@aether//pal:shm).
// Uses sequence-number spin-lock for tear-free reads and dirty-frame detection.
'use strict';

const EventEmitter = require('events');
const fs = require('fs');
const os = require('os');
const path = require('path');

const DEFAULT_SHM_NAME = '/inspector-frame';
const HEADER_SIZE = 64;
const SHM_BUFFER_COUNT = 3;

// macOS POSIX SHM name limit (31 usable bytes after leading '/').
const MACOS_PSHMNAMLEN = 31;

// Header field offsets (must match InspectorShmHeader in inspector_shm_server.h).
const OFF_MAGIC       = 0;   // u32
const OFF_VERSION     = 4;   // u32
const OFF_WIDTH       = 8;   // u32
const OFF_HEIGHT      = 12;  // u32
const OFF_WRITE_IDX   = 16;  // u32 (atomic latest completed buffer index)
const OFF_BUF_COUNT   = 20;  // u32 (3 for triple-buffering)
const OFF_FRAME_RDY   = 24;  // u64 (atomic monotonic frame counter)
const OFF_FPS         = 32;  // f32
const OFF_DELTA       = 36;  // f32
const OFF_DEBUG_VIEW  = 40;  // u8

// Inspector SHM magic: "INSP" = 0x494E5350
const INSPECTOR_SHM_MAGIC = 0x494E5350;
const INSPECTOR_SHM_VERSION = 1;

// Maximum spin iterations for tear-free read (sequence lock).
const MAX_SPIN_RETRIES = 4;

/**
 * Validate a POSIX SHM name for cross-platform correctness.
 * On macOS, shm_open names must be <= PSHMNAMLEN (31) bytes.
 * @param {string} shmName
 * @returns {{ valid: boolean, error?: string, normalized: string }}
 */
function validateShmName(shmName) {
  if (!shmName || typeof shmName !== 'string') {
    return { valid: false, error: 'SHM name must be a non-empty string', normalized: '' };
  }
  // Ensure leading slash for POSIX compliance.
  const normalized = shmName.startsWith('/') ? shmName : `/${shmName}`;
  if (os.platform() === 'darwin' && normalized.length > MACOS_PSHMNAMLEN) {
    return {
      valid: false,
      error: `SHM name '${normalized}' exceeds macOS PSHMNAMLEN limit of ${MACOS_PSHMNAMLEN} bytes`,
      normalized,
    };
  }
  // Reject embedded slashes (only leading '/' allowed in POSIX shm_open).
  if (normalized.indexOf('/', 1) !== -1) {
    return { valid: false, error: 'SHM name must not contain slashes after the leading "/"', normalized };
  }
  return { valid: true, normalized };
}

/**
 * Resolve a POSIX SHM name (e.g. "/inspector-frame-123") to its backing path.
 * On Linux, POSIX shm_open() creates files under /dev/shm/<name>.
 * On macOS / other platforms, falls back to os.tmpdir().
 * @param {string} shmName
 * @returns {string}
 */
function resolveShmFilePath(shmName) {
  if (!shmName || typeof shmName !== 'string') {
    throw new Error('Invalid SHM name');
  }
  // If the name is already a valid filesystem path, use it directly.
  if (fs.existsSync(shmName)) {
    return shmName;
  }
  const cleanName = shmName.replace(/^\/+/, '');
  // Linux: /dev/shm is the standard POSIX SHM backing store.
  const devShmPath = path.join('/dev/shm', cleanName);
  if (fs.existsSync('/dev/shm')) {
    return devShmPath;
  }
  // macOS / fallback: use tmpdir.
  return path.join(os.tmpdir(), cleanName);
}

/**
 * Parse a 64-byte InspectorShmHeader from a Buffer.
 * @param {Buffer} buf
 * @returns {object|null}
 */
function parseShmHeader(buf) {
  if (!Buffer.isBuffer(buf) || buf.length < HEADER_SIZE) {
    return null;
  }
  const magic = buf.readUInt32LE(OFF_MAGIC);
  const version = buf.readUInt32LE(OFF_VERSION);
  if (magic !== INSPECTOR_SHM_MAGIC || version !== INSPECTOR_SHM_VERSION) {
    return null;
  }
  const width = buf.readUInt32LE(OFF_WIDTH);
  const height = buf.readUInt32LE(OFF_HEIGHT);
  const writeIdx = buf.readUInt32LE(OFF_WRITE_IDX);
  const bufferCount = buf.readUInt32LE(OFF_BUF_COUNT) || SHM_BUFFER_COUNT;
  const frameReady = Number(buf.readBigUInt64LE(OFF_FRAME_RDY));
  const fps = buf.readFloatLE(OFF_FPS);
  const deltaTime = buf.readFloatLE(OFF_DELTA);
  const debugView = buf.readUInt8(OFF_DEBUG_VIEW);

  if (width === 0 || height === 0 || width > 7680 || height > 4320) {
    return null;
  }

  return {
    magic,
    version,
    width,
    height,
    writeIdx,
    bufferCount,
    frameReady,
    fps,
    deltaTime,
    debugView,
  };
}

class ShmFrameReader extends EventEmitter {
  constructor(shmName) {
    super();
    this._shmName = shmName || DEFAULT_SHM_NAME;
    this._fd = null;
    this._filePath = null;
    this._running = false;
    this._timer = null;
    this._lastFrameReady = 0;
    this._headerBuf = Buffer.alloc(HEADER_SIZE);
    this._headerBuf2 = Buffer.alloc(HEADER_SIZE);  // Second header read for spin-lock verification
  }

  /**
   * Open the shared memory backing file for reading.
   * @param {number} [_width]
   * @param {number} [_height]
   * @returns {boolean}
   */
  open(_width, _height) {
    this.close();
    try {
      const targetPath = resolveShmFilePath(this._shmName);
      if (!fs.existsSync(targetPath)) {
        return false;
      }
      this._fd = fs.openSync(targetPath, 'r');
      this._filePath = targetPath;
      this._lastFrameReady = 0;
      return true;
    } catch (_) {
      this._fd = null;
      this._filePath = null;
      return false;
    }
  }

  /**
   * Poll the SHM region once and emit 'frame' if a new completed frame is ready.
   * Uses a sequence-number spin-lock: reads header, then pixel data, then re-reads
   * header to verify frameReady has not changed (no torn frame).
   * @returns {{ header: object, pixels: Buffer }|null}
   */
  pollOnce() {
    if (this._fd === null) {
      if (!this.open()) return null;
    }

    try {
      const bytesRead = fs.readSync(this._fd, this._headerBuf, 0, HEADER_SIZE, 0);
      if (bytesRead < HEADER_SIZE) return null;

      const header = parseShmHeader(this._headerBuf);
      if (!header || header.frameReady === 0 || header.frameReady <= this._lastFrameReady) {
        return null;
      }

      const frameBytes = header.width * header.height * 4;
      const slot = header.writeIdx % header.bufferCount;
      const pixelOffset = HEADER_SIZE + slot * frameBytes;

      // Spin-lock: retry if writer changes frame during our read (torn frame detection).
      for (let attempt = 0; attempt < MAX_SPIN_RETRIES; attempt++) {
        const pixels = Buffer.allocUnsafe(frameBytes);
        const pixRead = fs.readSync(this._fd, pixels, 0, frameBytes, pixelOffset);
        if (pixRead < frameBytes) {
          return null;
        }

        // Re-read header to verify frame counter hasn't changed.
        const verifyRead = fs.readSync(this._fd, this._headerBuf2, 0, HEADER_SIZE, 0);
        if (verifyRead < HEADER_SIZE) return null;

        const verifyHeader = parseShmHeader(this._headerBuf2);
        if (!verifyHeader) return null;

        // If frameReady is still the same, we have a consistent (tear-free) read.
        if (verifyHeader.frameReady === header.frameReady) {
          this._lastFrameReady = header.frameReady;
          const frame = { header, pixels };
          this.emit('frame', frame);
          return frame;
        }
        // Frame changed during read — retry with the new frame.
        // Update header for next attempt.
        this._headerBuf2.copy(this._headerBuf);
      }

      // All retries exhausted (writer is faster than reader); skip this cycle.
      return null;
    } catch (_) {
      return null;
    }
  }

  /**
   * Start periodic polling for new frames.
   * @param {number} [intervalMs=16]
   */
  start(intervalMs = 16) {
    if (this._running) return;
    this._running = true;
    this._timer = setInterval(() => {
      if (this._running) {
        this.pollOnce();
      }
    }, intervalMs);
    if (this._timer.unref) {
      this._timer.unref();
    }
  }

  stop() {
    this._running = false;
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
  }

  close() {
    this.stop();
    if (this._fd !== null) {
      try {
        fs.closeSync(this._fd);
      } catch (_) { /* ignore */ }
      this._fd = null;
    }
    this._filePath = null;
  }
}

module.exports = {
  ShmFrameReader,
  parseShmHeader,
  resolveShmFilePath,
  validateShmName,
  DEFAULT_SHM_NAME,
  INSPECTOR_SHM_MAGIC,
  INSPECTOR_SHM_VERSION,
  HEADER_SIZE,
  SHM_BUFFER_COUNT,
  MACOS_PSHMNAMLEN,
};
