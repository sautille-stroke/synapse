// electron/shm-reader.js — Shared-memory frame reader for the Data Inspector.
// Reads triple-buffered frames written by InspectorShmServer (@aether//pal:shm).
// Uses strict sequence-number lock (Seqlock) for tear-free reads and dirty-frame detection.
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
const OFF_FRAME_RDY   = 24;  // u64 (atomic monotonic frame counter / Seqlock)
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
 * On macOS / other platforms, falls back to os.tmpdir() or /tmp.
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
  // Linux: /dev/shm is standard POSIX SHM backing store.
  const devShmPath = path.join('/dev/shm', cleanName);
  if (fs.existsSync(devShmPath)) {
    return devShmPath;
  }
  if (fs.existsSync('/dev/shm')) {
    return devShmPath;
  }
  // macOS / fallback: use tmpdir or /tmp
  const slashTmp = path.join('/tmp', cleanName);
  if (fs.existsSync(slashTmp)) {
    return slashTmp;
  }
  const tmpPath = path.join(os.tmpdir(), cleanName);
  if (fs.existsSync(tmpPath)) {
    return tmpPath;
  }
  return tmpPath;
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

  if (width === 0 || height === 0 || width > 8192 || height > 8192) {
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
    this._headerBuf2 = Buffer.alloc(HEADER_SIZE);
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
   * Uses strict Seqlock protocol:
   * 1. Reads header inside the retry loop.
   * 2. Checks if frameReady is odd (write in progress) -> spins/retries.
   * 3. Reads pixel buffer.
   * 4. Re-reads header and verifies (seq2 & 1) === 0 and seq1 === seq2 (no torn frame).
   * @returns {{ header: object, pixels: Buffer }|null}
   */
  pollOnce() {
    if (this._fd === null) {
      if (!this.open()) return null;
    }

    try {
      for (let attempt = 0; attempt < MAX_SPIN_RETRIES; attempt++) {
        const bytesRead = fs.readSync(this._fd, this._headerBuf, 0, HEADER_SIZE, 0);
        if (bytesRead < HEADER_SIZE) return null;

        const header1 = parseShmHeader(this._headerBuf);
        if (!header1) return null;

        const seq1 = header1.frameReady;
        // In Seqlock: odd sequence number indicates write actively in progress
        if ((seq1 & 1) !== 0) {
          continue;
        }

        // Deduplication: frame counter must advance beyond last processed frame
        if (seq1 === 0 || seq1 <= this._lastFrameReady) {
          return null;
        }

        const frameBytes = header1.width * header1.height * 4;
        const slot = header1.writeIdx % header1.bufferCount;
        const pixelOffset = HEADER_SIZE + slot * frameBytes;

        const pixels = Buffer.allocUnsafe(frameBytes);
        const pixRead = fs.readSync(this._fd, pixels, 0, frameBytes, pixelOffset);
        if (pixRead < frameBytes) {
          return null;
        }

        // Re-read header to verify frame counter hasn't changed during pixel read
        const verifyRead = fs.readSync(this._fd, this._headerBuf2, 0, HEADER_SIZE, 0);
        if (verifyRead < HEADER_SIZE) return null;

        const header2 = parseShmHeader(this._headerBuf2);
        if (!header2) return null;

        const seq2 = header2.frameReady;
        // Strict Seqlock verification: both even and identical
        if ((seq2 & 1) === 0 && seq1 === seq2) {
          this._lastFrameReady = seq1;
          const frame = { header: header2, pixels };
          this.emit('frame', frame);
          return frame;
        }
      }

      // Writer was faster than reader or write was in progress for all attempts
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
  validateShmName,
  resolveShmFilePath,
  parseShmHeader,
  DEFAULT_SHM_NAME,
  HEADER_SIZE,
  SHM_BUFFER_COUNT,
  INSPECTOR_SHM_MAGIC,
  INSPECTOR_SHM_VERSION,
  OFF_MAGIC,
  OFF_VERSION,
  OFF_WIDTH,
  OFF_HEIGHT,
  OFF_WRITE_IDX,
  OFF_BUF_COUNT,
  OFF_FRAME_RDY,
  OFF_FPS,
  OFF_DELTA,
  OFF_DEBUG_VIEW,
};
