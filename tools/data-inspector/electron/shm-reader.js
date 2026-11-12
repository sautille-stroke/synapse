// electron/shm-reader.js — Shared-memory frame reader for the Data Inspector.
// Reads triple-buffered frames written by InspectorShmServer (@aether//pal:shm).
'use strict';

const EventEmitter = require('events');
const fs = require('fs');
const os = require('os');
const path = require('path');

const DEFAULT_SHM_NAME = '/inspector-frame';
const HEADER_SIZE = 64;
const SHM_BUFFER_COUNT = 3;

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

/**
 * Resolve a POSIX SHM name (e.g. "/inspector-frame-123") to its backing path.
 * On Linux, POSIX shm_open() creates files under /dev/shm/<name>.
 * @param {string} shmName
 * @returns {string}
 */
function resolveShmFilePath(shmName) {
  if (!shmName || typeof shmName !== 'string') {
    throw new Error('Invalid SHM name');
  }
  if (fs.existsSync(shmName)) {
    return shmName;
  }
  const cleanName = shmName.replace(/^\/+/, '');
  const devShmPath = path.join('/dev/shm', cleanName);
  if (fs.existsSync('/dev/shm')) {
    return devShmPath;
  }
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
      const pixels = Buffer.allocUnsafe(frameBytes);

      const pixRead = fs.readSync(this._fd, pixels, 0, frameBytes, pixelOffset);
      if (pixRead < frameBytes) {
        return null;
      }

      this._lastFrameReady = header.frameReady;
      const frame = { header, pixels };
      this.emit('frame', frame);
      return frame;
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
  DEFAULT_SHM_NAME,
  INSPECTOR_SHM_MAGIC,
  INSPECTOR_SHM_VERSION,
  HEADER_SIZE,
  SHM_BUFFER_COUNT,
};
