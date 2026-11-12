// electron/shm-reader.js — SHM frame reader stub for the Data Inspector.
// In production, this loads the N-API shm_transport.node addon for zero-copy
// frame transport. For the initial scaffold, we use a stdin/stdout JSON
// protocol fallback with the native process.
'use strict';

const EventEmitter = require('events');

const DEFAULT_SHM_NAME = '/inspector-frame';
const HEADER_SIZE = 64;

// Header field offsets (must match inspector_shm_server.h).
const OFF_MAGIC      = 0;   // u32
const OFF_VERSION    = 4;   // u32
const OFF_WIDTH      = 8;   // u32
const OFF_HEIGHT     = 12;  // u32
const OFF_WRITE_IDX  = 16;  // u32
const OFF_FRAME_RDY  = 24;  // u64
const OFF_FPS        = 32;  // f32
const OFF_DELTA      = 36;  // f32

// Inspector SHM magic: "INSP" = 0x494E5350
const INSPECTOR_SHM_MAGIC = 0x494E5350;

/**
 * SHM frame reader — currently a stub that emits frame events
 * when pixels arrive via IPC from the main process.
 * Full N-API SHM transport will be added when the native addon is built.
 */
class ShmFrameReader extends EventEmitter {
  constructor(shmName) {
    super();
    this._shmName = shmName || DEFAULT_SHM_NAME;
    this._running = false;
  }

  open(_width, _height) {
    // Stub: SHM addon not yet available.
    return false;
  }

  start() {
    this._running = true;
  }

  stop() {
    this._running = false;
  }

  close() {
    this.stop();
  }
}

module.exports = {
  ShmFrameReader,
  DEFAULT_SHM_NAME,
  INSPECTOR_SHM_MAGIC,
  HEADER_SIZE,
};
