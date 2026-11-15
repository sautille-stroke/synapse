// test/test-bundle-loading.js — Tests for InspectionBundle loading, security,
// SHM protocol parsing, command whitelist, and native binary integration.
// Run with: node test/test-bundle-loading.js
// H6-P1-07: Static test/fixtures/ are read-only immutable. All dynamic/negative
// test cases use os.tmpdir() with try/finally cleanup. Zero git status pollution.
'use strict';

const path = require('path');
const fs = require('fs');
const os = require('os');
const assert = require('assert');
const { spawnSync } = require('child_process');

const {
  BundleManager,
  SCHEMA_VERSION,
  ACCEPTED_SCHEMA_VERSIONS,
  extractEntryPath,
  extractEntryLabel,
  isMultiInstanceKey,
} = require('../electron/bundle-manager');
const {
  InspectorProcessManager,
  validateNativeCommand,
} = require('../electron/native-process');
const {
  ShmFrameReader,
  parseShmHeader,
  validateShmName,
  INSPECTOR_SHM_MAGIC,
  INSPECTOR_SHM_VERSION,
  HEADER_SIZE,
  SHM_BUFFER_COUNT,
  MACOS_PSHMNAMLEN,
} = require('../electron/shm-reader');

// ─── Test fixtures (read-only, committed to git) ─────────────────
// These are STATIC and IMMUTABLE. Tests must NOT write to FIXTURES_DIR.
// All dynamic / negative test cases create temp dirs under os.tmpdir().

const FIXTURES_DIR = path.join(__dirname, 'fixtures');
const MINIMAL_BUNDLE = path.join(FIXTURES_DIR, 'minimal-bundle');
const FULL_BUNDLE = path.join(FIXTURES_DIR, 'full-bundle');

// Helper: create a temp directory for a test case.
function makeTempDir(suffix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), `inspector-test-${suffix}-`));
}

// Helper: create a minimal valid bundle in a temp directory.
function createTempBundle(tmpDir, manifestOverrides = {}) {
  const sourceDir = path.join(tmpDir, 'source');
  fs.mkdirSync(sourceDir, { recursive: true });
  const glbHeader = Buffer.alloc(12);
  glbHeader.writeUInt32LE(0x46546C67, 0);
  glbHeader.writeUInt32LE(2, 4);
  glbHeader.writeUInt32LE(12, 8);
  fs.writeFileSync(path.join(sourceDir, 'model.glb'), glbHeader);

  const manifest = {
    schema_version: SCHEMA_VERSION,
    bundle_id: 'temp_test',
    sources: { source_model: 'source/model.glb' },
    ...manifestOverrides,
  };
  fs.writeFileSync(
    path.join(tmpDir, 'bundle.json'),
    JSON.stringify(manifest, null, 2),
  );
  return tmpDir;
}

// ─── Tests ───────────────────────────────────────────────────────

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  PASS: ${name}`);
  } catch (err) {
    failed++;
    console.error(`  FAIL: ${name}`);
    console.error(`        ${err.message}`);
  }
}

console.log('=== Data Inspector Bundle, Security & SHM Tests ===\n');

const mgr = new BundleManager();

// ── Section 1: Static fixture tests (read-only, no writes) ──────

// Test 1: Load minimal bundle from static fixtures.
test('loads minimal bundle successfully', () => {
  const result = mgr.loadBundle(MINIMAL_BUNDLE);
  assert.strictEqual(result.ok, true, `Expected ok=true, got: ${JSON.stringify(result)}`);
  assert.strictEqual(result.manifest.schema_version, SCHEMA_VERSION);
  assert.strictEqual(result.manifest.bundle_id, 'test_minimal_001');
  assert.ok(result.inventory.fileCount >= 2, 'Should have at least 2 files');
});

// Test 2: Load full roundtrip bundle from static fixtures.
test('loads full roundtrip bundle successfully', () => {
  const result = mgr.loadBundle(FULL_BUNDLE);
  assert.strictEqual(result.ok, true, `Expected ok=true, got: ${JSON.stringify(result)}`);
  assert.strictEqual(result.manifest.bundle_id, 'test_full_roundtrip');
  assert.ok(result.manifest.sources.animations.length > 0);
  assert.ok(result.inventory.fileCount >= 7, 'Full bundle should have many files');
});

// Test 3: getBundleInfo after load.
test('getBundleInfo returns loaded state', () => {
  mgr.loadBundle(MINIMAL_BUNDLE);
  const info = mgr.getBundleInfo();
  assert.strictEqual(info.loaded, true);
  assert.ok(info.path.includes('minimal-bundle'));
  assert.ok(info.manifest);
  assert.ok(info.inventory);
});

// ── Section 2: Dynamic / negative tests (os.tmpdir(), try/finally) ──

// Test 4: Reject non-existent directory.
test('rejects non-existent directory', () => {
  const result = mgr.loadBundle(path.join(os.tmpdir(), 'does-not-exist-xyz-12345'));
  assert.strictEqual(result.ok, false);
  assert.ok(result.error.includes('not found'));
});

// Test 5: Reject directory without bundle.json.
test('rejects directory without bundle.json', () => {
  const tmpDir = makeTempDir('no-manifest');
  try {
    const result = mgr.loadBundle(tmpDir);
    assert.strictEqual(result.ok, false);
    assert.ok(result.error.includes('Missing'));
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

// Test 6: Reject wrong schema version.
test('rejects wrong schema version', () => {
  const tmpDir = makeTempDir('wrong-schema');
  try {
    createTempBundle(tmpDir, { schema_version: 'inspection.v999' });
    const result = mgr.loadBundle(tmpDir);
    assert.strictEqual(result.ok, false);
    assert.ok(result.error.includes('version mismatch'));
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

// Test 7: Reject path traversal.
test('rejects path traversal in sources', () => {
  const tmpDir = makeTempDir('path-traverse');
  try {
    createTempBundle(tmpDir, {
      sources: { source_model: '../../../etc/passwd' },
    });
    const result = mgr.loadBundle(tmpDir);
    assert.strictEqual(result.ok, false);
    assert.ok(result.error.includes('Unsafe path'));
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

// Test 8: Reject absolute path in sources.
test('rejects absolute path in sources', () => {
  const tmpDir = makeTempDir('abs-path');
  try {
    createTempBundle(tmpDir, {
      sources: { source_model: '/absolute/path/model.glb' },
    });
    const result = mgr.loadBundle(tmpDir);
    assert.strictEqual(result.ok, false);
    assert.ok(result.error.includes('Unsafe path'));
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

// Test 9: Reject missing source model file.
test('rejects missing source model file', () => {
  const tmpDir = makeTempDir('missing-model');
  try {
    fs.mkdirSync(tmpDir, { recursive: true });
    fs.writeFileSync(
      path.join(tmpDir, 'bundle.json'),
      JSON.stringify({
        schema_version: SCHEMA_VERSION,
        sources: { source_model: 'source/model.glb' },
      }),
    );
    const result = mgr.loadBundle(tmpDir);
    assert.strictEqual(result.ok, false);
    assert.ok(result.error.includes('not found'));
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

// Test 10: Schema version constant and accepted versions are correct.
test('schema version constants match expected values', () => {
  assert.strictEqual(SCHEMA_VERSION, 'inspection.v1.0');
  assert.ok(ACCEPTED_SCHEMA_VERSIONS.has('inspection.v1.0'));
  assert.ok(ACCEPTED_SCHEMA_VERSIONS.has('synapse.bundle.v1.0'));
});

// Test 10b: Load bundle where sources.source_model is a SourceEntry object.
test('loads bundle with SourceEntry object for source_model', () => {
  const tmpDir = makeTempDir('source-entry-obj');
  try {
    createTempBundle(tmpDir, {
      bundle_id: 'test_source_entry_obj',
      sources: {
        source_model: {
          path: 'source/model.glb',
          media_type: 'model/gltf-binary',
          sha256: 'a'.repeat(64),
        },
      },
    });
    const result = mgr.loadBundle(tmpDir);
    assert.strictEqual(result.ok, true, `Expected ok=true, got: ${JSON.stringify(result)}`);
    assert.strictEqual(result.manifest.bundle_id, 'test_source_entry_obj');
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

// Test 10c: Load bundle with synapse.bundle.v1.0 schema.
test('loads bundle with synapse.bundle.v1.0 schema version', () => {
  const tmpDir = makeTempDir('synapse-bundle-v1');
  try {
    createTempBundle(tmpDir, {
      schema_version: 'synapse.bundle.v1.0',
      bundle_id: 'test_synapse_bundle_v1',
      sources: {
        source_model: {
          path: 'source/model.glb',
          media_type: 'model/gltf-binary',
        },
      },
    });
    const result = mgr.loadBundle(tmpDir);
    assert.strictEqual(result.ok, true, `Expected ok=true, got: ${JSON.stringify(result)}`);
    assert.strictEqual(result.manifest.schema_version, 'synapse.bundle.v1.0');
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

// Test 10d: Reject path traversal when source entry is an object.
test('rejects path traversal when source entry is an object', () => {
  const tmpDir = makeTempDir('bad-source-obj');
  try {
    createTempBundle(tmpDir, {
      sources: {
        source_model: {
          path: '../../etc/passwd',
          media_type: 'model/gltf-binary',
        },
      },
    });
    const result = mgr.loadBundle(tmpDir);
    assert.strictEqual(result.ok, false);
    assert.ok(result.error.includes('Unsafe path'));
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

// ── Section 3: extractEntryPath / extractEntryLabel ─────────────

test('extractEntryPath handles string and object entries', () => {
  assert.strictEqual(extractEntryPath('source/model.glb'), 'source/model.glb');
  assert.strictEqual(extractEntryPath({ path: 'source/model.glb' }), 'source/model.glb');
  assert.strictEqual(extractEntryPath({ url: 'http://example.com/model.glb' }), 'http://example.com/model.glb');
  assert.strictEqual(extractEntryPath(null), null);
  assert.strictEqual(extractEntryPath(42), null);
});

test('extractEntryLabel returns display-safe strings', () => {
  assert.strictEqual(extractEntryLabel('source/model.glb'), 'source/model.glb');
  assert.strictEqual(extractEntryLabel({ path: 'source/model.glb' }), 'source/model.glb');
  assert.strictEqual(extractEntryLabel({ label: 'My Model' }), 'My Model');
  assert.strictEqual(extractEntryLabel(null), '');
  // Must NOT produce '[object Object]'.
  const label = extractEntryLabel({ path: 'a.glb', media_type: 'x' });
  assert.ok(!label.includes('[object'), `extractEntryLabel should not return [object Object]: got '${label}'`);
});

// ── Section 4: Multi-instance source keys ────────────────────────

test('isMultiInstanceKey identifies animation_*, generated_*, fgpp_* prefixes', () => {
  assert.strictEqual(isMultiInstanceKey('animation_walk'), true);
  assert.strictEqual(isMultiInstanceKey('animation_run'), true);
  assert.strictEqual(isMultiInstanceKey('generated_video_01'), true);
  assert.strictEqual(isMultiInstanceKey('fgpp_sequence_02'), true);
  assert.strictEqual(isMultiInstanceKey('source_model'), false);
  assert.strictEqual(isMultiInstanceKey('metrics'), false);
});

test('loads bundle with multi-instance source keys', () => {
  const tmpDir = makeTempDir('multi-instance');
  try {
    const animDir = path.join(tmpDir, 'anims');
    fs.mkdirSync(animDir, { recursive: true });
    const glbHeader = Buffer.alloc(12);
    glbHeader.writeUInt32LE(0x46546C67, 0);
    glbHeader.writeUInt32LE(2, 4);
    glbHeader.writeUInt32LE(12, 8);

    const sourceDir = path.join(tmpDir, 'source');
    fs.mkdirSync(sourceDir, { recursive: true });
    fs.writeFileSync(path.join(sourceDir, 'model.glb'), glbHeader);
    fs.writeFileSync(path.join(animDir, 'walk.glb'), glbHeader);
    fs.writeFileSync(path.join(animDir, 'run.glb'), glbHeader);

    fs.writeFileSync(
      path.join(tmpDir, 'bundle.json'),
      JSON.stringify({
        schema_version: SCHEMA_VERSION,
        bundle_id: 'test_multi_instance',
        sources: {
          source_model: 'source/model.glb',
          animation_walk: 'anims/walk.glb',
          animation_run: 'anims/run.glb',
        },
      }),
    );

    const result = mgr.loadBundle(tmpDir);
    assert.strictEqual(result.ok, true, `Expected ok=true, got: ${JSON.stringify(result)}`);
    assert.strictEqual(result.manifest.sources.animation_walk, 'anims/walk.glb');
    assert.strictEqual(result.manifest.sources.animation_run, 'anims/run.glb');
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

// ── Section 5: Command whitelist and validation ──────────────────

test('validateNativeCommand enforces command whitelist and schemas', () => {
  assert.strictEqual(
    validateNativeCommand({ type: 'camera', action: 'orbit', dx: 12.5, dy: -4.0 }).ok,
    true,
  );
  assert.strictEqual(
    validateNativeCommand({ type: 'set_debug_view', mode: 'wireframe' }).ok,
    true,
  );
  assert.strictEqual(
    validateNativeCommand({ type: 'set_debug_view', mode: 'depth' }).ok,
    true,
  );
  assert.strictEqual(
    validateNativeCommand({ type: 'set_debug_view', mode: 'cryptomatte' }).ok,
    true,
  );
  // H6-P0-10: New debug modes.
  assert.strictEqual(
    validateNativeCommand({ type: 'set_debug_view', mode: 'albedo' }).ok,
    true,
  );
  assert.strictEqual(
    validateNativeCommand({ type: 'set_debug_view', mode: 'motion' }).ok,
    true,
  );
  assert.strictEqual(
    validateNativeCommand({ type: 'set_debug_view', mode: 'normals' }).ok,
    true,
  );
  // 'normals' normalizes to 'normal'.
  const normalsCmd = validateNativeCommand({ type: 'set_debug_view', mode: 'normals' });
  assert.strictEqual(normalsCmd.command.mode, 'normal');

  const rgbNorm = validateNativeCommand({ type: 'set_debug_view', mode: 'rgb' });
  assert.strictEqual(rgbNorm.ok, true);
  assert.strictEqual(rgbNorm.command.mode, 'lit');
  const switchViewCmd = validateNativeCommand({ type: 'switch_view', view: 'normal' });
  assert.strictEqual(switchViewCmd.ok, true);
  assert.strictEqual(switchViewCmd.command.mode, 'normal');
  assert.strictEqual(
    validateNativeCommand({ type: 'set_entity_visible', name: 'hero_mesh', visible: false }).ok,
    true,
  );
  assert.strictEqual(
    validateNativeCommand({ type: 'load_bundle', path: MINIMAL_BUNDLE }).ok,
    true,
  );
  assert.strictEqual(
    validateNativeCommand({ type: 'quit' }).ok,
    true,
  );

  // Disallowed / malformed commands.
  assert.strictEqual(validateNativeCommand(null).ok, false);
  assert.strictEqual(validateNativeCommand({ type: 'exec_shell', cmd: 'rm -rf /' }).ok, false);
  assert.strictEqual(validateNativeCommand({ type: 'camera', action: 'fly', dx: 1, dy: 1 }).ok, false);
  assert.strictEqual(validateNativeCommand({ type: 'camera', action: 'orbit', dx: NaN, dy: 0 }).ok, false);
  assert.strictEqual(validateNativeCommand({ type: 'set_debug_view', mode: 'xray' }).ok, false);
  assert.strictEqual(validateNativeCommand({ type: 'load_bundle', path: 'bad"quote' }).ok, false);
});

// ── Section 6: Security hardening static assertions ──────────────

test('security hardening checks on main.js, preload.js, and app.js', () => {
  const mainSrc = fs.readFileSync(path.join(__dirname, '..', 'electron', 'main.js'), 'utf8');
  const preloadSrc = fs.readFileSync(path.join(__dirname, '..', 'electron', 'preload.js'), 'utf8');
  const appSrc = fs.readFileSync(path.join(__dirname, '..', 'renderer', 'src', 'app.js'), 'utf8');

  assert.ok(mainSrc.includes('sandbox: true'), 'main.js must enable sandbox: true');
  assert.ok(!mainSrc.includes('sandbox: false'), 'main.js must not contain sandbox: false');
  assert.ok(mainSrc.includes('will-navigate'), 'main.js must intercept will-navigate');
  assert.ok(mainSrc.includes('setWindowOpenHandler'), 'main.js must use setWindowOpenHandler');
  assert.ok(mainSrc.includes('NODE_ENV'), 'main.js must gate CDP on NODE_ENV');
  assert.ok(preloadSrc.includes('removeListener'), 'preload.js must return removeListener disposer');
  assert.ok(!appSrc.includes('.innerHTML'), 'renderer/src/app.js must not use innerHTML');
  assert.ok(appSrc.includes('extractDisplayLabel'), 'app.js must use extractDisplayLabel for source entries');
  assert.ok(appSrc.includes('.catch('), 'app.js must catch Promise rejections on nativeSendCommand');
});

// ── Section 7: SHM validation ────────────────────────────────────

test('validateShmName enforces POSIX rules and macOS PSHMNAMLEN', () => {
  const ok1 = validateShmName('/inspector-frame');
  assert.strictEqual(ok1.valid, true);
  assert.strictEqual(ok1.normalized, '/inspector-frame');

  const ok2 = validateShmName('inspector-frame');
  assert.strictEqual(ok2.valid, true);
  assert.strictEqual(ok2.normalized, '/inspector-frame');

  const bad1 = validateShmName('');
  assert.strictEqual(bad1.valid, false);

  const bad2 = validateShmName(null);
  assert.strictEqual(bad2.valid, false);

  // Reject embedded slashes.
  const bad3 = validateShmName('/a/b');
  assert.strictEqual(bad3.valid, false);
});

test('ShmFrameReader parses triple-buffered SHM header and pixel slot', () => {
  const w = 4;
  const h = 4;
  const frameBytes = w * h * 4;
  const totalBytes = HEADER_SIZE + frameBytes * SHM_BUFFER_COUNT;
  const buf = Buffer.alloc(totalBytes);

  buf.writeUInt32LE(INSPECTOR_SHM_MAGIC, 0);
  buf.writeUInt32LE(INSPECTOR_SHM_VERSION, 4);
  buf.writeUInt32LE(w, 8);
  buf.writeUInt32LE(h, 12);
  buf.writeUInt32LE(2, 16); // write_idx = slot 2
  buf.writeUInt32LE(SHM_BUFFER_COUNT, 20);
  buf.writeBigUInt64LE(5n, 24); // frame_ready = 5
  buf.writeFloatLE(60.0, 32);
  buf.writeFloatLE(0.016, 36);
  buf.writeUInt8(1, 40); // wireframe

  // Fill slot 2 with distinctive pixel values [42, 84, 126, 255].
  const slot2Offset = HEADER_SIZE + 2 * frameBytes;
  for (let i = 0; i < frameBytes; i += 4) {
    buf[slot2Offset + i + 0] = 42;
    buf[slot2Offset + i + 1] = 84;
    buf[slot2Offset + i + 2] = 126;
    buf[slot2Offset + i + 3] = 255;
  }

  const tmpShmFile = path.join(os.tmpdir(), `test-inspector-shm-${process.pid}.bin`);
  fs.writeFileSync(tmpShmFile, buf);

  try {
    const reader = new ShmFrameReader(tmpShmFile);
    let emittedFrame = null;
    reader.on('frame', (f) => { emittedFrame = f; });

    const polled = reader.pollOnce();
    assert.ok(polled, 'pollOnce should return frame');
    assert.ok(emittedFrame, 'reader should emit frame event');
    assert.strictEqual(polled.header.width, w);
    assert.strictEqual(polled.header.height, h);
    assert.strictEqual(polled.header.writeIdx, 2);
    assert.strictEqual(polled.header.frameReady, 5);
    assert.strictEqual(polled.header.debugView, 1);
    assert.strictEqual(polled.pixels[0], 42);
    assert.strictEqual(polled.pixels[1], 84);
    assert.strictEqual(polled.pixels[2], 126);
    assert.strictEqual(polled.pixels[3], 255);

    // Subsequent poll without frame_ready increment returns null (spin-lock dedup).
    assert.strictEqual(reader.pollOnce(), null);
    reader.close();
  } finally {
    fs.rmSync(tmpShmFile, { force: true });
  }
});

// ── Section 8: End-to-end native binary (conditional) ────────────

const inspectorRoot = path.resolve(__dirname, '..');
const procMgr = new InspectorProcessManager(inspectorRoot);
const nativeBin = procMgr._resolveBinaryPath();

if (nativeBin) {
  test('end-to-end native C++ binary renders into @aether//pal SHM and handles stdin commands', () => {
    const shmName = `/inspector-e2e-${process.pid}`;
    const shotPath = path.join(os.tmpdir(), `inspector-shot-${process.pid}.ppm`);
    const commands = [
      JSON.stringify({ type: 'load_bundle', path: MINIMAL_BUNDLE }),
      JSON.stringify({ type: 'camera', action: 'orbit', dx: 25.0, dy: 10.0 }),
      JSON.stringify({ type: 'set_debug_view', mode: 'normal' }),
      JSON.stringify({ type: 'export_screenshot', path: shotPath }),
    ].join('\n') + '\n';

    const run = spawnSync(
      nativeBin,
      [
        '--headless',
        '--rhi=null',
        '--width=32',
        '--height=32',
        `--shm-name=${shmName}`,
        '--keep-shm',
        `--bundle=${MINIMAL_BUNDLE}`,
      ],
      {
        input: commands,
        encoding: 'utf8',
        timeout: 10000,
      },
    );

    assert.strictEqual(run.status, 0, `Native binary exited with ${run.status}: ${run.stderr}`);
    assert.ok(
      run.stdout.includes('"type":"ack","command":"load_bundle","ok":true'),
      `Missing load_bundle ack. STDOUT was:\n${run.stdout}\nSTDERR was:\n${run.stderr}`,
    );
    assert.ok(run.stdout.includes('"type":"ack","command":"camera","ok":true'));
    assert.ok(run.stdout.includes('"type":"ack","command":"set_debug_view","ok":true'));
    assert.ok(run.stdout.includes('"type":"frame_ready"'));

    const reader = new ShmFrameReader(shmName);
    try {
      const frame = reader.pollOnce();
      assert.ok(frame, 'ShmFrameReader must read rendered frame from native SHM');
      assert.strictEqual(frame.header.width, 32);
      assert.strictEqual(frame.header.height, 32);
      assert.ok(frame.header.frameReady >= 1);
      assert.strictEqual(frame.header.debugView, 2); // kNormal = 2
      assert.strictEqual(frame.pixels.length, 32 * 32 * 4);

      // Verify non-black pixels.
      let nonZeroSum = 0;
      for (let i = 0; i < frame.pixels.length; i += 4) {
        nonZeroSum += frame.pixels[i] + frame.pixels[i + 1] + frame.pixels[i + 2];
      }
      assert.ok(nonZeroSum > 0, 'Rendered SHM frame must contain non-zero RGB pixels');

      // Verify exported PPM screenshot.
      assert.ok(fs.existsSync(shotPath), 'export_screenshot should write PPM file');
      const ppmData = fs.readFileSync(shotPath);
      assert.ok(ppmData.subarray(0, 3).toString('ascii') === 'P6\n');
      assert.ok(ppmData.length > 32 * 32 * 3);
    } finally {
      reader.close();
      try {
        const shmPath = path.join('/dev/shm', shmName.replace(/^\/+/, ''));
        if (fs.existsSync(shmPath)) fs.unlinkSync(shmPath);
      } catch (_) { /* ignore */ }
      fs.rmSync(shotPath, { force: true });
    }
  });
}

// ─── Summary ─────────────────────────────────────────────────────
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
process.exit(failed > 0 ? 1 : 0);
