// test/test-bundle-loading.js — Tests for InspectionBundle loading, security,
// SHM protocol parsing, command whitelist, and native binary integration.
// Run with: node test/test-bundle-loading.js
'use strict';

const path = require('path');
const fs = require('fs');
const os = require('os');
const assert = require('assert');
const { spawnSync } = require('child_process');

const { BundleManager, SCHEMA_VERSION } = require('../electron/bundle-manager');
const {
  InspectorProcessManager,
  validateNativeCommand,
} = require('../electron/native-process');
const {
  ShmFrameReader,
  parseShmHeader,
  INSPECTOR_SHM_MAGIC,
  INSPECTOR_SHM_VERSION,
  HEADER_SIZE,
  SHM_BUFFER_COUNT,
} = require('../electron/shm-reader');

// ─── Test fixtures ───────────────────────────────────────────────

const FIXTURES_DIR = path.join(__dirname, 'fixtures');

function ensureFixtures() {
  const minimalDir = path.join(FIXTURES_DIR, 'minimal-bundle');
  const sourceDir = path.join(minimalDir, 'source');
  fs.mkdirSync(sourceDir, { recursive: true });

  const glbHeader = Buffer.alloc(12);
  glbHeader.writeUInt32LE(0x46546C67, 0); // magic: "glTF"
  glbHeader.writeUInt32LE(2, 4);           // version: 2
  glbHeader.writeUInt32LE(12, 8);          // length: 12
  fs.writeFileSync(path.join(sourceDir, 'model.glb'), glbHeader);

  const manifest = {
    schema_version: SCHEMA_VERSION,
    bundle_id: 'test_minimal_001',
    created_at: new Date().toISOString(),
    description: 'Minimal test bundle for unit testing',
    sources: {
      source_model: 'source/model.glb',
    },
    scene: {
      up_axis: 'Y',
      units: 'meters',
    },
  };
  fs.writeFileSync(
    path.join(minimalDir, 'bundle.json'),
    JSON.stringify(manifest, null, 2),
  );

  const fullDir = path.join(FIXTURES_DIR, 'full-bundle');
  const fullSource = path.join(fullDir, 'source');
  const fullAnimations = path.join(fullDir, 'source', 'animations');
  const fullFgpp = path.join(fullDir, 'fgpp');
  const fullGenerated = path.join(fullDir, 'generated');
  const fullRecon = path.join(fullDir, 'reconstructed');
  const fullMetrics = path.join(fullDir, 'metrics');

  for (const d of [fullSource, fullAnimations, fullFgpp, fullGenerated,
                    fullRecon, fullMetrics]) {
    fs.mkdirSync(d, { recursive: true });
  }

  fs.writeFileSync(path.join(fullSource, 'model.glb'), glbHeader);
  fs.writeFileSync(path.join(fullAnimations, 'walk.glb'), glbHeader);
  fs.writeFileSync(path.join(fullFgpp, 'metadata.json'), '{}');
  fs.writeFileSync(path.join(fullGenerated, 'video.mp4'), Buffer.alloc(8));
  fs.writeFileSync(path.join(fullRecon, 'model.glb'), glbHeader);
  fs.writeFileSync(
    path.join(fullMetrics, 'comparison.json'),
    JSON.stringify({ chamfer_mm: 2.34 }),
  );

  const fullManifest = {
    schema_version: SCHEMA_VERSION,
    bundle_id: 'test_full_roundtrip',
    created_at: new Date().toISOString(),
    description: 'Full roundtrip test bundle',
    sources: {
      source_model: 'source/model.glb',
      animations: ['source/animations/walk.glb'],
      fgpp_sequence: 'fgpp/',
      generated_video: 'generated/video.mp4',
      reconstructed_model: 'reconstructed/model.glb',
      metrics: 'metrics/comparison.json',
    },
    scene: {
      up_axis: 'Y',
      units: 'meters',
      scenario_id: 't2_humanoid_walk_orbit',
      adapter_id: 'wan_2.1_controlnext',
    },
  };
  fs.writeFileSync(
    path.join(fullDir, 'bundle.json'),
    JSON.stringify(fullManifest, null, 2),
  );

  return { minimalDir, fullDir };
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

const { minimalDir, fullDir } = ensureFixtures();
const mgr = new BundleManager();

// Test 1: Load minimal bundle.
test('loads minimal bundle successfully', () => {
  const result = mgr.loadBundle(minimalDir);
  assert.strictEqual(result.ok, true, `Expected ok=true, got: ${JSON.stringify(result)}`);
  assert.strictEqual(result.manifest.schema_version, SCHEMA_VERSION);
  assert.strictEqual(result.manifest.bundle_id, 'test_minimal_001');
  assert.ok(result.inventory.fileCount >= 2, 'Should have at least 2 files');
});

// Test 2: Load full roundtrip bundle.
test('loads full roundtrip bundle successfully', () => {
  const result = mgr.loadBundle(fullDir);
  assert.strictEqual(result.ok, true, `Expected ok=true, got: ${JSON.stringify(result)}`);
  assert.strictEqual(result.manifest.bundle_id, 'test_full_roundtrip');
  assert.ok(result.manifest.sources.animations.length > 0);
  assert.ok(result.inventory.fileCount >= 7, 'Full bundle should have many files');
});

// Test 3: getBundleInfo after load.
test('getBundleInfo returns loaded state', () => {
  mgr.loadBundle(minimalDir);
  const info = mgr.getBundleInfo();
  assert.strictEqual(info.loaded, true);
  assert.ok(info.path.includes('minimal-bundle'));
  assert.ok(info.manifest);
  assert.ok(info.inventory);
});

// Test 4: Reject non-existent directory.
test('rejects non-existent directory', () => {
  const result = mgr.loadBundle('/tmp/does-not-exist-xyz-12345');
  assert.strictEqual(result.ok, false);
  assert.ok(result.error.includes('not found'));
});

// Test 5: Reject directory without bundle.json.
test('rejects directory without bundle.json', () => {
  const emptyDir = path.join(FIXTURES_DIR, 'empty-dir');
  fs.mkdirSync(emptyDir, { recursive: true });
  const result = mgr.loadBundle(emptyDir);
  assert.strictEqual(result.ok, false);
  assert.ok(result.error.includes('Missing'));
  fs.rmdirSync(emptyDir);
});

// Test 6: Reject wrong schema version.
test('rejects wrong schema version', () => {
  const wrongDir = path.join(FIXTURES_DIR, 'wrong-schema');
  const wrongSource = path.join(wrongDir, 'source');
  fs.mkdirSync(wrongSource, { recursive: true });
  fs.writeFileSync(path.join(wrongSource, 'model.glb'), Buffer.alloc(12));
  fs.writeFileSync(
    path.join(wrongDir, 'bundle.json'),
    JSON.stringify({ schema_version: 'inspection.v999', sources: { source_model: 'source/model.glb' } }),
  );
  const result = mgr.loadBundle(wrongDir);
  assert.strictEqual(result.ok, false);
  assert.ok(result.error.includes('version mismatch'));
  fs.rmSync(wrongDir, { recursive: true, force: true });
});

// Test 7: Reject path traversal.
test('rejects path traversal in sources', () => {
  const traverseDir = path.join(FIXTURES_DIR, 'path-traverse');
  const traverseSource = path.join(traverseDir, 'source');
  fs.mkdirSync(traverseSource, { recursive: true });
  fs.writeFileSync(path.join(traverseSource, 'model.glb'), Buffer.alloc(12));
  fs.writeFileSync(
    path.join(traverseDir, 'bundle.json'),
    JSON.stringify({
      schema_version: SCHEMA_VERSION,
      sources: { source_model: '../../../etc/passwd' },
    }),
  );
  const result = mgr.loadBundle(traverseDir);
  assert.strictEqual(result.ok, false);
  assert.ok(result.error.includes('Unsafe path'));
  fs.rmSync(traverseDir, { recursive: true, force: true });
});

// Test 8: Reject absolute path in sources.
test('rejects absolute path in sources', () => {
  const absDir = path.join(FIXTURES_DIR, 'abs-path');
  const absSource = path.join(absDir, 'source');
  fs.mkdirSync(absSource, { recursive: true });
  fs.writeFileSync(path.join(absSource, 'model.glb'), Buffer.alloc(12));
  fs.writeFileSync(
    path.join(absDir, 'bundle.json'),
    JSON.stringify({
      schema_version: SCHEMA_VERSION,
      sources: { source_model: '/absolute/path/model.glb' },
    }),
  );
  const result = mgr.loadBundle(absDir);
  assert.strictEqual(result.ok, false);
  assert.ok(result.error.includes('Unsafe path'));
  fs.rmSync(absDir, { recursive: true, force: true });
});

// Test 9: Reject missing source model file.
test('rejects missing source model file', () => {
  const missingDir = path.join(FIXTURES_DIR, 'missing-model');
  fs.mkdirSync(missingDir, { recursive: true });
  fs.writeFileSync(
    path.join(missingDir, 'bundle.json'),
    JSON.stringify({
      schema_version: SCHEMA_VERSION,
      sources: { source_model: 'source/model.glb' },
    }),
  );
  const result = mgr.loadBundle(missingDir);
  assert.strictEqual(result.ok, false);
  assert.ok(result.error.includes('not found'));
  fs.rmSync(missingDir, { recursive: true, force: true });
});

// Test 10: Schema version constant is correct.
test('schema version constant matches expected value', () => {
  assert.strictEqual(SCHEMA_VERSION, 'inspection.v1.0');
});

// Test 11: validateNativeCommand whitelist and schema checks.
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
    validateNativeCommand({ type: 'set_entity_visible', name: 'hero_mesh', visible: false }).ok,
    true,
  );
  assert.strictEqual(
    validateNativeCommand({ type: 'load_bundle', path: minimalDir }).ok,
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

// Test 12: Security hardening static assertions (sandbox, preload unsubscribe, no innerHTML).
test('security hardening checks on main.js, preload.js, and app.js', () => {
  const mainSrc = fs.readFileSync(path.join(__dirname, '..', 'electron', 'main.js'), 'utf8');
  const preloadSrc = fs.readFileSync(path.join(__dirname, '..', 'electron', 'preload.js'), 'utf8');
  const appSrc = fs.readFileSync(path.join(__dirname, '..', 'renderer', 'src', 'app.js'), 'utf8');

  assert.ok(mainSrc.includes('sandbox: true'), 'main.js must enable sandbox: true');
  assert.ok(!mainSrc.includes('sandbox: false'), 'main.js must not contain sandbox: false');
  assert.ok(preloadSrc.includes('removeListener'), 'preload.js must return removeListener disposer');
  assert.ok(!appSrc.includes('.innerHTML'), 'renderer/src/app.js must not use innerHTML');
});

// Test 13: ShmFrameReader parses triple-buffered SHM regions accurately.
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

    // Subsequent poll without frame_ready increment returns null.
    assert.strictEqual(reader.pollOnce(), null);
    reader.close();
  } finally {
    fs.rmSync(tmpShmFile, { force: true });
  }
});

// Test 14: End-to-end native binary + SHM + command verification (when compiled).
const inspectorRoot = path.resolve(__dirname, '..');
const procMgr = new InspectorProcessManager(inspectorRoot);
const nativeBin = procMgr._resolveBinaryPath();

if (nativeBin) {
  test('end-to-end native C++ binary renders into @aether//pal SHM and handles stdin commands', () => {
    const shmName = `/inspector-e2e-${process.pid}`;
    const shotPath = path.join(os.tmpdir(), `inspector-shot-${process.pid}.ppm`);
    const commands = [
      JSON.stringify({ type: 'load_bundle', path: minimalDir }),
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
        '--frames=3',
        `--shm-name=${shmName}`,
        '--keep-shm',
        `--bundle=${minimalDir}`,
      ],
      {
        input: commands,
        encoding: 'utf8',
        timeout: 10000,
      },
    );

    assert.strictEqual(run.status, 0, `Native binary exited with ${run.status}: ${run.stderr}`);
    assert.ok(run.stdout.includes('"type":"ack","command":"load_bundle","ok":true'));
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
