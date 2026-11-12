// test/test-bundle-loading.js — Tests for InspectionBundle loading and validation.
// Run with: node test/test-bundle-loading.js
'use strict';

const path = require('path');
const fs = require('fs');
const assert = require('assert');

// Import the BundleManager from the electron module.
const { BundleManager, SCHEMA_VERSION } = require('../electron/bundle-manager');

// ─── Test fixtures ───────────────────────────────────────────────

const FIXTURES_DIR = path.join(__dirname, 'fixtures');

function ensureFixtures() {
  // Create a minimal valid bundle.
  const minimalDir = path.join(FIXTURES_DIR, 'minimal-bundle');
  const sourceDir = path.join(minimalDir, 'source');
  fs.mkdirSync(sourceDir, { recursive: true });

  // Write a minimal glTF binary stub.
  const glbHeader = Buffer.alloc(12);
  glbHeader.writeUInt32LE(0x46546C67, 0); // magic: "glTF"
  glbHeader.writeUInt32LE(2, 4);           // version: 2
  glbHeader.writeUInt32LE(12, 8);          // length: 12 (header only)
  fs.writeFileSync(path.join(sourceDir, 'model.glb'), glbHeader);

  // Write bundle.json manifest.
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

  // Create a full roundtrip bundle.
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

console.log('=== Data Inspector Bundle Loading Tests ===\n');

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
  // Cleanup.
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
  // Cleanup.
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
  // Cleanup.
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
  // Cleanup.
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
  // Cleanup.
  fs.rmSync(missingDir, { recursive: true, force: true });
});

// Test 10: Schema version constant is correct.
test('schema version constant matches expected value', () => {
  assert.strictEqual(SCHEMA_VERSION, 'inspection.v1.0');
});

// ─── Summary ─────────────────────────────────────────────────────
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
process.exit(failed > 0 ? 1 : 0);
