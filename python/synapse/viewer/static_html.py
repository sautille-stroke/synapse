"""Self-contained interactive HTML/JS/CSS dashboard for Synapse Viewer."""

VIEWER_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Synapse — 3D & Neural Video Dataset Viewer</title>
  <style>
    :root {
      --bg-primary: #0d1117;
      --bg-secondary: #161b22;
      --bg-card: #21262d;
      --border-color: #30363d;
      --text-main: #f0f6fc;
      --text-muted: #8b949e;
      --accent-cyan: #388bfd;
      --accent-green: #2ea043;
      --accent-amber: #d29922;
      --accent-purple: #a371f7;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg-primary); color: var(--text-main); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

    /* Top Navigation Header */
    header {
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border-color);
      padding: 10px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
    }

    .brand { display: flex; align-items: center; gap: 12px; }
    .brand h1 { font-size: 18px; font-weight: 700; letter-spacing: -0.5px; }
    .brand .tag {
      background: rgba(56, 139, 253, 0.15);
      color: var(--accent-cyan);
      border: 1px solid rgba(56, 139, 253, 0.4);
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 600;
    }

    .meta-bar { display: flex; gap: 16px; font-size: 13px; align-items: center; }
    .meta-item { display: flex; gap: 6px; }
    .meta-label { color: var(--text-muted); }
    .badge {
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: 600;
      font-size: 12px;
    }
    .badge-train { background: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid rgba(46, 160, 67, 0.4); }
    .badge-val { background: rgba(210, 153, 34, 0.2); color: #e3b341; border: 1px solid rgba(210, 153, 34, 0.4); }

    /* Main Workspace */
    .workspace {
      display: flex;
      flex: 1;
      overflow: hidden;
    }

    /* Left: Multi-Channel Viewports */
    .viewport-panel {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 16px;
      overflow-y: auto;
      gap: 12px;
    }

    .channel-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      flex: 1;
    }

    .channel-card {
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .channel-header {
      padding: 8px 12px;
      background: var(--bg-card);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      font-weight: 600;
    }

    .channel-img-wrapper {
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 1;
      background: #080a0f;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    .channel-img-wrapper img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      image-rendering: pixelated;
    }

    /* Right: 3D Trajectory & Inspector Sidebar */
    .sidebar {
      width: 360px;
      background: var(--bg-secondary);
      border-left: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
    }

    .sidebar-section {
      padding: 14px;
      border-bottom: 1px solid var(--border-color);
    }
    .sidebar-section h3 {
      font-size: 13px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 10px;
    }

    #trajectory-canvas {
      width: 100%;
      height: 220px;
      background: #090c10;
      border-radius: 6px;
      border: 1px solid var(--border-color);
      cursor: crosshair;
    }

    .prop-table {
      width: 100%;
      font-size: 12px;
      border-collapse: collapse;
    }
    .prop-table td {
      padding: 5px 0;
      border-bottom: 1px solid #21262d;
    }
    .prop-table td:first-child { color: var(--text-muted); width: 40%; }
    .prop-table td:last-child { font-family: monospace; color: #79c0ff; text-align: right; }

    /* Bottom: Timeline Controller */
    .timeline-bar {
      background: var(--bg-secondary);
      border-top: 1px solid var(--border-color);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      gap: 16px;
      flex-shrink: 0;
    }

    .ctrl-btn {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 13px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .ctrl-btn:hover { background: #30363d; border-color: #8b949e; }
    .ctrl-btn.active { background: var(--accent-cyan); color: white; border-color: var(--accent-cyan); }

    .slider-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    input[type=range] {
      width: 100%;
      accent-color: var(--accent-cyan);
      cursor: pointer;
    }

    .time-labels {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      font-family: monospace;
      color: var(--text-muted);
    }

    .split-indicator {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Links & Downloads */
    .btn-link {
      color: var(--accent-cyan);
      text-decoration: none;
      font-size: 12px;
    }
    .btn-link:hover { text-decoration: underline; }
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="brand">
      <h1>SYNAPSE</h1>
      <span class="tag">FGPP & 3D Interop Hub</span>
      <span id="dataset-name" style="font-size: 14px; font-weight: 500; color: #c9d1d9;">Dataset</span>
    </div>
    <div class="meta-bar">
      <div class="meta-item"><span class="meta-label">Model:</span> <strong id="lbl-model">-</strong></div>
      <div class="meta-item"><span class="meta-label">Faces:</span> <span id="lbl-faces">-</span></div>
      <div class="meta-item"><span class="meta-label">Trajectory:</span> <span id="lbl-traj">-</span></div>
      <div class="meta-item"><span class="meta-label">Resolution:</span> <span id="lbl-res">-</span></div>
      <div class="meta-item">
        <span class="badge badge-train" id="lbl-train-count">Train: 0</span>
        <span class="badge badge-val" id="lbl-val-count">Val: 0</span>
      </div>
    </div>
  </header>

  <!-- Main View -->
  <div class="workspace">
    <!-- Viewports -->
    <div class="viewport-panel">
      <div class="channel-grid">
        <!-- 1. RGB -->
        <div class="channel-card">
          <div class="channel-header">
            <span>① RGB (Color / Radiance)</span>
            <span style="color: #3fb950;">PBR Shaded</span>
          </div>
          <div class="channel-img-wrapper">
            <img id="img-rgb" src="" alt="RGB">
          </div>
        </div>

        <!-- 2. Depth -->
        <div class="channel-card">
          <div class="channel-header">
            <span>② Depth (Linear Camera Depth)</span>
            <span style="color: #58a6ff;">Turbo Colormap</span>
          </div>
          <div class="channel-img-wrapper">
            <img id="img-depth" src="" alt="Depth">
          </div>
        </div>

        <!-- 3. World Normal -->
        <div class="channel-card">
          <div class="channel-header">
            <span>③ World Normal [-1,1]³</span>
            <span style="color: #d2a8ff;">RGB Normalized</span>
          </div>
          <div class="channel-img-wrapper">
            <img id="img-normal" src="" alt="Normal">
          </div>
        </div>

        <!-- 4. Optical Flow -->
        <div class="channel-card">
          <div class="channel-header">
            <span>④ Motion Vector (Optical Flow)</span>
            <span style="color: #f0883e;">HSV Wheel</span>
          </div>
          <div class="channel-img-wrapper">
            <img id="img-flow" src="" alt="Flow">
          </div>
        </div>
      </div>
    </div>

    <!-- Sidebar: 3D Trajectory & Inspector -->
    <div class="sidebar">
      <div class="sidebar-section">
        <h3>3D Camera Trajectory (Click to Jump)</h3>
        <canvas id="trajectory-canvas" width="330" height="220"></canvas>
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
          <span style="color: #3fb950;">● Train Cameras</span>
          <span style="color: #e3b341;">● Val Cameras</span>
          <span style="color: #f85149;">★ Current Active</span>
        </div>
      </div>

      <div class="sidebar-section">
        <h3>Frame Metadata & Inspector</h3>
        <table class="prop-table">
          <tr><td>Frame Index</td><td id="prop-frame">0 / 0</td></tr>
          <tr><td>Split Set</td><td id="prop-split"><span class="badge badge-train">TRAIN</span></td></tr>
          <tr><td>Camera Pos X</td><td id="prop-pos-x">0.000</td></tr>
          <tr><td>Camera Pos Y</td><td id="prop-pos-y">0.000</td></tr>
          <tr><td>Camera Pos Z</td><td id="prop-pos-z">0.000</td></tr>
          <tr><td>Distance to Origin</td><td id="prop-dist">0.000 m</td></tr>
          <tr><td>FOV (Field of View)</td><td id="prop-fov">50.0°</td></tr>
        </table>
      </div>

      <div class="sidebar-section" style="flex: 1;">
        <h3>Video & Artifact Exports</h3>
        <div style="display: flex; flex-direction: column; gap: 8px; font-size: 13px;">
          <a class="btn-link" id="link-video-all" href="video_all.mp4" target="_blank">🎬 Download Full Video (MP4)</a>
          <a class="btn-link" id="link-video-train" href="video_train.mp4" target="_blank">🟢 Download Train Set Video (MP4)</a>
          <a class="btn-link" id="link-video-val" href="video_val.mp4" target="_blank">🟡 Download Validation Video (MP4)</a>
          <a class="btn-link" id="link-contact" href="contact_sheet.png" target="_blank">🖼️ View Contact Sheet</a>
          <a class="btn-link" href="manifest.json" target="_blank">📄 View Dataset Manifest (JSON)</a>
        </div>
      </div>
    </div>
  </div>

  <!-- Timeline Controller -->
  <div class="timeline-bar">
    <button class="ctrl-btn" id="btn-play">▶ Play</button>
    <button class="ctrl-btn" id="btn-prev">⏮ Prev</button>
    <button class="ctrl-btn" id="btn-next">Next ⏭</button>

    <div class="slider-container">
      <input type="range" id="timeline" min="0" max="0" value="0">
      <div class="time-labels">
        <span id="lbl-time-cur">Frame 0</span>
        <span id="lbl-time-total">Frame 0</span>
      </div>
    </div>

    <div class="split-indicator">
      <span class="badge badge-train" id="badge-active-split">TRAIN</span>
    </div>

    <select id="sel-fps" class="ctrl-btn" style="padding: 6px 10px;">
      <option value="12">0.5x (12 fps)</option>
      <option value="24" selected>1.0x (24 fps)</option>
      <option value="48">2.0x (48 fps)</option>
    </select>
  </div>

  <script>
    const EMBEDDED_MANIFEST = /*__EMBEDDED_MANIFEST__*/ null;
    const EMBEDDED_IMAGES = /*__EMBEDDED_IMAGES__*/ {};

    let manifest = null;
    let currentFrame = 0;
    let isPlaying = false;
    let playInterval = null;
    let playbackFps = 24;

    const elTimeline = document.getElementById('timeline');
    const elBtnPlay = document.getElementById('btn-play');
    const elBtnPrev = document.getElementById('btn-prev');
    const elBtnNext = document.getElementById('btn-next');
    const elSelFps = document.getElementById('sel-fps');
    const canvasTraj = document.getElementById('trajectory-canvas');
    const ctxTraj = canvasTraj.getContext('2d');

    // Fetch or load embedded manifest
    async function init() {
      try {
        if (typeof EMBEDDED_MANIFEST !== 'undefined' && EMBEDDED_MANIFEST !== null) {
          manifest = EMBEDDED_MANIFEST;
        } else {
          const resp = await fetch('manifest.json');
          manifest = await resp.json();
        }
        setupUI();
        renderFrame(0);
        drawTrajectory();
      } catch (e) {
        console.error("Failed to load manifest.json", e);
        alert("Failed to load dataset manifest.json. Ensure viewer server is running.");
      }
    }

    function setupUI() {
      document.getElementById('dataset-name').textContent = manifest.dataset_id || "Synapse Dataset";
      document.getElementById('lbl-model').textContent = manifest.model?.name || "Model";
      document.getElementById('lbl-faces').textContent = (manifest.model?.num_faces || 0).toLocaleString();
      document.getElementById('lbl-traj').textContent = manifest.trajectory?.type || "orbit";
      document.getElementById('lbl-res').textContent = `${manifest.trajectory?.width || 512}x${manifest.trajectory?.height || 512}`;
      document.getElementById('lbl-train-count').textContent = `Train: ${manifest.splits?.train_count || 0}`;
      document.getElementById('lbl-val-count').textContent = `Val: ${manifest.splits?.val_count || 0}`;

      const totalFrames = manifest.frames.length;
      elTimeline.max = totalFrames - 1;
      document.getElementById('lbl-time-total').textContent = `Total: ${totalFrames} frames`;
    }

    function renderFrame(idx) {
      if (!manifest || idx < 0 || idx >= manifest.frames.length) return;
      currentFrame = idx;
      elTimeline.value = idx;

      const f = manifest.frames[idx];
      const padIdx = String(idx).padStart(4, '0');

      // Resolve image path: check embedded base64 first, fallback to relative URL
      function resolveImg(relPath) {
        if (typeof EMBEDDED_IMAGES !== 'undefined' && EMBEDDED_IMAGES && EMBEDDED_IMAGES[relPath]) {
          return EMBEDDED_IMAGES[relPath];
        }
        return relPath;
      }

      // Update images
      document.getElementById('img-rgb').src = resolveImg(`frames/frame_${padIdx}.png`);
      document.getElementById('img-depth').src = resolveImg(`visualizations/depth_${padIdx}.png`);
      document.getElementById('img-normal').src = resolveImg(`visualizations/normal_${padIdx}.png`);
      document.getElementById('img-flow').src = resolveImg(`visualizations/flow_${padIdx}.png`);

      const contactEl = document.getElementById('link-contact');
      if (contactEl && typeof EMBEDDED_IMAGES !== 'undefined' && EMBEDDED_IMAGES['contact_sheet.png']) {
        contactEl.href = EMBEDDED_IMAGES['contact_sheet.png'];
      }

      // Update metadata
      const isVal = f.split === 'val';
      const badgeSplit = document.getElementById('badge-active-split');
      const propSplit = document.getElementById('prop-split');

      if (isVal) {
        badgeSplit.className = 'badge badge-val';
        badgeSplit.textContent = 'VAL (Held-out)';
        propSplit.innerHTML = '<span class="badge badge-val">VALIDATION</span>';
      } else {
        badgeSplit.className = 'badge badge-train';
        badgeSplit.textContent = 'TRAIN';
        propSplit.innerHTML = '<span class="badge badge-train">TRAIN</span>';
      }

      document.getElementById('lbl-time-cur').textContent = `Frame ${idx} / ${manifest.frames.length - 1}`;
      document.getElementById('prop-frame').textContent = `${idx} / ${manifest.frames.length - 1}`;

      const pos = f.camera_position || [0, 0, 0];
      document.getElementById('prop-pos-x').textContent = pos[0].toFixed(3);
      document.getElementById('prop-pos-y').textContent = pos[1].toFixed(3);
      document.getElementById('prop-pos-z').textContent = pos[2].toFixed(3);
      const dist = Math.sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
      document.getElementById('prop-dist').textContent = `${dist.toFixed(3)} m`;
      document.getElementById('prop-fov').textContent = `${f.camera?.fov_degrees || 50.0}°`;

      drawTrajectory();
    }

    // 2D Projection of 3D Orbit Trajectory
    function drawTrajectory() {
      if (!manifest || !manifest.frames) return;
      const w = canvasTraj.width;
      const h = canvasTraj.height;
      ctxTraj.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const scale = Math.min(w, h) * 0.38;

      // Draw model center origin & coordinate axes
      ctxTraj.strokeStyle = '#30363d';
      ctxTraj.lineWidth = 1;
      ctxTraj.beginPath();
      ctxTraj.arc(cx, cy, 8, 0, 2 * Math.PI);
      ctxTraj.fillStyle = '#21262d';
      ctxTraj.fill();
      ctxTraj.stroke();

      // Draw trajectory line
      ctxTraj.beginPath();
      manifest.frames.forEach((f, i) => {
        const p = f.camera_position;
        // Project X-Z plane with slight isometric elevation tilt (Y)
        const px = cx + (p[0] * scale * 0.4);
        const py = cy + (p[2] * scale * 0.4) - (p[1] * scale * 0.15);
        if (i === 0) ctxTraj.moveTo(px, py);
        else ctxTraj.lineTo(px, py);
      });
      ctxTraj.strokeStyle = 'rgba(56, 139, 253, 0.35)';
      ctxTraj.lineWidth = 2;
      ctxTraj.stroke();

      // Draw camera points
      manifest.frames.forEach((f, i) => {
        const p = f.camera_position;
        const px = cx + (p[0] * scale * 0.4);
        const py = cy + (p[2] * scale * 0.4) - (p[1] * scale * 0.15);

        ctxTraj.beginPath();
        if (i === currentFrame) {
          // Highlight active camera
          ctxTraj.arc(px, py, 6, 0, 2 * Math.PI);
          ctxTraj.fillStyle = '#f85149';
          ctxTraj.fill();
          ctxTraj.strokeStyle = '#ffffff';
          ctxTraj.lineWidth = 2;
          ctxTraj.stroke();
        } else if (f.split === 'val') {
          ctxTraj.arc(px, py, 3, 0, 2 * Math.PI);
          ctxTraj.fillStyle = '#e3b341';
          ctxTraj.fill();
        } else {
          ctxTraj.arc(px, py, 2.5, 0, 2 * Math.PI);
          ctxTraj.fillStyle = '#3fb950';
          ctxTraj.fill();
        }
      });
    }

    // Canvas click to select frame
    canvasTraj.addEventListener('click', (e) => {
      if (!manifest) return;
      const rect = canvasTraj.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      const w = canvasTraj.width;
      const h = canvasTraj.height;
      const cx = w / 2;
      const cy = h / 2;
      const scale = Math.min(w, h) * 0.38;

      let closestIdx = 0;
      let minD = 1e9;

      manifest.frames.forEach((f, i) => {
        const p = f.camera_position;
        const px = cx + (p[0] * scale * 0.4);
        const py = cy + (p[2] * scale * 0.4) - (p[1] * scale * 0.15);
        const dist = Math.hypot(clickX - px, clickY - py);
        if (dist < minD) {
          minD = dist;
          closestIdx = i;
        }
      });

      renderFrame(closestIdx);
    });

    // Timeline control events
    elTimeline.addEventListener('input', (e) => renderFrame(parseInt(e.target.value)));
    elBtnPrev.addEventListener('click', () => {
      if (!manifest) return;
      const prev = (currentFrame - 1 + manifest.frames.length) % manifest.frames.length;
      renderFrame(prev);
    });
    elBtnNext.addEventListener('click', () => {
      if (!manifest) return;
      const next = (currentFrame + 1) % manifest.frames.length;
      renderFrame(next);
    });

    elBtnPlay.addEventListener('click', () => {
      isPlaying = !isPlaying;
      if (isPlaying) {
        elBtnPlay.textContent = "⏸ Pause";
        elBtnPlay.classList.add("active");
        playInterval = setInterval(() => {
          const next = (currentFrame + 1) % manifest.frames.length;
          renderFrame(next);
        }, 1000 / playbackFps);
      } else {
        elBtnPlay.textContent = "▶ Play";
        elBtnPlay.classList.remove("active");
        clearInterval(playInterval);
      }
    });

    elSelFps.addEventListener('change', (e) => {
      playbackFps = parseInt(e.target.value);
      if (isPlaying) {
        clearInterval(playInterval);
        playInterval = setInterval(() => {
          const next = (currentFrame + 1) % manifest.frames.length;
          renderFrame(next);
        }, 1000 / playbackFps);
      }
    });

    // Keyboard navigation (Left/Right arrows, Space to play/pause)
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        elBtnPlay.click();
      } else if (e.code === 'ArrowLeft') {
        elBtnPrev.click();
      } else if (e.code === 'ArrowRight') {
        elBtnNext.click();
      }
    });

    window.addEventListener('DOMContentLoaded', init);
  </script>
</body>
</html>
"""
