import subprocess
import time

script = """
echo "=== Mounts in cache ==="
mount | grep -i cache || true
echo "=== Running bazel/python processes ==="
ps aux | grep -E "bazel|pytest|synapse|fulgur" | grep -v grep || true
echo "=== Killing stale processes ==="
pkill -9 -f bazel || true
pkill -9 -f pytest || true
sleep 1
echo "=== Attempting force cleanup ==="
chmod -R u+w /root/.cache/bazel* 2>/dev/null || true
rm -rf /root/.cache/bazel 2>&1 || true
rm -rf /root/.cache/bazel-fulgur 2>&1 || true
rm -rf /root/.cache/bazel_repo_cache 2>&1 || true
rm -rf /root/.cache/bazel-aether 2>&1 || true
rm -rf /root/.cache/bazel-distdir 2>&1 || true
echo "=== Checking /root/.cache contents ==="
ls -la /root/.cache/
echo "=== Disk status ==="
df -h /
echo "=== Restarting logind ==="
systemctl restart systemd-logind 2>/dev/null || true
echo "STATUS_OK"
"""

print("[*] Running force cleanup...", flush=True)
res = subprocess.run(
    ["ssh", "-o", "IdentityAgent=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "ubuntu24", script],
    capture_output=True,
    text=True,
    timeout=300
)
print("=== STDOUT ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
