"""Run original-style fatal launch checks in separate, bounded processes."""
import json
import subprocess
import sys

cases = {
    "--invalid-stream": "prefill mmq: iota: invalid resource handle",
    "--sticky-error": "prefill mmq: iota: invalid resource handle",
    "--cold-capture": "MMQ scratch must be warmed before graph capture",
}
for mode, diagnostic in cases.items():
    result = subprocess.run([sys.argv[1], mode], capture_output=True, text=True, timeout=20)
    print(json.dumps({"mode": mode, "exit_code": result.returncode,
                      "stdout": result.stdout, "stderr": result.stderr}))
    if result.returncode != 1 or diagnostic not in result.stderr or "terminate" in result.stderr:
        raise SystemExit("fatal launch contract failed: " + mode)
print("PASS fatal_launch_checks=3 exit_code=1 teardown_does_not_mask_capture_error=1")
