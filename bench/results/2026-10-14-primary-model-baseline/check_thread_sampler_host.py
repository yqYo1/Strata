"""Host-only lifetime/sampling check; no inference or GPU work."""
import fcntl
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from owned_thread_sampler import Sampler

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent


def identity(pid):
    f = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(f[19]))


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix='strata-passive-sampler-host-') as tmp:
            root = Path(tmp); (root / 'owned').mkdir()
            child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.readline()'], stdin=subprocess.PIPE)
            sampler = None
            try:
                native = identity(child.pid)
                sampler = Sampler(native, root)
                sampler.start()
                time.sleep(1.2)
                child.stdin.write(b'finish\n'); child.stdin.flush()
                assert child.wait(timeout=5) == 0
                sampler.close()
                summary = json.loads((root / 'sampler-summary.json').read_text())
                assert summary['passed'] and summary['samples'] >= 1 and summary['bytes'] <= summary['byte_cap']
                assert (root / 'owned/native-maps.txt').stat().st_size <= 96 << 10
                samples = [json.loads(s) for s in (root / 'owned/thread-samples.jsonl').read_text().splitlines()]
                assert samples and all(x['most_active'] for x in samples)
                result = dict(passed=True, scope='host-only pinned-child passive sampler lifecycle; no JIT/GPU/state proof',
                              samples=len(samples), stub_exit=0, summary=summary)
                (D / 'thread-sampler-host-check.json').write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps(result))
            finally:
                if child.poll() is None:
                    child.kill(); child.wait()
                if sampler:
                    sampler.close()


if __name__ == '__main__':
    main()
