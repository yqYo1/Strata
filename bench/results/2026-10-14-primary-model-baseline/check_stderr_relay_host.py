"""Host-only pipe saturation/retention check; no model or GPU execution."""
import fcntl
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix='strata-relay-host-') as tmp:
            root = Path(tmp); (root / 'owned').mkdir()
            source = '''from bounded_stderr_relay import Relay
import subprocess,sys
from pathlib import Path
root=Path(sys.argv[1])
program="import sys;sys.stderr.buffer.write(b'x'*1023+b'\\\\n');sys.stderr.buffer.write((b'---> urTest\\\\n')*10000);sys.stderr.buffer.write(b'UR_RESULT_ERROR_TEST\\\\nstrata host diagnostic: __stack_chk_fail pid=1 tid=1 caller=0x1\\\\n');sys.stdout.buffer.write(b'protocol preserved\\\\n')"
child=subprocess.Popen([sys.executable,'-c',program],stderr=subprocess.PIPE)
relay=Relay(child.stderr,root,prefix_cap=256,selected_cap=1024);relay.start()
assert child.wait(timeout=10)==0
relay.close()
'''
            script = root / 'test.py'; script.write_text(source)
            env = dict(__import__('os').environ, PYTHONPATH=str(D.resolve()))
            with (root / 'retained.stderr').open('wb') as err:
                done = subprocess.run([sys.executable, str(script), str(root)], env=env,
                                      stdout=subprocess.PIPE, stderr=err, timeout=15)
            assert done.returncode == 0 and done.stdout == b'protocol preserved\n', done
            result = json.loads((root / 'stderr-relay-summary.json').read_text())
            assert result['passed'] and not result['oversized_lines']
            assert result['API_counts']['urTest'] == 10000 and result['error_counts']['UR_RESULT_ERROR_TEST'] == 1
            assert result['total_bytes'] > 100000 and result['prefix_bytes'] <= 256 and result['selected_bytes'] <= 1024
            assert b'__stack_chk_fail' in (root / 'retained.stderr').read_bytes()
            result.update(scope='host-only full pipe drain, stdout preservation, late-error retention and byte bounds',
                          recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            (D / 'stderr-relay-host-check.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result))


if __name__ == '__main__':
    main()
