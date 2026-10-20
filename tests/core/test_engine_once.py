#!/usr/bin/env python3
"""Owned synthetic child lifecycle; no model, native compilation or GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

WRAPPER=Path(__file__).resolve().parents[2]/'tools/sycl/engine_once.py'


class EngineOnceTests(unittest.TestCase):
    def test_boot_binding_collector_failure_and_no_second_native_launch(self):
        for collector_code in (0,2):
            with self.subTest(collector_code=collector_code),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                native=root/'native'
                native.write_text('#!/usr/bin/python3\nfrom pathlib import Path\nimport os,time\n'
                    'p=Path(os.environ["STRATA_MEASUREMENT_ROOT"])/"native-launches"\n'
                    'with p.open("a") as f:f.write("launched\\n")\ntime.sleep(.1)\n')
                native.chmod(0o700)
                reader=root/'reader.py'
                reader.write_text('from pathlib import Path\nimport json,sys\n'
                    'x=json.loads(Path(sys.argv[1]).read_text())\n'
                    'assert x["boot_id"]==Path("/proc/sys/kernel/random/boot_id").read_text().strip()\n'
                    f'raise SystemExit({collector_code})\n')
                def artifact(path):
                    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                (root/'plan.json').write_text(json.dumps(dict(binary=artifact(native),
                    resource_collection=dict(source=artifact(reader),seconds=1,interval=.2))))
                env=dict(os.environ,STRATA_MEASUREMENT_ROOT=str(root),PYTHONDONTWRITEBYTECODE='1')
                first=subprocess.run([sys.executable,'-B',str(WRAPPER)],env=env,capture_output=True,timeout=5)
                self.assertEqual(first.returncode,collector_code,first.stderr)
                end=json.loads((root/'engine-exit.json').read_text())
                self.assertEqual(end['exit_code'],0)
                self.assertEqual(end['resource_collector_exit_code'],collector_code)
                self.assertEqual(end['resource_collector']['sid'],end['native']['sid'])
                again=subprocess.run([sys.executable,'-B',str(WRAPPER)],env=env,capture_output=True,timeout=5)
                self.assertNotEqual(again.returncode,0)
                self.assertEqual((root/'native-launches').read_text(),'launched\n')


if __name__=='__main__':unittest.main()
