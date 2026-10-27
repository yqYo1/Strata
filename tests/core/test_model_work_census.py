"""Host-only regression against the real benchmark gate; repository LICENSE."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import measurement_owner as owner
import model_work_census as census


class BenchmarkGateContract(unittest.TestCase):
    def child(self, root, stages, interact):
        # Load the actual benchmark function, without invoking its CLI or model.
        script = ('import runpy,sys;sys.path.insert(0,' + repr(str(P/'tools')) + ');'
            'f=runpy.run_path(' + repr(str(P/'tools/strata_bench.py')) +
            ',run_name="_host_gate_contract")["profile_request_gate"];'
            + ';'.join('f(' + repr(stage) + ',"prefill-only",0)' for stage in stages))
        own = owner.Owner(root)
        entry, _, err = own.run('gate', [sys.executable, '-I', '-S', '-B', '-c', script],
            dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8'), P,
            interaction=interact, wall=10, cpu=10, rss_cap=256 << 20,
            text_cap=64 << 10, file_cap=64 << 10, total_cap=1 << 20)
        self.assertTrue(owner.closed(entry), entry)
        return entry, err.read_text()

    def test_controller_ack_advances_both_real_gates(self):
        seen = []
        def interact(io, leader):
            for stage in ('begin', 'end'):
                line = io.line(4)
                self.assertTrue(line.startswith('STRATA_PROFILE_GATE '))
                self.assertEqual(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')),
                    dict(stage=stage, phase='prefill-only', run=0))
                seen.append(stage)
                census.continue_benchmark(io)
            io.proc.stdin.close()
            io.exit(4)
        with tempfile.TemporaryDirectory() as tmp:
            entry, err = self.child(Path(tmp), ('begin', 'end'), interact)
        self.assertEqual(entry['exit_code'], 0)
        self.assertEqual(err, '')
        self.assertEqual(seen, ['begin', 'end'])

    def test_real_gate_rejects_wrong_ack(self):
        def interact(io, leader):
            self.assertTrue(io.line(4).startswith('STRATA_PROFILE_GATE '))
            io.send(b'X\n')
            io.proc.stdin.close()
            io.exit(4)
        with tempfile.TemporaryDirectory() as tmp:
            entry, err = self.child(Path(tmp), ('begin',), interact)
        self.assertEqual(entry['exit_code'], 1)
        self.assertIn('RuntimeError: profile request gate command', err)


if __name__ == '__main__':
    unittest.main()
