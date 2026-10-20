#!/usr/bin/env python3
"""Reject ambiguous request boundaries, missing restore clocks and unknown faults."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import prefill_resources as r

WARM = 'strata serve: prompt 74 tokens = 0 reused + 74 read, 24 generated\n'
LONG = 'strata serve: prompt 117489 tokens = 0 reused + 117489 read, 24 generated\n'
DENSE = ('strata native dense host profile: v=2 complete=1 outcome=success entry_ms=12 run_ms=123000 '
         'restore_ms=40 entry_done=1 run_returned=1 restored=1 intervals=NONEXCLUSIVE '
         'entry_scope=boundary_entry_through_Bank run_scope=MTP_release_prompt_callbacks_MTP_restore '
         'restore_scope=restore_drain_through_owner_PLE_graph_verifier_restore outer_startup_snapshot_unmeasured=1\n')
REFILL = 'strata trace: refilled 1124 slots on 1 stage(s) in 380.5 ms\n'


class Boundaries(unittest.TestCase):
    def test_disabled_legacy_census_is_not_an_enabled_observer(self):
        r.require_observer_environment({})
        r.require_observer_environment(dict(STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS='0'))
        for env in (dict(STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS='1'), dict(LD_PRELOAD='anything')):
            with self.subTest(env=env), self.assertRaises(RuntimeError): r.require_observer_environment(env)

    def test_warmup_incomplete_frame_does_not_qualify_or_invalidate_long_frame(self):
        warm_frame = DENSE.replace('complete=1', 'complete=0').replace('entry_done=1', 'entry_done=0')
        part = r.long_interval(warm_frame + WARM + DENSE + REFILL + LONG, dict(generated_tokens=24))
        result = r.restore_intervals(part, 1124)
        self.assertEqual(result['expert_cache_refill_wall_ms'], 380.5)
        self.assertEqual(result['native_dense']['restore_ms'], 40.)
        self.assertTrue(result['expert_refill_outside_native_dense'])
        with self.assertRaises(RuntimeError): r.restore_intervals(warm_frame + REFILL, 1124)

    def test_reject_partial_extra_reused_or_mismatched_requests(self):
        for log in (LONG, WARM + LONG + LONG, WARM + LONG.replace('0 reused', '1 reused'),
                    WARM + LONG.replace('24 generated', '23 generated'), WARM + LONG.replace('117489 read', '117488 read')):
            with self.subTest(log=log), self.assertRaises(RuntimeError):
                r.long_interval(log, dict(generated_tokens=24))

    def test_reject_missing_duplicate_unfinished_or_reordered_restoration(self):
        cases = [DENSE, REFILL, DENSE + DENSE + REFILL, DENSE + REFILL + REFILL, REFILL + DENSE,
            DENSE.replace('restored=1', 'restored=0') + REFILL,
            DENSE.replace('run_ms=123000', 'run_ms=nan') + REFILL,
            DENSE.replace('entry_ms=12', 'entry_ms=-1') + REFILL,
            DENSE + REFILL.replace('1124 slots', '1123 slots'),
            DENSE + REFILL.replace('1 stage', '2 stage'),
            DENSE.replace('complete=1', 'complete=1 complete=1') + REFILL]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(RuntimeError): r.restore_intervals(text, 1124)


class Terminal(unittest.TestCase):
    def finish(self, journal, *, persist_error=False, boot='same', closed=True):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            record = dict(passed=True)
            owner = SimpleNamespace(active=None if closed else object(), commands=[])
            def persist():
                if persist_error: raise OSError('write failed')
            with patch.object(r, 'RESULT', path), patch.object(r.subprocess, 'check_output', side_effect=journal), \
                patch.object(r.c, 'read_boot', return_value=boot), patch.object(r.guards, 'driver_guard', return_value={}):
                if persist_error:
                    with self.assertRaises(OSError): r.finish(record, owner, 'cursor', 'same', persist)
                else:
                    r.finish(record, owner, 'cursor', 'same', persist)
            self.assertEqual(record, json.loads(path.read_text()))
            return record

    def test_journal_failure_is_unknown_and_terminal_failed(self):
        record = self.finish(subprocess.CalledProcessError(1, ['journalctl']))
        self.assertFalse(record['passed'])
        self.assertIsNone(record['fresh_faults'])
        self.assertFalse(record['fault_probe_successful'])

    def test_fresh_fault_reboot_open_owner_or_persist_error_cannot_pass(self):
        for args in (dict(journal=lambda *a, **kw: 'xe GPU fault'),
                     dict(journal=lambda *a, **kw: '', boot='other'),
                     dict(journal=lambda *a, **kw: '', closed=False),
                     dict(journal=lambda *a, **kw: '', persist_error=True)):
            with self.subTest(args=args): self.assertFalse(self.finish(**args)['passed'])
        self.assertTrue(self.finish(lambda *a, **kw: '')['passed'])


if __name__ == '__main__': unittest.main()
