#!/usr/bin/env python3
"""Main-authored ownership/race controls, not an inference test. Repository LICENSE."""
import ast
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'tools/sycl'))
import measurement_owner as baseline
import owner_candidate as candidate

ROOT = Path(os.environ['STRATA_OWNER_TEST_ROOT'])
FIXTURE = P / 'bench/sycl/owner_fixture.py'
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')


def value(pid, birth, sid=100, state='S', rss=4096):
    return dict(pid=pid, start_ticks=birth, sid=sid, pgid=sid, ppid=100,
        state=state, rss_bytes=rss)


def registry(read, population=('100', '200'), pinned=None):
    # Compile the ACTUAL nested production functions, retaining their original
    # source locations. Mock only OS observations to exercise deterministic races.
    tree = ast.parse(Path(candidate.__file__).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Owner')
    run = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    functions = [n for n in run.body if isinstance(n, ast.FunctionDef) and n.name in ('anchor', 'register', 'scan')]
    leader = value(100, 10); closed = []; owned = {(100, 10): (1, leader)}
    if pinned: owned.update(pinned)
    scope = dict(vars(candidate), leader=leader, owned=owned,
        entry=dict(owners=[], peak_live_owned_identities=0), identity=read,
        os=SimpleNamespace(pidfd_open=lambda pid: pid + 10000, close=closed.append,
            listdir=lambda path: population, getuid=lambda: 1000,
            stat=lambda path: SimpleNamespace(st_uid=0)))
    exec(compile(ast.Module(body=functions, type_ignores=[]), candidate.__file__, 'exec'), scope)
    return scope, closed


class Races(unittest.TestCase):
    def test_text_parser_contract(self):
        tail = ['S', '101', '102', '103'] + ['0'] * 15 + ['123456', '0', '321']
        text = '200 (name with ) parentheses\nand newline) ' + ' '.join(tail)
        with patch.object(Path, 'read_text', return_value=text), patch('builtins.open', return_value=io.StringIO(text)):
            self.assertEqual(baseline.identity(200), candidate.identity(200))

    def test_missing_identity(self):
        with patch('builtins.open', side_effect=FileNotFoundError): self.assertIsNone(candidate.identity(200))

    def test_reused_disappeared_or_zombie_not_accounted(self):
        for ending in (None, value(200, 21), value(200, 20, state='Z')):
            reads = iter((value(200, 20), ending))
            s, closed = registry(lambda pid: value(100, 10) if pid == 100 else next(reads))
            self.assertEqual([v['pid'] for v in s['scan']()], [100])
            self.assertEqual(set(s['owned']), {(100, 10)})
            self.assertEqual(closed, [10200])

    def test_fresh_registration_accounting(self):
        reads = iter((value(200, 20, rss=111), value(200, 20, rss=999), value(200, 20, rss=999)))
        s, _ = registry(lambda pid: value(100, 10) if pid == 100 else next(reads))
        self.assertEqual([v['rss_bytes'] for v in s['scan']() if v['pid'] == 200], [999])

    def test_known_escape_and_foreign_exclusion(self):
        rows = {100: value(100, 10), 200: value(200, 20, sid=200), 300: value(300, 30, sid=300)}
        s, _ = registry(rows.get, ('100', '200', '300'), {(200, 20): (2, rows[200])})
        self.assertEqual({v['pid'] for v in s['scan']()}, {100, 200})

    def test_live_population_cap(self):
        rows = {100: value(100, 10), **{i: value(i, i) for i in range(1000, 1255)}, 200: value(200, 20)}
        s, closed = registry(rows.get, ('200',), {(i, i): (i, rows[i]) for i in range(1000, 1255)})
        with self.assertRaisesRegex(RuntimeError, 'population cap'): s['scan']()
        self.assertEqual(len(s['owned']), 256); self.assertIn(10200, closed)


class Processes(unittest.TestCase):
    def run_owned(self, name, mode, interaction=None):
        root = ROOT / name; root.mkdir()
        owner = None
        def persist():
            (root / 'execution.json').write_text(json.dumps(dict(active=bool(owner and owner.active),
                commands=owner.commands if owner else []), indent=2) + '\n')
        owner = candidate.Owner(root / 'owned', persist)
        entry, _, _ = owner.run(name, [sys.executable, '-I', '-S', '-B', str(FIXTURE), mode],
            ENV, P, interaction=interaction, wall=8, cpu=3, rss_cap=128 << 20,
            text_cap=1 << 20, file_cap=1 << 20, total_cap=4 << 20)
        self.assertIsNone(owner.active)
        self.assertTrue(entry['direct_child_reaped'] and entry['session_empty'])
        self.assertFalse(entry['survivors'])
        return entry

    def test_fast_exit_leader(self):
        self.assertTrue(candidate.closed(self.run_owned('normal', 'normal')))

    def interaction(self, cleanup):
        def interact(io, entry):
            pid = int(io.line(3).removeprefix('child=')); deadline = time.monotonic() + 3
            while not any(v['pid'] == pid for v in entry['owners']):
                io.poll(); self.assertLess(time.monotonic(), deadline); time.sleep(.02)
            io.send(b'go\n')
            expected = ('held=' if cleanup else 'escaped=') + str(pid)
            self.assertEqual(io.line(3), expected)
            if cleanup: raise RuntimeError('injected owned cleanup')
            self.assertNotEqual(candidate.identity(pid)['sid'], entry['pid'])
            deadline = time.monotonic() + .7
            while time.monotonic() < deadline: io.poll(); time.sleep(.02)
            io.send(b'stop\n'); self.assertEqual(io.line(3), 'done')
            io.proc.stdin.close(); io.exit(3)
        return interact

    def test_registered_escape_RSS(self):
        entry = self.run_owned('escape', 'family', self.interaction(False))
        self.assertTrue(candidate.closed(entry))
        self.assertGreater(entry['peak_session_rss_bytes'], 40 << 20)

    def test_owned_cleanup(self):
        entry = self.run_owned('cleanup', 'cleanup', self.interaction(True))
        self.assertIn('injected owned cleanup', entry['error'])
        self.assertFalse(candidate.closed(entry)); self.assertTrue(entry['cleanup'])
        known = {(v['pid'], v['start_ticks']) for v in entry['owners']}
        self.assertTrue(all((v['pid'], v['start_ticks']) in known for v in entry['cleanup']))


if __name__ == '__main__': unittest.main(failfast=True)
