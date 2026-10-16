#!/usr/bin/env python3
"""Retire one consumed candidate only after native/comparison evidence is in Git."""
import fcntl, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent; P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def main():
    assert len(sys.argv) == 2; name = sys.argv[1]
    assert name in ['firstB', 'firstC'] + [f'clean{x}{r}' for x in 'ABC' for r in range(3)]
    dst = A / f'qsa-large-resource-frame-retirement-{name}-v1.json'; assert not dst.exists()
    ownerpath = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert sha(ownerpath) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('qsa_large_frame_retirement_owner', ownerpath)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    paths = [A / f'own-qsa-large-resource-{name}-v1.json', A / f'own-qsa-large-resource-compare-{name}-v1.json']
    values = [json.loads(p.read_text()) for p in paths]
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for p, j in zip(paths, values):
            assert subprocess.check_output(['git', 'show', 'HEAD:' + str(p.relative_to(P))], cwd=P) == p.read_bytes()
            assert j['passed'] and j['complete'] and not j['active'] and j['ownership_closed']
            for c in j['commands']:
                assert m.closed(c)
                for old in c['owners'] + [c['direct_identity']]:
                    now = m.identity(old['pid']); assert now is None or now['start_ticks'] != old['start_ticks']
        frame = values[0]['cases'][0]['frame']; p = Path(frame['path'])
        assert p == B / f'main-qsa-large-resource-{name}-v1/qsa-0.bin'
        assert not p.is_symlink() and p.stat().st_size == frame['bytes'] and sha(p) == frame['sha256']
        rec = dict(schema='strata-qsa-large-resource-candidate-retirement-v1', passed=False,
            complete=False, GPU_executed=False, source_sha256=sha(__file__), original_statuses_unchanged=True,
            path=str(p), bytes=frame['bytes'], sha256=frame['sha256'], removed=False,
            committed_evidence=[dict(path=str(p), sha256=sha(p)) for p in paths],
            reason='Closed successful candidate fully finite/byte-compared to canonical; all samples/hash/exit/fault evidence committed. No mismatch or RESTORE consumer. Keep only canonical for next independent process.')
        dst.write_text(json.dumps(rec, indent=2) + '\n'); p.unlink()
        rec.update(removed=True, passed=True, complete=True); dst.write_text(json.dumps(rec, indent=2) + '\n')
        print(json.dumps(dict(manifest=str(dst), bytes=frame['bytes'], removed=True)))
if __name__ == '__main__': main()
