#!/usr/bin/env python3
"""Compare one closed native case to the canonical8192 frame, without deletion."""
import fcntl, hashlib, importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent; P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def pin(p):
    p = Path(p)
    with p.open('rb') as f: h = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h)
def main():
    assert len(sys.argv) == 2; name = sys.argv[1]
    assert name in ['firstB', 'firstC'] + [f'clean{x}{r}' for x in 'ABC' for r in range(3)]
    root = B / f'main-qsa-large-resource-compare-{name}-v1'
    dst = A / f'own-qsa-large-resource-compare-{name}-v1.json'
    assert not root.exists() and not dst.exists()
    ownerpath = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert pin(ownerpath)['sha256'] == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('qsa_large_comparison_owner', ownerpath)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    native_paths = [A / f'own-qsa-large-resource-{case}-v1.json' for case in ['firstA', name]]
    native = [json.loads(p.read_text()) for p in native_paths]
    for j in native:
        assert j['passed'] and j['complete'] and not j['active'] and j['ownership_closed'] and not j['fresh_faults']
        for c in j['commands']:
            assert m.closed(c)
            for old in c['owners'] + [c['direct_identity']]:
                now = m.identity(old['pid']); assert now is None or now['start_ticks'] != old['start_ticks']
    frames = [j['cases'][0]['frame'] for j in native]
    for v in frames:
        assert Path(v['path']).is_file() and not Path(v['path']).is_symlink()
        assert Path(v['path']).stat().st_size == v['bytes'] == 201326616
    helper = A / 'own-qsa-large-resource-control-v1/compare_frames.py'
    assert pin(helper)['sha256'] == 'e057abe02fd802544bc0f85791052ac59c116a073e71f692c14d332fffcbb7a1'
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); root.mkdir(mode=0o700)
        rec = dict(schema='strata-qsa-large-resource-comparison-v1', case_id=name, passed=False,
            complete=False, active=False, commands=[], GPU_executed=False, native_receipts=[pin(p) for p in native_paths],
            expected_frames=frames, helper=pin(helper), source=pin(__file__),
            scope='Both complete201326616byteinitialframes, everyFP32finite and everypayloadbyte. Native samples/lifetimes/fault results remain separate.')
        owner = None
        def persist():
            rec['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(rec, indent=2) + '\n')
        owner = m.Owner(root / 'owned', persist); rec['commands'] = owner.commands
        env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
        try:
            c, out, err = owner.run('compare-full-frames', ['/usr/bin/python3', str(helper), *[v['path'] for v in frames]],
                env, P, wall=90, cpu=120, rss_cap=512 << 20, text_cap=16 << 20, file_cap=16 << 20, total_cap=32 << 20)
            result = json.loads(out.read_text()); rec.update(result=result, stderr=err.read_text())
            assert m.closed(c) and c['exit_code'] == 0 and result['status'] == 'PASS'
            for observed, expected in zip(result['frames'], frames):
                assert observed['finite'] and observed['bytes'] == expected['bytes'] and observed['sha256'] == expected['sha256']
            assert result['frames'][1]['exact_reference']
            assert pin(helper) == rec['helper']
            rec.update(passed=True, complete=True)
        except BaseException as e:
            import traceback
            rec.update(error=repr(e), traceback=traceback.format_exc())
        finally:
            rec['ownership_closed'] = owner.active is None and all(m.closed(c) for c in owner.commands)
            persist(); dst.write_bytes((root / 'record.json').read_bytes())
        print(json.dumps(dict(receipt=str(dst), passed=rec['passed'], complete=rec['complete'], ownership_closed=rec['ownership_closed'], error=rec.get('error'))))
        return 0 if rec['passed'] and rec['ownership_closed'] else 1
if __name__ == '__main__': sys.exit(main())
