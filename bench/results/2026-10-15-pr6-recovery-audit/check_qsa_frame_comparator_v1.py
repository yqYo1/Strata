#!/usr/bin/env python3
"""Qualify full-frame comparison with independent finite/corrupt/large fixtures."""
import fcntl, hashlib, importlib.util, json, os, struct, sys
from pathlib import Path
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')

def pin(p):
    p = Path(p)
    with p.open('rb') as f: h = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h)

def main():
    root = B / 'main-qsa-frame-comparator-check-v1'
    dst = A / 'own-qsa-frame-comparator-check-v1.json'
    assert not root.exists() and not dst.exists()
    helper = A / 'own-qsa-large-resource-control-v1/compare_frames.py'
    assert pin(helper)['sha256'] == 'e057abe02fd802544bc0f85791052ac59c116a073e71f692c14d332fffcbb7a1'
    owner_path = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert pin(owner_path)['sha256'] == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('qsa_comparator_check_owner', owner_path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); root.mkdir(mode=0o700)
        fixtures = root / 'fixtures'; fixtures.mkdir()
        count = 128 * 24 * 256
        header = struct.pack('<8sIIQ', b'QRCV1LE\0', 128, 2051, count)
        good = header + struct.pack('<4I', 0, 0x80000000, 0x3f800000, 0xbf800000) * (count // 4)
        files = {}
        def put(name, payload):
            path = fixtures / name; path.write_bytes(payload); files[name] = path; return path
        put('good-a.bin', good); put('good-b.bin', good)
        bad = bytearray(good); bad[27] ^= 0x80; put('different.bin', bad)
        bad = bytearray(good); bad[-4:] = struct.pack('<I', 0x7fc00000)
        put('nonfinite-a.bin', bad); put('nonfinite-b.bin', bad)
        put('truncated.bin', good[:-4]); put('appended.bin', good + b'\0' * 4)
        os.link(files['good-a.bin'], fixtures / 'hardlink.bin')
        os.symlink(files['good-a.bin'], fixtures / 'symlink.bin')
        files['hardlink.bin'] = fixtures / 'hardlink.bin'
        files['symlink.bin'] = fixtures / 'symlink.bin'
        for name in ['large-a.bin', 'large-b.bin']:
            path = fixtures / name
            with path.open('xb') as f:
                f.write(struct.pack('<8sIIQ', b'QRCV1LE\0', 8192, 2051, 8192 * 24 * 256))
                f.truncate(24 + 8192 * 24 * 256 * 4)
            files[name] = path
        pins = {name: pin(path) for name, path in files.items() if name not in ('symlink.bin', 'hardlink.bin')}
        cases = [
            ('equal', 'good-a.bin', 'good-b.bin', True, None),
            ('finite-one-byte-difference', 'good-a.bin', 'different.bin', False, None),
            ('equal-nonfinite-rejected', 'nonfinite-a.bin', 'nonfinite-b.bin', False, 'nonfinite'),
            ('truncated-rejected', 'good-a.bin', 'truncated.bin', False, 'extent/header'),
            ('appended-rejected', 'good-a.bin', 'appended.bin', False, 'extent/header'),
            ('same-inode-rejected', 'good-a.bin', 'hardlink.bin', False, 'distinct files'),
            ('symlink-rejected', 'good-a.bin', 'symlink.bin', False, 'symbolic links'),
            ('mixed-shape-rejected', 'good-a.bin', 'large-a.bin', False, 'same nq'),
            ('large-full-finite-equal', 'large-a.bin', 'large-b.bin', True, None)]
        rec = dict(schema='strata-qsa-frame-comparator-check-v1', passed=False, complete=False,
            active=False, commands=[], GPU_executed=False, helper=pin(helper), fixture_pins=pins,
            cases=[], scope='Host-only independent synthetic IEEE-FP32 frames, including full201326616byte8192 frames; not GPU/model output or a performance measurement.',
            retention=dict(owner='Main', consumer='Full streaming comparator qualification',
                next_review='After this compact evidence is committed, retire all synthetic raw fixtures and successful output duplicates.'))
        owner = None
        def persist():
            rec['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(rec, indent=2) + '\n')
        owner = m.Owner(root / 'owned', persist); rec['commands'] = owner.commands
        env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
        try:
            for label, left, right, passed, error in cases:
                command, out, err = owner.run(label, ['/usr/bin/python3', str(helper), str(files[left]), str(files[right])], env, P,
                    wall=90, cpu=120, rss_cap=512 << 20, text_cap=16 << 20, file_cap=16 << 20, total_cap=512 << 20)
                assert m.closed(command)
                result = json.loads(out.read_text())
                assert command['exit_code'] == (0 if passed else 1) and result['status'] == ('PASS' if passed else 'FAIL')
                if error: assert error in result['error'], (label, result)
                if label == 'finite-one-byte-difference':
                    assert result['frames'][1]['first_mismatch_byte'] == 27 and not result['frames'][1]['exact_reference']
                if passed:
                    assert all(v['finite'] and v['bytes'] == pins[left]['bytes'] for v in result['frames'])
                    assert result['frames'][0]['sha256'] == result['frames'][1]['sha256'] == pins[left]['sha256']
                rec['cases'].append(dict(name=label, expected_pass=passed, exit_code=command['exit_code'], result=result, stderr=err.read_text()))
            assert pin(helper) == rec['helper']
            for name, value in pins.items(): assert pin(files[name]) == value
            rec.update(passed=True, complete=True)
        except BaseException as error:
            import traceback
            rec.update(error=repr(error), traceback=traceback.format_exc())
        finally:
            rec['ownership_closed'] = owner.active is None and all(m.closed(c) for c in owner.commands)
            for command in owner.commands:
                for old in command['owners'] + [command['direct_identity']]:
                    now = m.identity(old['pid']); assert now is None or now['start_ticks'] != old['start_ticks']
            persist(); dst.write_bytes((root / 'record.json').read_bytes())
        print(json.dumps(dict(receipt=str(dst), passed=rec['passed'], cases=len(rec['cases']), ownership_closed=rec['ownership_closed'], GPU_executed=False, error=rec.get('error'))))
        return 0 if rec['passed'] and rec['ownership_closed'] else 1

if __name__ == '__main__': sys.exit(main())
