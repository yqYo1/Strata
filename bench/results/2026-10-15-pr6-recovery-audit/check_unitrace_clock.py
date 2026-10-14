#!/usr/bin/env python3
"""CPU-only reproduction of the pinned profiler header's clock-domain mismatch."""
import fcntl
import importlib.util
import json
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, sha

O = B/'main-primary-unitrace-clock-source-root-v1'
PTI = Path('/home/yayoi/ghq/github.com/intel/pti-gpu')


def main():
    tool = json.loads((B/'main-primary-unitrace-build-root-v1/record.json').read_text())
    assert tool['passed'] and tool['complete'] and not tool['active']
    assert tool['source_revision'] == '6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033'
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists()
        O.mkdir(mode=0o700)
        source = D/'unitrace_clock_fixture.cpp'
        pins = {str(p): sha(p) for p in [source, PTI/'tools/unitrace/src/unitimer.h',
                PTI/'utils/utils.h', PTI/'utils/pti_assert.h']}
        env = dict(tool['environment'])
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        assert 'UNITRACE_SystemTime' not in env and 'LD_PRELOAD' not in env
        r = dict(active=True, passed=False, complete=False, GPU_executed=False, commands=[],
            source_pins=pins, profiler_source_revision=tool['source_revision'],
            profiler_build_receipt_sha256=sha(B/'main-primary-unitrace-build-root-v1/record.json'),
            recipe_sha256=sha(Path(__file__)), environment=env,
            external_UTC_correlation_qualified=False, historical_timestamps_modified=False,
            limits=dict(wall_seconds=30, raw_text_bytes=1 << 20, job_bytes=16 << 20),
            scope='Pinned unmodified UniTimer header in a standalone CPU process; compare RAW-based reported epoch to REALTIME brackets and BOOTTIME-RAW delta. This does not align an old or model trace.')

        def persist():
            (O/'record.json').write_text(json.dumps(r, indent=2)+'\n')

        owner = module.Owner(O/'owned', persist)
        r['commands'] = owner.commands
        persist()

        def run(label, argv, extra_env=None):
            runenv = dict(env, **(extra_env or {}))
            entry, out, err = owner.run(label, argv, runenv, P, wall=30, cpu=30,
                text_cap=1 << 20, total_cap=16 << 20, file_cap=8 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, err.read_text())
            return out.read_text()

        try:
            assert run('source-revision', ['git', '-C', str(PTI), 'rev-parse', 'HEAD']).strip() == tool['source_revision']
            assert not run('source-clean', ['git', '-C', str(PTI), 'status', '--porcelain']).strip()
            argv = ['/usr/bin/c++', '-std=c++17', '-O2', '-I'+str(PTI/'utils'),
                '-I'+str(PTI/'tools/unitrace/src'), str(source), '-o', str(O/'clock-fixture')]
            run('compile-clock-only', argv)
            r['binary_sha256'] = sha(O/'clock-fixture')
            samples = [list(map(int, line.split())) for line in run('clock-only', [str(O/'clock-fixture')]).splitlines()]
            assert len(samples) == 32 and [s[0] for s in samples] == list(range(32))
            values = []
            for i, rb, bb, raw, reported, ba, ra in samples:
                assert rb <= ra and bb <= ba and ra-rb < 1000000
                # At most 1ms residual after accounting for the *measured* clock
                # domains establishes this reproduction, without patching PTI.
                low, high = reported + bb-raw-ra, reported + ba-raw-rb
                assert low <= 1000000 and high >= -1000000
                assert abs(reported-(rb+ra)//2) > 1000000
                values.append(dict(index=i, realtime_before_ns=rb, boottime_before_ns=bb,
                    raw_ns=raw, reported_epoch_ns=reported, boottime_after_ns=ba, realtime_after_ns=ra,
                    observed_reported_minus_realtime_ns=[reported-ra, reported-rb],
                    boottime_minus_raw_ns=[bb-raw, ba-raw],
                    accounted_domain_residual_ns=[low, high]))
            r['samples'] = values
            raw_samples = [list(map(int, line.split())) for line in run('clock-only-raw-mode',
                [str(O/'clock-fixture')], {'UNITRACE_SystemTime': '1'}).splitlines()]
            assert len(raw_samples) == 32 and [s[0] for s in raw_samples] == list(range(32))
            assert all(raw == reported for _, _, _, raw, reported, _, _ in raw_samples)
            r['raw_mode_samples'] = raw_samples
            r['raw_mode_qualification'] = 'CPU-only pinned UniTimer header reports CLOCK_MONOTONIC_RAW exactly when UNITRACE_SystemTime=1. This is an internal flag; actual GPU trace in that mode is not yet qualified. Future collectors must record matching RAW bounds rather than relabel RAW as UTC.'
            assert all(sha(Path(p)) == pin for p, pin in pins.items())
            r.update(passed=True, complete=True, clock_domain_mismatch_reproduced=True)
        except BaseException as error:
            r['error'] = repr(error)
            raise
        finally:
            r['active'] = False
            persist()
            (D/'unitrace-clock-source-v1.json').write_text(json.dumps(r, indent=2)+'\n')
        print(json.dumps(dict(passed=True, samples=32, GPU_executed=False,
            external_UTC_correlation_qualified=False)), flush=True)


if __name__ == '__main__':
    main()
