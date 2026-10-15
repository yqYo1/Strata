#!/usr/bin/env python3
"""Main-owned CPU fault injection for the actual proposed PCIe helper."""
import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_path = A / 'selective-iq-clean-bench-full-0-v1.json'
        previous = json.loads(previous_path.read_text())
        require(not previous['active'] and previous.get('commands'), 'full OFF still active or not recorded')
        for c in previous['commands']:
            require(c.get('finished_utc') and c.get('direct_child_reaped') and
                    c.get('session_empty') and not c.get('survivors'), 'earlier command not retired')
            for identity in c.get('owners', []):
                current = module.identity(identity['pid'])
                require(current is None or current['start_ticks'] != identity['start_ticks'], 'earlier owned identity remains')
        proposal = A / 'checked-pcie-fetch-v1'
        inputs = {}
        for name in ('pins.json', 'artifact-sha256.json'):
            inputs[str((proposal / name).relative_to(P))] = sha(proposal / name)
        for rel, pin in json.loads((proposal / 'pins.json').read_text()).items():
            require(sha(P / rel) == pin, 'base changed before host qualification: ' + rel)
            inputs[rel] = pin
        for rel, pin in json.loads((proposal / 'artifact-sha256.json').read_text()).items():
            require(sha(proposal / rel) == pin, 'proposed artifact changed: ' + rel)
            inputs[str((proposal / rel).relative_to(P))] = pin
        for path in [Path(__file__), P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py',
                     A / 'owned-stall-stack-probe-v1/owned_stack_probe.py']:
            inputs[str(path.relative_to(P))] = sha(path)
        root = B / 'main-checked-pcie-fetch-host-root-v1'
        receipt = A / 'checked-pcie-fetch-host-v1.json'
        require(not root.exists() and not receipt.exists(), 'fresh qualification root required')
        compiler = Path('/usr/bin/c++').resolve()
        record = dict(schema='main-checked-pcie-fetch-host-v1', active=True, passed=False, complete=False,
            GPU_executed=False, product_applied=False, performance_qualified=False,
            quality_qualified=False, physical_262144_qualified=False, restore_qualified=False,
            source_pins=inputs, compiler=str(compiler), compiler_sha256=sha(compiler), commands=[],
            prior_terminal_sha256=sha(previous_path),
            scope='Actual production helper with fake enqueue/drain APIs and asserted fault cases. '
                  'No real Verifier/SYCL queue, device error, source pin registration, model or speed qualification.')
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = None
        root_created = False
        env = dict(PATH='/usr/bin:/bin', HOME=str(root), LANG='C.UTF-8', LC_ALL='C.UTF-8',
            ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
        fixture = proposal / 'cpu_fault_fixture.cpp'
        plans = []
        for variant in ('normal', 'asan'):
            exe = root / ('checked-pcie-fetch-' + variant)
            flags = ['-O2'] if variant == 'normal' else ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
            build = ['/usr/bin/c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-UNDEBUG',
                     *flags, str(fixture), '-o', str(exe)]
            plans.append((variant, exe, [(variant + '-build', build), (variant + '-run', [str(exe)])]))
        try:
            # All deterministic read-only preflight/plan/record setup is above.
            # Never delete or adopt an existing root on an initialization failure.
            root.mkdir(mode=0o700)
            root_created = True
            record['stage'] = 'owner_initialization'
            owner = module.Owner(root / 'owned', persist)
            record['commands'] = owner.commands
            record['owner_initialized'] = True
            persist()
            record['stage'] = 'commands'
            for variant, exe, commands in plans:
                for label, argv in commands:
                    c, out, err = owner.run(label, argv, env, P, wall=120, cpu=120,
                        rss_cap=2 << 30, text_cap=1 << 20, total_cap=128 << 20, file_cap=32 << 20)
                    require(module.closed(c) and c['exit_code'] == 0, label + ' failed: ' + err.read_text()[-3000:])
                    if label.endswith('-run'):
                        # This fixture is silent on success. Unexpected output is
                        # retained and rejected, including sanitizer diagnostics.
                        require(out.stat().st_size == err.stat().st_size == 0, 'unexpected fixture output')
                        record.setdefault('variants', []).append(dict(name=variant, binary_sha256=sha(exe),
                            source_sha256=sha(fixture), asserted=True, normal_exit=True))
                persist()
            require(all(sha(P / rel) == pin for rel, pin in inputs.items()), 'source changed during qualification')
            require(sha(compiler) == record['compiler_sha256'], 'compiler changed during qualification')
            require([v['name'] for v in record['variants']] == ['normal', 'asan'], 'both actual variants required')
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            if root_created:
                # A constructor failure has no Owner/process; do not invent a
                # closed command receipt. Once initialized, Owner is authoritative.
                record.update(active=owner is not None and owner.active is not None,
                              owner_initialized=owner is not None, finished_utc=module.utc())
                commands_closed = owner is not None and all(module.closed(c) for c in owner.commands)
                record['ownership_state'] = ('unresolved' if record['active'] else
                    'not_started' if owner is None or not owner.commands else
                    'closed' if commands_closed else 'incomplete')
                if record['active'] or (owner is not None and not commands_closed):
                    record.update(passed=False, complete=False)
                # Attempt the external terminal receipt even if local persistence
                # fails. Both paths may still fail on real filesystem errors.
                try:
                    persist()
                except BaseException as persist_error:
                    record.update(passed=False, complete=False, local_persist_error=repr(persist_error))
                    raise
                finally:
                    try:
                        with receipt.open('x') as stream:
                            json.dump(record, stream, indent=2)
                            stream.write('\n')
                    except BaseException as receipt_error:
                        record.update(passed=False, complete=False, receipt_write_error=repr(receipt_error))
                        persist()
                        raise
        print(json.dumps(dict(passed=record['passed'], GPU_executed=False)))


if __name__ == '__main__':
    main()
