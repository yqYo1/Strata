#!/usr/bin/env python3
"""Main's isolated checked-fetch build; requires real serial host qualification."""
import fcntl
import json
import os
from pathlib import Path
import shlex
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
        host_path = A / 'checked-pcie-fetch-host-v1.json'
        base_path = A / 'selective-iq-native-build-v1.json'
        host, base = [json.loads(p.read_text()) for p in (host_path, base_path)]
        for previous in (host, base):
            require(previous['passed'] and previous['complete'] and not previous['active'], 'closed host/base gate required')
            require(all(module.closed(c) for c in previous['commands']), 'earlier build/test owner not closed')
            for c in previous['commands']:
                for identity in c.get('owners', []):
                    current = module.identity(identity['pid'])
                    require(current is None or current['start_ticks'] != identity['start_ticks'], 'earlier identity remains')
        require([v['name'] for v in host['variants']] == ['normal', 'asan'], 'actual helper variants missing')
        require(all(sha(P / rel) == pin for rel, pin in host['source_pins'].items()), 'qualified helper/input changed')
        require(all(sha(P / rel) == pin for rel, pin in base['source_pins'].items()), 'frozen native base source changed')
        require(all(sha(Path(n)) == h for n, h in base['linked_inputs'].items()), 'frozen native link inputs changed')
        require(sha(Path(base['binary'])) == base['binary_sha256'], 'reference native changed')
        compiler = Path(base['compiler'])
        require(sha(compiler) == base['compiler_sha256'], 'SYCL compiler changed')
        baseline = B / 'main-primary-native-build-root-v1/build'
        database = baseline / 'compile_commands.json'
        rows = json.loads(database.read_text())
        proposal = A / 'checked-pcie-fetch-v1'
        overlay = proposal / 'files/include'
        header = overlay / 'strata/core/verify.hpp'
        helper = overlay / 'strata/core/checked_pcie_fetch.hpp'
        root = B / 'main-checked-pcie-fetch-build-root-v1'
        receipt = A / 'checked-pcie-fetch-build-v1.json'
        require(not root.exists() and not receipt.exists(), 'fresh isolated build root required')
        inputs = {**base['source_pins'], **host['source_pins'],
                  str(Path(__file__).relative_to(P)): sha(Path(__file__)),
                  str(database): sha(database)}
        commands = []
        objects = {}
        dependencies = {}
        for role, source, replacement in [
            ('verify', P / 'src/core/verify.cpp', proposal / 'files/src/core/verify.cpp'),
            ('generate', P / 'src/program/generate.cpp', P / 'src/program/generate.cpp')]:
            matches = [r for r in rows if r['file'] == str(source)]
            require(len(matches) == 1 and Path(matches[0]['directory']) == baseline, 'actual compile resolver: ' + role)
            argv = shlex.split(matches[0]['command'])
            require(argv[0] == str(compiler), 'unexpected source compiler')
            require(not any(x in argv for x in ('-include', '-include-pch', '-MF', '-MT', '-MJ')), 'unexpected preinclude/dependency side effect')
            version = [x for x in argv if x.startswith('-DSTRATA_VERSION=')]
            require(len(version) == 1 and version[0].split('=', 1)[1].replace('\\', '').replace('"', '') == '0.1.38',
                    'historical version macro changed')
            # compile_commands stores shell escaping. Supply an ordinary quoted
            # string macro in argv; do not interpret it through another shell.
            argv[argv.index(version[0])] = '-DSTRATA_VERSION="0.1.38"'
            exe_object = root / (role + '-checked-pcie.o')
            dep = root / (role + '.d')
            argv[argv.index('-o') + 1] = str(exe_object)
            argv[argv.index(str(source))] = str(replacement)
            argv[1:1] = ['-I' + str(overlay), '-MMD', '-MF', str(dep)]
            commands.append(('compile-' + role, argv))
            objects[role], dependencies[role] = exe_object, dep
            inputs[str(source.relative_to(P))] = sha(source)
            inputs[str(replacement.relative_to(P))] = sha(replacement)
        links = [c['argv'] for c in base['commands'] if c['label'] == 'link-native']
        require(len(links) == 1, 'one actual reference native link command required')
        link = list(links[0])
        old_gen = baseline / 'CMakeFiles/strata.dir/src/program/generate.cpp.o'
        require(link.count(str(old_gen)) == 1, 'reference generate link identity')
        i = link.index(str(old_gen))
        link[i:i + 1] = [str(objects['generate']), str(objects['verify'])]
        native = root / 'strata'
        link[link.index('-o') + 1] = str(native)
        maps = [x for x in link if x.startswith('-Wl,-Map=')]
        require(len(maps) == 1, 'one reference link map required')
        link[link.index(maps[0])] = '-Wl,-Map=' + str(root / 'native.map')
        commands.append(('link-native', link))
        cpu_limit = 900 * len(os.sched_getaffinity(0))
        require(all(isinstance(k, str) and isinstance(v, str) for k, v in base['environment'].items()),
                'explicit string build environment')
        record = dict(schema='main-checked-pcie-fetch-build-v1', active=True, passed=False, complete=False,
            GPU_executed=False, product_applied=False, performance_qualified=False,
            quality_qualified=False, physical_262144_qualified=False, restore_qualified=False,
            commands=[], source_pins=inputs, reference_linked_inputs=base['linked_inputs'],
            compiler=str(compiler), compiler_sha256=sha(compiler), environment=base['environment'],
            reference_build_sha256=sha(base_path), host_qualification_sha256=sha(host_path),
            reference_binary_sha256=base['binary_sha256'], historical_version_macro='0.1.38',
            scope='Both Verifier and its actual owning/calling TU rebuilt with one proposed class layout. '
                  'Other frozen native objects/libraries unchanged. Isolated candidate only; no GPU or model qualification.')
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = None
        root_created = False
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
            for label, argv in commands:
                c, out, err = owner.run(label, argv, base['environment'], baseline, wall=900,
                    cpu=cpu_limit, rss_cap=32 << 30,
                    text_cap=8 << 20, total_cap=2 << 30, file_cap=512 << 20)
                require(module.closed(c) and c['exit_code'] == 0, label + ' failed: ' + err.read_text()[-4000:])
                persist()
            for role, dep in dependencies.items():
                require(dep.is_file() and dep.stat().st_size <= 4 << 20, 'bounded compiler dependency proof')
                text = dep.read_text().replace('\\\n', ' ')
                require(str(header) in text and str(helper) in text, 'candidate header missing from ' + role + ' dependency proof')
                require(str(P / 'include/strata/core/verify.hpp') not in text, 'old Verifier layout reached ' + role)
            mapping = (root / 'native.map').read_text()
            require(all(str(p) in mapping for p in objects.values()), 'new objects absent from link map')
            require('verify.cpp.o' not in mapping and 'generate.cpp.o' not in mapping, 'old Verifier/generate object selected')
            require(all(sha(P / rel) == pin for rel, pin in inputs.items()), 'source changed during build')
            require(all(sha(Path(n)) == h for n, h in base['linked_inputs'].items()), 'reference objects changed during build')
            require(sha(compiler) == record['compiler_sha256'], 'SYCL compiler changed during build')
            require(sha(Path(base['binary'])) == base['binary_sha256'], 'reference native overwritten')
            require(sha(base_path) == record['reference_build_sha256'] and
                    sha(host_path) == record['host_qualification_sha256'], 'reference/qualification receipt changed')
            linked_inputs = {n: h for n, h in base['linked_inputs'].items() if n != str(old_gen)}
            linked_inputs.update({str(p): sha(p) for p in objects.values()})
            record.update(passed=True, complete=True, binary=str(native), binary_sha256=sha(native),
                linked_inputs=linked_inputs,
                objects={role: dict(path=str(p), sha256=sha(p)) for role, p in objects.items()},
                dependencies={role: dict(path=str(p), sha256=sha(p)) for role, p in dependencies.items()},
                link_map_sha256=sha(root / 'native.map'), dependency_and_link_review=True)
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
