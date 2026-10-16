#!/usr/bin/env python3
"""Main-only fresh ABI build after audited cohort closure and explicit source application."""
import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id


def main():
    module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        references = dict(previous=A/'license-clean-prefill-build-v2.json',
                          clean=A/'native-direct-clean-cohort-closure-v1.json',
                          proposal=A/'prefill-bank-profile-main-v1/pins.json')
        refs = {k:json.loads(p.read_text()) for k,p in references.items()}
        previous, clean, proposal = (refs[k] for k in ('previous', 'clean', 'proposal'))
        require(previous['passed'] and previous['complete'] and not previous['active'], 'prior actual ABI build')
        require(clean['closed'] and not clean['active'] and clean['safe_to_build'] and
                len(clean['samples']) == 4 and not clean['comparison_complete'] and
                not clean['all_six_passed'] and clean['no_automatic_retry'],
                'audited first-failure cohort closure, without performance adoption')
        for sample in clean['samples']:
            old_path = A/sample['receipt']
            require(sha(old_path) == clean['references'][sample['receipt']], 'clean sample receipt identity')
            old = json.loads(old_path.read_text())
            require(old['passed'] == sample['passed'] and not old['active'] and
                    old['boot_id'] == boot_id() and old['boot_unchanged'] and
                    not old['fresh_faults'] and not old['driver_dump_exists'], 'same-boot clean predecessor')
            if not old['passed']:
                require(sample['order'] == 3 and old['result']['real_engine_exit']['exit_code'] == 1 and
                        clean['error_excerpt'] == 'strata serve: prefill layer-major: requested layer-major GPU residual prefix plus staging exceeds free VRAM with headroom',
                        'only audited preallocation admission failure is waived for compilation')
            for command in old['commands']:
                require(module.closed(command), 'prior process closure')
                for identity in command['owners']:
                    now = module.identity(identity['pid'])
                    require(now is None or now['start_ticks'] != identity['start_ticks'], 'prior owner still present')
        require(sha(A/'prefill-bank-profile-main-v1/source.diff') == proposal['source_diff_sha256'], 'proposal diff pin')
        require(all(sha(P/n) == item['proposed_sha256'] for n,item in proposal['files'].items()),
                'Main must apply/review exact candidate before build')
        for n,h in previous['source_pins'].items():
            if n not in proposal['files']:
                require(sha(P/n) == h, 'unrelated source changed '+n)
        require(sha(Path(previous['binary'])) == previous['binary_sha256'], 'preserved clean reference binary')
        root = B/'main-canonical-bank-profile-build-root-v1'
        output = A/'canonical-bank-profile-build-v1.json'
        require(not root.exists() and not output.exists(), 'fresh owned build root')
        root.mkdir(mode=0o700)
        build = root/'build'
        argv = list(previous['configure_argv'])
        require(argv[argv.index('-S')+1] == str(P), 'own actual source resolver')
        argv[argv.index('-B')+1] = str(build)
        require('-DSTRATA_ENABLE_SYCL=ON' in argv and '-DSTRATA_ENABLE_CUDA=OFF' in argv and
                '-DSTRATA_ENABLE_HIP=OFF' in argv and '-DSTRATA_CPU_ARCH=znver3' in argv,
                'preserve qualified hardware compiler settings')
        compilers = [Path(a.split('=',1)[1]) for a in argv if a.startswith('-DCMAKE_CXX_COMPILER=')]
        require(len(compilers) == 1 and sha(compilers[0]) == previous['compiler_sha256'],
                'qualified compiler identity')
        binaries = ('strata', 'canonical_bank_policy_test', 'sycl_canonical_dense_bank_test',
                    'sycl_canonical_dense_bank_fault_test', 'sycl_canonical_native_reserve_fault_test',
                    'sycl_native_dense_bank_test', 'sycl_native_dense_bank_fault_test')
        sources = dict(previous['source_pins'])
        sources.update({n:item['proposed_sha256'] for n,item in proposal['files'].items()})
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[],
                      recipe_sha256=sha(Path(__file__)), references={p.name:sha(p) for p in references.values()},
                      environment=previous['environment'], compiler=previous['compiler'],
                      compiler_sha256=previous['compiler_sha256'], configure_argv=argv,
                      build_path=str(build), source_pins=sources, proposal_sha256=proposal['source_diff_sha256'],
                      preceding_performance_comparison_complete=False,
                      performance_qualified=False, quality_qualified=False, restore_qualified=False,
                      physical_262144_qualified=False,
                      scope='Fresh own SYCL product ABI build and manual fixture binaries only. '
                            'No device or fixture execution. Canonical bank and coarse profile default OFF.')
        def persist():
            (root/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(root/'owned', persist)
        record['commands'] = owner.commands
        persist()
        try:
            for label, command, wall in (
                    ('configure-canonical-profile', argv, 300),
                    ('build-canonical-profile', ['cmake', '--build', str(build), '--target', *binaries,
                                                 '--parallel', '2'], 1200)):
                c,out,err = owner.run(label, command, record['environment'], P,
                                      wall=wall, cpu=wall*12, rss_cap=32<<30,
                                      text_cap=16<<20, total_cap=1<<30, file_cap=512<<20)
                require(module.closed(c) and c['exit_code'] == 0,
                        label+': '+err.read_text()[-4000:]+'\n'+out.read_text()[-3000:])
            require(all(sha(P/n) == h for n,h in sources.items()), 'candidate source stable through build')
            require(sha(Path(previous['binary'])) == previous['binary_sha256'], 'reference product still preserved')
            compilation = json.loads((build/'compile_commands.json').read_text())
            proofs = {}
            for name in ('src/core/weights.cpp', 'src/core/native_dense.cpp',
                         'src/core/verify.cpp', 'src/prefill/prefill.cpp', 'src/program/generate.cpp'):
                matches = [c for c in compilation if Path(c['file']).resolve() == (P/name).resolve()]
                require(matches, 'actual own-source compiler entry '+name)
                proofs[name] = matches
            record.update(binary=str(build/'strata'), binary_sha256=sha(build/'strata'),
                          binaries={n:sha(build/n) for n in binaries}, compile_flag_proof=proofs,
                          compile_commands_sha256=sha(build/'compile_commands.json'), passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):
                record.update(passed=False, complete=False)
            persist()
            with output.open('x') as stream:
                json.dump(record, stream, indent=2); stream.write('\n')
        print(json.dumps(dict(passed=record['passed'], GPU_executed=False, binaries=record['binaries'])))


if __name__ == '__main__':
    main()
