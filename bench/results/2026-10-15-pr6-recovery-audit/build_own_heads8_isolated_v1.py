#!/usr/bin/env python3
"""Build reviewed attention fixture without changing the NLL cohort's sources/binary."""
import fcntl
import json
from pathlib import Path
import shlex
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
S = A / 'own-prompt-heads8-v2'
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = json.loads((A / 'own-residual-quality-fp32-a-v1.json').read_text())
        require(prior['passed'] and prior['complete'] and not prior['active'] and not prior['fresh_faults'] and prior['boot_unchanged'] and not prior['driver_dump_exists'], 'first long quality arm closed healthy')
        for c in prior['commands']:
            require(module.closed(c) and c['exit_code'] == 0, 'prior owners closed')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'prior owner alive')
        known = json.loads((A / 'own-dense-prefill-nll-build-v3.json').read_text())
        build = Path(known['build_path'])
        require(sha(build / 'strata') == prior['binary_sha256'], 'canonical native binary remains pinned')
        require(all(sha(P/n) == h for n,h in prior['source_pins'].items()), 'all NLL cohort compiled sources unchanged')
        m = json.loads((S / 'manifest.json').read_text())
        require(sha(S / 'source.diff') == m['patch_sha256'] == '3e0991d0a2a89e70eed80d8f336689854e16ee7c3eb294f4f32589458cf25f0c', 'reviewed complete source proposal')
        require(sha(P / 'src/kernels/sycl/prompt_attention.cpp') == m['base_sha256']['src/kernels/sycl/prompt_attention.cpp'], 'exact candidate kernel base')
        require(all(sha(S/'files'/n) == h for n,h in m['candidate_sha256'].items()), 'candidate package pins')
        require(sha(Path(known['compiler'])) == known['compiler_sha256'], 'official qualified compiler pin')
        root = B / 'main-own-heads8-isolated-build-root-v1'
        target = A / 'own-heads8-isolated-build-v1.json'
        require(not root.exists() and not target.exists(), 'fresh isolated component build')
        root.mkdir(mode=0o700)
        libraries = [build/n for n in ('libstrata_kernels.a','libstrata_core.a','libstrata_sycl_runtime.a')]
        libpins = {str(n):sha(n) for n in libraries}
        record = dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],
                      recipe_sha256=sha(Path(__file__)),proposal_manifest_sha256=sha(S / 'manifest.json'),
                      proposal_patch_sha256=sha(S / 'source.diff'),candidate_source_pins=m['candidate_sha256'],
                      canonical_source_pins=prior['source_pins'],canonical_native=str(build/'strata'),
                      canonical_native_sha256=sha(build/'strata'),library_pins=libpins,
                      environment=known['environment'],compiler=known['compiler'],compiler_sha256=known['compiler_sha256'],
                      scope='Isolated own MIT source-only proposal build; no CMake/prod source/native/library mutation, no GPU/test/performance/quality/adoption qualification.',
                      raw_consumer='Main-owned component smoke, full96 oracle cases and only then clean repeated component performance; next review after component decision',
                      build_budget_bytes=512 << 20)
        def persist():
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner = module.Owner(root/'owned',persist)
        record['commands'] = owner.commands
        persist()
        def run(label, argv, wall=900, cpu=18000, rss=32 << 30):
            c,out,err = owner.run(label,argv,known['environment'],build,wall=wall,cpu=cpu,rss_cap=rss,
                                  text_cap=64 << 20,total_cap=1 << 30,file_cap=512 << 20)
            require(module.closed(c) and c['exit_code'] == 0,label+': '+err.read_text()[-6000:])
            return out.read_text()
        try:
            cc = json.loads((build/'compile_commands.json').read_text())
            kernel = next(v for v in cc if v['file'] == str(P/'src/kernels/sycl/prompt_attention.cpp'))
            fixture = next(v for v in cc if 'sycl_prompt_k_cellmajor_test.dir' in v['command'])
            kernel_obj,fixture_obj = root/'candidate.o',root/'fixture.o'
            commands = []
            for entry,new_source,obj in ((fixture,S/'files/tests/sycl/prompt_heads8.cpp',fixture_obj),
                                         (kernel,S/'files/src/kernels/sycl/prompt_attention.cpp',kernel_obj)):
                args = shlex.split(entry['command'])
                require(args[0] == known['compiler'] and args.count('-o') == 1 and args.count('-c') == 1, 'exact compile command layout')
                args[args.index('-o')+1] = str(obj)
                args[args.index('-c')+1] = str(new_source)
                commands.append(args)
            linked = run('read-canonical-link',['ninja','-C',str(build),'-t','commands','sycl_prompt_k_cellmajor_test'],wall=60,cpu=120,rss=2 << 30).splitlines()[-1]
            tokens = shlex.split(linked)
            require(tokens[:2] == [':','&&'] and tokens[-2:] == ['&&',':'], 'generated link command envelope')
            args = tokens[2:-2]
            require(args[0] == known['compiler'] and args.count('-o') == 1, 'official linker pin')
            oldobj = 'CMakeFiles/sycl_prompt_k_cellmajor_test.dir/tests/sycl/prompt_k_cellmajor.cpp.o'
            require(args.count(oldobj) == 1, 'one canonical fixture object')
            i = args.index(oldobj)
            args[i:i+1] = [str(fixture_obj),str(kernel_obj)]
            output = root/'sycl_prompt_heads8_isolated'
            args[args.index('-o')+1] = str(output)
            for lib in libraries:
                require(args.count(lib.name) == 1, 'exact existing archive dependency')
                args[args.index(lib.name)] = str(lib)
            record['actual_compile_commands'] = commands
            record['actual_link_command'] = args
            record['canonical_compile_commands_sha256'] = sha(build/'compile_commands.json')
            persist()
            run('compile-fixture',commands[0])
            run('compile-candidate',commands[1])
            run('link-isolated',args)
            require(all(sha(P/n) == h for n,h in prior['source_pins'].items()) and sha(build/'strata') == record['canonical_native_sha256'] and all(sha(Path(n)) == h for n,h in libpins.items()), 'native sources, binary and archives remained unchanged')
            record.update(passed=True,complete=True,binary=str(output),binary_sha256=sha(output),
                          isolated_object_pins={str(n):sha(n) for n in (fixture_obj,kernel_obj)})
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None,finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):
                record.update(passed=False,complete=False)
            persist()
            with target.open('x') as f:
                json.dump(record,f,indent=2)
                f.write('\n')
        print(json.dumps(dict(passed=record['passed'],GPU_executed=False,binary_sha256=record['binary_sha256'],canonical_native_unchanged=True)),flush=True)


if __name__ == '__main__':
    main()
