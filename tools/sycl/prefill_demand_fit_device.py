"""Build only the private-bank component and exercise its ordinary B570 queue."""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


P = Path(__file__).resolve().parents[2]
MAIN = Path('/home/yayoi/orca/workspaces/Strata/sycl')
sys.dont_write_bytecode = True
sys.path.insert(0, str(MAIN / 'tools/sycl'))
import prefill_resident_input_build as bank
import prefill_demand_fit_coverage as coverage

require, sha, signature = bank.require, bank.sha, bank.signature
owner_module, validation = bank.builder.ownership, bank.basis.validation
ROOT = bank.B / 'prefill-demand-fit-device-v2'
KEY = 'prefill_demand_fit_device_v2'
RESULT = P / bank.RESULT.relative_to(MAIN)
TARGET = 'sycl_expert_bank_test'
SOURCE = 'ddeff7615019480c004433a931eb1522156affe2'
BASE = 'b043dfd3a2dd2939118c41d44fde001d1bd16c9d'
CHANGED = ('include/strata/prefill/expert_bank_upload.hpp', 'include/strata/prefill/fullfit_upload_plan.hpp',
           'src/prefill/layer_major.inc', 'src/prefill/moe_window.inc',
           'tests/prefill/fullfit_upload_plan_test.cpp', 'tests/sycl/expert_bank.cpp')
UNITS = {'tests/sycl/expert_bank.cpp', 'src/core/expert_cache.cpp', 'src/sycl/runtime.cpp',
         'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp'}
BUILD_LIMIT = dict(wall=540, cpu=1200, rss_cap=6 << 30, text_cap=8 << 20, file_cap=192 << 20, total_cap=512 << 20)
SMOKE_LIMIT = dict(wall=120, cpu=240, rss_cap=3 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=96 << 20)
read, write = bank.previous.read, bank.stage.write


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P, timeout=30)


def fixture_path(value):
    return value.replace(str(bank.ROOT), str(ROOT))


def commands(root):
    return subprocess.check_output(['/usr/bin/ninja', '-t', 'commands', TARGET], cwd=root / 'build',
                                   timeout=30, text=True).replace(str(root), '<root>').splitlines()


def parents():
    parent = read(bank.ROOT / 'record.json')
    original = bank.check(parent)
    require(parent['passed'] and parent['ownership_closed'] and not parent['active']
            and parent['binaries']['strata']['sha256'] == 'b35d66a4f29737a47abcdd7180c80eaf838b8658ec49ca2586948198d6a7613f',
            'require the exact coherent model and fixtures')
    for name in CHANGED:
        require(git('show', BASE + ':' + name) == (bank.ROOT / 'source' / name).read_bytes()
                and git('show', SOURCE + ':' + name) == (P / name).read_bytes(), 'overlay mismatch: ' + name)
    for script in (Path(__file__).resolve(), Path(coverage.__file__).resolve()):
        require(script.parent == P / 'tools/sycl' and
                git('show', 'HEAD:' + str(script.relative_to(P))) == script.read_bytes(),
                'commit the exact fixture and receipt controllers')
    return parent, original


def launch(child, name, argv, env, cwd, budget):
    item, out, err = child.run(name, argv, env, cwd, **budget)
    require(owner_module.closed(item) and item['exit_code'] == 0, name + ' failed: ' + err.read_text()[-1050:])
    return item, out, err


def object_inputs(root, name):
    out = subprocess.check_output(['/usr/bin/ninja', '-t', 'deps', name], cwd=root / 'build', text=True, timeout=20)
    lines = out.splitlines()
    require(lines and 'VALID' in lines[0], 'missing object dependencies: ' + name)
    deps = {(root / 'build' / line.strip()).resolve() for line in lines[1:] if line.startswith('    ')}
    require(deps, 'empty compiler dependency set')
    return deps


def compile_inputs(parent, source_hashes):
    units, compiled, objects = set(), {}, {}
    for entry in read(ROOT / 'build/compile_commands.json'):
        args = shlex.split(entry['command'])
        obj = args[args.index('-o') + 1]
        if not (ROOT / 'build' / obj).exists():
            continue
        units.add(str(Path(entry['file']).relative_to(ROOT / 'source')))
        objects[obj] = sha(ROOT / 'build' / obj)
        prior = object_inputs(bank.ROOT, obj)
        actual = object_inputs(ROOT, obj)
        require(actual == {Path(fixture_path(str(p))) for p in prior}, 'compiled headers differ: ' + obj)
        for path in prior:
            new = Path(fixture_path(str(path)))
            want = source_hashes[str(path.relative_to(bank.ROOT/'source'))] if path.is_relative_to(bank.ROOT/'source') else parent['actual_inputs']['compiled'][str(path)]
            require(sha(new) == want, 'unverified compilation input: ' + str(new))
            compiled[str(new)] = want
    require(units == UNITS, 'expected just the five freshly compiled fixture/runtime translation units')
    return dict(units=sorted(units), inputs=len(compiled), inputs_sha256=signature(compiled), objects=objects)


def build(row, parent, child):
    row['receipt_checks'] = coverage.check_reader()
    for folder in ('tmp', 'cache', 'source'):
        (ROOT / folder).mkdir(mode=0o700)
    source_hashes = dict(parent['source_files'])
    for name in source_hashes:
        dest = ROOT / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        raw = git('show', SOURCE + ':' + name) if name in CHANGED else (bank.ROOT / 'source' / name).read_bytes()
        dest.write_bytes(raw)
        if name in CHANGED:
            source_hashes[name] = sha(dest)
        require(sha(dest) == source_hashes[name], 'incorrect materialized source ' + name)
    row['source_files_sha256'] = signature(source_hashes)
    env = {k: fixture_path(v) for k, v in parent['recipe']['environment'].items()}
    configure = list(map(fixture_path, parent['recipe']['configure']))
    launch(child, 'configure', configure, env, P, BUILD_LIMIT)
    planned = commands(ROOT)
    require(planned == commands(bank.ROOT), 'actual fixture compile, archive or link commands differ')
    row['commands_sha256'] = signature(planned)
    _, out, _ = launch(child, 'compile', ['/usr/bin/cmake', '--build', str(ROOT/'build'), '--target', TARGET, '-j', '4'],
                       env, P, BUILD_LIMIT)
    compiled = compile_inputs(parent, source_hashes)
    expected = parent['actual_inputs']['links']
    links, fresh, trace = {}, {}, []
    temporary = 0
    for line in out.read_text().splitlines():
        name = line.strip()
        if not name or any(c.isspace() for c in name) or len(name) > 4096:
            continue
        path = Path(name)
        if not path.is_absolute() and path.suffix not in ('.o','.a') and '.so' not in path.name:
            continue
        path = (ROOT / 'build' / path).resolve()
        if not path.exists():
            require(path.is_relative_to(ROOT / 'tmp') and path.suffix == '.o', 'unidentified linker input: ' + name)
            temporary += 1
            trace.append('temporary/' + path.stem.rsplit('-',1)[0])
        elif path.is_relative_to(ROOT / 'build'):
            fresh[str(path.relative_to(ROOT / 'build'))] = sha(path)
            trace.append('build/' + str(path.relative_to(ROOT / 'build')))
        else:
            require(expected.get(str(path)) == sha(path), 'unexpected external linker input: ' + str(path))
            links[str(path)] = expected[str(path)]
            trace.append(str(path))
    # The verified parent linker trace has two unbundling passes. The
    # compiler names its temporary object files with fresh random suffixes.
    trace_hash = hashlib.sha256(('\n'.join(trace) + '\n').encode()).hexdigest()
    require(len(trace) == 83 and trace_hash == '3a8abfb580c8cd45d75dd80af986737a4191d75b3ca5461de1aa9d11e42ac7e3'
            and set(fresh) == {'libstrata_sycl_runtime.a'} and temporary == 5, 'fixture linker trace changed')
    row['build'] = dict(compiler=compiled, link_sha256=trace_hash, links=links, produced=fresh,
                        compiler_temporaries=temporary, binary=sha(ROOT / 'build' / TARGET), partial_archive_reuse=False)


def smoke(row, parent, original, child, outcome):
    require(row.get('build_passed') and sha(ROOT / 'build' / TARGET) == row['build']['binary']
            and commands(ROOT) == commands(bank.ROOT) and signature(commands(ROOT)) == row['commands_sha256'],
            'fixture differs from the committed build')
    have = {name:sha(ROOT/'source'/name) for name in parent['source_files']}
    require(signature(have) == row['source_files_sha256'] and compile_inputs(parent, have) == row['build']['compiler']
            and all(sha(ROOT/'build'/p) == h for p,h in row['build']['produced'].items()), 'fixture source or objects changed')
    for name, digest in row['build']['links'].items():
        require(sha(name) == digest, 'linked dependency changed: ' + name)
    selectable = {'jit': (0, 0), 'eager': (1, 0), 'demand-fit-only': (0, 1), 'eager-demand-fit': (1, 1)}
    for policy, (full, partial) in selectable.items():
        env = dict(parent['recipe']['environment'], **bank.device.FIXTURE_ENV)
        env.update(STRATA_SYCL_PREFILL_FULLFIT_UPLOAD=policy, TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
        _, out, _ = launch(child, 'policy-' + policy, [str(ROOT / 'build' / TARGET), '--prefill-upload-policy'],
                           env, P, SMOKE_LIMIT)
        require(out.read_text().splitlines() == [f'prefill_upload_policy full={full} partial={partial}'],
                'incorrect immutable selection for ' + policy)
    outcome['policy_selections'] = selectable
    outcome['device'] = {}
    for policy in ('demand-fit-only', 'eager-demand-fit'):
        outcome['device'][policy] = device_cases(parent, original, child, policy)
    old = json.loads(git('show', '9b39df072:' + str(RESULT.relative_to(P))))['prefill_demand_fit_device_v1']
    require(old['passed'] and old['ownership_closed'] and outcome['device']['demand-fit-only']['cases']
            == outcome['device']['eager-demand-fit']['cases'] == old['smoke_outcome']['cases'],
            'partial-bank fixture differs across policies or from the completed original')
    outcome['host_probe_accepted'] = True


def device_cases(parent, original, child, policy):
    d = ROOT / ('smoke-' + policy)
    for k in ('tmp','cache'):
        (d / k).mkdir(mode=0o700, parents=True)
    probe = original['native_validation_probe']
    env = dict(parent['recipe']['environment'], **bank.device.FIXTURE_ENV)
    env.update(STRATA_SYCL_PREFILL_FULLFIT_UPLOAD=policy, TMPDIR=str(d/'tmp'),
               XDG_CACHE_HOME=str(d/'cache'), LD_DEBUG='libs', LD_PRELOAD=probe['library']['path'],
               STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'], UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_TRACING='level:info;flush:info;output:stderr', UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(d/'validation.log'))
    item, out, err = launch(child, 'gpu-' + policy, [str(ROOT/'build'/TARGET), '--demand-fit-smoke'], env, P, SMOKE_LIMIT)
    log = err.read_text()
    lines = out.read_text().splitlines()
    cases = [dict(s.split('=',1) for s in line.split()[1:]) for line in lines if line.startswith('expert_bank_private_case ')]
    require([(int(v['experts']),int(v['layers']),int(v['unwind'])) for v in cases] == [(5,2,0),(6,2,0),(5,1,1)]
            and all(v['status']=='0' and all(v[k]=='PASS' for k in ('immutable_slab','guards','private_map','checked_drain')) for v in cases)
            and 'prefill_demand_fit PASS exact_fit overflow readers guards cancellation' in lines, 'partial-bank result differs')
    require('urEnqueue' in log and (not (d/'validation.log').exists() or (d/'validation.log').stat().st_size == 0), 'trace or validation failure')
    activation = validation.preload_evidence(log, item['pid'])
    require(activation['loader'] == probe['loader']['path'], 'validation loader differs')
    bank.device.diagnostics.diagnostics(log, item['pid'])
    providers = bank.device.loaded_providers(log, dict(environment=env, native_validation_probe=probe,
                                                       runtime_manifest=parent['foundation']['runtime_manifest']))
    return dict(cases=cases, validation=activation, providers=providers, environment_sha256=signature(env))


def main(action):
    parent, original = parents()
    reference = signature(bank.builder.compact(parent))
    program = signature({Path(path).name: sha(path) for path in (__file__, coverage.__file__)})
    if action == 'build':
        require(not ROOT.exists() and KEY not in read(RESULT), 'inspect the existing build attempt')
        ROOT.mkdir(mode=0o700)
        row = dict(source=SOURCE, base=BASE, parent=reference, controller=program,
                   plan='Five fresh fixture and runtime units. Four independent immutable startup policies. '
                        'B570 exact fit, overflow, guards and cancellation with partial-only and combined issue. '
                        'Compare with the committed V1 combined-mode outcome. Retire closed isolated fixture '
                        'artifacts after their original source recovery. No engine or speed qualification.')
    else:
        row = read(ROOT/'record.json')
        require(row == json.loads(git('show','HEAD:'+str(RESULT.relative_to(P))))[KEY] and row['passed']
                and row['stage']=='build' and row['ownership_closed'] and row['parent']==reference and row['controller']==program,
                'commit a passing identical build before GPU work')
    directory = ROOT / ('owner-' + action)
    child = owner_module.Owner(directory, lambda: write(ROOT/(action+'-ownership.json'), child.commands))
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursor, = (l.removeprefix('-- cursor: ') for l in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True,timeout=15).splitlines() if l.startswith('-- cursor: '))
    outcome = {}
    try:
        if action == 'build':
            build(row, parent, child)
            row['build_passed'] = True
            outcome['host_probe_accepted'] = True
        else:
            smoke(row,parent,original,child,outcome)
    except Exception as exc:
        outcome['error'] = repr(exc)
    finally:
        validation.finish(outcome,child,cursor,boot)
        row.update(stage=action, passed=outcome['passed'] and outcome.get('driver_after')==parent['driver'],
                   ownership_closed=outcome['ownership_closed'], **{action+'_outcome':outcome})
        write(ROOT/'record.json',row)
        data=read(RESULT);data[KEY]=row;write(RESULT,data)
    require(row['passed'],outcome.get('error', 'owner or driver did not pass'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('build','smoke') and sys.executable==bank.basis.reference.model.PYTHON,'use the pinned interpreter and one planned action')
    with (bank.B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        main(action)
