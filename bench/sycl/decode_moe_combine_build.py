"""Build one coherent T4 decode candidate on the measured B570 source image."""

import fcntl
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import prefill_large_gather_model_source as stock

bank = stock.bank
builder, require = bank.builder, bank.require
ROOT = bank.B / 'decode-moe-combine-native-v1'
KEY = 'decode_moe_combine_native_v1'
BASE = 'c4c891c09aef7c105608e9b03e98d2b8a27b8bee'
SOURCE = '05c365ecce626913967de1e29e46f722e20339fa'
PARENT_ELF = '85e3f202ef99661d52c03a17a8b609b8a8a44e9f584bfa8fc1a5c1c3d51aea7c'
TARGET = 'sycl_decode_moe_combine_test'
COMPONENT = 'decode_moe_combine_component_v1'
MODES = ('validate', 'time-forward', 'time-reverse', 'time-forward')
DEVICE_LIMITS = dict(wall=600, cpu=1200, rss_cap=4 << 30,
                     text_cap=24 << 20, file_cap=32 << 20, total_cap=64 << 20)
FIXTURE = 'tests/sycl/decode_moe_combine.cpp'
UNITS = ('src/core/verify.cpp', 'src/kernels/sycl/shared_expert.cpp', FIXTURE)
PROJECTED = {
    UNITS[0]: '5a79441b9ec90ef1b786eb584ffc36609221dd595b944d75c273a258732fac99',
    UNITS[1]: 'affe9c97194f7a7a2d02f5dadfc21f54014332474baf7530b1cd2fbc5a66753f',
    'include/strata/kernels/native_moe.hpp': 'ef2d80ade36e562dbe6a3607bc5ba4b8b48e1b9538e33296d3dddbc4c937f9bc',
    'cmake/sycl_backend.cmake': '666da4ce948432e60589910949ffa741fd9ecdf392b7463852f27d64cea83c13',
    FIXTURE: '1e14b56a60391cfba5d18f848d6f6f041fede876964d67c2ed09948fc872c01c',
}


def project(expected=None):
    native, foundation, source, recipe, _ = stock.build_inputs(expected=expected)
    record = bank.previous.read(stock.MODEL_ROOT / 'record.json')
    parent = record['binaries']['strata']
    require(record['complete'] and not record['active'] and record['ownership_closed']
            and record['passed'] and parent['sha256'] == PARENT_ELF and bank.sha(parent['path']) == PARENT_ELF
            and record['source_files'] == bank.stage.hashed(source) and record['recipe'] == recipe,
            'preserve and recover the measured reference build')
    require(FIXTURE not in source and all(n in source for n in PROJECTED if n != FIXTURE), 'unexpected projection base')
    patch = subprocess.check_output(
        ['git', 'diff', BASE, SOURCE, '--', *PROJECTED], cwd=bank.P, timeout=30)
    with tempfile.TemporaryDirectory(prefix='decode-combine-projection-') as temp:
        directory = Path(temp)
        for name in PROJECTED:
            if name == FIXTURE:
                continue
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source[name])
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', '-'],
                       cwd=directory, input=patch, check=True, timeout=30)
        require({p.relative_to(directory).as_posix(): bank.sha(p) for p in directory.rglob('*') if p.is_file()}
                == PROJECTED, 'changed candidate projection')
        source.update({n: (directory / n).read_bytes() for n in PROJECTED})
    old, new = str(stock.MODEL_ROOT), str(ROOT)
    recipe = dict(configure=[arg.replace(old, new) for arg in recipe['configure']],
                  environment={key: val.replace(old, new) for key, val in recipe['environment'].items()},
                  targets=[*recipe['targets'], TARGET])
    return native, foundation, source, recipe, dict(parent_binary=parent, parent_source=record['native_source_commit'],
        parent_record=bank.sha(stock.MODEL_ROOT / 'record.json'), projected=PROJECTED)


def check(row, committed=True):
    if committed:
        require(bank.basis.validation.committed(bank.RESULT, (KEY,)) == builder.compact(row), 'commit previous build stage')
    native, foundation, source, recipe, proof = project(row['foundation'])
    require(row['controller_sources'] == bank.stage.source.pin(bank.P, list(row['controller_sources']))['source_sha256']
            and row['native_source_commit'] == SOURCE and row['source_files'] == bank.stage.hashed(source)
            and row['foundation'] == foundation and row['projection'] == proof and row['recipe'] == recipe
            and row['driver'] == builder.guards.driver_guard() and row['external_inputs'] == native['external_inputs']
            and row['retained_decode_reference'] == builder.decode_reference(),
            'compiler, reference, native or controller identity changed')
    compiler = next(a.split('=', 1)[1] for a in recipe['configure'] if a.startswith('-DCMAKE_CXX_COMPILER='))
    external = row['external_inputs']
    require(bank.sha(external['path']) == external['sha256']
            and bank.previous.read(Path(external['path']))[compiler] == bank.sha(compiler)
            and all(bank.sha(p) == h for p, h in row['build_tools'].items()), 'build tools changed')
    materialized = ROOT / 'source'
    require({p.relative_to(materialized).as_posix() for p in materialized.rglob('*') if p.is_file()}
            == row['source_files'].keys(), 'unexpected file in the materialized build')
    require(all(bank.sha(ROOT / 'source' / n) == h for n, h in row['source_files'].items())
            and all(bank.sha(Path(dep['root']) / n) == h
                    for dep in native['dependencies'] for n, h in dep['source_sha256'].items())
            and all(bank.sha(n) == h for n, h in row.get('configured_files', {}).items()),
            'compiled source, dependencies or configured build input changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        require(not actual['partial_archive_reuse'] and actual['uncovered'] == 0
                and all(str(ROOT / 'source' / n) in actual['compiled'] for n in UNITS)
                and row['binaries'].keys() == set(recipe['targets'])
                and all(bank.sha(f['path']) == f['sha256'] for f in row['binaries'].values())
                and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in actual[group].items()),
                'fresh compiler dependencies or final executable identity missing')
    return native


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        files = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        files.extend((build / 'CMakeFiles').glob('*.ninja'))
        row['configured_files'] = {str(path): bank.sha(path) for path in files}
    bank.stage.save_build(row)


def prepare():
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT),
            'inspect the existing build before creating a candidate')
    controllers = [str(Path(m).resolve().relative_to(bank.P)) for m in
                   (__file__, stock.__file__, stock.fixture.__file__, stock.kv.__file__)]
    pins = bank.stage.source.pin(bank.P, [*controllers, 'tools/sycl', *bank.basis.POLICIES])
    native, foundation, source, recipe, proof = project()
    row = dict(stage='prepared', source_commit=pins['revision'], controller_sources=pins['source_sha256'],
               native_source_commit=SOURCE, source_files=bank.stage.hashed(source),
               foundation=foundation, projection=proof, recipe=recipe, external_inputs=native['external_inputs'],
               build_tools={p: bank.sha(p) for p in ('/usr/bin/cmake', '/usr/bin/ninja')},
               driver=builder.guards.driver_guard(), retained_decode_reference=builder.decode_reference(recover=True),
               commands=[], active=False, passed=False, complete=False, ownership_closed=True,
               component_plan=dict(order=MODES, hits=[4, 10, 20], pairs_per_process=4, replays=512,
                   minimum_ms=10, max_empty_clock_ratio=0.01, cpu_publication='Rewrite only misses after each drain.',
                   result='Candidate median faster both from launch and from publication at all three '
                   'hit counts in every process. '
                   'Independent capture and service comparison only.'),
               retention='Third build only during the same-kernel fixture and a gated coherent model comparison. '
                         'Keep the measured fast prefill and independent decode references. Retire this build '
                         'after recovering its source and recording a failed fixture or a completed comparison.')
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'tmp', 'cache', 'owned'):
        (ROOT / name).mkdir(mode=0o700)
    for name, content in source.items():
        path = ROOT / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    check(row, False)
    save(row)


def parse_pairs(text, reverse):
    normal = 'decode MoE combine numerical=PASS guards=PASS nodes=3/1 completed=1'
    lines = text.strip().splitlines()
    require(lines and lines[-1] == normal and lines.count(normal) == 1, 'fixture did not close normally')
    pattern = (r'decode MoE combine hits=(4|10|20) batch=([0-3]) arm=([01]) replays=512 '
               r'ms=(\d+\.\d+) full_ms=(\d+\.\d+) empty_ms=(\d+\.\d+)')
    samples = []
    for line in lines[:-1]:
        value = re.fullmatch(pattern, line)
        require(value is not None, 'unexpected timing row')
        h, n, a, ms, full, empty = value.groups()
        samples.append((int(h), int(n), int(a), float(ms), float(full), float(empty)))
    expected = [(h, n, a) for h in (4, 10, 20) for n in range(4)
                for a in (int(reverse) ^ (n % 2), 1 ^ int(reverse) ^ (n % 2))]
    require([s[:3] for s in samples] == expected and all(full >= ms >= 10 and empty <= ms * 0.01
                for _, _, _, ms, full, empty in samples), 'incomplete timing pairs or empty-clock limit')
    return samples


def component():
    row = bank.previous.read(ROOT / 'record.json')
    native = check(row)
    require(row['stage'] == 'built' and row['passed'] and row['complete'] and row['ownership_closed']
            and not row['active'] and COMPONENT not in bank.basis.validation.committed(bank.RESULT),
            'component requires one fresh owned comparison after the coherent build')
    probe = native['native_validation_probe']
    runtime = row['foundation']['runtime_manifest']
    fixture = row['binaries'][TARGET]
    root = ROOT / 'component'
    root.mkdir(mode=0o700)
    result = dict(stage='running', source_commit=row['source_commit'], native_source_commit=SOURCE, binary=fixture,
                  full_engine=row['binaries']['strata'], plan=row['component_plan'], processes=[],
                  passed=False, complete=False, ownership_closed=True, active=False)
    try:
        for i, mode in enumerate(MODES):
            case = root / str(i)
            case.mkdir(mode=0o700)
            for name in ('tmp', 'cache'):
                (case / name).mkdir(mode=0o700)
            env = {**row['recipe']['environment'], 'TMPDIR': str(case / 'tmp'), 'XDG_CACHE_HOME': str(case / 'cache'),
                   'LD_PRELOAD': probe['library']['path'], 'STRATA_UR_PROBE_EXPECTED_LOADER': probe['loader']['path'],
                   'LD_DEBUG': 'libs', 'UR_LOG_VALIDATION': 'level:warning;flush:warning;output:file,' + str(case / 'validation.log')}
            env.pop('UR_LOG_TRACING', None)
            env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
            if i == 0:
                env['UR_ENABLE_LAYERS'] = 'UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION'
                env['UR_LOG_TRACING'] = 'level:info;flush:info;output:stderr'
            require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['NEOReadDebugKeys'] == '1'
                    and env['EnableDirectSubmission'] == '0'
                    and bank.sha(fixture['path']) == fixture['sha256'], 'fixture or runtime safety settings changed')
            args = [fixture['path'], mode]
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            cursor, = (line.removeprefix('-- cursor: ') for line in subprocess.check_output(
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
                text=True, timeout=15).splitlines() if line.startswith('-- cursor: '))
            owner = builder.ownership.Owner(case / 'owned', lambda: bank.stage.write(case / 'ownership.json', owner.commands))
            outcome = dict(mode=mode, command=args, environment_sha256=bank.signature(env), budgets=DEVICE_LIMITS)
            try:
                entry, out, err = owner.run(mode, args, env, bank.P, **DEVICE_LIMITS)
                require(builder.ownership.closed(entry) and entry['exit_code'] == 0, 'component process failed')
                proof = bank.basis.validation.preload_evidence(err.read_text(), entry['pid'])
                require(proof['loader'] == probe['loader']['path'], 'fixture validation loader changed')
                warning = case / 'validation.log'
                require(not warning.exists() or warning.stat().st_size == 0, 'parameter validation warning')
                providers = bank.device.loaded_providers(err.read_text(), dict(
                    environment=env, native_validation_probe=probe, runtime_manifest=runtime))
                if i == 0:
                    require('urEnqueue' in err.read_text() and out.read_text().strip() ==
                            'decode MoE combine numerical=PASS guards=PASS nodes=3/1 completed=1', 'incomplete traced validation')
                else:
                    outcome['samples'] = parse_pairs(out.read_text(), mode == 'time-reverse')
                leader, = [o for o in entry['owners'] if o['pid'] == entry['pid']]
                outcome.update(host_probe_accepted=True, native_exit_code=0, native_identity=leader,
                               validation=proof, providers=providers)
            finally:
                bank.basis.validation.finish(outcome, owner, cursor, boot)
                outcome['passed'] &= outcome.get('driver_after') == row['driver']
                result['processes'].append(outcome)
            require(outcome['passed'], 'component execution or final device checks failed')
        by_hits = {}
        require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in result['processes']}) == len(MODES), 'independent processes required')
        for hits in (4, 10, 20):
            intervals = {}
            for i, name in enumerate(('graph', 'with_publication'), start=3):
                medians = [[statistics.median(s[i] for s in p['samples'] if s[0] == hits and s[2] == a)
                            for a in (0, 1)] for p in result['processes'][1:]]
                intervals[name] = [candidate / reference for reference, candidate in medians]
            by_hits[str(hits)] = intervals
        result.update(stage='closed', passed=True, complete=True, candidate_over_control=by_hits,
                      advance_to_model=all(r < 1 for intervals in by_hits.values()
                                           for ratios in intervals.values() for r in ratios))
    except BaseException as error:
        result.update(stage='failed', error=repr(error))
        raise
    finally:
        result['ownership_closed'] = all(p['ownership_closed'] for p in result['processes'])
        data = bank.previous.read(bank.RESULT)
        data[COMPONENT] = result
        bank.stage.write(bank.RESULT, data)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'configure', 'compile', 'component'), 'select prepare, configure, compile or component')
    builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, [*stock.bank.TARGETS, stock.fixture.TARGET, TARGET]
    builder.check, builder.save = check, save
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        elif action == 'component':
            component()
        else:
            builder.execute(action)
