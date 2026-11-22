"""Qualify the existing mapped GU input witness on the measured B570 runtime."""

import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_mixed_gather_measure as reference
from prefill_attention_bank_component import dependency_paths

bank, validation = reference.bank, reference.validation
owner, require = reference.owner, reference.require
P, B, RESULT = bank.P, bank.B, bank.RESULT
ROOT, KEY = B / 'prefill-mapped-gu-a-v2', 'prefill_mapped_gu_a_v2'
PARENT = reference.coherent_model.MODEL_ROOT
SRC, BUILD = PARENT / 'source', PARENT / 'build'
PROBE_SRC = ROOT / 'probe-source'
THIS = str(Path(__file__).resolve().relative_to(P))
PROBE = '19999c6c1df5ce11f12478958904d602c5eff9ac'
KERNEL = 'src/prefill/sycl/vllm_xe2/native16.cpp'
HEADER = 'src/prefill/sycl/vllm_xe2/native16.hpp'
FIXTURE = 'bench/sycl/moe_gu_swiglu.cpp'
TARGET = 'sycl_moe_gu_swiglu_bench'
IMAGE, HOST = 'strata_vllm_xe2_image.o', 'strata_vllm_xe2_host.o'
DEFINE = '-DSTRATA_PREFILL_GU_A_PROBE'
STAGES = ('layout', 'smoke', 'validate')
ROWS = {'layout': (1, 3, 16, 17, 128, 129, 211, 2334, 6640, 8191, 8192),
        'smoke': (3,), 'validate': (16, 17, 128, 129, 211, 2334)}
BOOT = Path('/proc/sys/kernel/random/boot_id')
BUDGET = dict(wall=480, cpu=840, rss_cap=12 << 30, text_cap=16 << 20,
              file_cap=64 << 20, total_cap=64 << 20)
sha, signature, write = bank.sha, bank.signature, bank.stage.write


def recipe():
    """Keep the original BMG device-finalization, host slice and final link modes."""
    entries = json.loads((BUILD / 'compile_commands.json').read_bytes())
    commands, outputs = {}, {}
    for label, source in (('kernel', KERNEL), ('fixture', FIXTURE)):
        entry, = (x for x in entries if x['file'] == str(SRC / source))
        cmd = shlex.split(entry['command'])
        require(Path(entry['directory']) == BUILD and '-fsycl' in cmd and DEFINE not in cmd and
                cmd[-4:] == ['-o', entry['output'], '-c', entry['file']], 'changed compile rule for ' + source)
        outputs[label] = entry['output']
        # The copied translation unit still needs its original sibling headers.
        commands[label] = [cmd[0], DEFINE, '-iquote', str((SRC / source).parent),
                           '-I' + str((PROBE_SRC / HEADER).parent), *cmd[1:-4],
                           '-MD', '-MF', str(ROOT / (label + '.d')), '-o', str(ROOT / (label + '.o')),
                           '-c', str(PROBE_SRC / source)]

    lines = subprocess.check_output(['/usr/bin/ninja', '-t', 'commands', TARGET], cwd=BUILD,
                                    text=True, timeout=25).splitlines()
    raw, = (line for line in lines if ' -fsycl-link ' in line and ' -unbundle ' in line and IMAGE in line)
    words = shlex.split(raw)
    require(words[:3] == ['cd', str(BUILD), '&&'] and words.count('&&') == 2,
            'changed native image/host rule')
    split = words.index('&&', 3)
    aot, unbundle = words[3:split], words[split + 1:]
    old_kernel = (BUILD / outputs['kernel']).resolve()
    require(Path(aot[-3]).resolve() == old_kernel and aot[-2:] == ['-o', str(BUILD / IMAGE)],
            'unexpected device registration inputs')
    commands['image'] = [*aot[:-3], str(ROOT / 'kernel.o'), '-o', str(ROOT / IMAGE)]
    bundle_input, = (i for i, word in enumerate(unbundle) if word.startswith('-input='))
    bundle_output, = (i for i, word in enumerate(unbundle) if word.startswith('-output='))
    require(Path(unbundle[bundle_input].removeprefix('-input=')).resolve() == old_kernel
            and unbundle[bundle_output] == '-output=' + str(BUILD / HOST), 'unexpected host slice inputs')
    unbundle[bundle_input] = '-input=' + str(ROOT / 'kernel.o')
    unbundle[bundle_output] = '-output=' + str(ROOT / HOST)
    commands['host-slice'] = unbundle

    words = shlex.split(lines[-1])
    require(words[:2] == [':', '&&'] and words[-2:] == ['&&', ':'] and words.count(TARGET) == 1 and
            words[words.index('-o') + 1] == TARGET and '-Wl,-t' in words, 'checked fixture link rule required')
    changed = {outputs['fixture']: str(ROOT / 'fixture.o'), TARGET: str(ROOT / TARGET),
               IMAGE: str(ROOT / IMAGE), HOST: str(ROOT / HOST)}
    require(all(words.count(old) == 1 for old in changed), 'fixture or native registration missing from final link')
    commands['link'] = [words[2], '-v', *(changed.get(v, v) for v in words[3:-2])]
    return commands


def probe_overlay(native):
    guard = rb'^#ifdef STRATA_PREFILL_GU_A_PROBE\n.*?^#endif\n'
    overlay = {}
    for name in (KERNEL, HEADER, FIXTURE):
        original = reference.fixture.git_file(PROBE + '^', name)
        candidate = reference.fixture.git_file(PROBE, name)
        require(original == (SRC / name).read_bytes() and sha(SRC / name) == native['source_files'][name],
                'retained source must match the probe parent at ' + name)
        stripped, count = re.subn(guard, b'', candidate, flags=re.M | re.S)
        require(count == 2 and list(filter(bytes.strip, stripped.splitlines())) ==
                list(filter(bytes.strip, original.splitlines())), 'unrelated native source change at ' + name)
        overlay[name] = candidate
    return overlay


def load_inputs():
    native, speed, _, build = reference.coherent_inputs()
    timing = validation.committed(RESULT, ('prefill_kv_gather_speed_65537_v1',))
    require(timing['stage'] == 'closed' and timing['complete'] and timing['passed'] and timing['ownership_closed']
            and not timing['active'] and timing['binary'] == native['binaries']['strata'] and
            timing['native_source_commit'] == native['native_source_commit'] and
            all(timing['decision']['contrast_passes'][part] for part in ('both/base', 'both/reuse', 'both/vec8')),
            'same native 65K selection required')
    overlay = probe_overlay(native)
    sources = {n: hashlib.sha256(raw).hexdigest() for n, raw in overlay.items()}
    allowed = {p: h for group in ('compiled', 'generated', 'links', 'objects', 'produced')
               for p, h in native['actual_inputs'][group].items()}
    allowed.update({str(SRC / p): h for p, h in native['source_files'].items()})
    allowed.update({str(PROBE_SRC / p): h for p, h in sources.items()})
    external = native['external_inputs']
    require(sha(external['path']) == external['sha256'], 'qualified build tool inputs changed')
    allowed.update(json.loads(Path(external['path']).read_bytes()))
    ninja = Path('/usr/bin/ninja').resolve()
    recipe_reader = dict(path=str(ninja), sha256=sha(ninja))
    commands = recipe()
    for cmd in commands.values():
        program = Path(cmd[0]).resolve()
        require(allowed.get(str(program), allowed.get(cmd[0])) == sha(program), 'new compiler or linker tool')
    seed, probe = speed['environment_template'], speed['native_validation_probe']
    env = {k: v.replace(seed['STRATA_MEASUREMENT_ROOT'], str(ROOT)) for k, v in seed.items()
           if k not in ('STRATA_TRACE', 'UR_LOG_TRACING', 'LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER')}
    build_env = dict(native['recipe']['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    require(env.get('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD') == '1' and env.get('EnableDirectSubmission') == '0'
            and speed['boot'] == BOOT.read_text().strip() and all(sha(a['path']) == a['sha256'] for a in probe.values()),
            'runtime or qualified validation changed')
    proof = dict(engine=native['binaries']['strata'], parent=build['build_sha256'],
                 native_source_commit=native['native_source_commit'], probe_source=PROBE, probe_inputs=sources,
                 recipe_reader=recipe_reader, boot=speed['boot'], driver=native['driver'], validation=probe,
                 runtime_manifest=native['foundation']['runtime_manifest'])
    return proof, allowed, commands, build_env, env, overlay


def controls(revision=None):
    scripts = [str(Path(m.__file__).resolve().relative_to(P)) for m in
               (reference, reference.fixture, reference.coherent_model, reference.coherent_model.kv)]
    pins = bank.stage.source.pin(P, [THIS, *scripts, 'tools/sycl'])
    require({THIS, *scripts} <= pins['source_sha256'].keys(), 'commit the complete controller first')
    sources = pins['source_sha256']
    if revision is not None:
        historical = {p: hashlib.sha256(raw).hexdigest()
                      for p, raw in bank.stage.source.blobs(P, revision, sources)}
        require(historical == sources, 'original controller inputs changed')
    return dict(revision=revision or pins['revision'], sha256=signature(sources))


def compact(row):
    return {name: val for name, val in row.items() if name not in ('build_env', 'runtime_env')}


def save(row):
    write(ROOT / 'record.json', row)
    public = bank.previous.read(RESULT)
    public[KEY] = compact(row)
    write(RESULT, public)


def prepare(inputs):
    proof, _, commands, build_env, runtime_env, overlay = inputs
    require(not ROOT.exists() and KEY not in validation.committed(RESULT), 'fresh mapped-A fixture only')
    row = dict(stage='prepared', controller=controls(), proof=proof, commands=commands,
        build_env=build_env, runtime_env=runtime_env, environment_sha256=signature((build_env, runtime_env)),
        stages=[], active=False, complete=False, passed=False, ownership_closed=True,
        plan=dict(build='Apply only the guarded probe code to the recorded parent sources and enable its '
                        'single macro for the probe object and fixture. Resolve sibling headers from the '
                        'original source directory. Finalize the '
                        'isolated BMG image and host slice and relink against the original checked archives.',
                  validation='Host lane layout, traced three-row smoke, then full exact bank-to-fragment '
                             'bits across both tile policies, tails and malformed maps. No model or timing.',
                  retention='Only the witness ELF is a new fixture. Review its live consumer at closure; '
                            'retain the primary prefill and decode engines.'), budgets=BUDGET)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, raw in overlay.items():
        path = PROBE_SRC / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(raw)
    save(row)


def check(row, inputs):
    proof, _, commands, build_env, runtime_env, overlay = inputs
    require(validation.committed(RESULT, (KEY,)) == compact(row) and row['controller'] == controls(row['controller']['revision']) and
            row['proof'] == proof and row['commands'] == commands and row['build_env'] == build_env and
            row['runtime_env'] == runtime_env and row['environment_sha256'] == signature((build_env, runtime_env))
            and row['budgets'] == BUDGET and not row['active'] and row['ownership_closed'],
            'commit exact prepared result and keep fixture inputs pinned')
    require(all((PROBE_SRC / p).read_bytes() == raw for p, raw in overlay.items()), 'private probe source changed')


def check_link(text, allowed, generated, host_slices=()):
    observed = {}
    slices = set()
    for line in text.splitlines():
        token = line.strip()
        if not token or len(token) > 4096 or any(c.isspace() for c in token):
            continue
        path = Path(token)
        if not path.is_absolute() and path.suffix not in ('.a', '.o', '.so'):
            continue
        path = (BUILD / path).resolve()
        if path.is_file():
            expected = generated.get(str(path), allowed.get(str(path)))
            require(expected == sha(path), 'unexpected link input ' + str(path))
            observed[str(path)] = expected
        else:
            require(path.is_relative_to(ROOT / 'tmp') and path.suffix == '.o', 'unowned link temporary')
            if str(path) in host_slices:
                slices.add(str(path))
    require(set(host_slices) == slices and {str(ROOT / IMAGE), str(ROOT / HOST)} <= observed.keys(),
            'incomplete actual AOT/host link inputs')
    return observed


def compiled_inputs(allowed, label, source):
    paths = dependency_paths((ROOT / (label + '.d')).read_text())
    required = {str(PROBE_SRC / HEADER), str(PROBE_SRC / source)}
    if label == 'kernel':
        required.add(str((SRC / KERNEL).with_name('xe_gemm_bsd.hpp')))
    require(required <= paths, 'actual probe translation unit or header missing')
    require(all(sha(p) == allowed.get(p) for p in paths), 'changed actual compiler input')
    return dict(count=len(paths), sha256=signature({p: allowed[p] for p in paths}))


def build(row, allowed):
    require(row['stage'] == 'prepared', 'inspect the existing build attempt')
    output = ROOT / 'compile'
    job = owner.Owner(output, lambda: write(output / 'execution.json', job.commands))
    outputs, dependencies = {}, {}
    try:
        for label, product in (('kernel', 'kernel.o'), ('fixture', 'fixture.o'), ('image', IMAGE),
                               ('host-slice', HOST), ('link', TARGET)):
            entry, out, err = job.run(label, row['commands'][label], row['build_env'], BUILD, **BUDGET)
            require(owner.closed(entry) and entry['exit_code'] == 0, label + ' failed: ' + err.read_text()[-900:])
            if label in ('kernel', 'fixture'):
                dependencies[label] = compiled_inputs(allowed, label, KERNEL if label == 'kernel' else FIXTURE)
            if label == 'link':
                slices = reference.fixture.fixture_host_slices(err.read_text(), ROOT / 'fixture.o')
                links = check_link(out.read_text(), allowed, outputs, slices)
                archives = {str((BUILD / v).resolve()) for v in row['commands']['link'] if v.endswith('.a')}
                require(archives and archives <= links.keys(), 'consumed runtime/engine archives missing')
            outputs[str(ROOT / product)] = sha(ROOT / product)
        row.update(stage='built', binary=dict(path=str(ROOT / TARGET), sha256=outputs[str(ROOT / TARGET)]),
                   compiler_inputs=dependencies, link_inputs=links, produced=outputs)
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        raise
    finally:
        row.update(active=bool(job.active), ownership_closed=not job.active and all(owner.closed(c) for c in job.commands))
        save(row)


def read_case(text, stage):
    lines = text.splitlines()
    mode = '--prefill-mapped-a-' + stage
    end = 'fixture_status=PASS mode=' + mode
    expected, words = [], 0
    for tile, block, replicas in (('compact', 16, 4), ('wide', 128, 2)):
        for n in ROWS[stage]:
            valid = n * 96 * replicas
            all_words = ((n + block - 1) // block) * block * 96 * replicas
            if stage == 'layout':
                expected.append(f'prefill_gu_a_layout rows={n} tile={tile} valid={valid} masked={all_words - valid}')
            else:
                for bad in ((0,) if stage == 'smoke' else (0, 1)):
                    expected.append(f'prefill_gu_a rows={n} tile={tile} invalid={bad} checked_words={2 * all_words} '
                                    'block2d_reorder=exact mapped=exact coverage=full guards=PASS')
                    words += 2 * all_words
    if stage == 'layout':
        end += ' GPU_submissions=0'
    else:
        require('{"kind":"topology","work_items":384,"status":"PASS"}' in lines, 'missing subgroup proof')
    require([line for line in lines if line.startswith(('prefill_gu_a', 'fixture_status='))] == [*expected, end],
            'case, lane coverage or native completion differs')
    return dict(completed_cases=len(expected), compared_device_words=words)


def run(row, stage, allowed):
    require(row['stage'] in ('built', 'checked') and STAGES[len(row['stages'])] == stage and
            all(s['passed'] for s in row['stages']) and
            row['compiler_inputs'] == {k: compiled_inputs(allowed, k, s) for k, s in
                                       (('kernel', KERNEL), ('fixture', FIXTURE))} and
            all(sha(path) == h for path, h in (row['produced'] | row['link_inputs']).items()),
            'closed matching build and exact prior stages required')
    root = ROOT / stage
    root.mkdir(mode=0o700)
    for d in ('tmp', 'cache'):
        (root / d).mkdir(mode=0o700)
    env = {k: v.replace(str(ROOT), str(root)) for k, v in row['runtime_env'].items()}
    probe = row['proof']['validation']
    if stage != 'layout':
        env.update(LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
                   UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    if stage == 'smoke':
        env.update(LD_DEBUG='libs', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_TRACING='level:info;flush:info;output:stderr')
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    job = owner.Owner(root / 'owned', lambda: write(root / 'execution.json', job.commands))
    sample = dict(stage=stage, environment_sha256=signature(env), gpu=stage != 'layout')
    try:
        entry, out, err = job.run(stage, [row['binary']['path'], '--prefill-mapped-a-' + stage], env, P, **BUDGET)
        require(owner.closed(entry) and entry['exit_code'] == 0, 'mapped input witness failed: ' + err.read_text()[-700:])
        sample.update(read_case(out.read_text(), stage))
        log = err.read_text()
        if stage != 'layout':
            sample['validation'] = validation.preload_evidence(log, entry['pid'])
            v = root / 'validation.log'
            require((not v.exists() or not v.stat().st_size) and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_', log),
                    'unexpected runtime validation message')
        if stage == 'smoke':
            require('urEnqueueKernelLaunch' in log, 'missing first-use kernel trace')
            sample['providers'] = bank.device.loaded_providers(log, dict(environment=env, native_validation_probe=probe,
                                                       runtime_manifest=row['proof']['runtime_manifest']))
        sample.update(native_exit_code=0, host_probe_accepted=True)
    except BaseException as error:
        sample['error'] = repr(error)
        raise
    finally:
        validation.finish(sample, job, cursor, row['proof']['boot'])
        sample['passed'] &= sample.get('driver_after') == row['proof']['driver']
        row['stages'].append(sample)
        closed = sample['passed'] and stage == STAGES[-1]
        row.update(stage=('closed' if closed else 'checked') if sample['passed'] else 'failed', passed=closed,
                   complete=closed, active=bool(job.active), ownership_closed=sample['ownership_closed'])
        save(row)
    require(sample['passed'], 'mapped input witness postcondition failed')


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', 'build', *STAGES),
            'select a prepared witness stage and qualified interpreter')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        inputs = load_inputs()
        if action == 'prepare':
            prepare(inputs)
        else:
            row = bank.previous.read(ROOT / 'record.json')
            check(row, inputs)
            if action == 'build':
                build(row, inputs[1])
            else:
                run(row, action, inputs[1])
