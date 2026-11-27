#!/usr/bin/env python3
"""Pin and build the standalone canonical two-column Q6 comparison fixture."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import build_q6_c5_component as existing

base, P, B, RESULT = existing.base, existing.P, existing.B, existing.RESULT
require, sha = existing.require, existing.sha
sys.path.insert(0, str(P / 'bench'))
import dflash2_decode_t4_build as source_build

ROOT = B / 'q6-t4-c2-component-v1'
KEY = 'q6_t4_c2_component_build_v1'
BASE_BUILD = '2dac4cbf2'
BASE_DEVICE = '460628c2a'
FOUNDATION = B / 'coherent-prefill-native-v1' / 'record.json'
THIS = str(Path(__file__).resolve().relative_to(P))
PREFIXES = ('CMakeLists.txt', 'cmake', 'src', 'include', 'third_party', 'bench/sycl',
            'tools/sycl', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md', source_build.THIS,
            source_build.fixture.THIS, source_build.retained.THIS)
CONTROL = ('--host-split-c2', 'HOST_SPLIT_C2 passed=1 cases=5 GPU=0')
ARC_GATE = 'dflash2_decode_t4_shared_plan_v2'
SHAPES = ((512, 33), (6144, 2560), (2560, 6144), (2560, 10240), (2560, 12288))
SUMMARY = 'SPLIT_C2_CHECK passed=1 cases=10 bitwise=1 guards=1 first_half=1'
ARC_LIMITS = dict(wall=180, cpu=600, rss_cap=4 << 30, text_cap=24 << 20,
                  file_cap=32 << 20, total_cap=64 << 20)


def reference():
    bank = source_build.bank
    receipt = source_build.ROOT / 'record.json'
    native = bank.previous.read(receipt)
    archived = bank.previous.historic(BASE_BUILD, source_build.KEY)
    require(source_build.retained.closed(native, 'built')
            and source_build.builder.compact(native) == archived, 'recover the committed T4 recipe and native identity')
    require(sha(FOUNDATION) == native['foundation']['original_record_sha256'],
            'the original compiler foundation changed')
    arc = bank.previous.historic(BASE_DEVICE, ARC_GATE)
    require(source_build.retained.closed(arc, 'closed')
            and arc['native_build_sha256'] == bank.signature(archived)
            and arc['driver'] == native['driver'] and arc['runtime_manifest'] == native['runtime_manifest'],
            'the validated T4 runtime and recipe differ')
    runtime = {path: digest for path, digest in native['source_files'].items()
               if path.startswith(('src/sycl/', 'include/strata/sycl/'))}
    require(runtime and all(sha(P / path) == digest for path, digest in runtime.items()),
            'the component must reuse the verified SYCL runtime source')
    identity = dict(record=dict(path=str(receipt), sha256=sha(receipt)),
                    build_commit=BASE_BUILD, source_commit=native['native_source_commit'],
                    reference_binary=native['binaries']['strata'], runtime_sources=runtime,
                    device_commit=BASE_DEVICE, runtime_manifest=arc['runtime_manifest'],
                    driver=arc['driver'], boot=arc['boot'], validation_probe=arc['native_validation_probe'])
    return native, bank.previous.read(FOUNDATION), identity


def numerical_plan(ctx):
    arc = source_build.bank.previous.historic(BASE_DEVICE, ARC_GATE)
    legacy_root = str(B / ARC_GATE.replace('_', '-'))
    env = {key: value.replace(legacy_root, str(ROOT)) for key, value in arc['environment'].items()}
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['NEOReadDebugKeys'] == '1'
            and env['EnableDirectSubmission'] == '0' and env['ZES_ENABLE_SYSMAN'] == '0',
            'preserve the verified Arc safety settings')
    return dict(command=[str(ROOT / 'build' / ctx.TARGET), '--check-split-c2'],
                environment=env, limits=ARC_LIMITS, shapes=[list(s) for s in SHAPES], summary=SUMMARY)


def context():
    ctx = existing.context()
    ctx.ROOT, ctx.KEY = ROOT, KEY

    def current(row, committed=True):
        if committed:
            proof = source_build.validation.committed(RESULT, (KEY,))
            require(ctx.compact(row) == proof, 'commit the exact C2 component stage')
        _, _, identity = reference()
        require(row['selected_native'] == identity and {*ctx.UNITS, THIS, existing.THIS}
                <= row['project']['source_sha256'].keys(), 'C2 build provenance or inputs changed')
        ctx.build.check_inputs(row)
        require(row['recipe']['targets'] == row['recipe']['build_targets'] == [ctx.TARGET]
                and not row['recipe']['environment'].get('LD_PRELOAD')
                and row['plan']['arc'] == numerical_plan(ctx), 'use the frozen standalone Q6 component plan')
        if 'binary' in row:
            binary = row['binary']
            require(Path(binary['path']).stat().st_size == binary['bytes']
                    and sha(binary['path']) == binary['sha256'], 'C2 fixture executable changed')
            actual = row['actual_inputs']
            require(sha(actual['path']) == actual['sha256'], 'the actual C2 compiler input census changed')
            inputs = json.loads(Path(actual['path']).read_text())
            require(all(sha(path) == digest for kind in ('compiled', 'generated', 'links')
                        for path, digest in inputs[kind].items()), 'an actual C2 build input changed')

    ctx.current = current
    return ctx


def prepare(ctx):
    require(not ROOT.exists() and KEY not in source_build.validation.committed(RESULT),
            'prepare one fresh C2 component build')
    native, compiler, identity = reference()
    previous_root = str(source_build.ROOT)
    args = [value.replace(previous_root + '/source', str(P)).replace(previous_root, str(ROOT))
            for value in native['recipe']['configure']]
    require(args[args.index('-S') + 1] == str(P) and args[args.index('-B') + 1] == str(ROOT / 'build'),
            'separate fixture and engine build paths required')
    recipe = dict(configure=args, environment={k: v.replace(previous_root, str(ROOT))
                                               for k, v in native['recipe']['environment'].items()},
                  targets=[ctx.TARGET], build_targets=[ctx.TARGET])
    row = dict(stage='prepared', active=False, passed=False, complete=False, ownership_closed=True,
               commands=[], fallbacks={}, selected_native=identity, project=ctx.source.pin(P, PREFIXES),
               dependencies=compiler['dependencies'], toolchain=compiler['toolchain'],
               external_inputs=native['external_inputs'], recipe=recipe, budgets=ctx.BUDGET,
               host_budget=dict(wall=120, cpu=120, rss_cap=1 << 30, text_cap=1 << 20,
                                file_cap=2 << 20, total_cap=8 << 20),
               plan=dict(compile_target=ctx.TARGET, initial_native_checks=['--host', '--host-c5', CONTROL[0]],
                         scope='Build the fixture only with the retained production DPC++ recipe.',
                         arc=numerical_plan(ctx),
                         next='Commit the same-build Arc bitwise and guard check before timing either control.'),
               retention='One standalone fixture through the C2 component decision. No serving engine build.')
    ctx.current(row, committed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    ctx.save(row)


def check(ctx):
    bank, validation = source_build.bank, source_build.validation
    row = bank.previous.read(ROOT / 'record.json')
    ctx.current(row)
    plan, identity = row['plan']['arc'], row['selected_native']
    require(row['stage'] == 'built' and row['passed'] and row['ownership_closed']
            and row['host']['passed'] and row['host']['ownership_closed']
            and plan['command'][0] == row['binary']['path']
            and 'numerical' not in row and not (ROOT / 'owned-arc').exists(),
            'require the same-build host outcomes and a fresh Arc check')
    runtime = identity['runtime_manifest']
    pins = bank.previous.read(Path(runtime['path']))
    require(sha(runtime['path']) == runtime['sha256']
            and len(pins) == runtime['files'] and all(sha(path) == digest for path, digest in pins.items())
            and all(sha(v['path']) == v['sha256'] for v in identity['validation_probe'].values())
            and base.feature.guards.driver_guard() == identity['driver']
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == identity['boot'],
            'the qualified C2 runtime, probe or driver changed')
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    owner = base.ownership.Owner(ROOT / 'owned-arc', lambda: ctx.write(
        ROOT / 'arc-execution.json', dict(active=bool(owner.active), commands=owner.commands)))
    outcome = dict(host_probe_accepted=False)
    try:
        entry, out, err = owner.run('check-split-c2', plan['command'], plan['environment'], P, **plan['limits'])
        outcome['native_exit_code'] = entry.get('exit_code')
        require(base.ownership.closed(entry) and outcome['native_exit_code'] == 0,
                'the Arc C2 fixture did not complete')
        lines = out.read_text().splitlines()
        admissions = [dict(p.split('=', 1) for p in line.split()[1:]) for line in lines if line.startswith('ADMISSION ')]
        require([[int(item['k']), int(item['n'])] for item in admissions] == plan['shapes']
                and all(item['graph_counts'] == '1/2' and item['reference'] == 'canonical_c4'
                        and item['additional_candidate_weight_bytes'] == '0' for item in admissions)
                and lines.count(plan['summary']) == 1, 'C2 numerical, graph, or shape outcome differs')
        log = err.read_text()
        proof = validation.preload_evidence(log, entry['pid'])
        warning = ROOT / 'validation.log'
        require(proof['loader'] == identity['validation_probe']['loader']['path'] and 'urEnqueue' in log
                and (not warning.exists() or warning.stat().st_size == 0), 'C2 runtime validation differs')
        providers = bank.device.loaded_providers(log, dict(runtime_manifest=runtime,
            native_validation_probe=identity['validation_probe'], environment=plan['environment']))
        outcome.update(host_probe_accepted=True, summary=SUMMARY, shapes=plan['shapes'],
                       validation=proof, providers=providers)
    except Exception as error:
        outcome['error'] = repr(error)
    finally:
        validation.finish(outcome, owner, cursor, identity['boot'])
        outcome['passed'] &= outcome.get('driver_after') == identity['driver']
        row['numerical'] = outcome
        ctx.save(row)
    require(outcome['passed'], outcome.get('error', 'C2 numerical or ownership gate failed'))


if __name__ == '__main__':
    require(sys.executable == source_build.bank.basis.reference.model.PYTHON
            and len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'host', 'check'),
            'use the pinned interpreter and one C2 component stage')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.feature.guards.no_model_owners([])
        ctx = context()
        if sys.argv[1] == 'prepare':
            prepare(ctx)
        elif sys.argv[1] == 'build':
            ctx.compile_fixture()
        elif sys.argv[1] == 'host':
            existing.host(ctx, extra_controls=(CONTROL,))
        else:
            check(ctx)
