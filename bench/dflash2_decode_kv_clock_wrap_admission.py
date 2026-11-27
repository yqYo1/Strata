"""Verify phase isolation and short-request parity on one clock-wrap model build."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_decode_kv_admission as native
import dflash2_decode_kv_clock_wrap_native as build
import dflash2_primary_continuation as primary
from check_dflash2_model import request_ids

bank, basis, require = primary.bank, primary.basis, primary.require
validation = basis.validation
FIRST_KEY = 'dflash2_decode_kv_clock_wrap_admission_v1'
FIRST_RESULT = 'ac4e646a2108b1ad45250c0a3d87fd906ec6e709'
KEY = 'dflash2_decode_kv_clock_wrap_admission_v2'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('scalar', 'clock-wrap')
FLAG = 'STRATA_SYCL_DECODE_KV_CLOCK_WRAP'
PLAN = dict(order=list(ORDER), input_tokens=82, output_tokens=32,
            scope='Same native build and per-layer stock placement. Change only the scalar clock-wrap selector.',
            routes='Keep all verifier prompt graphs on the stock scalar path. Exercise the actual scalar '
                   'decode selection in each captured width and match emitted IDs and draft counters.',
            limits='Bounded native route and parity with tracing and parameter validation. Independent clean '
                   '96K timing, agent work, physical context and RESTORE follow their own gates.')
CAPTURE = re.compile(r'^strata verify: captured the (\d+)-token window in (\d+) event-completed segments$', re.M)
COUNTER = re.compile(r'^strata verify KV_CLOCK_WRAP: phase=(prompt|decode) tokens=(\d+) enabled=([01]) calls=(\d+)$', re.M)
KV = re.compile(r'^strata verify KV_PLAN: ' + native.KV_PLAN.pattern + '$', re.M)


def routes(text, selected):
    captures = list(CAPTURE.finditer(text))
    plans = list(KV.finditer(text))
    switches = list(COUNTER.finditer(text))
    require(len(captures) == len(plans) == len(switches) and bool(captures)
            and text.count('strata verify KV_CLOCK_WRAP: ') == len(switches),
            'require one clock-wrap counter for each finalized verifier capture')
    result = []
    for capture, plan, switch in zip(captures, plans, switches, strict=True):
        width, segments = map(int, capture.groups())
        groups, rem = divmod(segments - 1, 48)
        phase = plan[1]
        switched = selected and phase == 'decode' and 1 <= width <= 4
        require(capture.end() < plan.start() < switch.start() and groups == 1 and rem == 0
                and plan.groups()[1:] == (str(width), '0', '12', '0')
                and switch.groups() == (phase, str(width), str(int(selected)), '12' if switched else '0'),
                'wrong verifier phase, geometry or scalar KV switch')
        result.append(dict(phase=phase, width=width, scalar_calls=12, clock_wrap_calls=int(switch[4])))
    return result


def baseline():
    reference = build.retained.baseline()
    row = bank.previous.read(build.ROOT / 'record.json')
    build.check(row)
    policy = row.get('policy', {})
    require(build.retained.closed(row, 'built') and policy.get('passed') and policy['exit_code'] == 0
            and not policy['gpu_executed'] and policy['binary'] == row['binaries'][build.POLICY]
            and row['native_base_commit'] == reference['native_source_commit']
            and row['base_binary'] == reference['binary'],
            'finish the same-build policy check on the coherent native trial')
    fields = dict(reference)
    fields['base_physical_context_sha256'] = fields.pop('physical_context_sha256')
    fields.update(binary=row['binaries']['strata'], native_source_commit=row['native_source_commit'],
                  native_build_sha256=bank.signature(build.builder.compact(row)), native_context_qualified=False)
    return fields


def check_profile(text, path):
    native.ranked.restore.check_loader(text, path, per_layer=True)
    require(text.count('strata generate: profile ') == 1
            and text.count('R4.2g PER-LAYER: each layer owns 26 slots (0..25).') == 1,
            'the actual per-layer profile and 26-slot layout are required')


def short_routes(text, selected):
    rows = routes(text, selected)
    require(text.count('strata prefill policy: requested=throughput route=cache-window') == 1
            and {('prompt', 1), ('decode', 1), ('decode', 4)} <= {(r['phase'], r['width']) for r in rows},
            'require the actual short prefill and both decode widths')
    return rows


def inputs(revision):
    reference = baseline()
    pins = primary.experiment().pins(revision)
    for name in (THIS, build.THIS, native.THIS, 'tools/sycl/check_dflash2_model.py'):
        path = bank.P / name
        require(path.read_bytes() == validation.git('show', revision + ':' + name),
                'commit the exact admission input: ' + name)
        pins[name] = bank.sha(path)
    cfg = primary.speed.generation.configuration(reference, dict(root=str(ROOT)))
    args = cfg['args']
    require('--expert-cache-per-layer' in args and '--no-spec-split' in args,
            'keep the validated primary placement and verifier')
    for flag, value in (('--kv', 'int8'), ('--kv-resident', '20480'), ('--max-context', '262144'), ('--spec', '4')):
        require(args.count(flag) == 1 and args[args.index(flag) + 1] == value, 'unexpected target geometry: ' + flag)
    require(args.count('--expert-profile') == 1, 'require the validated primary profile path')
    profile_path = args[args.index('--expert-profile') + 1]
    profile_sha = bank.sha(profile_path)
    require(profile_sha == bank.sha(bank.P / 'data/expert-profile.bin'),
            'the configured profile bytes differ from the primary benchmark')
    request = request_ids(Path(cfg['tokenizer']))
    require(len(request['ids']) == PLAN['input_tokens'] and request['generated'] == PLAN['output_tokens']
            and request['thinking'], 'the exact short probe changed')
    cases, environments = [], {}
    for index, arm in enumerate(ORDER):
        case = ROOT / arm
        env = native.native_environment(reference, primary.curve.environment(reference, dict(root=str(case), index=index)))
        require(env.get('STRATA_SYCL_KV_COPY_GROUPS') == '0' and FLAG not in env
                and env.get(native.FLAG, '0') == '0', 'keep scalar dispatch and the measured copy configuration')
        env.update({FLAG: str(int(arm == 'clock-wrap')), 'STRATA_TRACE': '1', 'LD_DEBUG': 'libs',
                    'UR_ENABLE_LAYERS': 'UR_LAYER_PARAMETER_VALIDATION',
                    'UR_LOG_VALIDATION': f'level:warning;flush:warning;output:file,{case / "validation.log"}'})
        environments[arm] = env
        cases.append(dict(arm=arm, root=str(case), command=[reference['binary']['path'], '--serve', *args],
                          environment_sha256=bank.signature(env), limits=primary.speed.generation.model.LIMITS))
    first = bank.previous.historic(FIRST_RESULT, FIRST_KEY)
    require(first == validation.committed(bank.RESULT, (FIRST_KEY,))
            and first['stage'] == 'failed' and first['ownership_closed'] and not first['active']
            and first['outcomes']['scalar']['native_exit_code'] == 0
            and not first['outcomes']['scalar']['fresh_faults']
            and first['request'] == request and first['binary'] == reference['binary']
            and [c['command'] for c in first['cases']] == [c['command'] for c in cases],
            'keep the original checker rejection and exact native commands')
    row = dict(first_result_sha256=bank.signature(first),
               stock=dict(path=profile_path, sha256=profile_sha),
               source_commit=revision, source_inputs=dict(files=len(pins), sha256=bank.signature(pins)),
               native_source_commit=reference['native_source_commit'], build_sha256=reference['native_build_sha256'],
               binary=reference['binary'], boot=reference['boot'], driver=reference['driver'],
               runtime_manifest=reference['runtime_manifest'], plan=PLAN, cases=cases,
               request=request, output_cap=PLAN['output_tokens'])
    return row, environments, reference


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    record = bank.previous.read(bank.RESULT)
    record[KEY] = row
    bank.stage.write(bank.RESULT, record)


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned Python and one clock-wrap model check stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one fresh admission cohort')
            revision = validation.git('rev-parse', 'HEAD').decode().strip()
            row, _, _ = inputs(revision)
            for case in row['cases']:
                for name in ('tmp', 'cache'):
                    (Path(case['root']) / name).mkdir(mode=0o700, parents=True)
            save(dict(row, stage='prepared', outcomes={}, complete=False, passed=False, ownership_closed=True, active=False))
            return
        row = bank.previous.read(ROOT / 'record.json')
        pinned, environments, reference = inputs(row['source_commit'])
        require(row == validation.committed(bank.RESULT, (KEY,)) and all(row[k] == v for k, v in pinned.items())
                and row['ownership_closed'] and not row['active'] and row['stage'] in ('prepared', 'running')
                and list(row['outcomes']) == list(ORDER[:ORDER.index(action)]), 'commit the exact next admission case')
        request = (row['request']['ids'], row['output_cap'])
        native.execute(row, environments[action], request, reference, action, order=ORDER, publish=save,
                       check_profile=lambda text: check_profile(text, row['stock']['path']),
                       read_routes=lambda text, **_: short_routes(text, action == 'clock-wrap'))


if __name__ == '__main__':
    main(*sys.argv[1:])
