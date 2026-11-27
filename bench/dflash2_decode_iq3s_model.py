"""Bind the isolated CPU IQ3_S candidate to actual DFlash2 decode work."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_decode_iq3s_build as build
import dflash2_decode_kv_width3_admission as prompt_fixture

kv = prompt_fixture.kv
bank, basis, ranked, primary = kv.bank, kv.basis, kv.ranked, kv.primary
require, validation = kv.require, kv.validation
census, footer = ranked.restore.census, primary.footer
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_iq3s_model_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('ggml', 'iq3s')
FLAG = 'STRATA_IQ_SINGLE_EXACT'
READ, OUTPUT, CHUNK = prompt_fixture.READ, prompt_fixture.OUTPUT, prompt_fixture.CHUNK
PLAN = dict(
    order=list(ORDER),
    scope='Same engine and stock expert placement. Change only the decode-only IQ3_S selector before exec.',
    checks='Replay the committed 33287-input thinking body with a 48-token cap. Check actual fresh layer-major '
           'prefill, DFlash2 output IDs and draft counters, all 48 admitted expert masks, and identical CPU '
           'format/width work with nonzero IQ3_S width-one jobs. Require quiet validation and closed ownership.',
    limits='The census proves eligible CPU jobs. This production build does not export the test-only per-row '
           'selector counter. The pinned selector and same-library parity fixture establish the branch conditions. '
           'Observer runs do not qualify speed, 262144-token fill, or RESTORE.')


def inputs(revision):
    reference = primary.retained.baseline()
    native = bank.previous.read(build.ROOT / 'record.json')
    build.check(native)
    identity = bank.signature(build.builder.compact(native))
    parity = validation.committed(bank.RESULT, ('dflash2_decode_iq3s_parity_v1',))
    require(primary.retained.closed(native, 'built') and primary.retained.closed(parity, 'closed')
            and parity['native_build_sha256'] == identity and parity['binary'] == native['binaries'][build.FIXTURE]
            and not parity['gpu_executed'] and parity['exit_code'] == 0,
            'require the coherent engine and exact same-build GGML parity')
    stock = ranked.restore_proof(reference, ranked.restore.KEY, ranked.restore.ORDER)
    profile = ranked.qualified_profile('stock', stock, None)
    paths = ranked.experiment('blend').pins(revision)
    for name in (THIS, build.THIS, kv.THIS, kv.scorer.THIS, prompt_fixture.THIS,
                 'serve/frontend.py', 'tools/strata_tokenizer.py', 'tools/strata_bench.py',
                 'tools/long_generation_bench.py', 'tools/ab_engine.py',
                 'tools/sycl/cpu_work_census.py', 'tools/sycl/decode_footer.py'):
        path = bank.P / name
        require(path.read_bytes() == validation.git('show', revision + ':' + name),
                'commit the exact IQ3_S model input: ' + name)
        paths[name] = bank.sha(path)
    cfg = ranked.configuration(dict(reference, qualified_profiles={'stock': profile}),
                               dict(arm='stock', root=str(ROOT)))
    args = cfg['args']
    for flag, value in (('--kv', 'int8'), ('--kv-resident', '20480'), ('--max-context', '262144'),
                        ('--prefill', str(CHUNK)), ('--spec', '4'), ('--pcie-frac', '0')):
        require(args.count(flag) == 1 and args[args.index(flag) + 1] == value, 'unexpected model option: ' + flag)
    require('--no-spec-split' in args, 'keep the verifier unsplit')
    manifest = Path(args[args.index('--pack') + 1]) / 'native_experts.txt'
    formats = census.manifest_formats(manifest)
    quota = ranked.restore.quota
    membership = quota.masks(quota.read_profile(profile['path'])[:quota.EXPECTED_SLOTS])
    ids, prompt = prompt_fixture.prompt(Path(cfg['tokenizer']))

    binary = native['binaries']['strata']
    device = primary.speed.generation.model
    cases, environments = [], {}
    for index, arm in enumerate(ORDER):
        root = ROOT / arm
        env = kv.native_environment(reference, primary.curve.environment(reference, dict(root=str(root), index=index)))
        require(env.get(FLAG, '0') == '0' and 'STRATA_NO_IQ256' not in env
                and env.get('STRATA_IQ_MT_MIN', '2') == '2', 'unexpected inherited CPU selector')
        env.update({FLAG: 'iq3s' if arm == 'iq3s' else '0', 'STRATA_TRACE': '1', 'LD_DEBUG': 'libs',
                    'STRATA_CPU_WORK_CENSUS': '1', 'STRATA_DECODE_TIMING': '1',
                    'UR_ENABLE_LAYERS': 'UR_LAYER_PARAMETER_VALIDATION',
                    'UR_LOG_VALIDATION': f'level:warning;flush:warning;output:file,{root / "validation.log"}'})
        env.pop('UR_LOG_TRACING', None)
        environments[arm] = env
        cases.append(dict(arm=arm, root=str(root), command=[binary['path'], '--serve', *args],
                          environment_sha256=bank.signature(env), limits=device.LIMITS))
    record = dict(source_commit=revision, source_inputs=dict(files=len(paths), sha256=bank.signature(paths)),
                  native_source_commit=native['native_source_commit'], build_sha256=identity, binary=binary,
                  parity_sha256=bank.signature(parity), boot=reference['boot'], driver=reference['driver'],
                  runtime_manifest=reference['runtime_manifest'], stock=profile,
                  manifest=dict(path=str(manifest), sha256=bank.sha(manifest)), plan=PLAN, cases=cases,
                  prompt=prompt, read_tokens=READ, output_cap=OUTPUT)
    return record, environments, (ids, OUTPUT), reference, formats, membership


def routes(text, formats, membership):
    expected = f'strata prefill policy: requested=throughput route=layer-major tokens={READ - 1} chunk={CHUNK}'
    headers = re.findall(footer.HEADER, text, re.M)
    timing = re.findall(footer.TIMING, text, re.M)
    copies = re.findall(footer.COPY, text, re.M)
    require(text.splitlines().count(expected) == 1 and len(headers) == len(timing) == len(copies) == 1
            and tuple(map(int, headers[0][:4])) == (READ, 0, READ, OUTPUT),
            'require fresh layer-major prefill and one completed bounded decode')
    windows = [tuple(map(int, fields)) for fields in re.findall(r'^strata trace: window (\d+) (\d+)$', text, re.M)]
    require(windows and windows[0] == (READ - 1, 1) and any(width > 1 for _, width in windows)
            and all(1 <= width <= 4 for _, width in windows)
            and all(1 <= b[0] - a[0] <= a[1] for a, b in zip(windows, windows[1:]))
            and len(windows) == int(timing[0][0]) and sum(width - 1 for _, width in windows) == int(headers[0][5])
            and not any(map(int, copies[0][:6])), 'unexpected DFlash2 windows or PCIe expert jobs')
    clocks = dict(zip(footer.LABELS, map(float, timing[0][1:]), strict=True))
    observed, = census.extract(text, [dict(
        prompt_tokens=READ, windows=len(windows), rows=sum(width for _, width in windows),
        cpu_jobs=int(copies[0][-2]), cpu_entries=int(copies[0][-1]),
        pool_ms=clocks['CPU_pool_run'] * len(windows))], formats, membership)
    jobs = sum(b['jobs_NT'][0] for b in observed['format_bins'] if b['gu'] == 21)
    require(jobs > 0, 'no eligible single-token IQ3_S CPU work')
    fields = ('windows', 'rows', 'cpu_jobs', 'cpu_entries', 'vram_entries', 'membership_sha256')
    return dict({name: observed[name] for name in fields}, iq3s_single_jobs=jobs,
                formats=[{name: b[name] for name in ('gu', 'down', 'layer_count', 'jobs_NT')}
                         for b in observed['format_bins']])


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned Python and select one IQ3_S admission stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one fresh IQ3_S admission')
            revision = validation.git('rev-parse', 'HEAD').decode().strip()
            row, *_ = inputs(revision)
            for case in row['cases']:
                for name in ('tmp', 'cache'):
                    (Path(case['root']) / name).mkdir(mode=0o700, parents=True)
            save(dict(row, stage='prepared', outcomes={}, complete=False, passed=False,
                      ownership_closed=True, active=False))
            return

        row = bank.previous.read(ROOT / 'record.json')
        pinned, environments, request, reference, formats, membership = inputs(row['source_commit'])
        require(row == validation.committed(bank.RESULT, (KEY,)) and all(row[k] == v for k, v in pinned.items())
                and row['ownership_closed'] and not row['active'] and row['stage'] in ('prepared', 'running')
                and list(row['outcomes']) == list(ORDER[:ORDER.index(action)]), 'commit the exact next case')

        def inspect(log, *, cooperative):
            observed = routes(log, formats, membership)
            require(observed['membership_sha256'] == row['stock']['membership_sha256'],
                    'live cache membership differs from the frozen map')
            if cooperative:
                require(observed == row['outcomes'][ORDER[0]]['cpu_work'], 'observed CPU format/width work differs')
            return observed

        kv.execute(row, environments[action], request, reference, action, read_routes=inspect, publish=save,
                   order=ORDER, route_key='cpu_work')


if __name__ == '__main__':
    action, = sys.argv[1:]
    main(action)
