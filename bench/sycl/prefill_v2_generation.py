"""Compare demand and batched uploads through the retained V2 model server."""

from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import prefill_split_continuation as continuation
import prefill_kv_upload_compare as upload

clock, suite, state, require = continuation.clock, continuation.suite, continuation.state, continuation.require
bank, basis = upload.bank, upload.basis
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = suite.ORDER
UPLOAD = upload.previous.UPLOAD
MODES = {'original': 'demand', 'candidate': 'batched'}
TERMINAL = 'd5d60596f40225727797ab9d18e630a1883ce1de'
RESTORE = '1697efe2f16cd35927c3ab320304633fa395f5b4'


def baseline():
    native = bank.previous.read(upload.build.ROOT / 'record.json')
    upload.build.context().check(native)
    measured = upload.experiment('time')
    speed = bank.previous.read(measured.ROOT / 'record.json')
    timing = bank.previous.historic(upload.UPLOAD_TIMING, upload.KEYS['time'])
    terminal = bank.previous.historic(TERMINAL, 'prefill_v2_terminal_v1')
    restored = bank.previous.historic(RESTORE, 'prefill_kv_upload_restore_v2')
    binary = native['binaries']['strata']
    require(all(upload.gather.closed(r) for r in (timing, restored, terminal))
            and basis.compact(speed) == timing
            and timing['binary'] == restored['plan']['binary'] == terminal['plan']['binary'] == binary
            and timing['native_source_commit'] == native['native_source_commit']
            and timing['decision']['repeated_1000']['batched']
            and terminal['outcomes']['full']['requests']['fill']['state']['L'] == '262144'
            and terminal['outcomes']['full']['requests']['restore-full']['snapshot_tokens'] == 262144,
            'require the measured V2 engine and its full-context and RESTORE evidence')
    manifest = speed['runtime_manifest']
    providers = bank.previous.read(Path(manifest['path']))
    require(bank.sha(manifest['path']) == manifest['sha256'] and len(providers) == manifest['files']
            and all(bank.sha(p) == h for p, h in providers.items())
            and speed['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and speed['driver'] == native['driver'] == bank.builder.guards.driver_guard(),
            'V2 runtime or boot changed')
    basis.comparison.shared.models.model_pins(speed['configuration'], speed['model_assets'])
    for i, plan in enumerate(speed['plans']):
        require(plan['mode'] == MODES[ORDER[i]]
                and basis.comparison.fingerprint(measured.environment(speed, plan)) == plan['environment_sha256'],
                'the exact measured V2 launch settings are required')
    return dict({k: speed[k] for k in (*clock.model.FIELDS, 'template', 'python_contract', 'python_selected')},
                binary=binary, native_source_commit=native['native_source_commit'],
                native_build_sha256=bank.signature(bank.builder.compact(native)),
                ordinary_control=speed['observer_admission']['ordinary_control'],
                correctness_references=[RESTORE, TERMINAL], speed_reference=upload.UPLOAD_TIMING)


def launch_environment(row, plan, *, split):
    if split:
        controls = {'ordinary_control': row['ordinary_control']}
        env = upload.experiment('time').environment(dict(row, observer_admission=controls), plan)
    else:
        env = clock.environment(row, dict(plan, arm='clean'))
        env.update({state.KV: '1', state.GATHER: '0', clock.model.OUTPUT: '0',
                    'STRATA_SYCL_KV_COPY_GROUPS': '0'})
    require(env['STRATA_SYCL_CACHE_SCRATCH_LEASE'] == env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1'
            and env['UR_ENABLE_LAYERS'] == 'UR_LAYER_PARAMETER_VALIDATION'
            and env.get('STRATA_SYCL_CACHE_LOAN_AUDIT', '0') == '0'
            and not any(k in env for k in ('STRATA_TRACE', 'STRATA_DECODE_TIMING', 'UR_LOG_TRACING')),
            'keep qualified quiet runtime settings')
    env[UPLOAD] = MODES[plan['arm']]
    return env


def split_upload_receipt(frame, arm, dispatch):
    records = upload.previous.workspace.records(frame, 'strata MoE split expert upload: ')
    experts = dispatch['dequants']
    require(experts > 0, 'no split expert dequantizations')
    mode = MODES[arm]
    if mode == 'batched':
        record, = records
        require(record['windows'] == '48' and record['final_drain'] == 'checked'
                and int(record['copy_commands']) == int(record['experts']) == experts,
                'the batched mode differs from its executed expert uploads')
    else:
        require(not records, 'demand mode emitted a batched upload')
    return dict(mode=mode, experts=experts, copy_commands=experts)


def routes(log, samples, arm, *, split):
    if split:
        result = continuation.routes(log, samples, arm, kv_reuse=True)
    else:
        result = clock.routes(log, samples, 'clean')
        continuation.standard.profiles.check_profile_startup(log, 'original')
        require('strata MoE split expert upload: ' not in log, 'unexpected split upload in official suite')
        result['fixed_work'] = {}
    heads = list(re.finditer(clock.footer.HEADER, log, re.M))
    for i, (head, sample) in enumerate(zip(heads, samples, strict=True)):
        if not sample['phase'].startswith('long-generation/'):
            continue
        frame = log[heads[i - 1].end() if i else 0:head.start()]
        name, rows = sample['workload'], sample['read_tokens'] - 1
        if split:
            result[name]['upload'] = split_upload_receipt(frame, arm, result[name]['native_work']['routes'])
        else:
            result['fixed_work'][name] = dict(
                gather=state.gather_counts(frame, False),
                kv=state.kv.check_clean_request(frame, 0, rows, 8192, reuse=True))
    return result


def decision(processes, *, split):
    if split:
        return continuation.decision(processes)
    require(tuple(p['arm'] for p in processes) == ORDER
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER)
            and all(p['complete'] and p['passed'] and p['ownership_closed'] and not p['fresh_faults']
                    for p in processes), 'six independent clean official suites required')
    result = suite.evidence_reader.decide(processes)
    rates = []
    for first, second in zip(processes[::2], processes[1::2], strict=True):
        a, b = (first, second) if first['arm'] == 'original' else (second, first)
        require(a['routes']['fixed_work'] == b['routes']['fixed_work'], 'ordinary prefill work differs')
        left, right = (p['evidence']['samples'] for p in (a, b))
        rates.append(dict(matched_agent_histories=[s['request_sha256'] for s in left if s['phase'] == 'agent']
                          == [s['request_sha256'] for s in right if s['phase'] == 'agent'],
                          fixed_prompts=[dict(phase=x['phase'], prompt_time=y['prompt_ms']/x['prompt_ms'],
                                              TTFT=y['ttft_s']/x['ttft_s'])
                                         for x, y in zip(left, right, strict=True)
                                         if x['phase'] in ('prompt/32K', 'long-generation/regular-code',
                                                           'long-generation/varied-prose')]))
    result['paired_prefill_and_agent'] = rates
    result['decode_preservation_with_matching_counts'] &= all(p['matched_agent_histories'] for p in rates)
    return result


def experiment(phase):
    split = phase == 'split'
    ctx = clock.experiment()
    ctx.KEY = 'prefill_v2_generation_' + phase + '_v1'
    ctx.ROOT, ctx.THIS, ctx.ORDER = bank.B / ctx.KEY.replace('_', '-'), THIS, ORDER
    ctx.baseline = baseline
    ctx.environment = lambda row, plan: launch_environment(row, plan, split=split)
    ctx.routes = lambda log, samples, arm: routes(log, samples, arm, split=split)
    ctx.evidence_reader.decide = lambda samples: decision(samples, split=split)
    if split:
        ctx.command, ctx.WORKLOAD_ID = continuation.command, continuation.WORKLOAD_ID
        ctx.evidence_reader.read_complete_suite = continuation.read_complete_suite
    ctx.protocol = dict(order=list(ORDER), arms=MODES, scope=phase, upload_only=True,
        comparison='Three opposite-order pairs on the same binary. Keep stock decode settings, KV reuse, '
                   'scalar gather and copied attention fixed. Use the 128-slot split for the 96K continuation '
                   'and the ordinary 384-slot route for the official long, agent and thinking control.',
        acceptance='Check actual route, complete thinking and answer, matching prefill work and fresh input. '
                   'Compare PP, client TTFT, per-workload TG and actual draft acceptance. Require both MTP '
                   'classes, matched decode work and per-workload paired geometric mean TG ratio at least one '
                   'for preservation. Keep initial and subsequent requests distinct. No automatic adoption.',
        stop='Stop on the first identity, ownership, GPU, route or workload eligibility failure. '
             'Retain all measured pairs without speed-based exclusion or replacement.')
    core_pins, upload_pins = ctx.pins, upload.experiment('time').pins
    core_save, core_current = ctx.save, ctx.current

    def pins(revision):
        values = core_pins(revision) | upload_pins(revision)
        for name in (THIS, clock.THIS, continuation.THIS, continuation.standard.THIS,
                     continuation.standard.profiles.THIS, *continuation.standard.profiles.profile.SOURCES):
            path = bank.P / name
            require(path.read_bytes() == suite.gen.git('show', revision + ':' + name),
                    'changed generation input: ' + name)
            values[name] = bank.sha(path)
        return values

    def invariant(row):
        identities = set()
        for plan in row['plans']:
            env = ctx.environment(row, plan)
            require(env.pop(UPLOAD) == MODES[plan['arm']], 'wrong upload arm')
            normalized = {key: value.replace(plan['root'], '<process>') for key, value in env.items()}
            identities.add(bank.signature(normalized))
        require(len(identities) == 1, 'benchmark arms differ outside upload scheduling')
        return identities.pop()

    def save(row):
        if row['stage'] == 'prepared':
            row['common_environment_sha256'] = invariant(row)
        core_save(row)

    def current(row):
        require(invariant(row) == row['common_environment_sha256'], 'common environment changed')
        return core_current(row)

    ctx.pins, ctx.save, ctx.current = pins, save, current
    return ctx


if __name__ == '__main__':
    phase = sys.argv.pop(1)
    require(phase in ('official', 'split'), 'select official or split')
    clock.main(experiment(phase))
