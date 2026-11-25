"""Screen one pinned request on the retained DFlash2 server."""

from functools import partial
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_primary_continuation as primary
import dflash2_reconciliation_client as reconciliation
from engine_once import native_environment

bank, basis, require = primary.bank, primary.basis, primary.require
clock, footer = primary.clock, primary.footer
client = reconciliation.client
THIS = str(Path(__file__).resolve().relative_to(bank.P))
CONTEXT = 262144
ORDER = ('dflash2',)
PLAN = dict(order=list(ORDER), input_tokens=reconciliation.INPUT, context=CONTEXT, thinking=True,
            output_cap=reconciliation.OUTPUT_CAP, wire_sha256=reconciliation.WIRE_SHA256,
            scope='One independent cold server on the selected primary build. Check a fresh reconciliation '
            'request through the production OpenAI SSE frontend with no benchmark warm-up.',
            acceptance='Require matching live input count, wire body, native segments and API history. '
            'Record answer correctness, exact native output and decode clock, client TTFT and observed draft acceptance. '
            'A later comparison needs at least 1024 native outputs and acceptance at most 40 percent.',
            limits='One exploratory request. Preserve the observed result even if it is ineligible. '
            'The official agent, prefills and sustained generation cohorts remain separate.')


def command(plan, *, source=THIS):
    return [basis.reference.model.PYTHON, str(bank.P / source), 'client', plan['root'], str(plan['port'])]


def collect(root, port, *, case=reconciliation.RECONCILIATION):
    expected_root = bank.B / f'dflash2-{case.name}-model-v1' / '0'
    result_path = root / f'{case.name}.json'
    require(root == expected_root and root == Path(os.environ['STRATA_MEASUREMENT_ROOT'])
            and 1024 <= port <= 65535 and not result_path.exists(),
            'run the workload client through the owned model controller')
    config = root / 'config.json'
    key = bank.previous.read(config).get('api_key') or ''
    address = client.server_base('127.0.0.1', port)
    require(not client.is_strata(address, key), 'the selected private port is already in use')
    server = client.start_server(config, port, None, root / 'server.log')
    try:
        require(client.wait_ready(address, key, server, 600), 'owned workload server did not become ready')
        print(f'Running the suite (one {case.name} request) ...', flush=True)
        observation = reconciliation.collect(address, key, root / 'engine.stderr', context=CONTEXT, workload=case)
        bank.stage.write(result_path, dict(workload=f'dflash2_{case.name}', observation=observation))
        print(f'{case.name.capitalize()} request completed', flush=True)
    finally:
        client.stop_server(server)


def read_result(root, workload, *, case=reconciliation.RECONCILIATION):
    result = bank.previous.read(root / f'{case.name}.json')
    require(result['workload'] == workload == f'dflash2_{case.name}',
            'the owned client did not record the expected workload')
    observation = result['observation']
    sample = observation['sample']
    require(sample['request_sha256'] == case.wire_sha256
            and sample['read_tokens'] == sample['prompt_tokens'] == case.input_tokens
            and sample['reused_tokens'] == 0, 'the observed workload input changed')
    basis.comparison.require_valid_client_ttft(sample)
    return dict(samples=[dict(sample, native_segments=observation['native_segments'])],
                observation={k: v for k, v in observation.items() if k not in ('sample', 'native_segments')})


def routes(log, samples, arm):
    sample, = samples
    require(arm == 'dflash2' and log.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
            and 'strata DFlash2: eager, ' in log
            and not re.search(r'(?m)^strata mtp: .*loaded|strata loan audit:|strata trace:|strata decode timing:', log),
            'expected the selected DFlash2 engine route without timing observers')
    segments = list(reconciliation.native_segments(log))
    work = [dict(zip(footer.COPY_KEYS, map(int, m.groups()), strict=True))
            for m in re.finditer(footer.COPY, log, re.M)]
    require(segments == sample['native_segments'] and len(work) == len(segments)
            and all(all(counts[k] == 0 for k in footer.COPY_KEYS[:6]) for counts in work)
            and any(0 < counts['cpu_unique_misses'] <= counts['cpu_entries'] for counts in work),
            'native segments or decode expert policy differs')
    return dict(expert_work=work, prefill_policy=re.findall(r'^strata prefill policy: ([^\n]+)$', log, re.M))


def decision(processes):
    process, = processes
    evidence = process['evidence']
    observation = evidence['observation']
    return dict(observed_low_acceptance=observation['weak_drafts_observed'],
                correct_answer=observation['answer_correct'],
                eligible_for_repeated_weak_trial=observation['weak_drafts_observed'] and observation['answer_correct'],
                client_speed_eligible=observation['eligibility']['speed_eligible'],
                sustained_70_observed=observation['native_generation_long_enough'] and observation['native_tg'] >= 70)


def live_proof(runtime, python_runtime, root, plan, row, env, providers, manifest, entry, *, source=THIS):
    actual = bank.previous.read(root / 'engine-start.json')['native']
    proc = Path('/proc', str(actual['pid']))
    live = runtime.supervision.identity(actual['pid'])
    require(live and live['start_ticks'] == actual['start_ticks'] and live['sid'] == entry['pid']
            and (proc / 'exe').resolve() == Path(plan['binary']['path']).resolve(),
            'owned native identity differs')
    expected = native_environment(row, env)
    environment = dict(v.split(b'=', 1) for v in (proc / 'environ').read_bytes().split(b'\0') if b'=' in v)
    require(all(environment.get(k.encode()) == v.encode() for k, v in expected.items())
            and set(environment) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
            'native pre-exec environment differs')
    mapped = runtime.executable_mappings((proc / 'maps').read_text())
    probe = row['native_validation_probe']
    require(mapped <= set(providers['runtime_files']) | {str(Path(plan['binary']['path']).resolve())}
            and {probe['loader']['path'], probe['library']['path'], env['ZE_ENABLE_ALT_DRIVERS']} <= mapped,
            'native providers differ')
    allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
    python_runtime.check_running(entry['pid'], bank.P / source, manifest['selected'],
                                 env, allowed, plan['command'][2:])
    return actual


def experiment(case=reconciliation.RECONCILIATION, *, source=THIS, protocol=PLAN, decide=decision):
    require(protocol['input_tokens'] == case.input_tokens and protocol['wire_sha256'] == case.wire_sha256
            and protocol['context'] == CONTEXT and protocol['output_cap'] == reconciliation.OUTPUT_CAP
            and protocol['thinking'] is True, 'workload and frozen protocol differ')
    ctx = primary.experiment()
    original_pins = ctx.pins
    ctx.KEY = f'dflash2_{case.name}_model_v1'
    ctx.ROOT, ctx.THIS, ctx.ORDER = bank.B / ctx.KEY.replace('_', '-'), source, ORDER
    ctx.command = partial(command, source=source)
    ctx.WORKLOAD_ID, ctx.routes, ctx.protocol = f'dflash2_{case.name}', routes, protocol
    ctx.evidence_reader = SimpleNamespace(read_complete_suite=partial(read_result, case=case), decide=decide)
    official = ctx.official
    ctx.official = SimpleNamespace(**vars(official))

    def native_live(root, plan, row, env, providers, manifest, entry):
        return live_proof(official.m.runtime, official.d.python_runtime,
                          root, plan, row, env, providers, manifest, entry, source=source)

    def pins(revision):
        values = original_pins(revision)
        for name in dict.fromkeys((source, THIS, 'bench/dflash2_reconciliation_client.py',
                                   reconciliation.RECONCILIATION.fixture_source, case.fixture_source)):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit exact workload input: ' + name)
            values[name] = bank.sha(path)
        reconciliation.request_body(case)
        return values

    ctx.official.native_live, ctx.pins = native_live, pins
    return ctx


if __name__ == '__main__':
    if sys.argv[1:2] == ['client']:
        collect(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        clock.main(experiment())
