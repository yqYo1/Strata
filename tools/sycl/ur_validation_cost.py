#!/usr/bin/env python3
"""Matched original-engine validation cost; Main owns serial measurements.

Repository LICENSE. No native build, inference selection or ordinary teardown.
"""
import copy
import fcntl
import json
from pathlib import Path
import select
import socket
import statistics
import subprocess
import sys
import time
sys.dont_write_bytecode = True
import cpu_capacity as models
import generation_engine as gen
import ur_hold_device as device

h, r, P, B = device.h, device.r, device.P, device.B
ROOT = B / 'ur-validation-cost-v1'
RESULT = gen.A / 'prefill-runtime-comparison-current.json'
TEMPLATE = gen.A / 'phase-prefill-speed-current.json'
KEY = 'validation_cost_control_v1'
require, digest, write = h.require, h.digest, h.write
ORDER = ('on', 'off', 'off', 'on', 'on', 'off')


def compact(row):
    return {k: v for k, v in row.items() if k not in ('controller_inputs', 'models', 'environment_template')}


def save(row):
    raw = RESULT.read_text(); desired = json.loads(raw); desired['historical_regression_audit'][KEY] = compact(row)
    literal = json.dumps(KEY) + ':'; value = json.dumps(compact(row), indent=2).replace('\n', '\n    ')
    if KEY in json.loads(raw)['historical_regression_audit']:
        require(raw.count(literal) == 1, 'ambiguous cost scope'); start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, size = json.JSONDecoder().raw_decode(raw[start:]); raw = raw[:start] + value + raw[start + size:]
    else:
        anchor = '"historical_regression_audit": {'; require(raw.count(anchor) == 1, 'ambiguous audit scope')
        raw = raw.replace(anchor, anchor + '\n    ' + literal + ' ' + value + ',', 1)
    require(json.loads(raw) == desired, 'cost update changed unrelated evidence')
    write(ROOT / 'record.json', row); RESULT.write_text(raw)


def qualification():
    helper, proof = h.native(); rt = r.validation.committed(r.RESULT); full = r.current(rt)
    gate = r.validation.committed(device.RESULT, ('event_reconciliation', 'fused_command_profile', device.KEY))
    original = json.loads((device.ROOT / 'record.json').read_text())
    require(gate == device.compact(original) and gate['stage'] == 'closed' and gate['passed'] and
        gate['complete'] and gate['ownership_closed'] and not gate['active'] and gate['helper'] == proof and
        gate['original_build'] == full['binary'] and gate['runtime_sha256'] == digest(r.RESULT) and
        len(gate['processes']) == 6 and all(v['passed'] and v['ownership_closed'] and
            v['live_environment_observed'] and v['parameter_warnings'] == 0 for v in gate['processes']) and
        gate['outcome']['fresh_faults'] == [], 'closed exact six-device gate required')
    require(h.controller(gate['source_commit'], worktree=False) == original['controller_inputs'],
        'original device gate source unrecoverable')
    return full, rt, dict(helper=proof, device_gate_sha256=h.identity(gate),
        reviewed=device.reviewed(), original_build=gate['original_build'])


def environment(row, root):
    template = row['environment_template']; old = template['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(root)) for k, v in template.items()}
    require(not any(k in env for k in ('LD_PRELOAD', 'STRATA_UR_HOLD_VALIDATION', 'STRATA_TRACE',
        'STRATA_STATE_HASH', 'STRATA_SNAPSHOT_VERIFY', 'STRATA_DECODE_TIMING', 'STRATA_VERIFY_PROFILE',
        'STRATA_CPU_WORK_CENSUS', 'ZE_ENABLE_ALT_DRIVERS')), 'unexpected base observer/provider')
    env.update(UR_ENABLE_LAYERS='', ZE_ENABLE_ALT_DRIVERS=row['NEO'],
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    return env


def current(row):
    require(r.validation.committed(RESULT, ('historical_regression_audit', KEY)) == compact(row), 'commit exact cost plan')
    full, rt, proof = qualification()
    require(proof == row['qualification'] and h.controller(row['source_commit']) == row['controller_inputs'],
        'qualified native/controller changed')
    require(digest(TEMPLATE) == row['template_sha256'] and
        gen.committed(TEMPLATE)['environment_template'] == row['environment_template'] and
        rt['boot'] == row['boot'] and rt['driver'] == row['driver'], 'template/platform changed')
    require(digest(row['expert_profile']['path']) == row['expert_profile']['sha256'], 'expert placement profile changed')
    for process in row['plans']:
        root = Path(process['root'])
        require(environment(row, root) == process['environment'], 'comparison environment changed')
        for name, sha in process['generated_sha256'].items(): require(digest(root / name) == sha, 'generated input changed')
        expected = dict(binary=full['binary'], native_validation_probe=process['probe'])
        require(json.loads((root / 'plan.json').read_text()) == expected, 'native plan differs')
    models.model_pins(row['configuration'], row['models'])
    return rt


def prepare():
    require(not ROOT.exists() and KEY not in gen.committed(RESULT)['historical_regression_audit'], 'single-use comparison')
    full, rt, proof = qualification(); revision = gen.git('rev-parse', 'HEAD').decode().strip()
    reference = gen.committed(RESULT)['historical_regression_audit']['near_dense_original_reference_v1']
    require(reference['stage'] == 'closed' and reference['passed'] and reference['ownership_closed'] and
        reference['binary'] == full['binary'], 'closed fresh same-engine reference required')
    expected = json.loads(gen.git('show', '1e676d554:' + str(TEMPLATE.relative_to(P))))['model_assets']
    cfg = copy.deepcopy(reference['configuration']); pinned_models = models.model_pins(cfg, expected, full=True)
    profile = cfg['args'][cfg['args'].index('--expert-profile') + 1]
    require(digest(profile) == '8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf',
        'historical expert placement profile differs')
    row = dict(stage='prepared', source_commit=revision, controller_inputs=h.controller(revision),
        qualification=proof, template_sha256=digest(TEMPLATE), environment_template=gen.committed(TEMPLATE)['environment_template'],
        configuration=cfg, models=pinned_models, binary=full['binary'],
        expert_profile=dict(path=profile, sha256=digest(profile)),
        NEO=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p), boot=rt['boot'], driver=rt['driver'],
        request_hashes=reference['request_hashes'], plans=[], processes=[],
        plan=dict(question='What is parameter-validation cost with the SAME held-reference control,original6e40 engine and placement?',
            order=list(ORDER), requests_per_process=4, independent_processes_per_arm=3,
            constants='Same22182f1a engine;1256-slot decode cache,131072 INT8 resident context,8192 chunks;'
                'same thinking-on four117489-fresh/zero-reuse/24-output wire requests. Both native children preload c8d080a3 '
                'and hold one loader ref. Common empty UR_ENABLE_LAYERS;config API alone toggles validation. '
                'No inference/kernel/decode/source/placement/runtime/safety change;no full traces or resource polling.',
            timing='Keep first use separate. Each process contributes median of three subsequent requests;'
                'adjacent pairs on/off,off/on,on/off. Report all samples,pair ratios and off/on geometric mean.',
            decision='Mechanism screen requires all three off/on pair ratios>1 and geometric mean>1.01. '
                'Otherwise no speedup qualification;retain signs/spreads,repeat under a new plan when variability prevents '
                'a decision. No speed-based early stop or retries. Historical reference comparisons are descriptive '
                'and cannot make observer cost explain the1003->975same-policy decline. No inference/default adoption.',
            stop='Stop first source/runtime/environment/provider/request/journal/route/validation/fault/ownership/budget '
                'failure;keep failed/incomplete status. No reset,service/package/clock change or ordinary UR teardown.',
            limits='Short24-token outputs do not qualify sustained decode,task quality,physical262144/session/RESTORE '
                'or new model route. API prefill timer includes decode handoff;no arithmetic/bus ceiling claim.'),
        budgets=dict(wall=3000, cpu=72000, rss_cap=104 << 30, text_cap=32 << 20, file_cap=64 << 20, total_cap=128 << 20),
        whole_bytes=768 << 20, complete=False, passed=False, active=False, ownership_closed=True,
        retention='Keep individual24 prompt samples,source/helper/runtime/config/environment and decision. '
            'Retire normal logs/JIT/process receipts and one-use controllers after closure and verified replay. '
            'Original13 assets unchanged;helper retained only through this comparison.')
    ROOT.mkdir(mode=0o700)
    for index, mode in enumerate(ORDER):
        root = ROOT / str(index); root.mkdir()
        for name in ('tmp', 'cache'): (root / name).mkdir()
        config = copy.deepcopy(cfg); config['log'] = str(root / 'engine.stderr')
        loader = rt['preload']['loader']
        probe = dict(library=proof['helper']['library'], loader=dict(path=loader, sha256=rt['runtime_files'][loader]),
            control='held-validation', mode=mode)
        write(root / 'config.json', config); write(root / 'plan.json', dict(binary=full['binary'], native_validation_probe=probe))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        row['plans'].append(dict(index=index, mode=mode, root=str(root), config=config, probe=probe,
            environment=environment(row, root), generated_sha256={name: digest(root / name) for name in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--prefill-only', '--prefill-tokens', '98304', '--think', '--runs', '4', '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
                '--out', str(root / 'bench')]))
    save(row); print('Prepared six processes,24long requests;commit frozen plan before run.', flush=True)


def run(row):
    rt = current(row); index = len(row['processes'])
    require(row['stage'] in ('prepared', 'running') and index < len(ORDER) and
        all(p['passed'] for p in row['processes']), 'ordered single-use processes;stop after failure')
    process = row['plans'][index]; root = Path(process['root']); env = process['environment']
    require(not (root / 'execution.json').exists(), 'never repeat native launch')
    initial = copy.deepcopy(row); owner = None; outcome = dict(complete=False, host_probe_accepted=False)
    start = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
    def persist(): write(root / 'execution.json', dict(outcome, active=bool(owner and owner.active),
        commands=owner.commands if owner else []))
    def interact(io, leader):
        for request in range(4):
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(1500)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                    dict(stage=stage, phase='prefill-only', run=request), 'request gate differs')
                require(not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size, 'parameter warning')
                if stage == 'begin':
                    claim = json.loads((root / 'engine-start.json').read_text()); pid = claim['native']['pid']
                    ident = r.supervision.identity(pid)
                    require(ident and ident['start_ticks'] == claim['native']['start_ticks'] and ident['sid'] == leader['pid'] and
                        str(Path('/proc', str(pid), 'exe').resolve()) == str(Path(row['binary']['path']).resolve()), 'native identity differs')
                    import engine_once
                    expected = engine_once.native_environment(dict(native_validation_probe=process['probe']), env)
                    actual = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
                    require(all(actual.get(k.encode()) == v.encode() for k, v in expected.items()) and
                        set(actual) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'}, 'native environment differs')
                    mapped = r.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
                    allowed = {p: v for p, v in rt['runtime_files'].items() if p != rt['preload']['library']['path']}
                    allowed.update({process['probe']['library']['path']: process['probe']['library']['sha256'],
                        str(Path(row['binary']['path']).resolve()): row['binary']['sha256']})
                    require(mapped <= set(allowed) and {row['NEO'], process['probe']['loader']['path']} <= mapped and
                        any('/libigdgmm.so.' in p for p in mapped), 'unexpected/missing actual provider')
                    outcome['verified_request_boundaries'] = request + 1
                else: print(f"Process {index}/{process['mode']}: long request {request + 1}/4 completed", flush=True)
                require(r.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole comparison byte budget')
                io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner = r.supervision.Owner(root / 'owned', persist)
        entry, _, _ = owner.run('bench', process['command'], env, P, interaction=interact, **row['budgets'])
        require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'benchmark/owner failed')
        result_file, = (root / 'bench').glob('*/results.json'); result = json.loads(result_file.read_text())
        samples = result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and len(samples) == 4 and
            outcome.get('verified_request_boundaries') == 4, 'incomplete official benchmark')
        for sample, expected in zip(samples, row['request_hashes']):
            require(sample['ok'] and sample['eligible'] and sample['stream_complete'] and sample['thinking_requested'] and
                sample['reasoning_observed'] and sample['request_sha256'] == expected and
                sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == 117489 and
                sample['actual_reused_tokens'] == 0 and sample['generated_tokens'] == 24 and
                sample['prompt_tps_source'] == 'engine' and type(sample['prompt_ms']) in (int, float) and
                10000 <= sample['prompt_ms'] < 1500000, 'workload/timing eligibility differs')
        sys.path.insert(0, str(P / 'tools')); from strata_bench import verify_request_journal
        journal, = (root / 'bench').glob('*/requests.jsonl'); verified = verify_request_journal(journal.parent)
        require(verified['valid'] and verified['complete'], 'journal invalid/incomplete')
        events = [json.loads(v) for v in journal.read_text().splitlines()]
        longs = [v for v in events if v['kind'] == 'result' and v['phase'] == 'prefill-only']
        require([dict(v['sample'], run=v['run']) for v in longs] == samples, 'journal/request binding differs')
        end = gen.native_end(root, dict(binary=row['binary'], native_validation_probe=process['probe']),
            dict(config=process['config']), row['boot'], entry['pid'])
        text = (root / 'engine.stderr').read_text()
        activation = device.evidence(text, end['native']['pid'], process['mode'], process['probe']['loader']['path'])
        require(text.count('strata prefill: layer-major candidate, tokens=117488 chunk=8192 residual_dtype=bf16') == 4 and
            text.count('strata MoE window configuration: requested_mode=10 effective_mode=10') == 4 and
            not any(s in text for s in ('strata trace:', 'strata decode timing:', 'STATE_HASH', 'SNAPSHOT_VERIFY',
                'UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')), 'route/runtime differs')
        current(initial); models.model_pins(row['configuration'], row['models'], full=True)
        fields = ('run', 'request_sha256', 'actual_prompt_tokens', 'actual_read_tokens', 'actual_reused_tokens',
            'generated_tokens', 'prompt_ms', 'thinking_requested')
        rates = [117489000 / s['prompt_ms'] for s in samples]
        outcome.update(complete=True, host_probe_accepted=True, index=index, mode=process['mode'],
            requests=[{k: s[k] for k in fields} for s in samples], tok_s=rates,
            first_use_tok_s=rates[0], subsequent_median=statistics.median(rates[1:]),
            subsequent_range=[min(rates[1:]), max(rates[1:])], native_identity=end['native'], validation=activation)
    except BaseException as error:
        outcome['error'] = repr(error)
        if (root / 'engine.stderr').exists(): outcome['error_excerpt'] = (root / 'engine.stderr').read_text()[-4096:]
    finally:
        if owner: r.validation.finish(outcome, owner, start, row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete']); persist()
        row['processes'].append(outcome); row.update(active=False, ownership_closed=outcome.get('ownership_closed', True))
        row['stage'] = 'running' if outcome['passed'] and len(row['processes']) < len(ORDER) else 'closed'
        row['complete'] = len(row['processes']) == len(ORDER) and all(p['passed'] for p in row['processes'])
        row['passed'] = row['complete']
        if row['complete']:
            pairs = []
            for offset in (0, 2, 4):
                group = {p['mode']: p['subsequent_median'] for p in row['processes'][offset:offset + 2]}
                pairs.append(group['off'] / group['on'])
            ratio = statistics.geometric_mean(pairs)
            row['analysis'] = dict(off_over_on_pair_ratios=pairs, geometric_mean=ratio,
                mechanism_speed_screen_passed=all(v > 1 for v in pairs) and ratio > 1.01,
                inference_adoption=False, historical_same_policy_cause_established=False)
        save(row)
    print(json.dumps(outcome), flush=True)
    return int(not outcome['passed'])


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); r.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        if sys.argv[1:] == ['run']: return run(json.loads((ROOT / 'record.json').read_text()))
        raise RuntimeError('use prepare or run;commit each closed process before next')


if __name__ == '__main__': sys.exit(main())
