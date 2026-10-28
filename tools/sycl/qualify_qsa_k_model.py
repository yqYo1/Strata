#!/usr/bin/env python3
"""Matched official long/agent/thinking qualification; never adopts a model.

Reuse the source-bound native/frontend gates, not a previous run's generic pass
field. Each process runs the maintained bench.sh suite without a client patch.
"""
import collections
import copy
import fcntl
import json
import math
from pathlib import Path
import re
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import compare_qsa_k_model as previous
import engine_once

d = previous.diagnostic
m, P, B, RESULT = previous.m, previous.P, previous.B, previous.RESULT
ROOT = B / 'qsa-direct-k-model-qualification-v1'
KEY = 'official_model_qualification'
ORDER = previous.ORDER
require, digest, write = previous.require, previous.digest, previous.write
DOCS = ('README.md', 'docs/HOW_IT_WORKS.md', 'docs/DETAILS.md', 'docs/INSTALL.md',
        'docs/MODELS.md', 'docs/COMMUNITY_BENCHMARKS.md')


def pins(revision):
    result = d.pins(revision)
    for name in DOCS:
        tracked = m.component.gen.git('ls-tree', '--name-only', revision, '--', name).strip()
        if not tracked:
            require(not (P / name).exists(), 'uncommitted agent document: ' + name)
            result[name] = None
        else:
            raw = m.component.gen.git('show', revision + ':' + name)
            require(raw == (P / name).read_bytes(), 'agent document changed: ' + name)
            result[name] = d.hashlib.sha256(raw).hexdigest()
    return result


def qualification():
    record, built, rt, manifest, model = previous.qualification()
    screen = record[m.KEY][previous.KEY]
    private = json.loads((previous.ROOT / 'record.json').read_text())
    require(screen == previous.compact(private) and screen['stage'] == 'closed' and
            screen['complete'] and screen['passed'] and screen['ownership_closed'] and
            not screen['active'] and len(screen['processes']) == 6 and
            all(p['passed'] and not p.get('error') for p in screen['processes']) and
            screen['decision']['small_gain_screen_passed'] and
            screen['post_closure_audit']['model_assets_full_post_digest_verified'],
            'audited closed six-process speed screen required')
    return record, built, rt, manifest, model


def compact(row):
    return {k:v for k,v in row.items() if k not in ('sources', 'model_assets')}


def accepted(outcome):
    # Generic UR finish predates this controller and does not reject an error
    # field. No exception or unfinished postcondition may become a qualified run.
    return bool(outcome.get('passed') and outcome.get('complete') and
                outcome.get('postconditions_complete') and not outcome.get('error'))


def prepare():
    record, built, rt, _, model = qualification()
    require(KEY not in record[m.KEY] and not ROOT.exists(), 'fresh qualification only')
    revision = m.component.gen.git('rev-parse', 'HEAD').decode().strip()
    source = pins(revision)
    assets = d.models.model_pins(model['configuration'], model['model_assets'], full=True)
    binaries = {'original':built['selected_binary'], 'candidate':built['binaries']['strata']}
    ROOT.mkdir(mode=0o700)
    plans = []
    for index, arm in enumerate(ORDER):
        root = ROOT / str(index); root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root / name).mkdir()
        config = copy.deepcopy(model['configuration']); config['log'] = str(root / 'engine.stderr')
        write(root / 'config.json', config)
        write(root / 'plan.json', dict(binary=binaries[arm], native_validation_probe=model['native_validation_probe']))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        command = ['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'),
            '--port', str(port), '--long', '--long-generation', '--think', '--runs', '1',
            '--workload-id', 'qsa_qualification_20261028', '--fail-fast-engine-error',
            '--ready-timeout', '600', '--out', str(root / 'bench')]
        plans.append(dict(index=index, arm=arm, root=str(root), binary=binaries[arm], command=command,
            generated={n:digest(root / n) for n in ('config.json', 'plan.json')},
            environment_sha256=d.fingerprint(previous.environment(root, rt))))
    row = dict(stage='prepared', source_commit=revision, sources=source, model_assets=assets,
        runtime_sha256=digest(m.runtime.RESULT), boot=rt['boot'], profile=model['profile'],
        native_validation_probe=model['native_validation_probe'], plans=plans, processes=[],
        budgets=dict(wall=5400, cpu=18000, rss_cap=104 << 30, text_cap=64 << 20,
                     file_cap=128 << 20, total_cap=512 << 20),
        plan=dict(order=list(ORDER), runs_per_process=1,
            question='Does the isolated prefill row-address hoist preserve decode performance and complete official long/agent and both sustained-thinking workloads?',
            constants='Same 131072 INT8 resident context,1256 decode expert slots,8192 chunks,BF16/vLLM/mode10,5workers,spec4,models/profile/client/safety.Only the native executable changes. No observer or environment mutation between phases.',
            samples='Six independent processes in three opposite-order adjacent pairs. Each official full suite has short warmups/calibration,chat,4K/16K/32K prompts,one complete six-turn agent session,and both >=32K-input/>=1K-output thinking workloads. Agent follow-up rate is the median of its five engine-reported decode rates;do not substitute API usage for the engine token count. Preserve every request,actual lengths and engine clocks;do not require identical generated text.',
            acceptance='All six executions,complete journals,agent sessions and both eligible long-generation cases must pass. Pool exactly three independent requests per workload/arm through the maintained strong>=0.60/weak<=0.40 MTP classifier. For each generation workload and agent follow-up decode,report all three paired candidate/original rate ratios;require geometric mean>=1 to screen preservation. A lower rate or unconfirmed MTP class remains pending/inconclusive;no noise waiver or automatic adoption. Matched 32K prompt rate and TTFT are descriptive in this gate.',
            stopping='First source/runtime/ownership/numerical/validation/fault,budget,incomplete response,missing answer,or workload admission failure stops the cohort. No speed-based early stopping,retries,replacement or post-hoc exclusions. Generic success cannot override an exception or unfinished final hashes.',
            limits='This qualifies observed execution,answer presence,MTP regimes and matched rates,not semantic task quality or physical262144 fill/tail/handoff/RESTORE. Short chat/prompt output is not sustained decode evidence. Input-length approximation remains scoped;no hardware-saturation or1000/70target claim.',
            retention='Retain compact individual samples,source/recipe/config/runtime and decisions;retire verified successful raw journals/logs/private caches after closure. Keep fallback and this candidate only for imminent adoption gates.'),
        complete=False, passed=False, active=False, ownership_closed=True)
    write(ROOT / 'record.json', row); record[m.KEY][KEY] = compact(row); write(RESULT, record)
    print('Prepared six matched official long/agent/thinking processes;commit before running.', flush=True)


def native_live(root, plan, row, env, rt, manifest, entry):
    claim = json.loads((root / 'engine-start.json').read_text()); actual = claim['native']
    pid = actual['pid']; live = m.runtime.supervision.identity(pid)
    require(live and live['start_ticks'] == actual['start_ticks'] and live['sid'] == entry['pid'] and
            str(Path('/proc', str(pid), 'exe').resolve()) == str(Path(plan['binary']['path']).resolve()),
            'owned native identity differs')
    expected = engine_once.native_environment(dict(native_validation_probe=row['native_validation_probe']), env)
    current = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
    require(all(current.get(k.encode()) == v.encode() for k,v in expected.items()) and
            set(current) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
            'native pre-exec environment differs')
    mapped = m.runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
    require(mapped <= set(rt['runtime_files']) | {str(Path(plan['binary']['path']).resolve())} and
            {row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'],
             env['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'native providers differ')
    allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
    d.python_runtime.check_running(entry['pid'], P / 'tools/strata_bench.py',
        manifest['selected'], env, allowed, plan['command'][2:])
    return actual


def evidence(root, row):
    files = list((root / 'bench').glob('*/results.json')); require(len(files) == 1, 'one official result required')
    result = json.loads(files[0].read_text()); out = result['results']
    journal = d.client.verify_request_journal(files[0].parent)
    require(result['complete'] and not out['failures'] and not out['skipped'] and
            result['params']['think'] and result['params']['long'] and result['params']['long_generation'] and
            journal['valid'] and journal['complete'] and not journal['pending'], 'incomplete official suite')
    events = [json.loads(line) for line in (files[0].parent / 'requests.jsonl').read_text().splitlines()]
    starts = {v['index']:v for v in events if v['kind'] == 'start'}
    samples = []
    for event in events:
        if event['kind'] != 'result': continue
        sample = event['sample']; start = starts[event['request_index']]
        require(sample['ok'] and sample['stream_complete'] and
                sample['request_sha256'] == start['request_sha256'] and
                start['thinking'] == (event['phase'] != 'calibration'), 'request/body/thinking evidence differs')
        samples.append(dict(phase=event['phase'], run=event['run'], turn=event['turn'], **sample))
    counts = collections.Counter(s['phase'] for s in samples)
    require(counts == collections.Counter({'warm-up/chat':1, 'calibration':2, 'warm-up/prompt':1,
            'chat':3, 'prompt/4K':1, 'prompt/16K':1, 'prompt/32K':1, 'agent':6,
            'long-generation/regular-code':1, 'long-generation/varied-prose':1}), 'suite scope differs')
    long_prompt = out['prompts']['32K']['requests']
    require(len(long_prompt) == 1 and long_prompt[0]['read_tokens'] == long_prompt[0]['prompt_tokens'] >= 32768 and
            long_prompt[0]['reused_tokens'] == 0 and long_prompt[0]['prompt_tps_source'] == 'engine',
            'standard long prompt lacks32Kfresh input/engine clock')
    agent = out['agent']
    require(agent['complete_sessions'] == 1 and agent['failed_sessions'] == agent['partial_sessions'] == 0 and
            len(agent['sessions']) == 1 and d.client.agent_complete(agent['sessions'][0]), 'agent session incomplete')
    generation = out['long_generation']; rows = m.component.gen.collect_samples(generation)
    m.component.gen.require_complete_generation(generation, rows)
    for s in agent['sessions'][0]:
        require(s['decode_tps_source'] == 'engine' and s['decode_tps'] > 0 and s['decode_ms'] > 0 and
                s['reasoning_chars'] > 0, 'agent thinking/engine decode clock missing')
    # Keep each request once. The generation result adds order/eligibility
    # fields after its journal row; retain those with the corresponding sample.
    by_workload = {s['workload']:s for s in rows}
    for sample in samples:
        if sample['phase'].startswith('long-generation/'):
            sample.update(by_workload[sample['phase'].split('/', 1)[1]])
    follow = agent['sessions'][0][1:]
    rates = {s['workload']:s['decode_tps'] for s in rows}
    rates['agent_followup'] = statistics.median(s['decode_tps'] for s in follow)
    return dict(samples=samples, decode_rates=rates, journal=journal)


def decision(processes):
    pairs = {}
    for name in ('regular-code', 'varied-prose', 'agent_followup'):
        ratios = []
        for offset in (0, 2, 4):
            pair = {p['arm']:p for p in processes[offset:offset+2]}
            ratios.append(pair['candidate']['evidence']['decode_rates'][name] /
                          pair['original']['evidence']['decode_rates'][name])
        mean = math.exp(statistics.mean(math.log(v) for v in ratios))
        pairs[name] = dict(pair_rate_ratios=ratios, geometric_mean_rate_ratio=mean, preservation_screen_passed=mean >= 1)
    classes = {}
    for arm in ('original', 'candidate'):
        cases = {name:dict(requests=[s for p in processes if p['arm'] == arm
                    for s in p['evidence']['samples'] if s.get('workload') == name])
                 for name in ('regular-code', 'varied-prose')}
        classes[arm] = d.client.long_generation.classify_pair(cases, 3)
    return dict(decode=pairs, MTP=classes,
        preservation_screen_passed=all(v['preservation_screen_passed'] for v in pairs.values()),
        MTP_pair_qualified=all(v['qualified'] for v in classes.values()),
        inference_adopted=False, physical262144_qualified=False)


def run(index):
    record, _, rt, manifest, _ = qualification(); row = json.loads((ROOT / 'record.json').read_text())
    require(record[m.KEY][KEY] == compact(row) and 0 <= index < len(ORDER) and
            len(row['processes']) == index and all(p['passed'] for p in row['processes']),
            'committed sequential cohort required;no retry/skip')
    require(pins(row['source_commit']) == row['sources'] and digest(m.runtime.RESULT) == row['runtime_sha256'],
            'source/runtime changed')
    plan = row['plans'][index]; root = Path(plan['root']); env = previous.environment(root, rt)
    require(not (root / 'execution.json').exists() and plan['arm'] == ORDER[index] and
            d.fingerprint(env) == plan['environment_sha256'] and
            all(digest(root / n) == sha for n,sha in plan['generated'].items()) and
            digest(plan['binary']['path']) == plan['binary']['sha256'], 'frozen launch plan differs')
    config = json.loads((root / 'config.json').read_text()); d.models.model_pins(config, row['model_assets'])
    profile = row['profile']
    require(d.models.file_identity(profile['path']) == profile['identity'] and
            digest(profile['path']) == profile['sha256'], 'expert profile differs')
    m.runtime.guards.no_model_owners([])
    cursor = re.findall(r'^-- cursor: (.+)$', subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15), re.M)
    require(len(cursor) == 1, 'fresh fault cursor required')
    outcome = dict(index=index, arm=plan['arm'], complete=False, host_probe_accepted=False, passed=False)
    owner = None
    def persist(): write(root / 'execution.json', dict(outcome, active=bool(owner and owner.active),
        commands=owner.commands if owner else []))
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def interact(io, entry):
        checked = False
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
            require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
            if select.select([io.proc.stdout], [], [], .05)[0]: io.read()
            while b'\n' in io.pending:
                line, _, tail = io.pending.partition(b'\n'); io.pending[:] = tail
                text = line.decode('utf-8').strip()
                if text.startswith('Running the suite'):
                    require(not checked, 'duplicate official suite start')
                    outcome['native_identity'] = native_live(root, plan, row, env, rt, manifest, entry)
                    checked = True; print(f'Process{index} {plan["arm"]} runtime verified', flush=True)
                if text.startswith(('agent:', 'long generation:', '32K prompt:', 'run ')):
                    print(f'Process{index}: {text}', flush=True)
        require(checked, 'runtime identity not observed'); io.pending.clear(); io.proc.stdin.close(); io.exit(1)
    try:
        owner = m.runtime.supervision.Owner(root / 'owned', persist)
        entry, _, _ = owner.run('bench', plan['command'], env, P, interaction=interact, **row['budgets'])
        require(m.runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'official process failed')
        outcome['evidence'] = evidence(root, row)
        require(pins(row['source_commit']) == row['sources'] and digest(m.runtime.RESULT) == row['runtime_sha256'],
                'source/runtime changed during suite')
        end = m.component.gen.native_end(root, dict(binary=plan['binary'], native_validation_probe=row['native_validation_probe']),
            dict(config=config), row['boot'], entry['pid'])
        log = (root / 'engine.stderr').read_text(); proof = m.runtime.validation.preload_evidence(log, end['native']['pid'])
        require(proof['loader'] == row['native_validation_probe']['loader']['path'] and
                'strata MoE window configuration: requested_mode=10 effective_mode=10' in log and
                not any(t in log for t in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')) and
                (not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size),
                'route/runtime/validation differs')
        for s in outcome['evidence']['samples']:
            if s['phase'].startswith(('prompt/32K', 'long-generation/')):
                require('strata prefill: layer-major candidate, tokens=' + str(s['read_tokens']-1) +
                        ' chunk=8192 residual_dtype=bf16' in log, 'actual long route missing')
        d.models.model_pins(config, row['model_assets'], full=index == len(ORDER)-1)
        outcome.update(complete=True, postconditions_complete=True, host_probe_accepted=True,
                       native_exit_code=0, validation=proof)
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        if owner:
            try: m.runtime.validation.finish(outcome, owner, cursor[0], row['boot'])
            except BaseException as error: outcome['error'] = repr(error); outcome['passed'] = False
        outcome['passed'] = accepted(outcome); persist()
        row['processes'].append(outcome); row.update(stage='running' if outcome['passed'] else 'failed',
            ownership_closed=outcome.get('ownership_closed', False), active=bool(owner and owner.active))
        if outcome['passed'] and len(row['processes']) == len(ORDER):
            row.update(stage='closed', complete=True, passed=True, decision=decision(row['processes']))
        write(ROOT / 'record.json', row); record[m.KEY][KEY] = compact(row); write(RESULT, record)
    print(json.dumps({k:outcome.get(k) for k in ('index', 'arm', 'passed', 'complete', 'fresh_faults', 'error')}), flush=True)
    require(outcome['passed'], 'official long/agent/thinking qualification failed')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if sys.argv[1] == 'prepare' else run(int(sys.argv[2]))
