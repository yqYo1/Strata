#!/usr/bin/env python3
"""Main-owned clean CPU-pool capacity discriminator; no engine rebuild."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import resource_engine as resources

P, B, A = gen.P, gen.B, gen.A
RESULT = A / 'decode-cpu-capacity-current.json'
ORDER = (5, 1, 1, 5, 5, 1)
SOURCE_KEYS = ('binary', 'build_source_commit', 'source_build_record', 'fallback_reference',
               'runtime_asset_delta', 'original_native_inputs_recovered')
WORK_KEYS = ('request_sha256', 'read_tokens', 'prompt_tokens', 'reused_tokens',
             'generated_tokens', 'drafts_offered', 'drafts_accepted')


def check_prompts(config, environment):
    # Match serve/server.py's actual tokenizer and template, without an optional
    # transformers/tokenizers package or a model/GPU launch.
    script = '''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,str(Path(sys.argv[1])/'tools'))
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate
p=Path(sys.argv[2]);v=json.loads((p/'vocab.json').read_text());tokens=[None]*len(v)
for text,index in v.items(): tokens[index]=text
t=Tokenizer(tokens,(p/'merges.txt').read_text().split('\\n'),json.loads((p/'token_type.json').read_text()))
template=ChatTemplate(p/'chat_template.jinja')
questions=['Explain why a queue dependency must precede a GPU memory reuse.',
           'Write a short Python function that sums a list of integers.']
print(json.dumps([t.encode(template.render([dict(role='user',content=q)],enable_thinking=True),parse_special=True) for q in questions]))
'''
    prompts = json.loads(subprocess.check_output([
        str(Path(environment['PATH'].split(':')[0])/'python'), '-c', script,
        str(P), config['tokenizer']], env=environment, cwd=P, timeout=30, text=True))
    guards.require(len(prompts) == 2 and all(p and all(type(x) is int and 0 <= x < 248320 for x in p)
        for p in prompts), 'invalid generated qualification tokens')
    return prompts


def comparison_plan():
    return dict(order=list(ORDER), pairs=[[0, 1], [3, 2], [4, 5]],
        question='Does CPU expert-pool capacity affect critical-path generation on this configured MTP4 route?',
        eligibility='All six independent clean processes, four successful official warmups each, same fresh request, output/draft work counters, startup placement and runtime. No internal or external observers. Preserve every sample, including failed/ineligible runs; no automatic retry.',
        decision='Let R be one-worker/five-worker decode latency per emitted token in each pair. Support reduced-capacity sensitivity only if all three R>1 and geometric mean R>1.10; support the opposite direction only if all R<1 and geometric mean R<1/1.10. Otherwise inconclusive. Partial, mismatched-work or failed cohorts cannot decide.',
        threshold=1.10,
        limits='CPU capacity changes scheduling, partition sizes, polling, cache/reuse and RAM demand together. Sensitivity does not isolate arithmetic from RAM or prove the five-worker baseline is saturated. One MTP-friendly request is diagnostic, not model quality, strong/weak adoption or70tok/s qualification.')


def summarize_processes(rows):
    guards.require(len(rows) == 6 and [r['cohort'] for r in rows] == list(range(6)) and
        [r['workers'] for r in rows] == list(ORDER), 'complete frozen six-process order required')
    identities = [(r['native_identity']['pid'], r['native_identity']['start_ticks']) for r in rows]
    guards.require(len(set(identities)) == 6, 'native process identity reused')
    guards.require(all(r['passed'] and r['ownership_closed'] and not r['fresh_faults'] and
        r['warmups']['successful_warmups'] == 4 and r['runtime_pool_confirmed'] and
        r['observer_records_absent'] and gen.sample_eligible(r['request']) for r in rows),
        'incomplete, observed, failed or ineligible process')
    matching = all(all(r['request'][k] == rows[0]['request'][k] for k in WORK_KEYS) for r in rows)
    if not matching:
        return dict(performance_eligible=False, decision='inconclusive_work_mismatch',
            limits=comparison_plan()['limits'])
    ratios = []
    for a, b in comparison_plan()['pairs']:
        base, reduced = rows[a]['request'], rows[b]['request']
        ratios.append((reduced['decode_ms'] / reduced['generated_tokens']) /
                      (base['decode_ms'] / base['generated_tokens']))
    geomean = math.exp(statistics.mean(math.log(x) for x in ratios))
    threshold = comparison_plan()['threshold']
    decision = ('reduced_capacity_slower' if all(x > 1 for x in ratios) and geomean > threshold else
                'reduced_capacity_faster' if all(x < 1 for x in ratios) and geomean < 1/threshold else
                'inconclusive_variability_or_effect_below_threshold')
    return dict(performance_eligible=True, paired_one_over_five_token_latency=ratios,
        latency_ratio_geomean=geomean, decision=decision,
        hardware_saturation_qualified=False, model_quality_qualified=False,
        limits=comparison_plan()['limits'])


def verify_pool(log, workers):
    markers = re.findall(r'^strata generate: (\d+) expert-pool workers(.*)$', log, re.M)
    guards.require(markers == [(str(workers), ' + the host thread')],
                   'actual native worker/host configuration differs')
    guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', log),
                   'native/runtime failure')


def prepare(stage, cohort):
    check = stage == 'check'
    previous = gen.committed(RESULT) if RESULT.exists() else None
    if check:
        guards.require(previous is None, 'no automatic retry of an existing capacity experiment')
        base = gen.committed(resources.RESULT)
        guards.require(base['complete'] and base['passed'] and base['ownership_closed'] and
            base['resource_admission_qualified'] and base['artifact_retirement']['complete'] and
            not base['active'] and not base['fresh_faults'], 'close resource capture first')
        source = gen.git('rev-parse', 'HEAD').decode().strip()
        reference = dict(commit=source, path=str(resources.RESULT.relative_to(P)),
            sha256=gen.common.digest(resources.RESULT))
        completed, qualification = [], None
    else:
        guards.require(previous and previous['complete'] and previous['passed'] and
            previous['ownership_closed'] and not previous['active'] and not previous['fresh_faults'] and
            previous['plan'] == comparison_plan() and
            ((cohort == 0 and previous['stage'] == 'check') or
             (cohort > 0 and previous['stage'] == 'clean' and previous['cohort'] == cohort-1)),
            'close the matching check/preceding process without changing the plan')
        reference, source = previous['reference'], previous['source_commit']
        raw = gen.git('show', reference['commit'] + ':' + reference['path'])
        guards.require(hashlib.sha256(raw).hexdigest() == reference['sha256'], 'reference changed')
        base = json.loads(raw)
        completed = previous['processes']
        qualification = previous['check_evidence']
        guards.require(len(completed) == cohort and qualification['passed'], 'missing small qualification')
    workers = 1 if check else ORDER[cohort]
    root = B / ('decode-cpu-capacity-v1-check' if check else f'decode-cpu-capacity-v1-{cohort}')
    record = {k: copy.deepcopy(base[k]) for k in SOURCE_KEYS}
    record.update(schema=1, mode='pool-capacity', stage=stage, cohort=cohort, workers=workers,
        source_commit=source, reference=reference, plan=comparison_plan(), processes=completed,
        check_evidence=qualification, complete=False, passed=False, active=False,
        performance_eligible=False, hardware_saturation_qualified=False, model_quality_qualified=False)
    guards.require(resources.source_proof(record) == record['original_native_inputs_recovered'], 'native proof differs')
    gen.runtime_assets_unchanged(record)
    pins = gen.benchmark_pins(source)
    settings = gen.committed(gen.SETTINGS)
    guards.require(not root.exists(), 'never overwrite/retry a process root')
    tokenizer_assets = {str(Path(base['config']['tokenizer'])/name):
        gen.common.digest(Path(base['config']['tokenizer'])/name)
        for name in ('vocab.json','merges.txt','token_type.json','chat_template.jinja')}
    if not check:
        guards.require(tokenizer_assets == previous['tokenizer_assets'] and
            gen.common.digest(gen.SETTINGS) == previous['settings_sha256'],
            'frozen tokenizer or shared settings changed between processes')
    prompts = check_prompts(base['config'], base['environment']) if check else None
    root.mkdir(mode=0o700)
    for name in ('cache', 'tmp'): (root/name).mkdir()
    old_root = base['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old_root, str(root)) for k, v in base['environment'].items()}
    guards.require(not any(k in env for k in ('UR_ENABLE_LAYERS', 'STRATA_TRACE',
        'STRATA_DECODE_CPU_PROFILE', 'STRATA_DECODE_TIMING')) and
        not any(k.startswith('UR_LOG_') for k in env), 'base must be observer-free')
    for k in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys',
              'EnableDirectSubmission', 'STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[k] == settings['environment'][k], 'qualified safety setting changed')
    if check:
        # Full model initialization previously exhausted the API-trace budget.
        # The runtime's small traced bank check is already recorded at db11be131.
        env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', STRATA_TRACE='1')
        env.update({k: v for k, v in settings['environment'].items()
                    if k.startswith('UR_LOG_') and k != 'UR_LOG_TRACING'})
    config = copy.deepcopy(base['config'])
    flags = config['args']
    guards.require(flags.count('--pool-workers') == 1 and '--no-host-worker' not in flags and
                   flags[flags.index('--pool-workers')+1] == '5', 'unexpected base pool')
    flags[flags.index('--pool-workers')+1] = str(workers)
    flags[flags.index('--expert-profile')+1] = str(root/'expert-profile.bin')
    config.update(log=str(root/'engine.stderr'), env={})
    guards.require(config['exe'] == str(P/'tools/sycl/engine_once.py'), 'one-launch wrapper required')
    shutil.copyfile(P/'data/expert-profile.bin', root/'expert-profile.bin')
    guards.require(gen.common.digest(root/'expert-profile.bin') == base['profile_sha256'], 'profile changed')
    (root/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    if check:
        record['check_prompts'] = prompts
        command = [record['binary']['path'], '--serve', *flags]
    else:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        command = ['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'),
            '--port', str(port), '--long-generation-only', '--think', '--runs', '1',
            '--workload-id', gen.WORKLOAD_ID, '--fail-fast-engine-error', '--ready-timeout', '600',
            '--out', str(root/'bench'), '--long-generation-workload', 'regular-code']
    record.update(root=str(root), environment=env, config=config, command=command,
        tokenizer_assets=tokenizer_assets,
        profile_sha256=base['profile_sha256'], config_sha256=gen.common.digest(root/'config.json'),
        settings_sha256=gen.common.digest(gen.SETTINGS), source_count=len(pins),
        source_aggregate_sha256=hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest(),
        budgets=dict(wall=900 if check else 1800, cpu=54000, rss_cap=96<<30,
                     text_cap=64<<20, file_cap=64<<20, total_cap=64<<20))
    (root/'plan.json').write_text(json.dumps(record, indent=2)+'\n')
    RESULT.write_text(json.dumps(record, indent=2)+'\n')
    print('Prepared '+stage+': commit the exact plan before run.')


def run():
    record = gen.committed(RESULT)
    root = Path(record['root'])
    guards.require(not record['complete'] and record == json.loads((root/'plan.json').read_text()), 'fresh frozen plan required')
    guards.require(not (root/'record.json').exists(), 'no automatic restart')
    pins = gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and
        hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest() == record['source_aggregate_sha256'], 'benchmark source changed')
    for path, expected in [(gen.SETTINGS, record['settings_sha256']),
                           (root/'config.json', record['config_sha256']),
                           (root/'expert-profile.bin', record['profile_sha256'])]:
        guards.require(gen.common.digest(path) == expected, 'measurement input changed: '+str(path))
    guards.require(resources.source_proof(record) == record['original_native_inputs_recovered'], 'native proof changed')
    gen.runtime_assets_unchanged(record)
    check = record['stage'] == 'check'
    for name, expected in record['tokenizer_assets'].items():
        guards.require(gen.common.digest(name) == expected, 'tokenizer asset changed: '+name)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True)
    cursor = next(x.split(': ',1)[1] for x in before.splitlines() if x.startswith('-- cursor: '))
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (root/'record.json').write_text(json.dumps(dict(record, owned_commands=owner.commands if owner else []), indent=2)+'\n')
    def interact(io, entry):
        while not io.line().startswith('READY '): pass
        outputs = []
        for ids in record['check_prompts']:
            io.send(('GEN 16 '+','.join(map(str, ids))+'\n').encode())
            tokens = []
            while True:
                line = io.line()
                guards.require(not line.startswith('ERR'), line)
                if line.startswith('T '): tokens.append(int(line.split()[1]))
                if line.startswith('DONE '):
                    fields = line.split()
                    guards.require(int(fields[1]) == len(tokens) and 0 < len(tokens) <= 16 and
                        fields[5] in ('length','stop','eos') and all(0 <= x < 248320 for x in tokens), 'incomplete small request')
                    outputs.append(dict(done=line, generated=len(tokens), output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()))
                    break
        record['small_requests'] = outputs
        io.send(b'QUIT\n'); io.proc.stdin.close(); io.exit(60)
    owner = supervision.Owner(root/'owned', persist)
    try:
        entry, out, _ = owner.run('check' if check else 'bench', record['command'], record['environment'], P,
            interaction=interact if check else None, **record['budgets'])
        record.update(complete=bool(supervision.closed(entry)), exit_code=entry.get('exit_code'))
        guards.require(record['complete'] and entry['exit_code'] == 0, 'owned command did not close successfully')
        log = (root/'owned/check.stderr' if check else root/'engine.stderr').read_text(errors='replace')
        verify_pool(log, record['workers']); record['runtime_pool_confirmed'] = True
        if not check:
            path, = (root/'bench').glob('*/results.json')
            suite = json.loads(path.read_text())['results']['long_generation']
            rows = gen.collect_samples(suite); record['requests'] = rows
            gen.require_complete_generation(suite, rows, ('regular-code',))
            journal, journal_rows = gen.journal_evidence(root)
            resources.require_journal_binding(rows, journal_rows)
            record['warmups'] = gen.warmup_evidence(root, journal)
            native = json.loads((root/'engine-exit.json').read_text())
            guards.require(native['exit_code'] == 0 and native['boot_id'] == boot and
                native['binary'] == record['binary'] and
                native['args'] == ['--serve', *record['config']['args']] and
                'resource_collector' not in native, 'native launch or observer differs')
            record['native_identity'] = native['native']
            guards.require('strata decode cpu: ' not in log and 'strata decode timing: ' not in log,
                           'clean run emitted observer records')
            record['observer_records_absent'] = True
        gen.runtime_assets_unchanged(record); record['passed'] = True
    except BaseException as error:
        record['error'] = str(error)
        if not check:
            try:
                journal, rows = gen.journal_evidence(root)
                record['request_journal'] = journal
                if 'requests' not in record:
                    record['requests'] = [dict(row, controller_eligible=gen.sample_eligible(row)) for row in rows]
            except Exception as journal_error:
                record['journal_recovery_error'] = str(journal_error)
        for path in (root/'engine.stderr', root/'owned/check.stderr'):
            if path.exists(): record['error_excerpt'] = path.read_text(errors='replace')[-4096:]
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'], text=True)
        record['fresh_faults'] = [x for x in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]
        record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        try: record['driver_after'] = guards.driver_guard()
        except Exception as error: record.update(driver_guard_error=str(error), passed=False)
        record['passed'] = bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])
        if check:
            record['check_evidence'] = dict(source_commit=record['source_commit'], binary=record['binary'],
                workers=1, passed=record['passed'], requests=record.get('small_requests', []),
                ownership_closed=record['ownership_closed'], fresh_faults=record['fresh_faults'],
                limits='Tiny native-protocol safety/configuration check only; no quality, numerical parity, throughput or full-context qualification.')
        else:
            record['processes'].append({k: record.get(k) for k in ('cohort','workers','passed','ownership_closed',
                'fresh_faults','warmups','native_identity','runtime_pool_confirmed','observer_records_absent')})
            requests = record.get('requests', [])
            record['processes'][-1]['request'] = requests[0] if requests else {}
            if record['passed'] and record['cohort'] == 5:
                try:
                    record['comparison_summary'] = summarize_processes(record['processes'])
                    record['performance_eligible'] = record['comparison_summary']['performance_eligible']
                except Exception as error: record.update(summary_error=str(error), passed=False)
        persist(); RESULT.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: record.get(k) for k in ('stage','cohort','workers','complete','passed','ownership_closed','error')}))
    return 0 if record['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run'))
    parser.add_argument('stage', nargs='?', choices=('check','clean'))
    parser.add_argument('cohort', nargs='?', type=int, choices=range(6), default=0)
    args = parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([]); guards.driver_guard()
        if args.action == 'prepare':
            guards.require(args.stage is not None, 'prepare requires check or clean')
            return prepare(args.stage, args.cohort)
        guards.require(args.stage is None, 'run uses the frozen committed stage')
        return run()


if __name__ == '__main__': sys.exit(main())
