#!/usr/bin/env python3
"""Run the maintained long/agent/thinking suite with an isolated store switch."""

import fcntl
import json
from pathlib import Path
import re
import select
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
import generation_engine as gen
import coherent_prefill_bench as historical
import official_suite_evidence as evidence_reader
import prefill_attention_bank_restore as restore
import qualify_qsa_k_model as official
import ur_validation as validation

P, B, RESULT, basis = restore.P, restore.B, restore.RESULT, restore.basis
build, require, sha, signature = restore.build, restore.require, restore.sha, restore.signature
ROOT, KEY = B / 'prefill-attention-bank-generation-v3', 'prefill_attention_bank_generation_v3'
THIS = 'tools/sycl/prefill_attention_bank_generation.py'
RESTORE_EVIDENCE = 'f2a044c7ff3f332327e7af1e748bfb3664c9c231'
PREVIOUS = 'prefill_attention_bank_generation_v2'
ORDINARY = 'prefill_ordinary_bank_comparison_v1'
ORDER = ('original', 'candidate', 'candidate', 'original', 'original', 'candidate')
WORKLOAD_ID = 'attention_bank_admission_20261119'
SELECTOR = restore.SELECTOR
OWNER = restore.owner_module
LIMITS = dict(wall=5400, cpu=21600, rss_cap=104 << 30, text_cap=24 << 20,
              file_cap=64 << 20, total_cap=160 << 20)


def reference():
    native, speed = restore.reference()
    prior = build.historic(RESTORE_EVIDENCE, restore.KEY)
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete']
            and prior['ownership_closed'] and not prior['active'] and prior['binary'] == native['binaries']['strata']
            and all(v['passed'] and v['ownership_closed'] and v['fresh_faults'] == []
                    for v in prior['outcomes'].values()), 'exact same-ELF server RESTORE gate required')
    return native, speed


def ordinary_controls(native):
    row = validation.committed(RESULT, (ORDINARY,))
    require(row['stage'] == 'closed' and row['complete'] and row['passed'] and row['ownership_closed']
            and not row['active'] and len(row['processes']) == len(row['plans']) == 9
            and all(p['complete'] and p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in row['processes']), 'complete all nine ordinary 64K processes')
    controls = {}
    for index, name, variant in ((1, 'original', 'copy'), (2, 'candidate', 'direct')):
        plan, done = row['plans'][index], row['processes'][index]
        require(plan['variant'] == variant and done['binary'] == native['binaries']['strata']
                and done['native_source_commit'] == native['source_commit'] and len(done['routes']) == 3,
                'exact ordinary same-ELF copy/direct controls required')
        controls[name] = {k: plan[k] for k in ('root', 'environment_sha256')}
    return controls


def pins(revision):
    sources = basis.core.pins(revision)
    sources.update(basis.comparison.build.controls(revision, controller=P / THIS))
    for name in official.DOCS + basis.POLICIES:
        committed = gen.git('ls-tree', '--name-only', revision, '--', name).strip()
        if committed:
            raw = gen.git('show', revision + ':' + name)
            require((P / name).read_bytes() == raw, 'committed agent or policy input changed: ' + name)
            sources[name] = sha(P / name)
        else:
            require(not (P / name).exists(), 'uncommitted agent or policy input: ' + name)
            sources[name] = None
    return sources


def environment(row, plan):
    env = historical.context().environment(row, dict(root=plan['root'], index=0, heads=1))
    env[SELECTOR] = str(int(plan['arm'] == 'candidate'))
    control = row['ordinary_controls'][plan['arm']]
    rebased = {k: v.replace(plan['root'], control['root']) for k, v in env.items()}
    require(basis.comparison.fingerprint(rebased) == control['environment_sha256']
            and env['STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS'] == '384'
            and env.get('STRATA_SYCL_PREFILL_MOE_PREFIX_ROUTER', 'off') == 'off'
            and env.get('STRATA_SYCL_PREFILL_PREFIX_EXPERT_COPY', 'sycl') == 'sycl'
            and env.get('STRATA_SYCL_PREFILL_PREFIX_EXPERT_DEFER_COMPLETION', '0') == '0',
            'full execution environment must match the checked ordinary route')
    env['PATH'] = str(Path(row['python_selected']['path']).parent) + ':/usr/bin:/bin'
    env.pop('STRATA_SNAPSHOT_VERIFY', None)
    require(not any(k in env for k in ('STRATA_TRACE', 'STRATA_STATE_HASH', 'STRATA_DECODE_TIMING',
                                      'STRATA_DRAFT_TIMING', 'STRATA_SNAPSHOT_VERIFY', 'LD_PRELOAD')),
            'benchmark observers must be disabled except the qualified startup validation')
    return env


def configuration(row, plan):
    return dict(row['configuration'], log=str(Path(plan['root']) / 'engine.stderr'))


def command(plan):
    root = Path(plan['root'])
    return ['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(plan['port']),
            '--long', '--long-generation', '--think', '--runs', '1', '--workload-id', WORKLOAD_ID,
            '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')]


def invariant(row, plan):
    env = environment(row, plan)
    env.pop(SELECTOR)
    return signature(json.dumps(dict(config=configuration(row, plan), environment=env), sort_keys=True)
                     .replace(plan['root'], '<process>'))


def save(row):
    build.stage.write(ROOT / 'record.json', row)
    result = build.read(RESULT)
    result[KEY] = basis.compact(row)
    build.stage.write(RESULT, result)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(RESULT), 'fresh full suite only')
    native, base = reference()
    failed = validation.committed(RESULT, (PREVIOUS,))
    require(failed['stage'] == 'failed' and not failed['passed'] and not failed['active']
            and failed['ownership_closed'] and len(failed['processes']) == 1
            and failed['binary'] == native['binaries']['strata'] and failed['processes'][0]['fresh_faults'],
            'preserve the failed 512-slot early-copy suite')
    revision = validation.git('rev-parse', 'HEAD').decode().strip()
    config = dict(base['configuration'], exe=str(P / 'tools/sycl/engine_once.py'), env={}, cwd=str(P))
    args = config['args']
    for flag, value in (('--max-context', '262144'), ('--expert-cache', '1256'), ('--kv-resident', '20480'),
                        ('--spec', '4'), ('--pcie-frac', '0'), ('--adapt-swaps', '0'),
                        ('--conversation-cache-mib', '0'), ('--prompt-cache', '0')):
        require(args.count(flag) == 1 and args[args.index(flag)+1] == value, 'qualified fixed argument differs: ' + flag)
    contract, manifest = basis.core.frontend.python_manifest()
    row = {k:base[k] for k in ('template','environment_template','model_assets','runtime_manifest',
                                'native_validation_probe','boot','driver')}
    row.update(stage='prepared', source_commit=revision, sources=pins(revision), configuration=config,
               binary=native['binaries']['strata'], native_source_commit=native['source_commit'],
               native_build_sha256=signature(build.builder.compact(native)), restore_evidence=RESTORE_EVIDENCE,
               previous_failed_comparison=PREVIOUS, ordinary_controls=ordinary_controls(native),
               python_contract=contract, python_selected=manifest['selected'], plans=[], processes=[],
               budgets=LIMITS, host_passed=False, active=False, ownership_closed=True, passed=False, complete=False,
               plan=dict(unit='Verified ordinary 384-slot configuration in both arms. Switch only the prefill store.',
                         execution='Six independent official bench.sh --long --long-generation --think --runs 1 '
                                   'processes in three alternating pairs. Every process includes the six-turn agent session.',
                         acceptance='Closed runtime/owner/fault, complete official v5 journal, ordinary 384-slot '
                                    'bank and actual store for the prior 16098-token prompt and both sustained '
                                    'workloads, matched fixed requests. Keep exact lengths and drafts. '
                                    'Report per-workload/arm MTP classes only if every eligible sample meets its threshold; '
                                    'screen per-workload and agent decode preservation by paired TG geometric mean >=1. '
                                    'Claim preservation for comparable decode work when all paired generated and MTP '
                                    'counts also match, including the actual agent input and reuse counts.',
                         scope='Agent histories may differ because they replay their own previous replies. Fixed long inputs '
                               'need >=32768 fresh tokens, >=1024 output including thinking and a visible answer. '
                               'The upstream nominal 32K prompt remains supplemental. This does not adopt an engine.',
                         stop='Stop on first safety, ownership, source/core identity, eligibility or finite budget failure. '
                              'Retain original samples and fastest/slowest pairs. No speed-based stopping or replacement.'))
    ROOT.mkdir(mode=0o700)
    for index, arm in enumerate(ORDER):
        root = ROOT / str(index)
        for name in ('tmp', 'cache'):
            (root / name).mkdir(parents=True, mode=0o700)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        plan = dict(index=index, arm=arm, root=str(root), port=port, binary=row['binary'])
        build.stage.write(root / 'config.json', configuration(row, plan))
        build.stage.write(root / 'plan.json', dict(binary=row['binary'], native_validation_probe=row['native_validation_probe']))
        plan.update(command=command(plan), generated={name:sha(root/name) for name in ('config.json','plan.json')},
                    environment_sha256=signature(environment(row, plan)))
        row['plans'].append(plan)
    row['invariant_sha256'] = invariant(row, row['plans'][0])
    require(all(invariant(row, p) == row['invariant_sha256'] for p in row['plans']), 'unplanned comparison difference')
    save(row)


def current(row):
    require(validation.committed(RESULT, (KEY,)) == basis.compact(row) and pins(row['source_commit']) == row['sources'],
            'commit exact benchmark inputs and comparison state')
    native, speed = reference()
    require(row['native_build_sha256'] == signature(build.builder.compact(native))
            and row['ordinary_controls'] == ordinary_controls(native)
            and row['binary'] == native['binaries']['strata'] and row['native_source_commit'] == native['source_commit']
            and all(row[k] == speed[k] for k in ('template','environment_template','model_assets','runtime_manifest',
                                               'native_validation_probe','boot','driver'))
            and sha(row['template']['path']) == row['template']['sha256'], 'executable or workload base changed')
    for index, (plan, arm) in enumerate(zip(row['plans'], ORDER, strict=True)):
        root = ROOT / str(index)
        require(plan['index'] == index and plan['arm'] == arm and plan['root'] == str(root)
                and plan['binary'] == row['binary'] and plan['command'] == command(plan)
                and plan['generated'].keys() == {'config.json','plan.json'}
                and build.read(root/'config.json') == configuration(row, plan)
                and build.read(root/'plan.json') == dict(binary=row['binary'], native_validation_probe=row['native_validation_probe'])
                and all(sha(root/name) == h for name,h in plan['generated'].items())
                and signature(environment(row, plan)) == plan['environment_sha256']
                and invariant(row, plan) == row['invariant_sha256'], 'frozen generated arm differs')
    contract, manifest = basis.core.frontend.python_manifest()
    require(contract == row['python_contract'] and manifest['selected'] == row['python_selected'], 'benchmark runtime changed')
    return build.read(Path(row['runtime_manifest']['path'])), manifest


def routes(text, samples, arm):
    require('strata MoE prefix expert upload layer:' not in text and
            'strata MoE prefix router: requested=early' not in text,
            'unexpected early expert upload on ordinary route')
    heads = list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read'
                             r'[^\n]*, (\d+) generated[^\n]*$', text, re.M))
    require(len(heads) == len(samples), 'native request footer count differs')
    start, result = 0, []
    for head, sample in zip(heads, samples, strict=True):
        prompt, reused, read, generated = map(int, head.groups())
        require((prompt,reused,read,generated) ==
                tuple(sample[k] for k in ('prompt_tokens','reused_tokens','read_tokens','generated_tokens')),
                'native footer and API counters differ')
        frame = text[start:head.start()]
        start = head.end()
        if sample['phase'] == 'prompt/16K' or sample['phase'].startswith('long-generation/'):
            require(sample['phase'] != 'prompt/16K' or read == 16098, 'the previous 16K stall shape changed')
            route = restore.require_store(frame, dict(direct=arm=='candidate'), dict(read=read))
            banks = re.findall(r'^strata MoE expert bank: requested_slots=(\d+) effective_slots=(\d+) '
                               r'staged=(\d+) host_slab_bytes=\d+ device_bytes=(\d+) ', frame, re.M)
            require(banks == [('384', '384', '1', '1022361600')] and route['gpu_rows'] == read-1
                    and route['window_mode'] == 10, 'expected ordinary all-GPU 384-slot residual route')
            result.append(dict(phase=sample['phase'], read=read, **route))
    require(len(result) == 3, 'ragged 16K and both sustained generation routes required')
    return result


def run(index):
    row = build.read(ROOT / 'record.json')
    providers, manifest = current(row)
    require(row['host_passed'] and row['ownership_closed'] and not row['active']
            and row['stage'] in ('ready','running') and len(row['processes']) == index < len(ORDER),
            'complete previous arms in order')
    plan = row['plans'][index]; root = Path(plan['root']); env = environment(row, plan)
    require(not (root/'execution.json').exists() and not (root/'launch.json').exists(), 'inspect the prior attempt')
    cursor = next(line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
        if line.startswith('-- cursor: '))
    outcome = dict(index=index, arm=plan['arm'], complete=False, host_probe_accepted=False)
    owner = None
    start_clock = time.clock_gettime_ns(time.CLOCK_BOOTTIME) - time.monotonic_ns()

    def persist():
        build.stage.write(root/'execution.json', dict(outcome=outcome,active=bool(owner and owner.active),
                                                      commands=owner.commands if owner else []))

    def observe(io, entry):
        checked = False
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            require(abs(time.clock_gettime_ns(time.CLOCK_BOOTTIME) - time.monotonic_ns() - start_clock) <= 50_000_000,
                    'suspend or clock discontinuity')
            if select.select([io.proc.stdout], [], [], .05)[0]:
                io.read()
            while b'\n' in io.pending:
                line, _, remaining = io.pending.partition(b'\n'); io.pending[:] = remaining
                text = line.decode().strip()
                if text.startswith('Running the suite'):
                    require(not checked, 'duplicate suite start')
                    outcome['native_identity'] = official.native_live(root, plan, row, env,
                        dict(runtime_files=providers), manifest, entry)
                    checked = True
                    persist()
                if text.startswith(('Running the suite','agent:','long generation:','32K prompt:','run ')):
                    print(f'Process{index} {plan["arm"]}: {text}',flush=True)
        require(checked, 'live provider identity not observed')
        io.pending.clear()
        io.proc.stdin.close()
        io.exit(1)

    try:
        owner = OWNER.Owner(root/'owned', persist)
        entry, _, _ = owner.run('bench', plan['command'], env, P, interaction=observe, **row['budgets'])
        require(OWNER.closed(entry) and entry['exit_code'] == 0,
                'official bench exit/ownership failed: ' + entry.get('error', 'output or return code'))
        evidence = evidence_reader.read_complete_suite(root, WORKLOAD_ID)
        outcome['evidence'] = evidence
        if row['processes']:
            official.matched(evidence['samples'], row['processes'][0]['evidence']['samples'])
        end = gen.native_end(root,row,dict(config=configuration(row,plan)),row['boot'],entry['pid'])
        log = (root/'engine.stderr').read_text()
        probe = validation.preload_evidence(log,end['native']['pid'])
        require(probe['loader'] == row['native_validation_probe']['loader']['path']
                and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain',log),
                'unexpected engine/runtime validation')
        basis.comparison.require_quiet_validation(root)
        outcome.update(routes=routes(log,evidence['samples'],plan['arm']),validation=probe,native_exit_code=0)
        current(row)
        if index+1 == len(ORDER):
            basis.comparison.shared.models.model_pins(row['configuration'],row['model_assets'],full=True)
        outcome.update(complete=True,host_probe_accepted=True,postconditions_complete=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if owner is not None:
            validation.finish(outcome,owner,cursor,row['boot'])
        outcome['passed'] = official.accepted(outcome) and outcome.get('driver_after') == row['driver']
        row['processes'].append(outcome)
        done = outcome['passed'] and index+1 == len(ORDER)
        row.update(stage='closed' if done else 'running' if outcome['passed'] else 'failed',
                   ownership_closed=outcome.get('ownership_closed',False),
                   active=bool(owner and owner.active),passed=done,complete=done)
        if done:
            row['decision'] = evidence_reader.decide(row['processes'])
        persist()
        save(row)
    print(json.dumps({k:outcome.get(k) for k in ('index','arm','passed','complete','fresh_faults','error')}),flush=True)
    require(outcome['passed'],'preserve the closed official comparison outcome')


if __name__ == '__main__':
    require(sys.executable == basis.reference.model.PYTHON, 'use pinned Python')
    action, = sys.argv[1:]
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        elif action == 'host':
            basis.comparison.host(SimpleNamespace(ROOT=ROOT,current=current,environment=environment,save=save))
        else:
            run(int(action))
