#!/usr/bin/env python3
"""Phase-specific PP/TTFT preservation after the matched KV decode campaign.

Main-authored from this fork's owned benchmark policies; repository LICENSE.
No native rebuild, XeStrata input, runtime environment mutation or adoption.
"""
import copy
import fcntl
import json
import math
import re
from pathlib import Path
import socket
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import decode_kv_bench as kv

P, B, RESULT = kv.P, kv.B, kv.RESULT
ROOT, KEY = B/'decode-kv-prefill-v1', 'decode_kv_prefill_v1'
THIS = 'tools/sycl/decode_kv_prefill.py'
PYTHON, RUNS = kv.PYTHON, 3
# Three adjacent pairs per length, with the middle pair reversed.
PAIR_READS = (32769, 65537, 65537, 32769, 32769, 65537)
CASES = [(read, arm) for i, read in enumerate(PAIR_READS)
         for arm in (('candidate', 'reference') if i in (2, 3) else ('reference', 'candidate'))]
ORDER = [(read, 1) for read, _ in CASES]
require, sha, signature = kv.require, kv.sha, kv.signature
compact, environment, configuration = kv.compact, kv.environment, kv.configuration
comparison, shared, build = kv.runner.comparison, kv.shared, kv.build
UNIT_TESTS = [
    'tools.test_benchmark_evidence.EvidenceTests.test_kv_prefill_medians_keep_first_and_warmed_independent',
    'tools.test_benchmark_evidence.EvidenceTests.test_kv_prefill_refuses_incomplete_or_nonfinite_cohorts']


def save(row):
    kv.core.write(ROOT/'record.json', row)
    value = json.loads(RESULT.read_text()); value[KEY] = compact(row); kv.core.write(RESULT, value)


def basis(expected=None):
    native = kv.device.native(expected, observers={kv.device.THIS, kv.model.THIS, kv.THIS, THIS},
        retained_frontend=kv.FRONTEND)
    prior = json.loads((kv.ROOT/'record.json').read_text())
    require(kv.device.common.validation.committed(RESULT, (kv.KEY,)) == compact(prior)
        and prior['stage'] == 'closed' and not prior['active']
        and all(prior[k] for k in ('passed', 'complete', 'ownership_closed'))
        and len(prior['processes']) == 6 and prior['decision'] == kv.decision(prior['processes'])
        and all(v['reproducible_TG_gain'] and v['reference_MTP_class_confirmed']
                for v in prior['decision']['workloads'].values())
        and all(v['native_exit_code'] == 0 and v['ownership_closed'] and not v['fresh_faults']
                and v['passed'] and v['complete'] for v in prior['processes'])
        and sha(P/kv.THIS) == prior['sources'][kv.THIS]
        and prior['binary'] == native['binaries']['strata']
        and prior['native_source_commit'] == native['native_source_commit']
        and prior['native_validation_probe'] == native['native_validation_probe'],
        'closed matched decode evidence and exact native build required')
    for key in (kv.device.KEY, kv.model.KEY):
        proof = kv.device.common.validation.committed(RESULT, (key,))
        require(proof['stage'] == 'closed' and not proof['active']
            and all(proof[k] for k in ('passed', 'complete', 'ownership_closed'))
            and proof['native_source_commit'] == native['native_source_commit'],
            'same-build closed device and model handoff required')
    manifest = prior['runtime_manifest']
    providers = json.loads(Path(manifest['path']).read_text())
    require(sha(manifest['path']) == manifest['sha256']
        and all(sha(p) == h for p, h in providers.items())
        and build.guards.driver_guard() == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
        'qualified runtime, driver or boot changed')
    shared.models.model_pins(prior['configuration'], prior['model_assets'], full=False)
    return native, prior, providers


def command(plan):
    root = Path(plan['root'])
    return ['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'),
        '--port', str(plan['port']), '--prefill-exact-tokens', str(plan['read']),
        '--think', '--runs', str(RUNS), '--profile-request-gates',
        '--workload-id', 'decode_kv_prefill_20261114', '--fail-fast-engine-error',
        '--ready-timeout', '600', '--out', str(root/'bench')]


def invariant(row, plan):
    value = dict(config=configuration(row, plan), environment=environment(row, plan))
    value['environment']['STRATA_SYCL_DECODE_KV_COALESCED'] = '<arm>'
    return signature(json.dumps(value, sort_keys=True).replace(plan['root'], '<process>'))


def current(row):
    require(kv.device.common.validation.committed(RESULT, (KEY,)) == compact(row)
        and kv.core.pins(row['source_commit']) == row['sources']
        and all(sha(P/p) == h for p, h in row['policies'].items()),
        'freeze committed comparison inputs and state')
    native, prior, files = basis(row['native_observer_admission'])
    require(row['prior_decode_evidence_sha256'] == signature(compact(prior))
        and row['binary'] == native['binaries']['strata']
        and row['native_source_commit'] == native['native_source_commit']
        and all(row[k] == prior[k] for k in ('configuration', 'environment_template', 'template',
            'boot', 'driver', 'runtime_manifest', 'model_assets', 'native_validation_probe')),
        'prior evidence, build or configuration changed')
    contract, python = kv.runner.model.python_manifest()
    require(contract == row['python_contract'] and python['selected'] == row['python_selected']
        and sha(row['template']['path']) == row['template']['sha256']
        and build.supervision.artifact_bytes(ROOT) <= row['whole_bytes']
        and len(row['plans']) == len(CASES), 'frontend, template or budget differs')
    for i, plan in enumerate(row['plans']):
        root = Path(plan['root'])
        require(plan['index'] == i and (plan['read'], plan['kernel']) == CASES[i]
            and plan['heads'] == 1 and plan['adapt_swaps'] == 0
            and plan['command'] == command(plan) and invariant(row, plan) == row['invariant_signature']
            and json.loads((root/'config.json').read_text()) == configuration(row, plan)
            and json.loads((root/'plan.json').read_text()) == dict(binary=row['binary'],
                native_validation_probe=row['native_validation_probe'])
            and all(sha(root/p) == h for p, h in plan['generated'].items())
            and comparison.fingerprint(environment(row, plan)) == plan['environment_sha256'],
            'frozen process, configuration, workload or environment differs')
    if row['host_passed']:
        unit = row.get('host_unit_evidence', {})
        require(unit.get('passed') is True and unit.get('ownership_closed') is True
            and unit.get('exit_code') == 0 and unit.get('tests') == UNIT_TESTS
            and unit.get('source_commit') == row['source_commit'], 'closed source-pinned unit controls required')
    return files, python


def host():
    comparison.host(context())
    row = json.loads((ROOT/'record.json').read_text())
    owner = build.supervision.Owner(ROOT/'unit')
    evidence = dict(passed=False, source_commit=row['source_commit'], tests=UNIT_TESTS)
    try:
        args = [PYTHON, '-B', '-m', 'unittest', *UNIT_TESTS]
        entry, out, err = owner.run('unit', args, environment(row,row['plans'][0]), P,
            wall=90,cpu=60,rss_cap=1<<30,text_cap=128<<10,file_cap=1<<20,total_cap=4<<20)
        text = out.read_text()+err.read_text()
        require(build.supervision.closed(entry) and entry['exit_code']==0
            and re.search(r'Ran 2 tests in [0-9.]+s',text) and text.rstrip().endswith('OK'),
            'preservation unit controls failed')
        evidence.update(passed=True,exit_code=0)
    except BaseException as error:
        evidence['error']=repr(error)
    finally:
        evidence['ownership_closed'] = owner.active is None and all(
            build.supervision.closed(v) for v in owner.commands)
        row['host_unit_evidence']=evidence
        row['host_passed']=bool(evidence['passed'] and evidence['ownership_closed'])
        if not row['host_passed']:row.update(stage='failed',passed=False,complete=False)
        save(row)
    require(row['host_passed'], 'unit controls or ownership did not qualify')


def native_context(row, plan):
    require((plan['read'], plan['kernel']) == CASES[plan['index']], 'unplanned phase comparison')
    return {k: row[k] for k in ('binary', 'native_source_commit', 'native_validation_probe', 'boot')}


def request_route(text, plan):
    out = kv.base.prefill.context().request_route(text, plan)
    out['KV_captures'] = kv.kv_captures(text, plan['kernel'] == 'candidate')
    require(any(x['phase'] == 'decode' and x['width'] == 4 for x in out['KV_captures']),
        'actual decode4 capture required even for PP-only request')
    footers = kv.base.copy_records(text, arm='reference')
    require(len(footers) == 1 and footers[0]['read_tokens'] == plan['read']
        and footers[0]['generated_tokens'] == 24, 'single fresh24-output native extent required')
    kv.base.clean_observers(text)
    return out


def check_sample(sample, count, read, minimum_ms=10000):
    comparison.check_sample(sample, count, read, minimum_ms)
    require(sample['generated_tokens'] == 24, 'exact24outputs required')


def decision(processes):
    require(len(processes) == len(CASES) and all(p['passed'] and p['complete']
        and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
        and p['index'] == i and (p['read'], p['kernel']) == CASES[i]
        and len(p['samples']) == RUNS
        and all(type(s.get(field)) in (int, float) and math.isfinite(s[field]) and s[field] > 0
            for s in p['samples'] for field in ('prompt_ms', 'ttft_s'))
        for i, p in enumerate(processes)), 'all twelve processes/thirty-six samples required')
    points = []
    for read in (32769, 65537):
        pairs = [{p['kernel']: p for p in processes[i:i+2]} for i in range(0, len(CASES), 2)
                 if CASES[i][0] == read]
        scopes = {}
        for scope in ('first', 'warmed'):
            metrics = {}
            for field in ('prompt_ms', 'ttft_s'):
                values = {arm: [p[arm]['samples'][0][field] if scope == 'first' else
                    statistics.median(s[field] for s in p[arm]['samples'][1:]) for p in pairs]
                    for arm in ('reference', 'candidate')}
                require(all(type(v) in (int, float) and math.isfinite(v) and v > 0
                    for group in values.values() for v in group), 'invalid preservation clock')
                ratios = [c/r for c, r in zip(values['candidate'], values['reference'])]
                ratio = statistics.median(values['candidate'])/statistics.median(values['reference'])
                metrics[field] = dict(values, paired_ratios=ratios, median_ratio=ratio,
                    paired_geometric_mean=math.exp(statistics.mean(math.log(v) for v in ratios)),
                    median_not_slower=ratio <= 1, every_pair_not_slower=all(v <= 1 for v in ratios))
            scopes[scope] = metrics
        points.append(dict(read=read, scopes=scopes))
    return dict(points=points, median_preservation_screen_passed=all(v['median_not_slower']
        for p in points for scope in p['scopes'].values() for v in scope.values()),
        inference_adopted=False, proof_of_zero_effect_or_physical_saturation=False,
        task_agent_physical262144_RESTORE_ordinary_teardown_pending=True)


def context():
    return SimpleNamespace(ROOT=ROOT, KEY=KEY, ORDER=ORDER, RUNS=RUNS, current=current,
        environment=environment, native_context=native_context, request_route=request_route,
        check_sample=check_sample, decision=decision, compact=compact, save=save,
        validate_model_text=kv.runner.generation.gdn.validate_model_text)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh committed campaign only')
    native, prior, _ = basis()
    row = {k: copy.deepcopy(prior[k]) for k in ('configuration', 'environment_template', 'template',
        'boot', 'driver', 'runtime_manifest', 'model_assets', 'native_validation_probe')}
    revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    row.update(stage='prepared', source_commit=revision, sources=kv.core.pins(revision),
        policies={p: sha(P/p) for p in ('AGENTS.md', 'docs/ARTIFACT_RETENTION.md')},
        binary=native['binaries']['strata'], native_source_commit=native['native_source_commit'],
        native_observer_admission=native['observer_admission'],
        prior_decode_evidence_sha256=signature(compact(prior)),
        python_contract=prior['python_contract'], python_selected=prior['python_selected'],
        plans=[], processes=[], request_hashes={}, host_passed=False,
        passed=False, complete=False, active=False, ownership_closed=True, whole_bytes=832<<20,
        budgets=dict(wall=1800, cpu=16200, rss_cap=104<<30, text_cap=16<<20, file_cap=32<<20, total_cap=64<<20),
        plan=dict(question='Does decode-only cooperative KV selection preserve first and warmed32K/64K PP and clientTTFT?',
            evidence='The preceding same-binary campaign must reproduce TG gains in all three pairs for both observed MTP classes. Keep its first-use PP/TTFT increases unresolved;do not dismiss them as noise or count them as accepted costs.',
            order='Twelve fresh processes:32R,32C,64R,64C,64C,64R,32C,32R,32R,32C,64R,64C. Each uses the official single short warmup and three distinct-nonce exact long requests,thinking enabled,24outputs. First long GPU use and subsequent2 medians separate. TTFT is post-count-preflight;model startup and count construction are outside request timing. Commit each closed process before the next.',
            constants='Same330405engine/13ELFs,static1256slots/5workers,pcie0/adapt0,spec4,logical262144/resident20480K8V8,8192throughput BF16 ordered-exp/bulk1/staging0/largeGU,head2/dense1. Only pre-exec DECODE_KV_COALESCED0/1 differs;copy-groups0. Qualified safety and quiet retained-reference validation remain charged.',
            acceptance='Source/build/recipe/compiler/provider/model/owner identities before each launch;fresh exact32769or65537read/zero reuse,thinking/24outputs,valid clocks/clientTTFT,journal and matched bodies. Actual typed largeGU/bank/handoff and12-QSA scalar/cooperative phase captures includingdecode4;zero expert copies,normal0/closed ownership/quiet validation/no freshfault/unchangedboot-driver. Reuse immediately preceding closed cohort final15full hashes only after identity checks;rehash all15after process11.',
            decision='Retain every sample and allthree paired C/R ratios for first and warmed prompt-time/TTFT at each length. Median screen passes only if candidate median is no slower for each of eight series;no tolerated regression or speed-based exclusion. Report all-pair consistency separately. A screen pass does not prove zero effect or adopt the model;any failed series keeps preservation unresolved. No revised criteria or extra samples within this campaign.',
            stopping='First identity,source,owner,provider,count,clock,route,validation,fault,budget or eligibility failure stops;exclusive attempt before preflight,no retry or pin refresh.',
            retention='No new native build;retain14508fallback and330405candidate through the immediate phase-preservation decision. Keep compact36samples and source/configuration/route evidence;retire successful raw output/cache at verified closure. Legacy provenance holds remain separate. No TG,agent/task,physical262144/RESTORE,ordinary teardown or hardware ceiling follows.'))
    ROOT.mkdir(mode=0o700)
    for i, (read, arm) in enumerate(CASES):
        root=ROOT/str(i);root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root/name).mkdir()
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0));port=sock.getsockname()[1]
        plan=dict(index=i, read=read, kernel=arm, heads=1, adapt_swaps=0, root=str(root), port=port)
        kv.core.write(root/'config.json', configuration(row, plan))
        kv.core.write(root/'plan.json', dict(binary=row['binary'], native_validation_probe=row['native_validation_probe']))
        plan.update(generated={n:sha(root/n) for n in ('config.json','plan.json')}, command=command(plan))
        plan['environment_sha256']=comparison.fingerprint(environment(row,plan));row['plans'].append(plan)
    row['invariant_signature']=invariant(row,row['plans'][0])
    require(all(invariant(row,p)==row['invariant_signature'] for p in row['plans']), 'undeclared arm difference')
    save(row);print('Prepared twelve PP/TTFT preservation processes;commit before host/native.',flush=True)


if __name__ == '__main__':
    require(sys.executable == PYTHON, 'use qualified benchmark interpreter')
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', *map(str,range(len(CASES)))),
        'prepare|host|0..11')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build.guards.no_model_owners([])
        action=sys.argv[1]
        if action=='prepare': prepare()
        else:
            with (ROOT/('host-attempt.json' if action=='host' else action+'/attempt.json')).open('x') as stream:
                json.dump(dict(action=action,result_key=KEY),stream)
            try:
                host() if action=='host' else comparison.run(int(action),context())
            except BaseException as error:
                row=json.loads((ROOT/'record.json').read_text())
                if row['stage']!='failed':row.update(stage='failed',passed=False,complete=False,error=repr(error),failed_action=action);save(row)
                raise
