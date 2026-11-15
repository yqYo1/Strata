#!/usr/bin/env python3
"""Matched official PP/TTFT comparison of two retained coherent generations.

Main-authored from owned benchmark and original-build readers; repository
LICENSE. No native rebuild, XeStrata input, environment mutation or adoption.
The preceding same-binary KV flag screen stays failed and unchanged.
"""
import copy
import fcntl
import json
from pathlib import Path
import re
import socket
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import coherent_original_reuse as original
import decode_kv_prefill as previous

kv = previous.kv
P, B, RESULT = previous.P, previous.B, previous.RESULT
ROOT, KEY = B/'coherent-generations-prefill-v1', 'coherent_generations_prefill_v1'
THIS = 'tools/sycl/compare_coherent_generations.py'
CASES, ORDER, RUNS, PYTHON = previous.CASES, previous.ORDER, previous.RUNS, previous.PYTHON
require, sha, signature = previous.require, previous.sha, previous.signature
compact, comparison, shared, build = previous.compact, previous.comparison, previous.shared, previous.build
SOURCES = dict(reference=original.SOURCE, candidate='1eec28c77f393d34814dc748502113514973ea8e')
SHAS = dict(reference='dcb8efd4698d4c9c6de913cb5f92e2229e945b242f24bd1fca2998e4cf89e6d1',
    candidate='330405370dab1d52995540e79c9dad5a3decf46d4e2007ab096dcc57e602d21e')
OBSERVERS = {kv.device.THIS, kv.model.THIS, kv.THIS, previous.THIS,
    'tools/sycl/coherent_original_reuse.py', THIS}
UNIT_TESTS = [
    'tools.test_coherent_generations.GenerationBindingTests',
    *previous.UNIT_TESTS]


def save(row):
    kv.core.write(ROOT/'record.json', row)
    value = json.loads(RESULT.read_text()); value[KEY] = compact(row); kv.core.write(RESULT, value)


def basis(expected=None):
    expected = expected or {}
    newer = kv.device.native(expected.get('candidate'), observers=OBSERVERS,
        retained_frontend=kv.FRONTEND)
    older, proof = original.verify_original(expected=expected.get('reference'))
    prior = json.loads((previous.ROOT/'record.json').read_text())
    require(kv.device.common.validation.committed(RESULT, (previous.KEY,)) == compact(prior)
        and prior['stage'] == 'closed' and not prior['active']
        and all(prior[k] for k in ('passed', 'complete', 'ownership_closed'))
        and len(prior['processes']) == 12 and prior['decision'] == previous.decision(prior['processes'])
        and not prior['decision']['median_preservation_screen_passed']
        and prior['binary'] == newer['binaries']['strata']
        and prior['native_source_commit'] == SOURCES['candidate'],
        'preserve closed failed KV flag screen;this is a different default0 build question')
    for plan in prior['plans']:
        require(all(sha(Path(plan['root'])/p) == h for p,h in plan['generated'].items()),
            'original matched-workload configuration/launcher plans changed')
    require(all(sha(P/p) == h for p,h in prior['sources'].items()),
        'prior benchmark/controller inputs changed')
    for key in (kv.device.KEY, kv.model.KEY):
        evidence = kv.device.common.validation.committed(RESULT, (key,))
        require(evidence['stage'] == 'closed' and not evidence['active']
            and all(evidence[k] for k in ('passed', 'complete', 'ownership_closed'))
            and evidence['native_source_commit'] == SOURCES['candidate'],
            'closed current-generation device/model handoff required')
    old_speed = kv.device.common.validation.committed(RESULT, ('coherent_prefill_speed_v1',))
    require(older['native_validation_probe'] == newer['native_validation_probe']
        == prior['native_validation_probe'] and old_speed['boot'] == prior['boot']
        and old_speed['driver'] == prior['driver'] and old_speed['configuration'] == prior['configuration']
        and old_speed['model_assets_sha256'] == signature(prior['model_assets']),
        'both original generations must bind the same runtime,configuration and model')
    manifest = prior['runtime_manifest']; files = json.loads(Path(manifest['path']).read_text())
    old_files = json.loads(Path(proof['runtime_manifest']['path']).read_text())
    require(files == old_files and sha(manifest['path']) == manifest['sha256']
        and all(sha(p) == h for p,h in files.items())
        and build.guards.driver_guard() == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
        'qualified providers,driver or boot changed')
    shared.models.model_pins(prior['configuration'], prior['model_assets'], full=False)
    native = {arm: dict(binary=n['binaries']['strata'], native_source_commit=SOURCES[arm],
        native_validation_probe=n['native_validation_probe'], boot=prior['boot'])
        for arm,n in [('reference',older), ('candidate',newer)]}
    require(all(n['binary']['sha256'] == SHAS[a] for a,n in native.items()),
        'exact retained native generations required')
    return native, dict(reference=proof,candidate=newer['observer_admission']), prior, files


def native_context(row, plan):
    require(type(plan['index']) is int and 0 <= plan['index'] < len(CASES)
        and (plan['read'],plan['kernel']) == CASES[plan['index']] and plan['heads'] == 1,
        'unplanned native-generation case')
    arm = plan['kernel']; native = row['native_builds'][arm]
    require(native['native_source_commit'] == SOURCES[arm]
        and native['binary']['sha256'] == SHAS[arm], 'native-generation identity differs')
    return native


def launch_plan(row, plan):
    native = native_context(row,plan)
    return {k:native[k] for k in ('binary','native_validation_probe')}


def configuration(row, plan):
    # The official server executes engine_once.py. Only its frozen plan.json
    # chooses the native ELF; substituting cfg.exe would bypass supervision.
    value = kv.configuration(row,plan)
    require(value['exe'] == str(P/'tools/sycl/engine_once.py'), 'preserve supervised production entry')
    return value


def environment(row, plan):
    env = kv.base.environment(row,plan)
    env.update(STRATA_SYCL_DECODE_KV_COALESCED='0', STRATA_SYCL_KV_COPY_GROUPS='0')
    return env


def invariant(row, plan):
    native_context(row,plan)
    return signature(json.dumps(dict(config=configuration(row,plan),
        environment=environment(row,plan)),sort_keys=True).replace(plan['root'],'<process>'))


def current(row):
    require(kv.device.common.validation.committed(RESULT,(KEY,)) == compact(row)
        and kv.core.pins(row['source_commit']) == row['sources']
        and all(sha(P/p) == h for p,h in row['policies'].items()), 'freeze committed source/plan/state')
    native,proofs,prior,files = basis(row['native_proofs'])
    require(native == row['native_builds'] and proofs == row['native_proofs']
        and row['prior_comparison_sha256'] == signature(compact(prior))
        and all(row[k] == prior[k] for k in ('configuration','environment_template','template','boot',
            'driver','runtime_manifest','model_assets','native_validation_probe')),
        'original generation or common comparison inputs changed')
    contract,python = kv.runner.model.python_manifest()
    require(contract == row['python_contract'] and python['selected'] == row['python_selected']
        and sha(row['template']['path']) == row['template']['sha256']
        and build.supervision.artifact_bytes(ROOT) <= row['whole_bytes']
        and len(row['plans']) == len(CASES), 'frontend/template/output budget differs')
    for i,plan in enumerate(row['plans']):
        root = Path(plan['root'])
        require(plan['index'] == i and plan['command'] == previous.command(plan)
            and plan['adapt_swaps'] == 0 and invariant(row,plan) == row['invariant_signature']
            and json.loads((root/'config.json').read_text()) == configuration(row,plan)
            and json.loads((root/'plan.json').read_text()) == launch_plan(row,plan)
            and all(sha(root/p) == h for p,h in plan['generated'].items())
            and comparison.fingerprint(environment(row,plan)) == plan['environment_sha256'],
            'frozen generation,configuration,workload,launcher or environment changed')
    if row['host_passed']:
        unit = row.get('host_unit_evidence',{})
        require(unit.get('passed') and unit.get('ownership_closed') and unit.get('exit_code') == 0
            and unit.get('tests') == UNIT_TESTS and unit.get('source_commit') == row['source_commit'],
            'closed source-pinned controller controls required')
    return files,python


def request_route(text, plan):
    out = kv.base.prefill.context().request_route(text,plan)
    footers = kv.base.copy_records(text,arm='reference')
    require(len(footers) == 1 and footers[0]['read_tokens'] == plan['read']
        and footers[0]['generated_tokens'] == 24, 'single fresh24-output native extent required')
    if plan['kernel'] == 'candidate':
        out['KV_captures'] = kv.kv_captures(text,False)
        require(any(c['phase'] == 'decode' and c['width'] == 4 for c in out['KV_captures']),
            'new-generation actual scalar decode4 capture required')
    else:
        require('strata verify KV_PLAN:' not in text, 'unexpected historical-generation hook')
        out['KV_policy'] = 'Original14508 has only its pinned scalar planner;no per-request KV_PLAN footer.'
    kv.base.clean_observers(text)
    return out


def decision(processes):
    require(all(p['native_source_commit'] == SOURCES[p['kernel']]
        and p['binary']['sha256'] == SHAS[p['kernel']] for p in processes),
        'timed process bound to wrong native generation')
    out = previous.decision(processes)
    out.update(question='Coherent330405 scalar0 versus original coherent14508,matched PP/TTFT.',
        old_KV_flag_screen_revised=False, decoder_gain_or_preservation_qualified=False)
    return out


def context():
    return SimpleNamespace(ROOT=ROOT,KEY=KEY,ORDER=ORDER,RUNS=RUNS,current=current,
        environment=environment,native_context=native_context,request_route=request_route,
        check_sample=previous.check_sample,decision=decision,compact=compact,save=save,
        validate_model_text=kv.runner.generation.gdn.validate_model_text)


def host():
    comparison.host(context()); row = json.loads((ROOT/'record.json').read_text())
    owner = build.supervision.Owner(ROOT/'unit')
    unit = dict(passed=False,source_commit=row['source_commit'],tests=UNIT_TESTS)
    try:
        entry,out,err = owner.run('unit',[PYTHON,'-B','-m','unittest',*UNIT_TESTS],
            environment(row,row['plans'][0]),P,wall=90,cpu=60,rss_cap=1<<30,
            text_cap=128<<10,file_cap=1<<20,total_cap=4<<20)
        text = out.read_text()+err.read_text()
        require(build.supervision.closed(entry) and entry['exit_code'] == 0
            and re.search(r'Ran 4 tests in [0-9.]+s',text) and text.rstrip().endswith('OK'),
            'generation binding or preservation controls failed')
        unit.update(passed=True,exit_code=0)
    except BaseException as error: unit['error'] = repr(error)
    finally:
        unit['ownership_closed'] = owner.active is None and all(build.supervision.closed(v) for v in owner.commands)
        row.update(host_unit_evidence=unit,host_passed=bool(unit['passed'] and unit['ownership_closed']))
        if not row['host_passed']: row.update(stage='failed',passed=False,complete=False)
        save(row)
    require(row['host_passed'],'host controls or ownership failed')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh committed comparison only')
    native,proofs,prior,_ = basis()
    row = {k:copy.deepcopy(prior[k]) for k in ('configuration','environment_template','template','boot',
        'driver','runtime_manifest','model_assets','native_validation_probe','python_contract','python_selected')}
    revision = shared.gen.git('rev-parse','HEAD').decode().strip()
    row.update(source_commit=revision,sources=kv.core.pins(revision),native_builds=native,native_proofs=proofs,
        policies={p:sha(P/p) for p in ('AGENTS.md','docs/ARTIFACT_RETENTION.md')},
        prior_comparison_sha256=signature(compact(prior)),stage='prepared',plans=[],processes=[],
        request_hashes={},host_passed=False,passed=False,complete=False,active=False,ownership_closed=True,
        whole_bytes=832<<20,budgets=copy.deepcopy(prior['budgets']),plan=dict(
            question='Does the current scalar0 native generation preserve the fastest coherent14508 PP/TTFT?',
            order='Twelve fresh processes:32R,32C,64R,64C,64C,64R,32C,32R,32R,32C,64R,64C. Official short warmup,then three thinking-enabled exact24-output requests. First long GPU use separate from subsequent2 medians. Commit each closed process before the next.',
            constants='Same pinned current engine_once.py/server/bench.sh/input ID decode_kv_prefill_20261114 and models. Same262144logical/20480residentK8V8,8192BF16throughput/largeGU/staged384,1256staticdecode slots,5workers,pcie0/adapt0,spec4,head2/dense1. KVcoalesced0/copygroups0 in both immutable environments. Only native plan ELF/source differs. Safety/quiet retained-reference validation charged.',
            acceptance='Exact original14508 Git source/recipe/census/12ELFs and current1eec/13ELFs;pin controller/helpers/policies/configurations/generated launcher plans/providers/Python/models. Actual fresh32769/65537read,zero reuse,thinking/exact24outputs/valid clocks/clientTTFT/journals/matched bodies. Actual typedQSA,GU/Down,placement/drains/handoff and zero expert copies. New generation must capture scalar decode4;original generation has no hook and its pinned scalar-only source/closed model evidence applies. Normal0/closed ownership/quiet validation/no freshfault/unchangedboot-driver. Reuse immediately preceding final15full hashes after identity checks;rehash after process11.',
            decision='Keep every sample and all three paired C/R prompt-time and TTFT ratios per length/first/warmed scope. The eight-series median preservation screen passes only if every candidate median is no slower;report all-pair directions and geometric means separately. No tolerated regression,extra samples,posthoc criteria or speed-based exclusion. A failure keeps the faster owned prefill reference eligible. Neither outcome changes the failed same-binary KVflag screen or adopts inference.',
            stopping='Stop at the first source/build/owner/provider/count/clock/route/validation/fault/budget/eligibility failure;exclusive attempt before admission,no retry or refreshed pins.',
            retention='No third native generation. Keep compact36samples/configuration/source/decision evidence;retire successful raw output/cache at verified closure. Keep both engines through this imminent comparison;separate three legacy provenance holds remain. No sustainedTG,agent/task,physical262144/RESTORE,ordinary teardown,hardware saturation or whole-goal qualification.'))
    ROOT.mkdir(mode=0o700)
    for i,(read,arm) in enumerate(CASES):
        root=ROOT/str(i);root.mkdir(mode=0o700)
        for n in ('tmp','cache'): (root/n).mkdir()
        with socket.socket() as sock: sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=i,read=read,kernel=arm,heads=1,adapt_swaps=0,root=str(root),port=port)
        kv.core.write(root/'config.json',configuration(row,plan));kv.core.write(root/'plan.json',launch_plan(row,plan))
        plan.update(command=previous.command(plan),generated={n:sha(root/n) for n in ('config.json','plan.json')})
        plan['environment_sha256']=comparison.fingerprint(environment(row,plan));row['plans'].append(plan)
    row['invariant_signature']=invariant(row,row['plans'][0])
    require(all(invariant(row,p) == row['invariant_signature'] for p in row['plans']), 'undeclared arm difference')
    save(row);print('Prepared matched coherent-generation comparison;commit before host/native.',flush=True)


if __name__ == '__main__':
    require(sys.executable == PYTHON and len(sys.argv) == 2
        and sys.argv[1] in ('prepare','host',*map(str,range(len(CASES)))),'qualified Python;prepare|host|0..11')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build.guards.no_model_owners([])
        action=sys.argv[1]
        if action == 'prepare': prepare()
        else:
            with (ROOT/('host-attempt.json' if action == 'host' else action+'/attempt.json')).open('x') as f:
                json.dump(dict(action=action,result_key=KEY),f)
            try: host() if action == 'host' else comparison.run(int(action),context())
            except BaseException as error:
                row=json.loads((ROOT/'record.json').read_text())
                if row['stage']!='failed': row.update(stage='failed',passed=False,complete=False,error=repr(error),failed_action=action);save(row)
                raise
