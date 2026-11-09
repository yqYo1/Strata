#!/usr/bin/env python3
"""Matched official long/agent/thinking comparison of four decode-DPAS cells.

Reuse Main's maintained official-suite evidence, ownership and runtime readers.
Native source and the already closed model proof keep their original identities.
Main-authored; repository LICENSE applies. No external implementation copied.
"""
import ast
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import benchmark_runtime
import engine_once
import official_model_entry as frontend
import q6_dense_model_entry as model
import q6_head_generation_speed as head_speed
import qualify_qsa_k_model as suite

build, guards, owner, validation = model.build, model.guards, model.owner, model.validation
P, B, RESULT = model.P, model.B, model.RESULT
ROOT, KEY, SCOPE = B / 'q6-dense-dpas-official-v2', 'q6_dense_dpas_official_v2', 'cohort'
OLD_ROOT, OLD_KEY = B / 'q6-dense-dpas-official-v1', 'q6_dense_dpas_official_v1'
OLD_EVIDENCE = 'da7511195b49705d5cb93728ef84a6769eaa8719'
THIS = 'tools/sycl/q6_dense_model_speed.py'
ORDER = ('original', 'candidate', 'candidate', 'original', 'original', 'candidate')
TEMPLATE = head_speed.ROOT / 'record.json'
require, sha, write, signature = model.require, model.sha, model.write, model.signature


def pins(revision):
    result = head_speed.pins(revision)
    for name in suite.DOCS:
        if not build.primary.source.git(P, 'ls-tree', '--name-only', revision, '--', name).strip():
            require(not (P / name).exists(), 'uncommitted agent prompt document: ' + name)
            result[name] = None
            continue
        raw = build.primary.source.git(P, 'show', revision + ':' + name)
        require((P / name).read_bytes() == raw, 'uncommitted agent prompt document: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    require(result.get(THIS) == sha(P / THIS), 'commit exact performance observer')
    return result


def native(prior):
    """Admit only this extra observer against the exact closed model inputs."""
    ctx = build.context(); n = json.loads((build.ROOT / 'record.json').read_text())
    require(validation.committed(RESULT, (build.KEY,)) == ctx.compact(n) and n['stage'] == 'built' and
        n['passed'] and n['complete'] and n['ownership_closed'] and not n['active'] and
        n['policy_check']['passed'], 'closed original native build required')
    before = n['admission_project']; a = before['source_sha256']
    require(a == {p: hashlib.sha256(raw).hexdigest() for p, raw in ctx.source.blobs(P, before['revision'], a)},
        'original build admission not recoverable')
    old = prior['native_observer_admission']; admitted = a | old['added_observers']
    require(signature(admitted) == old['observer_source_sha256'] and len(admitted) == old['observer_source_files'] and
        set(old['added_observers']) == {model.device.THIS, model.THIS} and
        old['added_observers'] == {p: hashlib.sha256(raw).hexdigest() for p, raw in
            ctx.source.blobs(P, prior['source_commit'], old['added_observers'])}, 'closed two-observer model proof changed')
    current = ctx.source.pin(P, ctx.PREFIXES); b = current['source_sha256']
    require(set(b) - set(admitted) == {THIS} and not set(admitted) - set(b) and
        all(b[p] == h for p, h in admitted.items()), 'only the new committed performance observer may be added')
    require(n['native_source_commit'] == n['project']['revision'] == n['actual_inputs']['source_commit'] == prior['native_source_commit'] and
        n['binaries'] == prior['native_build']['binaries'] and build.prerequisites() == n['prerequisites'],
        'original native/model/fallback identities changed')
    ctx.build.check_inputs(dict(n, project=current))
    require(all(sha(p) == h for p, h in (n['generated_inputs'] | n['configured_files']).items()),
        'generated/configured recipe changed')
    actual = n['actual_inputs']; require(sha(actual['path']) == actual['sha256'] and actual['uncovered'] == 0,
        'actual compiler census changed')
    census = json.loads(Path(actual['path']).read_text())
    require(all(sha(p) == h for field in ('compiled', 'generated', 'links') for p, h in census[field].items()) and
        all(sha(v['path']) == v['sha256'] and Path(v['path']).stat().st_size == v['bytes'] for v in n['binaries'].values()),
        'exact native inputs/five ELFs changed')
    proof = dict(native_source_commit=n['native_source_commit'], model_controller_source_commit=prior['source_commit'],
        original_model_source_files=len(admitted), current_source_files=len(b), current_source_sha256=signature(b),
        added_observer={THIS: b[THIS]}, exact_native_5ELFs_census_recipe=True,
        native_recompiled=False, earlier_speed_result_inherited=False)
    return n, proof


def historical_control(sources):
    """New analysis of the mandatory V1 control; never rewrite its failure."""
    old=json.loads((OLD_ROOT/'record.json').read_text())
    archived=json.loads(build.primary.source.git(P,'show',
        OLD_EVIDENCE+':'+str(RESULT.relative_to(P))))[OLD_KEY][SCOPE]
    require(archived == validation.committed(RESULT,(OLD_KEY,SCOPE)) == compact(old) and
        old['stage']=='failed' and not old['passed'] and not old['complete'] and
        not old['active'] and old['ownership_closed'] and len(old['processes'])==1,
        'preserved failed control/source record required')
    previous=old['processes'][0];late=old['post_closure_diagnostic']['late_postconditions']
    require(previous['index']==0 and previous['arm']=='original' and not previous['complete'] and
        not previous['passed'] and previous['error']==
        "RuntimeError('unexpected long-generation workload or thinking/output budget')" and
        previous['ownership_closed'] and previous['fresh_faults']==[] and previous['boot_unchanged'] and
        old['host_passed'] and late['passed'] and late['full15_asset_hashes_match'] and
        late['source_native_five_ELFs_census_recipe_runtime_generated_profile_match'] and
        late['V1_outcome_unchanged']=='failed' and
        late['native_admission_sha256']==signature(old['native_admission']) and
        late['asset_manifest_sha256']==signature(old['model_assets']),
        'late audited control postconditions incomplete')
    require(set(sources)==set(old['sources']) and all(
        sources[p]==h for p,h in old['sources'].items() if p!=THIS),
        'historical reuse allows only this postprocessing/controller change')
    original=build.primary.source.git(P,'show',old['source_commit']+':'+THIS)
    require(hashlib.sha256(original).hexdigest()==old['sources'][THIS],
        'original performance observer not recoverable')
    definitions=lambda text:{v.name:ast.dump(v,include_attributes=False) for v in ast.parse(text).body
        if isinstance(v,(ast.FunctionDef,ast.AsyncFunctionDef))}
    before,after=definitions(original),definitions((P/THIS).read_bytes())
    unchanged=('pins','native','environment','audit','native_live','decision')
    require(all(before[p]==after[p] for p in unchanged) and old['plan']['order']==list(ORDER),
        'timed launch/environment/route/decision helper changed')
    root=OLD_ROOT/'0';plan=old['plans'][0]
    prior=json.loads((model.ROOT/'record.json').read_text())
    runtime=dict(boot=prior['boot'],runtime_files=prior['runtime_files'])
    require(old['native_source_commit']==prior['native_source_commit'] and
        old['binary']==prior['native_build']['binaries']['strata'] and old['boot']==prior['boot'] and
        old['model_assets']==prior['models'] and sha(prior['runtime_manifest']['path'])==old['runtime_sha256'] and
        suite.d.fingerprint(environment(root,runtime))==plan['environment_sha256'] and
        all(sha(root/p)==h for p,h in plan['generated'].items()),
        'historical native/runtime/environment/generated configuration differs')
    profile=old['profile']
    require(suite.d.models.file_identity(profile['path'])==profile['identity'] and
        sha(profile['path'])==profile['sha256'],'historical expert profile differs')
    execution=json.loads((root/'execution.json').read_text());entries=execution['commands']
    require(not execution['active'] and len(entries)==1 and entries[0]['label']=='bench' and
        owner.closed(entries[0]) and entries[0]['exit_code']==0, 'historical bench owner/exit differs')
    config=json.loads((root/'config.json').read_text())
    end=suite.m.component.gen.native_end(root,
        dict(binary=plan['binary'],native_validation_probe=old['native_validation_probe']),
        dict(config=config),old['boot'],entries[0]['pid'])
    log=(root/'engine.stderr').read_text()
    require(validation.preload_evidence(log,end['native']['pid'])==late['validation'] and
        (root/'validation.log').stat().st_size==0 and
        not any(t in log for t in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','teardown uncertain')),
        'historical native quietness/pre-main validation differs')
    result=evidence(root,{'processes':[]})
    require(result['native_dense_route']==late['actual_route'] and
        'strata MoE window configuration: requested_mode=10 effective_mode=10' in log,
        'historical actual route differs')
    for sample in result['samples']:
        if sample['phase'].startswith(('prompt/32K','long-generation/')):
            require('strata prefill: layer-major candidate, tokens='+str(sample['read_tokens']-1)+
                ' chunk=8192 residual_dtype=bf16' in log,'historical long-prefill route missing')
    suite.d.models.model_pins(config,old['model_assets'])
    files=['record.json','0/config.json','0/plan.json','0/engine-start.json','0/engine-exit.json',
        '0/engine.stderr','0/validation.log','0/execution.json']
    files += [str(p.relative_to(OLD_ROOT)) for p in (root/'bench').glob('*/*')
        if p.name in ('results.json','requests.jsonl')]
    proof=dict(record_evidence_commit=OLD_EVIDENCE,original_controller_source_commit=old['source_commit'],
        original_outcome='failed',original_error=previous['error'],root=str(root),
        raw_sha256={p:sha(OLD_ROOT/p) for p in files},unchanged_helpers=list(unchanged),
        only_observer_changed=True,timed_native_client_workload_environment_unchanged=True,
        original_full_asset_audit_utc=late['checked_utc'],original_host_passed=True,
        original_runtime_live_proof=previous['native_identity'],
        limits='Historical control analysed under the new v5/no-Verifier-prompt reader;not a new process or retroactive V1 success. Same mandatory control,no speed-based selection or replacement.')
    process=dict(index=0,arm='original',execution='historical_reanalysis',complete=True,passed=True,
        postconditions_complete=True,host_probe_accepted=True,native_exit_code=0,ownership_closed=True,
        fresh_faults=[],boot_unchanged=True,evidence=result,historical_origin=proof)
    return old,proof,process


def qualification(require_host=True):
    prior = json.loads((model.ROOT / 'record.json').read_text()); ctx = model.context()
    require(validation.committed(RESULT, (model.KEY,)) == ctx.compact(prior) and prior['stage'] == 'closed' and
        prior['passed'] and prior['complete'] and prior['ownership_closed'] and not prior['active'] and
        prior['decision']['bounded_actual_model_routes_qualified'] and len(prior['results']) == 4 and
        all(v['passed'] and v['native_exit_code'] == 0 and v['fresh_faults'] == [] for v in prior['results']),
        'all four exact current model choices must close first')
    n, proof = native(prior)
    n['speed_observer_admission'] = proof
    # Supply the separately checked native input set to the unchanged model
    # reader. Its original controller hashes/outcomes are never refreshed.
    ctx.native = lambda: n
    ctx.verify(prior)
    reference = json.loads(TEMPLATE.read_text()); old = head_speed.context()
    require(validation.committed(RESULT, (head_speed.KEY,)) == old.compact(reference) and
        reference['stage'] == 'closed' and reference['passed'] and reference['complete'] and
        reference['ownership_closed'] and not reference['active'] and reference['model_assets'] == prior['models'],
        'closed tuned head-generation configuration required')
    contract, manifest = frontend.python_manifest()
    require(reference['python_contract'] == contract and reference['python_selected'] == manifest['selected'],
        'same qualified frontend Python required')
    if (ROOT / 'record.json').exists():
        row = json.loads((ROOT / 'record.json').read_text())
        require(validation.committed(RESULT, (KEY, SCOPE)) == compact(row) and row['native_admission'] == proof and
            sha(TEMPLATE) == row['template_record']['sha256'] and
            row['model_evidence_sha256'] == signature(ctx.compact(prior)), 'commit exact stable performance stage')
        _,historical,process=historical_control(row['sources'])
        require(row['historical_control']==historical and row['processes'][0]==process,
            'frozen historical-control analysis changed')
        if require_host:
            require(row['host_passed'] and row['stage'] in ('ready', 'running'), 'fresh frontend gates required')
    runtime = dict(boot=prior['boot'], runtime_files=prior['runtime_files'])
    data = json.loads(RESULT.read_text())
    return data, n, runtime, manifest, prior


def environment(root, runtime):
    reference = json.loads(TEMPLATE.read_text())
    index = int(root.name) if root.name.isdecimal() else 0
    old_plan = reference['plans'][index]
    env = head_speed.context(index).environment(reference, old_plan)
    env = {k: v.replace(old_plan['root'], str(root)) for k, v in env.items()}
    env.update(STRATA_SYCL_DECODE_Q6_DPAS_HEAD='2', STRATA_PREFILL_HEAD_HOST_BACKING='0',
        STRATA_SYCL_DECODE_Q6_DPAS_DENSE=str(int(ORDER[index] == 'candidate')))
    require(env['STRATA_MEASUREMENT_ROOT'] == str(root) and env['STRATA_SYCL_CACHE_SCRATCH_LEASE'] == '1' and
        env['STRATA_PREFILL_RELEASE_HEADS'] == env['STRATA_SYCL_Q6_ROUTE_AUDIT'] == '1' and
        env['STRATA_SYCL_DECODE_Q6_COLUMNS'] == '1' and env['STRATA_SYCL_DECODE_Q6_CENTERING'] == '0' and
        env['STRATA_SYCL_DECODE_Q6_NATIVE_HALF'] == '0' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and
        env['EnableDirectSubmission'] == env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0' and not env.get('LD_PRELOAD'),
        'preserve tuned source/placement and native-only validation preload')
    return env


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('sources', 'model_assets')}
    out['input_signatures'] = {k: signature(row[k]) for k in ('sources', 'model_assets')}
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data.setdefault(KEY, {})[SCOPE] = compact(row); write(RESULT, data)


def audit(text, dense, long_requests=(17, 18)):
    base, _ = model.previous.route_tables(text, 2)
    require({int(k[0]) for k in base} == set(range(1,19)), 'all18actual native request identities required')
    selected = {k: {} for k in base}; legacy = {k: {} for k in base}
    for line in text.splitlines():
        if not line.startswith(('strata Q6_DPAS_DENSE ', 'strata Q6_SHAPE ')): continue
        f = model.fields(line); key = tuple(f.get(k) for k in model.IDENTITY)
        require(key in base, 'orphaned production shape')
        if line.startswith('strata Q6_DPAS_DENSE '):
            require(dense == 1 and set(f) == set(model.IDENTITY) | {'ni','no','nc','layout',
                'extra_weight_bytes','canonical_table_excludes_dpas','calls','complete'} and
                f['nc'] == '4' and f['layout'] == 'canonical' and f['extra_weight_bytes'] == '0' and
                f['canonical_table_excludes_dpas'] == f['complete'] == '1' and
                all(f[k].isdecimal() for k in ('ni','no','calls')), 'actual dense schema/layout differs')
            shape = (int(f['ni']), int(f['no'])); calls = int(f['calls'])
            require(shape in model.SHAPES and shape not in selected[key] and
                (calls > 0) == (key[2:4] == ('decode','4')), 'dense escaped its four decode-C4 cells')
            selected[key][shape] = calls
        else:
            names = ('ni','no','nc','reference_calls','columns_calls','centered_calls','native_half_calls')
            require(set(f) == set(model.IDENTITY) | set(names) and all(f[k].isdecimal() for k in names),
                'canonical schema differs')
            shape = tuple(int(f[k]) for k in names[:3]); calls = tuple(int(f[k]) for k in names[3:])
            require(shape not in legacy[key] and shape[0] > 0 and shape[1] > 0 and 1 <= shape[2] <= 4 and
                sum(calls) > 0 and calls[2:] == (0,0) and
                (key[2] == 'decode' or calls[1] == 0), 'canonical phase/count policy differs')
            legacy[key][shape] = calls
    normalized, epochs, replays = {}, {}, {}
    expected = dict(zip(model.SHAPES, (40,14,22,5)))
    for key in sorted(base, key=lambda k: (int(k[0]), k[2], int(k[3]), int(k[4]))):
        require(set(selected[key]) == (set(model.SHAPES) if dense else set()) and
            base[key].get('shapes','').isdecimal() and int(base[key]['shapes']) == len(legacy[key]),
            'complete actual four-shape joins required')
        table = dict(legacy[key])
        if key[2:4] == ('decode','4'):
            for shape, count in expected.items():
                cell = (*shape,4)
                if dense: require(selected[key][shape] == count and cell not in table, 'dense count/double count differs')
                else: require(table.pop(cell, None) == (0,count,0,0), 'control columns call differs')
            require(table.get((2560,512,4)) == (21,0,0,0), 'small C4 fallback changed')
            capture = key[1:]; completed = int(base[key]['replay_completed'])
            previous = epochs.get(capture,0); require(completed >= previous, 'cumulative replay count decreased')
            replays[int(key[0])] = completed - previous; epochs[capture] = completed
        label = '/'.join(key[2:4]); value = {'x'.join(map(str,k)):list(v) for k,v in sorted(table.items())}
        digest = signature(value)
        require(label not in normalized or normalized[label] == digest, 'static canonical table changed')
        normalized[label] = digest
    require(all(replays.get(k,0) > 0 for k in long_requests), 'both actual long-generation requests must replay decode-C4')
    transfers = model.re.findall(r'^strata prefill heads: target detached bytes=(\d+) source=(\S+) D2H=(\d+) H2D-on-restore=(\d+)$', text, model.re.M)
    require(len(transfers) >= 2 and all(v == ('521472000','owned-RAM','521472000','521472000') for v in transfers),
        'actual common head handoffs differ')
    drafts = model.re.findall(r'^strata prefill heads: draft detached bytes=(\d+) retained-map-bytes=(\d+)$', text, model.re.M)
    require(drafts == [('0','425196')] * len(transfers),
        'single packed MTP ownership differs')
    lines = text.splitlines()
    require([v for v in lines if v.startswith('strata native head packed:')] ==
        ['strata native head packed: ready=1 columns=1..4 weight_bytes=521472000 canonical_device_weight_bytes=0 extra_device_weight_bytes=0 permanent_host_weight_bytes=0 default_adopted=0'] and
        [v for v in lines if v.startswith('strata mtp: shared packed head ')] ==
        ['strata mtp: shared packed head tokens=106299 subset_weight_bytes=0 subset_logit_bytes=425196 saved_device_bytes=222802704'] and
        not any(v.startswith('strata native head backing:') for v in lines), 'common packed/backing owner differs')
    return dict(dense_enabled=dense, selected_C4_calls={f'{k[0]}x{k[1]}':v for k,v in expected.items()},
        generic_small_C4_calls=21, canonical_phase_width_sha256=normalized,
        long_request_C4_replay_deltas={str(k):replays[k] for k in long_requests}, head_transitions=len(transfers),
        scope='Static actual captures and per-request replay deltas;logical copy requests,not physical traffic or utilization.')


def common_routes(reference, current):
    common = set(reference) & set(current)
    # Dedicated long prefill does not use the Verifier prompt capture. Its
    # mode10/BF16 route is checked separately by the unchanged suite runner.
    require('decode/4' in common and all(reference[k] == current[k] for k in common),
        'common canonical phase/width routing changed')
    return dict(common=sorted(common), reference_only=sorted(set(reference)-set(current)),
        candidate_only=sorted(set(current)-set(reference)), passed=True)


def generation_contract(generation, rows):
    # Private v5 contract, following the existing Main-authored pool_spin
    # reader. Preserve generation_engine's historical v4 contract unchanged.
    workloads = ('regular-code', 'varied-prose')
    require(generation.get('workload_version') == 5 and generation.get('thinking') is True and
        generation.get('reasoning_budget_tokens') == 0 and generation.get('max_tokens') == 16384,
        'unexpected v5 task or thinking/output budget')
    require(len(rows) == len(workloads) and
        [(r.get('workload'), r.get('run'), r.get('order')) for r in rows] ==
        [(name, 0, i) for i, name in enumerate(workloads)] and
        all(r.get('controller_eligible') is True for r in rows), 'incomplete/ineligible v5 generation')
    require(generation.get('diagnostic_subset') is False and
        generation.get('selected_workloads') == list(workloads), 'v5 workload selection differs')


def suite_context():
    spec = importlib.util.spec_from_file_location('dense_official_context', suite.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    gen = SimpleNamespace(**(vars(ctx.m.component.gen) |
        {'require_complete_generation':generation_contract}))
    component = SimpleNamespace(**(vars(ctx.m.component) | {'gen':gen}))
    ctx.m = SimpleNamespace(**(vars(ctx.m) | {'component':component}))
    return ctx


def audit_controls():
    def sample(dense, verifier_prompt=False):
        out = ['strata native head packed: ready=1 columns=1..4 weight_bytes=521472000 canonical_device_weight_bytes=0 extra_device_weight_bytes=0 permanent_host_weight_bytes=0 default_adopted=0',
            'strata mtp: shared packed head tokens=106299 subset_weight_bytes=0 subset_logit_bytes=425196 saved_device_bytes=222802704']
        for request in range(1,19):
            capture, completed = request * 4, 5 if request % 2 else 8
            out += ['strata prefill heads: target detached bytes=521472000 source=owned-RAM D2H=521472000 H2D-on-restore=521472000',
                'strata prefill heads: draft detached bytes=0 retained-map-bytes=425196']
            phases = [('decode',1),('decode',4)]
            if verifier_prompt: phases.insert(0,('prompt',4))
            for phase, tokens in phases:
                identity = f'request={request} device=0 phase={phase} tokens={tokens} capture_id={capture}'
                active = phase == 'decode' and tokens == 4; cells = [(2560,512,(21,0,0,0))]
                if not (dense and active):
                    cells += [(ni,no,(0,count,0,0) if phase == 'decode' else (count,0,0,0))
                        for (ni,no),count in zip(model.SHAPES,(40,14,22,5))]
                out += ['strata Q6_ROUTE '+identity+f' scope=live_capture_cumulative native_half_enabled=0 complete=1 shapes={len(cells)} replay_started={completed} replay_completed={completed}',
                    'strata Q6_PACKED_HEAD '+identity+' layout=sole-packed extra_weight_bytes=0 calls=1 complete=1']
                out += ['strata Q6_SHAPE '+identity+f' ni={ni} no={no} nc={tokens} reference_calls={c[0]} columns_calls={c[1]} centered_calls={c[2]} native_half_calls={c[3]}' for ni,no,c in cells]
                if dense:
                    out += ['strata Q6_DPAS_DENSE '+identity+f' ni={ni} no={no} nc=4 layout=canonical extra_weight_bytes=0 canonical_table_excludes_dpas=1 calls={count if active else 0} complete=1'
                        for (ni,no),count in zip(model.SHAPES,(40,14,22,5))]
        return '\n'.join(out)
    for verifier_prompt in (False,True):
        a,b = audit(sample(0,verifier_prompt),0),audit(sample(1,verifier_prompt),1)
        common_routes(a['canonical_phase_width_sha256'],b['canonical_phase_width_sha256'])
        require(a['canonical_phase_width_sha256'] == b['canonical_phase_width_sha256'] and
            b['long_request_C4_replay_deltas'] == {'17':5,'18':8}, 'full-suite source reader controls differ')
    text=sample(1);marker=next(v for v in text.splitlines() if v.startswith('strata Q6_DPAS_DENSE '))
    bad=[text.replace(marker,'',1),text+'\n'+marker,text.replace(marker,marker+' calls=0',1),
        text.replace(marker,marker.replace('calls=0','calls=1'),1),
        text.replace(marker,marker.replace('ni=6144','ni=512'),1),
        text.replace('extra_device_weight_bytes=0','extra_device_weight_bytes=1',1),
        text.replace('permanent_host_weight_bytes=0','permanent_host_weight_bytes=521472000',1),
        text.replace('nc=4 reference_calls=21','nc=4 reference_calls=20'),
        text.replace('H2D-on-restore=521472000','H2D-on-restore=0',1)]
    for value in bad:
        try:audit(value,1)
        except RuntimeError:continue
        raise RuntimeError('full-suite observer accepted corrupted routing/ownership')
    try:audit(text,0)
    except RuntimeError:pass
    else:raise RuntimeError('disabled selector accepted dense markers')
    for a,b in (({'decode/1':'a'},{'decode/1':'a'}),
                ({'decode/4':'a'},{'decode/4':'b'}),
                ({'decode/4':'a','prompt/4':'x'},{'decode/4':'a','prompt/4':'y'})):
        try: common_routes(a,b)
        except RuntimeError: continue
        raise RuntimeError('canonical isolation reader accepted a changed common route')
    root=OLD_ROOT/'0';path,=(root/'bench').glob('*/results.json')
    generation=json.loads(path.read_text())['results']['long_generation']
    rows=suite.m.component.gen.collect_samples(generation)
    generation_contract(generation,rows)
    mutations=[]
    for field,value in (('workload_version',4),('thinking',False),('reasoning_budget_tokens',1),
                        ('max_tokens',1024),('diagnostic_subset',True),('selected_workloads',['regular-code'])):
        changed=copy.deepcopy(generation);changed[field]=value;mutations.append((changed,rows))
    for field,value in (('run',1),('order',1),('controller_eligible',False)):
        changed=copy.deepcopy(rows);changed[0][field]=value;mutations.append((generation,changed))
    mutations.extend(((generation,rows[::-1]),(generation,rows[:-1])))
    for changed,samples in mutations:
        try: generation_contract(changed,samples)
        except RuntimeError: continue
        raise RuntimeError('current-v5 reader accepted changed version/order/eligibility/selection')
    current=evidence(root,{'processes':[]})
    require(len(current['samples'])==18 and
        set(current['native_dense_route']['canonical_phase_width_sha256'])=={'decode/1','decode/4'},
        'actual completed no-Verifier-prompt baseline does not pass the prospective reader')
    return dict(passed=True,positive_route_arms=4,corrupt_route_cases=len(bad)+4,
        current_v5_positive=1,current_v5_corrupt_cases=len(mutations),
        actual18request_baseline_reader_passed=True,GPU_executed=False)


def evidence(root, row):
    result = suite_context().evidence(root, row)
    dense = int(ORDER[int(root.name)] == 'candidate')
    result['native_dense_route'] = audit((root / 'engine.stderr').read_text(), dense)
    result['journal'] = {k: result['journal'][k] for k in ('valid','complete','pending')}
    if row['processes']:
        a = row['processes'][0]['evidence']['native_dense_route']['canonical_phase_width_sha256']
        b = result['native_dense_route']['canonical_phase_width_sha256']
        result['native_dense_route']['isolation'] = common_routes(a,b)
    return result


def native_live(root, plan, row, env, runtime, manifest, entry):
    claim = json.loads((root / 'engine-start.json').read_text())['native']; actual = owner.identity(claim['pid'])
    require(actual and actual['start_ticks'] == claim['start_ticks'] and actual['sid'] == entry['pid'] and
        str(Path('/proc',str(claim['pid']),'exe').resolve()) == str(Path(row['binary']['path']).resolve()),
        'actual owned native identity changed')
    expected = engine_once.native_environment(dict(native_validation_probe=row['native_validation_probe']), env)
    current = dict(v.split(b'=',1) for v in Path('/proc',str(claim['pid']),'environ').read_bytes().split(b'\0') if b'=' in v)
    require(all(current.get(k.encode()) == v.encode() for k,v in expected.items()) and
        set(current)-{k.encode() for k in expected} <= {b'PWD',b'OLDPWD',b'SHLVL',b'_'}, 'native pre-exec environment differs')
    proof = model.previous.model.runtime_evidence(dict(probe=row['native_validation_probe'], environment=expected,
        runtime_files=runtime['runtime_files'], native_build=dict(binaries=dict(strata=row['binary']))),
        claim['pid'], (root / 'engine.stderr').read_text())
    allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
    benchmark_runtime.check_running(entry['pid'], P / 'tools/strata_bench.py', manifest['selected'], env,
        allowed, plan['command'][2:])
    return dict(claim, runtime_evidence=proof)


def decision(processes):
    result = suite.decision(processes)
    for value in result['decode'].values(): value['gain_in_all_three_pairs'] = all(v > 1 for v in value['pair_rate_ratios'])
    phases = {}
    for phase in ('prompt/32K','long-generation/regular-code','long-generation/varied-prose'):
        pairs = {metric:[] for metric in ('prompt_tps','prompt_ms','ttft_s')}
        for offset in (0,2,4):
            arms = {p['arm']:p for p in processes[offset:offset+2]}
            samples = {arm:next(s for s in p['evidence']['samples'] if s['phase'] == phase) for arm,p in arms.items()}
            for metric in pairs: pairs[metric].append(samples['candidate'][metric]/samples['original'][metric])
        phases[phase] = dict(candidate_over_original=pairs,
            geometric_means={k:math.exp(statistics.mean(math.log(v) for v in values)) for k,values in pairs.items()},
            strict32K_input=phase.startswith('long-generation/'))
    result.update(prefill_TTFT=phases, current_physical_limiting_resource='unknown', inference_adopted=False,
        next='Judge each workload and prefill/TTFT/decode independently. Keep all individual samples and variance;no speed-based exclusions. Then assess backing separately and collect role-compatible resource evidence. Task/physical262144/RESTORE and ordinary teardown remain open.')
    return result


def context():
    ctx = suite_context()
    prior = json.loads((model.ROOT / 'record.json').read_text())
    runtime = SimpleNamespace(**(vars(ctx.m.runtime) | {'RESULT':Path(prior['runtime_manifest']['path'])}))
    ctx.m = SimpleNamespace(**(vars(ctx.m) | {'KEY':KEY,'runtime':runtime}))
    ctx.P, ctx.B, ctx.RESULT = P, B, RESULT
    ctx.ROOT, ctx.KEY, ctx.ORDER = ROOT, SCOPE, ORDER
    ctx.qualification, ctx.pins, ctx.compact = qualification, pins, compact
    ctx.previous = SimpleNamespace(environment=environment)
    ctx.native_live, ctx.evidence, ctx.decision = native_live, evidence, decision
    return ctx


def prepare():
    require(not ROOT.exists() and KEY not in build.primary.public(), 'fresh official cohort only;no overwrite/retry')
    _, n, runtime, manifest, prior = qualification(False)
    reference = json.loads(TEMPLATE.read_text()); revision = build.primary.source.git(P,'rev-parse','HEAD').decode().strip()
    sources=pins(revision);old,historical,control=historical_control(sources)
    config = copy.deepcopy(reference['configuration']); config['log'] = str(ROOT / 'engine.stderr')
    require(config['exe'] == str(P / 'tools/sycl/engine_once.py'), 'qualified engine wrapper required')
    plans=[]; children=[]; invariants=[]
    for index, arm in enumerate(ORDER):
        root=OLD_ROOT/'0' if index==0 else ROOT/str(index)
        child=copy.deepcopy(config);child['log']=str(root/'engine.stderr');children.append(child)
        if index==0:
            require(json.loads((root/'config.json').read_text())==child,
                'historical actual configuration differs')
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env=environment(root,runtime); masked=dict(env,STRATA_SYCL_DECODE_Q6_DPAS_DENSE='<arm>')
        invariants.append(signature(json.loads(json.dumps(dict(config=child,environment=masked)).replace(str(root),'<process>'))))
        plans.append(copy.deepcopy(old['plans'][0]) if index==0 else dict(index=index,arm=arm,root=str(root),binary=n['binaries']['strata'],
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--long','--long-generation','--think','--runs','1','--workload-id','q6_dense_official_v1',
                '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')],
            environment_sha256=suite.d.fingerprint(env)))
    require(len(set(invariants))==1 and invariants[0]==old['invariant_sha256'],
        'historical and fresh arms differ beyond dense selection/private process roots')
    profile=config['args'][config['args'].index('--expert-profile')+1]
    proof=n['speed_observer_admission']
    row=dict(stage='prepared',source_commit=revision,sources=sources,native_source_commit=n['native_source_commit'],
        binary=n['binaries']['strata'],native_admission=proof,native_validation_probe=prior['probe'],
        configuration=config,model_assets=prior['models'],boot=prior['boot'],driver=prior['driver'],
        template_record=dict(path=str(TEMPLATE),sha256=sha(TEMPLATE)),model_evidence_sha256=signature(model.context().compact(prior)),
        runtime_sha256=sha(prior['runtime_manifest']['path']),python_contract=reference['python_contract'],
        python_selected=manifest['selected'],profile=dict(path=profile,identity=suite.d.models.file_identity(profile),sha256=sha(profile)),
        invariant_sha256=invariants[0],plans=plans,processes=[control],historical_control=historical,
        fixed_request_inputs=None,host_passed=False,
        budgets=dict(wall=5400,cpu=43200,rss_cap=104<<30,text_cap=64<<20,file_cap=128<<20,total_cap=512<<20),
        plan=dict(question='Do the four individually winning C4decode-DPAS cells improve model decode while preserving prefill/TTFT?',
            order=list(ORDER),constants='Same a347/221e engine,head2,backing0,128KresidentINT8,1256cache,8192chunk,384bank,5workers,spec4,adapt96/every1,lease1,20ms spin,IQ3pair1 and existing BF16/GDN16/MTP16 paths. Only immutable decode-DPAS0/1 changes.',
            samples='One mandatory historical V1control plus five fresh processes,three opposite-order adjacent pairs. Same unchanged bench.sh --long --long-generation --think --runs1 and workload-id q6_dense_official_v1:18requests including six-turn agent and both >=32Kfresh/>=1Kthinking generation cases. Non-agent bodies/counts must match;generated text and optional MTP widths need not. Preserve the original failed V1source/outcome and explicit late-postcondition provenance;no speed-based replacement.',
            acceptance='First current-v5/no-Verifier-prompt reader controls,actual18request baseline reanalysis,and two fresh CPU/mock frontend probes. Then complete journals/streams and answer bodies;actual PID/providers/pre-main validation;normal native0,closed owners,quiet validation,no fresh fault;original exact source/5ELFs/census/recipe and assets. Both long requests must actually replay selected C4cells40/14/22/5;small generic21 and common observed canonical phase/width tables remain. Independent mode10/BF16long-prefill checks stay unchanged. Full15asset hashes from the immediately preceding committed late audit plus unchanged identities at preparation;full hashes again at final closure.',
            decision='Keep all3pair TG ratios/geomean and individual samples. Gain in all3pairs is credible gain with no arbitrary minimum floor;GM>=1 is only a preservation screen. Confirm strong>=.60/weak<=.40 from each arm sample. Judge PP/TTFT and decode independently;no noise waiver or offsetting a regression by another phase.',
            stopping='Stop first ownership/runtime/fault/budget/incomplete-output/source/admission failure. No retries,replacements or speed-based early stops;classification/speed outcomes never discard samples. Commit every closed stage. Freeze the whole FUSED file and benchmark inputs while a process is active.',
            limits='First control was executed under original8fa/V1plan before this reader-only repair;its known observations are retained without changing the original performance criteria or selecting by speed. Native/client/timed launch/environment and helper equivalence are checked. Nominal upstream32K may read32089 and is supplemental;added >=32768fresh generation inputs qualify strict long comparisons. Provider maps are checked at suite start,not throughout the complete executable address space/history. Validation/audits are charged;observer cost,semantic task quality,physical262144/RESTORE,ordinary provider teardown and hardware saturation remain unqualified.',
            retention='Keep individual metric/body hashes,source/config/native identities and decisions. Retire successful full journals/logs/private caches after closure and source recovery;keep a347/5fixtures and1db for the named imminent comparisons. Other named phase/feature consumers and legacy holds remain separate.'),
        complete=False,passed=False,active=False,ownership_closed=True)
    suite.d.models.model_pins(config,row['model_assets'])
    ROOT.mkdir(mode=0o700)
    for plan,child in zip(plans,children):
        if plan['index']==0:
            continue
        root=Path(plan['root']);root.mkdir(mode=0o700)
        for name in ('tmp','cache'):(root/name).mkdir(mode=0o700)
        write(root/'config.json',child)
        write(root/'plan.json',dict(binary=n['binaries']['strata'],native_validation_probe=prior['probe']))
        plan['generated']={p:sha(root/p) for p in ('config.json','plan.json')}
    save(row);print('Prepared one explicit historical control and five fresh same-engine official processes;commit before frontend probes.',flush=True)


def host():
    _, _, runtime, manifest, prior = qualification(False)
    row=json.loads((ROOT/'record.json').read_text())
    require(row['stage']=='prepared' and not row['host_passed'] and not (ROOT/'host-owned').exists(), 'fresh host gates only')
    scope=None
    try:
        require(pins(row['source_commit'])==row['sources'],'frontend/controller source changed')
        scope=owner.Owner(ROOT/'host-owned');host_root=ROOT/'host-probe';host_root.mkdir(mode=0o700)
        for name in ('tmp','cache'):(host_root/name).mkdir(mode=0o700)
        env=environment(host_root,runtime)
        row['host_reader_controls']=audit_controls()
        allowed={v['realpath'] for v in manifest['inventory']['files'].values()};reports=[]
        for role,path in (('benchmark','tools/strata_bench.py'),('server','serve/server.py')):
            command=[manifest['selected']['path'],str(benchmark_runtime.SOURCE),'probe',str(P/path),str(Path(row['configuration']['tokenizer'])/'chat_template.jinja')]
            e,out,err=scope.run(role,command,env,P,wall=45,cpu=30,rss_cap=512<<20,text_cap=512<<10,file_cap=2<<20,total_cap=8<<20)
            require(owner.closed(e) and e['exit_code']==0 and err.stat().st_size==0,'fresh frontend probe failed')
            observed=json.loads(out.read_text());require(str(Path(observed['executable']).resolve())==manifest['selected']['file']['realpath'],'frontend Python changed')
            for name in [*observed['modules'],*observed['code_maps']]:
                p=Path(name).resolve();require(str(p.relative_to(P)) in row['sources'] if p.is_relative_to(P) else str(p) in allowed,'unqualified frontend import/map')
            reports.append(dict(role=role,passed=True,modules=len(observed['modules']),code_maps=len(observed['code_maps'])))
        row.update(stage='ready',host_passed=True,host_results=reports)
    except BaseException as error:row.update(stage='failed',error=repr(error))
    finally:
        row['ownership_closed']=bool(scope and scope.active is None and all(owner.closed(v) for v in scope.commands))
        row['active']=bool(scope and scope.active);save(row)
    require(row['host_passed'] and row['ownership_closed'],row.get('error','frontend gates failed'))
    print('Both fresh frontend probes pass;commit before actual official comparison.',flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','host','1','2','3','4','5'),
        'q6_dense_model_speed.py prepare|host|1..5 (index0 is the explicit historical control)')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='host':host()
        else:context().run(int(sys.argv[1]))
