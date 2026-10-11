"""Main-run CPU fixtures only. Synthetic receipts are never model PASS evidence."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
A = B/'xe-native-dequant-prefill-quiet-timing-controller-source-v3'
sys.path.insert(0, str(A))
import run_root as controller
import protocol
import runtime

m = controller.m
CANDIDATE_SHA = 'c1e636e7544fa5ef6a14f6ec294b6454171b19c7b91e241dc22c7fc02a88a5f8'
OLD = B/'xe-native-dequant-prefill-model-controller-main-cpu-v6.py'
OLD_SHA = 'ee7aa0367b619762e9acd5f3a6ffc93e20abed718f3324488958a57861a2672c'
PINS = {'off':'01992f0ba3b15a135636e97ff4dd37fc03f1137486df948ff7631edc96a1c9d1',
        'on':'8f7ad6c80553ac091388f5146ae6e93571115b6f6831f5f01e3458f3d6cd9a54'}
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
root = Path(sys.argv[1]).resolve(strict=True)
assert root.parent.name == 'xe-native-dequant-prefill-quiet-timing-controller-cpu-root-v3'
assert root.name == 'fixtures' and m.sha(A/'run_root.py') == CANDIDATE_SHA
assert m.sha(OLD) == OLD_SHA
results = []

# Reuse the exact old Main protocol/trace fixture program, not a rewritten predicate.
old_tree = ast.parse(OLD.read_text())
old_harness = ast.literal_eval(next(x.value for x in old_tree.body if isinstance(x, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'HARNESS' for t in x.targets)))
protocol_prefix = old_harness.split('import ast, copy, hashlib, os, re, subprocess, sys, time\n', 1)[0]
namespace = {}
exec(compile(protocol_prefix, str(OLD)+'#unchanged-protocol-prefix', 'exec'), namespace)
assert len(namespace['positive']) == 9 and len(namespace['negative']) == 35
results.append(dict(case='unchanged_protocol_trace_fixtures', positive=9, negative=35))

diagnostics = {}
for arm, digest in PINS.items():
    path = B/'xe-native-dequant-prefill-model-root-v6'/f'n32768-g16-{arm}'/'record.json'
    assert m.sha(path) == digest and path.stat().st_size <= 8 << 20
    data = json.loads(path.read_text())
    assert data['passed'] and data['complete'] and not data['active']
    assert data['input_tokens'] == 32768 and data['validation']['generated_tokens'] == 16
    assert protocol.validate(data['request_protocol'], n=32768) == data['validation']
    diagnostics[arm] = data
assert [x for x in diagnostics['off']['request_protocol'] if x.startswith(('T ', 'LP '))] == [x for x in diagnostics['on']['request_protocol'] if x.startswith(('T ', 'LP '))]

# This is captured-data content, not a claim these GPU libraries are loaded by Python.
files = copy.deepcopy(diagnostics['off']['mapping_snapshots']['after_gen']['files'])
qualified = [dict(path=p, sha256=x['sha256']) for p,x in diagnostics['off']['observed_libraries'].items()]
for p,x in files.items():
    assert Path(x['canonical_path']).resolve() == Path(x['canonical_path'])
    assert controller.stat_identity(Path(x['canonical_path'])) == x['stat']
    assert m.sha(Path(x['canonical_path'])) == x['sha256']
assert any(not x['original_runtime'] for x in files.values())
snapshot = dict(complete=True, selected_rows_stable=True, files=files,
                fixture_synthetic=True, fixture_provenance='immutable OFF after-GEN capture reused as identical synthetic phases')
phases = {phase:copy.deepcopy(snapshot) for phase in ('ready_before_gen', 'after_gen')}
mapping_checks = controller.validate_mapping_phases(phases, qualified)
observed = {p:x for p,x in files.items() if x['original_runtime']}
runtime_check = runtime.validate_runtime(observed, qualified)
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
order = [dict(sample_index=i, arm=arm) for i in range(4)
         for arm in (('off','on') if i % 2 == 0 else ('on','off'))]
order_sha = hashlib.sha256(json.dumps(order, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
closed = dict(observation_complete=True, direct_child_reaped=True, session_empty=True,
              session_anchored=True, errors=[], cleanup=[], exit_code=0,
              fixture_synthetic=True, label='synthetic-no-execution-command')
assert controller.closed_owned(closed)
assert not controller.closed_owned(dict(closed, session_anchored=False))
assert controller.exact_health(controller.HEALTH_PROTOCOL)
assert not controller.exact_health(['bogus same vector']*6)
results.append(dict(case='exported_health_and_anchored_closure', positive=2, negative=2))

def make_case(name, ordinal):
    fixture_base = root/name
    fixture_base.mkdir()
    controller.B = fixture_base  # The redirected namespace exists only in this CPU fixture process.
    qualification_path = fixture_base/'qualification.synthetic.json'
    qualification_path.write_text('{"fixture_synthetic":true}\n')
    d = diagnostics['off']
    identity = {k:copy.deepcopy(d[k]) for k in ('input','model_inputs','argv','cpu_exclusion_proof',
        'transfer_expectations','diagnostic_helper_pins','source_applied_main_review_sha256',
        'controller_main_review_sha256','health_probe')}
    identity.update(diagnostic_qualification=dict(fixture_synthetic=True, pair=PINS),
        quiet_switches_absent=['STRATA_TRACE','STRATA_PREFILL_TRANSFER_COUNTS','STRATA_PREFILL_TIMING'],
        environment_off=dict(ENV, STRATA_PRELOAD_KERNELS='0', STRATA_PREFILL_FUSED='0', STRATA_PREFILL_NATIVE_DEQUANT_BATCH='0'),
        qualified_runtime_libraries=qualified,
        effective_settings_without_vram={k:x for k,x in d['effective_settings'].items() if k != 'vram_free_mib'})
    q = dict(sample_indices=list(range(4)), sample_order=copy.deepcopy(order), sample_order_sha256=order_sha,
        campaign='synthetic-'+name, campaign_identity=identity, published_head='0740b43488d812117f021ccbe4bd0e15d223f955',
        production_source_pins=d['source_pins'], free_release_binary=d['binary'], source_patch_sha256=d['working_diff_sha256'],
        diagnostic_pair={arm:dict(path=str(B/'xe-native-dequant-prefill-model-root-v6'/f'n32768-g16-{arm}'/'record.json'),sha256=sha) for arm,sha in PINS.items()})
    item = order[ordinal]
    args = SimpleNamespace(sample_index=item['sample_index'], batch=item['arm'], reference=None,
        qualification=qualification_path, qualification_sha256=m.sha(qualification_path))
    campaign = fixture_base/'xe-native-dequant-prefill-quiet-timing-root-v3'/q['campaign']
    paths = []
    for i, old in enumerate(order[:ordinal]):
        path = campaign/f"sample-{old['sample_index']:03d}"/f"n32768-g16-{old['arm']}"/'record.json'
        path.parent.mkdir(parents=True)
        v = copy.deepcopy(diagnostics[old['arm']])
        v.update(fixture_synthetic=True, fixture_only=True, fixture_provenance=PINS[old['arm']],
            commands=[copy.deepcopy(closed)], complete=True, passed=True, active=False, boot_id=boot,
            campaign=q['campaign'], sample_index=old['sample_index'], batch=old['arm'], sample_order=q['sample_order'],
            sample_order_sha256=order_sha, sample_ordinal=i, pair_position=i % 2, first_arm=order[i-i % 2]['arm'],
            qualification_sha256=args.qualification_sha256, qualification_path=str(qualification_path),
            controller_sha256=CANDIDATE_SHA, published_head=q['published_head'], campaign_identity=copy.deepcopy(identity),
            environment=dict(identity['environment_off'],STRATA_PREFILL_NATIVE_DEQUANT_BATCH='1' if old['arm']=='on' else '0'),
            quiet_diagnostic_absence=True, actual_environment_matches=True,
            mapping_snapshots=copy.deepcopy(phases), observed_libraries=copy.deepcopy(observed),runtime_mapping_validation=copy.deepcopy(runtime_check),
            diagnostic_reference_comparison=dict(output_ids_equal=True,selected_logprob_top5_equal=True,**q['diagnostic_pair'][old['arm']]),
            model_normal_close=True,kernel_gate_passed=True,boot_unchanged=True,
            runtime_error_lines=[],startup_runtime_error_lines=[],pre_gen_runtime_error_lines=[],
            performance_eligible=False,full_model_quality_qualified=False,full262144_qualified=False)
        for k in identity:
            if k not in ('environment_off','qualified_runtime_libraries','effective_settings_without_vram'):
                v[k] = copy.deepcopy(identity[k])
        v['health-before_protocol'] = copy.deepcopy(controller.HEALTH_PROTOCOL)
        v['health-after_protocol'] = copy.deepcopy(controller.HEALTH_PROTOCOL)
        if i % 2:
            v['reference'] = dict(path=str(paths[-1].resolve()),sha256=m.sha(paths[-1]))
            v['reference_comparison'] = dict(output_ids_equal=True,selected_logprob_top5_equal=True)
        path.write_text(json.dumps(v)+'\n')
        paths.append(path)
    if ordinal % 2:
        args.reference = paths[-1]
    output = campaign/f"sample-{item['sample_index']:03d}"/f"n32768-g16-{item['arm']}"
    return q,args,paths,output

def admit(q,args,ordinal):
    _,planned,digest,actual = controller.planned_order(q,args)
    assert actual == ordinal
    return controller.admit_predecessors(q,args,32768,16,planned,digest,actual,A/'run_root.py')

def rejection(name, action, expected=None):
    try:
        action()
    except (RuntimeError,ValueError,FileNotFoundError) as error:
        if expected:
            assert expected in str(error), (name,str(error))
        results.append(dict(case=name,rejected=True,error=repr(error),model_launch=False))
    else:
        raise AssertionError('accepted negative fixture '+name)

for ordinal in range(5):
    q,args,paths,out = make_case('positive-'+str(ordinal),ordinal)
    prior, admitted, actual_boot, _ = admit(q,args,ordinal)
    assert len(prior)==ordinal and admitted==out and not os.path.lexists(out) and actual_boot==boot
    results.append(dict(case='actual_exported_admission',ordinal=ordinal,first_arm=order[ordinal-ordinal%2]['arm'],
        pair_position=ordinal%2,completed_prior_pairs=ordinal//2,synthetic=True,output_created=False,model_launch=False))

for name,mutate in [
    ('same-first',lambda q:q.update(sample_order=[dict(sample_index=i,arm=arm) for i in range(4) for arm in ('off','on')])),
    ('unbalanced',lambda q:q['sample_order'].__setitem__(slice(2,4),[dict(sample_index=1,arm=arm) for arm in ('off','on')])),
    ('duplicate-index',lambda q:q['sample_indices'].__setitem__(3,2)),
    ('invalid-index',lambda q:q['sample_indices'].__setitem__(0,True)),
    ('wrong-order-hash',lambda q:q.update(sample_order_sha256='0'*64))]:
    q,args,_,out = make_case('plan-'+name,0)
    mutate(q)
    rejection('planned-'+name,lambda:controller.planned_order(q,args))
    assert not os.path.lexists(out)

mutations = {
    'failed':lambda v:v.update(passed=False),
    'unanchored':lambda v:v['commands'][0].update(session_anchored=False),
    'wrongboot':lambda v:v.update(boot_id='wrongboot'),
    'wrongconfig':lambda v:v['argv'].append('--wrong-config'),
    'wrongindex':lambda v:v.update(sample_index=99),
    'wrongqualificationhash':lambda v:v.update(qualification_sha256='0'*64),
    'wrongcontrollerhash':lambda v:v.update(controller_sha256='0'*64),
    'crosscampaign':lambda v:v.update(campaign='wrong-campaign'),
    'bogus-identical-health':lambda v:v.update({'health-before_protocol':['bogus']*6,'health-after_protocol':['bogus']*6}),
    'missing-core':lambda v:v['observed_libraries'].pop(next(p for p in v['observed_libraries'] if Path(p).name=='libsycl.so.9.0.0-0')),
}
for name,mutate in mutations.items():
    q,args,paths,out = make_case('prior-'+name,1)
    v=json.loads(paths[0].read_text());mutate(v);paths[0].write_text(json.dumps(v)+'\n')
    rejection('admission-'+name,lambda:admit(q,args,1))
    assert not os.path.lexists(out)

q,args,_,out=make_case('missing-prior',0)
args.sample_index=0;args.batch='on';args.reference=out/'record.json'
rejection('admission-missing-prior',lambda:admit(q,args,1),'missing')
assert not os.path.lexists(out)
q,args,_,out=make_case('wrong-reference',1);args.reference=root/'unrelated-record.json'
rejection('admission-wrong-reference',lambda:admit(q,args,1),'--reference')
assert not os.path.lexists(out)
q,args,paths,out=make_case('wrong-pair-refhash',2)
v=json.loads(paths[1].read_text());v['reference']['sha256']='0'*64;paths[1].write_text(json.dumps(v)+'\n')
rejection('admission-wrong-pair-reference-hash',lambda:admit(q,args,2),'completed pair')
assert not os.path.lexists(out)
for name,target in [('future-arm',order[3]),('skipped-arm',order[6])]:
    q,args,_,out=make_case(name,0)
    future=out.parents[1]/f"sample-{target['sample_index']:03d}"/f"n32768-g16-{target['arm']}"
    future.mkdir(parents=True)
    before=set(out.parents[1].rglob('*'))
    rejection('admission-'+name,lambda:admit(q,args,0),'out-of-order')
    assert set(out.parents[1].rglob('*'))==before and not os.path.lexists(out)
for name in ('leaf-symlink','ancestor-symlink','dangling-leaf-symlink'):
    q,args,_,out=make_case(name,0)
    target=root/(name+'-target');target.mkdir()
    if name=='ancestor-symlink':
        out.parent.parent.mkdir(parents=True);out.parent.symlink_to(target,target_is_directory=True)
    else:
        out.parent.mkdir(parents=True);out.symlink_to(target if name=='leaf-symlink' else root/'never-created-target',target_is_directory=True)
    rejection('admission-'+name,lambda:admit(q,args,0),'unaliased')
    assert not (out/'record.json').exists()

for name in ('compiler-phase-set-drift','compiler-phase-hash-drift','missing-original-core'):
    wrong=copy.deepcopy(phases)
    extra=next(p for p,x in wrong['after_gen']['files'].items() if not x['original_runtime'] and
        any(key in Path(p).name.lower() for key in ('igc','fcl','clang','iga','igdgmm')))
    if name=='compiler-phase-set-drift':del wrong['after_gen']['files'][extra]
    elif name=='compiler-phase-hash-drift':wrong['after_gen']['files'][extra]['sha256']='0'*64
    else:
        core=next(p for p in wrong['after_gen']['files'] if Path(p).name=='libsycl.so.9.0.0-0')
        del wrong['after_gen']['files'][core]
    rejection('mapping-'+name,lambda:controller.validate_mapping_phases(wrong,qualified))
optional_absent=copy.deepcopy(phases)
for phase in optional_absent.values():
    phase['files']={p:x for p,x in phase['files'].items() if Path(p).name not in runtime.OPTIONAL}
controller.validate_mapping_phases(optional_absent,qualified)
results.append(dict(case='optional-runtime-absent-accepted',runtime_original_selftest=runtime.selftest()))

# Actual host-only mapping snapshot and external ELF-note comparison; no GPU query.
entry=dict(pid=os.getpid(),direct_identity=m.identity(os.getpid()))
budget=dict(bytes=0)
live=controller.mapping_snapshot(entry,lambda:None,budget)
assert live['complete'] and live['selected_rows_stable'] and len(live['files'])>=2
for path,x in live['files'].items():
    assert m.sha(Path(path))==x['sha256']
    observed=subprocess.run(['/usr/bin/readelf','-n',x['canonical_path']],env=ENV,
        capture_output=True,text=True,timeout=10,check=True)
    assert set(re.findall(r'Build ID: ([0-9a-f]+)',observed.stdout))=={x['elf_build_id']}
results.append(dict(case='actual_CPU_maps_vs_readelf',files=live['files'],bytes=budget['bytes']))

# AST extraction happens only when Main executes this future CPU recipe.
tree=ast.parse((A/'run_root.py').read_text())
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
body=next(n for n in main.body if isinstance(n,ast.With)).body
run_fn=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name=='run')
try_body=next(n for n in body if isinstance(n,ast.Try)).body
interaction_fn=next(n for n in try_body if isinstance(n,ast.FunctionDef) and n.name=='interaction')
for post in (False,True):
    case=root/('owned-QUIT-post-DONE' if post else 'owned-QUIT-pre-GEN');case.mkdir()
    state=dict(gen_send_attempted=post,mapping_snapshots={'after_gen':dict(complete=post)})
    owned=m.Owner(case,lambda:None)
    def model_interaction(io,e):
        assert io.line(5)=='READY 262144';state['ready_seen']=True
        if post:
            assert io.line(5)=='DONE stop';state['request_done_seen']=True
        raise ValueError('injected primary gate failure')
    def no_capture(*unused):raise AssertionError('unexpected mapping capture')
    ns=dict(m=m,r=state,owner=owned,begin=time.monotonic(),time=time,W=A,BASE=ENV,
        closed_owned=controller.closed_owned,model_interaction=model_interaction,capture_maps=no_capture,save=lambda:None)
    fragment=ast.Module(body=[copy.deepcopy(run_fn),copy.deepcopy(interaction_fn)],type_ignores=[])
    exec(compile(ast.fix_missing_locations(fragment),str(A/'run_root.py'),'exec'),ns)
    child="import sys\nprint('READY 262144',flush=True)\n"+("print('DONE stop',flush=True)\n" if post else '')+"command=sys.stdin.readline();sys.exit(0 if command=='QUIT\\n' else 7)\n"
    try:ns['run']('model',['/usr/bin/python3','-u','-c',child],wall=15,interaction=ns['interaction'])
    except RuntimeError as error:assert 'injected primary gate failure' in str(error)
    else:raise AssertionError('primary failure masked')
    assert state['model_normal_close'] and not state['model_close']['forced']
    assert state['model_close']['quit_send_outcome']=='sent' and state['model_close']['quit_exit_outcome']=='exit observed'
    assert all(controller.closed_owned(c) and c['exit_code']==0 for c in owned.commands)
    results.append(dict(case='actual_owned_CPU_QUIT_preserves_primary',post_DONE=post,result=state,commands=owned.commands))

for explicit in (False,True):
    argv=['/usr/bin/python3',str(A/'run_root.py'),'--input-tokens','32768','--group','16','--sample-index','0','--batch','off']
    if explicit:argv+=['--qualification-sha256',m.sha(A/'qualification.pending.json')]
    child=subprocess.run(argv,env=ENV,capture_output=True,text=True,timeout=10)
    assert child.returncode==1 and ('no pending launch' if explicit else 'manifest hash') in child.stderr
    assert not (B/'xe-native-dequant-prefill-quiet-timing-root-v3'/'PENDING_MAIN_IMMUTABLE_QUIET_CAMPAIGN_NAME').exists()
    results.append(dict(case='pending_qualification_rejected',explicit_hash=explicit,exit_code=child.returncode,stderr=child.stderr[-2048:]))
assert all(m.sha(B/'xe-native-dequant-prefill-model-root-v6'/f'n32768-g16-{arm}'/'record.json')==digest for arm,digest in PINS.items())
assert m.sha(A/'run_root.py')==CANDIDATE_SHA
print(json.dumps(dict(passed=True,fixture_only=True,positive_protocol_cases=namespace['positive'],
    negative_protocol_cases=namespace['negative'],cases=results,actual_case_count=len(results),
    actual_rejected_case_count=sum(x.get('rejected',False) for x in results),
    gpu_executed=False,model_executed=False,synthetic_receipt_namespace=str(root),
    limitations='CPU exported-predicate fixtures, actual host mappings/readelf and owned CPU QUIT only; no GPU/model, full-context or performance qualification.')))
