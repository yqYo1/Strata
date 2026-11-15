#!/usr/bin/env python3
"""Observe host-only MoE uploads and independent decode handoff.

Main-authored from the maintained owned streamed-model controls; repo LICENSE.
This diagnostic uses the compiled candidate, not an inherited model result.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_moe_census_model as device
import moe_upload_census as geometry
import qsa16_stream_model_entry as original
import qsa16_stream_bench as benchmark
import qsa_bulk_mirror as component

P, B = device.P, device.B
ROOT, KEY = B / 'moe-host-census-model-v1', 'moe_host_census_model_v1'
THIS = 'tools/sycl/moe_host_census_model.py'
PYTHON = '/home/yayoi/.local/share/strata-sycl/venv/bin/python3'
require, sha = original.require, original.sha


def weight_formats():
    reference = json.loads(original.REFERENCE.read_text())
    argv = reference['argv']; pack = Path(argv[argv.index('--pack')+1])
    path = pack/'native_experts.txt'
    require(str(path) in reference['models'], 'native format descriptor must be a pinned model asset')
    rows = [v.split() for v in path.read_text().splitlines() if v and not v.startswith('#')]
    require(len(rows)==48 and [int(v[0]) for v in rows]==list(range(48)), '48 ordered native formats required')
    return [(int(v[1]),int(v[2]),int(v[4])) for v in rows]


def upload_counts(text, formats):
    def frames(label):
        return [json.loads(x,object_pairs_hook=geometry.unique_object) for x in
            re.findall(r'^strata MoE '+label+r' census: (.+)$',text,re.M)]
    planned, uploads = frames('planned job'), frames('upload')
    require(len(formats)==48 and len(planned)==len(uploads)==96,
        'two complete ordered planned/upload windows required')
    samples=[]
    for start in (0,48):
        ps,us=planned[start:start+48],uploads[start:start+48]
        shape=geometry.census(''.join('strata MoE planned job census: '+json.dumps(x)+'\n' for x in ps),32769)
        require([u.get('layer') for u in us]==list(range(48)), 'ordered accepted-upload coverage')
        folded={}
        for layer,(p,u,fmt) in enumerate(zip(ps,us,formats)):
            keys={'version','layer','first_row','rows','gu_type','down_type','blob_bytes',
                'accepted_uploads','cache_hits','requested_upload_bytes','final_drain',
                'accepted_upload_rows_histogram','cache_hit_rows_histogram'}
            require(set(u)==keys and all(type(u[k]) is int for k in keys-
                {'final_drain','accepted_upload_rows_histogram','cache_hit_rows_histogram'}),
                'exact integer upload metadata schema required')
            require(u['version']==1 and u['layer']==layer and u['first_row']==0 and
                u['rows']==32768 and (u['gu_type'],u['down_type'],u['blob_bytes'])==fmt and
                u['final_drain']=='checked', 'actual format/window/final-drain differs')
            hist=[]
            for key in ('accepted_upload_rows_histogram','cache_hit_rows_histogram'):
                require(isinstance(u[key],list), 'upload histogram list required')
                hist.append(geometry.histogram(u[key],1,8192) if u[key] else {})
            miss,hit=hist
            require(sum(miss.values())==u['accepted_uploads'] and sum(hit.values())==u['cache_hits']
                and 0<=u['accepted_uploads']<=512 and 0<=u['cache_hits']<=512
                and u['accepted_uploads']+u['cache_hits']==512-dict(p['jobs_per_expert_histogram']).get(0,0)
                and u['requested_upload_bytes']==u['accepted_uploads']*fmt[2],
                'accepted upload/hit/expert/byte conservation failed')
            jobs=dict(p['job_rows_histogram']);combined={n:miss.get(n,0)+hit.get(n,0) for n in set(miss)|set(hit)}
            require(all(v<=jobs.get(n,0) for n,v in combined.items()), 'first jobs must be a planned-job subset')
            if all(n<=1 for n,_ in p['jobs_per_expert_histogram']):
                require(combined==jobs,'unsplit first-job histogram must equal planned jobs')
            key='/'.join(map(str,fmt))
            f=folded.setdefault(key,dict(layers=0,accepted_uploads=0,cache_hits=0,
                requested_upload_bytes=0,accepted_first_job_rows={},hit_first_job_rows={}))
            f['layers']+=1
            for k in ('accepted_uploads','cache_hits','requested_upload_bytes'):f[k]+=u[k]
            for target,values in ((f['accepted_first_job_rows'],miss),(f['hit_first_job_rows'],hit)):
                for n,count in values.items():target[n]=target.get(n,0)+count
        for f in folded.values():
            for key in ('accepted_first_job_rows','hit_first_job_rows'):
                values=f[key]
                f[key]=[dict(lower=a,upper=b,count=sum(c for n,c in values.items() if a<=n<b),
                    rows=sum(n*c for n,c in values.items() if a<=n<b))
                    for a,b in zip(geometry.BOUNDS,geometry.BOUNDS[1:])
                    if any(a<=n<b for n in values)]
        samples.append(dict(planned=shape,formats=folded,
            accepted_uploads=sum(x['accepted_uploads'] for x in folded.values()),
            cache_hits=sum(x['cache_hits'] for x in folded.values()),
            requested_upload_bytes=sum(x['requested_upload_bytes'] for x in folded.values()),
            accepted_frames_sha256=original.signature(us)))
    return dict(requests=samples,actual_format_and_first_upload_counts=True,
        limits='Host accepted uploads/hits after checked existing drains. First-job rows are not all expert rows when split. Requested payload is not physical PCIe traffic,critical-path time,hardware saturation or speed.')


def parser_controls():
    docs=[];uploads=[];formats=[(22,20,1971200)]*48
    for layer in range(48):
        docs.append(dict(version=1,layer=layer,first_row=0,rows=32768,jobs=512,
            gu_nk=[1280,2560],down_nk=[2560,640],gu_down_storage='bf16',hidden_storage='fp16',
            allocated_weight_pairs=2,used_weight_pairs=1,weight_pair_bytes=9830400,
            requested_router_D2H_bytes=1310720,requested_map_H2D_bytes=2621440,
            job_rows_histogram=[[640,512]],jobs_per_expert_histogram=[[1,512]]))
        uploads.append(dict(version=1,layer=layer,first_row=0,rows=32768,gu_type=22,down_type=20,
            blob_bytes=1971200,accepted_uploads=256,cache_hits=256,requested_upload_bytes=504627200,
            final_drain='checked',accepted_upload_rows_histogram=[[640,256]],cache_hit_rows_histogram=[[640,256]]))
    def render(us):
        return ''.join('strata MoE planned job census: '+json.dumps(p)+'\n'+
            'strata MoE upload census: '+json.dumps(u)+'\n' for p,u in zip(docs*2,us))
    good=uploads*2
    value=upload_counts(render(good),formats)
    require(all(x['accepted_uploads']==x['cache_hits']==12288 and
        x['requested_upload_bytes']==24222105600 for x in value['requests']), 'literal mixed upload/hit totals')
    import copy
    for misses in (0,512):
        test=copy.deepcopy(good)
        for u in test:
            u.update(accepted_uploads=misses,cache_hits=512-misses,
                requested_upload_bytes=1971200*misses,
                accepted_upload_rows_histogram=[[640,misses]] if misses else [],
                cache_hit_rows_histogram=[[640,512-misses]] if misses<512 else [])
        value=upload_counts(render(test),formats)
        require(all(x['accepted_uploads']==48*misses and x['cache_hits']==48*(512-misses)
            and x['requested_upload_bytes']==48*1971200*misses for x in value['requests']),
            'literal empty/all-upload/all-hit histogram totals')
    cases=[]
    for key,bad in (('accepted_uploads',True),('cache_hits',255),('gu_type',18),('blob_bytes',1),
        ('requested_upload_bytes',0),('final_drain','unchecked'),('rows',8192),('layer',1),
        ('accepted_upload_rows_histogram',[[8193,256]]),('accepted_upload_rows_histogram',[[641,256]]),('cache_hit_rows_histogram',[[640,255]]),
        ('accepted_upload_rows_histogram',[[640,128],[640,128]])):
        test=copy.deepcopy(good);test[0][key]=bad;cases.append(render(test))
    cases.append(render(good[:-1]))
    cases.append(render(good).replace('\"version\": 1','\"version\": 1,\"version\": 1',1))
    for text in cases:
        try:upload_counts(text,formats)
        except RuntimeError:pass
        else:raise RuntimeError('invalid upload control accepted')
    return dict(passed=True,positive=3,negative=len(cases),GPU_executed=False,
        scope='Literal mixed miss/hit/byte,format/window/drain,coverage,duplicate-bin and first-job conservation controls;not model or bus evidence.')


def native(expected=None):
    """Admit the fresh nine-stage result, without a legacy repair receipt."""
    dev = device.context().context()
    n = dev.native(observers={dev.THIS, THIS})
    prior = dev.validation.committed(dev.RESULT, (device.KEY,))
    results = prior['results']
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete']
        and prior['ownership_closed'] and not prior['active']
        and prior['binaries'] == n['binaries']
        and prior['native_source_commit'] == n['native_source_commit']
        and [v['mode'] for v in results] == list(device.STAGES)
        and all(v['passed'] and v['ownership_closed'] and v['native_exit_code'] == 0
                for v in results), 'fresh closed nine-stage same-native result required')
    host = results[0]
    require(host['host_census_controls']['passed'] and not host['GPU_executed'] and host['reader_cases'] == 54
        and host['native_cases'] == 11 and host['policy_check']['passed']
        and host['policy_check']['native_exit_code'] == 0
        and host['selected_head_IR']['same_as_current_reference']
        and host['selected_head_IR']['engine_fixture_identical'],
        'fresh reader/refusal/policy/head-module controls required')
    require(all(v['GPU_executed'] and v['native_execution_attempted']
        and v['host_probe_accepted'] and v['fresh_faults'] == []
        and v['boot_unchanged'] and v['driver_after'] == prior['driver']
        for v in results[1:]), 'fresh normal device closures without faults required')
    qsa = results[6]['checks']
    require(len(qsa) == 1 and qsa[0]['name'] == 'numerical'
        and sum(v.startswith('case cells=') for v in qsa[0]['cases']) == 11
        and any(v.startswith('case cells=262144 live_cells=262144 ') for v in qsa[0]['cases']),
        'fresh actual262144 independent attention boundary required')
    producer = results[8]['checks']
    require([v['flag'] for v in producer] == ['0','1']
        and all(v['physical_cells'] == 262144 for v in producer)
        and producer[1]['bulk_calls'] == 33 and producer[1]['accepted_copy_commands'] == 136
        and producer[1]['requested_D2H_bytes'] == 262145*1056,
        'fresh off/on full producer fill and last-cell overwrite required')
    require(prior['native_observer_admission']['exact_seed_recipe_sha256']
        == n['observer_admission']['exact_seed_recipe_sha256'],
        'same exact qualified compiler recipe required')
    admission = prior['source_commit']
    source = dev.common.source.git(P, 'show', admission + ':' + dev.THIS)
    source_sha = hashlib.sha256(source).hexdigest()
    require(source_sha == prior['native_observer_admission']['added_observers'][dev.THIS]
        == n['observer_admission']['added_observers'][dev.THIS],
        'original qualified device-reader source must remain exact and recoverable')
    proof = dict(n['observer_admission'], device_key=device.KEY,
        device_evidence_sha256=original.signature(prior),
        qualified_device_observer_source=admission,
        qualified_device_observer_sha256=source_sha,
        fresh_nine_stage_controls=True,
        changed_controller_scope='This new model reader only;fresh nine-stage device '
            'evidence and exact original native/helper/fixture/recipe/SDK/ELF inputs. '
            'No historical controller-repair field or earlier model result inherited.',
        earlier_model_result_inherited=False)
    require(expected is None or proof == expected, 'cached-subgroup model admission changed')
    n['model_observer_admission'] = proof
    return n


def route(text):
    result = benchmark.fast_route(text)
    values = re.findall(r'^strata QSA mirror bulk: tokens=(\d+) pos0=(\d+) '
        r'segments=(\d+) copies=(\d+) requested_D2H_bytes=(\d+) '
        r'queue=same accepted=1 host_quantization=0$', text, re.M)
    expected = [(8192, pos, 1, 4, 8192 * 1056)
                for _ in range(2) for _ in range(12) for pos in (0, 8192, 16384, 24576)]
    require([tuple(map(int, v)) for v in values] == expected,
        'both actual 32K long-prefills must publish exact bulk mirror ranges')
    arithmetic = re.findall(r'^strata prefill: QSA BF16 operands, FP32 online state/accumulation; '
        r'KV=(\S+) transient_bytes=(\d+); source=resident-Mode0 score_pitch=(\d+) '
        r'operand_layout=(\S+) softmax=(\S+)$', text, re.M)
    require(len(arithmetic) == 2 and all(v[0] == 'active-logical-prefix' and int(v[1]) > 0
        and v[2:] == ('33','row-major','parallel-exp-ascending-sum') for v in arithmetic),
        'actual ordered-exp arithmetic must be selected for both long entries')
    for diagnostic in ('strata prefill: QSA cached_query=1 grf_policy=default source=qualified-component',
            'strata prefill: QSA k_staging=typed4 parallel_rows=1 source=qualified-component',
            'strata prefill: QSA v_staging=typed4 layout=row-major source=qualified-component'):
        require(text.splitlines().count(diagnostic) == 2, 'actual cached/vector operand selection missing')
    result['actual_attention_math'] = 'bf16-prefix-ordered-exp'
    result['moe_host_census'] = upload_counts(text, weight_formats())
    result.update(actual_bulk_calls=len(values), accepted_copy_commands=384,
        requested_D2H_bytes=2 * 12 * 32768 * 1056,
        bulk_route_observed=True, physical_bus_traffic_qualified=False)
    return result


def context():
    spec = importlib.util.spec_from_file_location('qsa_bulk_actual_model', original.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.device = device.context().context()
    ctx.native = native
    # Reuse the existing owned-stage host/execution checks. Preparation below
    # is fresh; it does not enter the old attempt's historical V2 branch.
    ctx.ATTEMPT = 'v2'
    ctx.route = route
    return ctx


def prepare():
    ctx = context()
    require(not ROOT.exists() and KEY not in ctx.device.common.public(), 'fresh model plan only')
    native = ctx.native()
    prior = ctx.validation.committed(ctx.RESULT, (device.KEY,))
    reference = json.loads(ctx.REFERENCE.read_text())
    require(reference['passed'] and reference['complete'] and reference['ownership_closed'],
        'closed original model/configuration assets required')
    argv = list(reference['argv'])
    argv[0] = native['binaries']['strata']['path']
    for flag, value in (('--max-context', '262144'), ('--kv-resident', '20480'), ('--prefill', '8192')):
        require(argv.count(flag) == 1, 'one argument required: ' + flag)
        argv[argv.index(flag) + 1] = value
    old = reference['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(ROOT)) for k, v in reference['environment'].items()}
    require('STRATA_KV_STAGE_OWN' not in env, 'owned stage must be explicit')
    env.update(benchmark.FAST, TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        STRATA_MEASUREMENT_ROOT=str(ROOT), STRATA_KV_STAGE_OWN='1',
        STRATA_SYCL_PREFILL_QSA_STREAM='1', STRATA_SYCL_DECODE_Q6_DPAS_HEAD='2',
        STRATA_SYCL_DECODE_Q6_DPAS_DENSE='1', STRATA_PREFILL_HEAD_HOST_BACKING='0',
        STRATA_SYCL_PREFILL_QSA_MATH='bf16-prefix-ordered-exp',
        STRATA_SYCL_PREFILL_MOE_CENSUS='1', STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
        STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING='0', STRATA_TRACE='1', UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'),
        **{component.FLAG: '1'})
    env.pop('UR_LOG_TRACING', None)
    require(env['STRATA_SYCL_PREFILL_NATIVE16'] == 'bf16'
        and env['STRATA_SYCL_PREFILL_NATIVE16_QSA'] == '1'
        and env['STRATA_SYCL_PREFILL_QSA_MATH'] == 'bf16-prefix-ordered-exp'
        and env['STRATA_PREFILL_LAYER_MAJOR_PROFILE'] == '0'
        and env['STRATA_SYCL_PREFILL_MOE_CENSUS'] == '1'
        and all(env.get(k) == v for k, v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
            NEOReadDebugKeys='1', EnableDirectSubmission='0', STRATA_PREFILL_RELEASE_HEADS='1',
            STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0').items()), 'preserve typed route and safety')
    owned = ctx.context(native['model_observer_admission'])
    row = dict(stage='prepared',
        source_commit=ctx.device.common.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        controller=owned.controller_pin(), native_source_commit=native['native_source_commit'],
        native_build=dict(binaries=native['binaries']),
        native_record=dict(path=str(ctx.device.NATIVE / 'record.json'),
            sha256=sha(ctx.device.NATIVE / 'record.json')),
        native_observer_admission=native['model_observer_admission'],
        device_evidence_sha256=ctx.signature(prior), models=reference['models'],
        models_sha256=reference['models_sha256'], runtime_manifest=prior['runtime_manifest'],
        runtime_files=reference['runtime_files'], probe=prior['native_validation_probe'],
        generator_runtime=ctx.previous.prompts.generator_runtime(), tokenizer=reference['tokenizer'],
        requests=ctx.requests(Path(reference['tokenizer'])), environment=env, argv=argv,
        boot=prior['boot'], driver=prior['driver'],
        capacity=dict(full_stage_payload_bytes=262144 * 1056,
            logical_host_KV_payload_bytes=12 * 262144 * 1056,
            stage_allocation='Separately owned264MiB;actual8192chunk still required.'),
        budgets=dict(wall=480, cpu=5760, rss_cap=104 << 30,
            text_cap=32 << 20, file_cap=32 << 20, total_cap=128 << 20),
        plan='Host INFO/stage checks,then one fresh thinking-enabled32769/82/32769input '
            'process with32outputs each. New97fda7ba/nativedc645342 and eight same-build '
            'fixtures;logical262144/resident20480INT8,own264MiBstage,8192chunks,1256decode '
            'slots,spec4,head2/dense1/Hostbacking0,nine original FAST settings and bulk mirror1. '
            'LM_PROFILE=0,CENSUS=1;require actual96bulk calls/384copies/830472192requested bytes,full fast-route '
            'bank/MTP restoration,48planned+accepted-upload frames per long request,model-native format agreement,job/expert/byte conservation and independent decode recapture,READY/INFO/source/provider/'
            'protocol/quiet-validation/normal0/closed-owner/unchanged-boot/no-fault checks. '
            'Full15model hashes before/after. No retries,pin refresh or environment mutation.',
        limits='Actual candidate route/handoff diagnostic only;validation/logging charged. '
            'No clean PP/TG/TTFT,gain,task quality,sustained decode,physical262144model/tails/'
            'RESTORE,ordinary-provider teardown or adoption. Follow with matched repeated '
            'same-binary off/on32K/64Kbench.sh and separate independent decode evidence.',
        upload_parser_controls=parser_controls(),
        results=[], commands=[], passed=False, complete=False, active=False,
        ownership_closed=True, model_executed=False, speed_eligible=False,
        model_adoption_qualified=False,
        retention='Keep compact actual dispatch/handoff and original identities. Retire '
            'successful full stage/UR logs and generated IDs after reduction. Five temporary '
            'temporary engine generations are unchanged;select candidate or56cad9 at '
            'matched phase closure and retire superseded build after source recovery.')
    owned.assets(row, full=True)
    ROOT.mkdir(mode=0o700)
    for part in ('tmp', 'cache'):
        (ROOT / part).mkdir(mode=0o700)
    owned.save(row)
    print('Prepared host-only MoE census long/short/long diagnostic;commit before host/model.', flush=True)


if __name__ == '__main__':
    require(sys.executable == PYTHON,
        'Use the existing pinned prompt-generator interpreter: ' + PYTHON)
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', 'run'),
        'moe_host_census_model.py prepare|host|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            context().execute(sys.argv[1])
