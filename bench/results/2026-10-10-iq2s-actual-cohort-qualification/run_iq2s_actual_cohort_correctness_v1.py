"""Root-owned CPU-only actual-weight IQ2_S numerical admission, no timing."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import importlib.util
import json
import re
import time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010')
G = Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
HELPER = B / 'run_native_service_calibration_cpu_v4.py'
BUILD = B / 'iq2s-actual-cohort-cpu-build-v1/record.json'
PRIOR = B / 'native-service-calibration-22-20-nt1-correctness-tasks6-batch6-r3/record.json'
OWNER = B / 'native-service-calibration-cohort-22-20-nt1-v2'
PACK = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
PRIMARY = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
PINS = {
    HELPER: '5d16615452013fbfdba190e382e750ab7ddaad87013daaa879b5348558b0e676',
    BUILD: '292849fedd2f0b9460304f3c3f317d3b9fcab483734e3d95103d23e4c1131de1',
    PRIOR: '62707c2acbc0c1ab8baef915b4a2bb3c24d2b795ee31cae1235f644a1000b8bb',
    OWNER / 'manifest.json': '6a5eac378b01531a7438fa512bb0829bc7e447028328cfa5aba9c64a02377304',
    OWNER / 'cohort.tsv': '49ab6545beda7a1d16cc02da07981b80d4fd36fc4fa9385f56039771fb4d21f1',
    PACK / 'native_experts.txt': 'd9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d',
    PACK / 'conversions.json': '51df3cd6ffd0d38c2da8e2d3a95a604b4a96c20b06fc639a9bf477282a42b86a',
}
ENV = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8')

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def canonical(v):
    return hashlib.sha256((json.dumps(v, sort_keys=True, separators=(',', ':'))+'\n').encode()).hexdigest()

def check_build(build):
    assert build['passed'] and build['complete'] and not build['active']
    assert not build['cleanup'] and not build['survivors']
    assert not build['gpu_work_submitted'] and not build['inference_run']
    for p, h in PINS.items():
        assert sha(p) == h, str(p)
    for p, h in build['pins'].items():
        assert sha(p) == h, p
    for p, h in build['production_source_pins'].items():
        assert sha(W/p) == h, p
    for p, h in build['ggml_source_pins'].items():
        assert sha(G/p) == h, p
    assert sha(build['binary']['path']) == build['binary']['sha256']

def extent_hashes(extents):
    result = []
    for item in extents:
        p = Path(item['path']); off, n = item['offset'], item['bytes']
        assert p.parent == PRIMARY.parent and 0 <= off <= p.stat().st_size-n
        assert 0 < n < 4 << 20
        with p.open('rb') as f:
            f.seek(off); data = f.read(n)
        assert len(data) == n
        assert hashlib.sha256(data).hexdigest() == item['sha256'], 'selected extent changed'
        result.append(item)
    assert len(result) == 1152 and sum(x['bytes'] for x in result) == 756940800
    return dict(extents=1152, bytes=756940800, canonical_json_sha256=canonical(result),
                expected_source_receipt_sha256=PINS[PRIOR], all_individual_sha256_match=True)

def validate(rows, manifest, extents):
    assert rows[-1] == ['RESULT','iq2s_index_correctness_only_pass','no_timing_no_adoption']
    assert [x for x in rows if x[:1] == ['INDEX_COMPLETE']] == [['INDEX_COMPLETE','384','245760','491520','1474560']]
    expected = [(co, co*192+i, x['layer'], x['expert'])
                for co, label in enumerate(('train','holdout')) for i,x in enumerate(manifest['cohorts'][label])]
    ids = [x for x in rows if x[:1] == ['ID']]
    assert len(ids) == 384 and [tuple(map(int,x[1:5])) for x in ids] == expected
    got_extents = [x for x in rows if x[:1] == ['EXTENT']]
    assert len(got_extents) == 1152 and all(len(x) == 9 for x in got_extents)
    actual = [dict(cohort=int(x[1]),layer=int(x[2]),expert=int(x[3]),role=int(x[4]),
                   path=x[5],tensor=x[6],offset=int(x[7]),bytes=int(x[8])) for x in got_extents]
    assert actual == [{k:v for k,v in x.items() if k!='sha256'} for x in extents]
    env = {x[1]:x[2] for x in rows if x[:1] == ['ENV']}
    assert len(env) == 16 and all(v=='<unset>' for v in env.values())
    meta = {x[1]:x[2] for x in rows if x[:1] == ['META']}
    mxcsr = {k:int(meta['index_mxcsr_'+k]) for k in ('entry','prepared','final')}
    assert all((x & 0xffc0) == 0x1f80 for x in mxcsr.values()), 'absolute MXCSR contract'
    assert meta['index_worker_mxcsr'] == 'not_observed_workers_not_started'
    assert not any(x[0] in ('TASK_PLAN','PLACEMENT','WARMUP','ROUND','REFERENCE','INDEX_FIRST_MISMATCH') for x in rows)
    checked = [x for x in rows if x[:1] == ['INDEX_ID']]
    assert len(checked) == 384 and all(len(x) == 36 for x in checked)
    assert [tuple(map(int,x[1:5])) for x in checked] == expected
    activation = {}; split_classes = [[0]*27 for _ in range(2)]
    for x in checked:
        co, _, layer, _ = map(int,x[1:5])
        assert x[5:7] == ['640','0'] and int(x[7]) > 0
        activation.setdefault(layer,x[8]); assert activation[layer] == x[8]
        values = list(map(int,x[9:])); assert all(n >= 0 for n in values)
        for j in range(0,27,3): assert values[j:j+3] == [640,0,0]
        split_classes[co] = [a+b for a,b in zip(split_classes[co],values)]
    assert len(activation) == 15
    splits = [x for x in rows if x[:1] == ['INDEX_SPLIT']]
    assert len(splits) == 2 and all(len(x) == 33 for x in splits)
    for co,x in enumerate(splits):
        assert x[1:5] == [str(co),'192','122880','0']
        assert list(map(int,x[6:])) == split_classes[co]
    return dict(experts=384,layers=15,row_pairs=245760,dot_calls=1474560,
                baseline_control_register_Gate_Up_finish_all_bits_equal=True,
                nonfinite_values=0,split_rows=[122880,122880],mxcsr=mxcsr,
                mxcsr_control=hex(0x1f80),rounding='nearest-even',FTZ=False,DAZ=False,
                activation_hash_by_layer=activation,no_pool_or_timing_records=True)

def main():
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for p,h in PINS.items(): assert sha(p)==h,str(p)
        spec=importlib.util.spec_from_file_location('root_cpu_child',HELPER)
        h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.W=W
        build=json.loads(BUILD.read_text());check_build(build)
        prior=json.loads(PRIOR.read_text());assert prior['passed'] and prior['complete'] and not prior['active'] and not prior['cleanup'] and not prior['survivors']
        manifest=json.loads((OWNER/'manifest.json').read_text());extents=prior['selected_extent_sha256']
        assert canonical(extents)=='b8a9834741ca63637ead3f0b72ad63e90ced6d4182f0b48e9687b7818962a58d'
        out=B/'iq2s-actual-cohort-correctness-v1';out.mkdir(mode=0o700)
        record=dict(active=True,complete=False,passed=False,controller_sha256=sha(__file__),
                    helper_sha256=PINS[HELPER],build_receipt_sha256=PINS[BUILD],binary=build['binary'],
                    input_pins={str(p):v for p,v in PINS.items()},commands=[],cleanup=[],survivors=[],
                    scope='Actual-weight three-arm CPU numerical admission, synthetic finite activations; no timing or model inference',
                    gpu_work_submitted=False,inference_run=False,performance_eligible=False,adopted=False,
                    full_lifecycle_passed=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    resource_policy=dict(AS_each_bytes=96<<30,RSS_group_poll_bytes=1536<<20,CPU_soft_seconds=900,
                                         CPU_hard_seconds=901,FSIZE_each_bytes=4<<20,text_budget_bytes=8<<20,CORE=0,NOFILE=256))
        h.save(record,out);start=time.monotonic()
        try:
            conversions=json.loads((PACK/'conversions.json').read_text())
            model=[h.identity(PRIMARY.parent/x['name']) for x in conversions['source_shards']]
            assert model==prior['model_identity'];record['model_identity']=model
            record['whole_model_payload_hashed']=False
            record['selected_extents_before']=extent_hashes(extents);h.save(record,out)
            def argv(gu=22,nt=1,tasks=0,batch=1,missing=False):
                return [build['binary']['path'],str(out/'missing-pack') if missing else str(PACK),
                        str(out/'missing-primary') if missing else str(PRIMARY),str(OWNER/'cohort.tsv'),
                        str(gu),'20',str(nt),str(tasks),str(batch),'2026101001','--iq2s-index-correctness-only']
            for label,kw in [('wrong-GU',dict(gu=21)),('wrong-NT',dict(nt=2)),('wrong-tasks',dict(tasks=6)),('wrong-batch',dict(batch=6))]:
                trace=out/(label+'-syscalls.txt')
                h.run_child(record,out,label,['/usr/bin/strace','-f','-yy','-s','256','-e','trace=open,openat,openat2,ioctl,execve','-o',str(trace)]+argv(missing=True,**kw),ENV,
                            expect_error='index mode requires GU22 Down20 NT1 tasks0 batch1',wall=20)
                text=trace.read_text();assert 'missing-pack' not in '\n'.join(x for x in text.splitlines() if 'open' in x)
                assert 'missing-primary' not in '\n'.join(x for x in text.splitlines() if 'open' in x)
                assert not re.search(r'/dev/(dri|nvidia|kfd)|lib(?:sycl|ur_|ze_loader|mkl|igc|intelocl)',text,re.I)
            trace=out/'correctness-syscalls.txt'
            launch=['/usr/bin/strace','-f','-yy','-s','512','-e','trace=open,openat,openat2,close,close_range,mmap,ioctl,execve','-o',str(trace)]+argv()
            rows,stderr=h.run_child(record,out,'correctness-observed',launch,{**ENV,'LD_DEBUG':'libs'},wall=120)
            record['numerical_admission']=validate(rows,manifest,extents)
            text=trace.read_text();assert trace.stat().st_size <= 512<<10
            assert not re.search(r'/dev/(dri|nvidia|kfd)|lib(?:sycl|ur_|ze_loader|mkl|igc|intelocl)',text,re.I)
            objects=sorted(set(re.findall(r'calling init: (.+)',stderr)))
            assert objects and ('transferring control: '+build['binary']['path']) in stderr
            assert not any(re.search(r'lib(?:sycl|ur_|ze_loader|mkl|igc|intelocl)',p,re.I) for p in objects)
            record['runtime_audit']=dict(scope='strace from pre-exec to exit plus LD_DEBUG=libs, no scoped device/runtime observed',
                                         initialized_objects=objects,trace_sha256=sha(trace),trace_bytes=trace.stat().st_size)
            record['selected_extents_after']=extent_hashes(extents)
            assert record['selected_extents_before']==record['selected_extents_after']
            assert all(h.identity(x['path'])==x for x in model)
            check_build(build);assert sha(__file__)==record['controller_sha256']
            record.update(complete=True,passed=True,negative_admission_cases=4)
        except BaseException as e:
            record['error']=type(e).__name__+': '+str(e)
        finally:
            try:
                check_build(build);assert sha(__file__)==record['controller_sha256']
                if 'model_identity' in record: assert all(h.identity(x['path'])==x for x in record['model_identity'])
                record['exit_source_pin_gate_passed']=True
            except BaseException as e:
                record['passed']=False;record['exit_source_error']=type(e).__name__+': '+str(e)
            record.update(active=False,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            h.save(record,out)
        print(json.dumps(dict(record=str(out/'record.json'),sha256=sha(out/'record.json'),passed=record['passed'],error=record.get('error'),elapsed_seconds=record['elapsed_seconds'])))
        return 0 if record['passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
