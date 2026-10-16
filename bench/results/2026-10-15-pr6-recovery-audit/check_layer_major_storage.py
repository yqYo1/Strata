#!/usr/bin/env python3
"""Serial conversion oracle and model storage-placement gates, not speed tests."""
import array
import fcntl
import importlib.util
import json
import math
import os
import re
import struct
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, DEVICE, FAULT, sha

BUILD=D/'layer-major-storage-build-v1.json'
O=B/'main-primary-layer-major-storage-small-root-v1'


def floats(path):
    values=array.array('f');values.frombytes(path.read_bytes())
    assert all(math.isfinite(x) for x in values)
    return values


def difference(reference,candidate):
    assert len(reference)==len(candidate) and len(reference)>0
    squared=math.fsum((float(a)-float(b))**2 for a,b in zip(reference,candidate))
    return dict(elements=len(reference),max_absolute=max(abs(a-b) for a,b in zip(reference,candidate)),
        relative_L2=math.sqrt(squared/math.fsum(float(a)**2 for a in reference)),
        equal_elements=sum(a==b for a,b in zip(reference,candidate)))


def logits_difference(reference,candidate):
    r=difference(reference,candidate)
    max_r,max_c=max(reference),max(candidate)
    er=[math.exp(float(x)-max_r) for x in reference]
    ec=[math.exp(float(x)-max_c) for x in candidate]
    zr,zc=math.fsum(er),math.fsum(ec)
    lr,lc=math.log(zr),math.log(zc)
    r['KL_reference_to_candidate_nats']=math.fsum(
        (w/zr)*((float(a)-max_r-lr)-(float(b)-max_c-lc))
        for w,a,b in zip(er,reference,candidate))
    tr=sorted(range(len(reference)),key=reference.__getitem__,reverse=True)[:5]
    tc=sorted(range(len(candidate)),key=candidate.__getitem__,reverse=True)[:5]
    r.update(reference_top5=tr,candidate_top5=tc,
        reference_top1_margin=float(reference[tr[0]])-float(reference[tr[1]]),
        candidate_top1_margin=float(candidate[tc[0]])-float(candidate[tc[1]]),
        scope='One synthetic-history next-token distribution, not task quality or multi-position NLL.')
    return r


def main():
    build=json.loads(BUILD.read_text())
    old_path=D/'layer-major-small-v1.json';old=json.loads(old_path.read_text())
    prior=json.loads((D/'natural-layout-long-pair-v2.json').read_text())
    assert build['passed'] and build['complete'] and not build['active'] and build['host_contracts_passed']
    assert old['passed'] and old['complete'] and not old['active'] and not old['fresh_faults']
    assert not old['driver_dump_exists'] and old['boot_unchanged']
    assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('owner',OWNER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert all(sha(P/n)==pin for n,pin in build['source_pins'].items())
        path=Path(build['build_path']);exe=path/'strata';component=path/'sycl_prefill_postops_test'
        assert sha(exe)==build['binaries']['strata'] and sha(component)==build['binaries'][component.name]
        assert not O.exists() and not (DEVICE/'devcoredump').exists()
        assert (DEVICE/'driver').resolve().name=='xe'
        reference=old['arms'][0]
        fixtures={Path(f['path']).name:Path(f['path']) for f in reference['fixtures']}
        assert all(sha(Path(f['path']))==f['sha256'] for f in reference['fixtures'])
        reference_logits=floats(fixtures['first-logits.bin'])
        reference_residual=fixtures['prefill-R.bin'].read_bytes()
        O.mkdir(mode=0o700)
        tokens=O/'tokens.txt';tokens.write_text(','.join(str(1000+i*7%512) for i in range(513))+'\n')
        assert sha(tokens)==old['tokens_sha256']
        args=list(old['engine_args']);args[args.index('--tokens-file')+1]=str(tokens)
        env=dict(prior['environment']);env.pop('STRATA_PREFILL_ATTN_LAYOUT')
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        r=dict(active=True,passed=False,complete=False,commands=[],arms=[],engine_args=args,
            environment=env,source_pins=build['source_pins'],build_receipt_sha256=sha(BUILD),
            reference_receipt_sha256=sha(old_path),binary_sha256=sha(exe),
            component_binary_sha256=sha(component),recipe_sha256=sha(Path(__file__)),
            actual_input_tokens=513,batched_input_tokens=512,chunk=257,max_new=8,
            tokens_sha256=sha(tokens),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,
            restore_qualified=False,adopted=False,
            limits=dict(wall_seconds_per_arm=900,RSS_bytes=88 << 30,
                raw_text_bytes_per_process=64 << 20,job_bytes=512 << 20),
            fixture_consumer='Main conversion/placement comparison only; keep canonical FP32 and one FP16 set until long diagnostic is prepared, retire other successful duplicates after compact comparison.',
            fixture_byte_budget=128 << 20,fixture_review='After the small and long numerical comparisons close',
            scope='Exhaustive half-bit conversion and independent host _Float16 oracle, '
                  'then512 batched rows in257/255 chunks; unchanged FP32 and FP16 placement parity. '
                  'Synthetic small context cannot qualify model quality, performance or physical256K.')

        def persist():
            (O/'record.json').write_text(json.dumps(r,indent=2)+'\n')

        owner=module.Owner(O/'owned',persist);r['commands']=owner.commands;persist()

        def run(label,argv,runenv,wall=900):
            entry,out,err=owner.run(label,argv,runenv,P,wall=wall,
                cpu=wall*len(os.sched_getaffinity(0)),text_cap=64 << 20,
                rss_cap=88 << 30,total_cap=512 << 20,file_cap=64 << 20)
            assert module.closed(entry) and entry['exit_code']==0, (
                label,entry.get('error'),err.read_text()[-5000:],out.read_text()[-3000:])
            return out,err

        def utility(label,argv):
            out,_=run(label,argv,build['environment'],30);return out.read_text()

        def fault_check(label,cursor):
            entries=[json.loads(v) for v in utility(label,
                ['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if v.strip()]
            faults=[v for v in entries if FAULT.search(str(v.get('MESSAGE','')))]
            assert not faults and not (DEVICE/'devcoredump').exists(),faults
            return entries

        cursor=None
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',
                ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'])).group(1)
            r['journal_before_cursor']=cursor
            diagnostic=dict(env,UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:stderr',XDG_CACHE_HOME=str(O/'component-cache'))
            out,err=run('gpu-fp16-conversion-oracle',[str(component)],diagnostic,180)
            assert out.read_text()=='SYCL prefill postops: normalization, mixing, residuals, stride guards and SwiGLU passed\n'
            r['conversion_oracle']=dict(passed=True,stdout_sha256=sha(out),stderr_sha256=sha(err),
                stdout=out.read_text(),environment=diagnostic,trace_bytes=err.stat().st_size,
                half_decode_patterns=65536,pack_and_roundtrip_values=513,
                host_and_device_roundtrips=True,independent_oracle='host _Float16 RNE',
                scope='Device conversion math, existing postops and checked host-copy ownership; not model quality.')
            persist();fault_check('journal-after-conversion',cursor)
            fp16_reference=None
            for label,dtype,gpu in [('fp32-auto',None,'auto'),('fp32-host','fp32','0'),
                    ('fp16-auto','fp16','auto'),('fp16-host','fp16','0'),('fp16-mixed','fp16','257')]:
                folder=O/label;folder.mkdir()
                runenv=dict(env,XDG_CACHE_HOME=str(folder/'cache'),STRATA_PREFILL_LAYER_MAJOR='1',
                    STRATA_PREFILL_LAYER_MAJOR_R_GPU=gpu,STRATA_DUMP_FIRST_LOGITS=str(folder/'first-logits.bin'),
                    STRATA_PREFILL_DUMP_R=str(folder/'prefill-R.bin'))
                if dtype is not None:runenv['STRATA_PREFILL_LAYER_MAJOR_R_DTYPE']=dtype
                print('storage small: starting '+label,flush=True)
                out,err=run(label,[str(exe),*args],runenv)
                text=err.read_text()
                assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|aborting with temporary owners live',text)
                values=floats(folder/'first-logits.bin');assert len(values)==248320
                raw=(folder/'prefill-R.bin').read_bytes();stride=8+10240*4
                assert len(raw)==len(reference_residual)
                residual=array.array('f');ref_r=array.array('f');positions=[]
                for off in range(0,len(raw),stride):
                    position=struct.unpack_from('q',raw,off)[0];positions.append(position)
                    assert raw[off:off+8]==reference_residual[off:off+8]
                    residual.frombytes(raw[off+8:off+stride]);ref_r.frombytes(reference_residual[off+8:off+stride])
                assert positions==list(range(0,512,64)) and all(math.isfinite(v) for v in residual)
                plan=re.search(r'layer-major candidate, tokens=(\d+) chunk=(\d+) residual_dtype=(fp32|fp16) residual_element_bytes=(\d+) residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) residual_host=(\d+) residual_staging=(\d+) layer_cache=(\d+)',text)
                assert plan and int(plan[1])==512 and int(plan[2])==257
                plan_values=dict(zip(['tokens','chunk','dtype','element_bytes','total_bytes','gpu_rows','gpu_bytes','host_bytes','staging_bytes','cache_bytes'],
                    [int(v) if i!=2 else v for i,v in enumerate(plan.groups())]))
                size=2 if dtype=='fp16' else 4
                assert plan_values['element_bytes']==size and plan_values['total_bytes']==512*10240*size
                assert plan_values['gpu_bytes']+plan_values['host_bytes']==plan_values['total_bytes']
                assert plan_values['gpu_rows']==(512 if gpu=='auto' else int(gpu))
                expected_stage=0 if size==4 or gpu=='auto' else min(257,512-int(gpu))*10240*2
                assert plan_values['staging_bytes']==expected_stage
                stats=re.search(r'strata generate: prefill (\d+) tokens in (\d+) chunks,.*experts streamed (\d+)',text)
                assert stats and list(map(int,stats.groups()))==[512,2,48*512]
                result=dict(label=label,environment=runenv,residual_plan=plan_values,
                    stdout_sha256=sha(out),stderr_sha256=sha(err),
                    logits_sha256=sha(folder/'first-logits.bin'),residual_sha256=sha(folder/'prefill-R.bin'),
                    positions=positions,finite=True,logit_comparison=logits_difference(reference_logits,values),
                    residual_comparison=difference(ref_r,residual),
                    diagnostic_excerpt=[s for s in text.splitlines() if s.startswith(('strata prefill:','strata generate: prefill','strata trace: layer-major'))],
                    fixtures=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in [folder/'first-logits.bin',folder/'prefill-R.bin']])
                r['arms'].append(result);persist()
                if size==4:
                    result['FP32_reference_bitwise_equal']=result['logits_sha256']==reference['logits_sha256'] and result['residual_sha256']==reference['residual_sha256']
                    assert result['FP32_reference_bitwise_equal']
                elif fp16_reference is None:
                    fp16_reference=result
                else:
                    result['FP16_placement_bitwise_equal']=result['logits_sha256']==fp16_reference['logits_sha256'] and result['residual_sha256']==fp16_reference['residual_sha256']
                    assert result['FP16_placement_bitwise_equal']
                persist();fault_check('journal-after-'+label,cursor)
                print('storage small: closed '+label,flush=True)
            assert all(sha(P/n)==pin for n,pin in r['source_pins'].items())
            assert sha(exe)==r['binary_sha256']
            r.update(passed=True,complete=True)
        except BaseException as error:
            r['error']=repr(error);raise
        finally:
            try:
                if cursor:
                    entries=[json.loads(v) for v in utility('journal-final',
                        ['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if v.strip()]
                    r['fresh_faults']=[v for v in entries if FAULT.search(str(v.get('MESSAGE','')))]
                    r['driver_dump_exists']=(DEVICE/'devcoredump').exists()
                    r['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot_id']
                    if r['fresh_faults'] or r['driver_dump_exists'] or not r['boot_unchanged']:
                        r.update(passed=False,complete=False)
            finally:
                r['active']=False;persist()
                (D/'layer-major-storage-small-v1.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],arms=len(r['arms']),fresh_faults=len(r['fresh_faults']))),flush=True)


if __name__=='__main__':
    main()
