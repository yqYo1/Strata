#!/usr/bin/env python3
"""Bounded paired selected-head distributions; diagnostics, never speed samples."""
import fcntl
import importlib.util
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from check_gemm_orientation import P,B,D,OWNER,DEVICE,FAULT,sha
from check_layer_major_storage import floats,logits_difference

BUILD=D/'current-diagnostics-build-v1.json'


def main():
    mode=sys.argv[1];assert mode in ['small','long']
    build=json.loads(BUILD.read_text());assert build['passed'] and build['complete'] and not build['active']
    storage=json.loads((D/'layer-major-storage-small-v1.json').read_text())
    natural=json.loads((D/'natural-layout-long-pair-v2.json').read_text())
    assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('owner',OWNER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    O=B/('main-primary-selected-prefill-'+mode+'-root-v1')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert storage['passed'] and storage['complete'] and not storage['active'] and not storage['fresh_faults']
        if mode=='long':
            gate=json.loads((D/'selected-prefill-small-v1.json').read_text())
            assert gate['passed'] and gate['complete'] and not gate['active'] and not gate['fresh_faults']
            assert gate['binary_sha256']==build['binaries']['strata']
        assert all(sha(P/n)==pin for n,pin in build['source_pins'].items())
        exe=Path(build['build_path'])/'strata';assert sha(exe)==build['binaries']['strata']
        assert not O.exists() and not (DEVICE/'devcoredump').exists()
        assert (DEVICE/'driver').resolve().name=='xe'
        prior=storage if mode=='small' else natural
        args=list(prior['engine_args']);tokens=Path(args[args.index('--tokens-file')+1])
        expected_tokens=prior['tokens_sha256'] if mode=='small' else natural['input']['tokens_sha256']
        assert sha(tokens)==expected_tokens
        ids=[int(v) for v in tokens.read_text().strip().split(',')]
        n=len(ids)-1
        positions=[0,1,63,64,255,256,257,510,511] if mode=='small' else [
            0,1,63,64,255,256,257,4095,4096,8191,8192,16383,16384,24575,24576,32767,32768,34600,34601]
        assert positions==sorted(set(positions)) and positions[-1]<n
        env=dict(storage['environment']);assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        O.mkdir(mode=0o700)
        r=dict(active=True,passed=False,complete=False,mode=mode,commands=[],arms=[],
            environment=env,engine_args=args,build_receipt_sha256=sha(BUILD),
            binary_sha256=sha(exe),source_pins=build['source_pins'],recipe_sha256=sha(Path(__file__)),
            tokens_sha256=sha(tokens),actual_input_tokens=len(ids),batched_input_tokens=n,positions=positions,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            quality_qualified=False,performance_qualified=False,physical_262144_qualified=False,
            restore_qualified=False,adopted=False,
            limits=dict(wall_seconds_per_arm=1800,RSS_bytes=88 << 30,text_bytes_per_process=64 << 20,job_bytes=512 << 20),
            fixture_consumer='Main paired selected-position distribution comparison; retire raw rows after full comparator and canonical reference review.',
            fixture_byte_budget=128 << 20,fixture_review='After this comparison and any identified divergent-position investigation close',
            scope='small: synthetic513 input IDs,9 selected predictions and old end-logit regression; '
                  'long: exact34603 thinking-template history,19 selected predictions. '
                  'Target NLL/KL are position samples, not full-token PPL, independent task quality or speed.')
        def persist():(O/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=module.Owner(O/'owned',persist);r['commands']=owner.commands;persist()
        def utility(label,argv):
            entry,out,err=owner.run(label,argv,build['environment'],P,wall=30,text_cap=64 << 20,total_cap=512 << 20)
            assert module.closed(entry) and entry['exit_code']==0
            return out.read_text()
        def journals(label,cursor):
            return [json.loads(v) for v in utility(label,['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if v.strip()]
        cursor=None
        rows_by_arm=[]
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',
                ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'])).group(1)
            r['journal_before_cursor']=cursor
            for dtype in ['fp32','fp16']:
                folder=O/dtype;folder.mkdir();selected=folder/'selected-logits.bin'
                runenv=dict(env,STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_LAYER_MAJOR_R_DTYPE=dtype,
                    STRATA_PREFILL_LAYER_MAJOR_R_GPU='auto',STRATA_PREFILL_LOGITS=str(selected),
                    STRATA_PREFILL_LOGITS_POSITIONS=','.join(map(str,positions)),
                    STRATA_DUMP_FIRST_LOGITS=str(folder/'first-logits.bin'),XDG_CACHE_HOME=str(folder/'cache'))
                print('selected '+mode+': starting '+dtype,flush=True)
                entry,out,err=owner.run(dtype,[str(exe),*args],runenv,P,wall=1800,
                    cpu=1800*len(os.sched_getaffinity(0)),text_cap=64 << 20,rss_cap=88 << 30,
                    total_cap=512 << 20,file_cap=64 << 20)
                assert module.closed(entry) and entry['exit_code']==0,(dtype,entry.get('error'),err.read_text()[-5000:])
                text=err.read_text()
                assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed|aborting with',text)
                raw=selected.read_bytes();vocab,count=struct.unpack_from('ii',raw)
                assert vocab==248320 and count==len(positions) and len(raw)==8+vocab*count*4
                meta=Path(str(selected)+'.meta');partial=Path(str(meta)+'.partial')
                assert meta.is_file() and sha(meta)==sha(partial)
                lines=meta.read_text().splitlines()
                descriptors=[s.split('\t') for s in lines if re.match(r'^\d+\t',s)]
                assert descriptors==[[str(i),str(p),str(ids[p+1])] for i,p in enumerate(positions)]
                assert lines[-1]=='complete\t'+str(count)+'\t'+str(len(raw))
                assert 'head=native' in lines and 'vocab=248320' in lines
                first=floats(folder/'first-logits.bin');assert len(first)==vocab
                if mode=='small':expected=storage['arms'][0 if dtype=='fp32' else 2]['logits_sha256']
                else:expected=natural['arms'][1]['logits_sha256'] if dtype=='fp32' else None
                result=dict(dtype=dtype,environment=runenv,finite=True,
                    first_logits_sha256=sha(folder/'first-logits.bin'),
                    selected_logits_sha256=sha(selected),metadata_sha256=sha(meta),
                    stdout_sha256=sha(out),stderr_sha256=sha(err),
                    diagnostic_excerpt=[s for s in text.splitlines() if s.startswith(('strata prefill:','strata generate: prefill'))],
                    fixtures=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in [selected,meta,partial,folder/'first-logits.bin']])
                r['arms'].append(result);persist()
                if expected is not None:
                    result['end_logit_reference_bitwise_equal']=result['first_logits_sha256']==expected
                    assert result['end_logit_reference_bitwise_equal']
                import array
                rows=[]
                for i in range(count):
                    values=array.array('f');values.frombytes(raw[8+i*vocab*4:8+(i+1)*vocab*4])
                    assert all(math.isfinite(v) for v in values)
                    rows.append(values)
                rows_by_arm.append(rows)
                faults=[v for v in journals('journal-after-'+dtype,cursor) if FAULT.search(str(v.get('MESSAGE','')))]
                assert not faults and not (DEVICE/'devcoredump').exists()
                persist();print('selected '+mode+': closed '+dtype,flush=True)
            r['position_comparisons']=[]
            for p,reference,candidate in zip(positions,*rows_by_arm):
                comparison=logits_difference(reference,candidate);target=ids[p+1]
                def nll(values):
                    maximum=max(values)
                    return maximum+math.log(math.fsum(math.exp(float(v)-maximum) for v in values))-float(values[target])
                comparison.update(position=p,target_token_id=target,reference_target_NLL_nats=nll(reference),
                    candidate_target_NLL_nats=nll(candidate),
                    scope='Selected teacher-forced prediction at a registered '+mode+' history position; not independent task quality or full-token PPL.')
                comparison['target_NLL_delta_nats']=comparison['candidate_target_NLL_nats']-comparison['reference_target_NLL_nats']
                r['position_comparisons'].append(comparison)
            assert all(sha(P/n)==pin for n,pin in r['source_pins'].items())
            assert sha(exe)==r['binary_sha256'];r.update(passed=True,complete=True)
        except BaseException as error:
            r['error']=repr(error);raise
        finally:
            try:
                if cursor:
                    r['fresh_faults']=[v for v in journals('journal-final',cursor) if FAULT.search(str(v.get('MESSAGE','')))]
                    r['driver_dump_exists']=(DEVICE/'devcoredump').exists()
                    r['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot_id']
                    if r['fresh_faults'] or r['driver_dump_exists'] or not r['boot_unchanged']:r.update(passed=False,complete=False)
            finally:
                r['active']=False;persist();(D/('selected-prefill-'+mode+'-v1.json')).write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],mode=mode,positions=len(positions),fresh_faults=len(r['fresh_faults']))),flush=True)


if __name__=='__main__':main()
