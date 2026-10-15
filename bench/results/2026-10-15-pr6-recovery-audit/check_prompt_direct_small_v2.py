#!/usr/bin/env python3
"""Main first-use diagnostic of existing int8 direct prompt output; never a speed sample."""
import fcntl
import importlib.util
import json
import os
import re
import struct
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, DEVICE, FAULT, sha


def main():
    build_path = D/'kv-copy-groups-build-v2.json'
    ref_path = D/'selected-prefill-small-v1.json'
    ids_path = D/'phase-audit-small-v1.json'
    build, ref, id_ref = [json.loads(p.read_text()) for p in [build_path, ref_path, ids_path]]
    assert all(d['passed'] and d['complete'] and not d['active'] for d in [build, ref, id_ref])
    assert not id_ref['fresh_faults'] and id_ref['boot_unchanged'] and not id_ref['driver_dump_exists']
    control = next(a for a in ref['arms'] if a['dtype'] == 'fp32')
    expected_ids = next(a['generated_ids'] for a in id_ref['arms'] if not a['audit_on'] and not a['phase_on'])
    exe = Path(build['build_path'])/'strata'
    assert sha(exe) == build['binaries'][exe.name]
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    args = ref['engine_args']; tokens = Path(args[args.index('--tokens-file')+1])
    assert sha(tokens) == ref['tokens_sha256']
    assert len(tokens.read_text().strip().split(',')) == 513
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        health=json.loads((D/'prompt-direct-small-faultcheck-v1.json').read_text())
        assert health['passed'] and health['complete'] and not health['active'] and not health['fresh_faults'] and not health['driver_dump_exists'] and health['boot_unchanged']
        assert all(sha(P/n) == h for n,h in build['source_pins'].items())
        assert (DEVICE/'driver').resolve().name == 'xe' and not (DEVICE/'devcoredump').exists()
        output = B/'main-primary-prompt-direct-small-root-v2'
        assert not output.exists(); output.mkdir(mode=0o700)
        first, selected = output/'first-logits.bin', output/'selected-logits.bin'
        seed = json.loads((D/'layer-major-storage-small-v1.json').read_text())
        env = dict(seed['environment'], XDG_CACHE_HOME=str(output/'cache'),
            STRATA_SYCL_PROMPT_DIRECT='1', STRATA_PREFILL_LAYER_MAJOR='1',
            STRATA_PREFILL_LAYER_MAJOR_R_DTYPE='fp32', STRATA_PREFILL_LAYER_MAJOR_R_GPU='auto',
            STRATA_PREFILL_RELEASE_MTP_EXPERTS='1', STRATA_PREFILL_PHASE_AUDIT='0',
            STRATA_PREFILL_BOUNDARY_AUDIT='0', STRATA_SYCL_KV_COPY_GROUPS='0',
            STRATA_PREFILL_LOGITS=str(selected), STRATA_PREFILL_LOGITS_POSITIONS=','.join(map(str,ref['positions'])),
            STRATA_DUMP_FIRST_LOGITS=str(first), UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION')
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        for k,v in {'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD':'1','NEOReadDebugKeys':'1','EnableDirectSubmission':'0',
                'STRATA_SYCL_PREFILL_SAME_QUEUE':'1','SYCL_CACHE_PERSISTENT':'0','STRATA_SYCL_VERIFY_NATIVE_CAPTURE':'0'}.items():
            assert env[k] == v
        record = dict(active=True, passed=False, complete=False, commands=[], environment=env, argv=[str(exe),*args],
            binary_sha256=sha(exe), source_pins=build['source_pins'], recipe_sha256=sha(Path(__file__)),
            build_receipt_sha256=sha(build_path), reference_sha256=sha(ref_path), output_reference_sha256=sha(ids_path),
            input_sha256=sha(tokens), actual_input_tokens=513, GPU_executed=True, adopted=False,
            performance_qualified=False, model_performance_qualified=False, quality_qualified=False,
            physical_262144_qualified=False, restore_qualified=False,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            limits=dict(wall_seconds=1200,RSS_bytes=88<<30,text_bytes=64<<20,job_bytes=256<<20),
            retention_consumer='Verify existing Direct output with exact selected-head/first-logit hashes and8IDs, then retire successful raw trace and duplicate tensors.',
            scope='Small first-use numerical/lifetime diagnostic only. Warning/error and parameter validation logs; full API trace exceeded the prior log budget. No short-input performance comparison.')
        def persist(): (output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(output/'owned',persist);record['commands']=owner.commands;persist()
        def utility(label,argv):
            e,o,s=owner.run(label,argv,build['environment'],P,wall=30,total_cap=256<<20)
            assert module.closed(e) and e['exit_code']==0,(label,e.get('error'));return o.read_text()
        cursor=None
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            record['journal_before_cursor']=cursor
            print('Existing int8 prompt Direct:513-token numerical/lifetime diagnostic',flush=True)
            e,out,err=owner.run('direct-small',[str(exe),*args],env,P,wall=1200,cpu=1200*len(os.sched_getaffinity(0)),
                rss_cap=88<<30,text_cap=64<<20,total_cap=256<<20,file_cap=64<<20)
            assert module.closed(e) and e['exit_code']==0,(e.get('error'),err.read_text()[-3000:])
            text=err.read_text()
            assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed|stopping with owners live',text)
            ids=re.findall(r'^output\s*:\s*([\d ]+)\s*$',out.read_text(),re.M)
            assert len(ids)==1
            got=[int(s) for s in ids[0].split()]
            meta=Path(str(selected)+'.meta');partial=Path(str(meta)+'.partial')
            raw=selected.read_bytes();vocab,count=struct.unpack_from('ii',raw)
            assert (vocab,count)==(248320,len(ref['positions'])) and len(raw)==8+vocab*count*4
            record['comparison']=dict(first_logits_bitwise=sha(first)==control['first_logits_sha256'],
                selected_logits_bitwise=sha(selected)==control['selected_logits_sha256'],
                metadata_equal=sha(meta)==sha(partial)==control['metadata_sha256'], generated_ids_equal=got==expected_ids,
                generated_ids=got, first_logits_sha256=sha(first), selected_logits_sha256=sha(selected), metadata_sha256=sha(meta))
            assert all(record['comparison'][k] for k in ['first_logits_bitwise','selected_logits_bitwise','metadata_equal','generated_ids_equal'])
            record['diagnostics']=[s for s in text.splitlines() if s.startswith(('strata prefill:','strata mtp:','strata trace: layer-major'))]
            record['trace']=dict(bytes=err.stat().st_size,sha256=sha(err),kernel_launch_mentions=text.count('urEnqueueKernelLaunch'),
                direct_kernel_mentions=text.count('launchILi1ELb1ELb0EE'),stdout_sha256=sha(out))
            record['fixtures']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in [first,selected,meta,partial]]
            assert all(sha(P/n)==h for n,h in build['source_pins'].items())
            record.update(passed=True,complete=True)
        except BaseException as error:
            record['error']=repr(error);raise
        finally:
            try:
                if cursor:
                    entries=[json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record['fresh_faults']=[s for s in entries if FAULT.search(str(s.get('MESSAGE','')))]
                    record['driver_dump_exists']=(DEVICE/'devcoredump').exists()
                    record['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:record.update(passed=False,complete=False)
            finally:
                record['active']=False;persist();(D/'prompt-direct-small-v2.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(passed=record['passed'],fresh_faults=len(record['fresh_faults']))),flush=True)


if __name__=='__main__':main()
