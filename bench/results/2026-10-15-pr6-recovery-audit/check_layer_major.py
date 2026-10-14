#!/usr/bin/env python3
"""Main serial numerical gates for the opt-in current-root scheduling candidate."""
import array
import fcntl
import importlib.util
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, DEVICE, FAULT, sha

BUILD = D/'layer-major-build-v2.json'


def main():
    mode = sys.argv[1]
    assert mode in ['small', 'long']
    build = json.loads(BUILD.read_text())
    prior = json.loads((D/'natural-layout-long-pair-v2.json').read_text())
    assert build['passed'] and build['complete'] and not build['active'] and build['host_contracts_passed']
    assert prior['passed'] and prior['complete'] and not prior['active']
    assert not prior['fresh_faults'] and prior['boot_unchanged'] and not prior['driver_dump_exists']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    O = B/('main-primary-layer-major-'+mode+'-root-v1')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if mode == 'long':
            gate = json.loads((D/'layer-major-small-v1.json').read_text())
            assert gate['passed'] and gate['complete'] and not gate['active']
            assert gate['binary_sha256'] == build['binaries']['strata'] and not gate['fresh_faults']
            assert gate['boot_unchanged'] and not gate['driver_dump_exists']
        assert not O.exists() and not (DEVICE/'devcoredump').exists()
        assert (DEVICE/'driver').resolve().name == 'xe'
        assert all(sha(P/n) == pin for n,pin in build['source_pins'].items())
        exe = Path(build['build_path'])/'strata'
        control = Path(build['control_binary']['path'])
        assert sha(exe) == build['binaries']['strata'] and sha(control) == build['control_binary']['sha256']
        O.mkdir(mode=0o700)
        env = dict(prior['environment'])
        env.pop('STRATA_PREFILL_ATTN_LAYOUT')
        args = list(prior['engine_args'])
        if mode == 'small':
            tokens = O/'tokens.txt'
            tokens.write_text(','.join(str(1000+i*7%512) for i in range(513))+'\n')
            args[args.index('--tokens-file')+1] = str(tokens)
            args[args.index('--prefill')+1] = '257'
            args[args.index('--max-new')+1] = '8'
            n = 512
            arms = [('control',control,None),('gpu-auto',exe,'auto'),('host',exe,'0')]
        else:
            tokens = Path(args[args.index('--tokens-file')+1])
            assert sha(tokens) == prior['input']['tokens_sha256']
            n = prior['input']['batched_input_tokens']
            arms = [('gpu-auto',exe,'auto')]
        r = dict(active=True,passed=False,complete=False,commands=[],mode=mode,arms=[],engine_args=args,
            environment=env,binary_sha256=sha(exe),control_binary=build['control_binary'],
            build_receipt_sha256=sha(BUILD),source_pins=build['source_pins'],
            recipe_sha256=sha(Path(__file__)),tokens_sha256=sha(tokens),actual_input_tokens=n+1,
            batched_input_tokens=n,max_new=int(args[args.index('--max-new')+1]),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,
            restore_qualified=False,adopted=False,
            limits=dict(wall_seconds_per_arm=1800,CPU_seconds_per_arm=1800*len(os.sched_getaffinity(0)),
                RSS_bytes=88 << 30,raw_text_bytes=64 << 20,job_bytes=512 << 20),
            fixture_consumer='Main first comparison of final-layer sampled residuals/logits across host/device residual paths; retire candidate tensors after comparison summaries verified',
            fixture_byte_budget=128 << 20,fixture_review='After this comparison closes',
            scope='small: synthetic513IDs,257/255chunks split a KV page, MTP4/maxnew8, auto GPU and host stores; long: exact34603-token thinking-template reference,9chunks. Numerical/lease diagnostic with validation/logging; no bench.sh speed claim or full-state quality proof.')

        def persist():
            (O/'record.json').write_text(json.dumps(r,indent=2)+'\n')

        owner = module.Owner(O/'owned',persist)
        r['commands'] = owner.commands
        persist()

        def utility(label,argv):
            entry,out,err=owner.run(label,argv,env,P,wall=30,text_cap=64 << 20,total_cap=512 << 20)
            assert module.closed(entry) and entry['exit_code']==0
            return out.read_text()

        cursor=None
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',
                ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'])).group(1)
            r['journal_before_cursor']=cursor
            for label,binary,gpu_rows in arms:
                folder=O/label;folder.mkdir()
                runenv=dict(env,XDG_CACHE_HOME=str(folder/'cache'),
                    STRATA_DUMP_FIRST_LOGITS=str(folder/'first-logits.bin'),STRATA_PREFILL_DUMP_R=str(folder/'prefill-R.bin'))
                if gpu_rows is not None:
                    runenv.update(STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU=gpu_rows)
                print('layer-major '+mode+': starting '+label,flush=True)
                entry,out,err=owner.run(label,[str(binary),*args],runenv,P,wall=1800,
                    cpu=r['limits']['CPU_seconds_per_arm'],text_cap=64 << 20,rss_cap=88 << 30,
                    total_cap=512 << 20,file_cap=64 << 20)
                assert module.closed(entry) and entry['exit_code']==0,(label,entry.get('error'),err.read_text()[-6000:])
                text=err.read_text()
                assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|aborting with temporary owners live',text)
                fs=array.array('f');fs.frombytes((folder/'first-logits.bin').read_bytes())
                assert len(fs)==248320 and all(math.isfinite(x) for x in fs)
                raw=(folder/'prefill-R.bin').read_bytes();stride=8+10240*4
                assert len(raw)%stride==0
                positions=[]
                for offset in range(0,len(raw),stride):
                    positions.append(struct.unpack_from('q',raw,offset)[0])
                    values=array.array('f');values.frombytes(raw[offset+8:offset+stride])
                    assert all(math.isfinite(v) for v in values)
                assert positions==list(range(0,n,64))
                diagnostic=[v for v in text.splitlines() if v.startswith((
                    'strata prefill:','strata generate:','strata draft kv:','strata verify:')) or 'MiB free' in v]
                result=dict(label=label,binary_sha256=sha(binary),environment=runenv,
                    diagnostic_excerpt=diagnostic,stdout_sha256=sha(out),stderr_sha256=sha(err),
                    logits_sha256=sha(folder/'first-logits.bin'),residual_sha256=sha(folder/'prefill-R.bin'),
                    residual_positions=positions,finite=True,
                    fixtures=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f))
                              for f in [folder/'first-logits.bin',folder/'prefill-R.bin']])
                if gpu_rows is not None:
                    plan=re.search(r'layer-major candidate, tokens=(\d+) chunk=(\d+) residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) residual_host=(\d+) layer_cache=(\d+)',text)
                    assert plan and int(plan[1])==n
                    result['residual_plan']=list(map(int,plan.groups()))
                    assert int(plan[3])==n*10240*4 and int(plan[5])+int(plan[6])==int(plan[3])
                    if mode=='small':
                        assert (int(plan[4])==n if gpu_rows=='auto' else int(plan[4])==0)
                    stats=re.search(r'strata generate: prefill (\d+) tokens in (\d+) chunks,.*experts streamed (\d+)',text)
                    assert stats and int(stats[1])==n and int(stats[3])==48*512
                reference=r['arms'][0] if mode=='small' and r['arms'] else prior['arms'][1] if mode=='long' else None
                if reference:
                    result['logits_bitwise_equal']=result['logits_sha256']==reference['logits_sha256']
                    result['residuals_bitwise_equal']=result['residual_sha256']==reference['residual_sha256']
                    # Persist numerical failure before its assertion.
                    r['arms'].append(result);persist()
                    assert result['logits_bitwise_equal'] and result['residuals_bitwise_equal'],result
                else:
                    r['arms'].append(result);persist()
                entries=[json.loads(v) for v in utility('journal-after-'+label,
                    ['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if v.strip()]
                assert not any(FAULT.search(str(v.get('MESSAGE',''))) for v in entries)
                assert not (DEVICE/'devcoredump').exists()
                print('layer-major '+mode+': closed '+label,flush=True)
            assert all(sha(P/n)==pin for n,pin in r['source_pins'].items())
            assert sha(exe)==r['binary_sha256']
            r.update(passed=True,complete=True)
        except BaseException as error:
            r['error']=repr(error)
            raise
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
                (D/('layer-major-'+mode+'-v1.json')).write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],mode=mode,arms=len(r['arms']),fresh_faults=len(r['fresh_faults']))),flush=True)


if __name__=='__main__':
    main()
