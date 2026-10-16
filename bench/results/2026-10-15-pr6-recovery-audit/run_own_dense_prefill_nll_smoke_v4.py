#!/usr/bin/env python3
"""Main model hook gate: default-OFF versus NLL with exact selected-logit oracle."""
import array
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha,boot_id
FAULT=re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
DEVICE=Path('/sys/bus/pci/devices/0000:05:00.0')


def main():
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        build=json.loads((A/'own-dense-prefill-nll-build-v3.json').read_text())
        require(build['passed'] and build['complete'] and not build['active'] and build['host_fixture_stdout']=='dense_prefill_nll_host PASS\n','host/build gate')
        for c in build['commands']:
            require(module.closed(c) and c['exit_code']==0,'build owners retired')
            for old in c['owners']:
                now=module.identity(old['pid']);require(now is None or now['start_ticks']!=old['start_ticks'],'build owner alive')
        require(all(sha(P/n)==h for n,h in build['source_pins'].items()),'same reviewed source')
        failed=json.loads((A/'own-dense-prefill-nll-smoke-v3.json').read_text())
        require(not failed['active'] and not failed['fresh_faults'] and failed['boot_unchanged'] and not failed['driver_dump_exists'], 'preceding configuration failure closedhealthy')
        for c in failed['commands']:
            require(module.closed(c), 'preceding failed command fully retired')
            for old in c['owners']:
                now=module.identity(old['pid']);require(now is None or now['start_ticks']!=old['start_ticks'], 'preceding owner alive')
        exe=Path(build['binary']);require(sha(exe)==build['binary_sha256'],'new NLL binary pin')
        prior=json.loads((A/'selective-iq-clean-bench-full-0-v1.json').read_text())
        libraries=json.loads((A/'own-qsa-select-model-a1-library-hashes-v1.json').read_text())
        require(all(sha(Path(v['path']))==v['sha256'] for v in libraries['files']),'qualified runtime files stable')
        for field in ('pack_metadata_sha256','MTP_metadata_sha256'):
            require(all(sha(Path(n))==h for n,h in prior[field].items()),'model metadata stable')
        for n,v in prior['model_identity'].items():
            st=Path(n).stat();require(st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],'model file identity')
        root=B/'main-own-dense-prefill-nll-smoke-root-v4';target=A/'own-dense-prefill-nll-smoke-v4.json'
        require(not root.exists() and not target.exists(),'fresh paired hook gate');root.mkdir(mode=0o700)
        tokens=list(range(129));positions=[0,63,127];inp=root/'tokens.txt';inp.write_text(','.join(map(str,tokens))+'\n')
        args=list(prior['engine_args'])
        for flag,value in {'--max-context':'4096','--kv-resident':'4096','--prefill':'64'}.items():args[args.index(flag)+1]=value
        args += ['--tokens-file',str(inp),'--max-new','1','--stream-token','--no-token-graph']
        env=dict(prior['environment'],XDG_CACHE_HOME=str(root/'cache'))
        env.update(STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='64',STRATA_PREFILL_LAYER_MAJOR_R_DTYPE='fp32',
            STRATA_PREFILL_RELEASE_HEADS='0',STRATA_PREFILL_NATIVE_DENSE_BANK='1',STRATA_PREFILL_CANONICAL_DENSE_BANK='1',
            STRATA_SYCL_PROMPT_DIRECT='1',STRATA_SYCL_PREFILL_SCRATCH_ALIAS='1',STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH='1',
            STRATA_SYCL_SELECT_RADIX4='1',STRATA_SYCL_SELECT_CYCLIC='0',STRATA_SYCL_PROMPT_K_CELL_MAJOR='0',
            STRATA_PREFILL_BF16X2='0',STRATA_PF_FUSED='0',STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='0',STRATA_PREFILL_CPU_WAIT_PROFILE='0',
            STRATA_TRACE='1',UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0',STRATA_SYCL_PREFILL_SAME_QUEUE='1',SYCL_CACHE_PERSISTENT='0',STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0',ONEAPI_DEVICE_SELECTOR='level_zero:gpu').items():require(env[k]==v,'same safety flags')
        record=dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],cases={},boot_id=boot_id(),
            recipe_sha256=sha(Path(__file__)),binary_sha256=sha(exe),source_pins=build['source_pins'],input_sha256=sha(inp),tokens=tokens,positions=positions,args=args,environment=env,
            model_identity=prior['model_identity'],pack_metadata_sha256=prior['pack_metadata_sha256'],runtime_files=libraries['files'],
            prior_configuration_failure_receipt='own-dense-prefill-nll-smoke-v3.json',prior_configuration_failure_sha256=sha(A/'own-dense-prefill-nll-smoke-v3.json'),scope='Synthetic129-input/MTP4 mechanics; two64-row layermajor chunks, FP32/mixedGPU64-host64 placement; observer transparency and three independent selectedrow NLL checks. NOT naturalquality/performance/fullcontext/RESTORE.',raw_tensor_byte_budget=12<<20,performance_qualified=False,FP16_quality_qualified=False,physical_262144_qualified=False)
        def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
        hostenv=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        def utility(label,argv):
            c,out,err=owner.run(label,argv,hostenv,P,wall=60,cpu=60,rss_cap=2<<30,text_cap=8<<20,total_cap=1<<30,file_cap=8<<20)
            require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-3000:]);return out.read_text()
        cursor=None
        try:
            require((DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists(),'GPU preflight')
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            selected={}
            for arm in ('off','on'):
                runroot=root/arm;runroot.mkdir()
                current=dict(env,STRATA_PREFILL_LOGITS=str(runroot/'selected.bin'),STRATA_PREFILL_LOGITS_POSITIONS=','.join(map(str,positions)))
                if arm=='on':current['STRATA_PREFILL_NLL']=str(runroot/'nll.bin')
                recipe=runroot/'recipe.json';recipe.write_text(json.dumps(dict(arm=arm,args=args,environment=current,source_pins=build['source_pins'],binary_sha256=sha(exe),input_sha256=sha(inp)),indent=2)+'\n')
                if arm=='on':current['STRATA_PREFILL_NLL_RECIPE_SHA256']=sha(recipe)
                record['GPU_executed']=True
                c,out,err=owner.run('native-'+arm,[str(exe),*args],current,P,wall=900,cpu=18000,rss_cap=88<<30,text_cap=64<<20,total_cap=1<<30,file_cap=64<<20)
                require(module.closed(c) and c['exit_code']==0,'native-'+arm+': '+err.read_text()[-6000:])
                text=err.read_text();require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:',text),'native errors')
                require(list(map(int,re.findall(r'layer-major layer (\d+) of 48: completed 2 chunks',text)))==list(range(48)),'all48 layers2chunks')
                data=(runroot/'selected.bin').read_bytes();meta=(runroot/'selected.bin.meta').read_text()
                vocab,count=struct.unpack_from('=ii',data);require(vocab==248320 and count==3 and len(data)==8+3*vocab*4,'selected shape')
                require(all(f'{i}\t{pos}\t{tokens[pos+1]}\n' in meta for i,pos in enumerate(positions)) and f'complete\t3\t{len(data)}\n' in meta,'exact selected target metadata')
                selected[arm]=data
                record['cases'][arm]=dict(recipe_sha256=sha(recipe),environment=current,selected_sha256=sha(runroot/'selected.bin'),selected_meta=meta,
                    native_stderr_sha256=sha(err),normal_closed=True,progress_layers=48,chunks_per_layer=2,
                    placement=[line for line in text.splitlines() if line.startswith('strata prefill: layer-major candidate,')])
                if arm=='off':require('strata dense prefill NLL:' not in text and not (runroot/'nll.bin').exists(),'default off hook absent')
                else:
                    require('strata dense prefill NLL: rows=128 ' in text,'all initial rows diagnosed')
                    raw=(runroot/'nll.bin').read_bytes();meta=(runroot/'nll.bin.meta').read_text()
                    require(raw[:8]==b'STNLLV2\n' and struct.unpack_from('<8I',raw,8)[:6]==(2,64,129,248320,128,0) and raw[40:64]==bytes(24) and len(raw)==64+12*128,'NLL exact header/coverage')
                    losses=list(struct.iter_unpack('<Id',raw[64:]));require(all(target==tokens[i+1] and math.isfinite(loss) and loss>=0 for i,(target,loss) in enumerate(losses)),'all targets/finite losses')
                    require('recipe_sha256='+sha(recipe)+'\n' in meta and 'complete=1\n' in meta and 'rows=128\n' in meta,'NLL completed metadata')
                    checks=[]
                    for i,pos in enumerate(positions):
                        values=array.array('f');values.frombytes(data[8+i*248320*4:8+(i+1)*248320*4])
                        require(all(math.isfinite(v) for v in values),'full selected row finite')
                        maximum=max(values);expected=(maximum-values[tokens[pos+1]])+math.log(math.fsum(math.exp(float(v)-maximum) for v in values))
                        require(abs(expected-losses[pos][1])<=1e-10,'independent fullvocabulary fsum NLL oracle')
                        checks.append(dict(position=pos,target=tokens[pos+1],NLL=losses[pos][1],independent_NLL=expected,absolute_difference=abs(expected-losses[pos][1])))
                    record['cases'][arm].update(nll_sha256=sha(runroot/'nll.bin'),nll_metadata=meta,all128_target_NLLs=[dict(target=t,loss=v) for t,v in losses],independent_selected_checks=checks)
                persist()
            require(selected['off']==selected['on'],'observer transparency: all3 fullvocabulary rows byteexact')
            require(all(sha(P/n)==h for n,h in build['source_pins'].items()),'frozen source throughout')
            record.update(passed=True,complete=True,all_three_fullvocabulary_rows_byteexact=True)
        except BaseException as error:record['error']=repr(error);raise
        finally:
            try:
                if cursor and owner.active is None:
                    rows=[json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in rows if FAULT.search(str(v.get('MESSAGE','')))],boot_unchanged=boot_id()==record['boot_id'],driver_dump_exists=(DEVICE/'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None,finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):record.update(passed=False,complete=False)
                persist()
                with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],cases=list(record['cases']),selected_byteexact=record['all_three_fullvocabulary_rows_byteexact'],checks=record['cases']['on']['independent_selected_checks'])),flush=True)


if __name__=='__main__':main()
