#!/usr/bin/env python3
"""Bounded installed llama.cpp control; Main authored, repository LICENSE.

This observes an external installation, without rebuilding or relabelling it as
an exactly recovered historical build. Source/recipe limitations stay explicit.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import runtime_probe as runtime

P, B = runtime.P, runtime.B
ROOT = B / 'llama9b-independent-control-v3'
RESULT = runtime.RESULT.with_name('prefill-resources-current.json')
KEY = 'independent_llama9b_control_v3'
LLAMA = Path('/home/yayoi/ghq/github.com/ggml-org/llama.cpp')
EXE = LLAMA / 'build/bin/llama-bench'
MODEL = Path('/data/llm_models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf')
SDK = Path('/opt/intel/oneapi/compiler/2026.1')
VENDOR_DIRS = [SDK / 'lib', Path('/opt/intel/oneapi/mkl/2026.1/lib'),
    Path('/opt/intel/oneapi/dnnl/2026.1/lib'), Path('/opt/intel/oneapi/tbb/2023.1/lib'),
    Path('/opt/intel/oneapi/umf/1.1/lib')]
INPUTS = ['tools/sycl/reference_llama.py', 'tools/sycl/runtime_probe.py',
    'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_guards.py',
    'tools/sycl/ur_validation.py', 'tools/sycl/build_engine.py',
    'tools/sycl/check_full_context_source.py', 'tools/sycl/measurement_environment.json']
PACKAGES = ['intel-oneapi-compiler-dpcpp-cpp-runtime-2026.1', 'intel-oneapi-dnnl-2026.1',
    'intel-oneapi-mkl-core-2026.1', 'intel-oneapi-tbb-2023.1', 'intel-oneapi-umf-1.1']
require, digest = runtime.require, runtime.digest


def write(path, data):
    text=json.dumps(data,indent=2)+'\n'
    samples=data.get('event_reconciliation',{}).get('fused_command_profile',{}).get(
        'prefill_upload_attribution_v3',{}).get('samples',{})
    # Preserve the existing dense numerical representation during unrelated runs.
    for rows in (samples.get('rows',[]),samples.get('profiles',{}).get('rows',[])):
        if not rows:
            continue
        lines=json.dumps(rows,indent=2).splitlines()
        for match in re.finditer(r'(?m)^( +)"rows": \[\n',text):
            indent=match.group(1);start=match.end()-2
            old='\n'.join(indent+line if i else line for i,line in enumerate(lines))
            if text[start:start+len(old)]==old:
                new='[\n'+''.join(indent+'  '+json.dumps(row,separators=(',',':'))+
                    (',' if i<len(rows)-1 else '')+'\n' for i,row in enumerate(rows))+indent+']'
                text=text[:start]+new+text[start+len(old):]
                break
    require(json.loads(text)==data,'result rendering lost information')
    path.write_text(text)


def pins(revision):
    out = {}
    for name, raw in runtime.build.blobs(P, revision, INPUTS):
        require(raw == (P / name).read_bytes(), 'commit exact controller input: ' + name)
        out[name] = hashlib.sha256(raw).hexdigest()
    return out


def external_source():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=LLAMA, text=True).strip()
    require(revision == 'abeada335e2e78bd3fe63febafab7e900ce75810', 'user reference revision differs')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
        cwd=LLAMA), 'installed reference tracked source changed')
    names = ['build/common/build-info.cpp', 'build/CMakeCache.txt',
        'build/CMakeFiles/3.28.3/CMakeCXXCompiler.cmake',
        'build/ggml/src/ggml-sycl/CMakeFiles/ggml-sycl.dir/flags.make']
    return dict(revision=revision, tracked_tree_clean=True,
        build_metadata_sha256={name:digest(LLAMA / name) for name in names},
        limit='Installed external reference: embedded11455/abeada335 and current clean source/build metadata agree. Complete build-time input receipt and historical binary hash are unavailable;do not claim exact historical binary/source reproducibility.')


def prepare():
    c = runtime.validation.committed(RESULT)
    require(KEY not in c and not ROOT.exists(), 'fresh single-use control; inspect existing ownership first')
    runtime.guards.no_model_owners([str(MODEL)])
    require(c['event_reconciliation']['fused_command_profile']['prefill_upload_attribution_v3']['retirement']['complete'],
        'close and retire prior component first')
    previous=c['independent_llama9b_control_v1']
    require(previous['stage']=='failed' and not previous['active'] and not previous['GPU_executed'] and
        previous['actual_ownership_reconciliation']['anchored_session_empty'] and
        previous['actual_ownership_reconciliation']['native_direct_child_reaped'],
        'preserve failed host observer and positively close its owned process first')
    previous=c['independent_llama9b_control_v2']
    require(previous['stage']=='failed' and not previous['active'] and previous['fresh_faults']==[] and
        previous['actual_ownership_reconciliation']['all3direct_children_reaped'] and
        previous['actual_ownership_reconciliation']['all3anchored_sessions_empty'],
        'preserve dynamic ICD observer refusal and close owned work first')
    revision = runtime.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    inputs = pins(revision)
    original = runtime.selected.baseline()[1]
    packages = subprocess.check_output(['dpkg-query','-W','-f=${Package} ${Version}\n',*PACKAGES],
        text=True,timeout=15).splitlines()
    verified = subprocess.run(['dpkg','--verify',*PACKAGES],capture_output=True,text=True,timeout=60)
    require(verified.returncode==0 and not verified.stdout and not verified.stderr,'official runtime package verification failed')
    preload = runtime.components()
    preload['loader'] = str((SDK / 'lib/libur_loader.so.0').resolve())
    files = {p:h for p,h in runtime.runtime_files(preload).items() if '.tools/intel-llvm-v711/' not in p}
    # Pin installed official vendor candidates before launch, including lazy MKL
    # providers. Unexpected executable mappings fail admission, never learn/retry.
    for directory in VENDOR_DIRS:
        require(directory.is_dir(), 'installed vendor directory absent: ' + str(directory))
        for path in directory.glob('*.so*'):
            if not path.is_file():
                continue
            with path.open('rb') as stream:
                elf = stream.read(4) == b'\x7fELF'
            if elf:
                files[str(path.resolve())] = digest(path)
    files[str(EXE.resolve())] = digest(EXE)
    files[str(Path(preload['canary']['path']).resolve())] = preload['canary']['sha256']
    icd=Path('/etc/OpenCL/vendors/intel64.icd')
    icd_provider=Path(icd.read_text().strip()).resolve()
    for path in [icd_provider,Path('/usr/lib/x86_64-linux-gnu/libOpenCL.so.1').resolve()]:
        files[str(path)]=digest(path)
    require(b'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD' in
        (SDK / 'lib/libur_adapter_level_zero_v2.so.0.12.0').read_bytes(), 'copy-offload avoidance unavailable')
    env = dict(runtime.validation.committed(runtime.SETTINGS)['environment'])
    env.update(LD_LIBRARY_PATH=':'.join(map(str,VENDOR_DIRS)),
        UR_ADAPTERS_FORCE_LOAD=str(SDK / 'lib/libur_adapter_level_zero_v2.so.0.12.0'),
        LD_PRELOAD=preload['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=preload['loader'],
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),OCL_ICD_VENDORS=str(icd))
    # Resolve the ELF interpreter's DT_NEEDED closure before admitting a child.
    # --list does not enter the target's constructors/main or initialize a GPU.
    pending={str(EXE.resolve()),str(Path(preload['library']['path']).resolve()),
        str(Path(preload['canary']['path']).resolve()),str(Path(env['UR_ADAPTERS_FORCE_LOAD']).resolve()),
        '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39758',str(icd_provider),
        str(Path('/usr/lib/x86_64-linux-gnu/libOpenCL.so.1').resolve())}
    listed=set()
    while pending:
        path=pending.pop()
        if path in listed:
            continue
        text=subprocess.check_output(['/lib64/ld-linux-x86-64.so.2','--list',path],
            env={'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH']},text=True,stderr=subprocess.STDOUT,timeout=15)
        require('not found' not in text,'unresolved ELF dependency: '+path)
        listed.add(path)
        for name in re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(',text,re.M):
            dependency=str(Path(name).resolve())
            if dependency not in files:
                files[dependency]=digest(dependency);pending.add(dependency)
    root_source = external_source()
    asset = dict(path=str(MODEL), bytes=MODEL.stat().st_size, sha256=digest(MODEL))
    args = ['-fa','1','-n','0','-ctk','q8_0','-ctv','q8_0','-m',str(MODEL),
        '-o','jsonl','--progress']
    row = dict(stage='prepared', source_commit=revision, input_sha256=inputs,
        external_source=root_source, binary=dict(path=str(EXE),sha256=digest(EXE),bytes=EXE.stat().st_size),
        model=asset, runtime_files=files, preload=preload, environment=env,packages=packages,
        external_config_sha256={str(icd):digest(icd)},
        ICD_selection={'scope':'This child only;no system/user configuration change.',
            'reference':'https://github.com/OCL-dev/ocl-icd/blob/master/doc/libOpenCL.7.txt.in',
            'local_document_sha256':digest('/usr/share/doc/ocl-icd-libopencl1/html/libOpenCL.html'),
            'reason':'The installed ocl-icd loader documents .icd-file override. Limit incidental oneDNN/OpenCL platform enumeration to Intel GPU;do not load NVIDIA or CPU OpenCL ICDs. This differs from unrecorded historical defaults.'},
        selected_original=original, boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        driver=runtime.guards.driver_guard(), active=False,ownership_closed=True,passed=False,
        GPU_executed=False,model_executed=False,model_adoption_qualified=False,
        cases=[dict(label='canary',argv=[preload['canary']['path']]),
            dict(label='probe2048',argv=[str(EXE),*args,'-p','2048','-r','1']),
            dict(label='pp131072',argv=[str(EXE),*args,'-p','131072','-r','3'])],
        criteria='Ordered host canary,one2048-token traced model check exercising512ubatches,then one independent131072-token process with full131072 warmup+3timed repetitions. All normal/reaped/empty-session,no fresh fault,same boot/driver/inputs/models/providers,pre-main parameter validation active,silent validation log,actual official SYCL/LevelZero/NEO maps. Parse exactly one JSON test with n_prompt,zero n_gen,FAon,K/Vq8_0,all GPU layers,SYCL backend,positive finite times. Stop first failure;no retries. Three within-process samples do not measure between-process variability.',
        question='Does a separately installed dense9B SYCL stack still reach the user-reported966.57±10.59tok/s while Strata same-binary historical1002.6507 now measures911.7371? Descriptive discrimination,not a causal model/source comparison or an architecture ceiling.',
        historical=dict(user_reported_tok_s=966.57,user_reported_stdev=10.59,build='abeada335(11455)',
            driver='1.17.39395+14',historical_binary_hash_available=False,
            limits='Historical environment/runtime hashes absent. Current NEO and explicit Intel-only OpenCL ICD selection differ;diagnostic retained-loader validation and immutable safety settings are charged. Repetition count3 versus user default5. No Strata/llama throughput equivalence or historical regression claim.'),
        budgets=dict(wall=1800,cpu=2400,rss_cap=24<<30,text_cap=128<<20,file_cap=128<<20,total_cap=320<<20),
        trace_budget_reason='128MiB finite full-API budget for2048probe covers512ubatch attention and incidental Intel OpenCL enumeration. Long131K has parameter validation without full API trace.',
        retention='Keep exact command/environment/source/executable/model/runtime identities,3individual timings,actual provider/fault/closure result and limits. No new build. Retire successful traces,logs,duplicate external receipt after extraction;user installed binary/model are not disposable project builds.')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir()
    write(ROOT/'record.json',row);c[KEY]=row;write(RESULT,c)
    print('Prepared installed-reference diagnostic;commit plan before running.',flush=True)


def admit(initial):
    require(pins(initial['source_commit'])==initial['input_sha256'],'controller changed')
    require(external_source()==initial['external_source'],'external tracked source/build metadata changed')
    require(digest(EXE)==initial['binary']['sha256'] and digest(MODEL)==initial['model']['sha256'],
        'external binary/model differs')
    require(all(digest(p)==h for p,h in initial['runtime_files'].items()),'runtime input changed')
    require(all(digest(p)==h for p,h in initial['external_config_sha256'].items()),'ICD selection file changed')
    require(subprocess.check_output(['dpkg-query','-W','-f=${Package} ${Version}\n',*PACKAGES],
        text=True,timeout=15).splitlines()==initial['packages'],'installed package versions changed')
    require(runtime.guards.driver_guard()==initial['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==initial['boot'],'driver/boot differs')
    runtime.guards.no_model_owners([str(MODEL)])


def run():
    c=runtime.validation.committed(RESULT);row=c[KEY]
    require(row['stage']=='prepared' and not row['active'],'single-use plan only;no retry')
    initial=json.loads(json.dumps(row));admit(initial)
    owner=None;cursor='';row.update(stage='running',passed=False)
    def persist():
        row['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(row,commands=owner.commands if owner else []))
    def monitor(io,entry):
        seen=set()
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            try:maps=Path('/proc',str(io.proc.pid),'maps').read_text()
            except (FileNotFoundError,ProcessLookupError):maps=''
            for path in runtime.executable_mappings(maps):
                require(path in row['runtime_files'] or path==row['preload']['canary']['path'],
                    'unplanned executable map: '+path)
                seen.add(path)
            if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
        io.exit(1);entry['runtime_seen']=sorted(seen)
    try:
        text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
        cursor=next(x.split(': ',1)[1] for x in text.splitlines() if x.startswith('-- cursor: '))
        owner=runtime.supervision.Owner(ROOT/'owned',persist)
        row['outcomes']=[]
        for case in row['cases']:
            admit(initial)
            label=case['label'];warning=ROOT/(label+'.validation');trace=ROOT/(label+'.trace')
            env=dict(row['environment'],UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(warning))
            if label=='probe2048':env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:file,'+str(trace))
            if label!='canary':row.update(GPU_executed=True,model_executed=True)
            persist();print('Starting '+label,flush=True)
            entry,out,err=owner.run(label,case['argv'],env,LLAMA,interaction=monitor,**row['budgets'])
            require(runtime.supervision.closed(entry) and entry['exit_code']==0,'native child failed: '+err.read_text(errors='replace')[-2500:])
            evidence=runtime.validation.preload_evidence(err.read_text(),entry['pid'])
            require(evidence['loader']==row['preload']['loader'],'wrong UR provider')
            require(not warning.exists() or not warning.stat().st_size,'parameter validation reported an error')
            outcome=dict(label=label,normal_closed=True,actual_code_maps=entry['runtime_seen'])
            if label=='canary':
                require(json.loads(out.read_text())==dict(pid=entry['pid'],initialized=0,checked=46,expected=46,
                    version=(1<<32)-1,closed=0),'host canary failed')
            else:
                required={str((SDK/'lib/libsycl.so.9').resolve()),str((SDK/'lib/libur_adapter_level_zero_v2.so.0').resolve()),
                    '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39758'}
                require(required<=set(entry['runtime_seen']),'actual GPU provider absent')
                parsed=[json.loads(s) for s in out.read_text().splitlines() if s.startswith('{')]
                require(len(parsed)==1,'one JSON benchmark result required');test=parsed[0]
                require(test['n_prompt']==(2048 if label=='probe2048' else 131072) and test['n_gen']==0 and
                    test['type_k']==test['type_v']=='q8_0' and test['n_gpu_layers']==-1 and
                    'SYCL' in test['backends'] and test['flash_attn']==1 and
                    test['build_commit']=='abeada335' and test['build_number']==11455 and
                    test['n_batch']==2048 and test['n_ubatch']==512 and not test['no_kv_offload'],
                    'benchmark route/config differs')
                ns=test['samples_ns'];require(len(ns)==(1 if label=='probe2048' else 3) and all(isinstance(v,int) and v>0 for v in ns),'bad sample clocks')
                outcome['benchmark']=test
                if label=='probe2048':require(trace.exists() and 'urEnqueue' in trace.read_text(),'small full API trace absent')
            probe=dict(host_probe_accepted=True);runtime.validation.finish(probe,owner,cursor,row['boot'])
            require(probe['passed'] and probe['driver_after']==row['driver'],'fault/closure check failed')
            row['outcomes'].append(outcome);persist();print(label+':closed,provider/validation/fault gates pass',flush=True)
        test=row['outcomes'][-1]['benchmark'];speeds=[131072*1e9/v for v in test['samples_ns']]
        row.update(stage='closed',complete=True,host_probe_accepted=True,
            summary=dict(tok_s=speeds,mean_tok_s=statistics.mean(speeds),median_tok_s=statistics.median(speeds),
                stdev_tok_s=statistics.stdev(speeds),versus_user_mean_ratio=statistics.mean(speeds)/966.57,
                historical_Strata_cause='unattributed',new_model_route_selected=False))
        admit(initial)
    except BaseException as error:
        row.update(stage='failed',passed=False,error=repr(error))
    finally:
        if owner is not None and cursor:runtime.validation.finish(row,owner,cursor,row['boot'])
        row['passed']=bool(row.get('passed') and row['stage']=='closed')
        persist();c[KEY]=row;write(RESULT,c)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','error','summary','ownership_closed')}),flush=True)
    return int(not row['passed'])


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if sys.argv[1]=='prepare':prepare()
        else:sys.exit(run())
