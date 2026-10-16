#!/usr/bin/env python3
"""UNEXECUTED Main controller proposal. Serial fixed-env probes and offline comparisons.
Never applies/builds sources, invokes existing env-switching tests, or deletes captures.
Use the SAME existing measurement lock as all other GPU/build/test/cleanup jobs.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import fcntl
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
from capture_common import HEX, file_sha, new_dir, open_dir, read_json, require, write_json

HERE=Path(__file__).absolute().parent
GLOBAL_BUDGET=64 << 20
LOG_BUDGET=4 << 20
CAPTURE_BUDGET=8 << 20
CONTROLLER_LIMIT_SECONDS=21600
CONTROLLER_END=None
ROUTE_KEYS=('STRATA_SYCL_PROMPT_DIRECT','STRATA_SYCL_PROMPT_HEAD_SPLIT',
            'STRATA_SYCL_PROMPT_NATIVE_KV_MIN','STRATA_PROMPT_ATTN_Q4')

def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def start_ticks(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    # comm can contain spaces/parentheses. Field22 is suffix index19 after ') '.
    return int(s[s.rfind(')')+2:].split()[19])

def read_external_json(path):
    p=Path(path); fd=open_dir(p.parent)
    try:
        return read_json(fd,p.name)
    finally:
        os.close(fd)

def freeze_release(path):
    r=read_external_json(path)
    require(r.get('schema')=='head-split-freeze-release-v2' and r.get('example_only') is not True, 'actual Main freeze-release receipt required; example is unusable')
    require(r.get('benchmark_binary_label')=='92acf71e', 'wrong frozen benchmark binary label')
    binary=r.get('benchmark_binary_sha256','')
    require(HEX.fullmatch(binary), 'full frozen benchmark binary SHA256 required')
    for key,direct in (('direct0','0'),('direct1','1')):
        arm=r.get(key,{})
        require(arm.get('status')=='CLOSED_FULL_COMPLETE' and arm.get('direct')==direct and
                arm.get('clean') is True and arm.get('thinking') is True and
                type(arm.get('runs')) is int and arm['runs']>=3 and arm.get('agent_session_complete') is True,
                f'{key} full clean thinking3runs/agent benchmark is not closed')
        require(arm.get('binary_sha256')==binary and arm.get('owners_retired') is True,
                f'{key} binary/retirement mismatch')
        require(HEX.fullmatch(arm.get('receipt_sha256','')) and
                file_sha(arm['receipt_path'])==arm['receipt_sha256'], f'{key} benchmark receipt hash mismatch')
    owners=r.get('owners')
    require(isinstance(owners,list) and owners and r.get('all_benchmark_owners_listed') is True,
            'all benchmark server/client/controller owners must be listed')
    current_boot=boot_id()
    for owner in owners:
        require(type(owner.get('pid')) is int and owner['pid']>0 and
                type(owner.get('start_ticks')) is int and owner['start_ticks']>0 and
                owner.get('retired') is True, 'invalid/unfinished owner retirement record')
        if owner.get('boot_id')==current_boot:
            require(start_ticks(owner['pid']) != owner['start_ticks'], 'original benchmark owner is still alive')
        else:
            require(isinstance(owner.get('boot_id'),str) and bool(owner['boot_id']), 'owner boot ID missing')
    return {'path':str(Path(path).absolute()),'sha256':file_sha(path),'benchmark_binary_sha256':binary}

def group_absent(pgid):
    try:
        os.killpg(pgid,0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

def targeted(env):
    return {k:env[k] if k in env else '@unset' for k in ROUTE_KEYS}

def env_marker(value):
    return '@unset' if value is None else '@empty' if value=='' else value

def tree_size(root):
    count=0; size=0
    for base,dirs,files in os.walk(root,followlinks=False):
        for name in dirs:
            require(not Path(base,name).is_symlink(), 'unexpected output symlink')
        for name in files:
            p=Path(base,name)
            require(not p.is_symlink(), 'unexpected output symlink')
            count+=1; require(count<=8192,'too many controller artifacts')
            size+=p.stat().st_size
    return size

def supervise(argv,env,job_dir,identity,route,capture_dir,deadline,prepared=False):
    if CONTROLLER_END is not None:
        remaining=CONTROLLER_END-time.monotonic()
        require(remaining>0,"whole-controller deadline exhausted; no new job")
        deadline=min(deadline,remaining)
    if not prepared:
        os.close(new_dir(job_dir))
    out_fd=os.open(job_dir/'stdout.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    err_fd=os.open(job_dir/'stderr.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    p=None; status='LAUNCH_FAILED'; reason=None; byte_count=0
    start_wall=time.time(); start_mono=time.monotonic(); ticks=None; current_boot=boot_id()
    argv0_sha=file_sha(argv[0])
    try:
        p=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           start_new_session=True,close_fds=True)
        ticks=start_ticks(p.pid)
        write_json(job_dir/'started.json',{'schema':'head-split-owned-start-v2','pid':p.pid,'start_ticks':ticks,
                   'boot_id':current_boot,'argv':argv,'argv0_file_sha256':argv0_sha,'route_environment':targeted(env),
                   'wall_start_unix':start_wall,'deadline_seconds':deadline})
        sel=selectors.DefaultSelector()
        for stream,dest in ((p.stdout,out_fd),(p.stderr,err_fd)):
            os.set_blocking(stream.fileno(),False); sel.register(stream,selectors.EVENT_READ,dest)
        while sel.get_map():
            if time.monotonic()-start_mono >= deadline:
                reason='DEADLINE_STOP'; break
            for key,_ in sel.select(timeout=min(0.25,max(0.,deadline-(time.monotonic()-start_mono)))):
                data=os.read(key.fileobj.fileno(),65536)
                if not data:
                    sel.unregister(key.fileobj); key.fileobj.close(); continue
                if byte_count+len(data)>LOG_BUDGET:
                    reason='TEXT_BUDGET_STOP'; break
                offset=0
                while offset<len(data):
                    n=os.write(key.data,data[offset:]); require(n>0,'supervisor log write failed'); offset+=n
                byte_count+=len(data)
            if reason:
                break
        sel.close()
        if not reason:
            try:
                p.wait(timeout=max(0.001,deadline-(time.monotonic()-start_mono)))
            except subprocess.TimeoutExpired:
                reason='DEADLINE_STOP'
        if reason:
            try: os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try: os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                try: p.wait(timeout=30)
                except subprocess.TimeoutExpired: reason+='__RETIREMENT_UNKNOWN'
        gone=p.poll() is not None
        absent=gone and group_absent(p.pid)
        stable=all(file_sha(identity[k+'_path'])==identity[k+'_sha256'] for k in ('source','binary','fixture'))
        status=('CLOSED_EXIT0' if gone and absent and not reason and p.returncode==0 and stable
                else reason or ('SOURCE_DRIFT' if not stable else 'ABNORMAL_OR_UNRETIRED'))
        receipt={'schema':'head-split-process-v2','status':status,'pid':p.pid,'start_ticks':ticks,
                 'boot_id':current_boot,'returncode':p.returncode,'owned_pid_retired':gone,
                 'process_group_absent':absent,'capture_dir':str(capture_dir) if capture_dir else None,
                 'route':route,'route_environment':targeted(env),'argv':argv,'argv0_file_sha256':argv0_sha,
                 'source_sha256':identity['source_sha256'],'binary_sha256':identity['binary_sha256'],
                 'fixture_sha256':identity['fixture_sha256'],'identities_stable_after_exit':stable,
                 'wall_start_unix':start_wall,'wall_end_unix':time.time(),
                 'supervision_wall_seconds':time.monotonic()-start_mono,'performance_sample':False,
                 'text_bytes':byte_count,'text_budget':LOG_BUDGET,'deadline_seconds':deadline,
                 'fault_gate':'EXTERNAL_MAIN_REQUIRED','timeout_is_gpu_completion_proof':False}
        write_json(job_dir/'process.json',receipt)
        require(status=='CLOSED_EXIT0',f'job stopped: {job_dir.name}: {status}; no subsequent job authorized')
        return receipt
    except BaseException:
        # Preserve failed artifacts. If supervision itself faults, stop issuing
        # all later jobs, and retire only the process group this controller owns.
        if p is not None and p.poll() is None:
            try: os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired: pass
        raise
    finally:
        for fd in (out_fd,err_fd):
            os.fsync(fd); os.close(fd)
        if p is not None:
            for stream in (p.stdout,p.stderr):
                if stream is not None and not stream.closed:
                    stream.close()

def main():
    global CONTROLLER_END
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-release',required=True)
    parser.add_argument('--check-freeze',action='store_true')
    parser.add_argument('--binary'); parser.add_argument('--source'); parser.add_argument('--fixture-source')
    parser.add_argument('--output-root'); parser.add_argument('--lock-file')
    parser.add_argument('--fault-helper',help='Main-reviewed bounded read-only journal adapter; REQUIRED for native jobs')
    parser.add_argument('--include-isolation',action='store_true')
    a=parser.parse_args()
    released=freeze_release(a.freeze_release)
    if a.check_freeze:
        print('BOTH_CLEAN_BENCHMARK_ARMS_CLOSED_AND_OWNERS_RETIRED')
        return 0
    require(all((a.binary,a.source,a.fixture_source,a.output_root,a.lock_file,a.fault_helper)), 'binary/source/fixture/output/EXISTING lock/Main fault helper are required')
    # Use the pre-existing common lock. Never create a new lock namespace.
    lock_fd=os.open(a.lock_file,os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC)
    root=Path(a.output_root)
    try:
        fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        released=freeze_release(a.freeze_release) # Recheck after obtaining lock.
        CONTROLLER_END=time.monotonic()+CONTROLLER_LIMIT_SECONDS
        os.close(new_dir(root))
        os.close(new_dir(root/'jobs')); os.close(new_dir(root/'comparisons'))
        identity={}
        for k,path in (('source',a.source),('binary',a.binary),('fixture',a.fixture_source)):
            identity[k+'_path']=str(Path(path).resolve(strict=True))
            identity[k+'_sha256']=file_sha(identity[k+'_path'])
        fault_helper=str(Path(a.fault_helper).resolve(strict=True))
        fault_helper_sha=file_sha(fault_helper)
        # Snapshot into a dictionary. Never write os.environ or a native child's environment after exec.
        base_env=dict(os.environ)
        for key in ROUTE_KEYS[:2]: base_env.pop(key,None)
        base_env['STRATA_SYCL_PROMPT_NATIVE_KV_MIN']='0'
        write_json(root/'controller-start.json',{'schema':'head-split-controller-start-v2','freeze_release':released,
                   'identity':identity,'global_byte_budget':GLOBAL_BUDGET,'whole_deadline_seconds':CONTROLLER_LIMIT_SECONDS,
                   'controller_sha256':file_sha(Path(__file__)),
                   'comparator_sha256':file_sha(HERE/'compare_captures.py'),
                   'capture_common_sha256':file_sha(HERE/'capture_common.py'),
                   'fault_helper':fault_helper,'fault_helper_sha256':fault_helper_sha,
                   'no_existing_env_switch_test':True})
        captures={}
        def room():
            require(time.monotonic()<CONTROLLER_END,"whole-controller deadline exhausted")
            require(tree_size(root) + CAPTURE_BUDGET + 3*LOG_BUDGET + 4*(1<<20) <= GLOBAL_BUDGET,
                    'controller total byte budget reserve exhausted')
        def fault_step(name,phase,job):
            require(file_sha(fault_helper)==fault_helper_sha,'Main fault helper source drift')
            supervise([fault_helper,phase,'--job-dir',str(job)],base_env.copy(),
                      root/'jobs'/(name+'_fault_'+phase),identity,'fault-helper',None,60)
            require(file_sha(fault_helper)==fault_helper_sha,'Main fault helper changed during check')
            fd=open_dir(job,private=True)
            try:
                before=read_json(fd,'fault-before.json')
                require(before.get('schema')=='head-split-fault-before-v2' and
                        before.get('boot_id')==boot_id() and isinstance(before.get('cursor'),str) and
                        bool(before['cursor']),'missing actual kernel-journal start boundary')
                if phase=='end':
                    after=read_json(fd,'fault-after.json')
                    require(after.get('schema')=='head-split-fault-after-v2' and
                            after.get('boot_id')==before['boot_id'] and after.get('before_cursor')==before['cursor'] and
                            after.get('status')=='PASS' and after.get('new_faults')==0 and
                            after.get('error_scan_success') is True and isinstance(after.get('after_cursor'),str) and
                            bool(after['after_cursor']) and isinstance(after.get('filter_id'),str) and
                            bool(after['filter_id']),'fresh fault gate missing/failed; stop all later jobs')
            finally:
                os.close(fd)
        def native(name,route,case_set='full',fmt=1,direct='1',split=None):
            room()
            for k in ('source','binary','fixture'):
                require(file_sha(identity[k+'_path'])==identity[k+'_sha256'],'source/binary changed before launch')
            env=base_env.copy()
            if direct is not None: env['STRATA_SYCL_PROMPT_DIRECT']=direct
            if split is not None: env['STRATA_SYCL_PROMPT_HEAD_SPLIT']=split
            job=root/'jobs'/name; capture=job/'capture'
            argv=[identity['binary_path'],'--route',route,'--case-set',case_set,'--capture-dir',str(capture),
                  '--expect-direct',env_marker(direct),'--expect-split',env_marker(split),
                  '--source-id',identity['source_sha256'],'--binary-id',identity['binary_sha256'],
                  '--fixture-id',identity['fixture_sha256'],'--format',str(fmt),'--capture-budget',str(CAPTURE_BUDGET)]
            os.close(new_dir(job))
            fault_step(name,'begin',job)
            supervise(argv,env,job,identity,route,capture,900,prepared=True)
            fault_step(name,'end',job)
            captures[name]=str(capture)
            return str(capture)
        def comparator(name,args):
            room()
            argv=[sys.executable,'-B',str(HERE/'compare_captures.py'),*args,'--output',str(root/'comparisons'/(name+'.json'))]
            supervise(argv,base_env.copy(),root/'jobs'/('compare_'+name),identity,'offline',None,120)
        room()
        supervise([identity['binary_path'],'--help'],base_env.copy(),root/'jobs'/'help',identity,'host',None,30)
        room()
        supervise([identity['binary_path'],'--host-only'],base_env.copy(),root/'jobs'/'host_only',identity,'host',None,60)
        legacy=native('full_legacy','legacy')
        c32=native('full_32','32',split='32')
        comparator('ch32',['--mode','ch32','--legacy',legacy,'--ch32',c32])
        c64=native('full_64','64',split='64')
        comparator('full',['--mode','full','--legacy',legacy,'--ch32',c32,'--ch64',c64])
        if a.include_isolation:
            configs=[(1,'1',s) for s in ('','0','off','032','+32','32x','64 ','64x','128')]
            configs += [(1,d,s) for d in ('0','','10','1x') for s in ('32','64')]
            configs += [(fmt,'1',s) for fmt in (0,2,3) for s in ('32','64')]
            require(len(configs)==23,'isolation plan count')
            for n,(fmt,direct,split) in enumerate(configs):
                left=native(f'iso_{n:02d}_left','legacy','isolation',fmt,direct,None)
                right=native(f'iso_{n:02d}_right','legacy','isolation',fmt,direct,split)
                comparator(f'iso_{n:02d}',['--mode','isolation','--left',left,'--right',right])
        write_json(root/'controller-complete.json',{'schema':'head-split-controller-complete-v2',
                   'status':'OUTPUT_COMPARISONS_AND_FAULT_GATES_COMPLETE_ROUTE_REVIEW_PENDING',
                   'full_cases_per_arm':28,'full_arms':3,'isolation_pairs':23 if a.include_isolation else 0,
                   'capture_dirs':captures,'total_bytes_before_completion':tree_size(root),
                   'fault_gate':'PASS_MAIN_HELPER','route_observation':'EXTERNAL_MAIN_REQUIRED',
                   'model_quality_or_speed_proved':False,'physical_256k_or_restore_proved':False,
                   'capture_consumer':'Offline pair comparisons plus Main first-failure review',
                   'next_retention_review':'After comparisons and diagnostic/quality decisions are recorded',
                   'cleanup_performed':False})
        print('OUTPUT_COMPARISONS_AND_MAIN_FAULT_GATES_COMPLETE; ROUTE REVIEW REQUIRED')
        return 0
    finally:
        os.close(lock_fd)

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print('STOP:',error,file=sys.stderr)
        sys.exit(1)
