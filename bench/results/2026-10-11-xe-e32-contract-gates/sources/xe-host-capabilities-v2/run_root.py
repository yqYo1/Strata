import fcntl, importlib.util, json, os, re, time, types
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE = Path(__file__).resolve().parent
OWNER = B/'xestrata-clean-64k-comparison-v4/direct_owner.py'
spec = importlib.util.spec_from_file_location('qualified_owner', OWNER)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.require(m.sha(OWNER) == '8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686', 'qualified owner')
HELPER = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011/sycl/tools/recover-xe.sh')
DUMP = Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
BASE = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
RUNTIME = dict(BASE, LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib:/opt/intel/oneapi/umf/1.1/lib:/usr/lib/x86_64-linux-gnu',
               SYCL_CACHE_PERSISTENT='0', UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',
               UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', ONEAPI_DEVICE_SELECTOR='level_zero:gpu', NEOReadDebugKeys='1', EnableDirectSubmission='0')

def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=B/'xe-host-capabilities-root-v2'; m.require(not out.exists(), 'immutable new output'); out.mkdir()
        record=dict(active=True, complete=False, passed=False, started_utc=m.utc(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    source_sha256=m.sha(HERE/'capabilities.cpp'), controller_sha256=m.sha(__file__), owner_sha256=m.sha(OWNER), commands=[],
                    queue_created=False, allocation_created=False, kernel_launched=False, model_launched=False, recovery_executed=False, reset_executed=False, retry_count=0)
        def save(): (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=m.Owner(out/'commands',save); record['commands']=owner.commands
        helper=types.ModuleType('read_only_helper')
        m.require(m.sha(HELPER)=='51f5ba140613ae6ee81905eeeea153243a6a41742488ac161d76bad241944457','health helper pin')
        exec(compile(HELPER.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(HELPER),'exec'),helper.__dict__)
        env=helper.diagnostic_environment(RUNTIME); record['environment']=env
        begin=time.monotonic(); save(); cursor=None
        def command(label,args,environment=BASE,wall=15):
            m.require(time.monotonic()-begin<240,'controller deadline')
            e,so,se=owner.run(label,args,environment,HERE,wall=wall,text_cap=32<<20,rss_cap=8<<30,cpu=int(wall),total_cap=64<<20)
            m.require(m.closed(e) and e['exit_code']==0,'closed normal '+label)
            return so.read_text()
        try:
            m.require(not DUMP.exists(),'no preexisting dump')
            boot=command('kernel-boot-before',['/usr/bin/journalctl','-k','-b','--no-pager','-o','json'])
            relevant=[json.loads(s) for s in boot.splitlines() if s.startswith('{')]
            relevant=[r for r in relevant if '0000:05:00.0' in r.get('MESSAGE','') or re.search(r'\bxe\b',r.get('MESSAGE',''))]
            record['kernel_preflight_entries']=relevant
            m.require(not any(helper.FAULT.search(v['MESSAGE']) for v in relevant),'no known fault on boot')
            compiler='/opt/intel/oneapi/compiler/2026.1/bin/icpx'
            record['compiler_sha256']=m.sha(compiler)
            record['compiler_version']=command('compiler-version',[compiler,'--version'])
            binary=out/'capabilities'
            command('compile',[compiler,'-std=c++20','-fsycl','-O2','-Wall','-Wextra','-Werror',str(HERE/'capabilities.cpp'),'-lze_loader','-o',str(binary)],wall=120)
            record['binary']=dict(path=str(binary),sha256=m.sha(binary),bytes=binary.stat().st_size)
            text=command('kernel-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager'])
            match=re.search(r'^-- cursor: (.+)$',text,re.M); m.require(match is not None,'fresh cursor'); cursor=match[1]
            record['kernel_cursor_before']=cursor; save()
            result=command('query',[str(binary),'0000:05:00.0'],env,30)
            record['query_output']=result.splitlines()
            m.require(result.splitlines()[0]=='STATUS 0000:05:00.0 0' and result.splitlines()[-1]=='PASS query only','status gated complete query')
            record['complete']=True
        except BaseException as exc:
            record['error']=repr(exc)
        finally:
            if cursor and owner.active is None:
                try:
                    after=command('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'])
                    relevant=[json.loads(s) for s in after.splitlines() if s.startswith('{')]
                    relevant=[r for r in relevant if '0000:05:00.0' in r.get('MESSAGE','') or re.search(r'\bxe\b',r.get('MESSAGE',''))]
                    record['kernel_device_entries']=relevant; record['new_fault_messages']=[r['MESSAGE'] for r in relevant if helper.FAULT.search(r['MESSAGE'])]
                    m.require(not record['new_fault_messages'] and not DUMP.exists(),'no fault/dump after query'); record['kernel_gate_passed']=True
                except BaseException as exc: record['kernel_gate_error']=repr(exc)
            record['source_stable']=record['source_sha256']==m.sha(HERE/'capabilities.cpp')
            record['boot_unchanged']=record['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            record['active']=owner.active is not None
            record['passed']=bool(record['complete'] and record.get('kernel_gate_passed') and record['source_stable'] and record['boot_unchanged'] and not record['active'] and not record.get('error') and not record.get('kernel_gate_error'))
            record['finished_utc']=m.utc(); save()
        print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],error=record.get('error'),output=record.get('query_output'))))
        return 0 if record['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
