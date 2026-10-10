"""One post-fault status/integer diagnostic; no reset, retry, or model run."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import time
import types

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE = Path(__file__).resolve().parent
OWNER = B/'xestrata-clean-64k-comparison-v4/direct_owner.py'
OWNER_SHA = '8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686'
spec = importlib.util.spec_from_file_location('qualified_direct_owner', OWNER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert m.sha(OWNER) == OWNER_SHA
require, sha, closed, identity = m.require, m.sha, m.closed, m.identity
HELPER = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/sycl/tools/recover-xe.sh')
HELPER_SHA = '2f90005f79d2309747d917cff51dc53d8ee6192d68cb1feca074f84af8b0796c'
FAULT_RECEIPT = B/'owned-xestrata39-hostusm4k-code32k-diagnostic-v8-r1/record.json'
FAULT_SHA = 'be30b5494b9f535150353e62e58b1450933072059f9362a7e3941003d14c1881'
DUMP = Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')


def snapshot():
    if not DUMP.exists():
        return dict(present=False)
    def meta(path, follow):
        s = path.stat() if follow else path.lstat()
        return dict(dev=s.st_dev, ino=s.st_ino, mode=s.st_mode,
                    ctime_ns=s.st_ctime_ns, mtime_ns=s.st_mtime_ns)
    return dict(present=True, symlink=str(DUMP.readlink()) if DUMP.is_symlink() else None,
                resolved=str(DUMP.resolve(strict=True)), node=meta(DUMP,False),
                target=meta(DUMP,True), data_metadata=meta(DUMP/'data',True))


def main():
    require(os.geteuid()!=0, 'ordinary account only')
    out=B/'postfault-status-health-root-r2'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not out.exists(), 'new output only')
        out.mkdir(mode=0o700)
        record=dict(active=True,completed=False,passed=False,model_executed=False,
                    reset_executed=False,recovery_executed=False,retry_count=0,
                    scope='new context: one status-gated 64KiB integer round only',
                    source_sha256=sha(HERE/'health.cpp'),controller_sha256=sha(__file__),
                    owner_sha256=OWNER_SHA,helper_sha256=HELPER_SHA,
                    old_fault_receipt_sha256=FAULT_SHA,old_fault_status='FAILED/incomplete',
                    started_utc=m.utc(),kernel_release=os.uname().release,
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        def save():
            tmp=out/'record.json.tmp'
            tmp.write_text(json.dumps(record,indent=2)+'\n')
            tmp.replace(out/'record.json')
        owner=m.Owner(out/'commands',save)
        record['commands']=owner.commands
        base=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        runtime=dict(base,LD_LIBRARY_PATH=':'.join([
            '/opt/intel/oneapi/compiler/2026.1/lib',
            '/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib',
            '/opt/intel/oneapi/umf/1.1/lib','/usr/lib/x86_64-linux-gnu']),
            SYCL_CACHE_PERSISTENT='0',
            UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',
            UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
            NEOReadDebugKeys='1',EnableDirectSubmission='0')
        begin=time.monotonic()
        def command(label,argv,env=base,wall=10):
            require(time.monotonic()-begin<240, 'whole controller deadline')
            e,so,se=owner.run(label,argv,dict(env),HERE,wall=min(wall,240-(time.monotonic()-begin)),
                              text_cap=8<<20,rss_cap=4<<30,cpu=max(5,int(wall)),total_cap=256<<20)
            require(closed(e) and e['exit_code']==0,'normal owned utility: '+label)
            return so.read_text()
        cursor=None
        save()
        try:
            require(sha(HELPER)==HELPER_SHA and sha(FAULT_RECEIPT)==FAULT_SHA,'immutable supporting evidence')
            prior=json.loads(FAULT_RECEIPT.read_text())
            require(prior['active'] is False and prior['completed'] is False
                    and prior['owned_gdb_closed'] is True and prior['auxiliary_ownership_closed'] is True
                    and prior['cleanup']==dict(forced=True,inferior_survived=False,gdb_survived=False),
                    'old fault ownership closed; preserve failed result')
            require(prior['boot_id']==record['boot_id'],'same boot as known closed fault')
            for key in ('inferior','debugger'):
                v=prior[key]; current=identity(v['pid'])
                require(not current or current['start_ticks']!=v['start_ticks'], 'known fault process gone')
            record['known_fault_owners_gone']=True
            require(shutil.disk_usage(B).free>2<<30, 'artifact capacity')
            helper=types.ModuleType('read_only_diagnostic')
            exec(compile(HELPER.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(HELPER),'exec'),helper.__dict__)
            env=helper.diagnostic_environment(runtime)
            record['environment']=env
            compiler=Path('/opt/intel/oneapi/compiler/2026.1/bin/icpx')
            record['compiler']=dict(path=str(compiler),sha256=sha(compiler))
            binary=HERE/'health'
            require(not binary.exists(), 'no binary overwrite')
            command('build',[str(compiler),'-fsycl','-O2',str(HERE/'health.cpp'),'-lze_loader','-o',str(binary)],runtime,180)
            record['binary']=dict(path=str(binary),sha256=sha(binary),bytes=binary.stat().st_size)
            # Invalid invocation exits before enumeration or any GPU API.
            e,so,se=owner.run('bad-argc',[str(binary)],runtime,HERE,wall=5,text_cap=65536,rss_cap=256<<20,cpu=5)
            require(closed(e) and e['exit_code']==1 and not so.read_text()
                    and 'one fixed B570 BDF is required' in se.read_text(),'CPU argument gate')
            record['CPU_argument_gate_passed']=True
            record['dump_before']=snapshot()
            text=command('kernel-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager'])
            match=re.search(r'^-- cursor: (.+)$',text,re.M)
            require(match is not None,'fresh journal cursor')
            cursor=match[1]; record['kernel_cursor_before']=cursor
            require(snapshot()==record['dump_before'],'stable known dump before launch')
            save()
            result=command('integer-health',[str(binary),'0000:05:00.0'],env,30)
            expected=['STATUS 0000:05:00.0 0','QUEUE 0000:05:00.0',
                      'H2D 65536 complete','KERNEL 16384 complete','D2H 65536 complete',
                      'PASS 0000:05:00.0: 1 round, 16384 exact words']
            require(result.splitlines()==expected, 'complete exact status/one-round result')
            record.update(completed=True,status_success=True,integer_exact_words=16384)
        except BaseException as exc:
            record['error']=repr(exc)
        finally:
            if cursor and owner.active is None:
                try:
                    interval=command('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'])
                    rows=[json.loads(v) for v in interval.splitlines() if v.startswith('{')]
                    relevant=[v for v in rows if '0000:05:00.0' in v.get('MESSAGE','') or re.search(r'\bxe\b',v.get('MESSAGE',''))]
                    record['kernel_device_entries']=relevant
                    record['new_fault_messages']=[v['MESSAGE'] for v in relevant if helper.FAULT.search(v['MESSAGE'])]
                    record['dump_after']=snapshot()
                    record['dump_unchanged']=record['dump_after']==record['dump_before']
                    require(not record['new_fault_messages'] and record['dump_unchanged'],'new fault or dump identity change')
                    record['kernel_gate_passed']=True
                except BaseException as exc:
                    record['kernel_gate_error']=repr(exc)
            record['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
            record['active']=owner.active is not None
            record['passed']=bool(record['completed'] and record.get('kernel_gate_passed')
                                  and record['boot_unchanged'] and not record['active']
                                  and not record.get('error') and not record.get('kernel_gate_error'))
            record['elapsed_seconds']=time.monotonic()-begin
            record['finished_utc']=m.utc()
            save()
        print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],
                             error=record.get('error'),kernel_gate_error=record.get('kernel_gate_error'))))
        return 0 if record['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
