#!/usr/bin/env python3
"""One separately planned syscall diagnostic of the closed, unused query error."""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import metric_inventory as inventory

P,B=inventory.P,inventory.B
ROOT=B/'b570-metric-query-diagnostic-v1'
KEY='event_query_syscall_diagnostic'


def inputs(revision):
    pins=inventory.inputs(revision)
    for name in ('bench/sycl/metric_query_setup.cpp','tools/sycl/metric_query_diagnostic.py'):
        raw=inventory.git('show',revision+':'+name)
        inventory.guards.require(raw==(P/name).read_bytes(),'diagnostic source changed')
        pins[name]=inventory.hashlib.sha256(raw).hexdigest()
    return pins


def admission(base,record):
    inventory.admission(base,True)
    original=base['event_query_setup']
    inventory.guards.require(original['stage']=='closed' and original['passed'] and original['complete'] and
        original['ownership_closed'] and not original['active'] and original['fresh_faults']==[] and
        original['setup_ready'] is False and original['summary']['last_api']=='zetMetricQueryPoolCreate' and
        original['summary']['result']==0x78000004 and original['normal_object_closure'],'closed original error required')
    for name in ('bench/sycl/metric_query_setup.cpp','bench/sycl/metric_inventory.cpp'):
        raw=inventory.git('show',original['source_commit']+':'+name)
        inventory.guards.require(raw==(P/name).read_bytes(),'original native input changed')
    inventory.guards.require(inputs(record['source_commit'])==record['inputs'] and
        inventory.sha(original['binary']['path'])==record['binary']['sha256']==original['binary']['sha256'] and
        inventory.sha(record['strace']['path'])==record['strace']['sha256'],'diagnostic source/tool/target changed')


def main(action):
    base=inventory.committed(inventory.RESULT)
    if action=='prepare':
        inventory.guards.require(KEY not in base and not ROOT.exists(),'fresh diagnostic only;no blind retry')
        revision=inventory.git('rev-parse','HEAD').decode().strip();original=base['event_query_setup']
        ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir();(ROOT/'cache').mkdir()
        record=dict(stage='prepared',source_commit=revision,inputs=inputs(revision),binary=original['binary'],
            strace=dict(path='/usr/bin/strace',sha256=inventory.sha('/usr/bin/strace')),
            environment=dict(original['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),
            command=['/usr/bin/strace','-f','-ttt','-s','1024','-e','trace=execve,openat,ioctl,write','-o',str(ROOT/'syscalls.txt'),'--',original['binary']['path']],
            plan='Original one-slot query setup returned0x78000004 despite valid zeroed descriptor/count1,event group andsuccessful activation. Compatible official NEO source can map metrics-library initialization/query creation/report-size failures to INVALID_ARGUMENT,so do not infer bad userarguments orpermissions. One separately committed strace ofunchanged original binary correlates libraryopen/ioctl results with APIprogress;no retryloop,no newqueue/commands/read/streamer orpermission change. Require originalapierror,normallyclosed context/deactivation/owner,nofault andpinnedELFs. Keepatmostone256KiB causallyrelevant diagnostic;retirefullsyscall/APIhistory afterbounded attribution.',
            passed=False,complete=False,active=False,ownership_closed=True,collection_qualified=False)
        admission(base,record);inventory.write(ROOT/'plan.json',record);base[KEY]=record;inventory.write(inventory.RESULT,base)
        print('Prepared separately scoped syscall diagnostic;commit before execution.',flush=True);return 0
    record=base[KEY];inventory.guards.require(record['stage']=='prepared' and not (ROOT/'record.json').exists(),'inspect existing diagnostic owner;no retry')
    admission(base,record);owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        inventory.write(ROOT/'record.json',dict(record,owned_commands=owner.commands if owner else []))
    owner=inventory.owner_module.Owner(ROOT/'owned',persist)
    record.update(stage='running',passed=False,complete=False,ownership_closed=False);persist()
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
    try:
        entry,out,err=owner.run('strace',record['command'],record['environment'],P,wall=60,cpu=120,
            rss_cap=2<<30,text_cap=8<<20,file_cap=16<<20,total_cap=96<<20)
        inventory.guards.require(inventory.owner_module.closed(entry) and entry['exit_code']==3,'strace/native closure or expected API exit failed')
        events=[json.loads(line) for line in out.read_text().splitlines()]
        summaries=[e for e in events if e['kind']=='summary']
        inventory.guards.require(summaries==[base['event_query_setup']['summary']],'diagnostic changed original API outcome')
        mapped={str(Path(e['path']).resolve()) for e in events if e['kind']=='elf'}
        inventory.guards.require(mapped<=set(base['runtime_pins']),'unqualified diagnostic target ELF')
        trace=ROOT/'syscalls.txt';text=trace.read_text();lines=text.splitlines()
        executable=record['binary']['path']
        inventory.guards.require(sum('execve("'+executable+'"' in line and line.endswith('= 0') for line in lines)==1,'target exec identity missing')
        markers=[i for i,line in enumerate(lines) if 'zetMetricQueryPoolCreate' in line]
        inventory.guards.require(markers,'pool API markers absent from syscall trace')
        window=lines[max(0,min(markers)-4):max(markers)+5]
        record.update(stage='closed',passed=True,complete=True,summary=summaries[0],
            native_API_failures=[e for e in events if e['kind']=='return' and e['result']!=0],
            pool_syscall_window_lines=len(window),pool_syscall_window_sha256=inventory.hashlib.sha256(('\n'.join(window)+'\n').encode()).hexdigest(),
            pool_failed_ioctls=[x for x in window if ('ioctl(' in x or '<... ioctl resumed>' in x) and '= -1' in x],
            pool_failed_library_opens=[x for x in window if ('openat(' in x or '<... openat resumed>' in x) and '= -1' in x],
            raw_trace=dict(path=str(trace),sha256=inventory.sha(trace),bytes=trace.stat().st_size),
            stdout_sha256=inventory.sha(out),stderr_sha256=inventory.sha(err))
        excerpt=ROOT/'pool-window.txt';excerpt.write_text('\n'.join(window)+'\n')
        inventory.guards.require(excerpt.stat().st_size<=256<<10,'diagnostic excerpt cap')
        record['excerpt']=dict(path=str(excerpt),sha256=inventory.sha(excerpt),bytes=excerpt.stat().st_size)
    except BaseException as error:record.update(stage='failed',passed=False,error=repr(error))
    finally:
        record['ownership_closed']=owner.active is None and all(inventory.owner_module.closed(c) for c in owner.commands)
        try:
            journal=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            record['fresh_faults']=[x for x in journal.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
            record['driver_after']=inventory.guards.driver_guard();record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except Exception as error:record['fault_probe_error']=repr(error)
        record['passed']=bool(record['passed'] and record['ownership_closed'] and record.get('fresh_faults')==[] and
            record.get('boot_unchanged') and 'fault_probe_error' not in record)
        persist();inventory.write(inventory.RESULT,base)
    print(json.dumps({k:record.get(k) for k in ('stage','passed','ownership_closed','pool_failed_ioctls','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    inventory.guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'use prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);inventory.guards.no_model_owners([]);inventory.guards.driver_guard()
        sys.exit(main(sys.argv[1]))
