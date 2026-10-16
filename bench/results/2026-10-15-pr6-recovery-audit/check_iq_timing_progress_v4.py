#!/usr/bin/env python3
"""One bounded owned diagnostic for the failed IQ4_XS/ne16 OFF case."""
import fcntl
import json
import os
from pathlib import Path
import re
import selectors
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
BASE=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEVICE=Path('/sys/bus/pci/devices/0000:05:00.0')
sys.path.insert(0,str(HERE/'owned-stall-stack-probe-v1'))
from owned_stack_probe import StackProbe, StallCaptured, load_owner, require, sha, boot_id
FAULT=re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
PHASE=re.compile(rb'^iq-progress-v3 seq=(\d+) steady_ns=(\d+) phase=([A-Za-z_]+) edge=([a-z_]+) index=(-?\d+)$')


class TimingProgressProbe(StackProbe):
    def __init__(self,*args,stderr,**kwargs):
        self.stderr=stderr
        self.offset=0
        self.pending=b''
        self.next_check=0
        self.sequence=-1
        self.steady_ns=-1
        super().__init__(*args,**kwargs)
        self.record['phases']=[]
        self.record['progress_policy']='Strict complete increasing-sequence/steady timestamp stderr markers; no repeated warnings.'

    def read_progress(self):
        now=time.monotonic()
        if now>=self.next_check:
            self.next_check=now+.05
            with self.stderr.open('rb') as stream:
                stream.seek(self.offset)
                consumed=0
                while True:
                    data=stream.read(65536)
                    if not data:
                        break
                    consumed+=len(data)
                    require(consumed<=4<<20,'total bounded progress scan budget')
                    self.offset+=len(data)
                    lines=(self.pending+data).split(b'\n')
                    self.pending=lines.pop()
                    require(len(self.pending)<=16384,'progress partial line budget')
                    for line in lines:
                        match=PHASE.fullmatch(line)
                        if not match:
                            continue
                        sequence,stamp=int(match[1]),int(match[2])
                        require(sequence==self.sequence+1 and stamp>=self.steady_ns,'nonmonotonic progress')
                        require(len(self.record['phases'])<160,'phase count budget')
                        self.sequence,self.steady_ns=sequence,stamp
                        self.last_progress=now
                        self.record['phases'].append(line.decode())

    def poll(self):
        now=time.monotonic()
        self.read_progress()
        age=now-self.entry['spawn_begin_monotonic']
        if not self.attempted and not self.capture_active and age>=100:
            self.original_poll()
            require(age<120,'safe attach reserve elapsed; do not attach near native deadline')
            self.requests_stopped=True
            self.record['requests_stopped']=True
            self.capture_now('latest_attach_reserve',silence_seconds=now-self.last_progress)
            raise StallCaptured('one reserved stack capture; no further cases')
        super().poll()


def main():
    module=load_owner(PROJECT/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    root=BASE/'main-iq-timing-progress-root-v4'
    with (BASE/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        build=json.loads((HERE/'next-diagnostics-build-v1.json').read_text())
        require(build['passed'] and build['complete'] and not build['active'],'diagnostic build gate')
        exe=Path(build['binary'])
        require(sha(exe)==build['binary_sha256'],'diagnostic binary changed')
        require(all(sha(PROJECT/n)==h for n,h in build['source_pins'].items()),'source changed')
        require(all(sha(Path(n))==h for n,h in {**build['linked_archives'],**build['linked_mkl']}.items()),'linked library changed')
        previous=json.loads((HERE/'parallel-iq-timing-wide-v2.json').read_text())
        require(not previous['active'] and all(c.get('direct_child_reaped') and c.get('session_empty') for c in previous['commands']),'previous job not retired')
        host=BASE/'main-owned-stack-host-preflight-root-v2'
        host_gate=json.loads((host/'preflight-v1/record.json').read_text())
        auto=json.loads((HERE/'owned-prefill-stall-host-auto-v1.json').read_text())
        require(host_gate['passed'] and host_gate['complete'] and not host_gate['active'] and auto['passed'] and auto['complete'] and not auto['active'],'qualified host attach/detach/automatic trigger gate')
        shim=host/'libowned_ptracer_parent.so'
        require(sha(shim)==host_gate['inputs'][str(shim)],'qualified shim changed')
        require((DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists(),'device driver/dump gate')
        require(not root.exists(),'fresh diagnostic root')
        root.mkdir(mode=0o700)
        record=dict(schema='main-iq-timing-progress-v4',active=True,passed=False,complete=False,
                    GPU_executed=False,commands=[],debugger_commands=[],probe={},boot_id=boot_id(),
                    binary_sha256=sha(exe),build_receipt_sha256=sha(HERE/'next-diagnostics-build-v1.json'),
                    recipe_sha256=sha(Path(__file__)),shim_sha256=sha(shim),performance_qualified=False,
                    original_failed_receipt_sha256=sha(HERE/'parallel-iq-timing-wide-v2.json'),
                    scope='Single progress/stack diagnostic. Does not repair failed wide comparison or prove intermittent fault resolved.')

        def persist():
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')

        owner=module.Owner(root/'native-owned',persist)
        debugger=module.Owner(root/'gdb-owned',persist)
        record['commands'],record['debugger_commands']=owner.commands,debugger.commands
        host_env=dict(PATH='/usr/bin:/bin',HOME=str(root),LANG='C',LC_ALL='C')

        def utility(label,argv):
            c,out,err=owner.run(label,argv,host_env,PROJECT,wall=30,cpu=30,rss_cap=2<<30,
                                text_cap=1<<20,total_cap=256<<20,file_cap=1<<20)
            require(module.closed(c) and c['exit_code']==0,label+' failed: '+err.read_text()[-1000:])
            return out.read_text()

        cursor=None
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            record['journal_before_cursor']=cursor
            original=[c for c in previous['commands'] if c['label']=='case-031']
            require(len(original)==1,'exact failed original case')
            env=dict(original[0]['environment'],STRATA_IQ_TIMING_PROGRESS='1',LD_PRELOAD=str(shim),
                     XDG_CACHE_HOME=str(root/'cache'),UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                     UR_LOG_TRACING='level:info;flush:info;output:stderr',
                     UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                     UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                     UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
            argv=[str(exe),*original[0]['argv'][1:]]

            def interaction(io,entry):
                io.proc.stdin.close()
                probe=TimingProgressProbe(module,debugger,io,entry,record['probe'],persist,PROJECT,
                                          stderr=root/'native-owned/diagnostic.stderr',silence_seconds=45)
                try:
                    with selectors.DefaultSelector() as ready:
                        ready.register(io.proc.stdout,selectors.EVENT_READ)
                        while not io.exit_watch.poll(0) or not io.eof:
                            try:
                                io.poll()
                            except StallCaptured:
                                record['stack_captured']=True
                                persist()
                            if ready.select(.02):
                                io.read()
                                io.pending.clear()
                    probe.next_check=0
                    probe.read_progress()
                finally:
                    probe.close()

            record['GPU_executed']=True
            c,out,err=owner.run('diagnostic',argv,env,PROJECT,interaction=interaction,wall=180,
                               cpu=180*len(os.sched_getaffinity(0)),rss_cap=8<<30,text_cap=4<<20,
                               total_cap=256<<20,file_cap=16<<20)
            record['native_exit_code']=c.get('exit_code')
            record['native_error']=c.get('error')
            require(module.closed(c) and c['exit_code']==0,'diagnostic native failed: '+repr(c.get('error'))+' '+err.read_text()[-2000:])
            sample=json.loads(out.read_text())
            require(sample['complete'] and sample['queue_retired'] and sample['full_checks']==2 and sample['guards']
                    and sample['full_half_oracle'] and sample['full_gemm_reference'] and sample['max_output_error']==0,
                    'diagnostic full correctness incomplete')
            require(record['probe']['phases'] and 'phase=retire edge=end' in record['probe']['phases'][-1],
                    'complete phase transcript absent')
            record.update(passed=True,complete=True,result=sample)
        except BaseException as error:
            record['error']=repr(error)
            raise
        finally:
            try:
                if cursor:
                    journal=[json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record['fresh_faults']=[e for e in journal if FAULT.search(str(e.get('MESSAGE','')))]
                    record['boot_unchanged']=boot_id()==record['boot_id']
                    record['driver_dump_exists']=(DEVICE/'devcoredump').exists()
                    if record['fresh_faults'] or not record['boot_unchanged'] or record['driver_dump_exists']:
                        record['passed']=False
            finally:
                record.update(active=owner.active is not None or debugger.active is not None,finished_utc=module.utc())
                persist()
                (HERE/'iq-timing-progress-v4.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(passed=record['passed'],capture=record['probe'].get('captures',[]))),flush=True)


if __name__=='__main__':
    main()
