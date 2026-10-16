#!/usr/bin/env python3
"""Unexecuted Main-owned serial subgroup array/route diagnostic; no speed gate."""
import collections
import fcntl
import json
import os
from pathlib import Path
import re
import selectors
import time
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
A = HERE.parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
sys.path.insert(0, str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import StackProbe, StallCaptured, load_owner, require, sha, boot_id
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
# One deliberately invalid value; each row is an independent immutable exec.
ARMS = [('off','0',''), ('on','1',''), ('sg16','0','sg16'), ('sg32','0','sg32'),
        ('on-sg16','1','sg16'), ('on-sg32','1','sg32'),
        ('invalid','0','SG16'), ('on-invalid','1','SG16')]


def closed_success(d):
    return d.get('passed') and d.get('complete') and not d.get('active')


def routes(path, mode, parallel, selector):
    # Same observed UR create->launch handle join used by Main's prior collector.
    create = re.compile(r'urKernelCreate\(.*\.pKernelName = 0x[0-9a-f]+ \(([^)]+)\), \.phKernel = 0x[0-9a-f]+ \((0x[0-9a-f]+)\)\) -> UR_RESULT_SUCCESS')
    launch = re.compile(r'<--- urEnqueueKernelLaunch(?:WithArgsExp)?\(.*?\.hKernel = (0x[0-9a-f]+),.* -> UR_RESULT_SUCCESS')
    kernels, counts, sg, per = {}, collections.Counter(), collections.Counter(), collections.Counter()
    with path.open(errors='replace') as stream:
        for line in stream:
            require(len(line) <= 65536, 'UR trace line cap')
            require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|FAIL ',line), 'runtime/fixture error in trace')
            if m := create.search(line): kernels[m[2]] = m[1]
            if m := launch.search(line):
                require(m[1] in kernels, 'launched kernel has no observed create')
                name = kernels[m[1]]; counts[name] += 1
                if m := re.search(r'try_iq3xxs_subgroupILj(16|32)ELb([01])',name):
                    sg[f'SG{m[1]}_GU{m[2]}'] += 1
                if m := re.search(r'parallel_iq_launchILi(\d+)ELb([01])',name):
                    per[f'type{m[1]}_GU{m[2]}'] += 1
    requested = selector in ('sg16','sg32')
    expected_sg = {f'SG{selector[2:]}_GU0':20, f'SG{selector[2:]}_GU1':10+(mode=='full')} if requested else {}
    expected_per = ({**{f'type{t}_GU1':1+(mode=='full') for t in (21,22,23)},
                     **{f'type{t}_GU0':2+(mode=='full') for t in (20,42)}} if parallel=='1' else {})
    require(dict(sg)==expected_sg, 'actual SG launches missing/wrong: fallback not accepted')
    require(dict(per)==expected_per, 'five-role mixed selector launch mismatch')
    require(sum(counts.values()) == 3*(75 if mode=='full' else 69)+6+int(requested), 'missing/extra kernel launches or unparsed trace')
    return dict(kernels=dict(counts), subgroup=dict(sg), parallel=dict(per),
                total_launches=sum(counts.values()), trace_sha256=sha(path), trace_bytes=path.stat().st_size,
                fallback_accepted=False)


class KernelProgressProbe(StackProbe):
    """Bounded UR completion-line observer; warning/poll spam is never progress."""
    def __init__(self,*args,stderr,wall,**kwargs):
        self.stderr=stderr;self.offset=0;self.pending=b'';self.next_check=0
        self.line_sequence=0;self.last_progress_offset=-1;self.wall=wall
        super().__init__(*args,**kwargs)
        self.record.update(progress_policy='New complete successful UR kernel create/enqueue records only; stdout ignored',
                           progress_count=0,last_progress_line=None,stderr_bytes_scanned=0,
                           quiet_seconds=60,latest_trigger_age_seconds=wall-65,
                           latest_safe_attach_age_seconds=wall-60)

    def read(self):
        # StackProbe.read treats stdout as progress; this fixture buffers cases.
        # Keep stdout transport unchanged but do not let arbitrary bytes reset quiet.
        self.original_read()

    def read_progress(self):
        now=time.monotonic()
        if now<self.next_check:return
        self.next_check=now+.05
        with self.stderr.open('rb') as stream:
            stream.seek(self.offset);consumed=0
            while True:
                data=stream.read(65536)
                if not data:break
                consumed+=len(data)
                require(consumed<=4<<20,'bounded UR progress scan per poll exceeded')
                self.offset+=len(data)
                lines=(self.pending+data).split(b'\n');self.pending=lines.pop()
                require(len(self.pending)<=16384,'UR partial-line budget')
                for line in lines:
                    self.line_sequence+=1
                    require(len(line)<=16384,'UR complete-line budget')
                    if not re.search(rb'<--- ur(?:KernelCreate|EnqueueKernelLaunch(?:WithArgsExp)?)\(.* -> UR_RESULT_SUCCESS\s*$',line):
                        continue
                    # Offset and local line sequence are strictly increasing. UR
                    # has no native phase seq; never invent one from warning lines.
                    require(self.line_sequence>self.record.get('last_progress_sequence',0) and
                            self.offset>=self.last_progress_offset,'nonmonotonic UR progress')
                    self.last_progress_offset=self.offset;self.last_progress=now
                    self.record['progress_count']+=1
                    self.record['last_progress_sequence']=self.line_sequence
                    self.record['last_progress_line']=line.decode(errors='replace')[:512]
        self.record['stderr_bytes_scanned']=self.offset

    def poll(self):
        self.original_poll()
        if self.io.exit_watch.poll(0):return
        self.read_progress()
        if self.attempted or self.capture_active:return
        now=time.monotonic();age=now-self.entry['spawn_begin_monotonic']
        quiet=now-self.last_progress
        if quiet>=60 or age>=self.wall-65:
            self.requests_stopped=True;self.record['requests_stopped']=True
            require(age<self.wall-60,'safe attach reserve missed; diagnostic incomplete, no late attach')
            self.capture_now('semantic_quiet' if quiet>=60 else 'deadline_reserve',silence_seconds=quiet)
            raise StallCaptured('one owned stack recorded; no next arm allowed')


def main():
    require(len(sys.argv)==2 and sys.argv[1] in ('quick','full'), 'usage: controller.py quick|full')
    mode=sys.argv[1]
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    proposal=A/'iq3xxs-subgroup-materializer-v2'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for rel,pin in json.loads((HERE/'pins.json').read_text()).items():
            require(sha(P/rel)==pin,'controller dependency changed: '+rel)
        bp=A/'iq3xxs-subgroup-build-v2.json'
        rp=A/'iq3xxs-subgroup-main-review-v1.json'
        build=json.loads(bp.read_text())
        # Never infer IR review from successful compilation. Main writes separate review.
        review=json.loads(rp.read_text())
        require(closed_success(build) and closed_success(review), 'build/Main review unfinished')
        require(review.get('IR_review_complete') is True and review.get('build_receipt_sha256')==sha(bp), 'explicit pinned Main IR review required')
        exe=Path(build['binary'])
        require(sha(exe)==build['binary_sha256']==review.get('binary_sha256'), 'binary review/hash mismatch')
        require(sha(Path(build['device_IR']))==build['device_IR_sha256']==review.get('device_IR_sha256'), 'device IR changed/unreviewed')
        require(review.get('sg16_supported') is True and review.get('sg32_supported') is True, 'required width not qualified; no fallback adoption')
        def pins_intact():
            for field in ('source_pins','reviewed_base_pins'):
                require(all(sha(P/n)==h for n,h in build[field].items()),field+' changed')
            require(all(sha(Path(n))==h for n,h in build['linked_archives'].items()),'linked archive changed')
            require(sha(exe)==build['binary_sha256'],'fixture changed')
            require(all(sha(P/n)==h for n,h in json.loads((HERE/'pins.json').read_text()).items()),'controller inputs changed')
        pins_intact()
        require(all(module.closed(c) for c in build['commands']),'build ownership not closed')
        host_root=B/'main-owned-stack-host-preflight-root-v2'
        host_path=host_root/'preflight-v1/record.json'
        auto_path=A/'owned-prefill-stall-host-auto-v1.json'
        host_gate=json.loads(host_path.read_text());auto_gate=json.loads(auto_path.read_text())
        require(closed_success(host_gate) and closed_success(auto_gate),'qualified host attach/detach/automatic trigger gates')
        shim=host_root/'libowned_ptracer_parent.so'
        require(sha(shim)==host_gate['inputs'][str(shim)],'qualified parent-only shim changed')
        require((DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists(),'driver/dump preflight')
        receipt=A/f'iq3xxs-subgroup-{mode}-v2.json'
        root=B/f'main-iq3xxs-subgroup-{mode}-root-v2'
        require(not receipt.exists() and not root.exists(),'fresh root/receipt required; no overwrite/retry')
        if mode=='full':
            q=json.loads((A/'iq3xxs-subgroup-quick-v2.json').read_text())
            require(closed_success(q) and not q['fresh_faults'] and q['boot_unchanged'] and not q['driver_dump_exists'],'quick arrays/routes/fault gate')
            require(q['binary_sha256']==sha(exe) and q['build_receipt_sha256']==sha(bp) and q['review_receipt_sha256']==sha(rp),'quick pins changed')
            require(all(module.closed(c) for c in q['commands']),'quick owners not closed')
            for c in q['commands']:
                for i in c['owners']:
                    now=module.identity(i['pid'])
                    require(now is None or now['start_ticks']!=i['start_ticks'],'quick owned identity still exists, including zombie')
        debug_root=B/f'main-iq3xxs-subgroup-{mode}-debug-root-v2'
        require(not debug_root.exists(),'fresh independent debugger root required')
        root.mkdir(mode=0o700);debug_root.mkdir(mode=0o700)
        total=2<<30 if mode=='full' else 1<<30
        record=dict(schema='iq3xxs-subgroup-controller-v2',mode=mode,active=True,passed=False,complete=False,
                    GPU_executed=False,commands=[],debugger_commands=[],arms=[],comparisons=[],boot_id=boot_id(),
                    binary_sha256=sha(exe),build_receipt_sha256=sha(bp),review_receipt_sha256=sha(rp),
                    recipe_sha256=sha(Path(__file__)),source_pins=build['source_pins'],
                    shim_sha256=sha(shim),host_gate_sha256=sha(host_path),automatic_host_gate_sha256=sha(auto_path),
                    debugger_root=str(debug_root),debugger_job_bytes=256<<20,stack_capture_attempted=False,
                    limits=dict(total_job_bytes=total,text_per_process_bytes=32<<20,arm_capture_bytes=192<<20,wall_per_arm_seconds=240 if mode=='quick' else 900),
                    adopted=False,performance_qualified=False,product_applied=False,
                    retention_consumer='Main strict all-arm arrays/actual routes; first distinct unresolved failure; review at completed boundary')
        def persist(): (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands
        debugger=module.Owner(debug_root/'owned',persist);record['debugger_commands']=debugger.commands
        host=dict(PATH='/usr/bin:/bin',HOME=str(root),LANG='C',LC_ALL='C',PYTHONDONTWRITEBYTECODE='1')
        def utility(label,argv):
            require(owner.active is None and debugger.active is None,'no utility while unresolved native/debugger ownership')
            e,o,err=owner.run(label,argv,host,P,wall=60,cpu=60,rss_cap=2<<30,text_cap=4<<20,total_cap=total,file_cap=8<<20)
            require(module.closed(e) and e.get('exit_code')==0,label+' failed')
            return o.read_text()
        def faults(label,cursor):
            rows=[json.loads(s) for s in utility(label,['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
            return dict(fresh_faults=[r for r in rows if FAULT.search(str(r.get('MESSAGE','')))],
                        driver_dump_exists=(DEVICE/'devcoredump').exists(),boot_unchanged=boot_id()==record['boot_id'])
        before=None;persist()
        try:
            previous_path=A/'prompt-direct-bench-full-0-v1.json'
            env=dict(json.loads(previous_path.read_text())['environment'])
            for k in ('STRATA_MEASUREMENT_ENGINE_START','STRATA_MEASUREMENT_ENGINE_EXIT'): env.pop(k,None)
            env.update(STRATA_SYCL_PROMPT_DIRECT='0',STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS='0',
                       UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                       UR_LOG_TRACING='level:info;flush:info;output:stderr',
                       UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                       UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                       UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
            require(not any('TOKEN' in k or 'API_KEY' in k or k in ('LD_PRELOAD','STRATA_TRACE') or k.startswith(('UNITRACE_','PTI_')) for k in env),'private/profiler/observer env')
            for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0',STRATA_SYCL_PREFILL_SAME_QUEUE='1',SYCL_CACHE_PERSISTENT='0',STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0').items():
                require(env[k]==v,'immutable safety flag changed: '+k)
            record['historical_failed_config_receipt_sha256']=sha(previous_path)
            before=re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            record['journal_before_cursor']=before
            for arm,parallel,selector in ARMS:
                pins_intact()
                arm_cursor=re.search(r'-- cursor: (\S+)',utility('journal-before-'+arm,['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
                arm_env=dict(env,LD_PRELOAD=str(shim),STRATA_SYCL_PARALLEL_IQ_DEQUANT=parallel,STRATA_SYCL_IQ3XXS_SUBGROUP=selector,XDG_CACHE_HOME=str(root/('cache-'+arm)))
                require(debugger.active is None and not record['stack_capture_attempted'],'no next arm after debugger attempt/unresolved ownership')
                require(sha(shim)==record['shim_sha256'],'qualified shim changed')
                item=dict(arm=arm,environment=arm_env,journal_before_cursor=arm_cursor,complete=False)
                record['arms'].append(item);record['GPU_executed']=True;persist()
                def interaction(io,entry):
                    io.proc.stdin.close()
                    item['probe']={}
                    probe=KernelProgressProbe(module,debugger,io,entry,item['probe'],persist,P,
                        stderr=root/('owned/device-'+arm+'.stderr'),
                        wall=record['limits']['wall_per_arm_seconds'],silence_seconds=60)
                    try:
                        with selectors.DefaultSelector() as ready:
                            ready.register(io.proc.stdout,selectors.EVENT_READ)
                            while not io.exit_watch.poll(0) or not io.eof:
                                try:io.poll()
                                except StallCaptured:
                                    record['stack_capture_attempted']=True
                                    item['stack_captured']=True;persist()
                                if ready.select(.02):
                                    io.read();io.pending.clear()
                        probe.next_check=0;probe.read_progress()
                    finally:
                        if probe.attempted:record['stack_capture_attempted']=True
                        probe.close();persist()
                e,out,err=owner.run('device-'+arm,[str(exe),arm,mode,str(root/arm)],arm_env,P,
                                   interaction=interaction,wall=record['limits']['wall_per_arm_seconds'],cpu=900*len(os.sched_getaffinity(0)),
                                   rss_cap=8<<30,text_cap=32<<20,total_cap=total,file_cap=192<<20)
                item['exit_code']=e.get('exit_code');item['owner_closed']=bool(module.closed(e));persist()
                if owner.active is None and debugger.active is None: item.update(faults('journal-after-'+arm,arm_cursor))
                persist()
                require(not record['stack_capture_attempted'],'stack capture consumed arm: original exit preserved, diagnostic not PASS')
                require(debugger.active is None and all(module.closed(c) for c in debugger.commands),'debugger unresolved/incomplete')
                grant=f'owned-ptracer grant pid={e["pid"]} parent={os.getpid()}\n'
                require(grant in err.read_text(),'native qualified direct-parent grant missing')
                require(module.closed(e) and e.get('exit_code')==0,'native abnormal/incomplete; original status preserved')
                require(not item['fresh_faults'] and item['boot_unchanged'] and not item['driver_dump_exists'],'fresh fault')
                cap=root/arm
                require(sum(p.stat().st_size for p in cap.iterdir() if p.is_file())<=192<<20,'capture cap')
                require('B570' in (cap/'device.txt').read_text(),'unexpected GPU')
                item['completion']=json.loads((cap/'complete.json').read_text())
                require(item['completion']['complete'] and item['completion']['cases']==(75 if mode=='full' else 69),'case completion')
                if selector in ('sg16','sg32'):
                    require((cap/'subgroup-probe.bin').exists() and not (cap/'subgroup-probe.txt').exists(),'SG declined; fallback not accepted')
                item['trace']=dict(path=str(err),bytes=err.stat().st_size,sha256=sha(err))
                try:item['routes']=routes(err,mode,parallel,selector)
                except BaseException as route_error:
                    item['route_check_error']=repr(route_error);item['fallback_accepted']=False;persist();raise
                if arm!='off':
                    report=root/('comparison-'+arm+'.json')
                    utility('compare-'+arm,['/usr/bin/python3','-B',str(proposal/'compare_captures.py'),str(root/'off'),str(cap),str(report)])
                    comparison=json.loads(report.read_text())
                    require(comparison['array_comparison']=='PASS','complete array mismatch')
                    compact=A/f'iq3xxs-subgroup-{mode}-{arm}-arrays-v2.json'
                    with compact.open('xb') as stream:stream.write(report.read_bytes())
                    record['comparisons'].append(dict(arm=arm,path=str(compact),sha256=sha(compact)))
                item['complete']=True;persist()
            pins_intact()
            require(sha(bp)==record['build_receipt_sha256'] and sha(rp)==record['review_receipt_sha256'],'review/build receipt changed')
            record.update(passed=True,complete=True)
        except BaseException as exc:
            record['error']=repr(exc);raise
        finally:
            try:
                if before and owner.active is None and debugger.active is None:record.update(faults('journal-final',before))
            except BaseException as exc:
                record['final_fault_error']=repr(exc);record.update(passed=False,complete=False);raise
            finally:
                record['active']=owner.active is not None or debugger.active is not None
                if record['stack_capture_attempted'] or not all(module.closed(c) for c in debugger.commands) or record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False,complete=False)
                record['finished_utc']=module.utc();persist()
                with receipt.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')


if __name__=='__main__':main()
