#!/usr/bin/env python3
"""Main serialized copy recheck, with read-only PCIe sysfs sampling.
Reuses an unchanged, previously qualified copy-only binary. No model/kernels.
"""
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time

sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
DEV=Path('/sys/bus/pci/devices/0000:05:00.0')
BOOT='cfda0b6a-7c13-4fe2-bdf5-8f9202770840'
FAULT=re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    root=B/'main-host-transfer-link-control-v3';target=A/'own-host-transfer-link-control-v3.json'
    assert not root.exists() and not target.exists()
    prior_path=A/'own-host-transfer-component-v1.json';prior=json.loads(prior_path.read_text())
    assert prior['passed'] and prior['complete'] and not prior['active'] and prior['boot_id']==BOOT
    exe=Path(prior['binary']);assert sha(exe)==prior['binary_sha256']=='2fccadda6686349ad4dbc081511d99f4f6cb85683055797cc21bb28e15f9894c'
    source=A.parents[2]/'tools/sycl/host_transfer_probe.cpp';assert sha(source)==prior['source_pins']['tools/sycl/host_transfer_probe.cpp']
    owner_path=A.parents[2]/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert sha(owner_path)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('host_transfer_link_owner',owner_path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    # Identical qualified runtime files. No new helper or excluded implementation.
    for v in prior['runtime_library_files']:assert sha(v['path'])==v['sha256']
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
        record=dict(schema='strata-host-transfer-link-control-v3',passed=False,complete=False,active=False,
            GPU_executed=False,commands=[],boot_id=BOOT,prior_receipt_sha256=sha(prior_path),
            source=str(source),source_sha256=sha(source),binary=str(exe),binary_sha256=sha(exe),
            runtime_library_files=prior['runtime_library_files'],cases=[],link_samples=[],
            reason='Controlled repeat of the preceding4.31/3.93GB/s copy recheck with only the external sysfs sampler disabled, to test observer interference; same binary, environment values, sizes, repetitions and process count.',
            scope='Ordinary RAM H2D/D2H capacity proxy, same320MiB full-residual size and qualified copy-only binary. External sysfs sampler disabled in this control; not a model throughput or implementation adoption.',
            retention=dict(owner='Main',consumer='Resolve sysfs versus actual copy-capacity discrepancy',
                next_review='After exact samples/checks/link-state summary extraction'))
        owner=None
        def persist():
            record['active']=bool(owner and owner.active is not None)
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=m.Owner(root/'owned',persist);record['commands']=owner.commands
        host=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
        env={**host,'LD_LIBRARY_PATH':str(L/'lib')+':/opt/intel/oneapi/mkl/2026.1/lib',
            'ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','UR_ADAPTERS_FORCE_LOAD':str(L/'lib/libur_adapter_level_zero_v2.so.0'),
            'SYCL_CACHE_PERSISTENT':'0','NEOReadDebugKeys':'1','EnableDirectSubmission':'0',
            'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD':'1','STRATA_SYCL_VERIFY_NATIVE_CAPTURE':'0',
            'STRATA_SYCL_PREFILL_SAME_QUEUE':'1','XDG_CACHE_HOME':str(root/'cache')}
        record['native_base_environment']=env
        def freeze():
            assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
            assert DEV.joinpath('driver').resolve().name=='xe' and not list(DEV.glob('devcoredump*'))
            assert sha(exe)==record['binary_sha256'] and sha(source)==record['source_sha256']
            for v in record['runtime_library_files']:assert sha(v['path'])==v['sha256']
        def run(label,argv,current=host,wall=30):
            c,out,err=owner.run(label,argv,current,A.parents[2],wall=wall,cpu=120,rss_cap=4<<30,
                text_cap=64<<20,file_cap=64<<20,total_cap=128<<20)
            assert m.closed(c) and c['exit_code']==0,(label,c,err.read_text()[-2048:])
            return c,out,err
        def journal(label,args):
            _,out,_=run(label,['/usr/bin/journalctl',*args]);return out.read_text()
        stop=threading.Event();sampling_errors=[]
        def sample():
            while not stop.is_set():
                try:
                    now=time.monotonic_ns()
                    values={n:DEV.joinpath(n).read_text().strip() for n in ('current_link_speed','current_link_width','max_link_speed','max_link_width')}
                    record['link_samples'].append(dict(monotonic_ns=now,**values))
                    assert len(record['link_samples'])<=12000
                except BaseException as e:sampling_errors.append(repr(e));return
                stop.wait(.02)
        sampler=None;cursor=None
        try:
            freeze();before=journal('journal-before',['-k','-b','-n','0','--show-cursor','--no-pager'])
            cursor=re.search(r'-- cursor: (\S+)',before)[1];record['journal_before_cursor']=cursor
            record['external_link_sampler_enabled']=False  # Controlled removal, no sampling thread.
            record['control_for']=dict(path=str(A/'own-host-transfer-link-recheck-v2.json'),sha256=sha(A/'own-host-transfer-link-recheck-v2.json'))
            for index,(size,repeats,traced) in enumerate([(4<<20,1,True),*[(320<<20,3,False)]*3]):
                freeze();current=dict(env)
                if traced:current.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                    UR_LOG_TRACING='level:info;flush:info;output:stderr',
                    UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                    UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                    UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
                record['GPU_executed']=True;begin=time.monotonic_ns()
                label='copy-'+str(index)
                c,out,err=run(label,['/usr/bin/taskset','-c','0',str(exe),'--host','ordinary','--bytes',str(size),'--repeats',str(repeats)],current,wall=90)
                finish=time.monotonic_ns();stdout=out.read_text();stderr=err.read_text()
                assert 'DEVICE name=Intel(R) Arc(TM) B570 Graphics driver=1.17.39395+14' in stdout
                assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|FAIL |retirement failed|Strata SYCL cleanup:',stdout+stderr)
                if traced:assert 'UR_RESULT_SUCCESS' in stderr
                assert 'DONE host=ordinary repeats='+str(repeats)+' warmups=1 scope=transfer_component_not_model' in stdout
                rows=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in stdout.splitlines() if line.startswith('COPY ')]
                checks=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in stdout.splitlines() if line.startswith('CHECK ')]
                assert len(rows)==len(checks)==2*(repeats+1)
                for row,check in zip(rows,checks):
                    assert row['direction']==check['direction'] and row['repetition']==check['repetition']
                    assert check['full']==check['guards']==check['source']=='PASS'
                    assert int(row['bytes'])==size and row['host']=='ordinary'
                    for key in list(row):
                        if key.endswith('_s') or key=='total_GBps':row[key]=float(row[key]);assert math.isfinite(row[key]) and row[key]>=0
                    assert row['total_wall_s']>0 and abs(row['total_GBps']-size/1e9/row['total_wall_s'])<.001
                record['cases'].append(dict(label=label,bytes=size,repeats=repeats,traced=traced,
                    start_monotonic_ns=begin,end_monotonic_ns=finish,environment=current,samples=rows,checks=checks,
                    stdout=stdout,stderr_sha256=sha(err)))
                persist()
            assert not sampling_errors
            record['summary']={d:dict(independent_processes=3,timed_samples=9,
                sample_GBps=[v['total_GBps'] for case in record['cases'] if not case['traced'] for v in case['samples'] if v['direction']==d and v['warmup']=='0'],
                median_of_process_medians_GBps=statistics.median(statistics.median(v['total_GBps'] for v in case['samples'] if v['direction']==d and v['warmup']=='0') for case in record['cases'] if not case['traced'])) for d in ('H2D','D2H')}
            record['link_states']=[json.loads(v) for v in sorted({json.dumps({k:v for k,v in s.items() if k!='monotonic_ns'},sort_keys=True) for s in record['link_samples']})]
            freeze();record.update(passed=True,complete=True)
        except BaseException as e:
            import traceback
            record.update(error=repr(e),traceback=traceback.format_exc())
        finally:
            stop.set()
            if sampler:sampler.join(timeout=2)
            record['sampling_errors']=sampling_errors
            try:
                assert owner.active is None and (sampler is None or not sampler.is_alive())
                if cursor:
                    end=journal('journal-end',['-k','-b','-n','0','--show-cursor','--no-pager'])
                    record['journal_after_cursor']=re.search(r'-- cursor: (\S+)',end)[1]
                    raw=journal('journal-after',['-k','-b','--after-cursor',cursor,'--no-pager','-o','json'])
                    entries=[json.loads(line) for line in raw.splitlines() if line.strip()]
                    assert all(v.get('_BOOT_ID')==BOOT.replace('-','') for v in entries)
                    record['fresh_faults']=[v for v in entries if FAULT.search(v.get('MESSAGE',''))]
                record['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
                record['driver_dump_entries']=[str(p) for p in DEV.glob('devcoredump*')]
                record['ownership_closed']=all(m.closed(c) for c in owner.commands)
                for c in owner.commands:
                    for old in c['owners']+[c['direct_identity']]:
                        now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
                if not record['ownership_closed'] or not record['boot_unchanged'] or record['driver_dump_entries'] or record.get('fresh_faults') or 'fresh_faults' not in record or record.get('error'):
                    record.update(passed=False,complete=False)
            except BaseException as e:record.update(closure_error=repr(e),passed=False,complete=False)
            persist();target.write_bytes((root/'record.json').read_bytes())
        print(json.dumps(dict(receipt=str(target),passed=record['passed'],samples=record.get('summary'),link_states=record.get('link_states'),error=record.get('error'),closure_error=record.get('closure_error'))))
        return 0 if record['passed'] else 1


if __name__=='__main__':sys.exit(main())
