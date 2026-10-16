#!/usr/bin/env python3
"""Main one-attempt API-only metric-pool diagnostic, with fixed private logging.
No kernels, query begin, command queues/lists, permissions or driver changes.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEV = Path('/sys/bus/pci/devices/0000:05:00.0')
BOOT = 'cfda0b6a-7c13-4fe2-bdf5-8f9202770840'
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
HOST = dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def pin(p):
    p=Path(p)
    return dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))


def check_pin(p,v):
    p=Path(p)
    assert str(p.resolve())==v['realpath'] and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],str(p)


def main():
    root=B/'main-metric-pool-diagnostic-v2';target=A/'own-metric-pool-diagnostic-v2.json'
    assert not root.exists() and not target.exists()
    owner_source=A.parents[2]/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert sha(owner_source)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    base_path=A/'own-vllm-resource-timeline-v3/manifest.json'
    assert sha(base_path)=='61da9469e3f975c72d770beee77d45118c988eb6d3d82b6dbe1b568cd9ea3f1c'
    build_path=A/'own-metric-pool-build-v2.json'
    assert sha(build_path)=='3a90062a96dc5b0d55319d7e08964ec18c71ab43c048a1e84f848bcbfa26b7c8'
    build=json.loads(build_path.read_text());assert build['passed'] and build['complete'] and build['ownership_closed'] and not build['GPU_executed']
    binary=build['binary'];check_pin(binary['path'],binary)
    assert binary['sha256']=='fbadcd96355a74d38649fb73fc25c64334dc39f4310da237fcd98ba9457a1974'
    review_path=A/'own-metric-pool-validation-layer-review-v1.json'
    review=json.loads(review_path.read_text());assert review['approved'] and review['sha256']=='5b5ead80d98959d0f0a27a8981bb911c354720d39a4aa2125dc18266ad67d970'
    allowed=dict(json.loads(base_path.read_text())['elf_pins'])
    assert review['realpath'] not in allowed
    allowed[review['realpath']]={k:review[k] for k in ('path','realpath','bytes','sha256')}
    allowed[binary['realpath']]=binary
    frozen={**allowed}
    for v in [build['actual_source'],build['config'],build['result_names'],build['main_patch'],*build['inputs'],*build['source_package'].values(),pin(review_path),pin(build_path),pin(Path(__file__))]:
        if v['realpath'] in frozen:
            old=frozen[v['realpath']]
            assert all(old[k]==v[k] for k in ('realpath','bytes','sha256'))
        frozen[v['realpath']]=v
    spec=importlib.util.spec_from_file_location('metric_pool_native_owner',owner_source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
        record=dict(schema='strata-metric-pool-api-diagnostic-v2',passed=False,complete=False,
            active=False,commands=[],boot_id=BOOT,native_executed=False,GPU_commands_authored=False,
            collection_qualified=False,numerical_result=False,build=pin(build_path),
            validation_layer_review=pin(review_path),source=build['actual_source'],binary=binary,
            config=build['config'],runtime_elf_pins=allowed,native_rows=[],library_stdout=[],snapshots=[],
            budgets=dict(native_wall_seconds=90,whole_child_cpu_seconds=120,raw_text_bytes=64<<20,private_bytes=128<<20),
            retention=dict(owner='Main',consumer='Distinguish pool/API rejection and MetricsLibrary logs',
                next_review='At actual diagnosis extraction; keep one representative per distinct unresolved cause'))
        owner=None
        def persist():
            record['active']=bool(owner and owner.active is not None)
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=m.Owner(root/'owned',persist);record['commands']=owner.commands
        def run(label,argv,env=HOST,interaction=None,wall=30):
            c,so,se=owner.run(label,argv,env,A.parents[2],interaction=interaction,wall=wall,cpu=120,
                rss_cap=2<<30,text_cap=64<<20,file_cap=64<<20,total_cap=128<<20)
            assert m.closed(c),(label,c)
            return c,so,se
        def journal(label,args):
            c,so,se=run(label,['/usr/bin/journalctl',*args]);assert c['exit_code']==0,(c,se.read_text())
            return so.read_text()
        def freeze():
            assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
            assert DEV.joinpath('driver').resolve().name=='xe'
            assert not list(DEV.glob('devcoredump*'))
            for p,v in frozen.items():check_pin(p,v)
        cursor=None
        native_env={**HOST,'ZET_ENABLE_METRICS':'1','ZE_FLAT_DEVICE_HIERARCHY':'FLAT',
            'ZE_ENABLE_VALIDATION_LAYER':'1','NEOReadDebugKeys':'1','EnableDirectSubmission':'0'}
        record['native_environment']=native_env
        def snapshot(io,entry,phase):
            ident=m.identity(io.proc.pid);assert ident and ident['start_ticks']==entry['direct_identity']['start_ticks']
            proc=Path('/proc')/str(io.proc.pid)
            assert str(proc.joinpath('exe').resolve())==binary['realpath']
            effective=dict(v.decode().split('=',1) for v in proc.joinpath('environ').read_bytes().split(b'\0') if v)
            assert effective==native_env
            maps=[]
            for line in proc.joinpath('maps').read_text().splitlines():
                f=line.split(None,5)
                if len(f)!=6 or 'x' not in f[1] or not f[5].startswith('/'):continue
                assert not f[5].endswith(' (deleted)')
                p=Path(f[5]).resolve(strict=True);assert str(p) in allowed,'unreviewed actual ELF: '+str(p)
                v=allowed[str(p)];check_pin(p,v);st=p.stat();major,minor=[int(x,16) for x in f[3].split(':')]
                assert st.st_ino==int(f[4]) and os.major(st.st_dev)==major and os.minor(st.st_dev)==minor
                maps.append(dict(mapping=line,pin=v))
            seen={v['pin']['realpath'] for v in maps}
            assert binary['realpath'] in seen and '/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0' in seen
            if phase=='post_probe' and any(v.get('api')=='zetMetricQueryPoolCreate' and v.get('kind')=='result' for v in record['native_rows']):
                assert '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395' in seen
            status={}
            for line in proc.joinpath('status').read_text().splitlines():
                if line.split(':',1)[0] in ('Uid','Gid','Groups','CapEff','CapBnd','NoNewPrivs'):status[line.split(':',1)[0]]=line.split(':',1)[1].strip()
            after=m.identity(io.proc.pid);assert after and after['start_ticks']==ident['start_ticks']
            check_pin(build['config']['path'],build['config'])
            record['snapshots'].append(dict(phase=phase,identity=ident,effective_environment=effective,mappings=maps,process_identity=status,
                scope='File-backed executable mappings at this hold; unloaded or anonymous code not attested'))
            io.send(b'X')
        def interaction(io,entry):
            holds=[];last=None
            for _ in range(2048):
                line=io.line(seconds=30)
                try:row=json.loads(line)
                except json.JSONDecodeError:
                    record['library_stdout'].append(line);continue
                if not isinstance(row,dict) or 'kind' not in row:
                    record['library_stdout'].append(line);continue
                record['native_rows'].append(row);kind=row['kind']
                if kind=='hold':
                    phase=row['phase'];assert phase==('pre_init' if not holds else 'post_probe') and len(holds)<2
                    snapshot(io,entry,phase);holds.append(phase)
                if kind=='process_summary':last=row;break
            assert last and holds==['pre_init','post_probe']
            record['process_summary']=last
            io.exit(seconds=5)
        try:
            freeze()
            record['driver_snapshot']=dict(driver=str(DEV.joinpath('driver').resolve()),
                vendor=DEV.joinpath('vendor').read_text().strip(),device=DEV.joinpath('device').read_text().strip(),
                current_link_speed=DEV.joinpath('current_link_speed').read_text().strip(),
                current_link_width=DEV.joinpath('current_link_width').read_text().strip())
            record['read_only_policies']={}
            for p in ['/proc/sys/dev/xe/observation_paranoid','/proc/sys/dev/i915/perf_stream_paranoid','/proc/sys/kernel/perf_event_paranoid']:
                try:record['read_only_policies'][p]=dict(value=Path(p).read_text().strip())
                except OSError as e:record['read_only_policies'][p]=dict(error=type(e).__name__,errno=e.errno)
            before=journal('journal-before',['-k','-b','-n','0','--show-cursor','--no-pager'])
            match=re.search(r'-- cursor: (\S+)',before);assert match;cursor=match[1];record['journal_before_cursor']=cursor
            record['native_executed']=True
            c,so,se=run('native-pool-probe',[binary['path'],'--main-owned-hold'],native_env,interaction,wall=90)
            record['native_exit_code']=c['exit_code'];record['library_stderr']=se.read_text()
            summary=record['process_summary'];assert c['exit_code']==summary['exit_code']
            assert c['exit_code'] in (0,125) and summary['pre_init_hold_status']=='ACK' and summary['post_probe_hold_status']=='ACK'
            assert summary['cleanup_complete']
            record['diagnostic_completed']=True
            record['status']='API_POOL_CREATED' if c['exit_code']==0 else 'API_POOL_REJECTED'
            record['passed']=c['exit_code']==0
            freeze()
        except BaseException as error:
            import traceback
            record.update(error=repr(error),traceback=traceback.format_exc(),status='INCOMPLETE')
        finally:
            try:
                assert owner.active is None
                if cursor:
                    end=journal('journal-end',['-k','-b','-n','0','--show-cursor','--no-pager'])
                    record['journal_after_cursor']=re.search(r'-- cursor: (\S+)',end)[1]
                    raw=journal('journal-after',['-k','-b','--after-cursor',cursor,'--no-pager','-o','json'])
                    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
                    assert all(v.get('_BOOT_ID')==BOOT.replace('-','') for v in rows)
                    record['fresh_faults']=[v for v in rows if FAULT.search(v.get('MESSAGE',''))]
                    record['fault_filter']=FAULT.pattern
                record['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
                record['driver_dump_entries']=[str(p) for p in DEV.glob('devcoredump*')]
                record['ownership_closed']=all(m.closed(c) for c in owner.commands)
                for c in owner.commands:
                    for old in c['owners']+[c['direct_identity']]:
                        now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
                record['complete']=bool(record.get('diagnostic_completed') and record['ownership_closed'] and
                    record['boot_unchanged'] and not record['driver_dump_entries'] and 'fresh_faults' in record and not record['fresh_faults'] and not record.get('error'))
            except BaseException as error:record['closure_error']=repr(error)
            persist();target.write_bytes((root/'record.json').read_bytes())
        print(json.dumps(dict(receipt=pin(target),status=record.get('status'),complete=record['complete'],
            passed=record['passed'],exit_code=record.get('native_exit_code'),faults=len(record.get('fresh_faults',[])),
            collection_qualified=False,error=record.get('error'),closure_error=record.get('closure_error'))))
        return 0 if record['complete'] else 1


if __name__=='__main__':
    sys.exit(main())
