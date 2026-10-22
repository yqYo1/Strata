#!/usr/bin/env python3
"""Main-owned, finite read-only metric inventory; no inference or counters run."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import measurement_guards as guards
import measurement_owner as owner_module

P=Path(__file__).resolve().parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT=B/'b570-metric-inventory-v1'
A=P/'bench/results/2026-10-15-pr6-recovery-audit'
RESULT=A/'metric-inventory-current.json'
SOURCE='bench/sycl/metric_inventory.cpp'
SETTINGS=P/'tools/sycl/measurement_environment.json'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def git(*args):return subprocess.check_output(['git',*args],cwd=P,timeout=30)


def committed(path):
    raw=path.read_bytes();guards.require(raw==git('show','HEAD:'+str(path.relative_to(P))),'commit exact input: '+str(path))
    return json.loads(raw)


def inputs(revision):
    names=(SOURCE,'tools/sycl/metric_inventory.py','tools/sycl/measurement_owner.py',
        'tools/sycl/measurement_guards.py','tools/sycl/measurement_environment.json')
    result={}
    for name in names:
        raw=git('show',revision+':'+name)
        guards.require(raw==(P/name).read_bytes(),'source changed: '+name)
        result[name]=hashlib.sha256(raw).hexdigest()
    return result


def native_assets():
    receipt=B/'main-native-gdn-engine/record.json'
    guards.require(sha(receipt)=='a38f477670fafcd1a931be2e20d90d6d9718df75b61954b421f361af9f6881b2','selected original build receipt changed')
    d=json.loads(receipt.read_text());assets=[d['binary'],*d['fixtures'].values()]
    guards.require(d['source_commit']=='6e40afc1beddd059ee518617f6c1efce1da14aca' and len(assets)==13,'selected original generation required')
    for item in assets:guards.require(sha(item['path'])==item['sha256'],'retained fallback artifact changed')
    return dict(source_commit=d['source_commit'],receipt_sha256=sha(receipt),engine=d['binary'],artifacts=13)


def admission(record,binary=False):
    guards.require(inputs(record['source_commit'])==record['inputs'],'inventory source changed')
    guards.require(native_assets()==record['retained_engine'],'original engine generation changed')
    manifest=record['dependencies'];guards.require(sha(manifest['path'])==manifest['sha256'],'dependency manifest changed')
    rows=json.loads(Path(manifest['path']).read_text());guards.require(len(rows)==manifest['files'],'dependency count changed')
    for row in rows:guards.require(sha(row['path'])==row['sha256'],'compiler/dependency changed: '+row['path'])
    for path,digest in record['runtime_pins'].items():guards.require(sha(path)==digest,'runtime asset changed: '+path)
    if binary:guards.require(sha(record['binary']['path'])==record['binary']['sha256'],'inventory executable changed')


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'single fresh attempt;inspect existing owner first')
    revision=git('rev-parse','HEAD').decode().strip();pins=inputs(revision)
    settings=committed(SETTINGS);component=committed(A/'decode-query-profile-component-current.json')
    guards.require(component['passed'] and component['complete'] and component['ownership_closed'] and not component['active'],
        'qualified original runtime component required')
    runtime={str(Path(name).resolve()):item['sha256'] for name,item in settings['runtime_elfs'].items()}
    for name,item in component['runtime_asset_delta'].items():
        guards.require(settings['runtime_elfs'][name]['sha256']==item['qualified_sha256'],'original runtime identity changed')
        runtime[str(Path(name).resolve())]=item['candidate_sha256']
    for name in ('/usr/lib/x86_64-linux-gnu/libze_validation_layer.so.1.32.0',
                 '/usr/lib/x86_64-linux-gnu/libigdmd.so.1.16.188'):
        runtime[str(Path(name).resolve())]=sha(name)
    for path,digest in runtime.items():guards.require(sha(path)==digest,'runtime input mismatch: '+path)
    origin=committed(A/'cpu-profiler-current.json')['unitrace_indexed_graph']['compiler_inputs']
    guards.require(sha(origin['path'])==origin['sha256'],'qualified compiler inventory changed')
    rows=json.loads(Path(origin['path']).read_text());guards.require(len(rows)==origin['files'],'SDK inventory count changed')
    for row in rows:guards.require(sha(row['path'])==row['sha256'],'SDK asset changed')
    paths={Path(row['path']) for row in rows}
    paths.update(Path('/usr/include/level_zero')/name for name in ('ze_api.h','zet_api.h'))
    compiler=json.loads((B/'main-native-gdn-engine/record.json').read_text())['compiler']
    guards.require(sha(compiler['path'])==compiler['sha256'],'compiler changed')
    for name in ('crtbeginS.o','crtendS.o','libstdc++.so','libgcc_s.so'):
        path=Path(subprocess.check_output([compiler['path'],'-print-file-name='+name],text=True,timeout=15).strip()).resolve()
        guards.require(path.is_file(),'link dependency missing');paths.add(path)
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir()
    manifest=ROOT/'compiler-inputs.json';write(manifest,[dict(path=str(path),sha256=sha(path)) for path in sorted(paths)])
    install=Path(compiler['path']).parents[1]
    record=dict(stage='prepared',source_commit=revision,inputs=pins,compiler=compiler,retained_engine=native_assets(),
        dependencies=dict(path=str(manifest),sha256=sha(manifest),files=len(paths)),runtime_pins=runtime,
        build_command=[compiler['path'],'-std=c++20','-O2','-g0','-Wall','-Wextra','-Werror',str(P/SOURCE),'-lze_loader','-ldl','-o',str(ROOT/'inventory')],
        build_environment=dict(PATH=str(install/'bin')+':/usr/bin:/bin',LANG='C',LC_ALL='C',TMPDIR=str(ROOT/'tmp'),
            LD_LIBRARY_PATH=str(install/'lib'),PYTHONDONTWRITEBYTECODE='1'),
        plan=dict(scope='Read-only Level Zero inventory for unique Intel E20C root device;no context,allocation,queue,command list,metric activation/query/streamer,Sysman or inference.',
            source_contract='LevelZero v1.32.0 d3b3efbdaa27a0aef9a9812cc8fa4260557c1e7c headers/loader logging+validation docs;installed ELF identity separately pinned. PTI4b6d742 chapters/metrics_collection/LevelZero.md API semantics only;no copied implementation.',
            order='One fresh ZET_ENABLE_METRICS=0 device-only process,then one separate ZET_ENABLE_METRICS=1 complete inventory process. All environment fixed beforeexec;standard safety settings preserved. No retries or permission/system changes.',
            gates='Normal finite owner closure,no fresh fault,same boot and unique E20C. Require baseline success before enabling metrics. Trace/parameter validation enabled;verify actual validation library mapped. Match all loaded absolute ELF identities to predeclared pins. Reconcile begin/return API calls,all group/member counts and one exact summary. API failure remains distinct from successful empty inventory;no counter-collection capability,physical traffic,bottleneck or speed claim.',
            retention='Keep compact result/API errors/group summary and only current-needed member catalog outsideGit for one source/semantic-selection consumer. Retire normal traces and one inventory binary after that consumer;source/recipe must remain recoverable. Original13engine/fixture artifacts andlegacy provenance holds remain unchanged.'),
        budgets=dict(build_wall_s=90,run_wall_s=60,cpu_s=120,rss_bytes=2<<30,text_bytes=16<<20,file_bytes=32<<20,total_bytes=96<<20),
        driver_before=guards.driver_guard(),existing_permissions={name:Path(name).read_text().strip() for name in
            ('/proc/sys/dev/xe/observation_paranoid','/proc/sys/kernel/perf_event_paranoid')},jobs=[],
        passed=False,complete=False,active=False,ownership_closed=True,inference_unchanged=True)
    for name,enabled,mode in (('device','0','--device-only'),('metrics','1','--metrics')):
        folder=ROOT/name;folder.mkdir();(folder/'tmp').mkdir();(folder/'cache').mkdir()
        env=dict(settings['environment'],TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),
            ZET_ENABLE_METRICS=enabled,ZEL_ENABLE_LOADER_LOGGING='1',ZEL_LOADER_LOG_CONSOLE='1',
            ZEL_LOADER_LOGGING_LEVEL='trace',ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='1',
            ZE_ENABLE_VALIDATION_LAYER='1',ZE_ENABLE_PARAMETER_VALIDATION='1')
        record['jobs'].append(dict(name=name,command=[str(ROOT/'inventory'),mode],environment=env))
    admission(record);write(ROOT/'plan.json',record);write(RESULT,record)
    print('Prepared read-only inventory source,recipe and2immutable child environments;commit beforebuild.',flush=True)


def execute(action):
    record=committed(RESULT);guards.require(record['stage']==('prepared' if action=='build' else 'built'),'inspect current attempt;no retry')
    admission(record,action=='run');root=ROOT/action;guards.require(not root.exists(),'existing owner receipt;no duplicate')
    root.mkdir();owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(root/'record.json',dict(record,owned_commands=owner.commands if owner else []))
    owner=owner_module.Owner(root/'owned',persist)
    record.update(stage=action+'ing',passed=False,complete=False,ownership_closed=False);persist()
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
    try:
        jobs=[dict(name='compiler',command=record['build_command'],environment=record['build_environment'])] if action=='build' else record['jobs']
        record['observations']=[]
        for job in jobs:
            admission(record,action=='run');b=record['budgets']
            print('Inventory '+action+' '+job['name'],flush=True)
            entry,out,err=owner.run(job['name'],job['command'],job['environment'],P,
                wall=b['build_wall_s'] if action=='build' else b['run_wall_s'],cpu=b['cpu_s'],rss_cap=b['rss_bytes'],
                text_cap=b['text_bytes'],file_cap=b['file_bytes'],total_cap=b['total_bytes'])
            guards.require(owner_module.closed(entry),'inventory process did not close normally')
            if action=='build':
                guards.require(entry['exit_code']==0,'inventory build failed')
                record['binary']=dict(path=str(ROOT/'inventory'),bytes=(ROOT/'inventory').stat().st_size,sha256=sha(ROOT/'inventory'))
                continue
            events=[json.loads(line) for line in out.read_text().splitlines()]
            summaries=[e for e in events if e['kind']=='summary'];guards.require(len(summaries)==1,'one complete status summary required')
            summary=summaries[0];begins=[e['api'] for e in events if e['kind']=='begin'];returns=[e for e in events if e['kind']=='return']
            guards.require(begins==[e['api'] for e in returns],'incomplete/reordered API calls')
            failures=[e for e in returns if e['result']!=0]
            mapped={str(Path(e['path']).resolve()) for e in events if e['kind']=='elf'}
            guards.require(mapped and mapped<=set(record['runtime_pins']),'unqualified loaded ELF: '+str(mapped-set(record['runtime_pins'])))
            guards.require(any('libze_validation_layer.so' in path for path in mapped),'validation library was not mapped')
            groups=[e for e in events if e['kind']=='group'];members=[e for e in events if e['kind']=='metric']
            if summary['status']=='success':
                guards.require(entry['exit_code']==0 and not failures and summary['complete'] and summary['GPU_commands']==0
                    and len(groups)==summary['groups'] and len(members)==summary['metrics'],'incomplete successful inventory')
                guards.require({g['group'] for g in groups}==set(range(len(groups))) and all(
                    len([m for m in members if m['group']==g['group']])==g['metrics'] for g in groups),'group/member conservation failed')
            else:
                guards.require(job['name']=='metrics' and entry['exit_code']==3 and summary['status']=='api_error'
                    and not summary['complete'] and len(failures)==1 and failures[0]['api']=='zetMetricGroupGet/count'
                    and failures[0]['result']==summary['result'],'unexpected API failure')
            record['observations'].append(dict(name=job['name'],exit_code=entry['exit_code'],owner_closed=True,
                summary=summary,API_calls=len(returns),API_failures=failures,mapped_ELFs=sorted(mapped),
                groups=groups,metric_members=len(members),stdout=dict(path=str(out),sha256=sha(out),bytes=out.stat().st_size),
                stderr_sha256=sha(err)))
        record.update(stage='built' if action=='build' else 'closed',passed=True,complete=True)
        if action=='run':record['metric_enumeration_succeeded']=record['observations'][-1]['summary']['status']=='success'
    except BaseException as error:record.update(stage='failed',error=repr(error),passed=False)
    finally:
        record['ownership_closed']=owner.active is None and all(owner_module.closed(c) for c in owner.commands)
        try:
            journal=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            record['fresh_faults']=[x for x in journal.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
            record['driver_after']=guards.driver_guard();record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except Exception as error:record['fault_probe_error']=repr(error)
        record['passed']=bool(record['passed'] and record['ownership_closed'] and record.get('fresh_faults')==[]
            and record.get('boot_unchanged') and 'fault_probe_error' not in record)
        persist();write(RESULT,record)
    print(json.dumps({k:record.get(k) for k in ('stage','passed','complete','ownership_closed','metric_enumeration_succeeded','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build','run'),'use prepare|build|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        sys.exit(prepare() if sys.argv[1]=='prepare' else execute(sys.argv[1]))
