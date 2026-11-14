#!/usr/bin/env python3
"""Build the host-aggregation-only PTI candidate; never load it or use a GPU."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode=True
import check_collector_aggregation as host
import collector_aggregation as proposal
import unitrace_launch as launch
import unitrace_probe as probe

tool=host.tool
P,B,RESULT=tool.P,tool.B,tool.RESULT
ROOT=B/'unitrace-indexed-collector-v1'
KEY='unitrace_indexed_collector'


def canonical(value):return tool.sha(json.dumps(value,sort_keys=True,separators=(',',':')).encode())
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def inputs(revision):
    rows=host.pins(revision)
    rows.update(probe.inputs(revision))
    for name in ('tools/sycl/build_indexed_collector.py','tools/sycl/unitrace_launch.py','tools/sycl/unitrace_probe.py'):
        data=tool.git(P,'show',revision+':'+name)
        tool.guards.require(data==(P/name).read_bytes(),'commit exact collector controller first')
        rows[name]=tool.sha(data)
    return rows


def generated():
    original=tool.git(tool.PTI,'show',proposal.REVISION+':'+proposal.HEADER)
    result={'ze_collector.h':proposal.overlay(original)}
    for name in ('tracer.cc','tracer.h'):
        result[name]=tool.git(tool.PTI,'show',proposal.REVISION+':tools/unitrace/src/'+name)
    return result


def command(container):
    result=list(container['unitrace_build']['commands']['collector'])
    result[result.index(str(tool.U/'src/tracer.cc'))]=str(ROOT/'tracer.cc')
    result[-1]=str(ROOT/'libunitrace_tool.so')
    result[1:1]=['-I',str(ROOT)]
    return result


def admission(record,container,binary=False):
    tool.guards.require(inputs(record['source_commit'])==record['input_sha256'],'collector source changed')
    closed=container['unitrace_aggregation_host_v2']
    tool.guards.require(closed['passed'] and closed['complete'] and closed['ownership_closed'] and not closed['active']
        and closed['performance_gate_passed'] and canonical(closed)==record['host_qualification_sha256'],
        'closed80-case host equivalence/performance required')
    host.admission(closed,True)
    tool.guards.require(probe.pin_assets(container)==record['original_assets'],'original collector/runtime/native assets changed')
    tool.guards.require(command(container)==record['command'],'collector compile command changed')
    for name,data in generated().items():
        tool.guards.require((ROOT/name).read_bytes()==data and tool.sha(data)==record['generated_sha256'][name],
            'generated collector source changed: '+name)
    manifest=record.get('compiler_inputs')
    if manifest:
        tool.guards.require(host.digest(manifest['path'])==manifest['sha256'],'compiler manifest changed')
        rows=json.loads(Path(manifest['path']).read_text())
        tool.guards.require(len(rows)==manifest['files'],'compiler dependency count changed')
        for row in rows:tool.guards.require(host.digest(row['path'])==row['sha256'],'compiler/dependency changed: '+row['path'])
    if binary:
        for artifact in record['artifacts'].values():
            tool.guards.require(host.digest(artifact['path'])==artifact['sha256'],'collector/launcher bytes changed')


def prepare(container):
    tool.guards.require(KEY not in container and not ROOT.exists(),'fresh tooling build only; inspect existing owner')
    closed=container['unitrace_aggregation_host_v2']
    tool.guards.require(closed['passed'] and closed['complete'] and closed['ownership_closed'] and closed['performance_gate_passed'],
        'actual80-case host gate must pass before candidate preparation')
    revision=tool.git(P,'rev-parse','HEAD').decode().strip();pins=inputs(revision)
    assets=probe.pin_assets(container)
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir()
    source=generated()
    for name,data in source.items():(ROOT/name).write_bytes(data)
    record=dict(stage='preparing',source_commit=revision,input_sha256=pins,original_assets=assets,
        host_qualification_sha256=canonical(closed),official_revision=proposal.REVISION,
        generated_sha256={n:tool.sha(v) for n,v in source.items()},command=command(container),
        environment=dict(PATH='/usr/bin:/bin',LANG='C',LC_ALL='C',TMPDIR=str(ROOT/'tmp'),PYTHONDONTWRITEBYTECODE='1'),
        provenance='Intel PTI MIT4b6d742 with only the source-pinned host AggregateDeviceTimeStats replacement; unchanged tracer.cc/tracer.h staged solely to resolve the local overlay first. Original notices retained; no XeStrata source.',
        scope='HOST BUILD ONLY. Same original launcher bytes will be copied beside the candidate library so its unchanged executable-relative lookup selects this library. No collector load,device enumeration,GPU work,metric request,engine rebuild or runtime/environment change.',
        retained_generations='One qualified6e40 engine+12fixtures untouched. One qualified original collector/launcher retained as fallback; this one candidate is for imminent4/10 graph coverage and a new official model-profile plan. One tiny graph fixture and one80-case host fixture serve that qualification. Explicit legacy provenance holds unchanged.',
        consumer='Bounded same-runtime graph/numerical/count/owner qualification,then actual complete model aggregate/lifetime check. A compile or host proxy gain alone does not qualify the collector.',
        retention='Keep exact source/recipe and useful outcomes. Retire normal build/preprocessor output after qualification; select one collector after actual model evidence. No extra model generation or full source snapshot archive.',
        budgets=dict(compile_wall_s=900,cpu_s=900,rss_bytes=8<<30,text_bytes=8<<20,file_bytes=64<<20,total_bytes=128<<20),
        active=False,ownership_closed=True,passed=False,complete=False,GPU_qualified=False,model_qualified=False)
    admission(record,container)
    owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ROOT/'prepare-record.json',dict(record,commands=owner.commands if owner else []))
    owner=tool.supervision.Owner(ROOT/'prepare-owned',persist)
    paths=set();args=record['command'];sources=[a for a in args if a.endswith('.cc')]
    prefix=[a for a in args[:args.index(sources[0])] if a!='-shared']
    try:
        for i,source in enumerate(sources):
            dep=ROOT/f'compiler-{i}.d'
            entry,_,_=owner.run('deps-'+str(i),prefix+['-M','-MT','target','-MF',str(dep),source],record['environment'],P,
                wall=120,cpu=120,rss_cap=8<<30,text_cap=8<<20,file_cap=64<<20,total_cap=128<<20)
            tool.guards.require(tool.supervision.closed(entry) and entry['exit_code']==0,'collector preprocessing failed')
            body=dep.read_text().replace('\\\n',' ').split(':',1)[1]
            tool.guards.require('\\' not in body,'unexpected dependency escaping')
            paths.update(Path(n).resolve() for n in body.split())
        baseline_manifest=container['unitrace_build']['dependencies']
        tool.guards.require(host.digest(baseline_manifest['path'])==baseline_manifest['sha256'],'original compiler manifest changed')
        # Include the verified original system toolchain/link assets as well as
        # this overlay's actual full preprocessing closure.
        paths.update(Path(r['path']).resolve() for r in json.loads(Path(baseline_manifest['path']).read_text()))
        rows=[dict(path=str(p),sha256=host.digest(p)) for p in sorted(paths)]
        manifest=ROOT/'compiler-inputs.json';write(manifest,rows)
        record.update(stage='prepared',compiler_inputs=dict(path=str(manifest),files=len(rows),sha256=host.digest(manifest)))
        admission(record,container)
    except BaseException as error:record.update(stage='failed',error=repr(error))
    finally:
        record['ownership_closed']=owner.active is None and all(tool.supervision.closed(c) for c in owner.commands)
        persist();container[KEY]=record;write(RESULT,container)
        if record['stage']=='prepared':write(ROOT/'plan.json',record)
    print(json.dumps({k:record.get(k) for k in ('stage','ownership_closed','error')}),flush=True)
    return 0 if record['stage']=='prepared' and record['ownership_closed'] else 1


def build(container):
    record=container[KEY]
    tool.guards.require(record['stage']=='prepared' and record==json.loads((ROOT/'plan.json').read_text())
        and not (ROOT/'build-record.json').exists(),'fresh committed build only; no retry')
    admission(record,container)
    owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ROOT/'build-record.json',dict(record,commands=owner.commands if owner else []))
    owner=tool.supervision.Owner(ROOT/'build-owned',persist);record['stage']='building';persist()
    try:
        entry,_,_=owner.run('collector',record['command'],record['environment'],P,wall=900,cpu=900,
            rss_cap=8<<30,text_cap=8<<20,file_cap=64<<20,total_cap=128<<20)
        tool.guards.require(tool.supervision.closed(entry) and entry['exit_code']==0,'collector compile failed')
        launcher=record['original_assets']['launcher'];shutil.copyfile(launcher['path'],ROOT/'unitrace');(ROOT/'unitrace').chmod(0o700)
        tool.guards.require(host.digest(ROOT/'unitrace')==launcher['sha256'],'copied original launcher changed')
        record['artifacts']={n:dict(path=str(ROOT/n),bytes=(ROOT/n).stat().st_size,sha256=host.digest(ROOT/n))
            for n in ('libunitrace_tool.so','unitrace')}
        admission(record,container,True)
        record.update(stage='built',passed=True,complete=True)
    except BaseException as error:record.update(stage='failed',error=repr(error))
    finally:
        record['ownership_closed']=owner.active is None and all(tool.supervision.closed(c) for c in owner.commands)
        record['passed']=bool(record['passed'] and record['ownership_closed']);persist();container[KEY]=record;write(RESULT,container)
    print(json.dumps({k:record.get(k) for k in ('stage','passed','ownership_closed','GPU_qualified','error')}),flush=True)
    return 0 if record['passed'] else 1


def main(action):
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);tool.guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        tool.guards.require(RESULT.read_bytes()==tool.git(P,'show','HEAD:'+str(RESULT.relative_to(P))),'commit exact result/plan first')
        return prepare(container) if action=='prepare' else build(container)


if __name__=='__main__':
    tool.guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build'),'use build_indexed_collector.py prepare|build')
    sys.exit(main(sys.argv[1]))
