#!/usr/bin/env python3
"""Qualify the indexed collector with the unchanged drained graph fixture."""
import fcntl
import hashlib
import json
from pathlib import Path
import secrets
import sys

sys.dont_write_bytecode=True
import build_indexed_collector as build
import graph_record_window as graph
from build_engine import blobs

tool=build.tool
P,B,RESULT=tool.P,tool.B,tool.RESULT
ROOT=B/'unitrace-indexed-graph-v1'
KEY='unitrace_indexed_graph'


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def inputs(revision):
    result=graph.inputs(revision);result.update(build.inputs(revision))
    name='tools/sycl/check_indexed_collector.py';raw=tool.git(P,'show',revision+':'+name)
    tool.guards.require(raw==(P/name).read_bytes(),'commit exact graph controller first')
    result[name]=tool.sha(raw)
    return result


def fixture(container):
    qualified=container['unitrace_graph_aggregate']
    tool.guards.require(qualified['passed'] and qualified['complete'] and qualified['ownership_closed']
        and not qualified['active'] and qualified['fresh_faults']==[] and qualified['boot_unchanged']
        and qualified['service_coverage_qualified'],'original aggregate coverage required')
    reference=qualified['plan_archive'];raw=tool.git(P,'show',reference['commit']+':'+reference['path'])
    tool.guards.require(tool.sha(raw)==reference['sha256'],'original fixture plan changed')
    original=json.loads(raw)['unitrace_graph_aggregate']
    recovered={n:tool.sha(v) for n,v in blobs(P,original['source_commit'],original['input_sha256'])}
    tool.guards.require(recovered==original['input_sha256'],'original188 fixture source inputs unrecoverable')
    differences=[]
    allowed={'tools/sycl/engine_once.py','tools/sycl/graph_record_window.py'}
    for name,sha in recovered.items():
        actual=build.host.digest(P/name)
        if actual!=sha:
            tool.guards.require(name in allowed,'unreviewed original fixture input difference: '+name)
            differences.append(dict(path=name,original_sha256=sha,current_sha256=actual,
                scope='Python controller/wrapper only; not compiled or run by the original fixture recipe. New graph controller is separately pinned.'))
    return original,dict(original_source_inputs_recovered=len(recovered),controller_only_differences=differences,
        binary_source_commit=original['binary_source_commit'],new_native_build=False,new_GPU_validation=False)


def admission(record,container,binary=True):
    tool.guards.require(inputs(record['source_commit'])==record['input_sha256'],'graph qualification source changed')
    candidate=container['unitrace_indexed_collector']
    tool.guards.require(candidate['stage']=='built' and candidate['passed'] and candidate['complete']
        and candidate['ownership_closed'] and not candidate['active'] and not candidate['GPU_qualified'],
        'closed host-only collector candidate required')
    build.admission(candidate,container,True)
    original,proof=fixture(container)
    tool.guards.require(proof==record['fixture_reuse'] and original['binary']==record['binary']
        and original['compiler_inputs']==record['compiler_inputs'],'original fixture/reuse proof differs')
    expected=dict(candidate['original_assets'],collector=candidate['artifacts']['libunitrace_tool.so'],
        launcher=candidate['artifacts']['unitrace'])
    tool.guards.require(record['assets']==expected,'graph collector/launcher selection differs')
    manifest=record['compiler_inputs'];tool.guards.require(build.host.digest(manifest['path'])==manifest['sha256'],'fixture compiler manifest changed')
    rows=json.loads(Path(manifest['path']).read_text());tool.guards.require(len(rows)==manifest['files'],'fixture compiler count differs')
    for row in rows:tool.guards.require(build.host.digest(row['path'])==row['sha256'],'original fixture compiler/header changed')
    for row in record['loader_diagnostic_evidence']['ELFs']:
        tool.guards.require(build.host.digest(row['path'])==row['sha256'],'inspected loader/driver changed')
    tool.guards.require(build.host.digest(record['binary']['path'])==record['binary']['sha256'],'fixture binary changed')


def prepare(container):
    tool.guards.require(KEY not in container and not ROOT.exists(),'fresh graph qualification only; no retry')
    candidate=container['unitrace_indexed_collector']
    tool.guards.require(candidate['stage']=='built' and candidate['passed'] and candidate['ownership_closed'] and not candidate['active'],
        'collector build must close before graph preparation')
    original,proof=fixture(container)
    revision=tool.git(P,'rev-parse','HEAD').decode().strip();pins=inputs(revision)
    ROOT.mkdir(mode=0o700)
    record=dict(schema='strata-indexed-collector-graph-v1',stage='check-prepared',source_commit=revision,input_sha256=pins,
        assets=dict(candidate['original_assets'],collector=candidate['artifacts']['libunitrace_tool.so'],launcher=candidate['artifacts']['unitrace']),
        binary=original['binary'],compiler_inputs=original['compiler_inputs'],fixture_reuse=proof,
        loader_diagnostic_evidence=original['loader_diagnostic_evidence'],
        passed=False,complete=False,active=False,ownership_closed=True,service_coverage_qualified=False,model_qualified=False,
        consumer='Require actual counts/service/numerical/owner/validation gates before a new official whole-model aggregate plan. Earlier failed attempts stay failed.',
        gates='Same original fixture227680B and8697 compiler inputs,unchanged safety settings and ordinary graph recipe. Traced/validated no-collector smoke,then initially recorded construction+2 warm replays,stay-paused versus3 later resumed replays. Require4/10 verbose named calls,one1164-byte copy,no readback leakage,exact positive time sums,numerical/guards/checked teardown,normal closure,unique session removal and no fresh fault. Any failure stops; no timing subtraction or changed adoption threshold.',
        limits='Collector service coverage only,not full model attribution,critical path,saturation,observer overhead,clean speed or70tok/s. Host80-case proxy did not test actual shutdown.',
        retention='Keep compact3-cell counts/service/owner/fault/identities; retire normal UR/loader logs after replay verification. Retain original collector fallback until actual model table and native shutdown qualify.',cells=[])
    base=json.loads((P/'tools/sycl/measurement_environment.json').read_text())['environment']
    for name,expected in (('smoke',None),('prefix-only',0),('prefix-window',6)):
        root=ROOT/name;root.mkdir()
        for folder in ('tmp','cache','trace'):(root/folder).mkdir()
        env=dict(base,TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),LD_DEBUG='libs',UNITRACE_SystemTime='1',
            STRATA_UR_PROBE_EXPECTED_LOADER=record['assets']['preload']['loader']['path'],
            UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'ur.log'))
        session='StrataIndexedGraph'+secrets.token_hex(12) if expected is not None else None
        command=[record['binary']['path']]
        if session:
            env.update(STRATA_UNITRACE_TARGET_PRELOAD=record['assets']['preload']['library']['path'],
                UNITRACE_FollowChildProcess='1',UNITRACE_TraceOutputDir=str(root/'trace'))
            command=[record['assets']['launcher']['path'],'--session',session,'--device-timing','--verbose',*command]
        else:env['LD_PRELOAD']=record['assets']['preload']['library']['path']
        record['cells'].append(dict(name=name,expected_kernels=expected,session=session,command=command,environment=env))
    admission(record,container);container[KEY]=record;write(RESULT,container);write(ROOT/'plan.json',record)
    print('Prepared bounded indexed-collector graph qualification; commit exact plan first.',flush=True)


def main(action):
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);tool.guards.no_model_owners([]);tool.guards.driver_guard()
        container=json.loads(RESULT.read_text())
        tool.guards.require(RESULT.read_bytes()==tool.git(P,'show','HEAD:'+str(RESULT.relative_to(P))),'commit graph result/plan first')
        if action=='prepare':return prepare(container)
        record=container[KEY]
        tool.guards.require(record==json.loads((ROOT/'plan.json').read_text()) and not (ROOT/'check-record.json').exists(),
            'inspect existing graph owner; no retry')
        graph.check(record,container,root=ROOT,admission_check=admission)
        container[KEY]=record;write(RESULT,container)
        print(json.dumps({k:record.get(k) for k in ('passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
        return 0 if record['passed'] else 1


if __name__=='__main__':
    tool.guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','check'),'use check_indexed_collector.py prepare|check')
    sys.exit(main(sys.argv[1]))
