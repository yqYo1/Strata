#!/usr/bin/env python3
"""Bind the existing graph collector to the current payload-audit generation.

Main-authored composition of this fork's admission/ownership/trace readers;
repository LICENSE. Reuse existing engine, fixture and collector bytes. This
small control does not qualify model coverage, overhead, speed or saturation.
"""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
import adaptive_payload_admission as admission
import profile_iq3_decode as previous

P,B,RESULT,base=admission.P,admission.B,admission.RESULT,admission.base
SELF=Path(__file__).resolve()
ROOT=B/'payload-gpu-coverage-v1'
KEY='adaptive_payload_collector_active_v1'
ENTRY='adaptive_payload96_native_entry_v1'
PREFLIGHT=Path('/home/yayoi/.agents/skills/xpu-runtime-preflight/scripts/check_runtime_preflight.sh')
require,digest=admission.require,admission.digest


def selected_basis():
    native=admission.compiled()
    data=base.committed(RESULT);prior=data[ENTRY]
    require(prior['stage']=='closed' and prior['passed'] and prior['complete'] and
        prior['ownership_closed'] and not prior['active'] and prior['model_executed'] and
        prior['native_source_commit']==native['source_commit'] and
        prior['binary']==native['binaries']['strata'],'closed exact current native entry required')
    outcome=prior['outcome']
    require(outcome['native_exit_code']==0 and outcome['fresh_faults']==[] and
        outcome['boot_unchanged'] and outcome['ownership_closed'] and
        outcome['driver_after']==prior['driver'],'current native entry did not close cleanly')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==prior['boot'] and
        base.guards.driver_guard()==prior['driver'],'requalify changed boot or driver separately')
    asset=prior['runtime_manifest']
    require(digest(asset['path'])==asset['sha256'],'current provider inventory changed')
    providers=json.loads(Path(asset['path']).read_text())
    require(len(providers)==asset['files']==84 and
        all(digest(path)==sha for path,sha in providers.items()),'actual current runtime bytes changed')
    for item in native['binaries'].values():
        require(digest(item['path'])==item['sha256'] and
            Path(item['path']).stat().st_size==item['bytes'],'current same-build executable changed')
    assets=previous.profile.collector_assets(data)
    return native,prior,providers,assets


def context():
    spec=importlib.util.spec_from_file_location('strata_payload_current_collector',previous.compat.__file__)
    ctx=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctx)
    ctx.ROOT=ctx.CONTROL_ROOT=ROOT;ctx.KEY=ctx.CONTROL_KEY=KEY
    ctx.SOURCE=SELF;ctx.ACTIVE_FROM_EXEC=True;ctx.basis=selected_basis

    def current(row):
        require(base.committed(RESULT)[KEY]==ctx.compact(row) and
            base.controls(row['source_commit'],controller=SELF)==row['sources'],
            'commit the exact current coverage controller and plan')
        native,prior,providers,assets=selected_basis()
        require(row['native_source_commit']==native['source_commit'] and
            row['engine']==native['binaries']['strata'] and
            row['binary']==native['binaries']['sycl_submission_test'] and
            row['native_validation_probe']==prior['native_validation_probe'] and
            row['environment_template']==prior['environment'] and
            row['boot']==prior['boot'] and row['driver']==prior['driver'] and row['assets']==assets and
            row['budgets']==ctx.BUDGET and row['whole_bytes']==128<<20 and
            row['profile']['mode']=='aggregate' and
            row['profile']['environment']['UNITRACE_TraceOutputDir']==str(ROOT/'trace') and
            all(row['profile'][k]==v for k,v in assets.items()),'current coverage inputs changed')
        preflight=row['preflight']
        require(preflight['passed'] and preflight['complete'] and preflight['ownership_closed'] and
            preflight['failures']==[] and digest(PREFLIGHT)==preflight['script_sha256'] and
            all(digest(p)==sha for p,sha in preflight['tools'].items()),
            'close and commit host readiness before GPU execution')
        return providers

    ctx.current=current
    return ctx


def prepare():
    ctx=context();ctx.prepare()
    row=json.loads((ROOT/'record.json').read_text())
    row['preflight']=dict(script=str(PREFLIGHT),script_sha256=digest(PREFLIGHT),
        command=['/bin/bash',str(PREFLIGHT),'--target-gpu','0','--out-dir',str(ROOT/'preflight')],
        tools={str(p):digest(p) for p in ('/usr/bin/bash','/usr/bin/python3','/usr/bin/timeout','/usr/bin/xpu-smi')},
        environment=dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',
            ZES_ENABLE_SYSMAN='0',NEOReadDebugKeys='1',EnableDirectSubmission='0',UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1'),
        budgets=dict(wall=120,cpu=120,rss_cap=2<<30,text_cap=4<<20,file_cap=8<<20,total_cap=16<<20),
        passed=False,complete=False,ownership_closed=True)
    row['plan'].update(
        native='Exact bbd345/e1b011 generation,all six checked ELFs and its actual df574bb8 compatibility fixture. No old b310/429b/cb7 binary or observer qualification is inherited.',
        fixture='No-argument frozen tests/sycl/compat.cpp. Audited257-element buffers,cross-queue completion,host handoff,retained recording resources,late replay and draining frees. Inventory1direct+12first+24host-handoff+24cross-queue+1late=62kernels. No in-process environment changes.',
        order='Commit source and fresh plan;run bounded installed read-only host preflight;commit its outcome;then one active-from-exec current compatibility fixture with Sysman0,native-capture0,128-event buffer and immutable runtime safety.',
        acceptance='Original numerical/events/graph/copy/lifetime pass;native0,closed owners,quiet parameter validation,no fresh driver fault,exact current provider/collector bytes. All62named kernel events must have unique native host-flow joins and contained raw-clock intervals. Both gates observe active collection;normal exit flushes trace and removes named session.',
        next='Only this current-ELF control may support a separately committed bounded thinking-on MTP4 model-window coverage/critical-path diagnostic. Qualify model event joins and observer cost separately;do not add overlapping service times.',
        limits='Host readiness and small compatibility/collector admission only;no model coverage,CPU work attribution,interior paused window,observer overhead,clean speed,physical RAM/VRAM/PCIe traffic,saturation,physical262144/RESTORE or inference adoption.',
        retention='Keep compact actual-source/provider/clock/count/outcome evidence. Successful readiness logs,full UR trace,timeline,JIT and transaction bookkeeping retire after useful joins and source/input recovery;keep a diagnostic only for a distinct unresolved failure. No new native build generation.')
    ctx.save(row)
    print('Prepared exact current payload GPU coverage control;commit before preflight.',flush=True)


def preflight():
    ctx=context();row=json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[KEY]==ctx.compact(row) and
        base.controls(row['source_commit'],controller=SELF)==row['sources'],'commit readiness plan first')
    selected_basis();frozen=row['preflight']
    require(not frozen['passed'] and not frozen['complete'] and
        digest(PREFLIGHT)==frozen['script_sha256'] and
        all(digest(p)==sha for p,sha in frozen['tools'].items()) and
        not (ROOT/'preflight-attempt.json').exists(),
        'fresh unchanged readiness attempt only')
    own=None;outcome=dict(passed=False,complete=False,ownership_closed=True)
    def persist():
        base.write(ROOT/'preflight-attempt.json',dict(outcome,active=bool(own and own.active),commands=own.commands if own else []))
    persist()
    try:
        own=base.supervision.Owner(ROOT/'preflight-owned',persist)
        item,_,_=own.run('readiness',frozen['command'],frozen['environment'],P,**frozen['budgets'])
        require(base.supervision.closed(item),'host readiness owner did not close')
        status=(ROOT/'preflight/status.tsv').read_text().splitlines()
        require(status and status[0]=='status\tcheck\tdetail','readiness status header missing')
        rows=[line.split('\t',2) for line in status[1:]]
        require(len(rows)>=6 and all(len(r)==3 and r[0] in ('PASS','WARN','FAIL','INFO') for r in rows),
            'incomplete readiness evidence')
        # Do not commit hostnames,proxy strings or raw environment details.
        retained=[dict(status=r[0],check=r[1]) for r in rows]
        failures=[r[1] for r in rows if r[0]=='FAIL']
        counts={key:sum(r[0]==key for r in rows) for key in ('PASS','WARN','FAIL','INFO')}
        outcome.update(failures=failures,status=retained,counts=counts,
            verdict='BLOCKED' if failures else 'READY WITH WARNINGS' if counts['WARN'] else 'READY',
            status_sha256=digest(ROOT/'preflight/status.tsv'),GPU_fixture_executed=False)
        require(item['exit_code']==0 and not failures,'host readiness blockers: '+repr(failures))
        require(base.guards.driver_guard()==row['driver'],'driver changed during readiness')
        outcome.update(passed=True,complete=True)
    except BaseException as error:outcome['error']=repr(error)
    finally:
        outcome['ownership_closed']=own is None or own.active is None and all(base.supervision.closed(c) for c in own.commands)
        outcome['passed']=bool(outcome['passed'] and outcome['ownership_closed']);persist()
        row['preflight'].update(outcome);ctx.save(row)
    require(outcome['passed'],outcome.get('error','readiness did not close'))
    print('Host readiness closed;commit before current GPU fixture.',flush=True)


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['preflight'],['check']),'use prepare|preflight|check')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        {'prepare':prepare,'preflight':preflight,'check':lambda:context().check()}[sys.argv[1]]()
