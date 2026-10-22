#!/usr/bin/env python3
"""One-slot event-query setup gate; never submit commands or read counter data."""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import metric_inventory as inventory

P,B=inventory.P,inventory.B
ROOT=B/'b570-metric-query-setup-v1'
KEY='event_query_setup'
SOURCE='bench/sycl/metric_query_setup.cpp'


def inputs(revision):
    pins=inventory.inputs(revision)
    for name in (SOURCE,'tools/sycl/metric_query_setup.py'):
        raw=inventory.git('show',revision+':'+name)
        inventory.guards.require(raw==(P/name).read_bytes(),'query setup input changed')
        pins[name]=inventory.hashlib.sha256(raw).hexdigest()
    return pins


def admission(base,record,binary=False):
    inventory.admission(base,True)
    inventory.guards.require(inputs(record['source_commit'])==record['inputs'],'query setup source changed')
    original=json.loads(inventory.git('show',record['inventory_reference']['commit']+':'+str(inventory.RESULT.relative_to(P))))
    inventory.guards.require(original['binary']==base['binary'] and original['runtime_pins']==base['runtime_pins'] and
        original['dependencies']==base['dependencies'] and original['observations']==base['observations'] and
        original['passed'] and original['complete'] and original['ownership_closed'], 'original complete inventory changed')
    catalog=record['catalog'];inventory.guards.require(inventory.sha(catalog['path'])==catalog['sha256'],'catalog changed')
    if binary:inventory.guards.require(inventory.sha(record['binary']['path'])==record['binary']['sha256'],'query probe changed')


def main(action):
    base=inventory.committed(inventory.RESULT)
    if action=='prepare':
        inventory.guards.require(KEY not in base and not ROOT.exists(),'fresh single query setup attempt;no retry')
        inventory.guards.require(base['stage']=='closed' and base['passed'] and base['metric_enumeration_succeeded'] and
            base['fresh_faults']==[],'successful closed local inventory required')
        revision=inventory.git('rev-parse','HEAD').decode().strip();pins=inputs(revision)
        ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir();(ROOT/'cache').mkdir()
        command=list(base['build_command']);command[command.index(str(P/inventory.SOURCE))]=str(P/SOURCE);command[-1]=str(ROOT/'query-setup')
        record=dict(stage='prepared',source_commit=revision,inputs=pins,
            inventory_reference=dict(commit=revision,path=str(inventory.RESULT.relative_to(P))),
            catalog=base['observations'][1]['stdout'],command=command,
            build_environment=dict(base['build_environment'],TMPDIR=str(ROOT/'tmp')),
            environment=dict(base['jobs'][1]['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),
            plan='Fresh owned context;uniqueE20C/ComputeBasic event group(domain3,83members),activate group,create one-slot PERFORMANCE query pool and one unused query,then query-destroy,pool-destroy,deactivate,context-destroy. Stop cleanup onfailed destroy,neverretire parent orretry. No queue/commandlist/deviceallocation API,noquerybegin/end/read,streamer,Sysman,permission change orworkload. API error remains a capability failure;setup success alone doesnotqualify collection/attribution/permissions atcommand submission. Layertrace/parameter validation fromexec;loadedassets pinned;normalowner closure andnofreshfault required.',
            budgets=dict(build_wall_s=90,run_wall_s=60,cpu_s=120,text_bytes=8<<20,rss_bytes=2<<30,file_bytes=32<<20,total_bytes=96<<20),
            passed=False,complete=False,active=False,ownership_closed=True,collection_qualified=False,inference_unchanged=True)
        admission(base,record);inventory.write(ROOT/'plan.json',record)
        base[KEY]=record;inventory.write(inventory.RESULT,base)
        print('Prepared one-slot query setup source/recipe andimmutable environment;commit beforebuild.',flush=True);return 0
    record=base[KEY];inventory.guards.require(record['stage']==('prepared' if action=='build' else 'built'),'inspect existing attempt;no retry')
    admission(base,record,action=='run');root=ROOT/action;inventory.guards.require(not root.exists(),'existing owner;no duplicate')
    root.mkdir();owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        inventory.write(root/'record.json',dict(record,owned_commands=owner.commands if owner else []))
    owner=inventory.owner_module.Owner(root/'owned',persist)
    record.update(stage=action+'ing',passed=False,complete=False,ownership_closed=False);persist()
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
    try:
        b=record['budgets'];entry,out,err=owner.run(action,record['command'] if action=='build' else [str(ROOT/'query-setup')],
            record['build_environment'] if action=='build' else record['environment'],P,
            wall=b['build_wall_s'] if action=='build' else b['run_wall_s'],cpu=b['cpu_s'],rss_cap=b['rss_bytes'],
            text_cap=b['text_bytes'],file_cap=b['file_bytes'],total_cap=b['total_bytes'])
        inventory.guards.require(inventory.owner_module.closed(entry),'query setup owner didnotclosenormally')
        if action=='build':
            inventory.guards.require(entry['exit_code']==0,'query setup compilefailed')
            record['binary']=dict(path=str(ROOT/'query-setup'),bytes=(ROOT/'query-setup').stat().st_size,sha256=inventory.sha(ROOT/'query-setup'))
        else:
            events=[json.loads(line) for line in out.read_text().splitlines()]
            summaries=[e for e in events if e['kind']=='summary'];inventory.guards.require(len(summaries)==1,'one exact setup summary required')
            summary=summaries[0];returns=[e for e in events if e['kind']=='return'];failed=[e for e in returns if e['result']!=0]
            inventory.guards.require([e['api'] for e in returns]==[e['api'] for e in events if e['kind']=='begin'],'partial API call')
            inventory.guards.require(summary['objects_closed'] and summary['GPU_commands']==0 and not summary['collection_qualified'],
                'query object/interpretation gatefailed')
            ready=summary['status']=='setup_ready'
            inventory.guards.require((ready and entry['exit_code']==0 and not failed and summary['result']==0) or
                (not ready and entry['exit_code']==3 and summary['status']=='api_error' and len(failed)==1 and
                 failed[0]['api']==summary['last_api'] and failed[0]['result']==summary['result']), 'unexpected setup result')
            mapped={str(Path(e['path']).resolve()) for e in events if e['kind']=='elf'}
            inventory.guards.require(mapped and mapped<=set(base['runtime_pins']) and any('libze_validation_layer.so' in x for x in mapped),
                'unqualified runtime/validation provider')
            record.update(setup_ready=ready,summary=summary,API_calls=len(returns),API_failures=failed,
                native_exit_code=entry['exit_code'],stdout_sha256=inventory.sha(out),stderr_sha256=inventory.sha(err),
                normal_object_closure=True,mapped_ELFs=sorted(mapped))
        record.update(stage='built' if action=='build' else 'closed',passed=True,complete=True)
    except BaseException as error:record.update(stage='failed',passed=False,error=repr(error))
    finally:
        record['ownership_closed']=owner.active is None and all(inventory.owner_module.closed(c) for c in owner.commands)
        try:
            journal=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            record['fresh_faults']=[x for x in journal.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
            record['driver_after']=inventory.guards.driver_guard();record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except Exception as error:record['fault_probe_error']=repr(error)
        record['passed']=bool(record['passed'] and record['ownership_closed'] and record.get('fresh_faults')==[]
            and record.get('boot_unchanged') and 'fault_probe_error' not in record)
        persist();inventory.write(inventory.RESULT,base)
    print(json.dumps({k:record.get(k) for k in ('stage','passed','complete','ownership_closed','setup_ready','summary','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    inventory.guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build','run'),'use prepare|build|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);inventory.guards.no_model_owners([]);inventory.guards.driver_guard()
        sys.exit(main(sys.argv[1]))
