#!/usr/bin/env python3
"""Check second-record refusal after real positive native cache copies.

Main-authored; repository LICENSE. Reuse the qualified engine and normal entry;
the expected native exit is 1. This control measures no inference throughput.
"""
import copy
import faulthandler
import fcntl
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
import check_adaptive_native_entry as positive

P, B, RESULT, base = positive.P, positive.B, positive.RESULT, positive.base
ROOT, KEY = B/'adaptive-record-failure-v1', 'adaptive_record_failure_native_v1'
SELF = Path(__file__).resolve()
require, digest, write = positive.require, positive.digest, positive.write
N, M = 1024, 256
ERROR = 'ERR an adaptive refill failed'


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('sources', 'commands')}
    out['source_inputs'] = dict(files=len(row['sources']), sha256=positive.width.signature(row['sources']))
    return out


def save(row):
    write(ROOT/'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def normal_entry():
    row = json.loads((positive.ROOT/'record.json').read_text())
    require(row['stage']=='closed' and row['passed'] and row['complete'] and row['ownership_closed'] and
            not row['active'] and row['outcome']['adaptive']['generations']==23 and
            row['outcome']['adaptive']['swaps']==184, 'closed actual positive replacement/loan entry required')
    return row


def configuration():
    config, env, models = positive.configuration()
    env = {k: v.replace(str(positive.ROOT), str(ROOT)) for k, v in env.items()}
    require('STRATA_SYCL_TEST_ADAPT_RECORD_FAILURE' not in env, 'no inherited fault injection')
    env['STRATA_SYCL_TEST_ADAPT_RECORD_FAILURE'] = '1'
    config['log'] = str(ROOT/'owned/native.stderr')
    return config, env, models


def audit(text):
    patterns = [re.compile(p) for p in (
        r'strata serve: ADAPT_AUDIT phase=prepared generation=(\d+) incoming=(\d+) outgoing=(\d+) slot=(\d+)',
        r'strata serve: ADAPT_AUDIT phase=masked generation=(\d+) swaps=(\d+) identities=3',
        r'strata serve: ADAPT_AUDIT phase=published generation=(\d+) swaps=(\d+) identities=3 payload_match=1',
        r'strata: ADAPT_RECORD_DIAGNOSTIC ordinal=(\d+) injected=(\d+) status=(\d+)',
        r'strata serve: ADAPT_SUBMIT swaps=(\d+) H2D_requested_bytes=(\d+)',
        r'strata serve: cannot record the adaptive refill event')]
    rows = {1: [], 2: []}; state = 0; submitted_bytes = None
    for line in text.splitlines():
        if not line.startswith(('strata serve: ADAPT_AUDIT ', 'strata: ADAPT_RECORD_DIAGNOSTIC ',
                                'strata serve: ADAPT_SUBMIT ', 'strata serve: cannot record')):
            continue
        found = [(k, p.fullmatch(line)) for k, p in enumerate(patterns) if p.fullmatch(line)]
        require(len(found)==1, 'unknown or malformed failure-control progress')
        kind, match = found[0]; v = tuple(map(int, match.groups()))
        require(all(0<=n<2**64 for n in v), 'failure-control counter bounds')
        if kind==0:
            generation = 1 if state==0 else 2 if state==4 else 0
            require(v[0]==generation and generation in rows and len(rows[generation])<8,
                    'prepare outside its exact generation')
            rows[generation].append(v[1:])
        elif kind==1:
            generation = 1 if state==0 else 2 if state==4 else 0
            require(generation in rows and v==(generation,len(rows[generation])) and rows[generation],
                    'mask generation/count/order')
            state += 1
        elif kind==2:
            require(state==3 and v==(1,len(rows[1])), 'stale or unordered publication')
            state = 4
        elif kind==3:
            require((state==1 and v==(1,0,0)) or (state==5 and v==(2,1,1)),
                    'actual first success/second cudaErrorInvalidValue required')
            state += 1
        elif kind==4:
            require(state==2 and v[0]==len(rows[1]) and v[1]>0, 'first positive submission differs')
            submitted_bytes = v[1]; state = 3
        else:
            require(state==6, 'event-record refusal before real second failure')
            state = 7
    require(state==7 and submitted_bytes is not None, 'incomplete second-record failure evidence')
    for swaps in rows.values():
        require(swaps and len({i for i,o,s in swaps})==len(swaps) and
                len({o for i,o,s in swaps})==len(swaps) and len({s for i,o,s in swaps})==len(swaps) and
                all(i!=o and i//512==o//512 and i<24576 and o<24576 and s<1256 for i,o,s in swaps),
                'positive replacement tuple conservation')
    return dict(first_record_status=0, second_record_status=1, successful_generation=1,
        first_swaps=len(rows[1]), failed_generation=2, queued_second_swaps=len(rows[2]),
        second_generation_masked=True, second_generation_published=False,
        first_H2D_requested_bytes=submitted_bytes, stale_event_not_published=True)


def controls():
    p = 'strata serve: ADAPT_AUDIT '
    good = [p+'phase=prepared generation=1 incoming=10 outgoing=3 slot=4',
        p+'phase=masked generation=1 swaps=1 identities=3',
        'strata: ADAPT_RECORD_DIAGNOSTIC ordinal=1 injected=0 status=0',
        'strata serve: ADAPT_SUBMIT swaps=1 H2D_requested_bytes=1024',
        p+'phase=published generation=1 swaps=1 identities=3 payload_match=1',
        p+'phase=prepared generation=2 incoming=11 outgoing=4 slot=5',
        p+'phase=masked generation=2 swaps=1 identities=3',
        'strata: ADAPT_RECORD_DIAGNOSTIC ordinal=2 injected=1 status=1',
        'strata serve: cannot record the adaptive refill event']
    audit('\n'.join(good))
    corrupt = [good[:n]+good[n+1:] for n in range(len(good))]
    corrupt += [good+[good[-1]], good+[good[4].replace('generation=1','generation=2')],
        good[:7]+[good[7].replace('status=1','status=0')]+good[8:],
        good[:7]+[good[7].replace('injected=1','injected=0')]+good[8:],
        good[:2]+[good[2].replace('status=0','status=1')]+good[3:],
        good[:5]+[good[5].replace('generation=2','generation=3')]+good[6:],
        [good[0].replace('incoming=10','incoming=513')]+good[1:],
        [good[0].replace('slot=4','slot=1256')]+good[1:],
        good[:1]+[good[1].replace('swaps=1','swaps=2')]+good[2:],
        good[:3]+[good[3].replace('bytes=1024','bytes=0')]+good[4:],
        good[:4]+good[5:7]+good[4:5]+good[7:], good[:5]+[good[5]]+good[5:]]
    for lines in corrupt:
        try: audit('\n'.join(lines))
        except RuntimeError: continue
        raise RuntimeError('corrupt event-record failure control accepted')
    return dict(passed=True, positive_fixture=1, negative=len(corrupt), GPU_executed=False)


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'fresh separate failure control only')
    host = controls(); normal = normal_entry(); positive.current(normal,full=True)
    config, env, models = configuration()
    revision = base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    row = dict(stage='prepared', source_commit=revision, sources=base.controls(revision,controller=SELF),
        positive_entry=dict(path=str(positive.ROOT/'record.json'), sha256=digest(positive.ROOT/'record.json')),
        binary=normal['binary'], native_source_commit=normal['native_source_commit'],
        native_validation_probe=normal['native_validation_probe'], runtime_manifest=normal['runtime_manifest'],
        boot=normal['boot'], driver=normal['driver'], configuration=config, environment=env, model_assets=models,
        python=normal['python'], input_tokens=N, max_output_tokens=M, input_sha256=positive.width.signature(positive.ids(N)),
        host_controls=host, budgets=dict(wall=900,cpu=3600,rss_cap=104<<30,text_cap=4<<20,
            file_cap=8<<20,total_cap=192<<20), whole_wall_s=1500,
        passed=False,complete=False,active=False,ownership_closed=True,GPU_execution_attempted=False,
        plan='One fresh owned engine429b with exactly the qualified normal-entry config/environment,except private paths and pre-exec STRATA_SYCL_TEST_ADAPT_RECORD_FAILURE=1. Synthetic1024 input/max256 output;real first event succeeds and generation1 publishes;second real cudaEventRecord(null) returns cudaErrorInvalidValue1 after accepted positive copies with owners live. Require generation2 prepared/masked but never published,no further generation,exact ERR an adaptive refill failed,no DONE,strict ordered wire prefix,normal owned native exit1 without supervisor cleanup,quiet strict pre-main validation,unchanged final15model assets/runtime/boot/driver,no fresh GPU fault. Stop once;no retry or safety-setting change.',
        limits='Intentional native error1 is expected control success,not successful generation,device fault injection,clean PP/TG,task quality,ordinary provider teardown or physical262144/RESTORE qualification.',
        retention='Keep compact first/second statuses,swap counts,nonpublication,wire extent/output hash,source/configuration and validation/closure/fault result. Retire successful raw output/JIT after verifying and pushing the result;one minimal distinct failure diagnostic only if unexpected.')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'): (ROOT/name).mkdir(mode=0o700)
    save(row); print('Prepared second-record refusal;commit before native execution.',flush=True)


def current(frozen,full=False):
    require(base.committed(RESULT)[KEY]==compact(frozen) and
            base.controls(frozen['source_commit'],controller=SELF)==frozen['sources'], 'failure-control source/plan changed')
    ref = frozen['positive_entry']; require(digest(ref['path'])==ref['sha256'], 'normal entry changed')
    normal = normal_entry(); providers = positive.current(normal,full=full)
    config, env, models = configuration()
    require(config==frozen['configuration'] and env==frozen['environment'] and models==frozen['model_assets'] and
            all(frozen[k]==normal[k] for k in ('binary','native_source_commit','native_validation_probe',
                'runtime_manifest','boot','driver','python')), 'failure-control native/environment drift')
    return providers


def run():
    row = json.loads((ROOT/'record.json').read_text()); frozen = copy.deepcopy(row)
    require(row['stage']=='prepared' and not (ROOT/'execution.json').exists() and faulthandler.is_enabled(),
            'one fresh control using -X faulthandler required')
    outcome = dict(passed=False,complete=False,host_probe_accepted=False); owner=None; cursor=None
    def persist(): write(ROOT/'execution.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    with (ROOT/'execution.json').open('x') as stream: stream.write('{}\n')
    started = time.monotonic()
    try:
        cursor = positive.profile_split.cursor(); providers = current(frozen)
        env = positive.engine_once.native_environment(row,row['environment'])
        owner = base.supervision.Owner(ROOT/'owned',persist)
        def interact(io,native):
            outcome['startup'] = positive.startup(io.line(600))
            require(io.line(600)=='READY 131072 stop', 'native READY differs')
            positive.live(native,row,env,providers); outcome['native_pid']=native['pid']; persist()
            io.send((f'GEN {M} '+','.join(map(str,positive.ids(N)))+'\n').encode())
            reader = positive.protocol.RequestProtocol(N,M)
            while True:
                line = io.line(600)
                if line==ERROR: break
                reader.feed(line)
                require(reader.done is None, 'request completed instead of second-record refusal')
            require(reader.resumed and reader.reused and reader.position==N-1 and 0<len(reader.tokens)<M,
                    'incomplete or empty ordered generation before expected refusal')
            outcome['wire'] = dict(error=ERROR,input_tokens=N,output_tokens=len(reader.tokens),
                output_sha256=positive.width.signature(reader.tokens),progress_reached=reader.position,
                progress_notifications=reader.progress_count,DONE_seen=False)
            persist(); io.proc.stdin.close(); io.exit(90)
        row.update(active=True,ownership_closed=False,GPU_execution_attempted=True); write(ROOT/'record.json',row)
        native,out,err = owner.run('native',[row['binary']['path'],'--serve',*row['configuration']['args']],
            env,P,interaction=interact,**row['budgets'])
        outcome.update(native_exit_code=native['exit_code'],native_normal_closed=base.supervision.closed(native))
        persist(); require(outcome['native_normal_closed'] and native['exit_code']==1, 'expected native exit1 did not close normally')
        text = err.read_text()
        outcome['validation'] = positive.validation.validation_log(text,native['pid'],row['native_validation_probe']['loader']['path'],False)
        require(not re.search(r'ADAPT_AUDIT_FAILURE|terminate called|retirement failed|Strata SYCL cleanup:',text),
                'unexpected audit/runtime failure')
        outcome['adaptive'] = audit(text); outcome.update(host_probe_accepted=True,complete=True); persist()
        current(frozen,full=True); require(time.monotonic()-started<row['whole_wall_s'], 'whole-control deadline')
    except BaseException as error: outcome['error']=repr(error)
    finally:
        if owner is not None and cursor is not None: positive.ur_validation.finish(outcome,owner,cursor,row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'] and
            outcome.get('driver_after')==row['driver'] and 'error' not in outcome)
        row.update(stage='closed' if outcome['passed'] else 'failed',passed=outcome['passed'],complete=outcome['complete'],
            active=bool(owner and owner.active),ownership_closed=outcome.get('ownership_closed',owner is None),
            outcome=outcome,commands=owner.commands if owner else [])
        persist(); save(row)
    require(row['passed'],str(outcome)); print('Second real record refused;generation2 never published;native exits1 normally.',flush=True)


if __name__=='__main__':
    require(sys.argv[1:] in (['controls'],['prepare'],['run']), 'use controls|prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); base.guards.no_model_owners([])
        if sys.argv[1]=='controls': print(json.dumps(controls()))
        else: (prepare if sys.argv[1]=='prepare' else run)()
