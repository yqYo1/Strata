#!/usr/bin/env python3
"""Official32K+ long-generation admission for checked cache scratch; Main only.

This parameter-validated check precedes a separately planned clean comparison.
It never retries an existing root and does not qualify throughput or256K.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
import cache_borrow as borrowed
import resource_engine as resources

q, gen, assets, guards, supervision = borrowed.q, borrowed.gen, borrowed.assets, borrowed.guards, borrowed.supervision
P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'cache-capacity-check-v1'
RESULT = A / 'cache-capacity-current.json'
SCOPE = 'cache_loan_long_context_capacity_scope'


def matches(pattern, text, count):
    values = list(re.finditer('^'+pattern+'$', text, re.M))
    guards.require(len(values) == count, 'missing/ambiguous evidence: '+pattern[:72])
    return values


def long_evidence(log, sample):
    """Bind all proof to the measured request, excluding successful warmups."""
    headers = matches(r'strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*', log, 5)
    identity = (sample['prompt_tokens'], 0, sample['read_tokens'], sample['generated_tokens'])
    guards.require(tuple(map(int, headers[-1].groups())) == identity and
        all(int(h[1]) < 32768 for h in headers[:-1]), 'actual measured native request differs')
    segments = [log[0 if i == 0 else headers[i-1].end():h.start()] for i,h in enumerate(headers)]
    generations, free = [], []
    for text in segments:
        layout, = matches(r'strata loan audit: phase=layout chunk=(\d+) first=(\d+) slots=1280 bytes=(\d+) need=(\d+) lent=(\d+) mask_match=1', text, 1)
        chunk, first, capacity, need, lent = map(int, layout.groups())
        guards.require(256 <= chunk <= 8192 and 128 <= first < 1280 and
            lent == 1280-first and capacity >= need > 0, 'loan extent/mask differs')
        pub = matches(r'strata cache loan: phase=(masked|ready) generation=(\d+) chunk=(\d+) first=(\d+) lent=(\d+) decode_ready=([01])', text, 2)
        guards.require([(m[1],m[6]) for m in pub] == [('masked','0'),('ready','1')] and
            pub[0].groups()[1:5] == pub[1].groups()[1:5] and
            tuple(map(int,pub[0].groups()[2:5])) == (chunk,first,lent) and
            layout.start() < pub[0].start() < pub[1].start(), 'publication identity/order differs')
        generations.append(int(pub[0][2]))
        payload, = matches(r'strata loan audit: phase=payload rows=1280 bytes=(\d+) lent=(\d+) same_slots=1 payload_match=1', text, 1)
        tables = matches(r'strata loan audit: phase=(lend|refill) table_entries=24576 table_match=1 free_bytes=(\d+) total_bytes=(\d+)', text, 2)
        guards.require(int(payload[1]) > capacity and int(payload[2]) == lent and
            [m[1] for m in tables] == ['lend','refill'] and
            tables[0].start() < pub[0].start() < payload.start() < tables[1].start() < pub[1].start() and
            text.count('strata loan audit:') == 4 and text.count('strata cache loan:') == 2,
            'exact payload/map refill evidence differs')
        for m in tables:
            guards.require(0 <= int(m[2]) <= int(m[3]) and int(m[3]) > 0, 'invalid VRAM observation')
            free.append(int(m[2]))
    guards.require(all(a > 0 and b > a for a,b in zip(generations,generations[1:])), 'loan generation reused')
    text = segments[-1]
    rows = sample['read_tokens']-1
    guards.require(32767 <= rows <= 65536, 'this admission covers the primary32K mixed window only')
    plan, = matches(r'strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) residual_host=0 residual_staging=0 layer_cache=1022361600; extra allocations, GPU headroom=536870912, RAM headroom=8589934592', text, 1)
    guards.require(tuple(map(int,plan.groups())) == (rows,rows*20480,rows,rows*20480) and
        len(re.findall(r'^strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows='+str(rows)+r' mixed_fp16_bytes=0$',text,re.M)) == 1 and
        'effective_mode=2' not in text, 'long all-GPU mode10 route differs')
    dense = matches(r'strata prefill dense: phase=(retired|bank|restored) residency=(Full|Bank) activations=(\d+) full_bytes=(\d+) bank_bytes=(\d+) snapshot_bytes=(\d+) overlay_rows=0 session_recapture=0 verifier=1 free_bytes=(\d+) total_bytes=(\d+)', text, 3)
    guards.require([(m[1],m[2],int(m[3])) for m in dense] ==
        [('retired','Full',0),('bank','Bank',0),('restored','Full',48)] and
        all(tuple(map(int,m.groups()[3:6])) == tuple(map(int,dense[0].groups()[3:6])) and
            0 <= int(m[7]) <= int(m[8]) for m in dense) and
        int(dense[0][4]) == int(dense[0][6]) > int(dense[0][5]) > 0,
        'dense retirement/banking/restoration differs')
    binding, = matches(r'strata prefill cache binding: generation=(\d+) parent_match=1 execution_distinct=1', text, 1)
    publications = list(re.finditer(r'^strata cache loan: phase=(masked|ready) ',text,re.M))
    guards.require(int(binding[1]) == generations[-1] and
        publications[0].start() < dense[0].start() < dense[1].start() < binding.start() < dense[2].start() < publications[1].start(),
        'actual cache substitution/dense phases escaped the request loan')
    windows = matches(r'strata MoE cache loan: generation=(\d+) layout_generation=(\d+) layer=(\d+) first=0 rows=(\d+) used=(\d+) capacity=(\d+)', text, 48)
    ptr = r'0x[0-9a-fA-F]+'
    entries = matches(r'strata MoE mixed owners: phase=entry layer=(\d+) first=0 rows=(\d+) primary='+ptr+r' primary_bytes=(\d+) region='+ptr+r' region_bytes=(\d+) tail=\(nil\) tail_offset=(\d+) tail_bytes=0 residual='+ptr+r' residual_bytes=(\d+) bo='+ptr+r' queue='+ptr+r' direct_owners=0 previous_attention=checked_drain', text, 48)
    retired = matches(r'strata MoE mixed owners: phase=retired layer=(\d+) first=0 rows=(\d+) route_status=0 shared_HC=checked_drain callbacks=after_return', text, 48)
    previous = binding.start()
    epoch, used, capacity = int(windows[0][2]), int(windows[0][5]), int(windows[0][6])
    guards.require(epoch > 0 and 0 < used <= capacity and capacity == int(re.findall(r'^strata loan audit: phase=layout chunk=8192 first=\d+ slots=1280 bytes=(\d+)',text,re.M)[0]), 'sealed loan extent differs')
    for layer,(w,e,d) in enumerate(zip(windows,entries,retired)):
        guards.require(tuple(map(int,w.groups())) == (generations[-1],epoch,layer,rows,used,capacity) and
            tuple(map(int,e.groups()[:2])) == tuple(map(int,d.groups())) == (layer,rows) and
            int(e[3]) == rows*5120 and int(e[4]) > 0 and int(e[5]) <= int(e[4]) and int(e[6]) == rows*20480 and
            previous < e.start() < w.start() < d.start() < dense[2].start(), 'borrowed FFN span/lifetime/order differs')
        previous = d.start()
    guards.require(text.count('strata MoE cache loan:') == 48 and text.count('strata MoE mixed owners:') == 96,
        'unparsed or extra mixed-owner records')
    return dict(completed_requests=5,loan_generations=generations,exact_payload_and_map_cycles=5,
        min_observed_free_bytes=min(free),prefill_rows=rows,residual_gpu_bytes=rows*20480,
        borrowed_ffn_entries=48,checked_retirements=48,dense_layers=48,
        execution_cache_parent_checked=True,primary_only=True,clean_speed_qualified=False)


def source_review():
    row = gen.committed(borrowed.RESEARCH)[SCOPE]
    guards.require(row.get('bounded_long_check_accepted') is True and row.get('worker_closed') is True,
        'complete and curate long-context source review before model admission')
    for name,expected in row['frozen_inputs'].items():
        guards.require(gen.common.digest(P/name) == expected and
            hashlib.sha256(gen.git('show',row['source_commit']+':'+name)).hexdigest() == expected,
            'reviewed native source changed: '+name)
    return {k:row[k] for k in ('source_commit','frozen_inputs','report_sha256','native_review_sha256')}


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(), 'fresh check only; never retry or overwrite')
    full,count = q.current_build()
    tiny,base = gen.committed(borrowed.RESULT),gen.committed(assets.RESULT)
    guards.require(tiny['complete'] and tiny['passed'] and tiny['ownership_closed'] and not tiny['active'] and
        not tiny['fresh_faults'] and tiny['binary'] == full['binary'] and
        tiny['checked_owner_evidence']['borrowed_ffn_entries'] == 48, 'matching completed tiny model prerequisite required')
    review = source_review()
    source = gen.git('rev-parse','HEAD').decode().strip()
    pins,settings = gen.benchmark_pins(source),gen.committed(gen.SETTINGS)
    config = copy.deepcopy(base['config']); flags = config['args']
    for name,value in (('--expert-cache','1280'),('--pool-workers','5'),('--pcie-frac','0'),
                       ('--expert-profile',str(ROOT/'expert-profile.bin'))):
        flags[flags.index(name)+1] = value
    guards.require(flags.count('--no-prefill-borrow') == 1, 'unexpected seed borrow placement')
    flags.remove('--no-prefill-borrow'); config.update(log=str(ROOT/'engine.stderr'),env={})
    old_root = base['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old_root,str(ROOT)) for k,v in base['environment'].items()}
    guards.require(not any(k == 'STRATA_TRACE' or k == 'UR_ENABLE_LAYERS' or k.startswith('UR_LOG_') for k in env), 'seed has observers')
    env.update(STRATA_SYCL_CACHE_SCRATCH_LEASE='1',STRATA_SYCL_CACHE_LOAN_AUDIT='1',
        STRATA_SYCL_VERIFY_ARENA_COPY='0',STRATA_SYCL_VERIFY_PCIE_EARLY='0',
        STRATA_SYCL_VERIFY_PCIE_EARLY_AUDIT='0',STRATA_SYCL_QUERY_PROFILE='0',
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',STRATA_TRACE='1')
    env.update({k:v for k,v in settings['environment'].items() if k.startswith('UR_LOG_') and k != 'UR_LOG_TRACING'})
    for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[key] == settings['environment'][key], 'qualified safety setting changed')
    model = assets.model_pins(config,full=True)
    ROOT.mkdir(mode=0o700)
    for name in ('cache','tmp'): (ROOT/name).mkdir()
    shutil.copyfile(P/'data/expert-profile.bin',ROOT/'expert-profile.bin')
    guards.require(gen.common.digest(ROOT/'expert-profile.bin') == base['profile_sha256'], 'profile changed')
    (ROOT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    with socket.socket() as sock: sock.bind(('127.0.0.1',0));port = sock.getsockname()[1]
    command = ['/bin/sh',str(P/'bench.sh'),'--config',str(ROOT/'config.json'),'--port',str(port),
        '--long-generation-only','--long-generation-workload','regular-code','--think','--runs','1',
        '--workload-id',gen.WORKLOAD_ID,'--fail-fast-engine-error','--ready-timeout','600','--out',str(ROOT/'bench')]
    record = dict(mode='cache-capacity-check',source_commit=source,build_source_commit=full['source_commit'],
        binary=full['binary'],source_build_record=dict(path=str(q.BUILD_RECORD),sha256=gen.common.digest(q.BUILD_RECORD)),
        original_native_inputs_recovered=count,component_result_sha256=gen.common.digest(q.COMPONENT),
        tiny_model_reference=dict(commit=source,path=str(borrowed.RESULT.relative_to(P)),sha256=gen.common.digest(borrowed.RESULT)),
        runtime_asset_delta=base['runtime_asset_delta'],source_review=review,root=str(ROOT),config=config,environment=env,command=command,
        config_sha256=gen.common.digest(ROOT/'config.json'),profile_sha256=base['profile_sha256'],
        settings_sha256=gen.common.digest(gen.SETTINGS),model_assets=model,model_assets_fingerprint=assets.fingerprint(model),
        machine_identity=assets.machine_identity(),source_count=len(pins),source_aggregate_sha256=assets.fingerprint(pins),
        tokenizer_assets={str(Path(config['tokenizer'])/n):gen.common.digest(Path(config['tokenizer'])/n)
            for n in ('vocab.json','merges.txt','token_type.json','chat_template.jinja')},
        complete=False,passed=False,active=False,clean_speed_qualified=False,model_quality_qualified=False,
        gate='One official process, four successful warmups and one32768..65537-input/1024+ output regular-code request with thinking and answer body. Exact1280-slot refill/map checks for all five requests; all-GPU BF16 residuals, actual mode10 and48 checked borrowed FFNs for the long request, Full restoration and actual decode. Normal closure, unchanged source/model/runtime/boot, no fresh fault. Stop on first refusal/budget/error; no retries. No clean timing comparison or adoption.',
        limits='Observed diagnostic timings include validation, trace and full cache readback. Primary mixed bank only. Strong-only workload is not strong/weak, agent-session, task-quality, physical262144/session/RESTORE or70tok/s qualification.',
        consumer='Admit or reject the1280-slot route for a later separately committed owned-small/borrowed-large clean comparison.',
        budgets=dict(wall=2400,cpu=54000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20))
    q.runtime_assets_unchanged(record)
    (ROOT/'plan.json').write_text(json.dumps(record,indent=2)+'\n');RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared official long cache check; commit exact plan before run.')


def run():
    record = gen.committed(RESULT)
    guards.require(record == json.loads((ROOT/'plan.json').read_text()) and not record['complete'] and
        not (ROOT/'record.json').exists(), 'fresh frozen plan required; no retry')
    full,count = q.current_build()
    guards.require(full['binary'] == record['binary'] and count == record['original_native_inputs_recovered'] and
        gen.common.digest(q.BUILD_RECORD) == record['source_build_record']['sha256'] and
        gen.common.digest(q.COMPONENT) == record['component_result_sha256'] and source_review() == record['source_review'], 'build/component/review changed')
    pins = gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and assets.fingerprint(pins) == record['source_aggregate_sha256'], 'benchmark source changed')
    for path,expected in [(gen.SETTINGS,record['settings_sha256']),(ROOT/'config.json',record['config_sha256']),
            (ROOT/'expert-profile.bin',record['profile_sha256']),*record['tokenizer_assets'].items()]:
        guards.require(gen.common.digest(path) == expected, 'input changed: '+str(path))
    q.runtime_assets_unchanged(record)
    guards.require(assets.model_pins(record['config'],record['model_assets']) == record['model_assets'] and
        assets.machine_identity() == record['machine_identity'], 'model/platform changed')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
    cursor = next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '));owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (ROOT/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    owner = supervision.Owner(ROOT/'owned',persist)
    try:
        entry,_,_ = owner.run('bench',record['command'],record['environment'],P,**record['budgets'])
        record.update(complete=supervision.closed(entry),exit_code=entry.get('exit_code'))
        guards.require(record['complete'] and entry['exit_code'] == 0, 'owned official benchmark failed')
        path, = (ROOT/'bench').glob('*/results.json')
        suite = json.loads(path.read_text())['results']['long_generation']
        rows = gen.collect_samples(suite);record['requests'] = rows
        gen.require_complete_generation(suite,rows,('regular-code',))
        journal,requests = gen.journal_evidence(ROOT);resources.require_journal_binding(rows,requests)
        record['warmups'] = gen.warmup_evidence(ROOT,journal)
        native = json.loads((ROOT/'engine-exit.json').read_text())
        guards.require(native['exit_code'] == 0 and native['boot_id'] == boot and native['binary'] == record['binary'] and
            native['args'] == ['--serve',*record['config']['args']] and 'resource_collector' not in native, 'native launch/closure differs')
        log = (ROOT/'engine.stderr').read_text(errors='replace');assets.verify_pool(log,5)
        record['long_capacity_evidence'] = long_evidence(log,rows[0])
        guards.require('strata decode cpu: ' not in log and 'strata decode timing: ' not in log, 'unexpected observer')
        q.current_build();q.runtime_assets_unchanged(record)
        guards.require(gen.benchmark_pins(record['source_commit']) == pins and assets.machine_identity() == record['machine_identity'] and
            assets.model_pins(record['config'],record['model_assets'],full=True) == record['model_assets'], 'inputs changed during run')
        record.update(passed=True,native_identity=native['native'])
    except BaseException as error:
        record['error'] = str(error)
        try:
            journal,rows = gen.journal_evidence(ROOT);record['request_journal'] = journal
            if 'requests' not in record: record['requests'] = rows
        except Exception as failure: record['journal_recovery_error'] = str(failure)
        for path in (ROOT/'engine.stderr',ROOT/'owned/bench.stderr'):
            if path.exists(): record['error_excerpt'] = path.read_text(errors='replace')[-4096:]
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
        record['fresh_faults'] = [x for x in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
        record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        try: record['driver_after'] = guards.driver_guard()
        except Exception as error: record.update(driver_guard_error=str(error),passed=False)
        record['passed'] = bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])
        persist();RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('complete','passed','ownership_closed','error')}))
    return 0 if record['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'));args = parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare() if args.action == 'prepare' else run()


if __name__ == '__main__': sys.exit(main())
