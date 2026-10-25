#!/usr/bin/env python3
"""Main-owned historical/current userspace crossover; no system installation.

Repository LICENSE. Extend our d53e15bda NEO-only controller, not third-party
application code. Recovered NEO/GMM/IGDML/zlib bytes must match historical hashes.
This diagnostic selects no inference implementation or production runtime.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import cpu_capacity as models
import generation_engine as gen
import measurement_owner as owners
import runtime_probe as runtime

P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'prefill-historical-userland-crossover-v1'
RESULT = A / 'prefill-runtime-comparison-current.json'
TEMPLATE = A / 'phase-prefill-speed-current.json'
ORDER = (1, 0, 0, 1, 1, 0)  # new,old,old,new,new,old
PAIRS = ((1, 0), (2, 3), (5, 4))  # old/new
require, digest, write = runtime.require, runtime.digest, runtime.write


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def recovered_assets():
    return gen.committed(RESULT)['historical_regression_audit']['runtime_dependency_audit']['historical_asset_recovery']


def scope():
    return gen.committed(RESULT)['historical_regression_audit']['userland_crossover']


def save(row):
    container = json.loads(RESULT.read_text())
    container['historical_regression_audit']['userland_crossover'] = compact(row)
    write(RESULT, container)


def compact(row):
    return {k: v for k, v in row.items() if k not in ('benchmark_pins', 'models', 'commands', 'environment_template')}


def environment(row, root, arm, trace=False):
    base = row['environment_template']; old = base['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(root)) for k, v in base.items()}
    require(not any(k in env for k in ('LD_PRELOAD', 'STRATA_TRACE', 'STRATA_STATE_HASH',
        'STRATA_SNAPSHOT_VERIFY', 'STRATA_DECODE_TIMING', 'STRATA_VERIFY_PROFILE',
        'STRATA_CPU_WORK_CENSUS', 'ZE_ENABLE_ALT_DRIVERS')), 'observer/provider override in template')
    if arm == 0:
        env['LD_LIBRARY_PATH'] = row['historical_library_root'] + ':' + env['LD_LIBRARY_PATH']
    env.update(ZE_ENABLE_ALT_DRIVERS=row['drivers'][arm]['path'],
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root/'validation.log'))
    if trace:
        env.update(LD_PRELOAD=row['preload']['library']['path'],
            STRATA_UR_PROBE_EXPECTED_LOADER=row['preload']['loader']['path'],
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root/'ur.trace'))
    return env


def check_maps(paths, allowed, expected, require_driver=True):
    require(paths <= set(allowed), 'unplanned executable provider: ' + repr(paths-set(allowed)))
    drivers = {p for p in paths if '/libze_intel_gpu.so.' in p}
    require(not drivers or drivers == {expected}, 'wrong or multiple NEO providers')
    if require_driver:
        require(drivers == {expected}, 'selected NEO was not observed')


def rates(requests, hashes):
    require(len(requests) == 4, 'four long requests required')
    out = []
    for sample, expected in zip(requests, hashes):
        require(sample['ok'] and sample['eligible'] and sample['stream_complete'] and
            sample['thinking_requested'] and sample['reasoning_observed'] and
            sample['request_sha256'] == expected and
            sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == 117489 and
            sample['actual_reused_tokens'] == 0 and sample['generated_tokens'] == 24 and
            sample['prompt_tps_source'] == 'engine' and
            type(sample['prompt_ms']) in (int, float) and 10000 <= sample['prompt_ms'] < 1500000,
            'workload/timing eligibility differs')
        out.append(117489000 / sample['prompt_ms'])
    return out


def summarize(processes):
    require(len(processes) == 6 and [p['arm'] for p in processes] == list(ORDER) and
        all(p['passed'] and p['ownership_closed'] for p in processes), 'closed ordered cohort required')
    medians = [statistics.median(p['tok_s'][1:]) for p in processes]
    ratios = [medians[a]/medians[b] for a, b in PAIRS]
    gm = statistics.geometric_mean(ratios)
    return dict(subsequent_process_medians=medians, old_over_new_pair_ratios=ratios,
        ratio_geomean=gm, old_faster_screen=all(x > 1 for x in ratios) and gm >= 1.01,
        new_faster_screen=all(x < 1 for x in ratios) and gm <= 1/1.01,
        limits='NEO/GMM/IGDML/zlib vary together;other pinned original ELFs/kernel stay fixed. '
            'Not individual-library attribution,historical clock/contention reproduction,hardware saturation,'
            'decode speed,runtime adoption or ordinary-runtime-teardown qualification.')


def current(row):
    require(scope() == compact(row), 'commit exact crossover plan/outcome')
    require(gen.benchmark_pins(row['source_commit']) == row['benchmark_pins'], 'benchmark/controller changed')
    require(digest(TEMPLATE) == row['template_sha256'] and digest(runtime.RESULT) == row['runtime_sha256'],
        'qualified template/runtime changed')
    require(row['environment_template'] == gen.committed(TEMPLATE)['environment_template'] and
        fingerprint(row['models']) == row['model_fingerprint'], 'private environment/model proof changed')
    rt = gen.committed(runtime.RESULT)
    require(rt['passed'] and rt['ownership_closed'] and not rt['active'] and
        rt['boot'] == row['boot'] and rt['driver'] == runtime.guards.driver_guard() and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'runtime/boot changed')
    for p, sha in rt['runtime_files'].items():
        require(digest(p) == sha, 'runtime provider changed: ' + p)
    require(fingerprint(recovered_assets()) == row['historical_assets_fingerprint'], 'historical asset proof changed')
    for files in row['variant_runtime_files']:
        for path, sha in files.items():
            require(digest(path) == sha, 'variant runtime asset changed: ' + path)
    for package in row['packages']:
        require(digest(package['path']) == package['sha256'], 'historical package changed')
    for q in recovered_assets()['packages']:
        require((Path(row['historical_library_root']) / q['alias']).resolve() == Path(q['library_path']),
                'historical library alias changed')
    for artifact in [row['binary'], *row['fixtures'].values(), *row['drivers']]:
        require(digest(artifact['path']) == artifact['sha256'], 'native/driver artifact changed')
    require(digest(row['build_proof']['receipt']['path']) == row['build_proof']['receipt']['sha256'],
        'original build proof changed')
    for i, plan in enumerate(row['process_plans']):
        root=ROOT/str(i);config=copy.deepcopy(row['configuration']);config['log']=str(root/'engine.stderr')
        require(plan['arm']==ORDER[i] and json.loads((root/'config.json').read_text())==config and
            json.loads((root/'plan.json').read_text())==dict(binary=row['binary'],native_validation_probe=row['preload']) and
            all(digest(p)==sha for p,sha in plan['inputs'].items()),'generated native inputs changed')
    models.model_pins(row['configuration'], row['models'])
    return rt


def prepare():
    require(not ROOT.exists() and 'userland_crossover' not in gen.committed(RESULT)['historical_regression_audit'],
            'fresh crossover only;never retry a closed root')
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    full, proof = runtime.selected.baseline()
    rt = gen.committed(runtime.RESULT); runtime.current(rt)
    template = gen.committed(TEMPLATE)
    previous = gen.committed(A/'restored-phase-bench-current.json')
    require(previous['stage'] == 'closed' and previous['ownership_closed'] and not previous['active'],
        'close prior model comparison first')
    config = copy.deepcopy(previous['configuration'])
    model_pins = models.model_pins(config, json.loads(gen.git('show', '1e676d554:' +
        str(TEMPLATE.relative_to(P))))['model_assets'])
    assets = recovered_assets()
    require(assets['all_assets_recovered'] and not assets['GPU_executed'] and
            not assets['system_packages_changed'], 'recover historical vendor assets first')
    original = {q['name']: q for q in assets['packages']}
    require(set(original) == {'neo', 'gmm', 'metrics', 'zlib'} and
            all(q['matched_historical_ELF'] for q in original.values()), 'historical dependencies incomplete')
    historical = Path(assets['root']) / 'lib'
    for q in original.values():
        require(digest(q['package_path']) == q['package_sha256'] and
                digest(q['library_path']) == q['library_sha256'] and
                (historical / q['alias']).resolve() == Path(q['library_path']), 'historical package/ELF/alias changed')
    new_driver = '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39758'
    drivers = [dict(path=original['neo']['library_path'], sha256=original['neo']['library_sha256']),
               dict(path=new_driver, sha256=rt['runtime_files'][new_driver])]
    packages = [dict(path=q['package_path'], sha256=q['package_sha256']) for q in original.values()]
    old_files = dict(rt['runtime_files'])
    for path in list(old_files):
        if any(key in path for key in ('/libze_intel_gpu.so.', '/libigdgmm.so.', '/libigdml.so.',
                                      '/libz.so.', '/libmetee.so.')):
            del old_files[path]
    old_files.update({q['library_path']: q['library_sha256'] for q in original.values()})
    variants = [old_files, dict(rt['runtime_files'])]
    preload = rt['preload']; preload = dict(library=preload['library'], loader=dict(
        path=preload['loader'], sha256=rt['runtime_files'][preload['loader']]))
    config['args'][config['args'].index('--expert-profile')+1] = str(P/'data/expert-profile.bin')
    require(digest(P/'data/expert-profile.bin') == '8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf',
        'expert profile changed')
    row = dict(stage='prepared', source_commit=revision, benchmark_pins=gen.benchmark_pins(revision),
        template_sha256=digest(TEMPLATE), runtime_sha256=digest(runtime.RESULT),
        environment_template=template['environment_template'], configuration=config,
        binary=full['binary'], fixtures=full['fixtures'], build_proof=proof, preload=preload,
        drivers=drivers, packages=packages, historical_library_root=str(historical),
        historical_assets_fingerprint=fingerprint(assets), variant_runtime_files=variants,
        boot=rt['boot'], driver=rt['driver'], models=model_pins,
        model_fingerprint=fingerprint(model_pins), request_hashes=[v['request_sha256'] for v in previous['processes'][0]['requests']],
        plan=dict(order=list(ORDER), pairs=[list(p) for p in PAIRS],
            question='Does the exact historical NEO/GMM/IGDML/zlib set restore original6e40 prefill speed?',
            constants='Same original6e40 binary,1256-slot decode layout,131072 resident INT8 context,8192 chunks, '
                'BF16/vLLM/mode10,384 packed expert bank,Q8 MTP,5 CPU workers,all safety settings. '
                'One short warmup then four117489-token thinking-on requests,24 output tokens,no prefix reuse.',
            gate='Before long model work,both dependency arms pass host-only canary,submission,graph0 and graph1 '
                'fixtures with pre-main parameter/API validation,numerical/guard markers,actual mapped NEO, '
                'normal owned exit and no fresh fault. Stop first failure,no retry.',
            decision='Six independent processes,new/old/old/new/new/old. Keep first use separate. '
                'Use subsequent-three median;old faster requires all three old/new ratios>1 and geomean>=1.01; '
                'new faster analogously. Otherwise unresolved. No speed-based stop or replacement. '
                'Commit every closed outcome before advancing. No inference or runtime adoption.',
            measurement='Official bench.sh --prefill-only --prefill-tokens98304 --think --runs4 with identical '
                'request gates and client in both arms. Engine clock is unchanged original6e40/libstdc++, '
                'qualified65536-pair endpoint records remain in both historical cohorts;requests>=10000ms. '
                'Warning-only parameter validation and retained-loader preload charged to both. '
                'No profiler,census,timer,state readbacks or resource sampling.',
            limits='NEO/GMM/IGDML/zlib vary together;27 original ELF hashes,loader and kernel stay fixed. '
                'This does not reproduce historical effective clocks,contention,paging or process history. '
                'Explicit alternative driver selection is applied to BOTH arms '
                'before exec and observed from maps. No system install,service change,reset or reboot.'),
        budgets=dict(wall=3000,cpu=72000,rss_cap=104<<30,text_cap=32<<20,file_cap=64<<20,total_cap=128<<20),
        qualification=[], processes=[], process_plans=[], active=False, ownership_closed=True,
        retention='Keep source/recipe,individual samples,provider/workload/fault checks and decision. '
            'Private old vendor ELFs/packages and successful raw/JIT output retire after crossover analysis; '
            'selected engine and twelve fixtures unchanged. oneMKL captures and14 legacy source holds remain separate.')
    ROOT.mkdir(mode=0o700)
    ports=set()
    for i,arm in enumerate(ORDER):
        root=ROOT/str(i);root.mkdir()
        for name in ('tmp','cache'):(root/name).mkdir()
        cfg=copy.deepcopy(config);cfg['log']=str(root/'engine.stderr')
        write(root/'config.json',cfg);write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=preload))
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        require(port not in ports,'duplicate prospective port;no retry');ports.add(port)
        command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
            '--prefill-only','--prefill-tokens','98304','--think','--runs','4','--profile-request-gates',
            '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]
        row['process_plans'].append(dict(arm=arm,root=str(root),command=command,
            inputs={str(root/name):digest(root/name) for name in ('config.json','plan.json')}))
    write(ROOT/'record.json', row); save(row)
    print('Prepared;commit the frozen plan before host/device execution.', flush=True)


def observe(io, paths, allowed, expected):
    while not io.exit_watch.poll(0) or not io.eof:
        io.poll()
        try:
            seen = runtime.executable_mappings(Path('/proc', str(io.proc.pid), 'maps').read_text())
        except (FileNotFoundError, ProcessLookupError):
            seen = set()
        check_maps(seen, allowed, expected, False); paths.update(seen)
        if select.select([io.proc.stdout], [], [], .01)[0]:
            io.read(); io.pending.clear()
    io.exit(1)


def journal_cursor():
    lines = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True, timeout=15)
    return next(s.split(': ', 1)[1] for s in lines.splitlines() if s.startswith('-- cursor: '))


def allowed_files(row, rt, arm):
    return dict(row['variant_runtime_files'][arm])


def qualify(row):
    rt = current(row)
    require(row['stage'] == 'prepared' and not (ROOT/'qualification').exists(), 'single-use qualification')
    initial = copy.deepcopy(row); outcome = dict(host_probe_accepted=False); owner = None; cursor = journal_cursor()
    root = ROOT/'qualification'; root.mkdir()
    def persist():
        write(root/'record.json', dict(outcome, commands=owner.commands if owner else [], active=bool(owner and owner.active)))
    try:
        owner = owners.Owner(root/'owned', persist)
        entry, _, _ = owner.run('host-controls', ['/usr/bin/python3','-B',str(P/'tools/sycl/test_prefill_userland_compare.py')],
            dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1'), P,
            wall=45,cpu=45,rss_cap=1<<30,text_cap=1<<20,file_cap=1<<20,total_cap=128<<20)
        require(owners.closed(entry) and entry['exit_code'] == 0, 'host controls failed')
        for arm in (1, 0):
            for name, fixture, args, marker in (
                ('canary',None,[],None),
                ('submission','sycl_submission_test',['--submission'],'fixture_status=PASS mode=submission'),
                ('graph0','sycl_graph_envelope_test',['--profile=0'],'numerical=PASS guards=PASS aliases=PASS checked_teardown=PASS support=NOT_ENABLED'),
                ('graph1','sycl_graph_envelope_test',['--profile=1'],'numerical=PASS guards=PASS aliases=PASS checked_teardown=PASS support=SUPPORTED')):
                case = root/(str(arm)+'-'+name);case.mkdir();(case/'tmp').mkdir();(case/'cache').mkdir()
                env = environment(row, case, arm, True)
                binary = row['fixtures'][fixture] if fixture else rt['preload']['canary']
                seen = set()
                allowed = allowed_files(row,rt,arm);allowed[str(Path(binary['path']).resolve())] = binary['sha256']
                entry, out, err = owner.run(str(arm)+'-'+name,[binary['path'],*args],env,P,
                    interaction=lambda io, entry: observe(io,seen,allowed,row['drivers'][arm]['path']),
                    wall=90,cpu=120,rss_cap=4<<30,text_cap=16<<20,file_cap=32<<20,total_cap=128<<20)
                require(owners.closed(entry) and entry['exit_code']==0, 'native fixture failed')
                if fixture:
                    require(marker in out.read_text() and any('/libigdgmm.so.' in p for p in seen),
                            'fixture markers or actual GMM mapping missing')
                else:
                    require(json.loads(out.read_text()) == dict(pid=entry['pid'], initialized=0,
                        checked=46, expected=46, version=(1<<32)-1, closed=0), 'host-only canary differs')
                check_maps(seen,allowed,row['drivers'][arm]['path'], fixture is not None)
                runtime.validation.preload_evidence(err.read_text(), entry['pid'])
                require(not (case/'validation.log').exists() or (case/'validation.log').stat().st_size==0, 'parameter warning')
                if fixture:
                    require((case/'ur.trace').exists() and 'urEnqueue' in (case/'ur.trace').read_text(), 'enqueue trace missing')
                probe=dict(host_probe_accepted=True);runtime.validation.finish(probe,owner,cursor,row['boot'])
                require(probe['passed'] and probe['driver_after']==row['driver'], 'fault/ownership gate failed')
                row['qualification'].append(dict(arm=arm,case=name,passed=True,provider=row['drivers'][arm],
                    mapped_variant_files=sorted(seen & set(row['variant_runtime_files'][arm])),
                    ownership_closed=True,fresh_faults=[]))
        current(initial);outcome['host_probe_accepted']=True
    except BaseException as e:
        outcome['error']=repr(e)
    finally:
        if owner: runtime.validation.finish(outcome,owner,cursor,row['boot'])
        row['ownership_closed']=outcome.get('ownership_closed',True)
        row['stage']='qualified' if outcome.get('passed') and len(row['qualification'])==8 else 'qualification-failed'
        row['qualification_outcome']=outcome;persist();write(ROOT/'record.json',row);save(row)
    print(json.dumps({'stage':row['stage'],'outcome':outcome}),flush=True)
    return int(row['stage']!='qualified')


def run(row):
    rt=current(row);require(row['stage'] in ('qualified','collecting') and len(row['processes'])<6 and
        all(x['passed'] for x in row['processes']), 'qualified ordered crossover only')
    initial=copy.deepcopy(row);i=len(row['processes']);arm=ORDER[i];root=ROOT/str(i)
    require(not (root/'owned').exists(),'existing attempt;inspect owner,never retry')
    config=json.loads((root/'config.json').read_text())
    env=environment(row,root,arm)
    command=row['process_plans'][i]['command']
    write(root/'inputs.json',dict(source_commit=row['source_commit'],command=command,environment=env,
        configuration=config,native_plan=json.loads((root/'plan.json').read_text())))
    cursor=journal_cursor();outcome=dict(arm=arm,passed=False,complete=False);owner=None
    def persist():
        write(root/'record.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    def interaction(io,leader):
        for request in range(4):
            for stage in ('begin','end'):
                while True:
                    line=io.line(1500)
                    if line.startswith('STRATA_PROFILE_GATE '):break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==dict(stage=stage,phase='prefill-only',run=request),'request gate differs')
                require(not (root/'validation.log').exists() or not (root/'validation.log').stat().st_size,'parameter warning')
                if stage=='begin':
                    claim=json.loads((root/'engine-start.json').read_text());pid=claim['native']['pid'];ident=owners.identity(pid)
                    require(ident and ident['start_ticks']==claim['native']['start_ticks'] and ident['sid']==leader['pid'] and
                        str(Path('/proc',str(pid),'exe').resolve())==str(Path(row['binary']['path']).resolve()),'native owner differs')
                    actual=dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                    expected=dict(env,LD_PRELOAD=row['preload']['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=row['preload']['loader']['path'])
                    require(all(actual.get(k.encode())==v.encode() for k,v in expected.items()) and
                        set(actual)-{k.encode() for k in expected}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'actual environment differs')
                    paths=runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                    allowed=allowed_files(row,rt,arm);allowed[str(Path(row['binary']['path']).resolve())]=row['binary']['sha256']
                    check_maps(paths,allowed,row['drivers'][arm]['path'])
                    require(any('/libigdgmm.so.' in p for p in paths), 'actual model GMM mapping missing')
                    outcome['mapped_variant_files'] = sorted(set(outcome.get('mapped_variant_files', [])) |
                        (paths & set(row['variant_runtime_files'][arm])))
                    outcome['verified_request_boundaries']=request+1
                io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic()<deadline,'benchmark teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner=owners.Owner(root/'owned',persist)
        entry,_,_=owner.run('bench',command,env,P,interaction=interaction,**row['budgets'])
        require(owners.closed(entry) and entry['exit_code']==0,'official benchmark/owner failed')
        result_file,=(root/'bench').glob('*/results.json');result=json.loads(result_file.read_text())
        samples=result['results']['prefill_only']['requests'];outcome['tok_s']=rates(samples,row['request_hashes'])
        require(result['complete'] and not result['results']['failures'] and outcome.get('verified_request_boundaries')==4,'incomplete benchmark')
        sys.path.insert(0,str(P/'tools'));from strata_bench import verify_request_journal
        journal,=(root/'bench').glob('*/requests.jsonl');verified=verify_request_journal(journal.parent)
        require(verified['valid'] and verified['complete'],'request journal incomplete')
        events=[json.loads(v) for v in journal.read_text().splitlines()]
        longs=[v for v in events if v['kind']=='result' and v['phase']=='prefill-only']
        require(len(longs)==4 and [dict(v['sample'],run=v['run']) for v in longs]==samples,'journal/request binding differs')
        claim=gen.native_end(root,dict(binary=row['binary'],native_validation_probe=row['preload']),dict(config=config),row['boot'],entry['pid'])
        text=(root/'engine.stderr').read_text();activation=runtime.validation.preload_evidence(text,claim['native']['pid'])
        require(text.count('strata prefill: layer-major candidate, tokens=117488 chunk=8192 residual_dtype=bf16')==4 and
            text.count('strata MoE window configuration: requested_mode=10 effective_mode=10')==4 and
            not any(s in text for s in ('strata trace:','strata decode timing:','STATE_HASH','SNAPSHOT_VERIFY',
                'UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','teardown uncertain')),'route/runtime differs')
        current(initial)
        if i==5:models.model_pins(config,row['models'],full=True)
        fields=('run','request_sha256','actual_prompt_tokens','actual_read_tokens','actual_reused_tokens','generated_tokens','prompt_ms','thinking_requested')
        outcome.update(requests=[{k:s[k] for k in fields} for s in samples],
            validation=activation,actual_provider=row['drivers'][arm],native_identity=claim['native'],
            journal_sha256=digest(journal),complete=True,host_probe_accepted=True)
    except BaseException as e:
        outcome['error']=repr(e)
        if (root/'engine.stderr').exists():outcome['error_excerpt']=(root/'engine.stderr').read_text()[-4096:]
    finally:
        if owner:runtime.validation.finish(outcome,owner,cursor,row['boot'])
        outcome['passed']=bool(outcome.get('passed') and outcome['complete'])
        if 'mapped_variant_files' in outcome:outcome['mapped_variant_files']=sorted(outcome['mapped_variant_files'])
        persist();row['processes'].append(outcome);row['ownership_closed']=outcome.get('ownership_closed',True)
        row['stage']='closed' if not outcome['passed'] or i==5 else 'collecting'
        if outcome['passed'] and i==5:row['analysis']=summarize(row['processes'])
        write(ROOT/'record.json',row);save(row)
    print(json.dumps({k:outcome.get(k) for k in ('arm','passed','complete','ownership_closed','tok_s','error')}),flush=True)
    return int(not outcome['passed'])


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']:prepare();return 0
        row=json.loads((ROOT/'record.json').read_text())
        if sys.argv[1:] == ['qualify']:return qualify(row)
        if sys.argv[1:] == ['run']:return run(row)
        raise RuntimeError('use prepare,qualify or run')


if __name__ == '__main__':
    sys.exit(main())
