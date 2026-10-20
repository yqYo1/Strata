#!/usr/bin/env python3
"""Main-owned finite producer/copy/drain diagnostic; no model run.

Commit source, prepare/commit the generated recipe, build/commit binary identity,
then run four traced smoke checks and twelve fixed measurement processes once.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import query_profile as q
import ur_validation as validation

gen, guards, supervision = q.gen, q.guards, q.supervision
P, B, A = q.P, q.B, q.A
ROOT = B / 'status-copy-pipeline-v4'
RESULT = A / 'status-copy-current.json'
FIXTURE = 'bench/sycl/status_copy.cpp'
CELLS = [('profiled', 'ordinary'), ('plain', 'usm'), ('plain', 'ordinary'), ('profiled', 'usm')]
ORDERS = [CELLS, list(reversed(CELLS)), CELLS[2:] + CELLS[:2]]


def digest(path):
    return supervision.sha(path)


def inputs(source):
    pins = gen.benchmark_pins(source)
    raw = gen.git('show', source + ':' + FIXTURE)
    guards.require(raw == (P / FIXTURE).read_bytes(), 'commit exact fixture input')
    pins[FIXTURE] = hashlib.sha256(raw).hexdigest()
    return pins


def runtime_check():
    settings = gen.committed(gen.SETTINGS)
    guards.require(set(q.DELTA) <= set(settings['runtime_elfs']), 'unknown qualified runtime delta')
    for path, item in settings['runtime_elfs'].items():
        expected = item['sha256']
        if path in q.DELTA:
            original, expected = q.DELTA[path]
            guards.require(original == item['sha256'], 'original baseline differs')
        guards.require(digest(path) == expected, 'runtime changed: ' + path)
    return settings


def admission(record):
    full, count = q.current_build()
    guards.require(full['source_commit'] == record['native_source_commit'] and
        digest(q.BUILD_RECORD) == record['native_build_record_sha256'] and
        count == record['native_inputs_recovered'], 'native source/build changed')
    pins = inputs(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and
        q.assets.fingerprint(pins) == record['source_fingerprint'] and
        digest(full['compiler']['path']) == record['compiler']['sha256'], 'fixture/controller/compiler changed')
    runtime_check()
    guards.require(digest(gen.SETTINGS) == record['settings_sha256'] and
        validation.preload_qualification() == record['validation_probe'], 'runtime/helper proof changed')
    guards.require(record['native_environment'].get('STRATA_UR_PROBE_EXPECTED_LOADER') ==
        record['validation_probe']['loader']['path'] and
        record['native_environment'].get('LD_PRELOAD') == record['validation_probe']['library']['path'],
        'configure exact native-child validation provider before execution')


def parse_output(text, profiling, host, mode):
    rows=[json.loads(line) for line in text.splitlines()]
    count,warmups=(1,0) if mode=='smoke' else (6,1)
    guards.require(len(rows)==count+warmups+2,'missing/extra protocol rows')
    begin=rows[0]
    expected=dict(type='begin',schema=4,profiling=profiling=='profiled',host=host,mode=mode,
        samples=count,warmups=warmups,fills=8,fill_bytes=(64<<10 if mode=='smoke' else 128<<20),
        copy_bytes=4,precopy_status_query=False)
    guards.require(set(begin)==set(expected)|{'clock_pair_cpu_s','clock_pair_wall_s','timer_resolution_ns'} and
        all(type(begin[k]) is type(v) and begin[k]==v for k,v in expected.items()),'begin protocol differs')
    def duration(v):return type(v) in (int,float) and math.isfinite(v) and v>=0
    guards.require(all(duration(begin[k]) for k in ('clock_pair_cpu_s','clock_pair_wall_s')) and
        type(begin['timer_resolution_ns']) is int and begin['timer_resolution_ns']>0,'invalid clock reference')
    samples,warmup=[],[]
    for rep,row in zip(range(-warmups,count),rows[1:-1]):
        expected=dict(type='warmup' if rep<0 else 'sample',rep=rep,completed_events=8,payload_and_guards_passed=True)
        durations={f'{part}_{kind}_s' for part in ('producer','enqueue','drain','whole') for kind in ('cpu','wall')}
        flags={f'{part}_timing_eligible' if part!='whole' else 'timing_eligible' for part in ('producer','enqueue','drain','whole')}
        guards.require(set(row)==set(expected)|durations|flags|{'gpu_fill_service_ns','gpu_fill_span_ns','gpu_clock_eligible'} and
            all(type(row[k]) is type(v) and row[k]==v for k,v in expected.items()) and
            all(duration(row[k]) for k in durations),'invalid sample/count/guard/clock')
        for part in ('producer','enqueue','drain','whole'):
            key=part+'_timing_eligible' if part!='whole' else 'timing_eligible'
            eligible=bool(row[part+'_wall_s']>=20e-6 and row[part+'_wall_s']>=100*begin['clock_pair_wall_s'] and
                row[part+'_cpu_s']>=100*begin['clock_pair_cpu_s'])
            guards.require(type(row[key]) is bool and row[key]==eligible,'timer eligibility mismatch')
        for kind in ('cpu','wall'):
            total=sum(row[p+'_'+kind+'_s'] for p in ('producer','enqueue','drain'))
            guards.require(math.isclose(total,row['whole_'+kind+'_s'],rel_tol=1e-8,abs_tol=1e-10),'phase conservation failed')
        guards.require(all(type(row[k]) is int and row[k]>=0 for k in ('gpu_fill_service_ns','gpu_fill_span_ns')) and
            type(row['gpu_clock_eligible']) is bool,'invalid device clock')
        if profiling=='plain':guards.require(not row['gpu_clock_eligible'] and row['gpu_fill_service_ns']==row['gpu_fill_span_ns']==0,'unprofiled event queried')
        if row['gpu_clock_eligible']:
            guards.require(profiling=='profiled' and 0<row['gpu_fill_service_ns']<=row['gpu_fill_span_ns'] and
                row['gpu_fill_span_ns']*1e-9<=row['whole_wall_s']+max(1e-6,8e-9*begin['timer_resolution_ns']),
                'invalid GPU timing conservation')
        (warmup if rep<0 else samples).append(row)
    guards.require(rows[-1]==dict(type='end',passed=True,owners_closed=True) and
        type(rows[-1]['passed']) is bool and type(rows[-1]['owners_closed']) is bool,'incomplete native closure')
    return begin,samples,warmup


def prepare():
    guards.require(not ROOT.exists(),'inspect prior job; no retry')
    previous=gen.committed(RESULT)
    guards.require(previous['schema']=='strata-status-query-observer-v3' and previous['stage']=='closed' and
        previous['complete'] and previous['passed'] and previous['ownership_closed'] and previous['fresh_faults']==[] and
        not previous['active'] and previous['diagnostic_timing_qualified'] and previous['binary']['retired'],
        'preserve the closed completed-event diagnostic and retire its unused standalone build')
    raw=gen.git('show','HEAD:'+str(RESULT.relative_to(P)))
    previous_ref=dict(commit=gen.git('rev-parse','HEAD').decode().strip(),path=str(RESULT.relative_to(P)),
        sha256=hashlib.sha256(raw).hexdigest(),outcome='v3 completed-event cost18.29..25.15ns quiet,11.75..12.16us full trace; no model/queue adoption. Original v1/v2 pending failures remain in its Git lineage.')
    full,count=q.current_build();settings=runtime_check();source=gen.git('rev-parse','HEAD').decode().strip();pins=inputs(source)
    compiler=full['compiler'];guards.require(digest(compiler['path'])==compiler['sha256'],'compiler differs')
    env=dict(full['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    native=dict(settings['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),
        UR_LOG_TRACING='level:info;flush:info;output:stderr',UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
        CUDA_MODULE_LOADING='EAGER',STRATA_SYCL_COMMAND_PROFILE='0')
    probe=validation.preload_qualification();native.update(LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'])
    command=[compiler['path'],'-O3','-DNDEBUG','-std=c++20','-fsycl','-fsycl-id-queries-range=size_t','-DSTRATA_ENABLE_SYCL=1',
        '-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),str(P/FIXTURE),
        *[str(P/('src/sycl/'+n+'.cpp')) for n in ('runtime','compat','command_profile')],'-o',str(ROOT/'status_copy')]
    providers=('libsycl.so','libur_loader.so','libur_adapter_level_zero_v2.so','libze_loader.so','libze_intel_gpu.so')
    paths=[str(Path(p).resolve()) for p in settings['runtime_elfs'] if any(Path(p).name.startswith(n) for n in providers)]
    guards.require(len(paths)==len(set(paths))==5,'ambiguous providers')
    record=dict(schema='strata-status-copy-pipeline-v4',previous_component=previous_ref,stage='prepared',source_commit=source,
        source_count=len(pins),source_fingerprint=q.assets.fingerprint(pins),native_source_commit=full['source_commit'],
        native_inputs_recovered=count,native_build_record_sha256=digest(q.BUILD_RECORD),compiler=compiler,
        settings_sha256=digest(gen.SETTINGS),compile_argv=command,compile_environment=env,native_environment=native,
        validation_probe=probe,expected_loaded_paths=sorted(paths),smoke_cells=CELLS,process_orders=ORDERS,
        gates='Four traced smoke cells first,8x64KiB fills each. Then three fixed four-cell validation-only cohorts: profile true/false x ordinary/USM-host destination. Each process has one separate warmup and6 samples of exactly8x128MiB fills,4-byte production compat D2H and checked drain. Retain all8 events, no explicit status/profiling query before copy. Record disjoint producer/enqueue/drain CPU/wall and whole; exact counts/guards, input/recipe/provider/first-init proof, native exit/owner/fault/boot checks. Whole/phase timing eligible only at>=20us wall and>=100x median128 clock-pair CPU/wall costs. Preserve below-floor phases as ineligible; no work enlargement/retry. Profiled events queried after drain only, positive ordered resolution-qualified GPU intervals with checked sum/span and span<=whole+max(1us,8xresolution); GPU eligibility separate from host execution.',
        comparison_gate='Use six steady samples per process median, three cohort ratios per comparison; warmups separate. For each host kind, plain/profiled whole-wall geomean<=0.95 and all3 ratios<1 qualifies profiling-cost mechanism screen. For each queue kind, USM/ordinary same rule qualifies destination mechanism screen. Require all72 whole-timing samples eligible. These screens permit only a production-consumer audit/next source candidate, not a model adoption or third engine build. Moving time from enqueue into drain alone is not a throughput gain.',
        budgets=dict(compile_wall_s=240,native_process_wall_s=45,text_bytes_per_command=16<<20,total_raw_bytes=64<<20,
            device_bytes=(128<<20)+140,measured_processes=12,steady_samples=72,warmup_samples=12,smoke_processes=4,
            logical_fill_bytes_per_sample=8*(128<<20)),
        limits='Finite producer/copy/drain source attribution only. No verified-pending claim or relaxed old pending gate. Requested fill bytes are not physical GDDR traffic/bandwidth. Profiled GPU intervals do not establish arithmetic saturation or model critical-path time. Quiet API counts are unobserved. No production queue default/math/elapsed-time consumer change.',
        retained_generations='396 fastest fallback and3b qualified status engine remain unchanged; both consumed by next model/source qualification. Closed standalone v2/v3 roots already retired; no third engine generation.',
        retention='Keep72 useful steady samples,12 separate warmups, clocks, GPU aggregates, component comparison/source/owner/fault decisions. Retire normal raw output and standalone root after compact evidence, exact source/recipe recovery and consumer closure.',
        complete=False,passed=False,active=False,GPU_executed=False)
    RESULT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(prepared=True,source=source,source_count=len(pins),GPU_executed=False)))


def compare(rows):
    eligible=all(r['timing_qualified'] for r in rows)
    by_key={(r['cohort'],*r['cell']):r for r in rows}
    result={}
    def criterion(name,pairs):
        ratios=[]
        for cohort,num,den in pairs:
            numerator=by_key[(cohort,*num)]['medians']['whole']['wall_s']
            denominator=by_key[(cohort,*den)]['medians']['whole']['wall_s']
            ratios.append(numerator/denominator if denominator>0 else None)
        mean=math.exp(sum(math.log(r) for r in ratios)/3) if all(r is not None and r>0 for r in ratios) else None
        result[name]=dict(ratios=ratios,latency_ratio_geomean=mean,eligible=eligible,
            mechanism_screen_passed=bool(eligible and mean is not None and mean<=.95 and all(r<1 for r in ratios)))
    for host in ('ordinary','usm'):criterion('plain_over_profiled_'+host,[(i,('plain',host),('profiled',host)) for i in range(3)])
    for queue in ('plain','profiled'):criterion('usm_over_ordinary_'+queue,[(i,(queue,'usm'),(queue,'ordinary')) for i in range(3)])
    return result


def execute(build):
    record=gen.committed(RESULT);guards.require(record['stage']==('prepared' if build else 'built'),'closed/mismatched stage; no retry');admission(record)
    if build:
        guards.require(not ROOT.exists(),'build root exists; inspect handle');ROOT.mkdir(mode=0o700)
        for name in ('tmp','cache'):(ROOT/name).mkdir()
    else:
        guards.require(digest(record['binary']['path'])==record['binary']['sha256'],'executable differs')
        guards.require(not (ROOT/'run.json').exists(),'run attempted; no retry')
    owner=None;output=ROOT/('build.json' if build else 'run.json')
    def persist():
        record['active']=bool(owner and owner.active is not None)
        output.write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    owner=supervision.Owner(ROOT/('owned-build' if build else 'owned-run'),persist);record.update(complete=False,passed=False)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();cursor=None;persist()
    try:
        if build:
            entry,out,err=owner.run('compile',record['compile_argv'],record['compile_environment'],P,
                wall=240,cpu=240,rss_cap=4<<30,text_cap=16<<20,file_cap=128<<20,total_cap=1<<30)
            guards.require(supervision.closed(entry) and entry.get('exit_code')==0,'compile failed: '+err.read_text()[-4096:])
            executable=ROOT/'status_copy';record.update(stage='built',binary=dict(path=str(executable),bytes=executable.stat().st_size,sha256=digest(executable)))
        else:
            guards.driver_guard();journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
            cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
            record.update(GPU_executed=True,cells_results=[],smoke_results=[])
            planned=[('smoke',-1,i,cell) for i,cell in enumerate(record['smoke_cells'])]+[
                ('measure',cohort,index,cell) for cohort,order in enumerate(record['process_orders']) for index,cell in enumerate(order)]
            for mode,cohort,index,cell in planned:
                runtime_check();guards.driver_guard()
                env=dict(record['native_environment'],UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if mode=='smoke' else 'UR_LAYER_PARAMETER_VALIDATION')
                entry,out,err=owner.run(f'{mode}{cohort}-{index}',[record['binary']['path'],*cell,mode],env,P,
                    wall=45,cpu=45,rss_cap=2<<30,text_cap=16<<20,file_cap=16<<20,total_cap=64<<20)
                guards.require(supervision.closed(entry) and entry.get('exit_code')==0,'cell failed: '+err.read_text()[-2048:])
                begin,samples,warmup=parse_output(out.read_text(),*cell,mode)
                log=err.read_text();proof=validation.preload_evidence(log,entry['pid'])
                guards.require(proof['loader']==record['validation_probe']['loader']['path'] and 'strata UR validation failure:' not in log,'validation provider/error')
                paths=re.findall(r'^status query loaded: (.+)$',log,re.M);guards.require(sorted(paths)==record['expected_loaded_paths'],'actual provider mismatch')
                api_counts={name:len(re.findall(r'^\s*---> '+name+r'\s*$',log,re.M)) for name in
                    ('urEventGetInfo','urEventGetProfilingInfo','urDeviceGetGlobalTimestamps','urEventRelease','urQueueFinish','urEnqueueUSMFill','urEnqueueUSMMemcpy')}
                guards.require((api_counts['urEnqueueUSMFill']>0)==(mode=='smoke'),'trace presence/absence mismatch')
                result=dict(cell=cell,cohort=cohort,order_index=index,clock_reference=begin,samples=samples,warmup=warmup,
                    timing_qualified=all(s['timing_eligible'] for s in samples),
                    gpu_timing_qualified=all(s['gpu_clock_eligible'] for s in samples) if cell[0]=='profiled' else None,
                    api_counts=api_counts,native_pid=entry['pid'],validation_activation=proof,loaded_paths=paths,native_exit=0,ownership_closed=True)
                if mode=='measure':
                    result['medians']={part:{kind+'_s':statistics.median(s[part+'_'+kind+'_s'] for s in samples) for kind in ('cpu','wall')} for part in ('producer','enqueue','drain','whole')}
                    record['cells_results'].append(result)
                else:
                    start=log.index('strata finite copy: sample=0 begin\n');end=log.index('strata finite copy: sample=0 end\n',start)
                    result['first_pipeline_api_sequence']=re.findall(r'^\s*---> (\w+)\s*$',log[start:end],re.M)
                    record['smoke_results'].append(result)
                persist()
            record.update(stage='closed',diagnostic_timing_qualified=all(r['timing_qualified'] for r in record['cells_results']),
                gpu_timing_qualified=all(r['gpu_timing_qualified'] for r in record['cells_results'] if r['cell'][0]=='profiled'),
                comparisons=compare(record['cells_results']),model_adoption_qualified=False)
        admission(record);record.update(complete=True,passed=True)
    except BaseException as error:record['error']=str(error)
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands);record['passed']=bool(record['passed'] and record['ownership_closed'])
        if not build and cursor is not None:
            record.update(fresh_faults=None,boot_unchanged=False)
            try:
                text=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                record['fresh_faults']=[x for x in text.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
                record['driver_after']=guards.driver_guard();record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            except Exception as error:record['final_probe_error']=str(error)
            record['passed']=bool(record['passed'] and record['fresh_faults']==[] and record['boot_unchanged'] and 'final_probe_error' not in record)
        persist();RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('stage','complete','passed','ownership_closed','error')}));return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','build','run'));action=parser.parse_args().action
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        if action=='prepare':return prepare()
        return execute(action=='build')


if __name__=='__main__':sys.exit(main())
