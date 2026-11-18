#!/usr/bin/env python3
"""Build and measure the isolated BF16 V tile comparison. Main only."""

import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import coherent_model_basis as basis
import measurement_guards as guards
import measurement_owner as owner
import qsa_cached_subgroup as diagnostics
import qsa_v_tile_sources as proposal
import ur_validation as validation

P, B, RESULT = basis.P, basis.B, basis.RESULT
ROOT, KEY = B / 'qsa-v-tile-v2', 'qsa_v_tile_v2'
THIS = 'tools/sycl/qsa_v_tile_trial.py'
PYTHON = basis.reference.model.PYTHON
COMPILER = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
INPUTS = ['src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
    'src/kernels/sycl/prompt_attention.cpp', proposal.PREFIX]
NUMERIC = 'tests/sycl/qsa_bf16_prefix.cpp'
BENCH = 'tests/sycl/qsa_bf16_prefix_layout_bench.cpp'
ADAPTER = 'bench/sycl/qsa_v_tile_experiment.hpp'
STAGES = ('build', 'smoke', 'numeric', '0', '1', '2')
ARMS = ('shared_late', 'separate_late', 'separate_early')
ORDERS = ((0,1,2),(2,1,0),(1,2,0),(0,2,1),(2,0,1),(1,0,2))
CASES = ((78453,8192,33,'gather'), (78453,8192,2051,'gather'), (78453,8192,2051,'clustered'))
BUILD_LIMIT = dict(wall=250, cpu=300, rss_cap=12 << 30, text_cap=8 << 20,
    file_cap=160 << 20, total_cap=512 << 20)
RUN_LIMIT = dict(wall=480, cpu=800, rss_cap=6 << 30, text_cap=32 << 20,
    file_cap=64 << 20, total_cap=112 << 20)
source = basis.original.coherent.build.common.source
require, sha, signature = basis.require, basis.sha, basis.signature
write = basis.reference.core.write


def frozen_source(revision):
    names = {THIS, ADAPTER, BENCH, NUMERIC, *INPUTS, *basis.POLICIES,
        'tools/test_qsa_v_tile_trial.py'}
    tree = source.git(P, 'ls-tree', '-r', '--name-only', revision, 'include', 'tools/sycl')
    names.update(tree.decode().splitlines())
    payload = dict(source.blobs(P, revision, sorted(names)))
    require(set(payload) == names and all((P/name).read_bytes() == data for name, data in payload.items()),
        'commit exact source, fixture, policy and controller inputs before preparing')
    return {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}


def platform():
    native, proof, speed = basis.basis()
    build = basis.original.coherent.build
    decode = validation.committed(RESULT, ('retained_builds_review_20261103', 'live_consumer_findings',
        'current_boundary_recheck', 'window_candidate_retirement'))
    require(decode['complete'] and decode['roots_absent'] and all(not Path(p).exists() for p in decode['roots'])
        and sha(decode['retained_decode']['path']) == decode['retained_decode']['sha256'],
        'retired window generation and preserved independent decode required')
    measured = validation.committed(RESULT, ('eager_request_profile_v2',))
    require(measured['stage'] == 'closed' and measured['ownership_closed']
        and measured['decision']['fresh_pair_within_observer_budget']
        and measured['closed_capture_retirement']['complete'], 'closed exact QSA resource evidence required')
    failed = validation.committed(RESULT, ('qsa_v_tile_v1',))
    require(failed['stage']=='failed' and failed['closed_partial_build']['complete']
        and failed['closed_partial_build']['all_children_reaped'] and not (B/'qsa-v-tile-v1').exists(),
        'preserve and retire the failed fixture compilation before this corrected trial')
    require(build.validation.committed(RESULT, ('qsa_prefix_calibration_v2',))['stage'] == 'closed',
        'original cap-2051 fixture baseline evidence required')
    manifest = speed['runtime_manifest']
    pin = dict(native_source=proof['original_native_source'], original_binary=proof['same_build_executables']['strata'],
        decode_binary=decode['retained_decode'], window_retirement_sha256=signature(decode),
        previous_profile_sha256=signature(measured), failed_fixture_sha256=signature(failed),
        external=native['external_inputs'],
        compiler=dict(path=COMPILER, sha256=native['toolchain'][COMPILER]),
        boot=speed['boot'], driver=speed['driver'], runtime_manifest=manifest,
        probe=speed['native_validation_probe'], environment_template=speed['environment_template'])
    return native, pin


def compile_recipe():
    flags = ['-std=c++20', '-O3', '-DNDEBUG', '-fsycl', '-fsycl-id-queries-range=size_t',
        '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13', '-DSTRATA_ENABLE_SYCL=1',
        '-DSTRATA_CPU_SERVICE_PROFILE=1', '-fno-fast-math', '-ffp-contract=off',
        '-I' + str(P/'include'), '-I' + str(P/'include/strata/sycl/compat'), '-I' + str(P/'bench/sycl')]
    paths = [P/name for name in INPUTS] + [ROOT/name for name in ('separate_late_v.cpp', 'separate_early_v.cpp')]
    paths += [P/NUMERIC, P/BENCH]
    objects = [ROOT/'build'/f'{i}.o' for i in range(len(paths))]
    commands=[]
    for path, obj in zip(paths, objects):
        extra = []
        if path == P/NUMERIC:
            extra = ['-Dqsa_bf16_prefix_batch=qsa_bf16_prefix_ordered_exp_batch',
                '-DSTRATA_QSA_PACKED_GUARD=17', '-DSTRATA_QSA_COMPARE_PREFILL_V_TILE=1']
        if path == P/BENCH:
            extra = ['-DSTRATA_QSA_BENCH_PREFILL_V_TILE=1']
        commands.append([COMPILER,*flags,*extra,'-MD','-MF',str(obj.with_suffix('.d')),
            '-c',str(path),'-o',str(obj)])
    for mode, obj in [('numeric',objects[-2]), ('benchmark',objects[-1])]:
        commands.append([COMPILER,*flags,*map(str,[*objects[:-2],obj]),'-Wl,-t','-pthread','-ldl',
            '-o',str(ROOT/'build'/('sycl_qsa_v_tile_'+mode))])
    return commands


def compare_samples(text, index, cases=CASES):
    arms = {k:i for i,k in enumerate(ARMS)}
    samples = []
    for line in text.splitlines():
        if line.startswith('sample '):
            fields = dict(word.split('=',1) for word in line.split()[1:])
            require(fields['status']==fields['exact_bits']==fields['guards']=='PASS',
                'correctness or guards failed')
            values = tuple(int(fields[n]) for n in ('cells','queries','cap'))
            case = (*values, fields['pattern'])
            j, k = int(fields['round']), int(fields['order'])
            arm = int(fields['variant'])
            require(0<=j<6 and 0<=k<3 and arm == ORDERS[(j+2*index)%6][k]
                and arms[fields['implementation']]==arm and int(fields['process'])==index
                and int(fields['compared'])==case[1]*24*256, 'timed ordinal, shape or actual arm differs')
            ms = float(fields['milliseconds'])
            require(0 < ms < 10000 and ms >= (0.5 if case[2]==33 else 10),
                'nonfinite or ineligible timed interval')
            samples.append((case,j,k,arm,ms))
    expect=[(case,j,k) for case in cases for j in range(6) for k in range(3)]
    require([(c,j,k) for c,j,k,a,ms in samples]==expect, 'incomplete or reordered timing coverage')
    markers = [x for x in text.splitlines() if x.startswith('case ')]
    require(len(markers)==len(cases) and all('samples=18 full_bits=PASS guards=PASS inputs=PASS status=PASS' in x
        for x in markers), 'immutable input checks are missing')
    result=[]
    for case in cases:
        times={arm:[ms for c,j,k,a,ms in samples if c==case and a==i] for i,arm in enumerate(ARMS)}
        result.append(dict(case=list(case), samples_ms=times,
            medians_ms={a:statistics.median(samples) for a,samples in times.items()}))
    return result


def decide(processes):
    require([p['mode'] for p in processes]==['0','1','2']
        and all(p['passed'] and p['ownership_closed'] for p in processes), 'three independent timing processes required')
    result=[]
    comparisons = [('storage', 1, 0), ('early_gather', 2, 1), ('combined', 2, 0)]
    for i,case in enumerate(CASES):
        require(all(p['points'][i]['case']==list(case) for p in processes),'comparison geometry differs')
        units={}
        for unit,new,old in comparisons:
            ratios=[p['points'][i]['medians_ms'][ARMS[new]]/p['points'][i]['medians_ms'][ARMS[old]]
                for p in processes]
            units[unit]=dict(process_ratios=ratios,geomean=statistics.geometric_mean(ratios),
                eligible=all(x<1 for x in ratios))
        result.append(dict(case=list(case), units=units))
    return dict(shapes=result, model_advance=False, engine_adopted=False,
        limits='Component admission, packing, attention and checked drain. Model throughput and hardware saturation unmeasured.')


def compact(row):
    out={k:v for k,v in row.items() if k not in ('sources','platform','compiled','actual_links','build_outputs')}
    for key in ('sources','platform','compiled','actual_links','build_outputs'):
        if key in row:
            out[key]=dict(count=len(row[key]),sha256=signature(row[key]))
    return out


def save(row):
    write(ROOT/'record.json',row)
    data=json.loads(RESULT.read_text())
    data[KEY]=compact(row)
    write(RESULT,data)


def admission(row):
    require(validation.committed(RESULT,(KEY,))==compact(row), 'commit this exact prepared stage first')
    require(row['sources']==frozen_source(row['source_commit']) and row['commands']==compile_recipe(),
        'source or recipe changed')
    _, pin=platform()
    require(pin==row['platform'], 'native source identity, runtime or provider changed')
    for name,data in proposal.generated(P/proposal.PREFIX).items():
        require((ROOT/name).read_bytes()==data and sha(ROOT/name)==row['projected'][name],
            'projected source changed')
    require(all(sha(v['path'])==v['sha256'] for v in row.get('binaries',{}).values()),
        'fixture executable differs')
    require(owner.artifact_bytes(ROOT)<row['whole_bytes'], 'whole cohort artifact limit')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh preparation only')
    revision=source.git(P,'rev-parse','HEAD').decode().strip()
    pins=frozen_source(revision)
    native, pin=platform()
    require(all(pins[name]==native['project']['source_sha256'][name] for name in INPUTS),
        'production comparison sources must equal original build')
    projected=proposal.generated(P/proposal.PREFIX)
    previous=validation.committed(RESULT, ('qsa_v_tile_v1',))
    require({n:hashlib.sha256(data).hexdigest() for n,data in projected.items()}==previous['projected'],
        'the fixture fix must leave both projected kernels byte-identical')
    compile_environment={k:v for k,v in native['recipe']['environment'].items()
        if k in ('PATH','LANG','LC_ALL','LD_LIBRARY_PATH')}
    compile_environment['TMPDIR']=str(ROOT/'tmp')
    ROOT.mkdir(mode=0o700)
    for n in ('build','tmp','compile'): (ROOT/n).mkdir(mode=0o700)
    for n,data in projected.items(): (ROOT/n).write_bytes(data)
    save(dict(stage='prepared',source_commit=revision,sources=pins,platform=pin,
        compile_environment=compile_environment,commands=compile_recipe(),
        projected={n:sha(ROOT/n) for n in projected},whole_bytes=1024<<20,
        scopes=[],active=False,passed=False,complete=False,ownership_closed=True,
        plan=dict(order=list(STAGES), shapes=[list(q) for q in CASES],
            reason='64K QSA prefix service about 14.4 s. Compare one extra V tile separately from earlier V gathering.',
            scope='Same owned prefix, BF16 precision and 20,036-byte shared local payload. Separate adds 16,384 bytes. Compare unchanged late gather at both sizes, then early vs late with identical allocation. No production dispatch edit.',
            build='Fresh two fixtures with identical flags and same three comparison paths. Counts as temporary third generation. Keep two independent phase binaries.',
            qualification='Traced small checks, then full 11-case numerical and state oracle including physical262144 address scope. No warnings, faults, leaks or output differences.',
            timing='Independent fresh processes 0..2. Six permutations per shape, one call per arm per permutation. Verify output and guards outside each complete admission-through-drain interval. Minimum 0.5ms at cap33, 10ms at cap2051. CLOCK_MONOTONIC measurement overhead median below 1us and p99 below 5us.',
            decision='All three process ratios below one per shape to advance any constituent. Preserve storage-only, early-only and combined decisions. No model qualification.',
            retention='Archive source and compact observations. Retire isolated fixtures when next consumer is resolved.')))
    print('Prepared exact QSA V tile source/build plans; commit before compiling.',flush=True)


def binary(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def audit_build(row, link_captures):
    allowed={str((P/name).resolve()):h for name,h in row['sources'].items()}
    allowed.update({str(ROOT/n):h for n,h in row['projected'].items()})
    external=row['platform']['external']
    require(sha(external['path'])==external['sha256'], 'compiler manifest changed')
    ext=json.loads(Path(external['path']).read_text())
    compiled={}
    base=json.loads((basis.original.ROOT/'record.json').read_text())['project']['source_sha256']
    dependencies=sorted((ROOT/'build').glob('*.d'))
    require(len(dependencies)==9 and len({p.stem for p in dependencies})==9, 'all nine TU depfiles required')
    for dep in dependencies:
        for name in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):
            path=Path(name).resolve();digest=sha(path)
            require((allowed|ext).get(str(path))==digest, 'unfrozen compiler input '+str(path))
            if path.is_relative_to(P/'include'):
                relative=str(path.relative_to(P))
                require(base.get(relative)==digest,'actual production header differs from original '+relative)
            compiled[str(path)]=digest
    require(len(compiled)>600, 'incomplete compiler dependency closure')
    links={}
    for log in link_captures:
        observed=0
        for line in log.splitlines():
            name=line.strip()
            if not name.startswith('/') or any(c.isspace() for c in name):
                continue
            path=Path(name).resolve()
            if path.is_relative_to(ROOT/'tmp') and not path.is_file():
                require(path.suffix=='.o','unexpected missing ephemeral link input')
                continue
            if path.is_relative_to(ROOT/'build'):
                require(path.is_file() and path.suffix=='.o','unknown built link input')
            else:
                require(path.is_file() and ext.get(str(path))==sha(path),'external link input not pinned '+str(path))
            links[str(path)]=sha(path);observed+=1
        require(observed>=12,'actual link-input capture incomplete')
    row.update(compiled=compiled, actual_links=links,
        build_outputs={str(path):sha(path) for path in (ROOT/'build').glob('*.o')},
        binaries={name:binary(ROOT/'build'/('sycl_qsa_v_tile_'+name)) for name in ('numeric','benchmark')})


def runtime_environment(row,path,trace):
    plat=row['platform'];old=plat['environment_template']['STRATA_MEASUREMENT_ROOT']
    env={k:v.replace(old,str(path)) for k,v in plat['environment_template'].items()}
    env.pop('STRATA_TRACE',None);env.pop('UR_LOG_TRACING',None)
    env.update(STRATA_TEST_MAIN_GATES='0',LD_PRELOAD=plat['probe']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=plat['probe']['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if trace else 'UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(path/'validation.log'))
    if trace:env['UR_LOG_TRACING']='level:info;flush:info;output:stderr'
    require(all(env[k]==v for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',
        EnableDirectSubmission='0',STRATA_SYCL_PREFILL_SAME_QUEUE='1',ZES_ENABLE_SYSMAN='0').items()),
        'qualified safety settings changed')
    return env


def run_stage(mode):
    row=json.loads((ROOT/'record.json').read_text());admission(row)
    require(row['stage'] not in ('closed','failed') and mode==STAGES[len(row['scopes'])],
        'only the next declared stage may run')
    if mode!='build':
        require(all(sha(p)==h for k in ('compiled','actual_links','build_outputs') for p,h in row[k].items()),
            'compiled input, object or linked provider changed')
    path=ROOT/mode if mode!='build' else ROOT/'compile'
    if mode!='build':
        path.mkdir(mode=0o700)
        for n in ('tmp','cache'): (path/n).mkdir(mode=0o700)
    child=owner.Owner(path/'owned',lambda:write(path/'execution.json',child.commands))
    result=dict(mode=mode,passed=False,ownership_closed=False)
    row.update(active=True,ownership_closed=False);save(row)
    cursor=None
    try:
        if mode=='build':
            outputs=[];deadline=time.monotonic()+1500
            for i,command in enumerate(row['commands']):
                print('Compile/link',i+1,'of',len(row['commands']),flush=True)
                wall=min(deadline-time.monotonic(),BUILD_LIMIT['wall'])
                require(wall>1,'whole build deadline')
                entry,out,err=child.run(str(i),command,row['compile_environment'],P,
                    **dict(BUILD_LIMIT,wall=wall))
                require(owner.closed(entry) and entry['exit_code']==0,
                    'build failed at command '+str(i)+': '+err.read_text()[-2200:])
                if '-Wl,-t' in command:outputs.append(out.read_text())
            audit_build(row,outputs)
            result.update(passed=True,GPU_executed=False)
        else:
            cursor=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],
                text=True,timeout=15).strip().split('-- cursor: ')[-1]
            require(cursor and '\n' not in cursor,'kernel cursor required')
            args={'smoke':[('numeric',['--smoke']),('benchmark',['--smoke'])],
                'numeric':[('numeric',['--component'])]}.get(mode,[('benchmark',['--component',mode])])
            if mode.isdecimal():
                ticks=[]
                for _ in range(10000):
                    begin=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                    ticks.append(time.clock_gettime_ns(time.CLOCK_MONOTONIC)-begin)
                result.update(clock_median_ns=statistics.median(ticks),clock_p99_ns=sorted(ticks)[9899])
                require(result['clock_median_ns']<1000 and result['clock_p99_ns']<5000,'timer overhead screen')
            checks=[]
            for name,options in args:
                env=runtime_environment(row,path,mode=='smoke')
                entry,out,err=child.run(name,[row['binaries'][name]['path'],*options],env,P,**RUN_LIMIT)
                require(owner.closed(entry) and entry['exit_code']==0,
                    'fixture failed: '+err.read_text()[-1600:])
                logs=err.read_text();text=out.read_text()
                probe=validation.preload_evidence(logs,entry['pid']);diagnostics.diagnostics(logs,entry['pid'])
                require(probe['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER']
                    and (not (path/'validation.log').exists() or not (path/'validation.log').stat().st_size)
                    and (mode!='smoke' or 'urEnqueue' in logs)
                    and 'fixture_status=PASS' in text, 'native validation, trace or fixture completion differs')
                if mode=='numeric':
                    numeric_lines=[x for x in text.splitlines() if x.startswith('case cells=')]
                    require(len(numeric_lines)==11 and any('cells=262144 live_cells=262144' in x for x in numeric_lines),
                        'full 11-case numeric/physical address check missing')
                    result['numeric_cases']=len(numeric_lines)
                if mode.isdecimal():result['points']=compare_samples(text,int(mode))
                checks.append(dict(role=name,pid=entry['pid'],first_init=probe,normal_exit=0))
            result.update(passed=True,host_probe_accepted=True,GPU_executed=True,checks=checks)
    except BaseException as error:
        result.update(error=repr(error),passed=False)
    finally:
        if cursor:
            validation.finish(result,child,cursor,row['platform']['boot'])
            result['passed']=bool(result['passed'] and 'error' not in result
                and result['driver_after']==row['platform']['driver'])
        else:
            result['ownership_closed']=child.active is None and all(owner.closed(c) for c in child.commands)
        result['passed']=bool(result['passed'] and result['ownership_closed'])
        row['scopes'].append(result)
        row.update(stage='running' if result['passed'] else 'failed',active=bool(child.active),
            ownership_closed=result['ownership_closed'])
        if mode==STAGES[-1] and result['passed']:
            row.update(stage='closed',complete=True,passed=True,
                decision=decide([p for p in row['scopes'] if p['mode'].isdecimal()]))
        save(row)
    require(result['passed'],result.get('error','validation/closure gate failed'))
    print(mode,'completed normally.',flush=True)


if __name__=='__main__':
    require(sys.executable==PYTHON and len(sys.argv)==2 and sys.argv[1] in ('prepare',*STAGES),
        'pinned Python: prepare|build|smoke|numeric|0|1|2')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:run_stage(sys.argv[1])
