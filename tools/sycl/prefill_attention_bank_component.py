#!/usr/bin/env python3
"""Build and compare the actual attention-bank kernel on the retained runtime."""

import fcntl
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import coherent_model_basis as basis
import measurement_owner as ownership
import prefill_prefix_expert_build as stage
import ur_validation as validation

B, P, RESULT = basis.B, basis.P, basis.RESULT
ROOT = B / 'prefill-attention-bank-component-v1'
KEY = 'prefill_attention_bank_component_v1'
REF = 'prefill_compat_wait_callsite_v1'
ELF = ROOT / 'sycl_prefill_attention_bank'
BOOT = Path('/proc/sys/kernel/random/boot_id')
FROZEN = ('src/prefill/prefill.cpp', 'src/prefill/layer_major.inc',
          'src/prefill/sycl/native16.cpp', 'include/strata/prefill/native16.hpp',
          'tests/sycl/prefill_attention_bank.cpp', 'src/prefill/moe_window.inc')
CONTROLLER = 'tools/sycl/prefill_attention_bank_component.py'
ORDER = ('correctness', 'copy', 'direct', 'direct', 'copy', 'copy', 'direct')
COUNT = 16
BYTES = 8192 * 10240 * 2 * COUNT
FLOOR = 10.
BUDGET = dict(wall=180, cpu=180, rss_cap=6 << 30, text_cap=4 << 20,
              file_cap=512 << 20, total_cap=512 << 20)
require, sha, signature = basis.require, basis.sha, basis.signature


def read(path):
    return json.loads(path.read_bytes())


def compact(row):
    return {k: v for k, v in row.items() if k not in ('sources', 'dependencies', 'environment')}


def save(row):
    stage.write(ROOT / 'record.json', row)
    data = read(RESULT)
    data[KEY] = compact(row)
    stage.write(RESULT, data)


def platform():
    r = read(stage.STAGE_ROOT / 'record.json')
    require(validation.committed(RESULT, (stage.STAGE_BUILD,)) == stage.builder.compact(r)
            and r['stage'] == 'built' and r['passed'] and r['ownership_closed'] and not r['active']
            and r['actual_inputs']['uncovered'] == 0, 'closed stage-all native reference required')
    diagnostic = validation.committed(RESULT, (REF,))
    top = diagnostic['failed_scope_diagnostic']['top_site']
    require(diagnostic['stage'] == 'failed' and diagnostic['original_outcome_retained']
            and diagnostic['physical_owner_reconciliation']['physical_executors_gone']
            and top['count'] == 192 and top['error_literal'] == 'prefill MoE window: attention store/drain'
            and diagnostic['binary'] == r['binaries']['strata'], 'original callsite diagnosis must remain incomplete')
    _,_,speed = basis.basis()
    manifest = speed['runtime_manifest']
    providers = read(Path(manifest['path']))
    require(sha(manifest['path']) == manifest['sha256']
            and all(sha(path) == h for path,h in providers.items()), 'runtime provider inventory changed')
    env = dict(speed['environment_template'])
    old_root = env['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old_root, str(ROOT)) for k,v in env.items()
           if k not in ('LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER', 'STRATA_TRACE')}
    require(env['ZES_ENABLE_SYSMAN'] == '0' and env['EnableDirectSubmission'] == '0'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
            and all(v == '0' for k,v in env.items() if '_PROFILE' in k or '_AUDIT' in k),
            'quiet pre-exec safety values differ')
    env.update(TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache'))
    driver = stage.builder.guards.driver_guard()
    boot = BOOT.read_text().strip()
    archive = stage.STAGE_ROOT / 'build/libstrata_sycl_runtime.a'
    compile_db = stage.STAGE_ROOT / 'build/compile_commands.json'
    engine = r['binaries']['strata']
    require(sha(archive) == r['actual_inputs']['produced'][str(archive)]
            and sha(compile_db) == r['configured_files'][str(compile_db)]
            and sha(engine['path']) == engine['sha256']
            and boot == diagnostic['boot'] and driver == diagnostic['driver'],
            'retained build recipe, runtime archive, or platform changed')
    return r, dict(stage_source=r['source_commit'], stage_binary=engine,
                   archive=dict(path=str(archive), sha256=sha(archive)),
                   compile_db=dict(path=str(compile_db), sha256=sha(compile_db)),
                   diagnostic_sha256=signature(diagnostic), probe=speed['native_validation_probe'],
                   providers_sha256=signature(providers), boot=boot, driver=driver), env


def dependency_paths(text):
    files = set()
    for line in text.replace('\\\n', ' ').splitlines():
        if not line.strip() or line.startswith(('# __CLANG_OFFLOAD_BUNDLE____START__ ',
                                               '# __CLANG_OFFLOAD_BUNDLE____END__ ')):
            continue
        target, sep, values = line.partition(':')
        require(sep and target.endswith('.o') and values.strip(), 'invalid host or device dependency rule')
        paths = [Path(x) for x in shlex.split(values)]
        require(all(p.is_absolute() and p.is_file() for p in paths), 'invalid dependency input')
        files.update(str(p.resolve()) for p in paths)
    require(files, 'empty dependency coverage')
    return files


def flags(entries, source):
    exact, = [x for x in entries if x['file'] == str(stage.STAGE_ROOT / 'source' / source)]
    command = shlex.split(exact['command'])
    cut = command.index('-o')
    require(command[cut+2:] == ['-c', exact['file']] and '-fsycl' in command[:cut],
            'unexpected native compiler recipe')
    return [command[0], '-I'+str(ROOT/'header'), *command[1:cut]]


def pins(revision):
    sources = basis.comparison.build.controls(revision, controller=P/CONTROLLER)
    for name, content in stage.source.blobs(P,revision,FROZEN):
        require((P/name).read_bytes() == content, 'commit exact '+name)
        sources[name] = sha(P/name)
    return sources


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(RESULT), 'fresh component plan required')
    native, admitted, env = platform()
    revision = stage.source.git(P,'rev-parse','HEAD').decode().strip()
    sources = pins(revision)
    ROOT.mkdir(mode=0o700)
    for child in ('tmp','cache','header'):
        (ROOT/child).mkdir(mode=0o700)
    header = ROOT/'header/strata/prefill/native16.hpp'
    header.parent.mkdir(parents=True, mode=0o700)
    header.write_bytes((P/FROZEN[3]).read_bytes())
    entries = read(Path(admitted['compile_db']['path']))
    native_flags = flags(entries,FROZEN[2])
    caller_flags = flags(entries,FROZEN[0])
    compiler = native_flags[0]
    # Preserve all meaningful original compile options and resolve the two
    # offload header streams independently for every new translation unit.
    units = ((FROZEN[0], caller_flags, 'entry.o'), (FROZEN[2], native_flags, 'native.o'),
             (FROZEN[4], native_flags, 'fixture.o'))
    commands, may_include = [], {}
    verified = set()
    for source, prefix, obj in units:
        listing = subprocess.check_output([*prefix, '-M', str(P/source)], env=env,text=True,timeout=55)
        headers = dependency_paths(listing)
        require(str(P/source) in headers and str(header) in headers, 'compiled source/header overlay missing')
        may_include[obj] = sorted(headers)
        verified.update(headers)
        if obj == 'entry.o':
            commands.append([*prefix, '-fsyntax-only', str(P/source)])
        else:
            commands.append([*prefix, '-MD', '-MF', str(ROOT/(obj+'.d')), '-c', str(P/source), '-o', str(ROOT/obj)])
    system_links = {str(Path(k).resolve()):v for k,v in native['actual_inputs']['links'].items()}
    toolchain = {str(Path(compiler).resolve()), str(Path('/usr/bin/ld').resolve()),
                 str(Path(sys.executable).resolve()),str(Path('/usr/bin/as').resolve())}
    for tool in ('clang-offload-wrapper','sycl-post-link','llvm-link','ld','as'):
        found = subprocess.check_output([compiler,'-print-prog-name='+tool],env=env,text=True,timeout=10).strip()
        path = Path(found)
        if not path.is_file(): path = Path(compiler).parent / found
        if path.is_file(): toolchain.add(str(path.resolve()))
    for path in verified:
        item = Path(path)
        snapshot = stage.STAGE_ROOT/'source'
        if item.is_relative_to(snapshot):
            require(native['source_files'][str(item.relative_to(snapshot))] == sha(item),
                    'original runtime/header differs at '+path)
        elif item.is_relative_to(P):
            require(sources.get(str(item.relative_to(P))) == sha(item),
                   'original caller .inc or changed source is missing from the committed pins')
    archive = admitted['archive']['path']
    deps = {p:sha(p) for p in sorted(verified | toolchain | set(system_links) | {archive})}
    require(all(deps[k] == v for k,v in system_links.items()), 'original link input changed')
    link = [compiler,'-O3','-fsycl','-fsycl-id-queries-range=size_t',
            '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13','-Wl,-t',
            str(ROOT/'fixture.o'),str(ROOT/'native.o'),archive,'-lze_loader','-o',str(ELF)]
    save(dict(stage='prepared', source_commit=revision, sources=sources,
              source_sha256=signature(sources), admitted=admitted, environment=env,
              environment_sha256=signature(env), dependencies=deps, dependencies_sha256=signature(deps),
              header_sets=may_include, header_sets_sha256=signature(may_include),
              commands=[*commands,link], budgets=BUDGET, order=ORDER, processes=[],
              plan=dict(unit='Only the GPU attention BF16 store changes. Separate production residual and bank '
                             'owners, same old BF16 residual, projection, inject, stream and checked drain.',
                        correctness='Actual production write/compat D2D pair compared bitwise with the direct '
                                    'operator, including broadcast initialization, 1/63/8191/8192 rows, '
                                    'independent zero-injection arithmetic, guards, alias/extent refusals and retirement.',
                        timing='Same standalone ELF, six independent ABBAAB 8192-row timing processes. '
                               'Five batches of 16, with all resets outside the clock and one checked drain per '
                               'measured store. Charge host span validation and the original copy in its arm.',
                        gate='Normal CPU/GPU exits with no fresh faults. Each batch at least 10 ms and '
                             'clock overhead at most 2%. Advance only if all 3 independent paired '
                             'time medians improve. This checks the complete store pair with drained input, '
                             'so qualifying the actual model route needs separate clean model evidence.',
                        retention='Keep existing 64K stage-all and phase references. This is a compact fixture, '
                                  'not a fourth full engine. Retire on closing its consumer.'),
              complete=False, passed=False, active=False, ownership_closed=True))
    print('Commit the exact component recipe before syntax and device builds.',flush=True)


def current(row):
    require(validation.committed(RESULT,(KEY,))==compact(row), 'commit the exact comparison state')
    _,admitted,env=platform()
    require(admitted==row['admitted'] and env==row['environment'] and signature(env)==row['environment_sha256']
            and pins(row['source_commit'])==row['sources']
            and signature(row['sources'])==row['source_sha256']
            and signature(row['header_sets'])==row['header_sets_sha256']
            and signature(row['dependencies'])==row['dependencies_sha256']
            and all(sha(p)==h for p,h in row['dependencies'].items()),'frozen component input changed')
    if row.get('binary'):
        require(sha(ELF)==row['binary']['sha256'] and all(sha(p)==h for p,h in row['actual_link_inputs'].items()),
                'tested fixture or linked native input changed')


def build(row):
    current(row)
    require(row['stage']=='prepared' and not (ROOT/'compile').exists(), 'fresh component build only')
    owner=ownership.Owner(ROOT/'compile')
    try:
        for i,label in enumerate(('native-entry-syntax','native-store','fixture','link')):
            command,out,err=owner.run(label,row['commands'][i],row['environment'],P,**BUDGET)
            require(ownership.closed(command) and command['exit_code']==0,'failed build stage '+label)
            if i in (1,2):
                obj=('native.o','fixture.o')[i-1]
                actual=dependency_paths((ROOT/(obj+'.d')).read_text())
                expected=set(row['header_sets'][obj])
                must={p for p in expected if p.startswith(str(P)+'/') or p.startswith(str(ROOT/'header')+'/')}
                require(must <= actual <= expected, 'compiler dependency escaped the conservative union')
        files={str(Path(x).resolve()) for x in (out.read_text()+'\n'+err.read_text()).splitlines()
               if x.startswith('/') and Path(x).is_file()}
        require(row['admitted']['archive']['path'] in files and any('libsycl' in p for p in files)
                and files <= set(row['dependencies']) | {str(ROOT/n) for n in ('native.o','fixture.o')},
                'missing or unpinned device fixture link input')
        row.update(stage='built', binary=dict(path=str(ELF),sha256=sha(ELF),bytes=ELF.stat().st_size),
                   actual_link_inputs={p:sha(p) for p in sorted(files)})
    except Exception as error:
        row.update(stage='failed',error=repr(error))
    finally:
        row.update(active=bool(owner.active), ownership_closed=owner.active is None and
                   all(ownership.closed(c) for c in owner.commands), commands_observed=len(owner.commands))
        save(row)
    require(row['stage']=='built' and row['ownership_closed'],row.get('error','component build did not retire'))
    print('Committed product-caller syntax and linked component required before testing.',flush=True)


def parse(text, arm):
    lines=text.splitlines()
    require(lines and lines[-1]=='attention_bank_fixture=PASS','missing native completion')
    if arm=='correctness':
        expected=[f'verified rows={r} layer_zero={z} bits={r*10240} guards=1 cancellation_lease=1'
                  for r,z in ((1,0),(63,1),(8191,0),(8192,1))]
        require(lines[:-1]==expected,'missing exact BF16/tail/guard cases')
        return dict(cases=len(expected),compared_BF16=sum(r*10240 for r in (1,63,8191,8192)),
                    checked_rejection_types=['overlap','zero extent','outside registered range'])
    require(len(lines)==7 and lines[0].startswith('timer_pair_ms='),'wrong timing samples')
    overhead=float(lines[0].split('=',1)[1])
    require(math.isfinite(overhead) and overhead > 0,'invalid timer overhead')
    samples=[]
    for i,line in enumerate(lines[1:-1]):
        expected=f'batch={i} rows=8192 repetitions={COUNT} d2d_bytes={BYTES if arm=="copy" else 0} wall_ms='
        require(line.startswith(expected),'unexpected shape, copy or batch')
        ms=float(line.removeprefix(expected))
        require(math.isfinite(ms) and ms>=FLOOR and COUNT*overhead<=.02*ms,'batch violates time or clock gate')
        samples.append(ms)
    return dict(timer_pair_ms=overhead, batches_ms=samples, median_ms=statistics.median(samples),
                requested_D2D_bytes_per_batch=BYTES if arm=='copy' else 0)


def run(row,index):
    current(row)
    require(row['ownership_closed'] and not row['active'] and row['stage'] in ('built','running')
            and tuple(row['order'])==ORDER and index==len(row['processes']), 'fresh owned arm required')
    arm=ORDER[index]
    root=ROOT/str(index)
    root.mkdir(mode=0o700)
    probe=row['admitted']['probe']
    require(all(sha(v['path'])==v['sha256'] for v in probe.values()),'native validation preload changed')
    env=dict(row['environment'],LD_PRELOAD=probe['library']['path'],
             STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'])
    cursor=next(line.split(': ',1)[1] for line in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=10)
        .splitlines() if line.startswith('-- cursor: '))
    owner=ownership.Owner(root/'owned')
    result=dict(index=index,arm=arm,host_probe_accepted=False)
    try:
        process,out,err=owner.run('fixture',[str(ELF),arm],env,P,**BUDGET)
        require(ownership.closed(process) and process['exit_code']==0,'native attention bank fixture failed')
        log=err.read_text()
        proof=validation.preload_evidence(log,process['pid'])
        require(proof['loader']==probe['loader']['path'] and not any(s in log for s in
                ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'retirement failed')), 'unexpected runtime validation')
        result.update(parse(out.read_text(),arm),native_exit_code=0,validation=proof,host_probe_accepted=True)
    except Exception as error:
        result['error']=repr(error)
    finally:
        validation.finish(result,owner,cursor,row['admitted']['boot'])
        if result.get('driver_after')!=row['admitted']['driver']:
            result['passed']=False
        row['processes'].append(result)
        row.update(stage='running' if result['passed'] else 'failed',
                   active=bool(owner.active),ownership_closed=result['ownership_closed'])
        if result['passed'] and len(row['processes'])==len(ORDER):
            timings=row['processes'][1:]
            pairs=[]
            for a,b in zip(timings[::2],timings[1::2],strict=True):
                copy,direct=sorted((a,b),key=lambda p:p['arm']=='direct')
                pairs.append(dict(copy_ms=copy['median_ms'],direct_ms=direct['median_ms'],
                                  ratio=direct['median_ms']/copy['median_ms']))
            row.update(stage='closed',complete=True,passed=True,decision=dict(
                pairs=pairs,geomean_ratio=math.prod(p['ratio'] for p in pairs)**(1/len(pairs)),
                model_eligible=all(p['ratio']<1 for p in pairs),model_tested=False))
        save(row)
    print(json.dumps({k:result.get(k) for k in ('arm','passed','fresh_faults','error')}),flush=True)
    require(result['passed'],result.get('error','device comparison did not qualify'))


if __name__=='__main__':
    require(sys.executable==basis.reference.model.PYTHON,'use pinned Python')
    action,=sys.argv[1:]
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stage.builder.guards.no_model_owners([])
        if action=='prepare':
            prepare()
        elif action=='build':
            build(read(ROOT/'record.json'))
        else:
            run(read(ROOT/'record.json'),int(action))
