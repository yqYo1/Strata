#!/usr/bin/env python3
"""Refresh production-compatible copy service under the qualified SYCL runtime.

Repository LICENSE. One tiny standalone tool, no new engine generation. Own all
builds and copies; prepare -> commit -> build -> commit -> run. No automatic retry.
"""
import ast
import copy
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
import build_fused_profile as closure
import measurement_owner as supervision
import measurement_guards as guards

r, P, B = closure.r, closure.P, closure.B
ROOT = B / 'transfer-refresh-39758-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'transfer_refresh'
BATCH = False
BINARY = 'host_transfer_probe'
NATIVE = ['src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
          'tools/sycl/host_transfer_probe.cpp']
FLAGS = ['-O3', '-DNDEBUG', '-std=c++17', '-fsycl', '-fsycl-id-queries-range=size_t',
         '-DSTRATA_ENABLE_SYCL=1', '-pthread', '-I', str(P / 'include'),
         '-I', str(P / 'include/strata/sycl/compat')]
BUILD_LIMITS = dict(wall=240, cpu=600, rss_cap=4 << 30, text_cap=8 << 20,
                    file_cap=128 << 20, total_cap=256 << 20)
RUN_LIMITS = dict(wall=90, cpu=90, rss_cap=3 << 30, text_cap=8 << 20,
                  file_cap=16 << 20, total_cap=64 << 20)
OFFICIAL = Path('/opt/intel/oneapi/compiler/2026.2')
UMF = Path('/opt/intel/oneapi/umf/1.2')
require, digest, write = guards.require, supervision.sha, closure.write


def git(*args):
    return r.build.git(P, *args)


def pins(revision):
    names = set(NATIVE) | set(git('ls-files', 'include').decode().splitlines())
    pending, seen = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path); names.add(str(path.relative_to(P)))
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [n.name for n in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                candidate = path.parent / ((module or '').split('.')[0] + '.py')
                if candidate.is_file():
                    pending.append(candidate)
    names.add('tools/sycl/measurement_environment.json')
    result = {}
    for name, raw in r.build.blobs(P, revision, sorted(names)):
        require(raw == (P / name).read_bytes(), 'commit transfer build input: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def public():
    data = json.loads(RESULT.read_text())
    committed = json.loads(git('show', 'HEAD:' + str(RESULT.relative_to(P))))
    require(data.get(KEY) == committed.get(KEY), 'commit exact transfer plan/result')
    return data


def commands(compiler):
    root = ROOT
    rows = [[compiler, *FLAGS, '-MD', '-MF', str(root / (str(i) + '.d')),
             '-c', str(P / source), '-o', str(root / (str(i) + '.o'))] for i, source in enumerate(NATIVE)]
    rows.append([compiler, '-fsycl', '-fsycl-id-queries-range=size_t', '-pthread',
                 *[str(root / (str(i) + '.o')) for i in range(len(NATIVE))], '-o', str(root / BINARY)])
    return rows


def runtime():
    rt = r.validation.committed(r.RESULT)
    require(rt['passed'] and rt['ownership_closed'] and not rt['active'] and not rt['fresh_faults'], 'qualified runtime required')
    full = r.current(rt)
    return rt, full


def official_packages():
    return subprocess.check_output(['dpkg-query','-W','-f=${Package} ${Version}\n',
        'intel-oneapi-compiler-dpcpp-cpp-runtime-2026.2','intel-oneapi-dpcpp-cpp-2026.2',
        'intel-oneapi-umf-1.2'], text=True,timeout=15).splitlines()


def official_snapshot():
    # Independent Main SDK recipe recovered from617e1845a; no legacy SDK
    # header/runtime/compiler reuse. Freeze this NEW scope before compiling.
    packages = official_packages()
    require(packages and all(len(p.split()) == 2 for p in packages), 'official package identities missing')
    original_path = B / 'prefill-expert-staging-v3/sdk-inputs.json'
    original_result = json.loads(git('show','e64229a83:bench/results/2026-10-15-pr6-recovery-audit/prefill-resources-current.json'))
    original = original_result['event_reconciliation']['fused_command_profile']['prefill_upload_staging_v3']['sdk']
    require(str(original_path) == original['path'] and digest(original_path) == original['sha256'] and
            packages == original['packages'], 'original independent official SDK receipt differs')
    prior = json.loads(original_path.read_text())
    roots = [OFFICIAL,UMF,Path('/usr/include')]
    roots += [p for p in Path('/usr/lib/gcc').glob('*/*/include*') if p.is_dir()]
    paths = {p.resolve() for root in roots for p in root.rglob('*') if p.is_file()}
    files = {str(p):digest(p) for p in sorted(paths)}
    vendor = lambda rows: {p:h for p,h in rows.items() if
        Path(p).is_relative_to(OFFICIAL) or Path(p).is_relative_to(UMF)}
    require(vendor(files) == vendor(prior), 'official vendor files differ from independently qualified bytes')
    write(ROOT / 'sdk-inputs.json',files)
    return dict(path=str(ROOT / 'sdk-inputs.json'),sha256=digest(ROOT / 'sdk-inputs.json'),
                files=len(files),packages=packages,origin='Official Intel apt packages;all3035vendor files exactly match original independently qualified snapshot.',
                original_manifest=dict(commit='e64229a83',path=str(original_path),sha256=original['sha256'],
                    vendor_files=len(vendor(files))),
                verification_limit='Original broad44-package preparation failed on pkgconfig metadata,including unrelated MKL/IPP;that failure remains. This new scope checks exact compiler/UMF vendor bytes against original Git-bound proof and separately freezes current host headers. No legacy SDK reuse or hash refresh.')


def official_admit(meta):
    require(digest(meta['path']) == meta['sha256'] and official_packages() == meta['packages'],
            'official SDK manifest/packages differ')
    files = json.loads(Path(meta['path']).read_text())
    require(len(files) == meta['files'] and all(digest(p) == h for p,h in files.items()),
            'official SDK/host input changed')
    return files


def verify(row):
    rt, full = runtime()
    require(digest(r.RESULT) == row['runtime_sha256'] and full['binary'] == row['selected_binary'], 'runtime/selected engine changed')
    require(guards.driver_guard() == row['driver'] and rt['boot'] == row['boot'], 'transfer boot/driver')
    if BATCH:
        official_admit(row['sdk'])
        require(all(digest(p) == h for p,h in row['runtime_files'].items()), 'proposed official runtime changed')
    else:
        require(closure.sdk_inputs() == row['sdk'], 'transfer compiler SDK changed')
    require(digest(row['inputs']['path']) == row['inputs']['sha256'], 'transfer input manifest changed')
    inputs = json.loads(Path(row['inputs']['path']).read_text())
    require(inputs['project'] == pins(row['source_commit']), 'transfer source closure changed')
    for name, sha in inputs['external'].items():
        require(digest(name) == sha, 'transfer external input changed: ' + name)
    compiler = str(OFFICIAL / 'bin/icpx') if BATCH else full['compiler']['path']
    require(commands(compiler) == row['commands'], 'transfer build command changed')
    return inputs, rt


def prepare(data):
    require(KEY not in data and not ROOT.exists(), 'fresh transfer plan only')
    rt, full = runtime()
    revision = git('rev-parse', 'HEAD').decode().strip(); project = pins(revision)
    sdk = None if BATCH else closure.sdk_inputs()
    external = closure.external_inputs(rt)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    if BATCH:
        sdk = official_snapshot()
        old_lib = Path(next(p for p in rt['runtime_files'] if '/libsycl.so.' in p)).parent
        external = {p:h for p,h in external.items() if not Path(p).is_relative_to(old_lib)}
    write(ROOT / 'inputs.json', dict(project=project, external=external))
    build_env = dict(full['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    native_env = dict(rt['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
                      STRATA_TRANSFER_PROBE_GATES='1')
    native_env.pop('UR_LOG_TRACING', None)
    native_env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
    cases = [dict(host=host, size=2 << 20, repeats=1, diagnostic=True) for host in ('ordinary', 'usm')]
    cases += [dict(host=host, size=size, repeats=3, diagnostic=False) for size in (2 << 20, 320 << 20)
              for host in ('ordinary', 'usm', 'usm', 'ordinary', 'ordinary', 'usm')]
    if BATCH:
        cases = [dict(diagnostic=False,host_canary=True,argv=[]),dict(diagnostic=True, argv=['--smoke'])]
        cases += [dict(diagnostic=False, argv=['--compare', str(i)]) for i in range(3)]
    row = dict(stage='prepared', source_commit=revision, inputs=dict(path=str(ROOT / 'inputs.json'),
               sha256=digest(ROOT / 'inputs.json'), project_files=len(project), external_files=len(external)),
               sdk=sdk, runtime_sha256=digest(r.RESULT), selected_binary=full['binary'],
               commands=commands(full['compiler']['path']), build_environment=build_env,
               native_environment=native_env, runtime_files=rt['runtime_files'],
               driver=guards.driver_guard(), boot=rt['boot'], cases=cases,
               build_limits=BUILD_LIMITS, run_limits=RUN_LIMITS,
               criteria='Two2MiB traced/validated cases before12 clean cases;each case fresh owned process. Exact B570,preinit/provider/executable/env gates,full payload/source/guards all repetitions,both directions,normal allocation retirement/exit,no fresh fault. Separate first copy and3 measured copies. Stop at first failed case.',
               scope='Current-runtime production Runtime/compat copy service and caller CPU/wall split. TwoMiB expert-sized and320MiB bulk transfers;3processes per allocation/size. No model speed,physicalPCIe byte counter,global hardware saturation,overlap or arithmetic ceiling claim.',
               provenance='Existing independent Main transfer tool with owned gates;unchanged production runtime/compat/profile implementation. SDK/compiler only,not XeStrata source.',
               retention='One tiny helper for imminent resource budgeting. Retire objects/normal logs after source recovery and compact individual samples;engine fallback/candidate untouched.',
               active=False, ownership_closed=True, passed=False)
    if BATCH:
        sdk_files = json.loads(Path(sdk['path']).read_text())
        runtime_files = {p:h for p,h in rt['runtime_files'].items() if not Path(p).is_relative_to(old_lib)}
        for p,h in sdk_files.items():
            path = Path(p)
            if (path.is_relative_to(OFFICIAL) or path.is_relative_to(UMF)) and '.so' in path.name:
                with path.open('rb') as stream: elf = stream.read(4) == b'\x7fELF'
                if elf: runtime_files[p] = h
        preload = copy.deepcopy(rt['preload'])
        preload['loader'] = str((OFFICIAL / 'lib/libur_loader.so.0').resolve())
        for env in (build_env,native_env):
            env.update(PATH=str(OFFICIAL / 'bin')+':/usr/bin:/bin',
                CPLUS_INCLUDE_PATH=str(OFFICIAL / 'include'),
                LD_LIBRARY_PATH=':'.join([str(OFFICIAL / 'lib'),str(UMF / 'lib'),
                    *[p for p in env['LD_LIBRARY_PATH'].split(':') if p != str(old_lib)]]),
                UR_ADAPTERS_FORCE_LOAD=str(OFFICIAL / 'lib/libur_adapter_level_zero_v2.so.0.12.0'))
        native_env['STRATA_UR_PROBE_EXPECTED_LOADER'] = preload['loader']
        # Resolve the actual official core libraries' DT_NEEDED closure before
        # launch (including host libraries absent from the legacy runtime).
        needed = [OFFICIAL / 'lib' / n for n in ('libsycl.so.9.0.0','libur_loader.so.0.12.0',
                                                'libur_adapter_level_zero_v2.so.0.12.0')]
        needed += [UMF / 'lib/libumf.so']
        for library in needed:
            raw = subprocess.check_output(['/usr/bin/ldd',str(library)],text=True,timeout=15,
                env=dict(PATH='/usr/bin:/bin',LD_LIBRARY_PATH=native_env['LD_LIBRARY_PATH']))
            require('not found' not in raw, 'official runtime dependency unresolved')
            for name in re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(',raw,re.M):
                path = str(Path(name).resolve()); runtime_files[path] = digest(path)
        require(b'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD' in Path(native_env['UR_ADAPTERS_FORCE_LOAD']).read_bytes(),
                'official adapter lacks mandatory safety key')
        row.update(commands=commands(str(OFFICIAL / 'bin/icpx')),runtime_files=runtime_files,preload=preload,
            runtime_scope='NEW official Intel2026.2/UMF1.2 component proposal;host loader canary and traced smoke required. No prior native/model/runtime qualification inherited;selected original engines unchanged.')
        row.update(
            scope='Bounded1/2/4/8-expert groups in one persistent in-order queue and128-slot private bank. Same destinations,packed ordinary CPU source and160active experts for every arm. Persistent workspace per payload/pattern;no host packing,overfetch,extra model VRAM or decode change. GPU full output/bank/guard affine oracle plus two timed input readers;small CPU cross-checks and corruption controls,final full CPU source check. Synthetic1/64-loop integer readers,not XMX or LLM throughput;event-recording queue.memcpy callbacks isolate scheduling,default compat path separately checked.',
            criteria='Official loader canary first,then4KiB traced10-case smoke with22admission/order refusals,3verifier corruption controls,CPU cross-checks at1/64loops,all1/2/4/8groups,bank/group tails,expert511,cancellation and default compat copy. Then3fresh processes,each288batches:2payloads(1971200/2662400),3patterns(contiguous/runs4/gapped),2integer workloads(1/64),one first rotating quartet plus5subsequent rotating quartets. Four arms group1/2/4/8;all160experts/320readers,128slots,identical slot=i%128,exact active bytes. Copy calls contiguous160/group,runs4160/min(group,4),gapped160. GPU output/bank/guards pass every batch;full immutable CPU source closure6cohorts/process. All normal provider/preinit/env/owner/exit/fault checks. Subsequent wall/device spans>=10ms,timerp99<=10us,median empty<=5percent smallest batch. A group qualifies only if every one of its36process/shape medians has candidate/group1 ratio<=0.98 for contiguous/runs4 and<=1.03 for gapped. Among fully qualifying groups choose lowest geometric mean of36medians,tie smaller group. No qualifying group means no model run. Passing permits only a new production-consumer plan;not inference adoption.',
            reference='Prior8db22cee7 V1 failed its original performance gate with432valid batches;that result stays failed. New source replaces128-expert bursts with bounded groups and prospectively changes both arms to persistent queue/workspace plus GPU verification. No physical bus,cache,staging-threshold or whole-model claim. Existing indexed reports do not rank2/4/8;native Luna review only informed this bounded comparison.',
            retention='One mechanism fixture for imminent prefill-only scheduling decision;no third engine generation. Keep original6e40 and9a79 engines/required fixtures. Retire normal output and this component build at decision closure after exact source/recipe recovery;retain compact paired samples and ownership/fault outcome.')
    data[KEY] = row; write(RESULT, data)
    print('Transfer source/recipe prepared;commit before build.', flush=True)


def build(data):
    row = data[KEY]; require(row['stage'] == 'prepared' and not (ROOT / 'build').exists(), 'one committed transfer build')
    inputs, _ = verify(row)
    supervisor = None
    supervisor = supervision.Owner(ROOT / 'build', lambda: write(ROOT / 'build-owner.json', supervisor.commands))
    try:
        row.update(stage='building',active=True,ownership_closed=False);write(RESULT,data)
        for i, command in enumerate(row['commands']):
            entry, out, err = supervisor.run('compile-' + str(i), command, row['build_environment'], P, **BUILD_LIMITS)
            require(supervision.closed(entry) and entry['exit_code'] == 0, 'transfer compiler/link failed')
        allowed = {str((P / name).resolve()): sha for name, sha in inputs['project'].items()}
        allowed.update(inputs['external'])
        allowed.update(official_admit(row['sdk']) if BATCH else
            {str(Path(x['path']).resolve()): closure.SYSTEM_HEADER_UPDATES.get(x['path'], (None, x['sha256']))[1]
             for x in json.loads(closure.lineage.SDK.read_text())})
        observed = {}
        for i in range(len(NATIVE)):
            body = (ROOT / (str(i) + '.d')).read_text().replace('\\\n', ' ').split(':', 1)[1]
            require('\\' not in body, 'unexpected transfer dependency escaping')
            for path in body.split():
                path = str(Path(path).resolve()); sha = digest(path)
                require(path in allowed and sha == allowed[path], 'unfrozen transfer compiler input: ' + path)
                observed[path] = sha
        require(all(str(P / name) in observed for name in NATIVE), 'incomplete transfer compiled source coverage')
        write(ROOT / 'compiled-inputs.json', observed)
        verify(row)
        binary = ROOT / BINARY
        row.update(stage='built', build_passed=True, binary=dict(path=str(binary), sha256=digest(binary), bytes=binary.stat().st_size),
                   compiled_inputs=dict(path=str(ROOT / 'compiled-inputs.json'), sha256=digest(ROOT / 'compiled-inputs.json'),
                                        files=len(observed), uncovered=0))
    except BaseException as error:
        row.update(stage='build-failed', build_passed=False, error=repr(error))
    finally:
        row.update(active=supervisor.active is not None,
                   ownership_closed=supervisor.active is None and all(supervision.closed(c) for c in supervisor.commands))
        write(RESULT, data)
    print(json.dumps({k: row.get(k) for k in ('stage', 'build_passed', 'ownership_closed', 'error')}), flush=True)


def run(data):
    row = data[KEY]
    require(row['stage'] == 'built' and row['build_passed'] and row['ownership_closed'], 'commit successful transfer build')
    _, rt = verify(row)
    require(digest(row['binary']['path']) == row['binary']['sha256'] and not (ROOT / 'native').exists(), 'fresh verified transfer run')
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(line.split(': ', 1)[1] for line in cursor_text.splitlines() if line.startswith('-- cursor: '))
    supervisor = None
    supervisor = supervision.Owner(ROOT / 'native', lambda: write(ROOT / 'native-owner.json', supervisor.commands))
    cases = row['measurements'] = []
    try:
        row.update(stage='running',active=True,ownership_closed=False,complete=False);write(RESULT,data)
        for i, case in enumerate(row['cases']):
            env = dict(row['native_environment'])
            warnings = ROOT / 'native' / ('case-' + str(i) + '-validation.log')
            trace = ROOT / 'native' / ('case-' + str(i) + '-trace.log')
            env['UR_LOG_VALIDATION'] = 'level:warning;flush:warning;output:file,' + str(warnings)
            if case['diagnostic']:
                env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                           UR_LOG_TRACING='level:info;flush:info;output:file,' + str(trace))
            if BATCH and case.get('host_canary'):
                asset = row['preload']['canary']
                require(digest(asset['path']) == asset['sha256'], 'host canary changed')
                entry,out,err = supervisor.run('case-'+str(i),[asset['path']],env,P,**RUN_LIMITS)
                require(supervision.closed(entry) and entry['exit_code'] == 0 and
                    json.loads(out.read_text()) == dict(pid=entry['pid'],initialized=0,checked=46,expected=46,
                        version=(1 << 32)-1,closed=0), 'official host canary failed')
                require(r.validation.preload_evidence(err.read_text(),entry['pid'])['loader'] == row['preload']['loader'],
                        'official host loader differs')
                require(not warnings.exists() or not warnings.stat().st_size, 'official host validation warning')
                cases.append(dict(case=case,passed=True,host_only=True,ownership_closed=True))
                write(RESULT,data)
                continue
            argv = [row['binary']['path'], *case['argv']] if BATCH else \
                [row['binary']['path'], '--host', case['host'], '--bytes', str(case['size']), '--repeats', str(case['repeats'])]
            gates, lines, code_maps = [], [], []

            def observe(io, entry):
                while True:
                    line = io.line(seconds=25); lines.append(line)
                    if not line.startswith('GATE stage='):
                        continue
                    stage = line.split('=', 1)[1]
                    require(len(gates) < 3 and stage == ['before-runtime', 'runtime-ready', 'before-exit'][len(gates)], 'transfer gate order')
                    proc = Path('/proc', str(entry['pid']))
                    require(str((proc / 'exe').resolve()) == row['binary']['path'] and digest(proc / 'exe') == row['binary']['sha256'], 'transfer actual executable')
                    actual_env = dict(v.split(b'=', 1) for v in (proc / 'environ').read_bytes().split(b'\0') if v)
                    require(actual_env == {k.encode(): v.encode() for k, v in env.items()}, 'transfer immutable child env')
                    mapped = r.executable_mappings((proc / 'maps').read_text())
                    providers = row['runtime_files'] if BATCH else rt['runtime_files']
                    allowed = set(providers) | {row['binary']['path']}
                    require(mapped <= allowed, 'unfrozen transfer provider: ' + repr(mapped - allowed))
                    for name in mapped:
                        require(digest(name) == (row['binary']['sha256'] if name == row['binary']['path'] else providers[name]),
                                'transfer provider changed: ' + name)
                    if stage != 'before-runtime':
                        require(any(line.startswith('DEVICE name=') and 'B570' in line for line in lines),
                                'actual transfer GPU differs before work')
                        require(any('libze_intel_gpu.so.1.17.39758' in name for name in mapped) and
                                any('libur_adapter_level_zero_v2.so' in name for name in mapped), 'expected transfer providers absent')
                    gates.append(stage); code_maps.append(dict(stage=stage, maps=len(mapped)))
                    io.send(b'continue\n')
                    if stage == 'before-exit':
                        io.exit(10); break

            entry, out, err = supervisor.run('case-' + str(i), argv, env, P, interaction=observe, **RUN_LIMITS)
            prefix = 'BATCH ' if BATCH else 'COPY '
            rows = [{part.split('=', 1)[0]: part.split('=', 1)[1] for part in line.split()[1:]} for line in lines if line.startswith(prefix)]
            checks = [line for line in lines if line.startswith('CHECK ')]
            source_checks = [{part.split('=',1)[0]:part.split('=',1)[1] for part in line.split()[1:]}
                             for line in lines if line.startswith('SOURCE ')]
            passed = supervision.closed(entry) and entry['exit_code'] == 0 and gates == ['before-runtime', 'runtime-ready', 'before-exit']
            if BATCH:
                passed = passed and len(rows) == (10 if case['diagnostic'] else 288) and all(
                    all(sample.get(k) == 'PASS' for k in ('source','guards','output','drain','gpu_verify')) for sample in rows)
                if case['diagnostic']:
                    passed = passed and sum(int(s['admission_rejections']) for s in rows) == 22 and \
                        sum(int(s['cancelled']) for s in rows) == 1 and \
                        sum(int(s['compat']) for s in rows) == 1 and \
                        sum(int(s['verifier_negative_controls']) for s in rows) == 3 and \
                        sum(int(s['cpu_reference']) for s in rows) == 10 and \
                        {s['loops'] for s in rows} == {'1','64'} and \
                        {s['pattern'] for s in rows} == {'contiguous','runs4','gapped','edge'} and \
                        any(s['capacity'] == '1' and s['copy_calls'] == '8' for s in rows) and \
                        any(s['cancelled'] == '1' and s['completed'] == '5' and
                            s['uploaded_experts'] == '6' and s['copy_calls'] == '2' for s in rows) and \
                        any(s['compat'] == '1' and s['copy_calls'] == '3' and s['copy_events'] == '0' for s in rows)
            else:
                passed = passed and len(rows) == len(checks) == 2 * (case['repeats'] + 1) and all(
                    'full=PASS guards=PASS source=PASS' in line for line in checks)
            require(any(line.startswith('DEVICE name=') and 'B570' in line for line in lines), 'actual transfer GPU differs')
            require(any(line.startswith('DONE ') for line in lines) and not any('FAIL' in line for line in lines), 'transfer completion failed')
            preload = r.validation.preload_evidence(err.read_text(), entry['pid'])
            require(preload['loader'] == (row['preload']['loader'] if BATCH else rt['preload']['loader']), 'transfer actual UR loader differs')
            require(not warnings.exists() or not warnings.stat().st_size, 'transfer parameter validation warning')
            if case['diagnostic']:
                require(trace.exists() and 'urEnqueue' in trace.read_text(), 'transfer enqueue trace absent')
            if BATCH:
                require(len(source_checks) == (10 if case['diagnostic'] else 6) and
                        all(c.get('full') == 'PASS' for c in source_checks), 'full immutable source closure missing')
                for sample in rows:
                    require(all(math.isfinite(float(sample[k])) and float(sample[k]) > 0
                                for k in ('wall_s','reader_s','device_span_s')) and
                            math.isfinite(float(sample['copy_s'])) and float(sample['copy_s']) >= 0 and
                            math.isfinite(float(sample['submission_s'])) and 0 <= float(sample['submission_s']) <= float(sample['wall_s']) and
                            0 <= float(sample['overlap_s']) <= min(float(sample['copy_s']), float(sample['reader_s'])),
                            'batch clock/overlap mismatch')
                controls = [{part.split('=',1)[0]:part.split('=',1)[1] for part in line.split()[1:]}
                            for line in lines if line.startswith(('CONTROL ', 'TIMER '))]
            else:
                for sample in rows:
                    require(sample['host'] == case['host'] and int(sample['bytes']) == case['size'] and
                            all(math.isfinite(float(sample[k])) and float(sample[k]) > 0 for k in ('total_GBps', 'total_wall_s')),
                            'transfer sample configuration/clock mismatch')
            cases.append(dict(case=case, passed=bool(passed), samples=rows, gates=code_maps,
                              process={k: entry.get(k) for k in ('pid', 'exit_code', 'elapsed_seconds', 'peak_session_rss_bytes',
                              'observation_complete', 'direct_child_reaped', 'session_empty', 'error')}))
            if BATCH: cases[-1].update(controls=controls,source_checks=source_checks)
            write(RESULT, data)
            require(passed, 'transfer case failed')
            guards.driver_guard()
        verify(row)
        require(digest(row['binary']['path']) == row['binary']['sha256'], 'transfer binary changed')
        if BATCH:
            row['decision'] = batch_decision(cases[1:])
        row.update(stage='closed', passed=True, complete=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        try:
            raw = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'], timeout=15)
            require(len(raw) < 2 << 20, 'transfer fault evidence budget')
            row['fresh_faults'] = [s for s in [json.loads(line).get('MESSAGE', '') for line in raw.decode().splitlines() if line]
                                   if re.search(r'(?:\bxe\b|drm|GPU).*(?:fault|hang|wedg|reset|timed out)', s, re.I)]
            row['driver_after'] = guards.driver_guard()
        except BaseException as error:
            row.update(passed=False, fault_evidence_error=repr(error))
        row.update(active=supervisor.active is not None,
                   ownership_closed=supervisor.active is None and all(supervision.closed(c) for c in supervisor.commands))
        row['passed'] = bool(row.get('passed') and row['ownership_closed'] and not row.get('fresh_faults'))
        write(RESULT, data)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'error')}), flush=True)


def batch_decision(cases):
    comparisons = [case for case in cases if case.get('case',{}).get('argv',[None])[0] == '--compare']
    require(len(comparisons) == 3 and {case['case']['argv'][-1] for case in comparisons} == {'0','1','2'},
            'three distinct batch comparison processes required')
    summaries, eligibility = [], []
    for process in comparisons:
        samples = process['samples']; require(process['passed'] and len(samples) == 288, 'batch sample coverage')
        keys = {(s['group'],s['bytes'],s['pattern'],s['loops'],s['repetition']) for s in samples}
        expected = {(str(group),str(size),pattern,str(loops),str(repetition)) for group in (1,2,4,8)
                    for size in (1971200,2662400) for pattern in ('contiguous','runs4','gapped')
                    for loops in (1,64) for repetition in range(-1,5)}
        require(keys == expected, 'batch configuration/repetition coverage differs')
        sources = process['source_checks']
        require(len(sources) == 6 and {(int(s['bytes']),s['pattern']) for s in sources} ==
                {(size,pattern) for size in (1971200,2662400) for pattern in ('contiguous','runs4','gapped')} and
                all(s['full'] == 'PASS' for s in sources), 'immutable source closure differs')
        for s in samples:
            group = int(s['group'])
            expected_calls = {'contiguous':160//group,'runs4':160//min(group,4),'gapped':160}[s['pattern']]
            require(all(int(s[k]) == 160 for k in ('experts','completed','uploaded_experts')) and
                    int(s['readers']) == 320 and int(s['capacity']) == 128 and
                    int(s['copy_calls']) == int(s['copy_events']) == expected_calls and
                    int(s['requested_upload_bytes']) == 160 * int(s['bytes']) and
                    all(int(s[k]) == 0 for k in ('cancelled','compat','admission_rejections','cpu_reference','verifier_negative_controls')) and
                    s['source_usm'] == 'unknown' and s['gpu_verify'] == 'PASS', 'batch payload/call/consumer conservation differs')
        timer = [c for c in process['controls'] if 'p99_ns' in c]
        empty = [float(c['wall_s']) for c in process['controls'] if int(c.get('repetition',-1)) >= 0]
        subsequent = [s for s in samples if int(s['repetition']) >= 0]
        valid = len(timer) == 1 and int(timer[0]['samples']) == 1000 and int(timer[0]['p99_ns']) <= 10000 and \
            len(empty) == 5 and statistics.median(empty) <= .05 * min(float(s['wall_s']) for s in subsequent) and \
            all(float(s['wall_s']) >= .01 and float(s['device_span_s']) >= .01 for s in subsequent)
        eligibility.append(valid)
        for size in (1971200,2662400):
            for pattern in ('contiguous','runs4','gapped'):
                for loops in (1,64):
                    selected = [s for s in subsequent if int(s['bytes']) == size and s['pattern'] == pattern and int(s['loops']) == loops]
                    for group in (2,4,8):
                        ratios = [float(next(s for s in selected if s['group'] == str(group) and int(s['repetition']) == i)['wall_s']) /
                                  float(next(s for s in selected if s['group'] == '1' and int(s['repetition']) == i)['wall_s']) for i in range(5)]
                        ratio = statistics.median(ratios)
                        summaries.append(dict(process=process['case']['argv'][-1],group=group,bytes=size,pattern=pattern,loops=loops,
                                              paired_wall_ratios=ratios,median_ratio=ratio,
                                              performance_passed=valid and ratio <= (1.03 if pattern == 'gapped' else .98)))
    groups = []
    for group in (2,4,8):
        rows = [s for s in summaries if s['group'] == group]
        require(len(rows) == 36, 'bounded-group comparison coverage')
        groups.append(dict(group=group,performance_passed=all(s['performance_passed'] for s in rows),
                           geometric_mean_ratio=math.exp(statistics.mean(math.log(s['median_ratio']) for s in rows))))
    eligible = [g for g in groups if g['performance_passed']]
    selected = min(eligible,key=lambda g:(g['geometric_mean_ratio'],g['group']))['group'] if eligible else None
    return dict(timing_eligible=all(eligibility),shapes=summaries,groups=groups,selected_group=selected,
                mechanism_screen_passed=selected is not None,inference_adopted=False,physical_saturation_qualified=False,
                next_action='Only a group passing all36matched process/shape gates permits a new production-consumer plan;otherwise reject without model run.')


def main():
    global KEY, ROOT, NATIVE, BATCH, BINARY, RUN_LIMITS
    require(len(sys.argv) == 2, 'choose prepare,build,run or batch-prepare,batch-build,batch-run')
    action = sys.argv[1]
    if action.startswith('batch-'):
        action = action.removeprefix('batch-')
        BATCH = True; KEY = 'prefill_copy_batch_v2'; ROOT = B / 'prefill-copy-batch-v2'
        BINARY = 'expert_batch_probe'
        NATIVE = [*NATIVE[:-1], 'tools/sycl/expert_batch_probe.cpp']
        RUN_LIMITS = dict(RUN_LIMITS,wall=240,cpu=360,rss_cap=5 << 30)
    require(action in ('prepare','build','run'), 'choose prepare,build,run')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        data = public(); {'prepare': prepare, 'build': build, 'run': run}[action](data)


if __name__ == '__main__':
    main()
