#!/usr/bin/env python3
"""Main-owned, standalone copy-engine probe; repository LICENSE.

Commit source -> prepare -> commit plan -> build -> commit ELF -> next (one
process per invocation). No inference build or automatic retry/integration.
"""
import fcntl
import json
from pathlib import Path
import re
import shlex
import sys
sys.dont_write_bytecode = True
import coherent_original_reuse as original
import engine_once
import measurement_guards as guards
import measurement_owner as owned
import ur_validation

P, B = original.P, original.B
ROOT = B/'prefill-copy-engine-v3'
RESULT = P/'bench/results/2026-10-15-pr6-recovery-audit/cpu-profiler-current.json'
KEY = 'prefill_copy_engine_v3'
SOURCE = 'tests/sycl/prefill_copy_engine.cpp'
PREFIXES = ['tools/sycl', SOURCE, 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md']
LOADER = '/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0'
VALIDATION = '/usr/lib/x86_64-linux-gnu/libze_validation_layer.so.1.32.0'
BUILD = dict(wall=180, cpu=240, rss_cap=4<<30, text_cap=8<<20, file_cap=64<<20, total_cap=128<<20)
RUN = dict(wall=90, cpu=180, rss_cap=4<<30, text_cap=8<<20, file_cap=8<<20, total_cap=128<<20)
require, sha = guards.require, owned.sha
write = original.coherent.build.common.write
git = original.coherent.build.common.source.git


def compact(row):
    value = {k:v for k,v in row.items() if k not in ('project', 'foundation', 'actual_inputs', 'commands')}
    value['project'] = {k:v for k,v in row['project'].items() if k != 'source_sha256'}
    value['foundation_sha256'] = original.fingerprint(row['foundation'])
    if 'actual_inputs' in row:
        value['actual_inputs'] = {k:dict(files=len(v), sha256=original.fingerprint(v)) for k,v in row['actual_inputs'].items()}
    return value


def save(row):
    write(ROOT/'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def check(row, committed=True):
    if committed:
        require(compact(row) == ur_validation.committed(RESULT, (KEY,)), 'commit exact local plan/stage first')
    pin = original.coherent.build.common.source.pin(P, PREFIXES)
    require(row['project']['source_sha256'] == pin['source_sha256'], 'frozen source/controller changed')
    native, _ = original.verify_original(expected=row['foundation'])
    require(sha(VALIDATION) == row['validation_provider']['sha256'], 'ZE validation provider changed')
    require(guards.driver_guard() == row['driver'] and row['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'boot/driver changed')
    require(sha(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'], 'runtime manifest changed')
    if 'binary' in row:
        require(sha(row['binary']['path']) == row['binary']['sha256'], 'actual executable changed')
        for group in row['actual_inputs'].values():
            require(all(sha(p) == h for p,h in group.items()), 'compiler/link inputs changed')
    return native


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'inspect existing handle; no retry')
    native, proof = original.verify_original()
    old = json.loads((B/'prefill-copy-engine-v1/record.json').read_text())
    require(old['stage'] == 'failed' and old['ownership_closed'] and not old['active']
        and old['binary']['sha256'] == sha(old['binary']['path']), 'closed V1 initialization refusal required')
    cc = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
    flags = ['-std=c++17', '-O2', '-fsycl', '-fsycl-host-only', '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13', '-Wall', '-Wextra', '-Werror']
    recipe = [
        ('compile', [cc, *flags, '-MD', '-MF', str(ROOT/'probe.d'), '-c', str(P/SOURCE), '-o', str(ROOT/'probe.o')]),
        ('link', [cc, *flags, str(ROOT/'probe.o'), LOADER, '-Wl,-t', '-o', str(ROOT/'probe')])]
    seed = json.loads((B/'coherent-generations-prefill-v1/record.json').read_text())
    keys = ['PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','ONEAPI_DEVICE_SELECTOR',
            'UR_ADAPTERS_FORCE_LOAD','NEOReadDebugKeys','EnableDirectSubmission',
            'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','STRATA_SYCL_PREFILL_SAME_QUEUE',
            'ZE_ENABLE_ALT_DRIVERS','ZES_ENABLE_SYSMAN','SYCL_CACHE_PERSISTENT']
    env = {k:v for k,v in seed['environment_template'].items() if k in keys}
    env.update(CUDA_MODULE_LOADING='EAGER', ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
               ZE_ENABLE_THREADING_VALIDATION='1', ZE_ENABLE_HANDLE_LIFETIME='1', ZEL_ENABLE_LOADER_LOGGING='1')
    cases = []
    def case(label, policy, size, iterations, batches, trace=False, speed=False):
        root = ROOT/label
        child = dict(env, TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'),
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if trace else 'UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'ur.trace'),
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'),
            ZEL_LOADER_LOG_CONSOLE='1', ZEL_LOADER_LOGGING_LEVEL='trace' if trace else 'warn',
            ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='1' if trace else '0')
        cases.append(dict(label=label, policy=policy, bytes=size, iterations=iterations, batches=batches,
            trace=trace, speed=speed, environment=child, root=str(root), argv=[str(ROOT/'probe'),
            '--policy',policy,'--bytes',str(size),'--iterations',str(iterations),'--batches',str(batches)]))
    for arm in ('sycl','copy'): case('metadata-'+arm,arm,0,0,0,True)
    for arm in ('sycl','copy'): case('tiny-'+arm,arm,256,2,1,True)
    for size in (1<<20, 256<<20):
        for arm in ('sycl','copy'): case('bytes-'+str(size)+'-'+arm,arm,size,1,1)
    for pair, order in enumerate([('sycl','copy'),('copy','sycl'),('sycl','copy')]):
        for arm in order: case('speed-'+str(pair)+'-'+arm,arm,256<<20,8,3,speed=True)
    row = dict(stage='prepared', project=original.coherent.build.common.source.pin(P,PREFIXES),
        foundation=proof, external_inputs=native['external_inputs'], runtime_manifest=seed['runtime_manifest'],
        validation_provider=dict(path=VALIDATION,sha256=sha(VALIDATION)),
        native_validation_probe=native['native_validation_probe'], recipe=recipe,
        build_environment=native['recipe']['environment'], cases=cases, results=[], commands=[],
        driver=guards.driver_guard(), boot=seed['boot'], passed=False, complete=False, active=False, ownership_closed=True,
        plan=dict(question='Synchronized pinned-host H2D throughput: qualified SYCL ordinal0 versus explicit immediate copy-only ordinal1/index0. Actual bus traffic/saturation remain unknown.',
            unit='Transfer primitive and its required queue creation/checked drain only. Same pinned source/device destination/payload/guards/host completion. No concurrent work,events,inference scheduling,math,bank layout,placement or decode changes.',
            baseline='Standalone queue reproduces qualified Runtime in-order/profiling properties and must return ordinal0/index0. This explicitly isolates a hardware diagnostic,not production dispatch or end-to-end model.',
            candidate='Same borrowed SYCL context/device;own async explicit-only in-order immediate list ordinal1/index0;5s finite host sync after every append,before readback/free. New unqualified queue policy despite unchanged copy-offload avoidance.',
            order='Metadata both arms,then256B traced both,then1MiB and256MiB quiet correctness both,then three independent alternating matched pairs with one warmup and three8-copy batches each.',
            timing='Each timed batch includes8 H2D submissions and checked per-copy host completion;minimum100ms;exact payload2147483648B. Initialization,allocation,pattern/reset and full guarded readback excluded. Preserve all batches,including ineligible ones;no automatic resizing/retry.',
            gate='Each previous process must pass before next:source/recipe/ELF/providers,pre-main UR validation,actual ordinals,ZE validation provider,ordered gates,full source/readback/edge guards,normal0,closed owners,quiet validation,no new fault,unchanged boot/driver. All9 speed batches per arm must exceed100ms to compare.',
            performance='Report every batch and process median requested decimalGB/s with spread,paired copy/sycl time ratios. No arbitrary adoption threshold;copy-only follow-up needs all three pair medians faster. No inference integration or model TPS claim from this probe.',
            teardown='All queues drain before list destruction and USM/context retirement. Uncertain drain fails _Exit before buffer destruction. External owned90s budget closes only own child;no reset/unbind or foreign signals. Qualified pre-main provider reference persists until process exit;general runtime teardown not qualified.',
            retention='One temporary standalone fixture beside unchanged fast14508 and namingA engine generations. Retire probe/objects/normal traces after compact resource decision and Git source/recipe recovery;no third inference image.',
            provenance='Main-authored from installed Intel SYCL interop/LevelZero1.17 API declarations;repository LICENSE,no external implementation or XeStrata source.',
            references=['https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md',
                'https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api.html#zecommandlisthostsynchronize',
                'https://github.com/oneapi-src/level-zero#logging-api-calls']))
    # A controller/logging change does not relabel or rebuild an unchanged TU.
    require(old['project']['source_sha256'][SOURCE] == sha(P/SOURCE)
        and sha(P/SOURCE) == __import__('hashlib').sha256(git(P,'show',old['project']['revision']+':'+SOURCE)).hexdigest(),
        'original probe native source must be recoverable and unchanged')
    row.update(recipe=old['recipe'],native_source_commit=old['project']['revision'],
        reused_native=dict(record_path=str(B/'prefill-copy-engine-v1/record.json'),
            record_sha256=sha(B/'prefill-copy-engine-v1/record.json'),binary=old['binary']))
    row['plan']['previous_failure'] = 'V1 metadata SYCL urContextCreate returned UNKNOWN before any work. Preserve V1failed. V2draft never executed. V3requires local plan to equal committed public plan and adds loader logging enablement;identical original C++image/recipe/compiler/SDK/runtime/safety and validation settings. Metadata gates precede any transfer.'
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache','owned'): (ROOT/name).mkdir(mode=0o700)
    save(row)
    print('Prepared standalone per-primitive probe;commit exact plan before build.',flush=True)


def permit(native,row):
    values = json.loads(Path(native['external_inputs']['path']).read_text())
    values.update(json.loads(Path(row['runtime_manifest']['path']).read_text()))
    values[str(P/SOURCE)] = sha(P/SOURCE)
    return values


def mappings(pid,row):
    allowed = json.loads(Path(row['runtime_manifest']['path']).read_text())
    for image in [row['binary'],row['validation_provider'],*row['native_validation_probe'].values()]:
        allowed[str(Path(image['path']).resolve())] = image['sha256']
    found = {}
    for line in Path('/proc',str(pid),'maps').read_text().splitlines():
        f = line.split(maxsplit=5)
        if len(f) == 6 and 'x' in f[1] and f[5].startswith('/'):
            p = str(Path(f[5]).resolve())
            require(allowed.get(p) == sha(p),'unfrozen executable provider: '+p)
            found[p] = allowed[p]
    return found


def execute(action):
    row = json.loads((ROOT/'record.json').read_text()); native = check(row)
    require(row['stage'] in (('prepared',) if action == 'build' else ('built','measuring')), 'ordered fresh stage required')
    supervisor = owned.Owner(ROOT/'owned',lambda:write(ROOT/'ownership.json',supervisor.commands))
    cursor = None
    try:
        if action == 'build':
            if 'reused_native' in row:
                reuse = row['reused_native']
                require(sha(reuse['record_path']) == reuse['record_sha256']
                    and sha(reuse['binary']['path']) == reuse['binary']['sha256'], 'original probe receipt/ELF changed')
                old = json.loads(Path(reuse['record_path']).read_text())
                require(all(sha(p) == h for g in old['actual_inputs'].values() for p,h in g.items()), 'original actual compiler/link/object inputs changed')
                row.update(stage='built',actual_inputs=old['actual_inputs'],binary=reuse['binary'],passed=True)
                print('Reused exact V1 native image;only committed logging/controller plan changes.',flush=True)
                return
            allowed = permit(native,row); actual = dict(compiled={},links={},objects={})
            for label,argv in row['recipe']:
                entry,out,err = supervisor.run(label,argv,row['build_environment'],P,**BUILD)
                require(owned.closed(entry) and entry['exit_code'] == 0,'compile/link failure: '+label)
                if label == 'compile':
                    dependencies = shlex.split((ROOT/'probe.d').read_text().replace('\\\n',' ').split(':',1)[1])
                    for name in dependencies:
                        p = str(Path(name).resolve()); h = sha(p)
                        require(allowed.get(p) == h,'unfrozen compiler input: '+p); actual['compiled'][p] = h
                    actual['objects'][str(ROOT/'probe.o')] = sha(ROOT/'probe.o')
                else:
                    allowed.update(actual['objects'])
                    for line in out.read_text().splitlines():
                        p = Path(line.strip())
                        if p.is_absolute() and p.is_file():
                            h = sha(p); key = str(p.resolve())
                            require(allowed.get(key) == h,'unfrozen link input: '+key); actual['links'][key] = h
            require(len(actual['compiled']) > 100 and len(actual['links']) > 8,'actual dependency census missing')
            row.update(stage='built',actual_inputs=actual,binary=dict(path=str(ROOT/'probe'),sha256=sha(ROOT/'probe')),passed=True)
        else:
            import subprocess
            index = len(row['results']); require(index < len(row['cases']),'all cases already closed')
            case = row['cases'][index]; root = Path(case['root'])
            require(not root.exists(),'existing case;inspect owner,never retry')
            root.mkdir(mode=0o700)
            for name in ('tmp','cache'): (root/name).mkdir(mode=0o700)
            journal = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
            cursor = journal.split('-- cursor: ',1)[1].strip()
            row.update(stage='measuring',active=True,ownership_closed=False); save(row)
            records = []; gates = []; providers = {}
            expected = ['before-runtime','runtime-ready'] + (['buffers-ready'] if case['bytes'] else []) + ['before-exit']
            def interaction(channel,_entry):
                while True:
                    item = json.loads(channel.line(30))
                    if item['kind'] == 'gate':
                        name = item['name']; require(name == expected[len(gates)],'gate order')
                        providers[name] = mappings(channel.proc.pid,row)
                        if name == 'runtime-ready':
                            require(VALIDATION in providers[name],'ZE validation layer not initialized')
                        gates.append(name); channel.send(b'continue\n')
                    else:
                        records.append(item)
                        if item['kind'] == 'done': channel.exit(20); break
            entry,out,err = supervisor.run(case['label'],case['argv'],engine_once.native_environment(row,case['environment']),P,
                interaction=interaction,**RUN)
            require(owned.closed(entry) and entry['exit_code'] == 0,'probe failed: '+str(err))
            validation = ur_validation.preload_evidence(err.read_text(),entry['pid'])
            require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size == 0,'UR validation failure')
            require(not re.search(r'\[(?:error|critical)\]|validation.*(?:error|failed)|LevelZero failure|copy probe:',err.read_text(),re.I),'ZE/probe error')
            if case['trace']:
                require('zeCommandListCreateImmediate' in err.read_text(),'actual ZE trace missing')
                if case['bytes'] and case['policy'] == 'copy':
                    require('zeCommandListAppendMemoryCopy' in err.read_text() and 'zeCommandListHostSynchronize' in err.read_text(),'raw append/drain trace missing')
            engines = [v for v in records if v['kind'] == 'engine']
            require(len(engines) == 1 and engines[0]['ordinal'] == (1 if case['policy']=='copy' else 0) and engines[0]['index'] == 0,'actual engine selection')
            batches = [v for v in records if v['kind'] == 'batch']
            require(len(batches) == (case['batches']+1 if case['bytes'] else 0) and gates == expected,'complete guarded batches/gates required')
            after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            fresh = [v for v in after.splitlines() if re.search(r'gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout)',v,re.I)]
            require(not fresh,'fresh GPU fault'); check(row,False)
            row['results'].append(dict(label=case['label'],passed=True,engine=engines[0],batches=batches,
                validation=validation,actual_initialized_providers=providers['runtime-ready'],
                native_exit_code=entry['exit_code'],ownership_closed=True,boot_unchanged=True,fresh_faults=fresh))
            row.update(passed=True,active=False,ownership_closed=True)
            if len(row['results']) == len(row['cases']):
                row.update(stage='closed',complete=True)
                row['timing_eligible'] = all(b['wall_s'] >= .1 for c,r in zip(row['cases'],row['results']) if c['speed'] for b in r['batches'] if not b['warmup'])
    except BaseException as exc:
        row.update(stage='failed',passed=False,complete=False,error=repr(exc)); raise
    finally:
        row['commands'].extend(supervisor.commands)
        row['active'] = supervisor.active is not None
        row['ownership_closed'] = supervisor.active is None and all(c.get('session_empty') and c.get('direct_child_reaped') for c in row['commands'])
        save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','complete','ownership_closed','timing_eligible')},sort_keys=True),flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'],['build'],['next']),'prepare|build|next')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        else: execute(sys.argv[1])
