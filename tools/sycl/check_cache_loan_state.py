#!/usr/bin/env python3
"""Main-only remaining same-build state gates before the cache-loan model trial.

Repository LICENSE. Runs existing independently authored fixtures with their
actual production settings. No new native code, model generation or speed claim.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_cache_loan_cli as loan
import check_dflash2 as device

P, B, STATUS = loan.P, loan.B, device.STATUS
ROOT = B / 'dflash2-cache-loan-state-v2'
KEY = 'dflash2_cache_loan_state_v2_check'
PACK = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/dflash2/qwen3.8-flash-next-dflash2.q4_0.pack')
PACK_SHA = 'cda40d62d0eb8f37ed6e272142641a1d2f0de4a32f9e14d0416e7a71534849f1'
INPUTS = list(dict.fromkeys(['tools/sycl/check_cache_loan_state.py', 'tools/sycl/check_dflash2.py', *loan.INPUTS]))
require, digest, write = loan.require, loan.digest, loan.write


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('controller', 'environment', 'runtime_files', 'commands', 'native_build')}
    out['controller'] = {k: row['controller'][k] for k in ('revision', 'count', 'aggregate_sha256')}
    out['environment_sha256'] = loan.build.signature(row['environment'])
    out['native_build'] = {k: row['native_build'][k] for k in ('source_commit', 'project_source_sha256', 'record')}
    out['commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'),
        ownership_closed=loan.ownership.closed(c), error=c.get('error'), errors=c.get('errors', []),
        cleanup=c.get('cleanup', [])) for c in row['commands']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    status = json.loads(STATUS.read_text()); status['current_sync'][KEY] = compact(row); write(STATUS, status)


def verify(row):
    require(loan.validation.committed(STATUS, ('current_sync', KEY)) == compact(
        json.loads((ROOT / 'record.json').read_text())), 'commit exact state plan/result')
    pin = row['controller']
    require({p: hashlib.sha256(raw).hexdigest() for p, raw in loan.source.blobs(
        P, pin['revision'], pin['source_sha256'])} == pin['source_sha256'] and
        loan.build.signature(pin['source_sha256']) == pin['aggregate_sha256'] and
        all(digest(P / p) == h for p, h in pin['source_sha256'].items()), 'controller changed/unrecoverable')
    _, native = loan.native(); require(native == row['native_build'], 'candidate build differs')
    prior = loan.validation.committed(STATUS, ('current_sync', 'dflash2_cache_loan_device_check'))
    require(loan.build.signature(prior) == row['prior_device_sha256'], 'original same-build16case result changed')
    probe, reuse = loan.current_probe()
    require(probe == row['native_validation_probe'] and reuse == row['runtime_reuse'], 'validation runtime proof changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
        json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'], 'runtime map changed')
    for p, h in row['runtime_files'].items(): require(digest(p) == h, 'runtime ELF changed: ' + p)
    require(PACK.stat().st_size == row['pack']['bytes'] and digest(PACK) == PACK_SHA, 'stock pack changed')
    require(loan.guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'driver/boot changed')
    loan.guards.no_model_owners([])


def steps():
    small = ['tail_transitions_2046_2047_2048', 'stock_pack_context', 'stock_block_forward',
        'draft_output_span_admission', 'narrow_draft_output_guards', 'physical_drafter_tail', 'failed_owner_containment']
    full = ['wrapped_context', 'prefix_rebuild', 'parked_branch_restore', 'fresh_replay_equivalence', 'missing_context_mask']
    common = dict(wall=360, cpu=360, rss_cap=8 << 30, text_cap=64 << 20, file_cap=64 << 20, total_cap=192 << 20)
    result = [dict(label='pack', target='dflash2_pack_test', args=[str(PACK)], count=19,
        markers=['DFlash2 pack metadata: 19 cases passed; no GPU execution'], wait=None,
        budgets=dict(common, wall=45, cpu=30, file_cap=1 << 30, total_cap=2 << 30), GPU=False),
        dict(label='submission', target='sycl_submission_test', args=['--submission'], count=10,
        markers=['case stream_copy_ordering=PASS', 'case explicit_stream_profiling=PASS', 'case default_stream_profiling=PASS',
            *[f'case queued_kernels={n} numerical=PASS guards=PASS event_timing=PASS kernel_profiling=PASS' for n in (1,119,120,121,241)],
            'case ordinary_host_roundtrips=3 status=PASS', 'case teardown=PASS', 'fixture_status=PASS mode=submission'],
        wait='COMPONENT_READY before-exit', budgets=common, GPU=True)]
    for phase, names in [('small', small), ('full', full)]:
        result.append(dict(label='ring_' + phase, target='sycl_dflash2_ring_test',
            args=['--' + phase, str(PACK), '--supervised'], count=len(names),
            markers=[f'dflash2 ring {n} PASS' for n in names] + [f'dflash2 ring {phase} cases={len(names)} PASS'],
            wait='DF2_RING_WAIT', budgets=dict(common, wall=600, cpu=600), GPU=True))
    result.append(dict(label='mask', target='sycl_dflash2_mask_test', args=['--supervised'], count=12,
        markers=[f'dflash2 mask variant={v} chunk={c} {n} PASS' for v in (1,2) for c in (64,128)
            for n in ('stale_epoch','wrong_position','live_control')] + ['dflash2 mask cases=12 PASS'],
        wait='DF2_MASK_WAIT', budgets=common, GPU=True))
    return result


def prepare():
    require(not ROOT.exists(), 'one finite state cohort;no automatic retry')
    _, native = loan.native()
    prior = loan.validation.committed(STATUS, ('current_sync', 'dflash2_cache_loan_device_check'))
    require(prior['passed'] and prior['ownership_closed'] and prior['numerical_cases_passed'] == 16 and
        not prior['fresh_faults'] and prior['native_build']['record'] == native['record'] and
        prior['fixture'] == native['binaries']['sycl_dflash2_test'], 'closed exact same-build16case fixture required')
    probe, reuse = loan.current_probe()
    reference = json.loads((device.REFERENCE / 'record.json').read_text())
    env = {k: reference['environment_template'][k] for k in loan.ENV_KEYS}
    env.update(CUDA_MODULE_LOADING='EAGER', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr', UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
        UR_LOG_LOADER='level:warning;flush:warning;output:stderr', UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
        STRATA_DF2_RING_LOG='1', STRATA_DF2_LOG_MIB='64', STRATA_DF2_RING_STATS='1')
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0' and
        env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0' and env['ZES_ENABLE_SYSMAN'] == '0', 'qualified safety settings')
    manifest = dict(path=str(device.REFERENCE / 'runtime-inputs.json'), sha256=digest(device.REFERENCE / 'runtime-inputs.json'))
    row = dict(stage='prepared', passed=False, ownership_closed=True, commands=[], results=[],
        controller=loan.source.pin(P, INPUTS), native_build=native,
        previous_failure_key='current_sync.dflash2_cache_loan_state_check', prior_device_sha256=loan.build.signature(prior),
        native_validation_probe=probe, runtime_reuse=reuse, pack=dict(path=str(PACK), bytes=522458112, sha256=PACK_SHA),
        environment=env, runtime_manifest=manifest, runtime_files=json.loads(Path(manifest['path']).read_text()),
        driver=loan.guards.driver_guard(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), order=steps(),
        criteria='One immutable process per declared step,in order. All19stock-index CPU admissions,10submission cases,'
            '7small-ring then5wrap/rewind/parked-restore cases and12mask controls must pass with normal0/reap/empty session.'
            'Require exact same-build output/source/runtime pins,same-PID before-main parameter-validation proof,'
            'actual executable maps at supervised runtime gates,quiet validation beyond the one intentional pre-main null call,'
            'no fresh GPU fault. Full-ring phase drops API tracing after successful traced small;parameter/stage diagnostics remain.'
            'Stop first failure,no retry/reset. Submission selects--submission,which returns before graph capture.'
            'Pack uses private sparse522458112byte metadata fixtures with1GiBfile/2GiBapparent-artifact budget;no weights copied.'
            'No target generation,cache-loan lifecycle,quality,speed,physical full target262144 or default adoption qualification.',
        retention='Retain compact individual case outcomes,source/binary/runtime/pack identities,actual-mapping count/hash and fault/closure.'
            'Delete successful full traces/private caches after result commit. Preserve native90f and5fixtures/f633/552b/e1b for imminent model decision.')
    require(row['driver'] == reference['driver'] and row['boot'] == reference['boot'] and digest(PACK) == PACK_SHA,
        'current runtime or stock pack differs')
    ROOT.mkdir(mode=0o700)
    for case in row['order']:
        for n in ('tmp','cache'): (ROOT / case['label'] / n).mkdir(parents=True, mode=0o700)
    save(row);print('Prepared53remaining cases in5owned processes;commit before execution.', flush=True)


def mappings(row, fixture, pid):
    actual = set()
    for line in Path('/proc', str(pid), 'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and 'x' in fields[1] and fields[5].startswith('/'):
            require(not fields[5].endswith(' (deleted)'), 'removed executable mapping')
            actual.add(str(Path(fields[5]).resolve()))
    allowed = {str(Path(p).resolve()): h for p, h in row['runtime_files'].items()}
    for a in (fixture, *row['native_validation_probe'].values()): allowed[str(Path(a['path']).resolve())] = a['sha256']
    required = {str(Path(p).resolve()) for p in [row['environment']['UR_ADAPTERS_FORCE_LOAD'],
        row['environment']['ZE_ENABLE_ALT_DRIVERS'], '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0',
        row['native_validation_probe']['library']['path'], row['native_validation_probe']['loader']['path']]}
    require(len(actual) >= 8 and not actual - allowed.keys() and not required - actual,
        'unexpected/missing runtime maps: ' + repr((sorted(actual-allowed.keys()), sorted(required-actual))))
    require(all(digest(p) == allowed[p] for p in actual), 'mapped ELF bytes changed')
    return dict(files=len(actual), sha256=loan.build.signature({p:allowed[p] for p in sorted(actual)}), unknown=0, missing=0)


def faults(cursor):
    text = device.kernel_log('--after-cursor', cursor, '-n', '2048')
    require(len(text.splitlines()) < 2048, 'incomplete kernel window')
    return [x for x in text.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]


def run():
    row = json.loads((ROOT / 'record.json').read_text());require(row['stage'] == 'prepared', 'no retry')
    verify(row)
    cursor = next(x.split(': ',1)[1] for x in device.kernel_log('-n','0','--show-cursor').splitlines() if x.startswith('-- cursor: '))
    owner = None
    def persist():write(ROOT / 'active.json',dict(active=bool(owner and owner.active is not None),commands=owner.commands if owner else []))
    try:
        owner = loan.ownership.Owner(ROOT / 'owned',persist)
        for case in row['order']:
            require(not faults(cursor), 'fault before next step');loan.guards.no_model_owners([])
            fixture = row['native_build']['binaries'][case['target']]
            env = dict(row['environment'], TMPDIR=str(ROOT/case['label']/'tmp'), XDG_CACHE_HOME=str(ROOT/case['label']/'cache'))
            if case['label'] == 'ring_full':env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',STRATA_TRACE='1')
            if case['label'] == 'submission':env['STRATA_TEST_MAIN_GATES']='1'
            env = loan.engine_once.native_environment(row,env)
            result = dict(label=case['label'],passed=False,cases=case['count'],GPU=case['GPU']);row['results'].append(result)
            def interact(io, entry):
                lines=[]
                while True:
                    line=io.line(seconds=case['budgets']['wall']-15);lines.append(line);require(len(lines)<=40,'stdout line cap')
                    if line=='COMPONENT_READY before-runtime':io.send(b'X')
                    elif line=='COMPONENT_READY runtime-ready':
                        result['runtime_maps']=mappings(row,fixture,entry['pid']);io.send(b'X')
                    elif line==case['wait']:
                        result['runtime_maps']=mappings(row,fixture,entry['pid'])
                        if case['label']=='submission':
                            io.send(b'X')
                            require(io.line(seconds=25)=='case teardown=PASS','missing teardown marker')
                            require(io.line(seconds=25)=='fixture_status=PASS mode=submission','missing final submission status')
                        else:io.send(b'QUIT\n')
                        io.exit(25);break
            entry,out,err=owner.run(case['label'],[fixture['path'],*case['args']],env,P,
                interaction=interact if case['wait'] else None,**case['budgets'])
            require(loan.ownership.closed(entry) and entry['exit_code']==0,'case failed/ownership not normal: '+case['label'])
            lines=out.read_text().splitlines();text=err.read_text(errors='replace')
            require(all(lines.count(s)==1 for s in case['markers']),'missing/duplicate fixture cases: '+case['label'])
            proof=loan.validation.preload_evidence(text,entry['pid'])
            require(Path(proof['loader']).resolve()==Path(row['native_validation_probe']['loader']['path']).resolve(),'pre-main provider differs')
            before,body=text.split(' phase=before-main ',1);body=body.split('\n',1)[1]
            expected_errors=[] if case['label']=='ring_full' else ['UR_RESULT_ERROR_INVALID_NULL_HANDLE']
            require(re.findall(r'UR_RESULT_ERROR_[A-Z_]+',before)==expected_errors and
                not re.search(r'UR_RESULT_ERROR_[A-Z_]+',body),'unexpected UR error')
            if case['GPU'] and case['label']!='ring_full':require('urEnqueueKernelLaunch' in body,'missing small API trace')
            if not case['GPU']:require(not re.search(r'ur(?:Enqueue|Queue|Device|ContextCreate|USM|PlatformGet\()',body),'pack gate executed device work')
            result.update(passed=True,ownership_closed=True,native_pre_main_validation=True,exit_code=0,
                stdout_sha256=digest(out),fresh_faults=faults(cursor));require(not result['fresh_faults'],'fresh GPU fault')
            print('PASS',case['label'],case['count'],'cases',flush=True)
        verify(row);row.update(passed=True,stage='passed')
    except BaseException as error:row.update(passed=False,stage='failed',error=repr(error))
    finally:
        row['commands']=owner.commands if owner else []
        row['ownership_closed']=owner is None or owner.active is None and all(loan.ownership.closed(c)for c in owner.commands)
        try:
            row['fresh_faults']=faults(cursor)
            require(not row['fresh_faults'] and loan.guards.driver_guard()==row['driver'] and
                Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'fault/driver/boot change')
        except BaseException as error:row.update(passed=False,stage='failed',fault_error=repr(error))
        if not row['ownership_closed']:row.update(passed=False,stage='failed')
        save(row)
    require(row['passed'],row.get('error',row.get('fault_error','state gate failed')))
    print('PASS53remaining cases;no target-model/cache-loan/performance qualification.',flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'use prepare or run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (prepare if sys.argv[1]=='prepare' else run)()
