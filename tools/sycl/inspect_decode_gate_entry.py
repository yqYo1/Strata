#!/usr/bin/env python3
"""CPU-only replay of the closed native gate request after its controller fault.

Main-authored; repository LICENSE. Preserve V3's failed overall outcome. No
native/model/GPU process is launched;the saved native0 evidence is checked.
"""
import copy
import fcntl
import json
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import decode_gate_entry as entry

base,RESULT = entry.base,entry.RESULT
ROOT = entry.ROOT / 'postcheck-v1'
KEY = 'decode_gate_model_entry_v3_postcheck_v1'
SELF = Path(__file__).resolve()
require,digest,write = entry.require,entry.digest,entry.write


def pins(revision):
    values = base.controls(revision,controller=SELF)
    values.update(entry.sources(revision))
    return values


def compact(row):
    value = {k:v for k,v in row.items() if k != 'sources'}
    value['source_inputs'] = dict(files=len(row['sources']),sha256=entry.width.signature(row['sources']))
    return value


def save(row):
    write(ROOT / 'record.json',row)
    result = json.loads(RESULT.read_text());result[KEY] = compact(row);write(RESULT,result)


def basis():
    failed = json.loads((entry.ROOT / 'record.json').read_text())
    require(failed['stage'] == 'failed' and not failed['passed'] and
            failed['diagnosis']['physical_closure']['confirmed'], 'preserve failed V3 first')
    raw = json.loads((entry.ROOT / 'execution.json').read_text())
    require(len(raw['commands']) == 1, 'one original native command required')
    native = raw['commands'][0]
    require(base.supervision.closed(native) and native['exit_code'] == 0 and not raw['active'] and
            len(raw['gates']) == 4 and raw['output_tokens'] == 128, 'original native not normally closed')
    for owner in native['owners']:
        now = base.supervision.identity(owner['pid'])
        require(now is None or now['start_ticks'] != owner['start_ticks'], 'original native identity live')
    return failed,raw,native


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'one fresh CPU replay only')
    failed,raw,native = basis();entry.current(failed)
    revision = base.basis.build.git(entry.P,'rev-parse','HEAD').decode().strip()
    sources = pins(revision)
    files = [entry.ROOT / n for n in ('record.json','execution.json','owned/native.stdout','owned/native.stderr')]
    row = dict(stage='prepared',source_commit=revision,sources=sources,
        original_files={str(p):digest(p) for p in files},
        python=dict(path=str(Path(sys.executable).resolve()),sha256=digest(sys.executable),version=sys.version),
        limits=dict(wall_s=420,cpu_s=360,address_space_bytes=8<<30,file_bytes=16<<20),
        plan='One CPU-only replay with -X faulthandler,stage persistence and finite limits. Reuse original native0/gates/output,never launch the native engine. Verify exact frozen failed V3 and raw hashes,current source/runtime/provider/native identities,ordered complete serving protocol,128token hash,four gates,strict UR validation,T1/T4 Q6 heads,all6resource snapshots/cycle recomputation,final full15asset hashes and kernel-journal/boot/driver/closed owners. Keep V3 failed;this result verifies retained native gate evidence separately. No retry or hash refresh.',
        scope='No new GPU execution,model speed,task/numerical quality,resource saturation,ordinary provider teardown,physical262144 or Python-crash fix claim.',
        retention='After verified result commit,retain compact protocol/gate/resource/validation/source facts and one host-fault excerpt;retire normal native output/JIT after the next model collector consumer.',
        passed=False,complete=False,active=False,GPU_executed=False,stages=[])
    ROOT.mkdir(mode=0o700);save(row)
    print('Prepared CPU-only closed-native replay;commit before execution.',flush=True)


def current(frozen):
    require(base.committed(RESULT)[KEY] == compact(frozen) and pins(frozen['source_commit']) == frozen['sources'],
            'CPU replay plan/source changed')
    require(all(digest(p) == h for p,h in frozen['original_files'].items()), 'original failed/raw evidence changed')
    require(frozen['python'] == dict(path=str(Path(sys.executable).resolve()),sha256=digest(sys.executable),
            version=sys.version), 'CPU replay interpreter changed')


def run():
    row = json.loads((ROOT / 'record.json').read_text());frozen = copy.deepcopy(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'execution.json').exists() and
            'faulthandler' in sys._xoptions, 'fresh CPU replay with faulthandler required')
    with (ROOT / 'execution.json').open('x') as stream:stream.write('{}\n')
    resource.setrlimit(resource.RLIMIT_CPU,(360,360))
    resource.setrlimit(resource.RLIMIT_AS,(8<<30,8<<30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(16<<20,16<<20))
    signal.alarm(420);started = time.monotonic()
    def stage(name):
        row['stages'].append(name);write(ROOT / 'execution.json',dict(active=True,stages=row['stages']))
        print('CPU replay: '+name,flush=True)
    try:
        current(frozen);failed,raw,native = basis();entry.current(failed);stage('source-native-runtime')
        lines = (entry.ROOT / 'owned/native.stdout').read_text().splitlines()
        entry.startup_info(lines[0]);require(lines[1] == 'READY 131072 stop', 'original READY differs')
        protocol = entry.RequestProtocol()
        for line in lines[2:]:protocol.feed(line)
        require(protocol.done is not None and entry.width.signature(protocol.tokens) == raw['output_token_sha256'] and
                protocol.progress_count == raw['prompt_progress']['notifications'], 'saved protocol extent/hash differs')
        require([(v['request'],v['sequence'],v['rounds']) for v in raw['gates']] ==
                [(1,i+1,n) for i,n in enumerate(entry.ROUNDS)] and all(
                    a['raw_ns'] < b['raw_ns'] for a,b in zip(raw['gates'],raw['gates'][1:])), 'saved gate order differs')
        text = (entry.ROOT / 'owned/native.stderr').read_text()
        row['validation'] = entry.validation.validation_log(text,native['pid'],failed['native_validation_probe']['loader']['path'],False)
        stage('protocol-four-gates-strict-validation')
        snapshots = entry.width.short.snapshots(text,1,1)
        require({v['tokens'] for v in snapshots['windows'] if v['phase'] == 'decode'} == {1,4} and
                sum(v['replay_completed'] for v in snapshots['windows'] if v['phase'] == 'decode') >= 24,
                'actual completed T1/T4 extent differs')
        row['Q6'] = dict(completed_head_captures=snapshots['completed_head_capture_ids'],
            windows=[{k:v[k] for k in ('phase','tokens','capture_id','shapes','replay_started','replay_completed')}
                     for v in snapshots['windows']])
        stage('Q6-live-replay-conservation')
        observations = raw['resource_observations']
        require([v['label'] for v in observations] == ['ready','rounds-0','rounds-8','rounds-16','rounds-24','done'] and
                all(v['available'] for v in observations), 'six complete resource snapshots required')
        intervals = entry.client_resources.CycleIntervals()
        target = {k:native['owners'][0][k] for k in ('pid','start_ticks','sid')}
        for value in observations:
            sample = {k:value[k] for k in ('begin_monotonic_ns','end_monotonic_ns','clients')}
            require(intervals.update(dict(sample,target=target)) == value['engine_intervals'], 'cycle recomputation differs')
        samples=[];ticks=__import__('os').sysconf('SC_CLK_TCK')
        for first,last in zip(observations[2:4],observations[3:5]):
            wall=((last['begin_monotonic_ns']+last['end_monotonic_ns'])-
                  (first['begin_monotonic_ns']+first['end_monotonic_ns']))/2e9
            delta=last['process']['cpu_ticks']-first['process']['cpu_ticks']
            require(wall>0 and delta>=0, 'invalid CPU interval')
            ccs=[v for v in last['engine_intervals'] if v['engine']=='ccs']
            require(len(ccs)==1 and ccs[0]['eligible'], 'eligible single CCS interval required')
            samples.append(dict(begin=first['label'],end=last['label'],wall_s=wall,
                process_cpu_s=delta/ticks,cpu_core_equivalents=delta/ticks/wall,ccs_busy_fraction=ccs[0]['busy_fraction']))
        row['resources'] = dict(intervals=samples,ticks_per_second=ticks,
            all_snapshot_requested_vram_resident=all(c['memory_bytes']['drm-total-vram0']==
                c['memory_bytes']['drm-resident-vram0'] for v in observations for c in v['clients']),
            snapshot_vram_bytes=[dict(label=v['label'],clients=[dict(client_id=c['client_id'],
                requested=c['memory_bytes']['drm-total-vram0'],resident=c['memory_bytes']['drm-resident-vram0'])
                for c in v['clients']]) for v in observations],
            limits='Short synthetic request;CPU includes polling,CCS busy includes stalls. Boundary snapshots do not exclude transient paging;no physical bandwidth/saturation or clean speed claim.')
        stage('resource-counter-recomputation')
        entry.assets.model_pins(failed['configuration'],failed['model_assets'],full=True);stage('final-full-model-assets')
        journal = subprocess.check_output(['journalctl','-k','-b','--since',native['started_utc'],'--no-pager'],text=True,timeout=15)
        row['fresh_GPU_faults']=[v for v in journal.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',v)]
        require(not row['fresh_GPU_faults'] and entry.base.guards.driver_guard()==failed['driver'] and
                Path('/proc/sys/kernel/random/boot_id').read_text().strip()==failed['boot'], 'fault/boot/driver differs')
        current(frozen);basis();require(time.monotonic()-started<420, 'CPU replay whole deadline')
        row.update(passed=True,complete=True,native_gate_evidence_verified=True,original_V3_remains_failed=True)
    except BaseException as error:row['error']=repr(error)
    row.update(stage='closed' if row['passed'] else 'failed',active=False)
    save(row);write(ROOT / 'execution.json',dict(active=False,passed=row['passed'],stages=row['stages'],error=row.get('error')))
    require(row['passed'],str(row.get('error')))
    print('Saved native gates/protocol/assets verified;V3 controller outcome remains failed.',flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'],['run']), 'use prepare|run')
    with (entry.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        (prepare if sys.argv[1]=='prepare' else run)()
