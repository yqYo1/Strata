#!/usr/bin/env python3
"""Bounded real-model state handoff after restored-prefill fixture qualification.

Repository LICENSE. Main-authored using this fork's native GEN, checked snapshot
and owned-process interfaces. Synthetic IDs diagnose state,not speed or quality.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_restore as native
import full_context_model as models
import measurement_owner as owner_api

gen, guards, runtime = native.gen, native.guards, native.runtime
validation = runtime.validation
P, B = native.P, native.B
ROOT = B / 'restored-model-state-v1'
RESULT = gen.A / 'prefill-restore-model-current.json'
COMPONENT = gen.A / 'prefill-restore-device-current.json'
TEMPLATE = gen.A / 'phase-prefill-speed-current.json'
PROFILE = P / 'data/expert-profile.bin'
write, digest, fingerprint = native.write, native.build.digest, native.native.fingerprint


def requests():
    long = [(512 + i * 17) % 4096 for i in range(16385)]
    return [([i for i in range(33)], 16, 0), (long, 1, 0),
        ([128 + i for i in range(33)], 16, 0), (long + [256 + i for i in range(16)], 16, 16385)]


def wire(ids, count):
    return ('GEN ' + str(count) + ' ' + ','.join(map(str, ids)) + '\n').encode()


def component(revision):
    result = gen.committed(COMPONENT)
    guards.require(result['schema'] == 'restored-prefill-components-v1' and result['passed'] and
        result['complete'] and result['ownership_closed'] and not result['active'] and
        result['final_checks']['fresh_faults'] == [] and result['verification']['restore_smoke_checks'] == 113,
        'closed restored same-build component qualification required')
    original = result['original_plan']
    guards.require(hashlib.sha256(gen.git('show', original['commit'] + ':' + original['path'])).hexdigest() ==
        original['sha256'], 'original fixture plan cannot recover')
    full = json.loads((native.ROOT / 'record.json').read_text())
    native.current(full, controller_revision=revision)
    guards.require(full['source_commit'] == result['build_source_commit'] and
        full['assets']['strata'] == result['binary'] and fingerprint(full['assets']) ==
        result['native_build_reference']['assets_fingerprint'] and digest(native.ROOT / 'record.json') ==
        result['native_build_reference']['private_record_sha256'] and digest(runtime.RESULT) ==
        result['runtime_reference']['sha256'], 'qualified native package/runtime differs')
    return result


def recipe():
    base = models.setup()
    args = list(base['config']['args'])
    for key, value in (('--expert-profile', str(PROFILE)), ('--conversation-cache-mib', '8192'),
            ('--conversation-cache-slots', '2'), ('--conversation-cache-min-free-mib', '4096')):
        if key in args:
            args[args.index(key) + 1] = value
        else:
            args += [key, value]
    guards.require(all(args[args.index(k) + 1] == v for k, v in
        (('--max-context', '131072'), ('--kv', 'int8'), ('--kv-resident', '131072'),
         ('--expert-cache', '1256'), ('--prefill', '8192'), ('--pool-workers', '5'),
         ('--spec', '4'), ('--prompt-cache', '0'), ('--pcie-frac', '0'), ('--adapt-swaps', '0'))),
        'preserve selected decode placement/configuration')
    template = gen.committed(TEMPLATE)['environment_template']
    previous_root = template['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(previous_root, str(ROOT)) for k, v in template.items()}
    qualified = gen.committed(runtime.RESULT)
    env.update(LD_PRELOAD=qualified['preload']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=qualified['preload']['loader'],
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', STRATA_TRACE='1',
        STRATA_STATE_HASH='1', STRATA_SNAPSHOT_VERIFY='1', STRATA_DECODE_TIMING='1',
        STRATA_SYCL_CACHE_LOAN_AUDIT='1')
    env.pop('UR_LOG_TRACING', None)
    guards.require('STRATA_SYCL_PREFILL_SEPARATE_LAYOUT' not in env, 'retired layout must stay absent')
    for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission',
            'STRATA_SYCL_PREFILL_SAME_QUEUE', 'SYCL_CACHE_PERSISTENT'):
        guards.require(env[key] == qualified['environment'][key], 'qualified safety setting changed: ' + key)
    return base, args, env


def compact(row):
    return {k: v for k, v in row.items() if k not in ('model_assets', 'environment', 'commands')}


def current(row):
    guards.require(gen.committed(RESULT) == compact(row), 'commit exact model diagnostic plan')
    pins = gen.benchmark_pins(row['controller_source_commit'])
    guards.require(len(pins) == row['controller_files'] and fingerprint(pins) == row['controller_fingerprint'],
        'model controller/benchmark changed')
    result = component(row['controller_source_commit'])
    guards.require(result['binary'] == row['binary'] and result['build_source_commit'] == row['native_source_commit']
        and digest(COMPONENT) == row['component_sha256'], 'component qualification changed')
    base, args, env = recipe()
    guards.require(base['reference'] == row['base_reference'] and base['models'] == row['model_assets']
        and fingerprint(row['model_assets']) == row['model_fingerprint'] and args == row['args']
        and env == row['environment'] and fingerprint(env) == row['environment_fingerprint']
        and digest(TEMPLATE) == row['template_sha256'] and digest(PROFILE) == row['profile_sha256']
        and gen.git('show', row['controller_source_commit'] + ':data/expert-profile.bin') == PROFILE.read_bytes(),
        'model assets,profile,recipe or effective environment changed')
    models.model_pins(base['config'], row['model_assets'])


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(), 'fresh single-use real-model gate')
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    result = component(revision)
    base, args, env = recipe()
    models.model_pins(base['config'], base['models'], full=True)
    pins = gen.benchmark_pins(revision)
    guards.require('tools/sycl/restored_model.py' in pins, 'commit model controller before preparing')
    vocab = set(json.loads((Path(base['config']['tokenizer']) / 'vocab.json').read_text()).values())
    guards.require(set(range(4096)) <= vocab and min(vocab) == 0, 'synthetic input vocabulary differs')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir()
    cases = [dict(prompt=len(ids), max_new=count, reused=reused, wire_bytes=len(wire(ids, count)),
        wire_sha256=hashlib.sha256(wire(ids, count)).hexdigest()) for ids, count, reused in requests()]
    row = dict(stage='prepared', controller_source_commit=revision, controller_files=len(pins),
        controller_fingerprint=fingerprint(pins), native_source_commit=result['build_source_commit'],
        binary=result['binary'], component_sha256=digest(COMPONENT), args=args, environment=env,
        environment_fingerprint=fingerprint(env), template_sha256=digest(TEMPLATE),
        base_reference=base['reference'], model_assets=base['models'], model_fingerprint=fingerprint(base['models']),
        model_files=len(base['models']), profile_sha256=digest(PROFILE), vocab_size=max(vocab) + 1,
        requests=cases, boot=result['runtime_reference']['boot'], driver=result['runtime_reference']['driver'],
        budgets=dict(wall=1200, cpu=7200, rss_cap=104 << 30, text_cap=32 << 20,
            file_cap=32 << 20, total_cap=64 << 20),
        criteria='One fresh owned native serve process after the host protocol gate;four disjoint-prefix requests '
            '33/16385/33/16401 with16/1/16/16 output limits and last prefix reuse16385. Allow a valid earlier '
            'EOS with positive output;no generated-text identity requirement. Require startup/read/reuse '
            'extents,valid IDs,callbacks,draft precompute,actual BF16/vLLM/mode10 long dispatch,Full48 dense '
            'restoration and1256-slot payload/map restoration. RESTORE must verify all12 main layers,draft '
            'and running state before the last continuation. Check MTP committed positions,actual mapped '
            'ELF allowlist at READY and each DONE,pre-main/silent parameter validation,15 final full model '
            'hashes,normal owned exit,boot/driver and no fresh fault. Stop first failure;no retry. '
            'No identical-text,speed,task,HTTP/agent,physical262144 or ordinary runtime teardown claim.',
        retention='Keep compact individual request/route/state/restore/owner/fault outcomes and source/recipe '
            'identities. Retire successful stage logs after committed replay. Original6e40 remains selected;'
            'this restored generation stays only for imminent independent long-input/decode comparisons.',
        protocol=None, observation=None, active=False, ownership_closed=True, passed=False, complete=False,
        production_selected=False, speed_qualified=False, decode_preservation_qualified=False,
        physical262144_qualified=False, ordinary_runtime_teardown_qualified=False)
    write(ROOT / 'record.json', row); write(RESULT, compact(row))
    print('Prepared bounded restored-model state plan;commit before protocol/model execution.', flush=True)


PEER = r'''
import hashlib,json,sys
print('INFO kv=int8 kv_resident=0 expert_slots=1256 spec=4 pool_workers=5 conversation_cache_mib=8192 conversation_cache_slots=2 conversation_cache_min_free_mib=4096')
print('READY 131072 stop',flush=True)
for expected in json.loads(sys.argv[1]):
 line=sys.stdin.buffer.readline();assert len(line)==expected['wire_bytes'] and hashlib.sha256(line).hexdigest()==expected['wire_sha256']
 _,count,ids=line.decode().strip().split(' ',2);n=len(ids.split(','));count=int(count)
 assert n==expected['prompt'] and count==expected['max_new']
 print('PP',n-1)
 for _ in range(count):print('T 1')
 print('DONE',count,n,0,0,'length',0,0,expected['reused'],0,0,0,0,0,n-expected['reused'],flush=True)
assert sys.stdin.buffer.readline()==b'QUIT\n'
'''


def interact(io, row, observation, real):
    info = {}
    def mappings():
        files = runtime.executable_mappings(Path('/proc', str(io.proc.pid), 'maps').read_text())
        allowed = gen.committed(runtime.RESULT)['runtime_files']
        guards.require(all(p in allowed or p == str(Path(row['binary']['path']).resolve()) for p in files)
            and any('libze_intel_gpu.so.1.17.39758' in p for p in files), 'actual runtime provider differs')
        observation['mapped_runtime_boundaries'] = observation.get('mapped_runtime_boundaries', 0) + 1
    while True:
        line = io.line(600)
        guards.require(not line.startswith('ERR'), line)
        if line.startswith('INFO '):
            info.update(v.split('=', 1) for v in line.split()[1:] if '=' in v)
        if line.startswith('READY '):
            guards.require(line == 'READY 131072 stop', 'ready context differs'); break
    # INFO reports0 for fully resident Mode0;131072 is the CLI allocation
    # limit,not a Mode1 ring extent. Check the actual configured context at READY.
    for key, value in dict(kv='int8', kv_resident='0', expert_slots='1256', spec='4', pool_workers='5',
            conversation_cache_mib='8192', conversation_cache_slots='2', conversation_cache_min_free_mib='4096').items():
        guards.require(info.get(key) == value, 'actual startup differs: ' + key)
    if real:
        mappings()
    observation['requests'] = []
    for (ids, count, reused), expected in zip(requests(), row['requests']):
        payload = wire(ids, count)
        guards.require(len(payload) == expected['wire_bytes'] and hashlib.sha256(payload).hexdigest() ==
            expected['wire_sha256'], 'request wire changed')
        for offset in range(0, len(payload), 256 << 10):
            io.send(payload[offset:offset + (256 << 10)])
        tokens, publications = [], []
        while True:
            line = io.line(600); guards.require(not line.startswith('ERR'), line)
            if line.startswith('PP '): publications.append(int(line.split()[1]))
            if line.startswith('T '): tokens.append(int(line.split()[1]))
            if not line.startswith('DONE '): continue
            f = line.split()[1:]; guards.require(len(f) == 14, 'DONE schema differs')
            actual = dict(generated=int(f[0]), prompt=int(f[1]), finish=f[4], accepted=int(f[5]),
                offered=int(f[6]), reused=int(f[7]), read=int(f[13]), ids=tokens)
            guards.require(actual['generated'] == len(tokens) and 0 < len(tokens) <= count and actual['prompt'] == len(ids)
                and actual['reused'] == reused and actual['read'] == len(ids) - reused
                and 0 <= actual['accepted'] <= actual['offered'] and all(0 <= t < row['vocab_size'] for t in tokens)
                and (actual['finish'] == 'stop' or actual['finish'] == 'length' and len(tokens) == count)
                and publications[-1:] == [len(ids) - 1]
                and all(reused < v <= len(ids) - 1 for v in publications)
                and all(a < b for a, b in zip(publications, publications[1:])), 'request extent/position differs')
            actual['publications'] = dict(count=len(publications), last=publications[-1], sha256=fingerprint(publications))
            observation['requests'].append(actual); break
        if real: mappings()
    io.send(b'QUIT\n'); io.exit(90)


def evidence(text, observation, row, pid):
    activation = validation.preload_evidence(text, pid)
    guards.require(activation['loader'] == gen.committed(runtime.RESULT)['preload']['loader'], 'UR provider differs')
    boundary = re.search(r'^strata UR validation: .*phase=before-main[^\n]*\n', text, re.M)
    body = text[boundary.end():]
    guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed', body),
        'runtime error after canary')
    states = re.findall(r'^strata serve: STATE_HASH L=.*$', body, re.M)
    windows = [int(v) for v in re.findall(r'^strata decode timing: (\d+) windows,', body, re.M)]
    guards.require(len(states) == len(windows) == 4, 'state/window accounting incomplete')
    frames = re.split(r'^strata serve: STATE_HASH L=.*$', body, flags=re.M)
    for i, (state, nw, actual, frame) in enumerate(zip(states, windows, observation['requests'], frames)):
        fields = dict(re.findall(r'(\w+)=([0-9a-f,-]+)', state))
        guards.require(set(fields) == {'L','gdn','ple','tail','pooled','kv','mtp','stale','dead','pooled_full','ple_prev'}
            and int(fields['L']) == actual['prompt'] - 1 + nw + actual['accepted'] <= 131072,
            'MTP committed-position/state-domain conservation failed')
        draft = re.findall(r'^strata serve: DRAFT_PREFILL path=(batched|token) mode=([012]) cells=(\d+)$', frame, re.M)
        cells = [int(v[2]) for v in draft]; end = actual['reused']; ends = []
        for count in cells: end += count; ends.append(end)
        guards.require(cells and all(0 < n <= 8192 for n in cells) and end == actual['prompt'] - 1 and
            actual['publications'] == dict(count=len(ends), last=ends[-1], sha256=fingerprint(ends)),
            'draft callbacks do not cover the prompt')
        guards.require(len(re.findall(r'^strata loan audit: phase=payload rows=1256 bytes=2575206400 '
            r'lent=\d+ same_slots=1 payload_match=1$', frame, re.M)) == 1 and all(len(re.findall(
            r'^strata loan audit: phase=' + phase + r' table_entries=24576 table_match=1 ', frame, re.M)) == 1
            for phase in ('lend','refill')), 'decode cache payload/map not restored')
        layers = 48 if i == 1 else 0
        guards.require(re.search(r'^strata prefill dense: phase=restored residency=Full activations=' +
            str(layers) + r' ', frame, re.M), 'dense owner handoff differs')
        if i == 1:
            guards.require('tokens=16384 chunk=8192 residual_dtype=bf16 residual_element_bytes=2' in frame and
                'effective_mode=10 ' in frame and 'QSA BF16 operands, FP32 online state/accumulation' in frame and
                'phase=after-restore ' in frame, 'actual long-prefill dispatch/restoration differs')
        else:
            guards.require('layer-major candidate,' not in frame and 'phase=before-detach ' not in frame,
                'short request withdrew dense/MTP weights')
        actual.update(state_L=int(fields['L']), windows=nw, state_sha256=fingerprint(fields),
            prefill_path='layer-major' if i == 1 else 'chunk-major', dense_activations=layers,
            cache_slots_restored=1256, cache_payload_bytes=2575206400)
    restores = re.findall(r'^strata serve: conversation cache: restored (\d+) tokens \((live|checkpoint)\)', body, re.M)
    proofs = [dict(re.findall(r'(\w+)=([^ ]+)', s)) for s in re.findall(r'^strata serve: SNAPSHOT_VERIFY .*$', body, re.M)]
    guards.require(restores == [('16385', 'live')] and len(proofs) == 1, 'required live RESTORE missing')
    proof = proofs[0]
    guards.require(proof.get('result') == 'PASS' and int(proof.get('upto', '-1')) == 16385
        and int(proof.get('main_layers', '-1')) == 12 and all(int(proof.get(k, '0')) > 0
            for k in ('main_bytes','draft_bytes','running_bytes')), 'full main/draft/running RESTORE verification missing')
    observation['restore'] = {k: proof[k] for k in ('upto','main_layers','main_bytes','draft_bytes','running_bytes','result')}


def run(row, protocol):
    current(row)
    guards.require((protocol and row['stage'] == 'prepared' and row['protocol'] is None) or
        (not protocol and row['stage'] == 'protocol-closed' and row['protocol']['passed']), 'ordered single-use stages')
    initial, observation, owner = copy.deepcopy(row), dict(passed=False, GPU_executed=not protocol), None
    label = 'protocol' if protocol else 'model'
    guards.require(not (ROOT / label).exists(), 'never retry a closed or partial stage')
    cursor_text = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True, timeout=15)
    cursor = next(v.split(': ', 1)[1] for v in cursor_text.splitlines() if v.startswith('-- cursor: '))
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else []))
    try:
        owner = owner_api.Owner(ROOT / label, persist)
        argv = ['/usr/bin/python3','-B','-c',PEER,json.dumps(row['requests'])] if protocol else [row['binary']['path'],'--serve',*row['args']]
        env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', PYTHONDONTWRITEBYTECODE='1') if protocol else row['environment']
        budgets = dict(wall=60,cpu=60,rss_cap=1 << 30,text_cap=1 << 20,file_cap=1 << 20,total_cap=4 << 20) if protocol else row['budgets']
        entry, out, err = owner.run(label, argv, env, P,
            interaction=lambda io, entry: interact(io, row, observation, not protocol), **budgets)
        guards.require(owner_api.closed(entry) and entry['exit_code'] == 0, 'normal owned exit required')
        if protocol: guards.require(err.stat().st_size == 0, 'protocol peer stderr')
        else:
            evidence(err.read_text(), observation, row, entry['pid'])
            base = models.setup(); models.model_pins(base['config'], row['model_assets'], full=True)
        current(initial)
        observation.update(host_probe_accepted=True, exit_code=0, passed=True)
    except BaseException as error:
        observation['error'] = repr(error)
    finally:
        if owner is not None: validation.finish(observation, owner, cursor, row['boot'])
        observation['passed'] = bool(observation.get('passed') and observation.get('driver_after') == row['driver'])
        row['active'] = bool(owner and owner.active is not None)
        row['ownership_closed'] = observation.get('ownership_closed', owner is None)
        row['protocol' if protocol else 'observation'] = observation
        row['stage'] = ('protocol-closed' if protocol else 'model-closed') if observation.get('passed') else 'failed'
        row['passed'] = row['complete'] = bool(not protocol and observation.get('passed'))
        persist(); write(RESULT, compact(row))
    print(json.dumps({k: observation.get(k) for k in ('passed','exit_code','ownership_closed','fresh_faults','error')}), flush=True)
    return int(not observation.get('passed'))


if __name__ == '__main__':
    guards.require(sys.argv[1:] in (['prepare'], ['protocol'], ['model']), 'prepare|protocol|model')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare()
        else: sys.exit(run(json.loads((ROOT / 'record.json').read_text()), sys.argv[1:] == ['protocol']))
