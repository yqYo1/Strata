#!/usr/bin/env python3
"""Observe Q6 head routing and repeated real-model head handoffs; Main only.

Small thinking-enabled diagnostics are not speed or task-quality evidence.
Main-authored; repository LICENSE applies. No external implementation copied.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import check_q6_head_model as device
import check_dflash2_model as model
import check_prefill_drafter_heads as prompts
import native_request_protocol as protocol

build, guards, owner, validation = device.build, device.guards, device.owner, device.validation
P, B, RESULT = device.P, device.B, device.RESULT
ROOT, KEY = B / 'q6-head-model-entry-v3', 'q6_head_model_entry_v3'
REFERENCE = B / 'decode-iq3-audit-speed-v2/record.json'
STATUS = P / 'bench/results/2026-10-15-pr6-recovery-audit/upstream-status.json'
require, sha, write, signature = device.require, device.sha, device.write, build.signature


def native():
    """Reuse exact qualified native/fixture bytes with two new observer files.

    Keep native d174 and its recipe distinct from this controller admission;
    the failed V2 model request gate is not inherited as a success.
    """
    row = json.loads((build.ROOT / 'record.json').read_text())
    require(validation.committed(RESULT, (build.KEY,)) == build.compact(row) and
        row['stage'] == 'built' and row['passed'] and row['complete'] and row['ownership_closed'] and
        row['actual_inputs']['uncovered'] == 0, 'closed original native qualification required')
    previous = row['admission_project']; current = build.source.pin(P, build.PREFIXES)
    before, after = previous['source_sha256'], current['source_sha256']
    require(before == {n: hashlib.sha256(raw).hexdigest() for n, raw in
        build.source.blobs(P, previous['revision'], before)}, 'original admission source not recoverable')
    require(set(before) == set(after) and {n for n in before if before[n] != after[n]} ==
        {'tools/sycl/q6_head_model_entry.py', 'tools/sycl/q6_head_generation_speed.py'},
        'only named model/cohort observers may change;no native,fixture or build-reader change')
    check = dict(row, project=current); build.build.check_inputs(check)
    for name, digest in (row.get('generated_inputs', {}) | row.get('configured_files', {})).items():
        require(sha(name) == digest, 'original configured/recipe input changed')
    inputs = row['actual_inputs']; require(sha(inputs['path']) == inputs['sha256'], 'native census changed')
    census = json.loads(Path(inputs['path']).read_text())
    require(all(sha(p) == digest for field in ('compiled', 'generated', 'links')
        for p, digest in census[field].items()), 'actual native compiler/link input changed')
    require(row['native_source_commit'] == row['project']['revision'] and
        all(sha(v['path']) == v['sha256'] and Path(v['path']).stat().st_size == v['bytes']
            for v in row['binaries'].values()), 'original engine/fixture/source identity changed')
    return row


def check_placement(fields):
    # layer.cpp:kv_plan defaults to slots=pages/mode0. generate.cpp:INFO emits
    # a nonzero resident count only for streamed mode1;zero means fully resident.
    expected = dict(context='131072', kv='int8', kv_resident='0', expert_slots='1256',
        spec='4', pool_workers='5', pcie_frac='0.00', mtp_max='0')
    require(all(fields.get(k) == v for k, v in expected.items()), 'actual placement changed')


def memory_snapshot(pid):
    """Read owned DRM clients without a new GPU context or clocked observer."""
    clients = {}; skipped = 0
    try:
        for path in Path('/proc', str(pid), 'fdinfo').iterdir():
            try:
                with path.open() as stream: text = stream.read(65537)
            except FileNotFoundError: continue
            if len(text) > 65536: skipped += 1; continue
            fields = dict((k.strip(), v.strip()) for line in text.splitlines()
                          if ':' in line for k, v in [line.split(':', 1)])
            if fields.get('drm-pdev') != '0000:05:00.0': continue
            key = fields.get('drm-client-id', 'fd-' + path.name)
            clients[key] = {k: v for k, v in fields.items() if k in ('drm-pdev', 'drm-client-id', 'drm-driver') or
                            k.startswith(('drm-memory-', 'drm-resident-', 'drm-shared-', 'drm-total-'))}
        return dict(status='observed' if clients else 'unavailable', clients=list(clients.values()),
            skipped_oversized=skipped, scope='Owned per-client counters with their original units;not global free memory,live-tensor extent,traffic or saturation. Do not sum shared clients.')
    except OSError as error:
        return dict(status='unavailable', error=repr(error), scope='No memory or saturation claim.')


def routes(text, arm):
    base, dpas = {}, {}
    for line in text.splitlines():
        if not line.startswith(('strata Q6_ROUTE ', 'strata Q6_DPAS_HEAD ')): continue
        fields = dict(p.split('=', 1) for p in line.split()[2:])
        key = tuple(fields[k] for k in ('request', 'device', 'phase', 'tokens', 'capture_id'))
        require(fields['complete'] == '1', 'incomplete production head route')
        if line.startswith('strata Q6_ROUTE '):
            require(key not in base and fields['native_half_enabled'] == '0' and
                fields['replay_started'] == fields['replay_completed'], 'invalid production graph identity/count')
            base[key] = fields
        else:
            require(key not in dpas and fields['canonical_table_excludes_dpas'] == '1' and
                fields['layout'] == 'canonical' and fields['extra_weight_bytes'] == '0' and
                fields['calls'] == str(int(fields['phase'] == 'decode' and fields['tokens'] == '4')),
                'canonical DPAS head escaped its decode-C4/no-extra-weight policy')
            dpas[key] = fields
    require(base and ((set(base) == set(dpas)) if arm else not dpas), 'missing/unexpected DPAS route joins')
    ids = sorted({int(k[0]) for k in base})
    require(ids == [1, 2, 3], 'all three actual request route snapshots required')
    decode = []
    for request in ids:
        found = [(k, f) for k, f in base.items() if k[0] == str(request) and k[2:4] == ('decode', '4')]
        require(len(found) == 1 and int(found[0][1]['replay_completed']) > 0,
                'decode-C4 must actually replay in every request')
        k, fields = found[0]
        decode.append(dict(request=request, capture_id=int(k[4]),
            replay_completed=int(fields['replay_completed']), dpas_calls=int(dpas[k]['calls']) if arm else 0))
    require(decode[0]['capture_id'] != decode[-1]['capture_id'], 'long prefill must retire and recapture the head reader')
    return dict(snapshots=len(base), requests=ids, decode_C4=decode,
                dpas_only_decode_C4=bool(arm), prompt_and_C1_dpas_calls=0)


def parser_controls():
    def sample(arm):
        lines = []
        for request, capture in ((1, 1), (2, 1), (3, 2)):
            for phase, tokens in (('prompt', 4), ('decode', 1), ('decode', 4)):
                shared = f'request={request} device=0 phase={phase} tokens={tokens} capture_id={capture}'
                lines.append('strata Q6_ROUTE ' + shared + ' native_half_enabled=0 complete=1 replay_started=1 replay_completed=1')
                if arm: lines.append('strata Q6_DPAS_HEAD ' + shared +
                    ' layout=canonical extra_weight_bytes=0 canonical_table_excludes_dpas=1 calls=' + str(int(phase == 'decode' and tokens == 4)) + ' complete=1')
        return '\n'.join(lines)
    for arm in (0, 1): routes(sample(arm), arm)
    text = sample(1)
    variants = [text.replace('calls=0', 'calls=1', 1),
        '\n'.join(v for v in text.splitlines() if not v.startswith('strata Q6_DPAS_HEAD request=2')),
        text.replace('canonical_table_excludes_dpas=1', 'canonical_table_excludes_dpas=0', 1),
        text.replace('layout=canonical', 'layout=packed', 1),
        text.replace('extra_weight_bytes=0', 'extra_weight_bytes=521472000', 1),
        text.replace('replay_completed=1', 'replay_completed=0', 1),
        text + '\n' + text.splitlines()[0], text.replace('request=3', 'request=4')]
    for bad in variants:
        try: routes(bad, 1)
        except RuntimeError: continue
        raise RuntimeError('production head parser accepted a corrupted join/policy/count')
    fields = dict(context='131072', kv='int8', kv_resident='0', expert_slots='1256',
        spec='4', pool_workers='5', pcie_frac='0.00', mtp_max='0')
    check_placement(fields)
    for key, value in [('kv_resident', '131072'), ('kv_resident', '4096'),
                       ('context', '4096'), ('kv', 'fp16'), ('expert_slots', '1255')]:
        try: check_placement(dict(fields, **{key: value}))
        except RuntimeError: continue
        raise RuntimeError('incorrect/streamed model placement accepted')
    return dict(passed=True, positive_arms=2, corrupt_variants=8,
        full_resident_placement_positive=1, incorrect_placement_variants=5, GPU_executed=False)


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('controller', 'models', 'runtime_files', 'generator_runtime', 'commands')}
    out['controller'] = {k: v for k, v in row['controller'].items() if k != 'source_sha256'}
    out['generator_runtime'] = {k: v for k, v in row['generator_runtime'].items() if k != 'inputs'}
    out['commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'), ownership_closed=owner.closed(c),
        error=c.get('error'), errors=c.get('errors', [])) for c in row['commands']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    d = json.loads(RESULT.read_text()); d[KEY] = compact(row); write(RESULT, d)


def assets(row, full=False):
    require(signature(row['models']) == row['models_sha256'] and len(row['models']) == 15, 'model manifest changed')
    deadline = time.monotonic() + 300
    for name, value in sorted(row['models'].items()):
        p = Path(name)
        require(model.model_identity(p) == value['identity'], 'actual model identity changed: ' + name)
        if full:
            digest = hashlib.sha256()
            with p.open('rb', buffering=0) as stream:
                while chunk := stream.read(8 << 20):
                    require(time.monotonic() < deadline, 'whole-model digest300s budget')
                    digest.update(chunk)
            require(digest.hexdigest() == value['sha256'] and
                model.model_identity(p) == value['identity'], 'model bytes changed: ' + name)
            print('Verified model asset ' + p.name, flush=True)


def verify(row):
    require(validation.committed(RESULT, (KEY,)) == compact(row), 'commit exact model-entry stage first')
    pin = row['controller']
    require(signature(pin['source_sha256']) == pin['aggregate_sha256'] and
        {p: hashlib.sha256(raw).hexdigest() for p, raw in build.source.blobs(P, pin['revision'],
            pin['source_sha256'])} == pin['source_sha256'] and
        all(sha(P / p) == h for p, h in pin['source_sha256'].items()), 'controller changed')
    n = native()
    require(n['binaries'] == row['native_build']['binaries'] and
        sha(build.ROOT / 'record.json') == row['native_record']['sha256'], 'new engine changed')
    require(signature(validation.committed(RESULT, (device.KEY,))) == row['device_evidence_sha256'], 'device prerequisite changed')
    require(sha(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
        all(sha(p) == h for p, h in row['runtime_files'].items()), 'runtime files changed')
    require(prompts.generator_runtime() == row['generator_runtime'] and
        prompts.requests(Path(row['tokenizer'])) == row['requests'], 'thinking workload generation changed')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    assets(row); guards.no_model_owners([])


def prepare():
    require(not ROOT.exists() and KEY not in build.public(), 'fresh model-entry plan only')
    n = native(); prior = validation.committed(RESULT, (device.KEY,))
    require(prior['passed'] and prior['complete'] and prior['ownership_closed'] and
        all(any(r.get('mode') == arm and r['passed'] for r in prior['results'])
            for arm in ('submission', 'phase', 'reference', 'canonical')) and
        prior['binaries'] == n['binaries'], 'same-build device closure required')
    reference = json.loads(REFERENCE.read_text())
    old = validation.committed(STATUS, ('current_sync', 'dflash2_prefill_heads_check'))
    argv = list(old['argv']); argv[0] = n['binaries']['strata']['path']
    for flag in ('--draft', '--draft-model', '--draft-fallback', '--draft-conf', '--mtp-draft-vocab'):
        i = argv.index(flag); del argv[i:i + 2]
    args = reference['configuration']['args']; argv += ['--mtp', args[args.index('--mtp') + 1]]
    # The packed model failed at this exact resident-context/cache admission.
    # Exercise those allocations now, while keeping first model requests small.
    for flag, value in (('--max-context', '131072'), ('--kv-resident', '131072'), ('--prefill', '8192')):
        require(argv.count(flag) == 1, 'one explicit placement argument required: ' + flag)
        argv[argv.index(flag) + 1] = value
    env = dict(reference['environment_template'], STRATA_PREFILL_RELEASE_HEADS='1', STRATA_PREFILL_RELEASE_MTP_EXPERTS='0',
        STRATA_SYCL_CACHE_SCRATCH_LEASE='0', STRATA_MTP_BATCH_F16='0', STRATA_SNAPSHOT_VERIFY='0',
        STRATA_SYCL_PREFILL_MOE_WINDOW='0', STRATA_SYCL_PREFILL_MIXED_SCRATCH='0',
        STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR='0', STRATA_SYCL_PREFILL_SCORE_GEMM='0',
        STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS='0', STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED='0',
        STRATA_SYCL_DECODE_Q6_COLUMNS='1', STRATA_SYCL_DECODE_Q6_CENTERING='0',
        STRATA_SYCL_DECODE_Q6_NATIVE_HALF='0', STRATA_SYCL_Q6_ROUTE_AUDIT='1', STRATA_TRACE='1',
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0',
        LD_PRELOAD=prior['native_validation_probe']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=prior['native_validation_probe']['loader']['path'])
    env.pop('UR_LOG_TRACING', None); env.pop('STRATA_DECODE_TIMING', None)
    require(env['STRATA_PREFILL_NATIVE_DENSE_BANK'] == '1' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
        and env['EnableDirectSubmission'] == '0' and env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0', 'preserve runtime safety')
    row = dict(stage='prepared', source_commit=build.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        controller=build.source.pin(P, ['tools/sycl', 'tools/strata_tokenizer.py', 'data/expert-profile.bin']),
        native_source_commit=n['project']['revision'], native_build=dict(binaries=n['binaries']),
        native_qualification_reuse=dict(native_build_key=build.KEY, device_key=device.KEY,
            device_evidence_commit='51fbdc3812774bef1b7b0a0bc73501dbfa82fd2f',
            native_source_commit=n['project']['revision'], binaries_unchanged=True,
            actual_compiler_and_fixture_and_recipe_inputs_unchanged=True,
            changed_observers=['tools/sycl/q6_head_model_entry.py', 'tools/sycl/q6_head_generation_speed.py'],
            previous_failed_model_key='q6_head_model_entry_v2', previous_failure_commit='4fa9f6e9aaea658c1f3652cc3698af966c5f75ef',
            scope='Controller-only new plan;reuse exact original four native ELFs/fixtures/recipe and closed GPU owner proof. No failed model result or speed inherited.'),
        native_record=dict(path=str(build.ROOT / 'record.json'), sha256=sha(build.ROOT / 'record.json')),
        device_evidence_sha256=signature(prior), models=reference['model_assets'],
        models_sha256=signature(reference['model_assets']), model_files=15,
        model_bytes=sum(v['identity']['bytes'] for v in reference['model_assets'].values()),
        runtime_manifest=prior['runtime_manifest'], runtime_files=json.loads(Path(prior['runtime_manifest']['path']).read_text()),
        probe=prior['native_validation_probe'], generator_runtime=prompts.generator_runtime(),
        tokenizer=reference['configuration']['tokenizer'], requests=prompts.requests(Path(reference['configuration']['tokenizer'])),
        environment=env, argv=argv, boot=prior['boot'], driver=prior['driver'],
        budgets=dict(wall=900, cpu=10800, rss_cap=104 << 30, text_cap=64 << 20, file_cap=64 << 20, total_cap=256 << 20),
        plan='Committed parser controls first;fresh head0then1 processes,long513/short82/long513 thinking-on32output'
             ' requests in each. Context131072,KVresident131072,chunk8192,short-read128,1256cache,MTP4,'
             'no token/coupled graphs. This qualifies full-placement admission,not full-context processing.'
             ' Require actual READY/INFO,immutable environment,pre-main validation and pinned file-backed maps,'
             ' native full-resident INFO kv_resident=0,owned DRM snapshots at READY/after requests,'
             ' complete GEN/drafts,all request route joins,positive C4replays,two target-head retirements and'
             ' fresh capture identities after restoration. No output-equality requirement. Full15asset hashes'
             ' at preparation and final cohort closure;stat checks before each process. Stop first failure.',
        limits='Small ownership/routing diagnosis only;no clean PP/TTFT/TG claim,task quality,physical262144/'
               'RESTORE,ordinary provider teardown or default adoption. Long/agent/strong-and-weak thinking'
               ' benchmarks and independent prefill/decode preservation remain required.',
        results=[], commands=[], passed=False, complete=False, ownership_closed=True, model_executed=False,
        speed_eligible=False, model_adoption_qualified=False,
        retention='Commit compact per-request extents/draft counts/route identities/ownership and limitations.'
                  ' Retire successful stage logs and superseded standalone head roots after exact input recovery;'
                  ' keep new engine for imminent long comparison and preserve90f/552b/e1b references.')
    require({k: v['prompt_tokens'] for k, v in row['requests'].items()} == {'long': 513, 'short': 82}, 'known diagnostic geometry')
    assets(row, full=True)
    ROOT.mkdir(mode=0o700)
    for arm in (0, 1):
        path = ROOT / str(arm); path.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (path / name).mkdir(mode=0o700)
    save(row); print('Prepared same-engine head0/1 real-model entry;commit before controls.', flush=True)


def execute(mode):
    row = json.loads((ROOT / 'record.json').read_text()); verify(row)
    require(row['stage'] == ('prepared' if mode == 'host' else 'host_passed'), 'no duplicate model attempt')
    if mode == 'host':
        row['parser_controls'] = parser_controls(); row['stage'] = 'host_passed'; save(row)
        print('Production route parser controls passed;commit before native execution.', flush=True); return
    scope = None
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if v.startswith('-- cursor: '))
    def persist(): write(ROOT / 'execution.json', dict(active=bool(scope and scope.active), commands=scope.commands if scope else []))
    try:
        scope = owner.Owner(ROOT / 'owned', persist)
        for arm in (0, 1):
            assets(row); guards.no_model_owners([])
            env = dict(row['environment'], TMPDIR=str(ROOT / str(arm) / 'tmp'), XDG_CACHE_HOME=str(ROOT / str(arm) / 'cache'),
                PYTHONPYCACHEPREFIX=str(ROOT / str(arm) / 'cache/python'), STRATA_MEASUREMENT_ROOT=str(ROOT / str(arm)),
                STRATA_SYCL_DECODE_Q6_DPAS_HEAD=str(arm), UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / str(arm) / 'validation.log'))
            result = dict(arm=arm, passed=False, requests=[]); row['results'].append(result)
            def interact(io, command):
                startup = []
                while True:
                    line = io.line(480); startup.append(line); require(len(startup) <= 32, 'startup stdout cap')
                    if line.startswith('READY '): break
                require(line == 'READY 131072 stop', 'actual context/READY differs')
                info = [v[5:] for v in startup if v.startswith('INFO ')]
                require(len(info) == 1, 'one actual INFO required')
                result['placement'] = dict(v.split('=', 1) for v in info[0].split())
                check_placement(result['placement'])
                actual = dict(v.split('=', 1) for v in Path('/proc', str(command['pid']), 'environ').read_text().split('\0') if '=' in v)
                require(actual == env, 'native environment changed')
                evidence_row = dict(row, environment=env)
                result.update(model.runtime_evidence(evidence_row, command['pid'],
                    (ROOT / 'owned' / f'head{arm}.stderr').read_text(errors='replace')))
                result['memory_at_READY'] = memory_snapshot(command['pid'])
                print(f'Head{arm}: READY with pinned runtime and actual placement.', flush=True)
                for label in ('long', 'short', 'long'):
                    request = row['requests'][label]; parser = protocol.RequestProtocol(request['prompt_tokens'], request['generated'])
                    io.send(model.wire(request['ids'], request['generated']))
                    while parser.done is None: parser.feed(io.line(480))
                    require(int(parser.done[7]) > 0 and int(parser.done[6]) > 0, 'MTP must offer and accept actual drafts')
                    result['requests'].append(dict(label=label, prompt_tokens=request['prompt_tokens'],
                        generated=len(parser.tokens), accepted=int(parser.done[6]), offered=int(parser.done[7]),
                        output_ids_sha256=signature(parser.tokens), complete=True))
                    print(f'Head{arm}: {label} request complete.', flush=True)
                result['memory_after_requests'] = memory_snapshot(command['pid'])
                io.send(b'QUIT\n'); io.exit(30)
            e, out, err = scope.run(f'head{arm}', row['argv'], env, P, interaction=interact, **row['budgets'])
            require(owner.closed(e) and e['exit_code'] == 0, 'native model closure failed')
            text = err.read_text(errors='replace')
            result['routes'] = routes(text, arm)
            detached = re.findall(r'^strata prefill heads: target detached bytes=(\d+) source=owned-RAM D2H=(\d+) H2D-on-restore=(\d+)$', text, re.M)
            expected = ('521472000', '521472000', '521472000')
            require(detached == [expected, expected], 'two actual long-prefill target retire/restore transitions required')
            admission = 'strata native head DPAS: ready=1 columns=4 layout=canonical canonical_bytes=521472000 extra_device_weight_bytes=0 extra_host_weight_bytes=0 default_adopted=0'
            require([v for v in text.splitlines() if v.startswith('strata native head DPAS:')] == ([admission] if arm else []) and
                not re.search(r'UR_RESULT_ERROR_[A-Z_]+', text), 'head layout/UR failure')
            log = ROOT / str(arm) / 'validation.log'
            require(not log.exists() or log.stat().st_size == 0, 'native parameter-validation error')
            result.update(head_transitions=[dict(freed=int(v[0]), D2H=int(v[1]), H2D=int(v[2])) for v in detached],
                host_probe_accepted=True, native_exit_code=0)
            validation.finish(result, scope, cursor, row['boot'])
            require(result['passed'], 'model fault/closure gate failed')
            row['model_executed'] = True
        assets(row, full=True); row.update(stage='closed', passed=True, complete=True)
    except BaseException as error: row.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        row['commands'] = scope.commands if scope else []
        row['ownership_closed'] = not scope or scope.active is None and all(owner.closed(c) for c in scope.commands)
        final = dict(host_probe_accepted=row['passed']); validation.finish(final, scope, cursor, row['boot']) if scope else None
        row['fresh_faults'] = final.get('fresh_faults'); row['passed'] = bool(row['passed'] and final.get('passed') and row['ownership_closed'])
        row['final_fault_check'] = {k: v for k, v in final.items() if k != 'host_probe_accepted'}
        if not row['passed']: row['stage'] = 'failed'
        save(row)
    require(row['passed'], row.get('error', 'small head model entry failed'))
    print('Closed real-model head0/1 routes and repeated handoffs;speed/adoption remain unqualified.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['host'], ['run']), 'q6_head_model_entry.py prepare|host|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
