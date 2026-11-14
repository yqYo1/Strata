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
ROOT, KEY = B / 'q6-head-packed-model-entry-v1', 'q6_head_packed_model_entry_v1'
REFERENCE = B / 'decode-iq3-audit-speed-v2/record.json'
STATUS = P / 'bench/results/2026-10-15-pr6-recovery-audit/upstream-status.json'
require, sha, write, signature = device.require, device.sha, device.write, build.signature


def native():
    """Check the sole-packed native source, recipe, four ELFs and actual inputs.

    The shared reader preserves native6dd and admissionf518 identities, rather
    than relabeling the retained engine with this new model-controller commit.
    """
    return device.native()


def controller_pin():
    pin = build.source.pin(P, ['tools/sycl', 'tools/strata_tokenizer.py', 'data/expert-profile.bin'])
    # This later consumer imports us; we never import or execute it. Keeping it
    # out of this plan lets the closed model proof qualify its subsequent plan
    # without pretending that a changed measurement dependency is unchanged.
    del pin['source_sha256']['tools/sycl/q6_head_generation_speed.py']
    pin['aggregate_sha256'] = signature(pin['source_sha256'])
    return pin


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


def requests(tokenizer):
    # Keep the real8192chunk/cache placement. The old513-token fixture could
    # never enter uses_layer_major and therefore could not test head retirement.
    content = 'Review the following stable merge sort requirements, then implement and explain the algorithm.\n'
    content += '\n'.join(f'Requirement {i}: preserve the original order of equal elements and explain the merge invariant.'
                         for i in range(512))
    # The shared small-model helper intentionally caps requests at2048. Keep
    # that contract intact and build this bounded multi-chunk fixture here.
    sys.path.insert(0, str(P / 'tools'))
    from strata_tokenizer import Tokenizer
    from jinja2 import Environment
    vocab = json.loads((tokenizer / 'vocab.json').read_text()); tokens = [None] * len(vocab)
    for name, i in vocab.items(): tokens[i] = name
    config = json.loads((tokenizer / 'tokenizer.json').read_text())
    require(config['pre'] == 'qwen35' and len(tokens) == 248320, 'unexpected target tokenizer')
    tok = Tokenizer(tokens, (tokenizer / 'merges.txt').read_text().split('\n'),
        json.loads((tokenizer / 'token_type.json').read_text()), config['pre'], config['special_ids'])
    text = Environment().from_string((tokenizer / 'chat_template.jinja').read_text()).render(
        messages=[dict(role='user', content=content)], tools=None, add_generation_prompt=True, enable_thinking=True)
    require(text.endswith('<think>\n'), 'thinking prompt suffix changed')
    ids = tok.encode(text, parse_special=True)
    require(8192 < len(ids) <= 32768 and all(type(i) is int and 0 <= i < 248320 for i in ids) and
        tok.decode(ids) == text, 'multi-chunk real prompt/tokenizer round trip')
    long = dict(ids=ids, generated=32, thinking=True, rendered_sha256=hashlib.sha256(text.encode()).hexdigest(),
        wire_sha256=hashlib.sha256(model.wire(ids, 32)).hexdigest(), prompt_tokens=len(ids))
    short = model.request_ids(tokenizer)
    require(8192 < long['prompt_tokens'] <= 32768 and short['prompt_tokens'] == 82,
            'diagnostic must exceed one8192chunk and retain the original short request')
    long['generator'] = 'q6_head_model_entry.requests:512numbered stable-merge requirements,thinking template'
    short['generator'] = 'check_dflash2_model.request_ids:original82ID thinking prompt'
    return dict(long=long, short=short)


def route_tables(text, arm):
    require(type(arm) is int and arm in (0, 2), 'explicit canonical0 or sole-packed2 arm')
    base, packed = {}, {}
    for line in text.splitlines():
        require(not line.startswith('strata Q6_DPAS_HEAD '), 'unexpected canonical-DPAS trial marker')
        if not line.startswith(('strata Q6_ROUTE ', 'strata Q6_PACKED_HEAD ')): continue
        pairs = [p.split('=', 1) for p in line.split()[2:]]
        require(all(len(p) == 2 for p in pairs), 'malformed head route fields')
        fields = dict(pairs)
        require(len(fields) == len(pairs) and
            all(k in fields for k in ('request', 'device', 'phase', 'tokens', 'capture_id', 'complete')),
            'missing or duplicate head route identity')
        require(all(fields[k].isdecimal() for k in ('request', 'tokens', 'capture_id')) and
            int(fields['request']) > 0 and int(fields['capture_id']) > 0 and fields['device'] == '0' and
            fields['phase'] in ('prompt', 'decode') and 1 <= int(fields['tokens']) <= 4,
            'unplanned head route geometry')
        key = tuple(fields[k] for k in ('request', 'device', 'phase', 'tokens', 'capture_id'))
        require(fields['complete'] == '1', 'incomplete production head route')
        if line.startswith('strata Q6_ROUTE '):
            require(key not in base and fields.get('native_half_enabled') == '0' and
                fields.get('scope') == 'live_capture_cumulative' and
                str(fields.get('replay_started', '')).isdecimal() and
                fields['replay_started'] == fields.get('replay_completed'),
                'invalid production graph identity/count')
            base[key] = fields
        else:
            require(key not in packed and fields.get('layout') == 'sole-packed' and
                fields.get('extra_weight_bytes') == '0' and fields.get('calls') == '1',
                'sole-packed head layout/call/zero-extra-weight policy')
            packed[key] = fields
    require(base and ((set(base) == set(packed)) if arm == 2 else not packed),
            'missing/unexpected sole-packed route joins')
    return base, packed


def routes(text, arm):
    base, packed = route_tables(text, arm)
    ids = sorted({int(k[0]) for k in base})
    require(ids == [1, 2, 3], 'all three actual request route snapshots required')
    decode = []
    for request in ids:
        found = [(k, f) for k, f in base.items() if k[0] == str(request) and k[2:4] == ('decode', '4')]
        require(len(found) == 1 and int(found[0][1]['replay_completed']) > 0,
                'decode-C4 must actually replay in every request')
        k, fields = found[0]
        decode.append(dict(request=request, capture_id=int(k[4]),
            replay_completed=int(fields['replay_completed']), packed_calls=int(packed[k]['calls']) if arm else 0))
    require(decode[0]['capture_id'] != decode[-1]['capture_id'], 'long prefill must retire and recapture the head reader')
    epochs = {}
    for item in decode:
        capture, completed = item['capture_id'], item['replay_completed']
        require(completed >= epochs.get(capture, 0), 'cumulative replay count decreased within one capture')
        epochs[capture] = completed
    return dict(snapshots=len(base), requests=ids, decode_C4=decode,
        actual_decode_C4_replays=sum(epochs.values()), sole_packed_all_live_head_windows=arm == 2,
        scope='Per-request live-capture snapshots;coalesce repeated capture identities before summing replays.')


def handoffs(text, arm, transitions):
    require(type(arm) is int and arm in (0, 2) and transitions in (1, 2), 'explicit model handoff arm/count')
    target = re.findall(r'^strata prefill heads: target detached bytes=(\d+) source=owned-RAM D2H=(\d+) H2D-on-restore=(\d+)$', text, re.M)
    expected = ('521472000', '521472000', '521472000')
    require(target == [expected] * transitions, 'actual long-prefill target retire/restore transitions required')
    draft = re.findall(r'^strata prefill heads: draft detached bytes=(\d+) retained-map-bytes=(\d+)$', text, re.M)
    require(draft == [('0' if arm == 2 else '223227900', '425196')] * transitions,
            'actual MTP subset retirement/map extent changed')
    admission = 'strata native head packed: ready=1 columns=1..4 weight_bytes=521472000 canonical_device_weight_bytes=0 extra_device_weight_bytes=0 permanent_host_weight_bytes=0 default_adopted=0'
    mtp = 'strata mtp: shared packed head tokens=106299 subset_weight_bytes=0 subset_logit_bytes=425196 saved_device_bytes=222802704'
    lines = text.splitlines()
    require([v for v in lines if v.startswith('strata native head packed:')] == ([admission] if arm == 2 else []) and
        [v for v in lines if v.startswith('strata mtp: shared packed head ')] == ([mtp] if arm == 2 else []) and
        not any(v.startswith('strata native head DPAS:') for v in lines), 'head/MTP owner admission changed')
    return dict(target=[dict(freed=int(v[0]), D2H=int(v[1]), H2D=int(v[2])) for v in target],
        draft=[dict(freed=int(v[0]), retained_map_bytes=int(v[1])) for v in draft],
        sole_packed_owner=arm == 2, scope='Requested allocation/copy extents,not physical bus traffic or tensor residency.')


def parser_controls():
    def sample(arm):
        lines = []
        for request, capture, completed in ((1, 2, 10), (2, 2, 18), (3, 6, 10)):
            for phase, tokens in (('prompt', 4), ('decode', 1), ('decode', 4)):
                shared = f'request={request} device=0 phase={phase} tokens={tokens} capture_id={capture}'
                lines.append('strata Q6_ROUTE ' + shared + f' scope=live_capture_cumulative native_half_enabled=0 complete=1 replay_started={completed} replay_completed={completed}')
                if arm: lines.append('strata Q6_PACKED_HEAD ' + shared +
                    ' layout=sole-packed extra_weight_bytes=0 calls=1 complete=1')
        return '\n'.join(lines)
    for arm in (0, 2):
        require(routes(sample(arm), arm)['actual_decode_C4_replays'] == 28, 'cumulative replay snapshots double-counted')
    text = sample(2)
    variants = [text.replace('calls=1', 'calls=0', 1),
        '\n'.join(v for v in text.splitlines() if not v.startswith('strata Q6_PACKED_HEAD request=2')),
        text.replace('layout=sole-packed', 'layout=canonical', 1),
        text.replace('extra_weight_bytes=0', 'extra_weight_bytes=521472000', 1),
        text.replace('replay_completed=10', 'replay_completed=0', 1),
        text + '\n' + text.splitlines()[0], text.replace('request=3', 'request=4'),
        text.replace('tokens=4', 'tokens=8', 1), text.replace('capture_id=2', 'capture_id=0', 1),
        text.replace('scope=live_capture_cumulative', 'scope=per_request', 1),
        text.replace('replay_started=18 replay_completed=18', 'replay_started=9 replay_completed=9'),
        text + '\nstrata Q6_DPAS_HEAD request=1 device=0 phase=decode tokens=4 capture_id=2',
        text.replace('native_half_enabled=0', 'native_half_enabled=1', 1),
        text.replace('request=1 device=0', 'request=1 request=1 device=0', 1),
        text.replace('complete=1', 'complete=0', 1)]
    for bad in variants:
        try: routes(bad, 2)
        except RuntimeError: continue
        raise RuntimeError('production head parser accepted a corrupted join/policy/count')
    try: routes(text, 0)
    except RuntimeError: pass
    else: raise RuntimeError('reference accepted packed head markers')
    target = 'strata prefill heads: target detached bytes=521472000 source=owned-RAM D2H=521472000 H2D-on-restore=521472000'
    draft = 'strata prefill heads: draft detached bytes=0 retained-map-bytes=425196'
    admission = 'strata native head packed: ready=1 columns=1..4 weight_bytes=521472000 canonical_device_weight_bytes=0 extra_device_weight_bytes=0 permanent_host_weight_bytes=0 default_adopted=0'
    mtp = 'strata mtp: shared packed head tokens=106299 subset_weight_bytes=0 subset_logit_bytes=425196 saved_device_bytes=222802704'
    handoff = '\n'.join([admission, mtp, target, draft, target, draft])
    handoffs(handoff, 2, 2)
    handoffs('\n'.join([target, draft.replace('bytes=0 ', 'bytes=223227900 ')] * 2), 0, 2)
    invalid = [handoff.replace('H2D-on-restore=521472000', 'H2D-on-restore=0', 1),
        handoff.replace('draft detached bytes=0', 'draft detached bytes=223227900', 1),
        handoff.replace('retained-map-bytes=425196', 'retained-map-bytes=0', 1),
        handoff.replace('canonical_device_weight_bytes=0', 'canonical_device_weight_bytes=521472000'),
        handoff.replace('subset_weight_bytes=0', 'subset_weight_bytes=223227900'),
        handoff + '\n' + mtp, handoff.replace(target, '', 1), handoff.replace(draft, '', 1)]
    for bad in invalid:
        try: handoffs(bad, 2, 2)
        except RuntimeError: continue
        raise RuntimeError('head/MTP parser accepted corrupted owner/transfer evidence')
    fields = dict(context='131072', kv='int8', kv_resident='0', expert_slots='1256',
        spec='4', pool_workers='5', pcie_frac='0.00', mtp_max='0')
    check_placement(fields)
    for key, value in [('kv_resident', '131072'), ('kv_resident', '4096'),
                       ('context', '4096'), ('kv', 'fp16'), ('expert_slots', '1255')]:
        try: check_placement(dict(fields, **{key: value}))
        except RuntimeError: continue
        raise RuntimeError('incorrect/streamed model placement accepted')
    return dict(passed=True, positive_arms=2, corrupt_route_variants=len(variants) + 1,
        handoff_positive_arms=2, corrupt_handoff_variants=len(invalid), coalesced_decode_replays=28,
        full_resident_placement_positive=1, incorrect_placement_variants=5, GPU_executed=False)


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('controller', 'models', 'runtime_files', 'generator_runtime', 'commands')}
    out['controller'] = {k: v for k, v in row['controller'].items() if k != 'source_sha256'}
    out['generator_runtime'] = {k: v for k, v in row['generator_runtime'].items() if k != 'inputs'}
    out['requests'] = {k: {n: v for n, v in r.items() if n != 'ids'} for k, r in row['requests'].items()}
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
        requests(Path(row['tokenizer'])) == row['requests'], 'thinking workload generation changed')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    assets(row); guards.no_model_owners([])


def prepare():
    require(not ROOT.exists() and KEY not in build.public(), 'fresh model-entry plan only')
    n = native(); prior = validation.committed(RESULT, (device.KEY,))
    require(prior['passed'] and prior['complete'] and prior['ownership_closed'] and
        all(any(r.get('mode') == arm and r['passed'] for r in prior['results'])
            for arm in ('submission', 'phase', 'reference', 'canonical', 'sole-packed')) and
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
        controller=controller_pin(),
        native_source_commit=n['project']['revision'], native_build=dict(binaries=n['binaries']),
        native_qualification_reuse=dict(native_build_key=build.KEY, device_key=device.KEY,
            device_evidence_commit='5145e8eb7229cdec7b33711f708cc95949d853cd',
            native_source_commit=n['project']['revision'], binaries_unchanged=True,
            actual_compiler_and_fixture_and_recipe_inputs_unchanged=True,
            changed_observers=['tools/sycl/q6_head_model_entry.py'],
            earlier_canonical_model_key='q6_head_model_entry_v4', earlier_model_evidence_commit='bde0c43bd1ff6fe92f98914d83c991abf2a798a6',
            scope='Controller-only new plan;reuse exact original four native ELFs/fixtures/recipe and closed GPU owner proof. No failed model result or speed inherited.'),
        native_record=dict(path=str(build.ROOT / 'record.json'), sha256=sha(build.ROOT / 'record.json')),
        device_evidence_sha256=signature(prior), models=reference['model_assets'],
        models_sha256=signature(reference['model_assets']), model_files=15,
        model_bytes=sum(v['identity']['bytes'] for v in reference['model_assets'].values()),
        runtime_manifest=prior['runtime_manifest'], runtime_files=json.loads(Path(prior['runtime_manifest']['path']).read_text()),
        probe=prior['native_validation_probe'], generator_runtime=prompts.generator_runtime(),
        tokenizer=reference['configuration']['tokenizer'], requests=requests(Path(reference['configuration']['tokenizer'])),
        environment=env, argv=argv, boot=prior['boot'], driver=prior['driver'],
        budgets=dict(wall=900, cpu=10800, rss_cap=104 << 30, text_cap=64 << 20, file_cap=64 << 20, total_cap=256 << 20),
        plan='Committed parser controls first;fresh head0then2 processes,long>8192/short82/long>8192 thinking-on32output'
             ' requests in each. Context131072,KVresident131072,chunk8192,short-read128,1256cache,MTP4,'
             'no token/coupled graphs. This qualifies full-placement admission,not full-context processing.'
             ' Require actual READY/INFO,immutable environment,pre-main validation and pinned file-backed maps,'
             ' native full-resident INFO kv_resident=0,owned DRM snapshots at READY/after requests,'
             ' complete GEN/drafts,all prompt/C1/C4 sole-packed route joins,positive C4replays,two target-head retirements,'
             ' actual canonical223227900B versus packed0B MTP-subset retirements,immutable map extents and'
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
    require(8192 < row['requests']['long']['prompt_tokens'] <= 32768 and
        row['requests']['short']['prompt_tokens'] == 82, 'known multi-chunk diagnostic geometry')
    assets(row, full=True)
    ROOT.mkdir(mode=0o700)
    for arm in (0, 2):
        path = ROOT / str(arm); path.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (path / name).mkdir(mode=0o700)
    save(row); print('Prepared same-engine head0/2 real-model entry;commit before controls.', flush=True)


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
        for arm in (0, 2):
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
                    row['model_executed'] = True
                    print(f'Head{arm}: {label} request complete.', flush=True)
                result['memory_after_requests'] = memory_snapshot(command['pid'])
                io.send(b'QUIT\n'); io.exit(30)
            e, out, err = scope.run(f'head{arm}', row['argv'], env, P, interaction=interact, **row['budgets'])
            require(owner.closed(e) and e['exit_code'] == 0, 'native model closure failed')
            text = err.read_text(errors='replace')
            result['routes'] = routes(text, arm)
            transfers = handoffs(text, arm, 2)
            require(not re.search(r'UR_RESULT_ERROR_[A-Z_]+', text), 'native UR failure')
            log = ROOT / str(arm) / 'validation.log'
            require(not log.exists() or log.stat().st_size == 0, 'native parameter-validation error')
            result.update(head_transitions=transfers['target'], draft_transitions=transfers['draft'],
                sole_packed_owner=transfers['sole_packed_owner'],
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
    print('Closed real-model head0/2 routes and repeated handoffs;speed/adoption remain unqualified.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['host'], ['run']), 'q6_head_model_entry.py prepare|host|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
