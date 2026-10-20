#!/usr/bin/env python3
"""Main-owned 1256-slot/117K cache-loan admission, with both mixed windows.

This uses the retained, qualified engine unchanged. The short output checks
prefill-to-decode restoration; it does not qualify sustained decode or speed.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
import cache_capacity as cap

q, gen, assets, guards, supervision = cap.q, cap.gen, cap.assets, cap.guards, cap.supervision
P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'cache-long-check-v1'
RESULT = A / 'cache-long-current.json'
SCOPE = 'cache_1256_long_mixed_admission_scope'
ROWS, SLOTS, CACHE_BYTES = 117488, 1256, 2575206400
WINDOWS = ((0, 81920), (81920, 35568))
TAIL_OFFSET, PRIMARY, TAIL = 1059131904, 335544320, 83886080
LONG_PREFIXES = ('strata prefill: layer-major candidate,', 'strata MoE mixed banks:',
    'strata MoE window configuration:', 'strata MoE accumulator admission:',
    'strata MoE mixed owners:', 'strata MoE cache loan:', 'strata prefill cache binding:')


def records(text, prefix, count):
    found = []
    for match in re.finditer(r'^'+re.escape(prefix)+r': (.+)$', text, re.M):
        fields = match[1].split()
        guards.require(all(t.count('=') == 1 for t in fields), 'malformed '+prefix)
        row = dict(t.split('=', 1) for t in fields)
        guards.require(len(row) == len(fields), 'duplicate '+prefix+' field')
        found.append((match.start(), row))
    guards.require(len(found) == count == text.count(prefix+':'), 'missing/extra '+prefix)
    return found


def integer(row, name):
    value = row.get(name, '')
    guards.require(re.fullmatch(r'\d+', value) is not None and int(value) < 1 << 64,
                   'invalid uint64 '+name)
    return int(value)


def address(row, name):
    value = row.get(name, '')
    guards.require(re.fullmatch(r'0x[0-9a-fA-F]+', value) is not None and
                   0 < int(value, 16) < 1 << 64, 'invalid pointer '+name)
    return int(value, 16)


def loan_cycle(text):
    audit = records(text, 'strata loan audit', 4)
    pub = records(text, 'strata cache loan', 2)
    layout, lend, payload, refill = [r for _, r in audit]
    masked, ready = [r for _, r in pub]
    guards.require([r['phase'] for _, r in audit] == ['layout', 'lend', 'payload', 'refill'] and
        [r['phase'] for _, r in pub] == ['masked', 'ready'] and
        audit[0][0] < audit[1][0] < pub[0][0] < audit[2][0] < audit[3][0] < pub[1][0],
        'loan/refill publication order differs')
    chunk, first, capacity, need, lent = [integer(layout, k) for k in ('chunk', 'first', 'bytes', 'need', 'lent')]
    guards.require(256 <= chunk <= 8192 and 128 <= first < SLOTS and
        integer(layout, 'slots') == SLOTS and lent == SLOTS-first and capacity >= need > 0 and
        layout['mask_match'] == '1' and integer(payload, 'rows') == SLOTS and
        integer(payload, 'bytes') == CACHE_BYTES > capacity and integer(payload, 'lent') == lent and
        payload['same_slots'] == payload['payload_match'] == '1', 'loan/payload extent differs')
    generation = integer(masked, 'generation')
    guards.require(generation > 0 and masked['decode_ready'] == '0' and ready['decode_ready'] == '1' and
        all(integer(r, 'generation') == generation and
            tuple(integer(r, k) for k in ('chunk', 'first', 'lent')) == (chunk, first, lent)
            for r in (masked, ready)), 'loan generation/ready identity differs')
    for table in (lend, refill):
        guards.require(table['table_match'] == '1' and integer(table, 'table_entries') == 24576 and
            0 <= integer(table, 'free_bytes') <= integer(table, 'total_bytes') and
            integer(table, 'total_bytes') > 0, 'residency table or VRAM observation differs')
    return dict(generation=generation, chunk=chunk, first=first, lent=lent,
        capacity=capacity, need=need, min_observed_free_bytes=min(integer(r, 'free_bytes') for r in (lend, refill))), pub


def mixed_evidence(log, sample):
    headers = cap.matches(r'strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*', log, 2)
    guards.require(tuple(map(int, headers[-1].groups())) ==
        (ROWS+1, 0, ROWS+1, sample['generated_tokens']) and
        sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == ROWS+1 and
        0 < int(headers[0][1]) < 32768 and int(headers[0][2]) == 0 and
        int(headers[0][1]) == int(headers[0][3]) and 1 <= int(headers[0][4]) <= 24,
        'measured request/native header identity differs')
    segments = [log[:headers[0].start()], log[headers[0].end():headers[1].start()]]
    guards.require(not any(prefix in log[headers[-1].end():] for prefix in
        (*LONG_PREFIXES, 'strata loan audit:', 'strata cache loan:', 'strata prefill dense:')),
        'proof records outside completed requests')
    guards.require(not any(prefix in segments[0] for prefix in LONG_PREFIXES),
        'long-window proof leaked into short warmup/preamble')
    cycles, publications = [], []
    for text in segments:
        cycle, pub = loan_cycle(text)
        cycles.append(cycle); publications.append(pub)
    guards.require(0 < cycles[0]['generation'] < cycles[1]['generation'] and
        cycles[1]['chunk'] == 8192, 'request generations/chunk differ')
    text, cycle, pub = segments[-1], cycles[-1], publications[-1]
    for prefix in LONG_PREFIXES:
        count=192 if prefix == 'strata MoE mixed owners:' else 96 if prefix == 'strata MoE cache loan:' else 1
        guards.require(text.count(prefix) == count, 'extra/malformed long-request proof: '+prefix)
    placement, = cap.matches(r'strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) residual_host=0 residual_staging=0 layer_cache=1022361600; extra allocations, GPU headroom=536870912, RAM headroom=8589934592', text, 1)
    guards.require(tuple(map(int, placement.groups())) == (ROWS, ROWS*20480, ROWS, ROWS*20480),
        '117K all-GPU BF16 residual admission differs')
    configuration, = cap.matches(r'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=81920 mixed_fp16_bytes=0', text, 1)
    guards.require('effective_mode=2' not in text, 'fallback route is not admitted')
    banks, = records(text, 'strata MoE mixed banks', 1)
    guards.require(banks[1] == dict(storage='scratch', primary_bytes=str(PRIMARY), tail_bytes=str(TAIL),
        tail_offset=str(TAIL_OFFSET), extra_allocation_bytes='0', phase='FFN_only_after_attention_drain'),
        'mixed tail backing differs')
    admission, = records(text, 'strata MoE accumulator admission', 1)
    admitted=admission[1]
    guards.require(tuple(integer(admitted,k) for k in ('requested_mode','rows','gpu_prefix_rows','mixed_fp16_bytes')) ==
        (10,81920,ROWS,PRIMARY+TAIL) and integer(admitted,'shared_arena_bytes') >= TAIL_OFFSET+TAIL and
        integer(admitted,'free_gpu_bytes') >= integer(admitted,'fixed_including_headroom_bytes') >=
        ROWS*20480+1022361600+536870912, 'actual accumulator allocation budget differs')
    dense = records(text, 'strata prefill dense', 3)
    guards.require([(r['phase'], r['residency'], integer(r, 'activations')) for _, r in dense] ==
        [('retired', 'Full', 0), ('bank', 'Bank', 0), ('restored', 'Full', 48)], 'dense restoration differs')
    sizes = tuple(integer(dense[0][1], k) for k in ('full_bytes', 'bank_bytes', 'snapshot_bytes'))
    guards.require(sizes[0] == sizes[2] > sizes[1] > 0 and all(
        tuple(integer(r, k) for k in ('full_bytes', 'bank_bytes', 'snapshot_bytes')) == sizes and
        r['overlay_rows'] == r['session_recapture'] == '0' and r['verifier'] == '1' and
        0 <= integer(r, 'free_bytes') <= integer(r, 'total_bytes') for _, r in dense), 'dense sizes/owners differ')
    binding, = records(text, 'strata prefill cache binding', 1)
    guards.require(binding[1] == dict(generation=str(cycle['generation']), parent_match='1', execution_distinct='1') and
        pub[0][0] < dense[0][0] < dense[1][0] < admission[0] < banks[0] < configuration.start() <
        binding[0] < placement.start() < dense[2][0] < pub[1][0],
        'dense/cache binding escaped loan')
    owners = records(text, 'strata MoE mixed owners', 192)
    loans = records(text, 'strata MoE cache loan', 96)
    previous, epoch, used, pointers = placement.end(), None, None, None
    for i, (layer, first, rows) in enumerate((l, f, n) for l in range(48) for f, n in WINDOWS):
        (ep, entry), (dp, done), (lp, loan) = owners[2*i], owners[2*i+1], loans[i]
        identity = (layer, first, rows)
        guards.require(entry['phase'] == 'entry' and done['phase'] == 'retired' and
            all(tuple(integer(r, k) for k in ('layer', 'first', 'rows')) == identity for r in (entry, done, loan)) and
            previous < ep < lp < dp < dense[2][0] and
            entry['direct_owners'] == '0' and entry['previous_attention'] == 'checked_drain' and
            done['route_status'] == '0' and done['shared_HC'] == 'checked_drain' and done['callbacks'] == 'after_return',
            'missing, reordered or incorrectly retired FFN window')
        primary, tail = min(rows, 65536)*5120, max(0, rows-65536)*5120
        guards.require(tuple(integer(entry, k) for k in ('primary_bytes', 'tail_offset', 'tail_bytes', 'residual_bytes')) ==
            (primary, TAIL_OFFSET, tail, rows*20480) and
            integer(entry,'region_bytes') == integer(admitted,'shared_arena_bytes') >= TAIL_OFFSET+TAIL,
            'mixed window byte extent differs')
        addresses = tuple(address(entry, k) for k in ('primary', 'region', 'residual', 'bo', 'queue'))
        stable = (addresses[0], addresses[1], addresses[2]-first*20480, addresses[3], addresses[4])
        if pointers is None: pointers = stable
        guards.require(stable == pointers and all(x % 256 == 0 for x in addresses[:4]), 'mixed owners moved or are misaligned')
        # The tail is inside the allocated region, after the live plan prefix.
        # Comparing it against the full allocation would reject intentional aliasing.
        spans = [(addresses[0], primary), (addresses[1], TAIL_OFFSET),
                 (addresses[2], rows*20480), (addresses[3], 8192*5120)]
        if tail:
            guards.require(address(entry, 'tail') == addresses[1]+TAIL_OFFSET, 'tail does not follow live arena planes')
            spans.append((address(entry, 'tail'), tail))
        else:
            guards.require(entry['tail'] == '(nil)', 'unexpected second-window tail')
        guards.require(all(a+n < 1 << 64 for a, n in spans) and all(
            a+n <= b or b+m <= a for j, (a, n) in enumerate(spans) for b, m in spans[j+1:]),
            'mixed typed spans overlap or overflow')
        if epoch is None: epoch, used = integer(loan, 'layout_generation'), integer(loan, 'used')
        guards.require(epoch > 0 and 0 < used <= cycle['need'] <= cycle['capacity'] and
            tuple(integer(loan, k) for k in ('generation', 'layout_generation', 'used', 'capacity')) ==
            (cycle['generation'], epoch, used, cycle['capacity']), 'sealed loan extent/generation differs')
        previous = dp
    return dict(completed_requests=2, exact_payload_and_map_cycles=2,
        loan_generations=[r['generation'] for r in cycles], slots=SLOTS,
        long_loan={k:v for k,v in cycle.items() if k != 'generation'},
        actual_allocation_budget={k:integer(admitted,k) for k in
            ('free_gpu_bytes','fixed_including_headroom_bytes','shared_arena_bytes')},
        min_observed_free_bytes=min(r['min_observed_free_bytes'] for r in cycles),
        prefill_rows=ROWS, residual_gpu_bytes=ROWS*20480, windows_per_layer=[list(w) for w in WINDOWS],
        borrowed_ffn_entries=96, checked_retirements=96, dense_layers=48,
        secondary_bank_windows=48, execution_cache_parent_checked=True, clean_speed_qualified=False)


def official_evidence(root, result):
    guards.require(result['params']['mode'] == 'prefill-only' and result['params']['runs'] == 1 and
        result['params']['think'] is True and result['params']['prefill_tokens'] == 98304 and
        result['complete'] and not result['results']['failures'], 'official prefill suite incomplete')
    cohort = result['results']['prefill_only']
    guards.require(cohort['complete'] and cohort['eligible_runs'] == 1 and len(cohort['requests']) == 1 and
        cohort['max_tokens'] == 24 and cohort['thinking_requested'] is True, 'wrong prefill-only scope')
    sample = cohort['requests'][0]
    guards.require(sample['run'] == 0 and all(sample.get(k) is True for k in
        ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed')) and
        sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == ROWS+1 and
        sample['actual_reused_tokens'] == 0 and 1 <= sample['generated_tokens'] <= 24 and
        sample['prompt_tps_source'] == 'engine', 'ineligible117K sample')
    journal, _ = gen.journal_evidence(root)
    path, = (root/'bench').glob('*/requests.jsonl')
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    guards.require(journal['valid'] and journal['complete'] and len(entries) == 5 and
        entries[0]['kind'] == 'identity' and entries[0]['phase'] == 'suite' and
        all(r['kind'] != 'identity' for r in entries[1:]), 'incomplete/extra/misplaced journal identity')
    identity, comparison = entries[0], result['comparison_identity']
    from strata_bench import client_fingerprint
    workload=dict(workload_id='native_gdn_20261018',nonce='native_gdn_20261018',nonce_policy='fixed-id')
    guards.require(identity['mode'] == comparison['mode'] == 'prefill-only' and
        identity['requested_runs'] == 1 and identity['thinking'] is comparison['thinking'] is True and
        identity['context'] == comparison['context'] == result['machine']['context'] == 131072 and
        identity['source'] == comparison['client'] == client_fingerprint() and
        identity['workload'] == comparison['workload'] == result['params']['workload'] == workload and
        identity['prefill_plan'] == comparison['prefill_plan'] and
        all(cohort[k] == value for k,value in identity['prefill_plan'].items()) and
        all(identity[k] == comparison[k] == result['machine'][k] for k in ('model','quant')),
        'journal model/client/workload identity differs')
    events=entries[1:]
    hashes = []
    for i, (phase, run) in enumerate((('warm-up/chat', None), ('prefill-only', 0))):
        start, end = events[2*i:2*i+2]
        guards.require(start['kind'] == 'start' and end['kind'] == 'result' and
            all(r['phase'] == phase and r.get('run') == run and r.get('turn') is None for r in (start, end)) and
            start['max_tokens'] == 24 and start['thinking'] is True and end['sample']['ok'] is True and
            end['request_index'] == start['index'] and end['sample']['request_sha256'] == start['request_sha256'] and
            re.fullmatch('[0-9a-f]{64}', start['request_sha256']) is not None,
            'warmup/final journal identity differs')
        hashes.append(start['request_sha256'])
    guards.require(events[-1]['sample'] == {k:v for k,v in sample.items() if k != 'run'}, 'journal/suite sample differs')
    return sample, dict(successful_warmups=1, successful_long_requests=1, request_sha256=hashes,
        journal_sha256=gen.common.digest(path), state_sha256=gen.common.digest(path.with_name('journal-state.json')))


def reviewed():
    prior = cap.source_review()
    row = gen.committed(cap.borrowed.RESEARCH)[SCOPE]
    guards.require(row.get('worker_closed') is True and row.get('bounded_117K_check_accepted') is True,
        'curated source review required before117K admission')
    for name, expected in row['frozen_inputs'].items():
        guards.require(gen.common.digest(P/name) == expected and
            cap.hashlib.sha256(gen.git('show', row['source_commit']+':'+name)).hexdigest() == expected,
            'reviewed long-check source changed: '+name)
    return dict(prior=prior, source_commit=row['source_commit'], report_sha256=row['report_sha256'],
        native_review_sha256=row['native_review_sha256'], frozen_inputs=row['frozen_inputs'])


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(), 'fresh root/result required; never retry')
    full, count = q.current_build()
    before, seed = gen.committed(cap.RESULT), gen.committed(assets.RESULT)
    guards.require(before['passed'] and before['complete'] and before['ownership_closed'] and
        before['long_capacity_evidence']['checked_retirements'] == 48 and
        before['binary'] == full['binary'] and not before['fresh_faults'], 'matching32K admission required')
    review, source = reviewed(), gen.git('rev-parse', 'HEAD').decode().strip()
    pins, settings = gen.benchmark_pins(source), gen.committed(gen.SETTINGS)
    config = copy.deepcopy(seed['config']); flags = config['args']
    for name, value in (('--expert-cache', str(SLOTS)), ('--pool-workers', '5'), ('--pcie-frac', '0'),
                        ('--expert-profile', str(ROOT/'expert-profile.bin'))):
        flags[flags.index(name)+1] = value
    guards.require(flags.count('--no-prefill-borrow') == 1 and
        all(flags[flags.index(k)+1] == v for k,v in (('--max-context','131072'), ('--kv','int8'),
            ('--kv-resident','131072'), ('--prefill','8192'), ('--adapt-swaps','0'))), 'seed placement changed')
    flags.remove('--no-prefill-borrow'); config.update(log=str(ROOT/'engine.stderr'), env={})
    old_root = seed['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old_root,str(ROOT)) for k,v in seed['environment'].items()}
    guards.require(not any(k == 'STRATA_TRACE' or k == 'UR_ENABLE_LAYERS' or k.startswith('UR_LOG_') for k in env), 'seed has observers')
    env.update(STRATA_SYCL_CACHE_SCRATCH_LEASE='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1',
        STRATA_SYCL_VERIFY_ARENA_COPY='0', STRATA_SYCL_VERIFY_PCIE_EARLY='0',
        STRATA_SYCL_VERIFY_PCIE_EARLY_AUDIT='0', STRATA_SYCL_QUERY_PROFILE='0',
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', STRATA_TRACE='1')
    env.update({k:v for k,v in settings['environment'].items() if k.startswith('UR_LOG_') and k != 'UR_LOG_TRACING'})
    for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[key] == settings['environment'][key], 'qualified safety setting changed')
    model = assets.model_pins(config, full=True)
    ROOT.mkdir(mode=0o700)
    for name in ('cache','tmp'): (ROOT/name).mkdir()
    shutil.copyfile(P/'data/expert-profile.bin', ROOT/'expert-profile.bin')
    guards.require(gen.common.digest(ROOT/'expert-profile.bin') == seed['profile_sha256'], 'profile changed')
    (ROOT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
    command = ['/bin/sh', str(P/'bench.sh'), '--config', str(ROOT/'config.json'), '--port', str(port),
        '--prefill-only', '--prefill-tokens', '98304', '--think', '--runs', '1', '--workload-id',
        'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(ROOT/'bench')]
    record = dict(mode='cache-capacity-check', diagnostic='1256-slot117K-mixed-tail', source_commit=source,
        build_source_commit=full['source_commit'], binary=full['binary'], original_native_inputs_recovered=count,
        source_build_record=dict(path=str(q.BUILD_RECORD),sha256=gen.common.digest(q.BUILD_RECORD)),
        component_result_sha256=gen.common.digest(q.COMPONENT), source_review=review,
        prerequisite=dict(commit=source,path=str(cap.RESULT.relative_to(P)),sha256=gen.common.digest(cap.RESULT)),
        runtime_asset_delta=seed['runtime_asset_delta'], root=str(ROOT), config=config, environment=env, command=command,
        config_sha256=gen.common.digest(ROOT/'config.json'), profile_sha256=seed['profile_sha256'],
        settings_sha256=gen.common.digest(gen.SETTINGS), model_assets=model, model_assets_fingerprint=assets.fingerprint(model),
        machine_identity=assets.machine_identity(), source_count=len(pins), source_aggregate_sha256=assets.fingerprint(pins),
        tokenizer_assets={str(Path(config['tokenizer'])/n):gen.common.digest(Path(config['tokenizer'])/n)
            for n in ('vocab.json','merges.txt','token_type.json','chat_template.jinja')},
        complete=False, passed=False, active=False, clean_speed_qualified=False, model_quality_qualified=False,
        physical262144_RESTORE_qualified=False,
        gate='One official process, one short thinking warmup and one fresh117489-input request,24 output cap. Exact1256-slot payload/map restoration twice; all117488 BF16 residual rows on GPU; actual mode10,81920+35568 windows in each48 layers,96 checked retirements,48 secondary-bank windows, bank-to-Full restoration and actual decode. Normal closure, unchanged inputs/runtime/boot, no fresh fault. First refusal/error/budget closes the run; no retries.',
        limits='Parameter validation, stage trace and cache readback are diagnostic overhead. No clean speed, sustained decode, strong/weak/agent adoption, broad quality or physical262144/session/RESTORE qualification.',
        consumer='Admit or reject retained old-owned128/current-borrowed1256 long-input comparison; do not rebuild engine before that comparison closes.',
        retention='Commit compact sample, extent/conservation/fault results and decision; retire successful raw logs/JIT/transaction files after verification.',
        budgets=dict(wall=1800,cpu=36000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20))
    q.runtime_assets_unchanged(record)
    (ROOT/'plan.json').write_text(json.dumps(record,indent=2)+'\n')
    RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared1256-slot117K admission; commit exact plan before run.')


def run():
    record = gen.committed(RESULT)
    guards.require(record == json.loads((ROOT/'plan.json').read_text()) and not record['complete'] and
        not (ROOT/'record.json').exists(), 'fresh committed plan required; never retry')
    full, count = q.current_build()
    guards.require(full['binary'] == record['binary'] and count == record['original_native_inputs_recovered'] and
        gen.common.digest(q.BUILD_RECORD) == record['source_build_record']['sha256'] and
        gen.common.digest(q.COMPONENT) == record['component_result_sha256'] and reviewed() == record['source_review'],
        'build/component/review changed')
    pins = gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and assets.fingerprint(pins) == record['source_aggregate_sha256'], 'benchmark inputs changed')
    for path, expected in [(gen.SETTINGS,record['settings_sha256']), (ROOT/'config.json',record['config_sha256']),
            (ROOT/'expert-profile.bin',record['profile_sha256']), *record['tokenizer_assets'].items()]:
        guards.require(gen.common.digest(path) == expected, 'input changed: '+str(path))
    q.runtime_assets_unchanged(record)
    guards.require(assets.model_pins(record['config'],record['model_assets']) == record['model_assets'] and
        assets.machine_identity() == record['machine_identity'], 'model/platform changed')
    record['plan_commit'] = gen.git('rev-parse','HEAD').decode().strip()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
    cursor = next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (ROOT/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    owner = supervision.Owner(ROOT/'owned',persist)
    try:
        entry, _, _ = owner.run('bench',record['command'],record['environment'],P,**record['budgets'])
        record.update(complete=supervision.closed(entry),exit_code=entry.get('exit_code'))
        guards.require(record['complete'] and entry['exit_code'] == 0, 'owned official benchmark failed')
        path, = (ROOT/'bench').glob('*/results.json')
        sample, journal_proof = official_evidence(ROOT,json.loads(path.read_text()))
        record['requests'], record['request_journal'] = [sample], journal_proof
        native = json.loads((ROOT/'engine-exit.json').read_text())
        guards.require(native['exit_code'] == 0 and native['boot_id'] == boot and native['binary'] == record['binary'] and
            native['args'] == ['--serve',*record['config']['args']] and 'resource_collector' not in native, 'native launch/closure differs')
        log = (ROOT/'engine.stderr').read_text(errors='replace'); assets.verify_pool(log,5)
        record['long_capacity_evidence'] = mixed_evidence(log,sample)
        guards.require('strata decode cpu: ' not in log and 'strata decode timing: ' not in log, 'unexpected observer')
        q.current_build(); q.runtime_assets_unchanged(record)
        guards.require(gen.benchmark_pins(record['source_commit']) == pins and assets.machine_identity() == record['machine_identity'] and
            assets.model_pins(record['config'],record['model_assets'],full=True) == record['model_assets'], 'inputs changed during run')
        record.update(passed=True,native_identity=native['native'],full_model_hashes_before_and_after_match=True)
    except BaseException as error:
        record['error'] = str(error)
        paths = list((ROOT/'bench').glob('*/results.json'))
        if len(paths) == 1:
            record['requests'] = json.loads(paths[0].read_text()).get('results',{}).get('prefill_only',{}).get('requests',[])
        for path in (ROOT/'engine.stderr',ROOT/'owned/bench.stderr'):
            if path.exists(): record['error_excerpt'] = path.read_text(errors='replace')[-4096:]
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
        record['fresh_faults'] = [x for x in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
        record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        try: record['driver_after'] = guards.driver_guard()
        except Exception as error: record.update(driver_guard_error=str(error),passed=False)
        record['passed'] = bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])
        persist(); RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('complete','passed','ownership_closed','error')}))
    return 0 if record['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run')); args = parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([]); guards.driver_guard()
        return prepare() if args.action == 'prepare' else run()


if __name__ == '__main__': sys.exit(main())
