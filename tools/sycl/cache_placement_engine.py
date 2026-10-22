#!/usr/bin/env python3
"""Main-owned, same-native cache-placement discriminator; no adoption or rebuild."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import cache_placement as placement
import cpu_work_census as census
import generation_engine as gen
import resource_engine as resources
from check_expert_bank import check_packed_cases

assets, restored, comparison = resources.modules()
guards, supervision = resources.guards, resources.supervision
P, B, A = gen.P, gen.B, gen.A
ROOT, RESULT = B/'cache-placement-v2', A/'cache-placement-current.json'
WORKLOADS = ('regular-code',)
PREVIOUS_COMMIT = 'eb20952da4bde54018d1ec7e373ba7e3d390ea44'
PREVIOUS_SHA256 = '0d41acf9ec37d4ccb858c03b56deff538ec407b00c704cc07003022f00857e74'


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def prospective():
    return dict(order=['traced-packed-smoke', 'per-layer', 'global-ranked'],
        question='Does existing global profile placement with a smaller protected byte prefix reduce actual target-verifier CPU logical weight visits at no larger packed allocation?',
        source_audit='Existing generate.cpp global prefix uses align256(blob_bytes) and open_sized;ExpertCache::admit assigns monotonic slots,with adapt-swaps0 no heterogeneous replacement. All48 layers occur in this prefix. device_slot/Drive/VerifyHits carry actual slot_offsets;Verifier::init checks owner/base/offsets/slot count. prepare_scratch_loan protects128 slots and uses actual offsets;payload/map snapshots and refill must match before decode. No kernel,collective,queue,graph implementation or environment mutation is introduced;component qualification covers sized slots and scratch owner lifetime.',
        constants='Original6e40 engine and12 fixtures;unchanged model/runtime/precision;131072 residentINT8 KV,8192 prefill,spec4,5workers,pcie0,adapt0,caches0. Both arms use a deterministically reordered profile: stable ascending blob size only inside original admitted prefix. Prove original per-layer slot map and global membership unchanged. Only remove per-layer admission and replace cache1256 with the offline derived global knob between arms. Immutable native environment;existing work/timing/IQ2/loan diagnostics;no passive collector.',
        admission='Fresh v2 roots;original v1 admission failure remains closed/archived and its unused raw root retired. Commit controller,host tests and generated profile recipe/hash/plan before execution. Recover9853 native inputs/recipe and exact13 artifacts,full model hashes before cohort and after each model process,stat identities at other boundaries. Offline candidate maximum loan must cover v1 actual8192 need2319958528 bytes. Small traced packed-bank fixture must pass first. Before long request verify4 official warmups,native PID/start/SID/executable/environment,exact cache slot count/policy and no native error. Bind all5 cache loan/refill cycles to exact offsets and membership;require unchanged8192 chunk,GPU BF16 mode10/private-bank route and48 dense layers restored.',
        workload='Pinned official bench.sh,thinking-enabled regular-code subset,runs1,four warmups,>=32768 fresh input,>=1024 generated and substantive answer. Independent baseline then candidate processes. This diagnostic is not clean speed or repeated adoption evidence.',
        decision='Advance only with both complete/closed/fault-free processes and same request/read extent,strong observed MTP>=0.60. Candidate logical CPU job-blob bytes per actual verifier row must be <=0.95 of baseline,VRAM entry fraction must increase,and exact decode_ms/generated_tokens must not increase. Otherwise reject this screen;no retry or posthoc threshold change. Generated text need not match;changed work is recorded. A pass permits only a separately committed repeated clean comparison,not adoption.',
        retention='Keep2 individual long requests,compact logical work/NT bins,loan extent/restoration summaries,identities,fault/closure and decision. Retire successful raw API/polling/model logs andJIT after evidence replay. No tensors or full retained timeline.',
        limits='Logical visits are not physical RAM/PCIe traffic or hardware saturation. One ordered observed pair does not prove clean speed,quality,agent-session,prefill preservation,physical262144/session/RESTORE or70tok/s.')


def config(setup, root, candidate, geometry):
    value = restored.recipe.config(setup, root)
    args = value['args']
    guards.require(args.count('--expert-cache') == args.count('--expert-cache-per-layer') == 1 and
        args[args.index('--expert-cache')+1] == '1256' and
        args[args.index('--adapt-swaps')+1] == args[args.index('--pcie-frac')+1] == '0',
        'fixed original cache/adaptive/PCIe policy required')
    if candidate:
        args.remove('--expert-cache-per-layer')
        args[args.index('--expert-cache')+1] = str(geometry['candidate']['expert_cache_argument'])
    return value


def pins(revision):
    values = gen.benchmark_pins(revision)
    for name in ('data/expert-profile.bin', 'tests/core/test_cache_placement.py',
                 'tests/core/test_cache_placement_engine.py'):
        raw = gen.git('show', revision+':'+name)
        guards.require((P/name).read_bytes() == raw, 'uncommitted placement input: '+name)
        values[name] = hashlib.sha256(raw).hexdigest()
    return values


def current(record, full_models=False):
    full, proof, setup = resources.qualification()
    guards.require(record['binary'] == full['binary'] and record['qualification'] == proof and
        record['plan'] == prospective(), 'native/plan qualification changed')
    actual = pins(record['source_commit'])
    guards.require(len(actual) == record['source_count'] and
        assets.fingerprint(actual) == record['source_fingerprint'] and
        gen.common.digest(gen.SETTINGS) == record['settings_sha256'], 'source/settings changed')
    manifest = Path(setup['config']['args'][setup['config']['args'].index('--pack')+1])/'native_experts.txt'
    guards.require(placement.plan(P/'data/expert-profile.bin', manifest, protect_prefix=True) == record['geometry'] and
        record['geometry']['candidate']['maximum_loan_bytes'] >= record['required_prefill_bytes'] == 2319958528,
        'offline geometry/profile/manifest changed')
    for path, digest in record['generated_inputs'].items():
        guards.require(gen.common.digest(path) == digest, 'generated placement input changed: '+path)
    for i, row in enumerate(record['process_plans']):
        root = Path(row['root'])
        guards.require(row['config'] == config(setup, root, i == 1, record['geometry']) and
            row['environment'] == restored.environment(setup, root, '0'), 'configuration/environment changed')
    guards.require(assets.machine_identity() == record['machine_identity'] and
        assets.model_pins(setup['config'], record['model_assets'], full=full_models) == record['model_assets'],
        'model/platform changed')
    gen.runtime_assets_unchanged(record)
    return full, setup


def prepare():
    guards.require(not ROOT.exists(), 'single-use v2 placement root;never retry')
    previous = gen.committed(RESULT)
    guards.require(previous['schema'] == 'strata-cache-placement-closed-v1' and previous['stage'] == 'closed' and
        previous['archive']['commit'] == PREVIOUS_COMMIT and previous['archive']['sha256'] == PREVIOUS_SHA256 and
        previous['candidate_admission']['native_launched'] is False and previous['artifact_retirement']['complete'] and
        previous['ownership_closed'] and not previous['active'] and not (B/'cache-placement-v1').exists(),
        'closed archived v1 admission failure and raw retirement required;no retry')
    raw = gen.git('show', PREVIOUS_COMMIT+':'+str(RESULT.relative_to(P)))
    guards.require(hashlib.sha256(raw).hexdigest() == PREVIOUS_SHA256, 'original v1 evidence changed')
    full, proof, setup = resources.qualification()
    source = gen.git('rev-parse', 'HEAD').decode().strip()
    sources = pins(source)
    manifest = Path(setup['config']['args'][setup['config']['args'].index('--pack')+1])/'native_experts.txt'
    geometry = placement.plan(P/'data/expert-profile.bin', manifest, protect_prefix=True)
    required = previous['candidate_admission']['actual_required_prefill_bytes']
    guards.require(geometry['candidate']['maximum_loan_bytes'] >= required == 2319958528, 'protected prefix still cannot preserve8192')
    sizes = [row['blob_bytes'] for row in placement.read_manifest(manifest)]
    pairs = placement.reorder_cached_prefix(placement.read_profile(P/'data/expert-profile.bin'),sizes,
        geometry['profile_generation']['cached_prefix_slots'])
    profile_bytes = placement.encode_profile(pairs)
    guards.require(hashlib.sha256(profile_bytes).hexdigest() == geometry['profile_generation']['sha256'], 'generated profile differs')
    models = assets.model_pins(setup['config'], setup['model_assets'], full=True)
    ROOT.mkdir(mode=0o700)
    generated, plans = {}, []
    for i, name in enumerate(('per-layer', 'global-ranked')):
        root = ROOT/name; root.mkdir(mode=0o700)
        for part in ('tmp', 'cache'): (root/part).mkdir()
        (root/'expert-profile.bin').write_bytes(profile_bytes)
        cfg, env = config(setup, root, i == 1, geometry), restored.environment(setup, root, '0')
        write(root/'config.json', cfg)
        write(root/'plan.json', dict(binary=full['binary'], native_validation_probe=None))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        plans.append(dict(root=str(root), config=cfg, environment=env,
            command=['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'), '--port', str(port),
                '--long-generation-only', '--long-generation-workload', 'regular-code', '--think', '--runs', '1',
                '--profile-request-gates', '--workload-id', comparison.WORKLOAD_ID, '--fail-fast-engine-error',
                '--ready-timeout', '600', '--out', str(root/'bench')]))
        generated.update({str(root/n): gen.common.digest(root/n) for n in ('config.json','plan.json','expert-profile.bin')})
    smoke = ROOT/'smoke'; smoke.mkdir(mode=0o700)
    for part in ('tmp', 'cache'): (smoke/part).mkdir()
    env = dict(json.loads(gen.SETTINGS.read_text())['environment'], TMPDIR=str(smoke/'tmp'), XDG_CACHE_HOME=str(smoke/'cache'))
    record = dict(schema='strata-cache-placement-model-v2', mode='engine-compare', stage='prepared',
        previous_screen=previous['archive'], required_prefill_bytes=required,
        source_commit=source, binary=full['binary'], build_source_commit=full['source_commit'], qualification=proof,
        plan=prospective(), geometry=geometry, model_assets=models, process_plans=plans, smoke_environment=env,
        source_count=len(sources), source_fingerprint=assets.fingerprint(sources),
        settings_sha256=gen.common.digest(gen.SETTINGS), machine_identity=assets.machine_identity(),
        runtime_asset_delta=setup['runtime_asset_delta'], generated_inputs=generated,
        native_validation_probe=None, smoke=None, processes=[], active=False, ownership_closed=True,
        clean_speed_qualified=False, adoption_qualified=False, complete=False, passed=False)
    current(record)
    write(ROOT/'frozen-plan.json', record); write(RESULT, record)
    print('Prepared immutable placement diagnostic;commit plan before smoke/model execution.', flush=True)


def startup(text, layout, candidate):
    matches = re.findall(r'^strata generate: expert cache (\d+) slots, [^\n]+$', text, re.M)
    guards.require(matches == [str(layout['slots'])], 'actual startup cache admission differs')
    guards.require(('R4.2g PER-LAYER:' in text) is (not candidate) and
        ('PROFILE, ranked by routing frequency, no eviction.' in text) is candidate,
        'actual startup placement policy differs')
    guards.require(f'pre-filled {layout["slots"]} of {layout["slots"]} slots from the profile; slot 0 verified' in text,
        'complete verified cache fill required')
    assets.verify_pool(text, 5)


def loan_cycle(text, selected, sizes, dense_required=False):
    """Check exact candidate slot offsets, not the old1256-slot parser constants."""
    admission = comparison.bank.admission
    audits, publications = admission.records(text, 'strata loan audit', 4), admission.records(text, 'strata cache loan', 2)
    layout, lend, payload, refill = [r for _, r in audits]
    masked, ready = [r for _, r in publications]
    n, integer = len(selected), admission.integer
    extents = [placement.aligned(sizes[l]) for l, _ in selected]
    guards.require([r['phase'] for _, r in audits] == ['layout','lend','payload','refill'] and
        [r['phase'] for _, r in publications] == ['masked','ready'] and
        audits[0][0] < audits[1][0] < publications[0][0] < audits[2][0] < audits[3][0] < publications[1][0],
        'loan/refill publication order differs')
    chunk, first, capacity, need, lent = [integer(layout, k) for k in ('chunk','first','bytes','need','lent')]
    guards.require(256 <= chunk <= 8192 and 128 <= first < n and integer(layout,'slots') == n and
        lent == n-first and capacity == sum(extents[first:]) >= need > 0 and layout['mask_match'] == '1' and
        integer(payload,'rows') == n and integer(payload,'bytes') == sum(extents) > capacity and
        integer(payload,'lent') == lent and payload['same_slots'] == payload['payload_match'] == '1',
        'actual packed loan/payload extent differs')
    generation = integer(masked,'generation')
    guards.require(generation > 0 and masked['decode_ready'] == '0' and ready['decode_ready'] == '1' and
        all(integer(r,'generation') == generation and
            tuple(integer(r,k) for k in ('chunk','first','lent')) == (chunk,first,lent) for r in (masked,ready)),
        'loan generation/ready publication differs')
    for row in (lend,refill):
        guards.require(row['table_match'] == '1' and integer(row,'table_entries') == 24576 and
            0 <= integer(row,'free_bytes') <= integer(row,'total_bytes') and integer(row,'total_bytes') > 0,
            'residency map or VRAM observation differs')
    if dense_required:
        dense = admission.records(text, 'strata prefill dense', 3)
        guards.require([(r['phase'],r['residency'],r['activations']) for _,r in dense] ==
            [('retired','Full','0'),('bank','Bank','0'),('restored','Full','48')] and
            publications[0][0] < dense[0][0] < dense[1][0] < dense[2][0] < publications[1][0],
            'dense retirement/restoration escaped loan')
    return dict(generation=generation, chunk=chunk, first=first, lent=lent, capacity=capacity, need=need,
        cache_bytes=sum(extents), payload_and_map_match=True, restored_dense_layers=48 if dense_required else None)


def work_evidence(text, rows, plan, root, layout, candidate):
    args = plan['config']['args']; manifest = Path(args[args.index('--pack')+1])/'native_experts.txt'
    formats = census.manifest_formats(manifest); sizes = [r[2] for r in formats]
    pairs = placement.read_profile(root/'expert-profile.bin')
    selected = (placement.ranked(pairs,sizes,layout['nominal_cap_bytes']) if candidate
                else placement.per_layer(pairs,layout['slots']))
    timing = gen.decode_timing_evidence(text, rows)
    bindings = [dict(prompt_tokens=s['prompt_tokens'],windows=t['windows'],rows=t['work']['target_rows'],
        cpu_jobs=t['work']['cpu_unique_jobs'],cpu_entries=t['work']['cpu_entries'],
        pool_ms=t['ms_per_window']['CPU_pool_run']*t['windows']) for s,t in zip(rows,timing)]
    work = census.extract(text, bindings, formats, placement.membership(selected))
    paired = gen.iq2_pair_evidence(text, rows, 1, restored.recipe.component.base.machine()['physical_cpus'])
    for audit, observed in zip(paired,work):
        guards.require(audit['row_ops'] == 640*observed['cpu_entries'] and
            audit['nt1_rows'] == 640*sum(b['jobs_NT'][0] for b in observed['format_bins'] if b['gu'] == 22),
            'IQ2 callback/census conservation differs')
    return work, paired, timing, selected, sizes


def evidence(root, record, plan, setup, boot, entry, candidate):
    guards.require(supervision.closed(entry) and entry.get('exit_code') == 0, 'official benchmark owner failed')
    path, = (root/'bench').glob('*/results.json')
    samples, warmups, events = comparison.official(root, json.loads(path.read_text()), WORKLOADS)
    sample, = samples
    guards.require(sample['acceptance_rate'] >= .60 and sample['request_sha256'] == setup['request_sha256'],
        'fixed strong-MTP request required')
    native = gen.native_end(root, record, plan, boot, entry['pid'])
    text = (root/'engine.stderr').read_text(errors='replace')
    layout = record['geometry']['candidate' if candidate else 'baseline']
    startup(text, layout, candidate)
    rows = [e['sample'] for e in events if e['kind'] == 'result']
    work, paired, timing, selected, sizes = work_evidence(text,rows,plan,root,layout,candidate)
    headers = list(re.finditer(r'^strata serve: prompt .*$', text, re.M))
    guards.require(len(headers) == len(rows) == 5, 'exact warmup/long request framing required')
    loans, previous = [], 0
    for i, header in enumerate(headers):
        segment = text[previous:header.start()]; previous = header.end()
        loans.append(loan_cycle(segment, selected, sizes, dense_required=i == 4))
        if i == 4:
            cells = sample['read_tokens']-1
            expected = (cells, cells*20480, cells, cells*20480)
            matches = re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) residual_host=0 residual_staging=0 ', segment, re.M)
            guards.require(len(matches) == 1 and tuple(map(int,matches[0])) == expected and
                segment.count(f'requested_mode=10 effective_mode=10 maximum_rows={cells} mixed_fp16_bytes=0') == 1,
                'actual GPU BF16/mode10 route differs')
            bank = comparison.bank.geometry(segment, 1)
    guards.require(all(a['generation'] < b['generation'] for a,b in zip(loans,loans[1:])) and
        loans[-1]['chunk'] == 8192, 'loan generation reused or long chunk changed')
    return dict(request=sample,warmups=warmups,native_exit_code=native['exit_code'],
        native_identity=native['native'],census=work[-1],iq2_dispatch=paired[-1],decode_timing=timing[-1],
        loans=loans,private_bank=bank,
        evidence_sha256={p.name:gen.common.digest(p) for p in (path,root/'engine.stderr',root/'engine-start.json',root/'engine-exit.json')})


def decide(rows):
    guards.require(len(rows) == 2 and all(r['passed'] and r['complete'] for r in rows), 'complete pair required')
    a,b = rows
    if (a['request']['request_sha256'] != b['request']['request_sha256'] or
            a['request']['read_tokens'] != b['request']['read_tokens']):
        return dict(advance_to_new_clean_plan=False,reason='request/read mismatch',adoption_qualified=False,
            clean_speed_qualified=False)
    byte_ratio = (b['census']['logical_job_blob_bytes']/b['census']['rows'])/(a['census']['logical_job_blob_bytes']/a['census']['rows'])
    gpu = [r['census']['vram_entries']/(r['census']['cpu_entries']+r['census']['vram_entries']) for r in rows]
    latency_ratio = (b['request']['decode_ms']/b['request']['generated_tokens'])/(a['request']['decode_ms']/a['request']['generated_tokens'])
    return dict(logical_bytes_per_row_ratio=byte_ratio,vram_entry_fraction=gpu,token_latency_ratio=latency_ratio,
        advance_to_new_clean_plan=byte_ratio <= .95 and gpu[1] > gpu[0] and latency_ratio <= 1,
        adoption_qualified=False,clean_speed_qualified=False)


def run():
    record = gen.committed(RESULT); full,setup = current(record)
    frozen = json.loads((ROOT/'frozen-plan.json').read_text())
    mutable = {'stage','smoke','processes','complete','passed','active','ownership_closed','decision'}
    guards.require({k:v for k,v in record.items() if k not in mutable} ==
        {k:v for k,v in frozen.items() if k not in mutable} and not record['active'] and record['ownership_closed'] and
        record['schema'] == 'strata-cache-placement-model-v2' and record['stage'] in ('prepared','collecting') and
        not record['complete'] and all(r['passed'] for r in record['processes']), 'closed/changed/failed plan')
    smoke = record['smoke'] is None
    i = len(record['processes']); guards.require(smoke or i < 2, 'pair complete')
    if not smoke: guards.require(record['smoke']['passed'], 'successful traced smoke required')
    root = ROOT/'smoke' if smoke else Path(record['process_plans'][i]['root'])
    guards.require(not (root/'record.json').exists() and not (root/'launch.json').exists(), 'existing attempt;inspect owner,never retry')
    row, owner = dict(passed=False,complete=False,active=False,ownership_closed=True), None
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor = next(s.split(': ',1)[1] for s in before.splitlines() if s.startswith('-- cursor: '))
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(root/'record.json', dict(row,commands=owner.commands if owner else []))
    def interact(io, leader):
        for stage in ('begin','end'):
            while True:
                line = io.line(1500)
                if line.startswith('STRATA_PROFILE_GATE '): break
            guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                dict(stage=stage,phase='long-generation/regular-code',run=0), 'request gate differs')
            if stage == 'begin':
                native = json.loads((root/'engine-start.json').read_text())['native']
                identity = supervision.identity(native['pid'])
                guards.require(identity and identity['start_ticks'] == native['start_ticks'] and identity['sid'] == leader['pid'] and
                    Path('/proc',str(native['pid']),'exe').resolve() == Path(record['binary']['path']).resolve(), 'native owner/executable differs')
                actual = dict(x.split(b'=',1) for x in Path('/proc',str(native['pid']),'environ').read_bytes().split(b'\0') if b'=' in x)
                env = record['process_plans'][i]['environment']
                guards.require(all(actual.get(k.encode()) == v.encode() for k,v in env.items()) and
                    b'LD_PRELOAD' not in actual, 'native environment differs')
                path, = (root/'bench').glob('*/requests.jsonl')
                events = [json.loads(x) for x in path.read_text().splitlines()]
                guards.require(len(events) == 10 and events[-1]['kind'] == 'start' and
                    events[-1]['phase'] == 'long-generation/regular-code', 'warmup journal differs')
                gen.warmup_reconciliation(events[:-1],workloads=())
                text = (root/'engine.stderr').read_text(errors='replace')
                previous_rows = [e['sample'] for e in events[:-1] if e['kind'] == 'result']
                gen.request_evidence(text,previous_rows)
                layout = record['geometry']['candidate' if i else 'baseline']
                startup(text,layout,i == 1)
                _,_,_,selected,sizes = work_evidence(text,previous_rows,record['process_plans'][i],root,layout,i == 1)
                headers = list(re.finditer(r'^strata serve: prompt .*$',text,re.M))
                guards.require(len(headers) == 4, 'four complete native warmups required')
                previous = 0
                for header in headers:
                    loan_cycle(text[previous:header.start()],selected,sizes); previous = header.end()
                row.update(native_identity=native,effective_environment_verified=True); persist()
            io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); guards.require(time.monotonic() < deadline, 'teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    owner = supervision.Owner(root/'owned', persist)
    try:
        if smoke:
            fixture = full['fixtures']['sycl_expert_bank_test']
            entry,out,err = owner.run('packed-smoke',[fixture['path'],'--packed-smoke'],record['smoke_environment'],P,
                wall=180,cpu=600,rss_cap=8<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
            stdout,stderr = out.read_text(),err.read_text()
            guards.require(supervision.closed(entry) and entry.get('exit_code') == 0 and
                not re.search(comparison.FAILURE,stderr) and 'urEnqueue' in stderr and 'urUSM' in stderr,
                'traced tiny fixture incomplete/failed')
            cases = check_packed_cases(stdout,'packed-smoke')
            row.update(cases=len(cases),fixture=fixture,trace_sha256=gen.common.digest(err))
        else:
            plan = record['process_plans'][i]
            entry,_,_ = owner.run('bench',plan['command'],plan['environment'],P,interaction=interact,
                wall=1800,cpu=54000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
            observed = evidence(root,record,plan,setup,boot,entry,i == 1)
            guards.require(row.get('native_identity') == observed['native_identity'] and
                row.get('effective_environment_verified') is True, 'native identity differs from warmup handshake')
            row.update(observed)
        current(record,full_models=not smoke)
        row.update(passed=True,complete=True)
    except BaseException as error:
        row['error'] = str(error)
        if (root/'engine.stderr').exists(): row['error_excerpt'] = (root/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        row.update(ownership_closed=owner.active is None and all(supervision.closed(e) for e in owner.commands),
            fresh_faults=None,fault_probe_successful=False,boot_unchanged=False)
        try:
            after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            row.update(fresh_faults=[s for s in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)],
                fault_probe_successful=True,boot_unchanged=boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            guards.driver_guard()
        except Exception as error: row['fault_probe_error'] = str(error)
        row['passed'] = bool(row['passed'] and row['ownership_closed'] and row['fault_probe_successful'] and
            row['fresh_faults'] == [] and row['boot_unchanged'] and 'fault_probe_error' not in row)
        persist()
        if smoke: record['smoke'] = row
        else: record['processes'].append(row)
        record['stage'] = 'closed' if not row['passed'] or len(record['processes']) == 2 else 'collecting'
        record['complete'] = bool(len(record['processes']) == 2 and all(r['complete'] for r in record['processes']))
        record['ownership_closed'] = row['ownership_closed']
        record['passed'] = bool(record['complete'] and all(r['passed'] for r in record['processes']))
        if record['passed']: record['decision'] = decide(record['processes'])
        write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ('passed','complete','ownership_closed','error')}),flush=True)
    return 0 if row['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run')); args = parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([]); guards.driver_guard()
        return prepare() if args.action == 'prepare' else run()


if __name__ == '__main__':
    sys.exit(main())
