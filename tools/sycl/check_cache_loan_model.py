#!/usr/bin/env python3
"""Main's bounded owned/checked-borrowed DFlash2 model handoff diagnostic.

Repository LICENSE. Reuses independently authored production code and qualified
fixture/runtime readers. Payload readbacks are charged; this is not a benchmark.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_cache_loan_cli as loan
import check_cache_loan_state as state
import check_prefill_drafter_heads as heads
import check_dflash2_model as model

P, B, STATUS, ROOT = loan.P, loan.B, state.STATUS, loan.B / 'dflash2-cache-loan-model-v1'
KEY = 'dflash2_cache_loan_model_v1_check'
HOST_COMMIT = '3c7ccce8f20c6f05fcb37be87660b666d2320773'
HOST_KEY = 'dflash2_cache_loan_heads_host_v1'
INPUTS = list(dict.fromkeys([*state.INPUTS, *loan.INPUTS, *[
    'tools/sycl/' + name for name in ('check_cache_loan_model.py', 'check_prefill_drafter_heads.py',
    'check_dflash2_model.py', 'check_dflash2_mask.py', 'check_dflash2_ring.py', 'native_request_protocol.py')],
    'tools/strata_tokenizer.py', 'data/expert-profile.bin']))
require, digest, write, signature = loan.require, loan.digest, loan.write, loan.build.signature


class ResumedProtocol(model.protocol.RequestProtocol):
    """Keep the strict cold reader, checking actual prefix counters first."""
    def __init__(self, prompt, output, resume):
        super().__init__(prompt, output)
        require(0 <= resume < prompt - 1, 'invalid diagnostic prefix')
        self.resume, self.actual_done = resume, None

    def feed(self, line):
        fields = line.split()
        require(bool(fields), 'empty native line')
        marker = fields[0]
        if marker in ('RESUME', 'REUSED'):
            require(line == f'{marker} {self.resume}', 'unexpected live prefix extent')
            super().feed(marker + ' 0')
            if marker == 'RESUME': self.position = self.resume
        elif marker == 'DONE':
            require(len(fields) == 15 and fields[8] == str(self.resume) and
                fields[14] == str(self.prompt_tokens - self.resume), 'actual reused/read extent differs')
            normalized = list(fields); normalized[8] = '0'; normalized[14] = str(self.prompt_tokens)
            super().feed(' '.join(normalized)); self.actual_done = fields
        else: super().feed(line)


def compact(row):
    out = {k: v for k, v in row.items() if k not in
        ('controller', 'models', 'runtime_files', 'environment', 'commands', 'native_build', 'generator_runtime')}
    out['controller'] = {k: row['controller'][k] for k in ('revision', 'count', 'aggregate_sha256')}
    out['native_build'] = {k: row['native_build'][k] for k in ('source_commit', 'project_source_sha256', 'record')}
    out['environment_sha256'] = signature(row['environment'])
    out['generator_runtime'] = {k: v for k, v in row['generator_runtime'].items() if k != 'inputs'}
    out['commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'), error=c.get('error'),
        ownership_closed=loan.ownership.closed(c), errors=c.get('errors', []), cleanup=c.get('cleanup', []))
        for c in row['commands']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    public = json.loads(STATUS.read_text()); public['current_sync'][KEY] = compact(row); write(STATUS, public)


def prerequisites(native):
    device = loan.validation.committed(STATUS, ('current_sync', 'dflash2_cache_loan_device_check'))
    remaining = loan.validation.committed(STATUS, ('current_sync', state.KEY))
    cli = loan.validation.committed(loan.STATUS, (loan.KEY,))
    for name, result in [('device', device), ('state', remaining), ('CLI', cli)]:
        require(result['passed'] and result['ownership_closed'] and not result['fresh_faults'] and
            result['native_build']['record'] == native['record'], 'unqualified same-build ' + name)
    require(device['numerical_cases_passed'] == 16 and len(remaining['results']) == 5 and
        sum(r['cases'] for r in remaining['results']) == 53 and all(r['passed'] for r in remaining['results']) and
        len(cli['case_results']) == 12, 'incomplete same-build fixture/CLI census')
    actual = cli['actual_inputs']
    require(actual['uncovered'] == 0 and digest(actual['path']) == actual['sha256'], 'actual input audit changed')
    original = json.loads(loan.source.git(P, 'show', HOST_COMMIT + ':' + str(loan.STATUS.relative_to(P))))[HOST_KEY]
    require(original['passed'] and original['complete'] and original['ownership_closed'] and not original['active'] and
        original['result'] == dict(kind='prefill_head_scope', cases=38, checks=854, status='PASS'), 'host gate incomplete')
    n, _ = loan.native()
    files = ('src/program/generate.cpp', 'include/strata/core/cache_scratch_lease.hpp', 'tests/sycl/prefill_head_scope.cpp')
    recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in loan.source.blobs(P, original['source_commit'], files)}
    require(all(n['project']['source_sha256'][p] == sha for p, sha in recovered.items()), 'host policy/native source differ')
    return dict(device=signature(device), state=signature(remaining), CLI=signature(cli),
        host_evidence_commit=HOST_COMMIT, host_source_commit=original['source_commit'], host_native_inputs=recovered,
        actual_inputs=actual)


def verify(row, full=False):
    require(loan.validation.committed(STATUS, ('current_sync', KEY)) == compact(
        json.loads((ROOT / 'record.json').read_text())), 'commit exact model plan/result')
    pin = row['controller']
    require({p: hashlib.sha256(raw).hexdigest() for p, raw in loan.source.blobs(P, pin['revision'],
        pin['source_sha256'])} == pin['source_sha256'] and signature(pin['source_sha256']) == pin['aggregate_sha256'] and
        all(digest(P / p) == sha for p, sha in pin['source_sha256'].items()), 'controller changed/unrecoverable')
    _, native = loan.native(); require(native == row['native_build'], 'candidate build changed')
    require(prerequisites(native) == row['prerequisites'], 'qualification evidence changed')
    probe, reuse = loan.current_probe()
    require(probe == row['native_validation_probe'] == row['probe'] and reuse == row['runtime_reuse'], 'runtime reuse changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
        json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'], 'runtime map changed')
    for p, sha in row['runtime_files'].items(): require(digest(p) == sha, 'runtime ELF changed: ' + p)
    require(heads.generator_runtime() == row['generator_runtime'], 'workload generator runtime changed')
    require(loan.guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'driver/boot changed')
    model.model_pins(row, full=full)
    loan.guards.no_model_owners([])


def payload(row, case):
    ids = [i for part in case['parts'] for i in row['requests'][part]['ids']]
    wire = model.wire(ids, case['generated'])
    require(len(ids) == case['prompt_tokens'] and hashlib.sha256(wire).hexdigest() == case['wire_sha256'], 'wire differs')
    return wire


def prepare():
    require(not ROOT.exists(), 'one finite cohort;no automatic retry')
    _, native = loan.native(); prior = prerequisites(native); probe, reuse = loan.current_probe()
    old = json.loads((heads.ROOT / 'record.json').read_text())
    reference = json.loads((model.REFERENCE / 'record.json').read_text())
    argv = list(old['argv']); argv[0] = native['binaries']['strata']['path']
    argv[argv.index('--prompt-cache') + 1] = '1'; argv += ['--prompt-cache-every', '0']
    env = {k: v for k, v in old['environment'].items() if k not in
        ('LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER', 'TMPDIR', 'XDG_CACHE_HOME',
         'STRATA_MEASUREMENT_ROOT', 'PYTHONPYCACHEPREFIX')}
    env.update(STRATA_PREFILL_RELEASE_HEADS='1', STRATA_SYCL_CACHE_SCRATCH_LEASE='0', STRATA_SYCL_CACHE_LOAN_AUDIT='0')
    require(env['UR_ENABLE_LAYERS'] == 'UR_LAYER_PARAMETER_VALIDATION' and
        env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0' and
        env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0' and env['ZES_ENABLE_SYSMAN'] == '0' and
        env['STRATA_PREFILL_RELEASE_MTP_EXPERTS'] == '0' and env['STRATA_PREFILL_NATIVE_DENSE_BANK'] == '1', 'safety/policy settings')
    requests = heads.requests(Path(old['tokenizer']))
    require(requests['short']['prompt_tokens'] == 82 and requests['long']['prompt_tokens'] == 513, 'known route geometry')
    order = []
    for name, parts, generated, resume in [('short', ['short'], 32, 0), ('long', ['long'], 32, 0),
        ('known_live', ['short'], 1, 0), ('resumed_long', ['short', 'long'], 32, 82)]:
        ids = [i for part in parts for i in requests[part]['ids']]
        order.append(dict(label=name, parts=parts, prompt_tokens=len(ids), generated=generated, resume=resume,
            fresh_tokens=len(ids) - resume, route='layer-major' if len(ids) - resume > 128 else 'cache-window',
            wire_sha256=hashlib.sha256(model.wire(ids, generated)).hexdigest()))
    models = reference['model_assets']
    row = dict(stage='prepared', passed=False, ownership_closed=True, commands=[], results=[],
        controller=loan.source.pin(P, INPUTS), native_build=native, prerequisites=prior, native_validation_probe=probe,
        probe=probe, runtime_reuse=reuse, runtime_manifest=old['runtime_manifest'], runtime_files=old['runtime_files'],
        models=models, models_sha256=model.fingerprint(models), model_files=len(models),
        model_bytes=sum(v['identity']['bytes'] for v in models.values()), pack=dict(path=str(state.PACK), sha256=state.PACK_SHA, bytes=522458112),
        generator_runtime=heads.generator_runtime(), tokenizer=old['tokenizer'], requests=requests, order=order,
        environment=env, argv=argv, arms=[dict(label='owned', loan=False), dict(label='checked_borrowed', loan=True)],
        driver=loan.guards.driver_guard(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        budgets=dict(wall=900, cpu=10800, text_cap=64 << 20, file_cap=64 << 20, total_cap=256 << 20, rss_cap=104 << 30),
        criteria='Two fresh same90f processes,in fixed owned then checked-borrowed order;immutable environments.'
            'Each completes short82/long513/short82one-output/resumed595 with32/32/1/32thinking outputs.'
            'Require exact RESUME/REUSED0/0/0/82 and fresh82/513/82/513;PP absolute extent/order;valid tokens/DONE.'
            'Nontrivial positive offered/accepted drafts for32output requests;one-output control has no offers.'
            'Actual routes2cache-window/2layer-major,2target-head transitions with absent-MTP DFlash2,no draft-head detach.'
            'Borrowed arm:positive balanced monotone checked mask/ready leases,matching layout/mask capacity,'
            'lend/refill device-residency equality,all resident expert payload bytes equal RAM source in original slots before publication.'
            'Require at least one completed loan per long request;no loan audit in owned control.'
            'Same-PID before-main validation and actual mapped ELFs;quiet validation,no fresh fault,normal0/reap/empty.'
            'Full15model asset hashes and controller/native/runtime pins before and after.Stop first failure,no retry/reset.'
            'No generated text identity requirement. Known-live one-output avoids unprinted accepted-token overcommit.'
            'This diagnostic is not task quality,performance,ordinary-provider teardown,physical262144/tails/RESTORE or adoption.',
        speed_eligible=False, model_adoption_qualified=False,
        retention='Commit compact per-arm/request ownership,extent/counters,loan/payload totals,identities and fault result.'
            'Then delete normal full stage logs/cache after source/recipe recovery. Keep original plan/workload in Git.'
            'At this first model decision select feature generation and retire extra build after exact original-input recovery;'
            'preserve independent552b/e1b phase references and carry explicit unresolved legacy provenance holds.')
    model.model_pins(row, full=True)
    require(row['driver'] == reference['driver'] and row['boot'] == reference['boot'], 'qualified current host changed')
    ROOT.mkdir(mode=0o700)
    for arm in row['arms']:
        for name in ('tmp', 'cache'): (ROOT / arm['label'] / name).mkdir(mode=0o700, parents=True)
    save(row); print('Prepared2processes/8requests;commit exact plan before native execution.', flush=True)


def lease_evidence(text, borrowed, required):
    events = re.findall(r'strata cache loan: phase=(masked|ready) generation=(\d+) chunk=(\d+) first=(\d+) lent=(\d+) decode_ready=([01])', text)
    layouts = re.findall(r'strata loan audit: phase=layout chunk=(\d+) first=(\d+) slots=(\d+) bytes=(\d+) need=(\d+) lent=(\d+) mask_match=1', text)
    payloads = re.findall(r'strata loan audit: phase=payload rows=(\d+) bytes=(\d+) lent=(\d+) same_slots=1 payload_match=1', text)
    tables = re.findall(r'strata loan audit: phase=(lend|refill) table_entries=(\d+) table_match=1 free_bytes=(\d+) total_bytes=(\d+)', text)
    if not borrowed:
        require(not events and not layouts and not payloads and not tables, 'owned arm executed cache loan')
        return dict(pairs=0)
    require(len(events) % 2 == 0, 'open cache lease')
    pairs = len(events) // 2
    require(pairs >= int(required) and len(layouts) == len(payloads) == pairs and len(tables) == 2 * pairs,
        'missing/extra loan audit stages')
    previous = 0; payload_bytes = 0; lent_rows = []
    for i in range(pairs):
        masked, ready = events[2 * i:2 * i + 2]
        require(masked[0] == 'masked' and ready[0] == 'ready' and masked[1:5] == ready[1:5] and
            masked[5] == '0' and ready[5] == '1', 'loan publication order/state differs')
        generation, chunk, first, lent = map(int, masked[1:5])
        require(generation > previous and chunk > 0 and first >= 128 and lent > 0, 'invalid/nonmonotone lease')
        previous = generation
        lc, lf, slots, size, need, ll = map(int, layouts[i]); rows, size_read, pl = map(int, payloads[i])
        require((lc, lf, ll) == (chunk, first, lent) and first < slots and size >= need > 0 and
            lent <= slots - first and pl == lent and rows >= lent and size_read > 0, 'layout/refill extent differs')
        a, b = tables[2 * i:2 * i + 2]
        require(a[0] == 'lend' and b[0] == 'refill' and a[1] == b[1] and int(a[1]) >= rows and
            all(0 < int(t[2]) <= int(t[3]) for t in (a, b)), 'device residency audit differs')
        payload_bytes += size_read; lent_rows.append(lent)
    require(text.count('strata cache loan:') == 2 * pairs and text.count('strata loan audit:') == 4 * pairs,
        'unparsed loan audit event')
    return dict(pairs=pairs, generations=[int(e[1]) for e in events[::2]], lent_rows=lent_rows,
        resident_payload_bytes_checked=payload_bytes, exact_payloads=True, mask_match=True, same_slots=True,
        ready_after_refill=True)


def run():
    row = json.loads((ROOT / 'record.json').read_text()); require(row['stage'] == 'prepared', 'no retry')
    verify(row, full=True)
    cursor = next(s.split(': ', 1)[1] for s in model.mask.ring.device.kernel_log('-n', '0', '--show-cursor').splitlines()
        if s.startswith('-- cursor: '))
    owner = None
    def persist(): write(ROOT / 'active.json', dict(active=bool(owner and owner.active), commands=owner.commands if owner else []))
    try:
        owner = loan.ownership.Owner(ROOT / 'owned-logs', persist)
        for arm in row['arms']:
            require(not state.faults(cursor), 'fault before next child'); loan.guards.no_model_owners([])
            label, borrowed = arm['label'], arm['loan']
            env = dict(row['environment'], TMPDIR=str(ROOT / label / 'tmp'), XDG_CACHE_HOME=str(ROOT / label / 'cache'),
                STRATA_MEASUREMENT_ROOT=str(ROOT / label), STRATA_SYCL_CACHE_SCRATCH_LEASE=str(int(borrowed)),
                STRATA_SYCL_CACHE_LOAN_AUDIT=str(int(borrowed)))
            env = loan.engine_once.native_environment(row, env)
            argv = list(row['argv']) + ([] if borrowed else ['--no-prefill-borrow'])
            observed = dict(label=label, loan=borrowed, passed=False, requests=[], argv_sha256=signature(argv), environment_sha256=signature(env))
            row['results'].append(observed); errpath = ROOT / 'owned-logs' / (label + '.stderr')
            def interact(io, entry):
                lines = []
                while True:
                    line = io.line(seconds=480); lines.append(line); require(len(lines) <= 32, 'startup line budget')
                    if line.startswith('READY '): break
                require(line == 'READY 4096 stop', 'unexpected READY')
                info = [s[5:] for s in lines if s.startswith('INFO ')]; require(len(info) == 1, 'one INFO required')
                placement = dict(s.split('=', 1) for s in info[0].split())
                require(all(placement.get(k) == v for k, v in dict(context='4096', kv='int8', kv_resident='0',
                    expert_slots='1256', spec='4', mtp_max='0', lookup='0', pool_workers='5', pcie_frac='0.00',
                    conversation_cache_mib='0').items()), 'INFO placement differs')
                observed['placement'] = placement
                observed.update(model.runtime_evidence(dict(row, environment=env), entry['pid'], errpath.read_text(errors='replace')))
                for case in row['order']:
                    offset = errpath.stat().st_size
                    parser = ResumedProtocol(case['prompt_tokens'], case['generated'], case['resume'])
                    io.send(payload(row, case))
                    while parser.done is None: parser.feed(io.line(seconds=480))
                    done = parser.actual_done
                    require(int(done[7]) > 0 and int(done[6]) > 0 if case['generated'] > 1 else
                        int(done[7]) == 0 and int(done[6]) == 0, 'draft path not exercised as declared')
                    with errpath.open('rb') as stream: stream.seek(offset); segment = stream.read().decode(errors='replace')
                    other = 'cache-window' if case['route'] == 'layer-major' else 'layer-major'
                    require(segment.count('route=' + case['route']) == 1 and 'route=' + other not in segment,
                        'actual request route differs')
                    lease = lease_evidence(segment, borrowed, case['route'] == 'layer-major')
                    observed['requests'].append(dict(label=case['label'], prompt_tokens=case['prompt_tokens'],
                        generated=len(parser.tokens), resumed=int(done[8]), fresh_tokens=int(done[14]),
                        prompt_ms=float(done[3]), decode_ms=float(done[4]), finish=done[5], accepted=int(done[6]),
                        offered=int(done[7]), progress_count=parser.progress_count, route=case['route'], lease=lease,
                        output_ids_sha256=signature(parser.tokens), complete=True))
                    print(label + ' completed ' + case['label'] + ' loan_pairs=' + str(lease['pairs']), flush=True)
                io.send(b'QUIT\n'); io.exit(30)
            entry, _, err = owner.run(label, argv, env, P, interaction=interact, **row['budgets'])
            require(loan.ownership.closed(entry) and entry['exit_code'] == 0, 'native did not close normally: ' + str(entry.get('error')))
            text = err.read_text(errors='replace')
            target = list(map(int, re.findall(r'strata prefill heads: target detached bytes=(\d+)', text)))
            require(len(target) == 2 and all(0 < n <= 1 << 30 for n in target) and
                'strata prefill heads: draft detached' not in text and text.count('draft_reader=absent-dflash2') == 2,
                'joint target-head/absent-MTP transitions differ')
            require('UR_RESULT_ERROR_' not in text and not state.faults(cursor), 'validation or fresh device fault')
            observed.update(passed=True, ownership_closed=True, target_head_detached_bytes=target,
                lease_totals=lease_evidence(text, borrowed, True))
            print(label + ' normal0/reaped/empty;no fresh fault', flush=True)
        verify(row, full=True); row.update(passed=True, stage='passed')
    except BaseException as error: row.update(passed=False, stage='failed', error=repr(error))
    finally:
        row['commands'] = owner.commands if owner else []
        row['ownership_closed'] = owner is None or owner.active is None and all(loan.ownership.closed(c) for c in row['commands'])
        try:
            row['fresh_faults'] = state.faults(cursor)
            require(not row['fresh_faults'] and loan.guards.driver_guard() == row['driver'] and
                Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'fault/driver/boot changed')
        except BaseException as error: row.update(passed=False, stage='failed', fault_error=repr(error))
        if not row['ownership_closed']: row.update(passed=False, stage='failed')
        save(row)
    require(row['passed'], row.get('error', row.get('fault_error', 'model handoff failed')))
    print('PASS2processes/8requests;diagnostic only,no speed/full-context/default adoption.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'use prepare or run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (prepare if sys.argv[1] == 'prepare' else run)()
