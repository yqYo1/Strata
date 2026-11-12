#!/usr/bin/env python3
"""Bounded existing decode H2D-route admission; Main owns every GPU child.

Main-authored from maintained Strata controls; repository LICENSE applies.
Short requests establish execution/ownership, never speed, quality or adoption.
"""
import copy
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import coherent_native_reuse as reuse
import coherent_prefill_model as original
from q6_canonical_component import validated_trace
from native_request_protocol import RequestProtocol

P, B, RESULT = reuse.P, reuse.B, reuse.RESULT
ROOT, KEY = B / 'coherent-decode-transfer-smoke-v2', 'coherent_decode_transfer_smoke_v2'
LONG = False
THIS = 'tools/sycl/coherent_decode_transfer.py'
PYTHON = original.PYTHON
require, sha = reuse.require, reuse.sha
CASES = (('success', 'S'), ('large', 'L'), ('bounds', 'B'))


def native(expected=None):
    row, proof = reuse.verify(observers={THIS}, expected=expected)
    return dict(row, model_observer_admission=proof)


def requests(tokenizer):
    if LONG:
        return original.original.requests(tokenizer)
    short = original.original.requests(tokenizer)['short']
    require(short['prompt_tokens'] == 82 and short['generated'] == 32 and short['thinking'],
        'preserve the bounded thinking-enabled request')
    return dict(long=copy.deepcopy(short), short=short)


def parse_info(text):
    # INFO prints two decimals; the actual frozen argv uses exact finite1/16.
    # Reuse the established schema/placement checks for all other fields.
    normalized, count = re.subn(r'(?<!\S)pcie_frac=0\.06(?=\s|$)',
        'pcie_frac=0.00', text)
    require(count == 1, 'actual finite1/16 display required')
    fields = original.original.parse_info(normalized)
    fields['pcie_frac'] = '0.06'
    return fields


def copy_evidence(text, transcript, specs):
    lines = transcript.splitlines()
    require(len(lines) >= 2 and lines[0].startswith('INFO ') and lines[1] == 'READY 262144 stop',
        'one native startup before the exact request stream')
    parse_info(lines[0][5:])
    at, actual = 2, []
    for spec in specs:
        parser = RequestProtocol(spec['prompt_tokens'], spec['generated'])
        while parser.done is None:
            require(at < len(lines), 'incomplete request in actual stdout')
            parser.feed(lines[at]); at += 1
        actual.append(parser)
    require(at == len(lines), 'unplanned native stdout after completed requests')
    selected = [line for line in text.splitlines() if line.startswith(
        ('strata serve: prompt ', 'strata serve: SYCL_EXPERT_COPY '))]
    require(len(selected) == 2 * len(actual), 'one ordered prompt/copy pair per completed request')
    retained = []
    for index, parser in enumerate(actual):
        prompt = re.fullmatch(r'strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read '
            r'in (\d+) ms \([0-9.]+ tok/s\), (\d+) generated in (\d+) ms \([0-9.]+ tok/s\), '
            r'drafts accepted (\d+) of (\d+), (\d+) checkpoints', selected[2*index])
        require(prompt is not None, 'prompt footer must precede its copy footer')
        n, reused, read, pp_ms, generated, tg_ms, accepted, offered, checkpoints = map(int, prompt.groups())
        require(n == read == parser.prompt_tokens and reused == 0
            and generated == len(parser.tokens) == parser.output_tokens
            and accepted == int(parser.done[6]) and offered == int(parser.done[7])
            and checkpoints == 0 and abs(pp_ms-float(parser.done[3])) <= 1
            and abs(tg_ms-float(parser.done[4])) <= 1,
            'prompt/copy ordinal must bind the actual stdout request and rounded clocks')
        copy_row = re.fullmatch(r'strata serve: SYCL_EXPERT_COPY submitted_blobs=(\d+) submitted_bytes=(\d+) '
            r'completed_blobs=(\d+) completed_bytes=(\d+) host_view_groups=(\d+) '
            r'gpu_miss_groups=(\d+) cpu_unique_misses=(\d+) cpu_entries=(\d+)', selected[2*index+1])
        require(copy_row is not None, 'copy footer must immediately follow its prompt in the filtered stream')
        submitted, byte_count, completed, completed_bytes, views, groups, cpu_jobs, entries = map(int, copy_row.groups())
        require(submitted == completed == views == groups and submitted > 0
            and byte_count == completed_bytes
            and submitted * 1510400 <= byte_count <= submitted * 2662400
            and cpu_jobs > 0 and entries >= cpu_jobs,
            'actual private-Arena host-view copy/retirement counters do not conserve')
        retained.append(dict(request_ordinal=index+1, prompt_tokens=n, generated_tokens=generated,
            accepted=accepted, offered=offered, output_ids_sha256=original.original.signature(parser.tokens),
            submitted_blobs=submitted, requested_bytes=byte_count,
            completed_blobs=completed, completed_bytes=completed_bytes,
            resident_host_view_groups=views, GPU_miss_groups=groups,
            CPU_unique_misses=cpu_jobs, CPU_entries=entries))
    return retained


def route(text):
    require('SYCL_PCIE_EARLY' not in text, 'keep early dispatch off')
    prefill = original.route(text) if LONG else {}
    if not LONG:
        require('layer-major candidate' not in text, 'preserve the short-input route')
    marker = re.findall(r'^strata verify: ordinary-Arena copy enabled mode=0 '
        r'staging_bytes=(\d+) arena_bytes=(\d+)$', text, re.M)
    require(len(marker) == 1 and int(marker[0][0]) == 16 * 2662400
        and int(marker[0][1]) > int(marker[0][0]), 'actual explicit-copy/staging admission required')
    frozen = json.loads((ROOT/'record.json').read_text())
    specs = [frozen['requests'][name] for name in ('long','short','long')]
    retained = copy_evidence(text, (ROOT/'owned/model.stdout').read_text(), specs)
    return dict(actual_requests=len(retained), actual_pcie_fraction_display='0.06',
        frozen_fraction='1/16', frozen_pcie_num=16, mode=0, early_dispatch=False,
        copy_records=retained, prefill=prefill, request_footer_pairing_checked=True,
        per_window_copy_bins_unavailable=True,
        physical_bus_traffic_or_saturation_qualified=False,
        numerical_task_or_speed_qualified=False)


def context():
    ctx = original.context()
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.native, ctx.requests, ctx.parse_info, ctx.route = native, requests, parse_info, route
    return ctx


def prepare():
    ctx = context()
    require(not ROOT.exists() and KEY not in ctx.device.common.public(), 'fresh admission plan required')
    n = native()
    prior = json.loads((original.ROOT / 'record.json').read_text())
    require(ctx.validation.committed(RESULT, (original.KEY,)) == ctx.previous.compact(prior)
        and prior['stage'] == 'closed' and all(prior[k] for k in
            ('passed', 'complete', 'ownership_closed')) and not prior['active'],
        'closed coherent handoff/assets/configuration required')
    inherited = ('models', 'models_sha256', 'runtime_manifest', 'runtime_files', 'probe',
        'generator_runtime', 'tokenizer', 'boot', 'driver')
    row = {k: copy.deepcopy(prior[k]) for k in inherited}
    old_root = prior['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old_root, str(ROOT)) for k, v in prior['environment'].items()}
    env.update(STRATA_SYCL_VERIFY_ARENA_COPY='1', STRATA_SYCL_VERIFY_PCIE_EARLY='0',
        STRATA_SYCL_VERIFY_OVERLAP='1')
    argv = list(prior['argv'])
    require(argv.count('--pcie-frac') == 1 and argv[argv.index('--pcie-frac')+1] == '0',
        'preserve and explicitly replace the zero-share configuration')
    argv[argv.index('--pcie-frac')+1] = '0.0625'
    require(argv[0] == n['binaries']['strata']['path'] and all(env.get(k) == v for k, v in
        dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1',
            EnableDirectSubmission='0', STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0',
            STRATA_SYCL_PREFILL_SAME_QUEUE='1').items()), 'preserve actual executable and runtime safety')
    row.update(stage='prepared', source_commit=ctx.device.common.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        controller=ctx.context(n['model_observer_admission']).controller_pin(),
        native_source_commit=n['native_source_commit'], native_build=dict(binaries=n['binaries']),
        native_record=dict(path=str(reuse.ROOT / 'record.json'), sha256=sha(reuse.ROOT / 'record.json')),
        native_observer_admission=n['model_observer_admission'],
        device_evidence_sha256=ctx.signature(ctx.validation.committed(RESULT, (ctx.device.KEY,))),
        controller_python=dict(path=sys.executable, prefix=sys.prefix, sha256=sha(sys.executable)),
        controller_commands=[[PYTHON, '-B', THIS, a] for a in ('prepare', 'host', 'device', 'run')],
        requests=requests(Path(row['tokenizer'])), environment=env, argv=argv,
        budgets=dict(wall=900, cpu=3600, rss_cap=104 << 30, text_cap=16 << 20,
            file_cap=32 << 20, total_cap=64 << 20),
        plan=dict(question='Can existing private-Arena explicit H2D safely offload a finite1/16 of decode misses?',
            order='Commit plan;host parser/extent/request-join/no-retry controls;three external no-fork retained-fixture cases '
                'success128B,large16*2662400B,bounds6;commit closure;then one serve process with three '
                + ('thinking-on32769/82/32769input/32output requests.' if LONG else
                   'thinking-on82input/32output requests.') +
                ' Early dispatch off. Stop first failure,no retry.',
            mechanism='Existing per-expert copies only,not rejected copy-run batching. Finite0.0625 '
                'rounds to16/256;floor(distinct_misses/16),with existing16slot staging cap. Same '
                'prefill strategies,weights,kernels,1256cache slots,20480residentINT8KV and262144capacity.',
            acceptance='All source/build/12ELF/recipe/runtime/15asset identities,pre-main validation,normal0 '
                'owned closure and no fresh fault. Actual INFO0.06,mode0Arena opt-in marker,positive '
                'submitted/completed copy counts/bytes for all3requests,nonzero CPU remainder,thinking '
                'and actual MTPoffers. No early-dispatch output. Device source/memory/guards checked. '
                'Pair every copy footer with its immediately preceding prompt footer and the actual '
                'stdout DONE extent/drafts/rounded times;host-view groups must equal all submitted groups. '
                'Full15asset hashes after model. No speed or exact-text decision.',
            stopping='Any source/provider/workload/owner/budget/driver/fault/copy-counter mismatch stops '
                'this plan. Do not refresh hashes,relax gates,retry or advance an unrun/failed stage.',
            next='Only after normal validated closure,commit a separate >=32K official bench.sh '
                'agent/strong-and-weak-MTP >=1Kthinking-generation comparison and phase-preservation gates.'),
        source_review=dict(report_index_key='decode_pcie_share_admission_20261112',
            report_sha256='f6bc0f0a21fb8903f4408dcd1091c6c8dbbfae58db7eaa9f9e8896fd91464865',
            native_review_sha256='cf8f8f0031a5f724678102b76c008d5dcccf728518ee28746937963486debe6c',
            Main_additional_headers={p: sha(P/p) for p in
                ('include/strata/core/pinned.hpp', 'include/strata/core/expert_layer_view.hpp')},
            remaining='The request parser has an independent nonfinite-to-int source hazard. This '
                'owned plan supplies only finite1/16. Fix arbitrary-client admission before adoption; '
                'no untrusted request values or direct/UVA mode in this check.'),
        limits='Native route/state admission only;not clean speed,numerical/task quality,agent benchmark,physical '
            '262144/tails/RESTORE,ordinary provider teardown or inference adoption. Requested bytes '
            'and staging counters are not physical bus traffic or resource saturation.',
        retention='Keep compact actual per-request/case counts,source/configuration and normal/fault '
            'outcomes for the imminent long comparison. Retire successful traces/JIT at closure. '
            'Use the existing coherent generation;no new engine or fixture build.',
        device_results=[], results=[], commands=[], passed=False, complete=False,
        active=False, ownership_closed=True, model_executed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT/name).mkdir(mode=0o700)
    ctx.context(n['model_observer_admission']).save(row)
    print('Prepared bounded1/16decode-copy admission;commit before host/device/model.', flush=True)


def claim_attempt(path, row, kind):
    # Exclusive creation precedes journal/probe work, so even a failure before
    # Owner exists cannot silently make the frozen stage retryable.
    with path.open('x') as stream:
        json.dump(dict(stage='attempt_started', kind=kind, source_commit=row['source_commit'],
            result_key=KEY), stream)
        stream.write('\n')


def pairing_controls(good):
    wire = lambda f: ' '.join(k+'='+v for k, v in f.items())
    transcript = ['INFO '+wire(good), 'READY 262144 stop']
    footers = []
    for i in range(3):
        accepted, offered = 5+i, 10+i
        transcript += ['RESUME 0', 'PP 81 82 10.0 8.1', 'REUSED 0']
        transcript += ['T '+str(i*32+t) for t in range(32)]
        transcript += [f'DONE 32 82 10.0 20.0 length {accepted} {offered} 0 0 0 0 0 0.0 82']
        footers += [f'strata serve: prompt 82 tokens = 0 reused + 82 read in 10 ms (8200.0 tok/s), '
            f'32 generated in 20 ms (1600.0 tok/s), drafts accepted {accepted} of {offered}, 0 checkpoints',
            'strata serve: SYCL_EXPERT_COPY submitted_blobs=4 submitted_bytes=6041600 '
            'completed_blobs=4 completed_bytes=6041600 host_view_groups=4 '
            'gpu_miss_groups=4 cpu_unique_misses=9 cpu_entries=10']
    stdout, text = '\n'.join(transcript)+'\n', '\n'.join(footers)+'\n'
    specs = [dict(prompt_tokens=82, generated=32)]*3
    require(len(copy_evidence(text, stdout, specs)) == 3, 'paired actual-request control failed')
    bad = [(text.replace('host_view_groups=4', 'host_view_groups=0'), stdout),
        ('\n'.join([footers[1],footers[0],*footers[2:]])+'\n', stdout),
        ('\n'.join(footers[1:])+'\n', stdout),
        (text.replace('drafts accepted 5 of 10', 'drafts accepted 6 of 10'), stdout),
        (text.replace('completed_bytes=6041600', 'completed_bytes=6041601'), stdout),
        (text+'\n'+footers[1], stdout),
        (text, '\n'.join(transcript[:-1])+'\n'),
        (text, stdout.replace('DONE 32 82', 'DONE 31 82', 1)),
        (text.replace('read in 10 ms', 'read in 12 ms'), stdout)]
    for stderr, output in bad:
        refused = False
        try:
            copy_evidence(stderr, output, specs)
        except RuntimeError:
            refused = True
        require(refused, 'unpaired/wrong-provider/incomplete request control accepted')
    path = ROOT/'tmp/host-attempt-control.json'
    require(not path.exists(), 'fresh exclusive-attempt host control')
    claim_attempt(path, dict(source_commit='host-fixture'), 'host-control')
    refused = False
    try:
        claim_attempt(path, dict(source_commit='host-fixture'), 'host-control')
    except FileExistsError:
        refused = True
    require(refused, 'attempt could be silently retried'); path.unlink()
    return dict(valid_three_request_streams=1, corrupt_streams_rejected=len(bad),
        exclusive_attempt_rejected=True)


def host_checks():
    ctx = context(); row = json.loads((ROOT/'record.json').read_text())
    owned = ctx.context(row['native_observer_admission']); owned.verify(row)
    require(row['stage'] == 'prepared', 'one host admission,no retry')
    previous = json.loads((original.ROOT/'record.json').read_text())['results'][0]['placement']
    good = dict(previous, pcie_frac='0.06')
    wire = lambda f: ' '.join(k+'='+v for k, v in f.items())
    parse_info(wire(good)); parse_info(wire(dict(reversed(list(good.items())))))
    bad = [wire(dict(good, pcie_frac=v)) for v in ('0.00', 'nan', 'inf', '0.0625')]
    bad += [wire(dict(good, **{k: v})) for k, v in
        (('context', '131072'), ('expert_slots', '1536'), ('kv_resident', '0'))]
    bad += [wire(good)+' pcie_frac=0.06', wire({k:v for k,v in good.items() if k!='engine'})]
    for text in bad:
        refused = False
        try:
            parse_info(text)
        except RuntimeError:
            refused = True
        require(refused, 'changed/nonfinite/duplicate INFO control accepted')
    joins = pairing_controls(good)
    row.update(stage='host_passed', host_controls=dict(positive=2, negative=len(bad), passed=True,
        request_pairing=joins, GPU_executed=False))
    owned.save(row)
    print('Host placement controls pass;commit before device checks.', flush=True)


def host():
    row = json.loads((ROOT/'record.json').read_text())
    require(row['stage'] == 'prepared', 'one host attempt,no retry')
    claim_attempt(ROOT/'host-attempt.json', row, 'host')
    try:
        host_checks()
    except BaseException as error:
        failed = json.loads((ROOT/'record.json').read_text())
        failed.update(stage='failed', passed=False, complete=False, active=False,
            ownership_closed=True, host_error=repr(error), GPU_executed=False)
        context().context(failed['native_observer_admission']).save(failed)
        raise


def device():
    ctx = context(); row = json.loads((ROOT/'record.json').read_text())
    owned = ctx.context(row['native_observer_admission'])
    require(row['stage'] == 'host_passed' and not row['device_results'], 'one device gate,no retry')
    claim_attempt(ROOT/'device-attempt.json', row, 'device')
    result = dict(passed=False); scope = None; cursor = None
    def persist():
        ctx.write(ROOT/'device-execution.json', dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []))
    try:
        owned.verify(row)
        cursor = next(v.split(': ',1)[1] for v in subprocess.check_output(
            ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True, timeout=15).splitlines()
            if v.startswith('-- cursor: '))
        scope = ctx.owner.Owner(ROOT/'device-owned', persist)
        env = dict(row['environment'], UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:stderr')
        for name, marker in CASES:
            owned.verify(row)
            binary = row['native_build']['binaries']['sycl_verifier_retirement_test']
            event, out, err = scope.run(name, [binary['path'], '--case', name, '1'], env, P,
                wall=90, cpu=180, rss_cap=4<<30, text_cap=16<<20, file_cap=16<<20, total_cap=64<<20)
            text = err.read_text(errors='replace')
            proof, body = validated_trace(text, event['pid'])
            require(ctx.owner.closed(event) and event['exit_code'] == 0 and out.read_text() == marker
                and 'urEnqueue' in body
                and not re.search(r'terminate called|teardown uncertain', text)
                and (not (ROOT/'validation.log').exists() or (ROOT/'validation.log').stat().st_size == 0),
                'actual no-fork copy fixture/validation failed')
            result.setdefault('cases', []).append(dict(case=name, marker=marker, exit_code=0,
                ownership_closed=True, pre_main_validation=proof))
        result['host_probe_accepted'] = True
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        if scope:
            ctx.validation.finish(result, scope, cursor, row['boot'])
            row['commands'] += scope.commands
        else:
            result.update(ownership_closed=True, native_execution_attempted=False,
                host_probe_accepted=False, preflight_failure=True)
        result['passed'] = bool(result.get('passed') and len(result.get('cases', [])) == len(CASES)
            and result.get('driver_after') == row['driver'])
        row['device_results'].append(result)
        row.update(stage='host_passed' if result['passed'] else 'failed',
            device_checks_complete=result['passed'],
            active=bool(scope and scope.active), ownership_closed=bool(result.get('ownership_closed')))
        owned.save(row)
    require(result['passed'], result.get('error', 'copy preflight failed'))
    print('Three no-fork copy fixtures pass;commit before model.', flush=True)


def run():
    row = json.loads((ROOT/'record.json').read_text())
    require(row['stage'] == 'host_passed' and row.get('device_checks_complete')
        and len(row['device_results']) == 1 and row['device_results'][0]['passed'],
        'closed copy device gate required')
    claim_attempt(ROOT/'model-attempt.json', row, 'model')
    try:
        context().execute('run')
    except BaseException as error:
        failed = json.loads((ROOT/'record.json').read_text())
        # The inherited executor records its own native outcomes. A failure
        # before its try/cursor must also be durable,without inventing a child.
        receipt = ROOT/'execution.json'
        state = json.loads(receipt.read_text()) if receipt.exists() else dict(active=False,commands=[])
        closed = not state['active'] and all(original.original.owner.closed(c) for c in state['commands'])
        for command in state['commands']:
            for identity in command.get('owners', []):
                live = original.original.owner.identity(identity['pid'])
                if live and live['start_ticks'] == identity['start_ticks']:
                    closed = False
        failed.update(stage='failed', passed=False, complete=False,
            controller_outer_error=repr(error), active=not closed, ownership_closed=closed)
        owned = context().context(failed['native_observer_admission']); owned.save(failed)
        raise


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == 'long':
        LONG = True; args = args[1:]
        ROOT, KEY = B/'coherent-decode-transfer-long-v2', 'coherent_decode_transfer_long_v2'
    require(sys.executable == PYTHON and len(args) == 1
        and args[0] in ('prepare','host','device','run'), 'Use qualified Python:[long] prepare|host|device|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); original.original.guards.no_model_owners([])
        globals()[args[0]]()
