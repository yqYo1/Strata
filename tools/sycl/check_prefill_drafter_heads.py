#!/usr/bin/env python3
"""Main's bounded real target/MTP-head ownership check; not a speed benchmark.

One new binary, immutable per-child settings, long/short/long requests. Retained
component fixtures keep their original native identities. No default adoption.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_dflash2_model as model

P, B, STATUS = model.P, model.B, model.STATUS
build, source, guards, ownership = model.build, model.source, model.guards, model.ownership
ROOT = B / 'prefill-drafter-heads-model-v1'
KEY = 'dflash2_prefill_heads_check'
HOST_KEY = 'optional_mtp_prefill_heads_host_v3'
REVIEW_KEY = 'optional_mtp_prefill_heads_source_audit_20261108'
require, digest, write = model.require, model.digest, model.write


def compact(row):
    result = {k: v for k, v in row.items() if k not in
        ('native_build', 'controller', 'models', 'runtime_files', 'environment', 'commands', 'requests')}
    if 'native_build' in row:
        result['native_build'] = {k: row['native_build'][k] for k in ('source_commit', 'project_source_sha256', 'record')}
    if 'controller' in row:
        result['controller_source'] = dict(commit=row['controller']['revision'], sha256=row['controller']['aggregate_sha256'])
    if 'requests' in row:
        result['requests'] = {k: {n: v for n, v in r.items() if n != 'ids'} for k, r in row['requests'].items()}
    result['commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'), error=c.get('error'),
        ownership_closed=ownership.closed(c), errors=c.get('errors', []), cleanup=c.get('cleanup', []))
        for c in row.get('commands', [])]
    return result


def save(row):
    write(ROOT / 'record.json', row)
    status = json.loads(STATUS.read_text()); status['current_sync'][KEY] = compact(row); write(STATUS, status)


def requests(tokenizer):
    content = 'Review the following stable merge sort requirements, then implement and explain the algorithm.\n'
    content += '\n'.join(f'Requirement {i}: preserve the original order of equal elements and explain the merge invariant.' for i in range(24))
    long = model.request_ids(tokenizer, content)
    short = model.request_ids(tokenizer)
    require(256 < long['prompt_tokens'] <= 2048 and short['prompt_tokens'] <= 128, 'request geometry changed')
    long['generator'] = 'check_prefill_drafter_heads.requests:24numbered stable-merge requirements,thinking template'
    short['generator'] = 'check_dflash2_model.request_ids:original82ID thinking prompt'
    return dict(long=long, short=short)


def prior_contract(native):
    """Scope old arithmetic/ring evidence separately from NEW ownership code."""
    previous = json.loads((model.ROOT / 'record.json').read_text())
    require(model.validation.committed(STATUS, ('current_sync', 'dflash2_model_check')) == model.compact(previous)
        and previous['passed'] and previous['ownership_closed'] and not previous['fresh_faults'] and
        len(previous['commands']) == 1 and ownership.closed(previous['commands'][0]) and
        previous['commands'][0]['exit_code'] == 0, 'closed original target diagnostic required')
    old = previous['native_build']
    require(native['recipe']['configure'] == old['recipe']['configure'] and
        native['recipe']['environment'] == old['recipe']['environment'] and native['toolchain'] == old['toolchain'] and
        native['external_inputs'] == old['external_inputs'] and
        {p['root']: p['revision'] for p in native['dependencies']} == old['dependency_revisions'], 'native recipe/dependencies changed')
    excluded = {'src/program/generate.cpp', 'include/strata/core/prefill_head_scope.hpp',
        'tests/sycl/prefill_head_scope.cpp', 'tools/sycl/build_upstream_features.py'}
    pins = native['project']['source_sha256']; matched = {}
    for name, raw in source.blobs(P, old['source_commit'], pins):
        if name in excluded: continue  # new ownership/caller code is under test, not reused qualification
        if name == 'src/core/verify.cpp':
            raw = raw.replace(b'idle intact single-device native MTP4 window', b'idle intact single-device native spec4 window')
        require(hashlib.sha256(raw).hexdigest() == pins[name], 'component native input changed: ' + name)
        matched[name] = pins[name]
    fixtures = {}
    for name, value in old['binaries'].items():
        if name == 'strata': continue
        require(native['binaries'][name] == value and digest(value['path']) == value['sha256'], 'original fixture changed')
        fixtures[name] = value
    return dict(original_model_record=dict(path=str(model.ROOT / 'record.json'), sha256=digest(model.ROOT / 'record.json')),
        original_native_source=old['source_commit'], matching_native_files=len(matched),
        matching_native_sha256=build.signature(matched), excluded_new_ownership_inputs=sorted(excluded),
        verifier_difference='Error text only;exact replacement checked,no arithmetic/retirement change', fixtures=fixtures,
        scope='Reuse original DFlash2 arithmetic/tap/ring evidence and exact original fixture bytes only.New generator/head ownership,graph handoff and new engine are unqualified until this run.')


def verify(row, full=False):
    require(model.validation.committed(STATUS, ('current_sync', KEY)) ==
        compact(json.loads((ROOT / 'record.json').read_text())), 'commit exact head plan/result')
    pin = row['controller']
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in source.blobs(P, pin['revision'], pin['source_sha256'])}
    require(recovered == pin['source_sha256'] and build.signature(recovered) == pin['aggregate_sha256'] and
        all(digest(P / n) == h for n, h in recovered.items()), 'controller/input changed')
    native = json.loads((build.ROOT / 'record.json').read_text())
    require(build.compact(native) == row['native_build'] == build.committed_plan() and
        native['passed'] and native['ownership_closed'], 'native build changed')
    build.check_inputs(native)
    for artifact in native['binaries'].values(): require(digest(artifact['path']) == artifact['sha256'], 'native output changed')
    require(prior_contract(native) == row['component_reuse'], 'original component contract changed')
    host = model.validation.committed(model.RESULT, (HOST_KEY,))
    require(host == row['host_policy'] and host['passed'] and host['ownership_closed'] and
        host['result']['cases'] == 29 and host['result']['checks'] == 845, 'host ownership gate changed')
    host_row = json.loads(Path(row['host_record']['path']).read_text())
    require(digest(row['host_record']['path']) == row['host_record']['sha256'] and
        host_row['passed'] and host_row['ownership_closed'] and len(host_row['commands_executed']) == 2 and
        all(ownership.closed(c) and c['exit_code'] == 0 for c in host_row['commands_executed']) and
        all(digest(P / n) == host_row['sources'][n] for n in
            ('include/strata/core/prefill_head_scope.hpp', 'tests/sycl/prefill_head_scope.cpp')), 'tested host policy native inputs changed')
    review = model.validation.committed(model.validation.INDEX, (REVIEW_KEY,))
    require(review['state'] == 'reviewed_no_ownership_blocker' and review['native_review'] == row['source_review'] and
        digest(row['source_review']['path']) == row['source_review']['sha256'], 'review incomplete/changed')
    probe, reuse = model.recover_probe(json.loads((model.REFERENCE / 'record.json').read_text()))
    require(probe == row['probe'] and reuse == row['runtime_reuse'], 'pre-main probe reuse changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
        json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'], 'runtime inventory changed')
    for p, h in row['runtime_files'].items(): require(digest(p) == h, 'runtime changed: ' + p)
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    require(requests(Path(row['tokenizer'])) == row['requests'] and
        model.fingerprint(row['environment']) == row['environment_sha256'], 'request/environment changed')
    model.model_pins(row, full=full); guards.no_model_owners([])


def prepare():
    require(not ROOT.exists(), 'one finite ownership plan;no automatic retry')
    native = json.loads((build.ROOT / 'record.json').read_text())
    require(native['stage'] == 'built' and native['passed'] and native['ownership_closed'] and
        build.compact(native) == build.committed_plan(), 'commit completed native build')
    host = model.validation.committed(model.RESULT, (HOST_KEY,))
    review = model.validation.committed(model.validation.INDEX, (REVIEW_KEY,))
    require(review['state'] == 'reviewed_no_ownership_blocker', 'source review first')
    previous = json.loads((model.ROOT / 'record.json').read_text())
    reference = json.loads((model.REFERENCE / 'record.json').read_text()); probe, reuse = model.recover_probe(reference)
    args = list(previous['argv'][2:])  # after engine,--serve;keep exact target/placement/draft assets
    for flag, value in [('--prefill', '256'), ('--short-read', '128')]: args[args.index(flag) + 1] = value
    environment = dict(previous['environment'], STRATA_PREFILL_RELEASE_MTP_EXPERTS='0',
        STRATA_SYCL_CACHE_SCRATCH_LEASE='0', STRATA_MTP_BATCH_F16='0', STRATA_SNAPSHOT_VERIFY='0')
    row = dict(stage='prepared', passed=False, ownership_closed=True, commands=[], cases=[],
        controller=source.pin(P, ['tools/sycl', 'tools/strata_tokenizer.py', 'data/expert-profile.bin']),
        native_build=build.compact(native), component_reuse=prior_contract(native),
        host_policy=host, host_record=dict(path=str(B / 'prefill-drafter-head-scope-v3/record.json'),
            sha256=digest(B / 'prefill-drafter-head-scope-v3/record.json')),
        source_review=review['native_review'], probe=probe, runtime_reuse=reuse,
        runtime_manifest=previous['runtime_manifest'], runtime_files=previous['runtime_files'],
        models=previous['models'], models_sha256=previous['models_sha256'], driver=previous['driver'], boot=previous['boot'],
        tokenizer=reference['configuration']['tokenizer'], requests=requests(Path(reference['configuration']['tokenizer'])),
        argv=[native['binaries']['strata']['path'], '--serve', *args],
        mtp_path=reference['configuration']['args'][reference['configuration']['args'].index('--mtp') + 1],
        environment=environment, environment_sha256=model.fingerprint(environment),
        order=[dict(label='df2_head0', draft='dflash2', heads=0), dict(label='df2_head1', draft='dflash2', heads=1),
               dict(label='mtp_head1', draft='mtp', heads=1)],
        rejection_order=[dict(label='reject_width1', args=['--mtp-max-t', '1'], environment={}, expected='--mtp-max-t 0 or >=2'),
            dict(label='reject_mtp_experts', args=[], environment={'STRATA_PREFILL_RELEASE_MTP_EXPERTS': '1'},
                expected='DFlash2 requires STRATA_PREFILL_RELEASE_MTP_EXPERTS=0 and STRATA_SYCL_CACHE_SCRATCH_LEASE=0'),
            dict(label='reject_cache_scratch', args=[], environment={'STRATA_SYCL_CACHE_SCRATCH_LEASE': '1'},
                expected='DFlash2 requires STRATA_PREFILL_RELEASE_MTP_EXPERTS=0 and STRATA_SYCL_CACHE_SCRATCH_LEASE=0')],
        request_order=['long', 'short', 'long'],
        budgets=dict(wall=900, cpu=10800, text_cap=64 << 20, file_cap=64 << 20, total_cap=256 << 20, rss_cap=104 << 30),
        criteria='One source/binary,three owned processes in declared order,long/short/long32output requests in each.'
            'Thinking enabled,short-read128/chunk256:long layer-major twice and short windows once.'
            'Each request starts RESUME0,completes extent,offers drafts and accepts at least one;no output equality requirement.'
            'Release1 emits two target detaches and expected optional draft reader;DF2 never detaches MTP weights.'
            'MTP release1 retains original duplicate-head ordering.Verifier/dense/canonical restored before decode;'
            'all children exit0/reaped/session empty,pre-main proof,actual INFO and READY maps,no fresh fault.'
            'Three admission children(width1,absent-MTP expert release,DFlash2 cache-scratch)must exit2 before GPU startup.'
            'No speed/task/full262144/RESTORE/standalone/default or ordinary provider teardown qualification.',
        speed_eligible=False, model_adoption_qualified=False,
        retention='Keep compact request hashes,configuration,per-request counters,head/free-memory observations and closures.'
            'Generated IDs are reproduced from committed generator+hash-pinned tokenizer;verify exact wire before launch.'
            'Discard successful stage/API text and caches after extracting useful evidence;no new engine copy.')
    ROOT.mkdir(mode=0o700)
    for case in row['order']:
        d = ROOT / case['label'];d.mkdir(mode=0o700)
        for n in ('tmp', 'cache'): (d / n).mkdir(mode=0o700)
    model.model_pins(row, full=True); save(row)
    print('Prepared optional-head model ownership plan;commit before native execution.', flush=True)


def run():
    row = json.loads((ROOT / 'record.json').read_text()); require(row['stage'] == 'prepared', 'no duplicate/retry')
    verify(row, full=True)
    kernel = model.mask.ring.device.kernel_log
    cursor = next(s.split(': ', 1)[1] for s in kernel('-n', '0', '--show-cursor').splitlines() if s.startswith('-- cursor: '))
    monitor = None
    def persist(): write(ROOT / 'active.json', dict(active=bool(monitor and monitor.active), commands=monitor.commands if monitor else []))
    def fresh_faults():
        text = kernel('--after-cursor', cursor, '-n', '2048'); require(len(text.splitlines()) < 2048, 'incomplete kernel journal')
        return [s for s in text.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', s)]
    try:
        monitor = ownership.Owner(ROOT / 'owned', persist)
        row['admission_results'] = []
        for case in row['rejection_order']:
            cli = row['argv'] + case['args']
            env = dict(row['environment'], UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:stderr', **case['environment']);env.pop('LD_PRELOAD', None)
            entry, out, err = monitor.run(case['label'], cli, env, P,
                wall=30, cpu=60, text_cap=1 << 20, file_cap=1 << 20, total_cap=4 << 20, rss_cap=2 << 30)
            text = err.read_text(errors='replace')
            require(ownership.closed(entry) and entry['exit_code'] == 2 and case['expected'] in text and
                'strata generate:' not in text and not out.read_bytes() and 'UR_PROBE' not in text and
                not re.search(r'--->\s*ur[A-Z]|<---\s*ur[A-Z]', text), 'pre-GPU admission failed: ' + case['label'])
            row['admission_results'].append(dict(label=case['label'], exit_code=2, ownership_closed=True,
                expected_error_seen=True, observed_UR_calls=0, argv_sha256=model.fingerprint(cli),
                environment_sha256=model.fingerprint(env)))
        for case in row['order']:
            require(not fresh_faults(), 'fault before next child'); guards.no_model_owners([])
            label = case['label']; env = dict(row['environment'], TMPDIR=str(ROOT / label / 'tmp'),
                XDG_CACHE_HOME=str(ROOT / label / 'cache'), STRATA_MEASUREMENT_ROOT=str(ROOT / label),
                STRATA_PREFILL_RELEASE_HEADS=str(case['heads']))
            argv = list(row['argv'])
            if case['draft'] == 'mtp':
                for flag in ('--draft', '--draft-model', '--draft-fallback', '--draft-conf', '--mtp-draft-vocab'):
                    i = argv.index(flag); del argv[i:i + 2]
                argv += ['--mtp', row['mtp_path']]
            observed = dict(label=label, draft=case['draft'], heads=case['heads'], requests=[],
                argv_sha256=model.fingerprint(argv), environment_overrides={k: v for k, v in env.items() if row['environment'].get(k) != v},
                environment_sha256=model.fingerprint(env))
            row['cases'].append(observed)
            def interact(io, owned):
                lines = []
                while True:
                    line = io.line(seconds=480); lines.append(line); require(len(lines) <= 32, 'startup line cap')
                    if line.startswith('READY '): break
                require(line == 'READY 4096 stop', 'unexpected READY')
                info = [s[5:] for s in lines if s.startswith('INFO ')];require(len(info) == 1, 'one INFO')
                placement = dict(s.split('=', 1) for s in info[0].split())
                require(all(placement.get(k) == v for k, v in
                    dict(context='4096', kv='int8', kv_resident='0', expert_slots='1256', spec='4', mtp_max='0',
                         lookup='0', pool_workers='5', pcie_frac='0.00', conversation_cache_mib='0').items()), 'INFO placement differs')
                observed['placement'] = placement
                observed.update(model.runtime_evidence(dict(row, environment=env), owned['pid'],
                    (ROOT / 'owned' / (label + '.stderr')).read_text(errors='replace')))
                for kind in row['request_order']:
                    request = row['requests'][kind]; parser = model.protocol.RequestProtocol(request['prompt_tokens'], request['generated'])
                    io.send(model.wire(request['ids'], request['generated']))
                    while parser.done is None: parser.feed(io.line(seconds=480))
                    require(int(parser.done[7]) > 0, 'no draft offered')
                    observed['requests'].append(dict(kind=kind, prompt_tokens=request['prompt_tokens'],
                        generated=len(parser.tokens), accepted=int(parser.done[6]), offered=int(parser.done[7]),
                        finish=parser.done[5], output_ids_sha256=model.fingerprint(parser.tokens), complete=True))
                    require(int(parser.done[6]) > 0, 'accepted-token handoff not exercised')
                    print(label + ' completed ' + kind, flush=True)
                io.send(b'QUIT\n'); io.exit(30)
            entry, out, err = monitor.run(label, argv, env, P, interaction=interact, **row['budgets'])
            require(ownership.closed(entry) and entry['exit_code'] == 0, 'native closure failed: ' + str(entry.get('error')))
            text = err.read_text(errors='replace')
            detaches = re.findall(r'strata prefill heads: target detached bytes=(\d+)', text)
            draft_detaches = re.findall(r'strata prefill heads: draft detached bytes=(\d+)', text)
            require(len(detaches) == 2 * case['heads'] and all(0 < int(n) <= 1 << 30 for n in detaches), 'target head transition count')
            require(len(draft_detaches) == (2 if case['heads'] and case['draft'] == 'mtp' else 0), 'optional MTP head transition count')
            if case['heads']:
                require(text.count('draft_reader=' + ('mtp4' if case['draft'] == 'mtp' else 'absent-dflash2')) == 2, 'actual reader selection')
            require(text.count('route=layer-major') == 2 and text.count('route=cache-window') == 1, 'actual long/short route')
            require('UR_RESULT_ERROR_' not in text, 'UR error')
            observed.update(target_head_detached_bytes=list(map(int, detaches)), draft_head_detached_bytes=list(map(int, draft_detaches)),
                dense_memory=[dict(x.split('=', 1) for x in line.split() if '=' in x) for line in text.splitlines()
                    if line.startswith('strata prefill dense: phase=')], passed=True, ownership_closed=True)
            require(not fresh_faults(), 'fresh GPU fault')
            print(label + ' ownership/handoff closed', flush=True)
        verify(row, full=True);row['passed'] = True
    except BaseException as error: row['error'] = repr(error)
    finally:
        row['commands'] = monitor.commands if monitor else []
        row['ownership_closed'] = not monitor or monitor.active is None and all(ownership.closed(c) for c in row['commands'])
        try:
            row['fresh_faults'] = fresh_faults()
            require(not row['fresh_faults'] and guards.driver_guard() == row['driver'] and
                Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'fresh fault/driver/boot change')
        except BaseException as error: row.update(passed=False, fault_error=repr(error))
        row['passed'] = bool(row['passed'] and row['ownership_closed']);row['stage'] = 'passed' if row['passed'] else 'failed';save(row)
    require(row['passed'], row.get('error', row.get('fault_error', 'ownership gate failed')))
    print('Bounded real prefill-head ownership passed;no speed/full-context adoption.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'use prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'run': run}[sys.argv[1]]()
