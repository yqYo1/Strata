#!/usr/bin/env python3
"""Main-only first target-model DFlash2 diagnostic, not a speed benchmark.

Reuse the later closed native-child validation evidence under its exact current
runtime. Preserve the older canary's runtime mismatch; never refresh its pins.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
import check_dflash2_mask as mask
import native_request_protocol as protocol
import ur_validation as validation

build, source, guards, ownership = mask.build, mask.source, mask.guards, mask.ownership
P, B, STATUS = mask.P, mask.B, mask.STATUS
ROOT = B / 'dflash2-target-model-v1'
REFERENCE = mask.ring.REFERENCE
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
REFERENCE_KEY = 'decode_iq3_audit_speed_v2'
REVIEW_KEY = 'implemented_dflash2_mask_discriminator_audit_20261108'
INPUTS = [*mask.INPUTS, 'tools/sycl/check_dflash2_model.py', 'tools/sycl/ur_validation.py',
          'tools/sycl/native_request_protocol.py', 'tools/strata_tokenizer.py',
          'bench/sycl/ur_validation_preload.c', 'bench/sycl/ur_validation_canary.c', 'data/expert-profile.bin']
require, write, digest = guards.require, build.write, source.digest


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def compact(row):
    out = {k: v for k, v in row.items() if k not in ['native_build', 'runtime_files', 'controller', 'models', 'environment']}
    if 'native_build' in row:
        out['native_build'] = {k: row['native_build'][k] for k in ['source_commit', 'project_source_sha256', 'record']}
    if 'controller' in row:
        out['controller_source'] = dict(commit=row['controller']['revision'], sha256=row['controller']['aggregate_sha256'])
    out['commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'), error=c.get('error'),
        ownership_closed=ownership.closed(c), errors=c.get('errors', []), cleanup=c.get('cleanup', [])) for c in row['commands']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    status = json.loads(STATUS.read_text()); status['current_sync']['dflash2_model_check'] = compact(row)
    write(STATUS, status)


def recover_probe(reference):
    public = validation.committed(RESULT, (REFERENCE_KEY,))
    payload = {k: v for k, v in reference.items() if k not in ['sources', 'model_assets']}
    payload['source_inputs'] = dict(files=len(reference['sources']), aggregate_sha256=fingerprint(reference['sources']))
    require(fingerprint(payload) == fingerprint(public) and reference['stage'] == 'closed' and reference['passed'] and
            reference['complete'] and reference['ownership_closed'] and not reference['active'], 'later reference not committed/closed')
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in source.blobs(P, reference['source_commit'], reference['sources'])}
    require(recovered == reference['sources'], 'later validation caller source unrecoverable')
    probe = reference['native_validation_probe']
    component = validation.committed(validation.INDEX, ('UR_silent_validation_native_child_scope', 'preload_component'))
    require(component['passed'] and component['complete'] and component['ownership_closed'] and
            not component['fresh_faults'] and component['provider_mismatch_stopped_before_body'] and
            probe['library'] == component['library'], 'probe is not the qualified native library')
    ref = component['original_plan']; raw = source.git(P, 'show', ref['commit'] + ':' + ref['path'])
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'original probe recipe changed')
    original = json.loads(raw)['UR_silent_validation_native_child_scope']['canary_plan']
    require(original['source_commit'] == component['source_commit'] and len(original['build_commands']) == 2,
            'original probe source/recipe identity differs')
    project = {str(Path(p).relative_to(P)): h for p, h in original['inputs'].items() if Path(p).is_relative_to(P)}
    require({n: hashlib.sha256(raw).hexdigest() for n, raw in source.blobs(P, component['source_commit'], project)} == project,
            'original probe project inputs unrecoverable')
    for p, h in original['inputs'].items():
        # The original controllers may evolve. The C implementation and every
        # original external compiler/header input must still match its recipe.
        if not Path(p).is_relative_to(P) or Path(p).suffix == '.c':
            require(digest(p) == h, 'original probe build input changed: ' + p)
    for artifact in probe.values(): require(digest(artifact['path']) == artifact['sha256'], 'probe/loader ELF changed')
    require(len(reference['processes']) == 6, 'six later independent processes required')
    for case in reference['processes']:
        proof = case['validation']
        require(case['passed'] and case['complete'] and case['ownership_closed'] and case['native_exit_code'] == 0 and
                case['boot_unchanged'] and not case['fresh_faults'] and proof['pid'] == case['native_identity']['pid'] and
                Path(proof['loader']).resolve() == Path(probe['loader']['path']).resolve() and
                proof['threads_before'] == 1 and proof['preinit'] == 37 and proof['first_init'] and
                proof['config_enabled'] and proof['dispatch_result'] == 46 and proof['output_untouched'] and
                proof['retained_reference_until_exit'] and not proof['runtime_teardown_qualified'], 'later pre-main proof incomplete')
    return probe, dict(original_source_commit=component['source_commit'], original_plan=ref,
        build_commands_sha256=fingerprint(original['build_commands']), original_input_files=len(original['inputs']),
        original_inputs_sha256=fingerprint(original['inputs']), later_source_commit=reference['source_commit'],
        later_committed_key=REFERENCE_KEY, later_processes=6, later_pre_main_proofs_checked=True,
        legacy_refusal_preserved=True, ordinary_runtime_teardown_qualified=False)


def model_identity(path):
    s = path.stat()
    return dict(realpath=str(path.resolve()), dev=s.st_dev, ino=s.st_ino, bytes=s.st_size,
        mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def model_pins(row, full=False):
    deadline = time.monotonic() + 300
    require(fingerprint(row['models']) == row['models_sha256'], 'model manifest changed')
    cfg = json.loads((REFERENCE / 'record.json').read_text())['configuration']['args']
    paths = set(Path(cfg[cfg.index('--native') + 1]).parent.glob('*.gguf'))
    for flag in ['--pack', '--mtp']: paths.update(p for p in Path(cfg[cfg.index(flag) + 1]).rglob('*') if p.is_file())
    require(set(map(str, paths)) == set(row['models']) and len(paths) == 15, 'model inventory changed')
    for p in sorted(paths):
        identity = model_identity(p)
        require(identity == row['models'][str(p)]['identity'], 'model stat changed: ' + str(p))
        if full:
            h = hashlib.sha256()
            with p.open('rb', buffering=0) as stream:
                while block := stream.read(8 << 20):
                    require(time.monotonic() < deadline, 'whole-model digest300s budget'); h.update(block)
            require(h.hexdigest() == row['models'][str(p)]['sha256'] and model_identity(p) == identity, 'model bytes changed: ' + str(p))
            print('Verified model asset ' + p.name, flush=True)
    require(mask.ring.PACK.stat().st_size == 522458112 and digest(mask.ring.PACK) == mask.ring.PACK_SHA, 'stock draft pack changed')


def request_ids(tokenizer):
    # Use the maintained tokenizer and the model's own thinking-enabled template.
    # Commit the resulting small ID list/wire hash in the prepared plan; the
    # native run consumes those exact IDs, not another tokenizer invocation.
    sys.path.insert(0, str(P / 'tools'))
    from strata_tokenizer import Tokenizer
    from jinja2 import Environment
    vocab = json.loads((tokenizer / 'vocab.json').read_text()); tokens = [None] * len(vocab)
    for name, i in vocab.items(): tokens[i] = name
    config = json.loads((tokenizer / 'tokenizer.json').read_text())
    require(config['pre'] == 'qwen35' and len(tokens) == 248320, 'unexpected target tokenizer')
    tok = Tokenizer(tokens, (tokenizer / 'merges.txt').read_text().split('\n'),
        json.loads((tokenizer / 'token_type.json').read_text()), config['pre'], config['special_ids'])
    messages = [dict(role='user', content='Write a Python implementation of a stable merge sort with type annotations, then explain its correctness and time complexity. Work through the design before writing the code.')]
    text = Environment().from_string((tokenizer / 'chat_template.jinja').read_text()).render(
        messages=messages, tools=None, add_generation_prompt=True, enable_thinking=True)
    require(text.endswith('<think>\n'), 'thinking prompt suffix changed')
    ids = tok.encode(text, parse_special=True)
    require(16 <= len(ids) <= 2048 and all(type(x) is int and 0 <= x < 248320 for x in ids) and tok.decode(ids) == text,
            'small real prompt/tokenizer round trip')
    return dict(ids=ids, generated=32, thinking=True, rendered_sha256=hashlib.sha256(text.encode()).hexdigest(),
        wire_sha256=hashlib.sha256(wire(ids, 32)).hexdigest(), prompt_tokens=len(ids))


def wire(ids, count):
    return ('GEN ' + str(count) + ' ' + ','.join(map(str, ids)) + '\n').encode()


def verify(row):
    public = validation.committed(STATUS, ('current_sync', 'dflash2_model_check'))
    require(public == compact(json.loads((ROOT / 'record.json').read_text())), 'commit exact model plan/result')
    pin = row['controller']
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in source.blobs(P, pin['revision'], pin['source_sha256'])}
    require(recovered == pin['source_sha256'] and build.signature(recovered) == pin['aggregate_sha256'] and
            all(digest(P / p) == h for p, h in recovered.items()), 'model controller changed')
    native = json.loads((build.ROOT / 'record.json').read_text())
    require(build.compact(native) == row['native_build'], 'native build changed')
    m = json.loads((mask.ROOT / 'record.json').read_text()); mask.verify(m)
    require(digest(mask.ROOT / 'record.json') == row['mask_reference']['sha256'] and m['passed'] and m['cases_passed'] == 12,
            'mask semantic gate changed')
    review = validation.committed(validation.INDEX, (REVIEW_KEY,))
    require(review['state'] == 'reviewed_no_component_blocker' and
            digest(row['source_review']['path']) == row['source_review']['sha256'] == review['native_review']['sha256'], 'source review changed/incomplete')
    require(digest(REFERENCE / 'record.json') == row['reference_record']['sha256'], 'later runtime reference changed')
    reference = json.loads((REFERENCE / 'record.json').read_text())
    probe, reuse = recover_probe(reference)
    require(probe == row['probe'] and reuse == row['runtime_reuse'] and
            fingerprint(row['environment']) == row['environment_sha256'], 'probe reuse/environment changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
            json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'], 'runtime manifest changed')
    require(len(row['runtime_files']) == 84, 'full runtime inventory required')
    for p, h in row['runtime_files'].items(): require(digest(p) == h, 'runtime ELF changed: ' + p)
    require(guards.driver_guard() == row['driver'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    require(hashlib.sha256(wire(row['request']['ids'], row['request']['generated'])).hexdigest() == row['request']['wire_sha256'], 'request wire changed')
    model_pins(row); guards.no_model_owners([])


def prepare():
    require(not ROOT.exists(), 'one finite model diagnostic;no automatic retry')
    ROOT.mkdir(mode=0o700)
    row = dict(stage='preparing', commands=[], passed=False, ownership_closed=True)
    try:
        row['controller'] = source.pin(P, INPUTS); require(row['controller']['count'] == len(INPUTS), 'commit controller inputs')
        native = json.loads((build.ROOT / 'record.json').read_text()); require(build.committed_plan() == build.compact(native), 'commit feature build')
        m = json.loads((mask.ROOT / 'record.json').read_text()); mask.verify(m)
        require(m['passed'] and m['ownership_closed'] and m['cases_passed'] == 12 and not m['fresh_faults'], 'mask component must pass')
        review = validation.committed(validation.INDEX, (REVIEW_KEY,))
        require(review['state'] == 'reviewed_no_component_blocker', 'source review first')
        reference = json.loads((REFERENCE / 'record.json').read_text()); probe, reuse = recover_probe(reference)
        require(all(reference['runtime_manifest'][k] == m['runtime_manifest'][k] for k in ['path', 'sha256']),
                'later pre-main evidence and component runtime differ')
        row.update(native_build=build.compact(native), mask_reference=dict(path=str(mask.ROOT / 'record.json'), sha256=digest(mask.ROOT / 'record.json')),
            source_review=dict(path=review['native_review']['path'], sha256=review['native_review']['sha256']),
            reference_record=dict(path=str(REFERENCE / 'record.json'), sha256=digest(REFERENCE / 'record.json')),
            runtime_manifest=reference['runtime_manifest'], runtime_files=json.loads(Path(reference['runtime_manifest']['path']).read_text()),
            probe=probe, runtime_reuse=reuse, driver=m['driver'], boot=m['boot'], models=reference['model_assets'],
            models_sha256=fingerprint(reference['model_assets']), model_files=15)
        model_pins(row, full=True)
        args = list(reference['configuration']['args']); vocab = str(Path(args[args.index('--mtp') + 1]) / 'draft_vocab.bin')
        i = args.index('--mtp'); del args[i:i + 2]
        for flag, value in [('--max-context', '4096'), ('--kv-resident', '4096'), ('--prefill', '1024'), ('--vram-reserve-mib', '1500')]:
            args[args.index(flag) + 1] = value
        args += ['--draft', 'dflash2', '--draft-model', str(mask.ring.PACK), '--draft-fallback', 'none', '--draft-conf', '0', '--mtp-draft-vocab', vocab]
        overrides = dict(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'), STRATA_MEASUREMENT_ROOT=str(ROOT),
            LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=str(Path(probe['loader']['path']).resolve()),
            CUDA_MODULE_LOADING='EAGER', STRATA_PREFILL_RELEASE_MTP_EXPERTS='0', STRATA_PREFILL_RELEASE_HEADS='0',
            STRATA_SYCL_CACHE_SCRATCH_LEASE='0', STRATA_MTP_BATCH_F16='0', STRATA_SNAPSHOT_VERIFY='0',
            STRATA_DF2_RING_LOG='1', STRATA_DF2_LOG_MIB='64', STRATA_DF2_RING_STATS='1',
            STRATA_DF2_TIMING='1', STRATA_TRACE='1', STRATA_DECODE_TIMING='1',
            UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            UR_LOG_LOADER='level:warning;flush:warning;output:stderr', UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        env = dict(reference['environment_template'], **overrides); env.pop('UR_LOG_TRACING', None)
        require(all(env[k] == m['environment'][k] for k in mask.ring.device.ENV_KEYS), 'qualified safety/provider environment differs')
        row.update(environment=env, environment_overrides=overrides, environment_sha256=fingerprint(env),
            argv=[native['binaries']['strata']['path'], '--serve', *args],
            request=request_ids(Path(reference['configuration']['tokenizer'])), budgets=dict(wall=600, cpu=7200,
                text_cap=64 << 20, file_cap=64 << 20, total_cap=256 << 20, rss_cap=104 << 30),
            meaning='One thinking-enabled real prompt,32output tokens,actual stock DFlash2 and target embeddings/head. Require positive offered drafts,complete GEN,normal exit,pre-main validation,qualified maps and no fresh fault. Diagnostic only;no speed,task quality,full-context,RESTORE,phase preservation or default adoption.',
            speed_eligible=False, model_adoption_qualified=False, stage='prepared')
        for name in ['tmp', 'cache']: (ROOT / name).mkdir(mode=0o700)
        save(row); print('Prepared first native target-model diagnostic;commit before launch.', flush=True)
    except BaseException as error:
        row.update(stage='preparation_failed', error=repr(error)); save(row); raise


def run():
    row = json.loads((ROOT / 'record.json').read_text()); require(row['stage'] == 'prepared', 'no duplicate/retry')
    verify(row)
    before = mask.ring.device.kernel_log('-n', '0', '--show-cursor')
    cursor = next(s.split(': ', 1)[1] for s in before.splitlines() if s.startswith('-- cursor: '))
    monitor = None
    def persist(): write(ROOT / 'active.json', dict(active=bool(monitor and monitor.active), commands=monitor.commands if monitor else []))
    try:
        monitor = ownership.Owner(ROOT / 'owned', persist)
        def interact(io, entry):
            lines = []
            while True:
                line = io.line(seconds=480); lines.append(line); require(len(lines) <= 32, 'startup stdout cap')
                if line.startswith('READY '): break
            require(line == 'READY 4096 stop', 'unexpected native READY')
            stderr = ROOT / 'owned/model.stderr'; proof = validation.preload_evidence(stderr.read_text(errors='replace'), entry['pid'])
            require(Path(proof['loader']).resolve() == Path(row['probe']['loader']['path']).resolve(), 'actual pre-main loader differs')
            actual = set()
            for line in Path('/proc', str(entry['pid']), 'maps').read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and 'x' in fields[1] and fields[5].startswith('/'):
                    require(not fields[5].endswith(' (deleted)'), 'mapped code removed'); actual.add(str(Path(fields[5]).resolve()))
            allowed = {str(Path(p).resolve()): h for p, h in row['runtime_files'].items()}
            for p, h in [(row['argv'][0], row['native_build']['binaries']['strata']['sha256']),
                         (row['probe']['library']['path'], row['probe']['library']['sha256'])]: allowed[str(Path(p).resolve())] = h
            required = {str(Path(p).resolve()) for p in [row['probe']['library']['path'], row['probe']['loader']['path'],
                row['environment']['UR_ADAPTERS_FORCE_LOAD'], row['environment']['ZE_ENABLE_ALT_DRIVERS'],
                '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0']}
            require(len(actual) >= 8 and actual <= allowed.keys() and required <= actual, 'actual executable maps unqualified')
            require(all(digest(p) == allowed[p] for p in actual), 'mapped ELF bytes changed')
            row['pre_main_validation'] = proof; row['mapping'] = dict(files=len(actual), sha256=build.signature({p: allowed[p] for p in sorted(actual)}))
            print('Native READY with actual pre-main validation and executable mappings.', flush=True)
            request = row['request']; parser = protocol.RequestProtocol(request['prompt_tokens'], request['generated'])
            io.send(wire(request['ids'], request['generated']))
            while parser.done is None: parser.feed(io.line(seconds=480))
            require(int(parser.done[7]) > 0, 'no actual speculative draft was offered')
            row['request_result'] = dict(prompt_tokens=request['prompt_tokens'], generated=len(parser.tokens),
                drafts_accepted=int(parser.done[6]), drafts_offered=int(parser.done[7]), finish=parser.done[5],
                output_ids_sha256=fingerprint(parser.tokens), complete=True)
            io.send(b'QUIT\n'); io.exit(30)
        entry, stdout, stderr = monitor.run('model', row['argv'], row['environment'], P, interaction=interact, **row['budgets'])
        require(ownership.closed(entry) and entry['exit_code'] == 0, 'native closure failed: ' + str(entry.get('error')))
        text = stderr.read_text(errors='replace')
        require('strata DFlash2: eager,' in text and not re.search(r'UR_RESULT_ERROR_[A-Z_]+', text), 'missing actual DFlash2/UR error')
        row['pre_main_validation'] = validation.preload_evidence(text, entry['pid'])
        verify(row); model_pins(row, full=True); row['passed'] = True
    except BaseException as error: row['error'] = repr(error)
    finally:
        row['commands'] = monitor.commands if monitor else []; row['native_launched'] = bool(row['commands'])
        row['ownership_closed'] = not monitor or monitor.active is None and all(ownership.closed(c) for c in row['commands'])
        try:
            after = mask.ring.device.kernel_log('--after-cursor', cursor, '-n', '2048')
            require(len(after.splitlines()) < 2048, 'incomplete kernel journal')
            row['fresh_faults'] = [s for s in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', s)]
            require(not row['fresh_faults'] and guards.driver_guard() == row['driver'] and
                Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'fresh fault/boot/driver change')
        except BaseException as error: row.update(passed=False, fault_error=repr(error))
        row['passed'] = bool(row['passed'] and row['ownership_closed']); row['stage'] = 'passed' if row['passed'] else 'failed'; save(row)
    require(row['passed'], row.get('error', row.get('fault_error', 'model gate failed')))
    print('First target-model DFlash2 diagnostic passed;no speed/task/full-context qualification.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'run'], 'use prepare or run')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (prepare() if sys.argv[1] == 'prepare' else run())
