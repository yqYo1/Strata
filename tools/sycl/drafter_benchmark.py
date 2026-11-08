#!/usr/bin/env python3
"""Own matched bench.sh drafter runs with the already qualified native engine.

Main-authored from this repository's measurement/runtime interfaces; repository
LICENSE. Small ownership tests do not qualify the tuned long-input route. Keep
that admission, sustained decode, clean speed and default adoption separate.
"""
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import check_prefill_drafter_heads as heads
import benchmark_runtime as python_runtime
import engine_once
import generation_engine as evidence
import decode_footer

P, B = heads.P, heads.B
RESULT = heads.model.RESULT
ROOT = B / 'drafter-long-admission-v3'
KEY = 'drafter_long_admission_v3'
PREVIOUS = B / 'drafter-long-admission-v2'
FAILURE_COMMIT = '1381e9a9b997115308fa2cc1dc462e73d376bfa8'
REVIEW_KEYS = ('dflash2_tuned_long_shape_admission_20261108', 'dflash2_bench_clock_counter_contract_20261108')
SELF = Path(__file__).resolve()
READ, RUNS = 32769, 2
ORDER = ('mtp', 'dflash2')
require, digest, write = heads.require, heads.digest, heads.write


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def request_route(text, drafter, sample):
    """Check the observed route without assuming an optional MTP reader exists."""
    def records(prefix):
        return [dict(re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)', line))
                for line in text.splitlines() if line.startswith(prefix)]
    def one(prefix):
        rows = records(prefix); require(len(rows) == 1, 'missing/extra route record:' + prefix)
        return rows[0]
    n = READ-1
    main = one('strata prefill: layer-major candidate, ')
    require(all(main.get(k) == v for k,v in dict(tokens=str(n), chunk='8192', residual_dtype='bf16',
        residual_gpu_rows=str(n), residual_element_bytes='2', residual_total=str(n*20480),
        residual_gpu=str(n*20480), residual_host='0', residual_staging='0').items()) and
        text.count('strata prefill: native16 BF16 HC ') == 1, 'native16/full GPU residual route differs')
    bank = one('strata MoE expert bank: '); finish = one('strata MoE expert bank complete: ')
    require(all(bank.get(k) == v for k,v in dict(requested_slots='384', effective_slots='384', staged='1',
        residual_gpu_rows=str(n), general_map='private_scratch_never_published').items()) and
        bank['device_bytes'] == main['layer_cache'] and 0 < int(bank['device_bytes']) <= int(bank['allocation_limit_bytes']) and
        finish['total_slots'] == '384' and finish['final_drain'] == 'checked', 'staged bank/drain differs')
    window = one('strata MoE window configuration: '); work = one('strata MoE accumulator window: ')
    require(window['requested_mode'] == window['effective_mode'] == '10' and
        window['maximum_rows'] == work['maximum_rows'] == str(n) and work['windows'] == '48' and
        work['GPU_resident_windows_only'] == '1' and work['operator_max'] == '8192', 'MoE window route differs')
    split = one('strata MoE split admission: ')
    require(split['active'] == split['decode_changed'] == '0' and split['rows'] == str(n), 'unexpected split/decode placement')
    targets = records('strata prefill heads: target detached ')
    drafts = records('strata prefill heads: draft detached ')
    require(len(targets) == 1 and targets[0]['source'] == 'owned-RAM' and
        targets[0]['bytes'] == targets[0]['D2H'] == targets[0]['H2D-on-restore'] == '521472000', 'target head retirement differs')
    require([r['bytes'] for r in drafts] == (['223227900'] if drafter == 'mtp' else []) and
        one('strata prefill heads: draft_reader=')['draft_reader'] == ('mtp4' if drafter == 'mtp' else 'absent-dflash2'),
        'optional MTP reader retirement differs')
    callbacks = re.findall(r'^strata serve: DRAFT_PREFILL path=(\w+) mode=(-?\d+) cells=(\d+)$', text, re.M)
    activations = re.findall(r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=(\d+) vram_bytes=(\d+)$', text, re.M)
    if drafter == 'mtp':
        require(callbacks == [('batched', '2', '8192')]*4 and not activations, 'MTP prompt callbacks differ')
    else:
        require(not callbacks and len(activations) == 1 and activations[0][:2] == ('2046', '1') and
            0 < int(activations[0][2]) <= 2 << 30, 'DFlash2 actual tail/activation differs')
    header = re.findall(decode_footer.HEADER, text, re.M)
    require(len(header) == 1 and tuple(map(int, header[0])) == tuple(sample[k] for k in
        ('prompt_tokens', 'reused_tokens', 'read_tokens', 'generated_tokens', 'drafts_accepted', 'drafts_offered')) and
        type(sample['drafts_offered']) is int and type(sample['drafts_accepted']) is int and
        0 <= sample['drafts_accepted'] <= sample['drafts_offered'] and sample['drafts_offered'] > 0,
        'actual native/client request and draft counters differ')
    clocks = re.findall(r'^strata serve: prompt \d+ tokens = \d+ reused \+ \d+ read in (\d+) ms '
        r'\((\d+(?:\.\d+)?) tok/s\), \d+ generated in (\d+) ms \((\d+(?:\.\d+)?) tok/s\), drafts accepted ', text, re.M)
    require(len(clocks) == 1, 'one completed native phase clock required')
    prompt_ms, prompt_pp, decode_ms, _ = map(float, clocks[0])
    api_ms = sample['prompt_ms']; require(api_ms > .05, 'invalid native/API prompt clock')
    rate = 1000*READ/api_ms
    require(abs(prompt_ms-api_ms) <= .550001 and abs(decode_ms-sample['decode_ms']) <= .550001 and
        abs(sample['prompt_tps']-rate) <= .050001 and
        abs(prompt_pp-rate) <= .050001+1000*READ*.05/(api_ms*(api_ms-.05)),
        'native/API rounded phase clocks or rate differ')
    gdn = [v for v in text.splitlines() if v.startswith('strata prefill: GDN columns=')]
    require(gdn == ['strata prefill: GDN columns=4 workgroups=192 lanes=128 barriers=3'
        ' epilogue=GPU_FP32_same_order scratch=existing_y'], 'actual GDN schedule differs')
    return dict(rows=n, residual_gpu_bytes=n*20480, residual_host_bytes=0, chunk=8192,
        prefill_slots=384, window_mode=10, windows=48, staged_bank_checked_drain=True,
        target_detached_bytes=521472000, MTP_detached_bytes=223227900 if drafter=='mtp' else 0,
        draft_reader='mtp4' if drafter=='mtp' else 'absent-dflash2',
        DFlash2_tail_rows=2046 if activations else 0, nominal_MTP_callback_cells=n if callbacks else 0,
        native_prompt_ms_rounded=prompt_ms, native_prompt_pp_rounded=prompt_pp,
        prompt_includes_repeated_DFlash2_activation=bool(activations), GDN_columns=4, GDN_barriers=3,
        score_GEMM_requested=True, score_GEMM_command_dispatch_observed=False,
        physical_resource_saturation_qualified=False)


def previous_failure():
    raw = (PREVIOUS / 'record.json').read_bytes(); old = json.loads(raw)
    archived = json.loads(evidence.git('show', FAILURE_COMMIT + ':' + str(RESULT.relative_to(P))))['drafter_long_admission_v2']
    require(old['stage'] == archived['stage'] == 'failed' and not old['passed'] and
        not old['ownership_closed'] and len(old['processes']) == 1 and
        old['failure_diagnosis'] == archived['failure_diagnosis'] and
        old['failure_diagnosis']['native_exit_code'] == 0 and
        old['failure_diagnosis']['benchmark_exit_code'] == 1 and
        old['processes'][0]['commands'][0]['direct_child_reaped'] and
        old['processes'][0]['commands'][0]['session_empty'], 'preserve the original model/count/capacity failure')
    return dict(path=str(PREVIOUS / 'record.json'), sha256=hashlib.sha256(raw).hexdigest(),
        evidence_commit=FAILURE_COMMIT, source_commit=old['source_commit'],
        GPU_executed=True, native_executed=True, qualified_ownership_closed=False, status='failed',
        earlier_host_failure_evidence='1fa4b829af8e8d34aee0015bd238a56784101f74')


def source_reviews():
    result = {}
    for key in REVIEW_KEYS:
        row = heads.model.validation.committed(heads.model.validation.INDEX, (key,))
        require(row.get('long_benchmark_admission') is True, 'source/counter review corrections not admitted:' + key)
        pin = row['native_review']; require(digest(pin['path']) == pin['sha256'], 'review digest changed')
        result[key] = {k:row[k] for k in ('native_review', 'long_benchmark_admission', 'admission_corrections')}
    return result


def committed():
    return heads.model.validation.committed(RESULT, (KEY,))


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('sources', 'qualification')}
    out['source_inputs'] = dict(files=len(row['sources']), sha256=fingerprint(row['sources']))
    out['qualification_record'] = dict(path=str(heads.ROOT / 'record.json'),
                                      canonical_sha256=fingerprint(row['qualification']))
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def sources(revision):
    names = evidence.git('ls-tree', '-r', '--name-only', revision, '--',
                         'serve', 'strata_control', 'tools', 'bench.sh',
                         'data/expert-profile.bin').decode().splitlines()
    names = [n for n in names if n in ('bench.sh', 'data/expert-profile.bin') or
             n.endswith('.py') and '/hip/' not in n and '/cuda/' not in n]
    values = {n: hashlib.sha256(raw).hexdigest() for n, raw in heads.source.blobs(P, revision, names)}
    require(str(SELF.relative_to(P)) in values and all(digest(P / n) == h for n, h in values.items()),
            'uncommitted or untracked executed benchmark/controller input')
    return values


def current(row):
    require(committed() == compact(row) and sources(row['source_commit']) == row['sources'],
            'commit exact benchmark plan and unchanged source')
    heads.verify(row['qualification'])
    require(row['qualification']['passed'] and row['qualification']['ownership_closed'] and
            not row['qualification'].get('fresh_faults'), 'closed native ownership qualification required')
    require(previous_failure() == row['previous_failure'] and source_reviews() == row['source_reviews'],
            'failed original plan or source/counter review changed')
    alias = row['asset_alias']
    require(Path(alias['path']).is_symlink() and os.readlink(alias['path']) == alias['link'] and
            str(Path(alias['path']).resolve()) == alias['target'], 'drafter asset alias changed')
    manifest = python_runtime.read_manifest(row['python_contract'])
    python_runtime.validate_manifest(manifest)
    for plan in row['plans']:
        env = plan['environment']; python_runtime.require_environment(env)
        require(python_runtime.selection(env) == row['python_selected'] and
                all(digest(Path(plan['root']) / n) == h for n, h in plan['generated'].items()),
                'generated config/native plan or Python selection changed')
    return manifest


def prepare():
    require(not ROOT.exists(), 'fresh prospective cohort only;no retry')
    previous = previous_failure(); reviews = source_reviews()
    qualified = json.loads((heads.ROOT / 'record.json').read_text())
    heads.verify(qualified, full=True)
    reference = json.loads((heads.model.REFERENCE / 'record.json').read_text())
    revision = evidence.git('rev-parse', 'HEAD').decode().strip()
    pins = sources(revision)
    alias = Path.home() / '.local/share/strata-sycl/dflash2'
    require(alias.is_symlink(), 'explicit existing stock-drafter asset alias required')
    row = dict(stage='prepared', source_commit=revision, sources=pins, qualification=qualified,
        previous_failure=previous, source_reviews=reviews,
        binary=qualified['native_build']['binaries']['strata'],
        native_source_commit=qualified['native_build']['source_commit'],
        native_validation_probe=qualified['probe'], boot=qualified['boot'], driver=qualified['driver'],
        python_contract=reference['python_contract'], python_selected=reference['python_selected'],
        asset_alias=dict(path=str(alias), link=os.readlink(alias), target=str(alias.resolve()),
            reason='Shared frontend data-dir convention points to the already verified stock pack beside the current model;no model bytes moved or duplicated.'),
        template=reference['template'], plans=[], host=[], processes=[],
        passed=False, complete=False, ownership_closed=True, active=False,
        budgets=dict(wall=1200, cpu=14400, rss_cap=104 << 30, text_cap=16 << 20,
                     file_cap=32 << 20, total_cap=64 << 20),
        plan=dict(question='Can both named drafters run the existing tuned8192 long-prefill route through the current production frontend?',
            order='Two fresh native/bench processes:MTP then DFlash2. Two exact32769-token thinking requests each,24output budget. Tiny benchmark warmup and count-only preflight are outside timed long samples;first long use and subsequent long request stay separate.',
            constants='Same f633 native,65536 INT8 resident context,256 expert-cache budget,5workers,8192chunk,spec4,greedy,suffix0,zero caches/adaptation,throughput BF16/native16,GDN16,mode10 mixed/split scratch,384packed bank,score GEMM requested,head retirement1. Force owned buffers with --no-prefill-borrow;MTP expert release/cache scratch0in BOTH arms. V2measured8192buffers2212MiB versus1122MiB free,then automatic1536chunks. This prospectively halves context and reduces cache768->256 to fund the fixed buffers and verifier/drafter headroom. Actual8192/full-GPU/bank/drain admission must pass,without fallback. This is not best-MTP tuning,a speed comparison to retained fast references,or131072/262144capacity qualification.',
            routing='Named configuration must yield exactly committed actual engine arguments at host admission and native exec. No Auto fallback,restart,benchmark-specific model renaming or forged capability markers. Verify native DFlash2 on/absent MTP versus MTP callbacks and per-request generated/accepted/offered counters.',
            acceptance='Each long sample has exactly32769actual/read tokens,zero reuse,thinking/reasoning,1..24generated,finite engine PP/prompt and client TTFT,valid complete journal. Actual BF16 layer-major8192/full-GPU residual path,quiet validation,pre-main proof,qualified live providers,normal benchmark AND native exit0,reaped empty session,unchanged models/source/runtime and no fresh driver fault are required.',
            decision='Parameter validation and STRATA_TRACE stage progress remain charged;not clean speed/adoption or sustained TG evidence. Keep first/subsequent PP/time/TTFT/counters separately. DFlash2 is reset and loaded/bound for EVERY fresh long prompt,and that cost belongs to prompt_ms;first target verify belongs to decode/TTFT. Generic draft offers=sum(T-1),accepted=leading a before final output clipping;not emitted drafted tokens. 32K does not exercise mode10 rollover or split. Next is matched >=1K-thinking-output long-generation and repeated long/agent comparisons,then physical262144/RESTORE. Hardware limiting resource remains unknown.',
            stopping='Stop at first input,ownership,route,counter,clock,validation,fault or budget failure;no retry,replacement,speed-based stop,hash refresh or relaxed gate. Commit each closed process before the next.',
            retention='Keep compact samples/actual arguments/source/config/limits. Retire successful raw logs/journals/caches after evidence verification;no new native engine copy.'))
    ROOT.mkdir(mode=0o700)
    original = reference['environment_template']
    for index, drafter in enumerate(ORDER):
        root = ROOT / drafter; root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root / name).mkdir(mode=0o700)
        env = {k: v.replace(original['STRATA_MEASUREMENT_ROOT'], str(root)) for k, v in original.items()}
        # This admission is observed, not clean speed. Presence enables tracing.
        env['STRATA_TRACE'] = '1'
        env.pop('STRATA_DECODE_TIMING', None)
        for k in env:
            if '_PROFILE' in k or '_AUDIT' in k: env[k] = '0'
        env.update(STRATA_MEASUREMENT_ROOT=str(root), TMPDIR=str(root / 'tmp'),
            XDG_CACHE_HOME=str(root / 'cache'), PYTHONPYCACHEPREFIX=str(root / 'cache/python'),
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'),
            STRATA_PREFILL_RELEASE_MTP_EXPERTS='0', STRATA_SYCL_CACHE_SCRATCH_LEASE='0',
            STRATA_POOL_SPIN_US='20000')
        pack = Path(reference['configuration']['tokenizer']).parent
        token_files = ('vocab.json','merges.txt','token_type.json','tokenizer.json','chat_template.jinja')
        token_pins = {name:digest(pack/'tokenizer'/name) for name in token_files}
        env.update(STRATA_BENCH_REAL_PACK=str(pack), STRATA_BENCH_REAL_PACK_SHA256=hashlib.sha256(
            json.dumps(token_pins,sort_keys=True,separators=(',',':')).encode()).hexdigest())
        require(not env.get('LD_PRELOAD') and env['CUDA_MODULE_LOADING'] == 'EAGER' and
            env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0' and
            env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1', 'qualified immutable native safety required')
        config = copy.deepcopy(reference['configuration'])
        for flag, value in (('--expert-cache','256'),('--max-context','65536'),('--kv-resident','65536')):
            config['args'][config['args'].index(flag)+1] = value
        config['args'].append('--no-prefill-borrow')
        config.update(backend='sycl', family='qwen', drafting=drafter,
                      user_args=['--spec', '4', '--draft-conf', '0', '--suffix-draft', '0'],
                      log=str(root / 'engine.stderr'))
        write(root / 'config.json', config)
        write(root / 'plan.json', dict(binary=row['binary'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        row['plans'].append(dict(index=index, drafter=drafter, root=str(root), environment=env,
            generated={n: digest(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--prefill-exact-tokens', str(READ), '--think', '--runs', str(RUNS), '--profile-request-gates',
                '--profile-startup-gate',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
                '--out', str(root / 'bench')]))
    save(row)
    print('Prepared named-drafter tuned long admission;commit before host probes.', flush=True)


def route_config(path):
    """Production argument construction only; no mocked hardware or native exec."""
    sys.path.insert(0, str(P))
    from serve import server
    cfg = json.loads(Path(path).read_text())
    args = server.engine_args(cfg, exe=cfg['exe'], announce=False)
    def value(flag): return args[args.index(flag) + 1] if flag in args else None
    require(all(value(k) == v for k, v in {'--spec':'4', '--prefill':'8192',
        '--max-context':'65536', '--kv-resident':'65536', '--expert-cache':'256', '--suffix-draft':'0',
        '--pool-workers':'5', '--prefill-strategy':'throughput', '--short-read':'0'}.items()) and
        args.count('--no-prefill-borrow') == 1 and cfg['family'] == 'qwen' and
        cfg['model_name'] == 'qwen3.8-flash-next-iq3_s-primary',
        'frontend changed frozen placement/shape')
    if cfg['drafting'] == 'dflash2':
        require(value('--draft') == 'dflash2' and value('--draft-fallback') == 'none' and
                value('--draft-conf') == '0' and value('--draft-model') is not None and
                Path(value('--draft-model')).resolve() == heads.model.mask.ring.PACK.resolve() and
                value('--mtp-draft-vocab') == str(Path.home()/'.local/share/strata-sycl/mtp/rt/draft_vocab.bin') and
                '--mtp' not in args,
                'named DFlash2 did not survive production configuration')
    else: require(value('--mtp') == str(Path.home()/'.local/share/strata-sycl/mtp/rt') and
                  '--draft' not in args, 'named MTP changed')
    print(json.dumps(dict(args=args, backend=server.drafting.detect_hardware())), flush=True)


def host():
    row = json.loads((ROOT / 'record.json').read_text()); manifest = current(row)
    require(row['stage'] == 'prepared' and not row['host'], 'fresh host stage only')
    frozen = copy.deepcopy(row)
    allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
    owner = heads.ownership.Owner(ROOT / 'host-owned')
    try:
        for script in ('tools/strata_bench.py', 'serve/server.py'):
            role = Path(script).stem
            argv = [row['python_selected']['path'], '-B', str(python_runtime.SOURCE), 'probe',
                    str(P / script), row['template']['path']]
            entry, out, err = owner.run(role, argv, row['plans'][0]['environment'], P,
                wall=60, cpu=60, rss_cap=1 << 30, text_cap=1 << 20, file_cap=2 << 20, total_cap=8 << 20)
            require(heads.ownership.closed(entry) and entry['exit_code'] == 0 and err.stat().st_size == 0,
                    'fresh current-source Python probe failed')
            meta = json.loads(out.read_text())
            require(meta['executable'] == row['python_selected']['path'], 'host interpreter differs')
            for name in [*meta['modules'], *meta['code_maps']]:
                path = Path(name).resolve()
                require(row['sources'].get(str(path.relative_to(P))) == digest(path)
                    if path.is_relative_to(P) else str(path) in allowed, 'unfrozen host provider:' + str(path))
            row['host'].append(dict(role=role, passed=True, exit_code=0, ownership_closed=True))
        for plan in row['plans']:
            entry, out, _ = owner.run('args-' + plan['drafter'], [row['python_selected']['path'], '-B',
                str(Path(__file__).resolve()), 'route-config', str(Path(plan['root']) / 'config.json')],
                plan['environment'], P, wall=60, cpu=60, rss_cap=1 << 30,
                text_cap=1 << 20, file_cap=2 << 20, total_cap=8 << 20)
            require(heads.ownership.closed(entry) and entry['exit_code'] == 0, 'production argument admission failed')
            actual = json.loads(out.read_text())
            require(actual['backend'][0] == 'sycl', 'actual hardware backend differs')
            plan['native_args'] = actual['args']
            row['host'].append(dict(role='args-' + plan['drafter'], passed=True, exit_code=0, ownership_closed=True))
        entry, out, err = owner.run('count-api', [row['python_selected']['path'], '-B', '-m', 'unittest', '-v',
            'tools.test_strata_bench.RealPackExactCounts'], row['plans'][0]['environment'], P,
            wall=300, cpu=120, rss_cap=2 << 30, text_cap=2 << 20, file_cap=4 << 20, total_cap=8 << 20)
        require(heads.ownership.closed(entry) and entry['exit_code'] == 0 and
            re.search(r'^Ran 1 test in ',err.read_text(),re.M) and
            re.search(r'^OK$',err.read_text(),re.M) and out.read_text().count('EXACT_REAL_PACK ') == 4,
            'real tokenizer/count-only HTTP preflight failed or skipped;no native run allowed')
        row['host'].append(dict(role='count-api', passed=True, exit_code=0, ownership_closed=True,
            real_tokenizer_exact_extents=[73729,73730,81921,81922], model_executed=False))
        current(frozen)
        row['stage'] = 'ready'
    except BaseException as error: row.update(stage='failed', error=repr(error))
    finally:
        row['ownership_closed'] = owner.active is None and all(heads.ownership.closed(v) for v in owner.commands)
        row['host_commands'] = owner.commands; save(row)
    require(row['stage'] == 'ready' and row['ownership_closed'], row.get('error', 'host closure failed'))
    print('Current-source frontend/providers and actual named arguments passed;commit before GPU.', flush=True)


def run(index):
    row = json.loads((ROOT / 'record.json').read_text()); manifest = current(row)
    require(row['stage'] in ('ready', 'running') and len(row['processes']) == index < len(ORDER) and
            all(p['passed'] for p in row['processes']), 'fresh frozen process order required')
    plan = row['plans'][index]; root = Path(plan['root']); env = plan['environment']
    require(not (root / 'execution.json').exists(), 'no repeat of an existing process')
    kernel = heads.model.mask.ring.device.kernel_log
    cursor = next(s.split(': ', 1)[1] for s in kernel('-n', '0', '--show-cursor').splitlines() if s.startswith('-- cursor: '))
    outcome = dict(index=index, drafter=plan['drafter'], passed=False, complete=False, model_executed=False)
    owner = None; windows = []
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def fresh_faults():
        text = kernel('--after-cursor', cursor, '-n', '2048')
        require(len(text.splitlines()) < 2048, 'kernel journal overflow')
        return [s for s in text.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', s)]
    def persist():
        write(root / 'execution.json', dict(outcome, active=bool(owner and owner.active),
                                          commands=owner.commands if owner else []))
    def live(entry):
        claim = json.loads((root / 'engine-start.json').read_text()); pid = claim['native']['pid']
        now = heads.ownership.identity(pid)
        require(now and now['start_ticks'] == claim['native']['start_ticks'] and now['sid'] == entry['pid'] and
            str(Path('/proc', str(pid), 'exe').resolve()) == row['binary']['path'] and
            claim['args'] == ['--serve', *plan['native_args']], 'actual owned native route/arguments differ')
        expected = engine_once.native_environment(dict(native_validation_probe=row['native_validation_probe']), env)
        actual = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
        require(all(actual.get(k.encode()) == v.encode() for k, v in expected.items()) and
            set(actual) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'}, 'native environment changed')
        runtime = heads.model.validation
        import runtime_probe
        maps = runtime_probe.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
        require(maps <= set(row['qualification']['runtime_files']) | {row['binary']['path']} and
            {row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'],
             env['ZE_ENABLE_ALT_DRIVERS']} <= maps, 'unqualified native provider mapping')
        runtime.preload_evidence((root / 'engine.stderr').read_text(errors='replace'), pid)
        python_runtime.check_running(entry['pid'], P / 'tools/strata_bench.py', manifest['selected'], env,
            {v['realpath'] for v in manifest['inventory']['files'].values()}, plan['command'][2:])
        outcome.update(model_executed=True, native_identity=claim['native'], file_backed_executable_mappings=len(maps))
    def interact(io, entry):
        while True:
            line = io.line(900)
            if line.startswith('STRATA_PROFILE_GATE '): break
        require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
            dict(stage='ready',phase='startup',run=-1), 'startup admission gate missing')
        live(entry)
        text = (root/'engine.stderr').read_text(errors='replace')
        chunks = re.findall(r'^strata prefill: FP16 MoE compact .*?chunk=(\d+),',text,re.M)
        free = re.findall(r'^strata serve: (\d+) MiB of VRAM free with everything loaded$',text,re.M)
        require(chunks == ['8192'] and len(free) == 1 and
            int(free[0]) >= (1536 if plan['drafter']=='dflash2' else 512) and
            'chunk\'s prompt buffers need' not in text, 'actual startup chunk or decode/drafter VRAM headroom differs')
        outcome['startup_admission'] = dict(actual_chunk=8192, free_MiB=int(free[0]),
            required_free_MiB=1536 if plan['drafter']=='dflash2' else 512, before_benchmark_warmup=True)
        print(plan['drafter']+' startup8192/headroom admitted',flush=True);io.send(b'continue\n')
        for request in range(RUNS):
            offsets = {}
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(900)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                    dict(stage=stage, phase='prefill-only', run=request), 'unexpected request gate')
                delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
                require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'clock/suspend discontinuity')
                live(entry)
                warnings = root / 'validation.log'
                require(not warnings.exists() or warnings.stat().st_size == 0, 'validation warning/error')
                offsets[stage] = (root / 'engine.stderr').stat().st_size
                print(f'{plan["drafter"]} long request{request} {stage}', flush=True); io.send(b'continue\n')
            windows.append(offsets)
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'benchmark/native closure deadline')
            if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        require(not fresh_faults(), 'fresh fault before native launch')
        owner = heads.ownership.Owner(root / 'owned', persist)
        entry, _, _ = owner.run('bench', plan['command'], env, P, interaction=interact, **row['budgets'])
        require(heads.ownership.closed(entry) and entry['exit_code'] == 0, 'benchmark/native ownership failed')
        outputs = list((root / 'bench').glob('*/results.json')); require(len(outputs) == 1, 'one benchmark result required')
        result = json.loads(outputs[0].read_text()); cohort = result['results']['prefill_only']
        samples, counts = cohort['requests'], cohort['count_preflight']
        require(result['complete'] and not result['results']['failures'] and len(samples) == len(counts) == len(windows) == RUNS,
                'incomplete cohort')
        sys.path.insert(0, str(P / 'tools')); import strata_bench as client
        require(client.verify_request_journal(outputs[0].parent)['valid'], 'invalid request journal')
        end = evidence.native_end(root, row, dict(config=dict(args=plan['native_args'])), row['boot'], entry['pid'])
        raw = (root / 'engine.stderr').read_bytes(); text = raw.decode()
        require(text.count('strata serve: the prompt path allocates its own buffers ') == 1 and
                'prompt path borrows' not in text, 'owned prompt buffer admission differs')
        heads.model.validation.preload_evidence(text, end['native']['pid'])
        require(not any(s in text for s in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')),
                'native runtime failure')
        routes = []
        for sample, count, window in zip(samples, counts, windows):
            require(all(sample[k] is True for k in ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed')) and
                sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == count['counted_input_tokens'] == READ and
                sample['actual_reused_tokens'] == 0 and sample['request_sha256'] == count['body_sha256'] and
                1 <= sample['generated_tokens'] <= 24 and sample['prompt_tps_source'] == 'engine' and
                all(type(sample[k]) in (int,float) and math.isfinite(sample[k]) and sample[k] > 0
                    for k in ('prompt_ms', 'prompt_tps', 'ttft_s', 'total_s')) and sample['ttft_s'] <= sample['total_s'],
                'workload/thinking/clock differs')
            part = raw[window['begin']:window['end']].decode()
            routes.append(request_route(part, plan['drafter'], sample))
        if index == len(ORDER)-1: heads.model.model_pins(row['qualification'], full=True)
        outcome.update(passed=True, complete=True, native_exit_code=end['exit_code'], routes=routes,
            samples=samples, count_preflight=counts, ownership_closed=True, speed_adoption_qualified=False)
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        outcome['ownership_closed'] = not owner or owner.active is None and all(heads.ownership.closed(c) for c in owner.commands)
        outcome['commands'] = owner.commands if owner else []
        try:
            outcome['fresh_faults'] = fresh_faults(); require(not outcome['fresh_faults'], 'fresh GPU fault')
            current(row)
        except BaseException as error: outcome.update(passed=False, finalization_error=repr(error))
        outcome['passed'] = bool(outcome['passed'] and outcome['ownership_closed'])
        row['processes'].append(outcome)
        row.update(stage='running' if outcome['passed'] else 'failed', active=False,
                   ownership_closed=outcome['ownership_closed'])
        if len(row['processes']) == len(ORDER) and outcome['passed']:
            row.update(stage='closed', passed=True, complete=True)
        save(row)
    require(outcome['passed'], outcome.get('error', outcome.get('finalization_error', 'long admission failed')))
    print(plan['drafter'] + ' tuned long admission closed;no sustained TG/default adoption.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) >= 2, 'use prepare|host|run INDEX|route-config FILE')
    if sys.argv[1] == 'route-config':
        require(len(sys.argv) == 3, 'one configuration path'); route_config(sys.argv[2])
    else:
        with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); heads.guards.no_model_owners([])
            if sys.argv[1:] == ['prepare']: prepare()
            elif sys.argv[1:] == ['host']: host()
            elif len(sys.argv) == 3 and sys.argv[1] == 'run': run(int(sys.argv[2]))
            else: raise RuntimeError('unknown stage')
