#!/usr/bin/env python3
"""Current same-engine MTP prefill Q8/FP16 comparison; repository LICENSE.

Main-authored from owned exact-input benchmark controls. The closed 32K
long-generation comparison is recoverable at65fad3a32; it does not qualify
this native generation or sustained decode. Main owns serialized execution.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics
import sys

sys.dont_write_bytecode = True
import compare_prefill_strategies as gdn

model, comparison, shared, build = gdn.model, gdn.comparison, gdn.shared, gdn.build
P, B, RESULT = gdn.P, gdn.B, gdn.RESULT
ROOT, KEY = model.gdn16.ROOT / 'mtp16-speed-v1', 'gdn16_mtp16_model_speed_v1'
READ, RUNS = 117489, 4
ARMS = ('candidate', 'reference', 'reference', 'candidate', 'candidate', 'reference')
CASES = [(READ, arm) for arm in ARMS]
ORDER = [(read, 1) for read, _ in CASES]
CHANGED = {'tools/sycl/mtp_prefill.py', 'tools/sycl/check_prefill_strategy_model.py'}
require, digest, write, compact = model.require, model.digest, model.write, model.compact


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def proof(revision):
    model.configure('gdn16-mtp16-throughput')
    native, _, providers = model.proof()
    data = build.committed(RESULT)
    probe = json.loads((model.ROOT / 'record.json').read_text())
    require(data[model.KEY] == compact(probe) and probe['passed'] and probe['complete']
            and probe['host_passed'] and probe['ownership_closed'] and not probe['active']
            and probe['binary'] == native['binaries']['strata']
            and probe['native_source_commit'] == native['source_commit']
            and probe['outcome']['native_exit_code'] == 0 and not probe['outcome']['fresh_faults']
            and len(probe['outcome']['routes']) == 2
            and all(r['GDN_FP16_storage_calls'] == 144
                    and r['GDN_FP16_storage_tokens'] == 36 * 32768
                    and r['MTP_prefill']['callbacks'] == 4
                    and r['MTP_prefill']['nominal_cells'] == 32768
                    for r in probe['outcome']['routes']), 'closed current GDN16/MTP16 production model check required')
    prior = json.loads((model.gdn16.ROOT / 'model-speed-v2/record.json').read_text())
    require(data['gdn16_model_speed_v2'] == compact(prior) and prior['passed'] and prior['complete']
            and prior['ownership_closed'] and not prior['active']
            and prior['arms']['candidate']['binary'] == native['binaries']['strata'],
            'closed same-engine GDN16 speed basis required')
    recovered = {n: hashlib.sha256(raw).hexdigest()
                 for n, raw in shared.gen.blobs(P, prior['source_commit'], prior['sources'])}
    current = model.pinned_sources(revision)
    require(recovered == prior['sources'] and set(current) == set(recovered)
            and {n for n in current if current[n] != recovered[n]} == CHANGED,
            'undeclared benchmark/controller change since closed speed basis')
    def draft_function(rev):
        text = shared.gen.git('show', rev + ':src/prefill/prefill.cpp').decode()
        return text[text.index('bool Prefill::draft_kv('):text.index('bool Prefill::set_peer(')].encode()
    original = draft_function('3b4998f326101fa913788fd368e1674b111dea03')
    require(original == draft_function(native['source_commit']), 'previously executed MTP function changed')
    return native, providers, prior, probe, dict(bytes=len(original), sha256=hashlib.sha256(original).hexdigest())


def native_context(row, plan):
    require((plan['read'], plan['kernel']) == CASES[plan['index']] and plan['heads'] == 1,
            'undeclared process/placement')
    native = row['arms'][plan['kernel']]
    return dict(binary=native['binary'], native_source_commit=native['native_source_commit'],
                native_validation_probe=row['native_validation_probe'], boot=row['boot'])


def environment(row, plan):
    native_context(row, plan)
    env = comparison.environment(row, plan)
    env.update(STRATA_SYCL_PREFILL_GDN_BARRIERS3='1', STRATA_SYCL_PREFILL_GDN_FP16='1',
               STRATA_SYCL_PREFILL_GDN_CHUNK64='0', STRATA_MTP_BATCH_F16=str(int(plan['kernel'] == 'candidate')),
               STRATA_SNAPSHOT_VERIFY='1')
    require(env['STRATA_PREFILL_LAYER_MAJOR'] == '1'
            and env['STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL'] == 'original'
            and env['STRATA_SYCL_PREFILL_GDN_COLUMNS'] == '4'
            and not env.get('STRATA_DRAFT_TIMING') and not env.get('STRATA_TRACE')
            and not env.get('LD_PRELOAD') and env['CUDA_MODULE_LOADING'] == 'EAGER',
            'selected route or observer envelope differs')
    return env


def invariant(row, plan):
    value = dict(config=json.loads((Path(plan['root']) / 'config.json').read_text()),
                 environment=environment(row, plan))
    value['environment']['STRATA_MTP_BATCH_F16'] = '<arm>'
    return hashlib.sha256(json.dumps(value, sort_keys=True).replace(plan['root'], '<process>').encode()).hexdigest()


def request_route(text, plan):
    # The unchanged GDN path is FP16 in BOTH MTP arms.
    result = gdn.request_route(text, dict(plan, kernel='candidate'))
    result.update(executable_arm=plan['kernel'], MTP_projection_FP16=plan['kernel'] == 'candidate',
                  MTP_prefill=model.mtp_route(text, plan['read'] - 1),
                  dispatch_proof='Both arms use observed long native16/GDN3 headers and immutable GDN16 selector1. '
                  'MTP callback markers prove successful batched mode2 and nominal chunk cells. The unchanged '
                  'draft_kv source and immutable MTP_BATCH_F16 selector determine Q8/FP16 for needed work; '
                  'internal needed rows, batch shape and individual dispatches are not observed here.')
    return result


validate_model_text = gdn.validate_model_text


def decision(processes):
    require(len(processes) == len(CASES) and all(p['passed'] and p['complete'] and p['ownership_closed']
            and not p['fresh_faults'] and (p['read'], p['kernel']) == CASES[i]
            for i, p in enumerate(processes)), 'complete frozen cohort required')
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes})
            == len(CASES), 'native process reused')
    ratios, ttft, speeds, refs = [], [], [], []
    for offset in (0, 2, 4):
        pair = {p['kernel']: p for p in processes[offset:offset + 2]}
        a, b = pair['candidate'], pair['reference']
        require(all(math.isfinite(p['subsequent_median_ms']) and p['subsequent_median_ms'] > 10000
                    for p in pair.values()), 'invalid process clock')
        ratios.append(a['subsequent_median_ms'] / b['subsequent_median_ms'])
        ttft.append(statistics.median(s['ttft_s'] for s in a['samples'][1:]) /
                    statistics.median(s['ttft_s'] for s in b['samples'][1:]))
        speeds.append(READ * 1000 / a['subsequent_median_ms'])
        refs.append(READ * 1000 / b['subsequent_median_ms'])
    gm = math.exp(statistics.mean(map(math.log, ratios)))
    return dict(candidate_over_reference_prompt_time=ratios, geometric_mean_time_ratio=gm,
                reproducible_prompt_gain=all(v < 1 for v in ratios) and gm < 1,
                minimum_percentage_gain_required=False, candidate_over_reference_TTFT=ttft,
                candidate_subsequent_process_pp=speeds, reference_subsequent_process_pp=refs,
                all_candidate_processes_at_least1000=all(v >= 1000 for v in speeds),
                inference_adopted=False, sustained_decode_qualified=False,
                physical262144_qualified=False, physical_resource_saturation_qualified=False)


def selfcheck():
    model.configure('gdn16-mtp16-throughput')
    controls = model.route_controls()
    for rows in (32768, 117488):
        chunks = [min(8192, rows - i) for i in range(0, rows, 8192)]
        text = ''.join(f'strata serve: DRAFT_PREFILL path=batched mode=2 cells={n}\n' for n in chunks)
        model.mtp_route(text, rows)
        for wrong in ('\n'.join(text.splitlines()[:-1]) + '\n', text + text.splitlines()[0] + '\n',
                      text.replace('batched', 'token'), text.replace('mode=2', 'mode=0'),
                      text.replace('cells=' + str(chunks[-1]), 'cells=' + str(chunks[-1] - 1))):
            try: model.mtp_route(wrong, rows)
            except RuntimeError: pass
            else: raise RuntimeError('incomplete/changed MTP callback coverage accepted')
    def item(i):
        read, arm = CASES[i]
        return dict(read=read, kernel=arm, passed=True, complete=True, ownership_closed=True,
                    fresh_faults=[], native_identity=dict(pid=i + 1, start_ticks=i + 100),
                    subsequent_median_ms=110000 if arm == 'candidate' else 120000,
                    samples=[dict(ttft_s=130 if arm == 'candidate' else 125) for _ in range(RUNS)])
    rows = [item(i) for i in range(len(CASES))]
    good = decision(rows)
    require(good['reproducible_prompt_gain'] and good['all_candidate_processes_at_least1000']
            and all(x > 1 for x in good['candidate_over_reference_TTFT']), 'PP/TTFT/target decisions coupled')
    rows[0]['subsequent_median_ms'] = 120001
    require(not decision(rows)['reproducible_prompt_gain']
            and not decision(rows)['all_candidate_processes_at_least1000'], 'one slow pair accepted')
    rows[0] = item(0)
    rows[1]['native_identity'] = rows[0]['native_identity']
    try: decision(rows)
    except RuntimeError: pass
    else: raise RuntimeError('duplicate native identity accepted')
    return dict(passed=True, GPU_executed=False, model_route_controls=controls,
                additional_callback_positive=2, additional_callback_negative=10, decision_controls=3)


def prepare():
    controls = selfcheck()
    revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    native, providers, prior, probe, source_contract = proof(revision)
    require(not ROOT.exists() and KEY not in build.committed(RESULT), 'fresh comparison only')
    contract, manifest = model.python_manifest()
    require(contract == prior['python_contract'] and manifest['selected'] == prior['python_selected'],
            'frontend provider differs')
    assets = shared.models.model_pins(prior['configuration'], prior['model_assets'], full=True)
    ROOT.mkdir(mode=0o700)
    write(ROOT / 'runtime-inputs.json', providers)
    row = dict(stage='prepared', source_commit=revision,
        arms={a: dict(binary=native['binaries']['strata'], native_source_commit=native['source_commit']) for a in ARMS},
        native_validation_probe=prior['native_validation_probe'], configuration=copy.deepcopy(prior['configuration']),
        environment_template=copy.deepcopy(prior['environment_template']), python_contract=contract,
        python_selected=manifest['selected'], template=prior['template'], boot=prior['boot'], driver=prior['driver'],
        model_assets=assets, sources=model.pinned_sources(revision),
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'), sha256=digest(ROOT / 'runtime-inputs.json'), files=len(providers)),
        qualification_reuse=dict(native_source=native['source_commit'], current_MTP16_probe=probe['source_commit'],
            clean_GDN16_basis=prior['source_commit'], recovered_controller_files=len(prior['sources']),
            declared_controller_changes=sorted(CHANGED), previously_executed_draft_function=source_contract,
            scope='Same nine actual ELFs/native inputs/recipe and qualified providers; no new native build. '
                  'Current two32K traced/validated GDN16/MTP16 entries pass. Broader adoption remains pending.'),
        plans=[], processes=[], request_hashes={}, controls=controls, host_passed=False, passed=False,
        complete=False, active=False, ownership_closed=True, inference_adopted=False,
        whole_bytes=512 << 20, budgets=dict(wall=1800, cpu=5400, rss_cap=104 << 30, text_cap=16 << 20,
                                          file_cap=16 << 20, total_cap=64 << 20),
        plan=dict(question='Does existing FP16 draft-KV projection improve current117K long prefill over Q8 with GDN16 fixed?',
            attribution='Closed current resource profile has36Q80 prompt-product kernels/7.035882796s summed service. '
                        'draft_kv is unchanged from the previously executed3b function. This is an optimization candidate, '
                        'not a physical ceiling, warmed cost or predicted end-to-end gain.',
            order='6fresh processes,3pairs at117489read:candidate/reference,reference/candidate,candidate/reference. '
                  'Four exact-input thinking-on requests per process;first use separate,subsequent3median is the process unit.',
            differences='Only immutable STRATA_MTP_BATCH_F16 differs. Both arms fix GDN16on/GDN3,throughput strategy, '
                        'native16BF16 residuals,131072INT8resident context,8192chunk,128staged slots,checked split, '
                        'head retirement1,1256decode slots,5workers,spec4,Top-k0,originalDown and qualified runtime safety. '
                        'Both enable SNAPSHOT_VERIFY for one nominal MTP callback marker per chunk;prompt and conversation '
                        'caches are0,excluding its heavy verification branch. No trace/draft-timing/API observers.',
            acceptance='Every existing request/body/actual-token/reuse/thinking/clock/route/provider/owner/native-exit/quiet-validation/fault '
                       'gate must pass. MTP requires batched mode2 and8192/tail nominal callback conservation. '
                       'All3paired subsequent prompt-time ratios<1 and GM<1 allow follow-up,without a minimum percentage floor. '
                       'Judge TTFT separately;1000target requires all3candidate process medians>=1000. '
                       'Keep all24samples and actual short draft acceptance;no sustained TG or inference adoption.',
            stopping='Stop on first failed source/provider/owner/clock/route/validation/fault/budget gate. No speed-based early stop, '
                     'retry,replacement,exclusion or hash refresh. Commit each process before the next;full model hashes before/aftercohort.',
            limits='Minimal callback-marker timing with identical observers,not zero-observer/overhead proof. '
                   'Nominal callbacks do not expose internal needed rows,batch size or individual kernels. '
                   'FP16 changes draft cache values and may change acceptance;decode code/config remain unchanged but handoff and '
                   'repeated>=1K strong/weak-MTP decode,agent/task quality,TTFT option,physical262144/RESTORE remain separate gates. '
                   'One117K point does not update a length curve or generalize mixed placement families.',
            retention='Keep compact individual samples,source/config/qualification/route checks and decisions. '
                      'Retire successful raw logs,journals and ownership output after closed evidence verification. '
                      'No new build generation;retain current9ELFs and the named8ELF fallback for remaining qualification.'))
    for index, (read, arm) in enumerate(CASES):
        root = ROOT / str(index)
        root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root / name).mkdir()
        config = copy.deepcopy(row['configuration'])
        config['log'] = str(root / 'engine.stderr')
        for flag in ('--prompt-cache', '--conversation-cache-mib'):
            require(config['args'][config['args'].index(flag) + 1] == '0', 'callback marker requires disabled snapshot caches')
        write(root / 'config.json', config)
        write(root / 'plan.json', dict(binary=native['binaries']['strata'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        plan = dict(index=index, kernel=arm, read=read, heads=1, root=str(root),
            generated={n: digest(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                     '--prefill-exact-tokens', str(read), '--think', '--runs', str(RUNS), '--profile-request-gates',
                     '--workload-id', 'native_mtp16_20261101', '--fail-fast-engine-error', '--ready-timeout', '600',
                     '--out', str(root / 'bench')])
        plan['environment_sha256'] = comparison.fingerprint(environment(row, plan))
        row['plans'].append(plan)
    row['invariant_signature'] = invariant(row, row['plans'][0])
    require(all(invariant(row, plan) == row['invariant_signature'] for plan in row['plans']),
            'undeclared arm difference')
    save(row)
    print('Prepared6matched current-engine MTP16/Q8 processes;commit before frontend probes.', flush=True)


def current(row):
    require(build.committed(RESULT)[KEY] == compact(row), 'commit exact campaign state')
    native, providers, _, _, _ = proof(row['source_commit'])
    require(row['arms'] == {a: dict(binary=native['binaries']['strata'], native_source_commit=native['source_commit']) for a in ARMS},
            'native generation changed')
    require(model.pinned_sources(row['source_commit']) == row['sources'], 'frozen model/controller inputs changed')
    inv = row['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'] and json.loads(Path(inv['path']).read_text()) == providers
            and len(providers) == inv['files'], 'provider inventory changed')
    require(digest(row['template']['path']) == row['template']['sha256']
            and build.guards.driver_guard() == row['driver'], 'template/platform changed')
    require(build.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole output budget')
    shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    contract, manifest = model.python_manifest()
    require(contract == row['python_contract'] and manifest['selected'] == row['python_selected'], 'frontend provider changed')
    require(all(invariant(row, plan) == row['invariant_signature'] for plan in row['plans']), 'arm invariant changed')
    return providers, manifest


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.guards.no_model_owners([])
        args = sys.argv[1:]
        if args == ['selfcheck']: print(json.dumps(selfcheck()))
        elif args == ['prepare']: prepare()
        elif args == ['host']: comparison.host(sys.modules[__name__])
        elif len(args) == 1 and args[0].isdigit() and int(args[0]) < len(CASES):
            comparison.run(int(args[0]), sys.modules[__name__])
        else: raise RuntimeError('mtp_prefill.py selfcheck|prepare|host|INDEX0..5')
