"""Measure aligned KV reuse through the official exact-count DFlash2 prefill workload."""

from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_prefill_kv_stage_model as gate

bank, basis, require = gate.bank, gate.basis, gate.require
generation, build = gate.generation, gate.build
comparison = basis.comparison
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_kv_stage_speed_v4'
ROOT = bank.B / KEY.replace('_', '-')
SPLIT_KEY = 'dflash2_prefill_kv_stage_split_speed_v1'
SPLIT_ROOT = bank.B / SPLIT_KEY.replace('_', '-')
SPEED_64K = '6e5bb20067e0d3f8abc45f547d226f2f16e1f4f3'
MODEL_PROOF = '91186794c8624121f245067433c387785dffd757'
FINGERPRINT_REFUSAL = '16755c8184acb52fc6b7aba7a8d86113628de70d'
PROFILE_PATH_STOP = 'fe958812755d6dce0faded3dd41073edf7371196'
SERIALIZATION_STOP = '2da6d6261ff7328136a63ef055151b954563e3a8'
PROFILE_SOURCE = 'data/expert-profile.bin'
READ, RUNS, HEADS = 65537, 3, 0
ORDER = ('reload', 'retain', 'retain', 'reload', 'reload', 'retain')
WORKLOAD_ID = 'native_gdn_20261018'
KV = generation.v2.state.kv
SELECTOR = generation.v2.state.KV


def baseline(read=READ):
    previous = bank.previous.historic(FINGERPRINT_REFUSAL, 'dflash2_prefill_kv_stage_speed_v1')
    refusal = previous['prelaunch_refusal']
    require(previous['stage'] == 'failed' and previous['ownership_closed'] and not previous['active']
            and not previous['processes'] and not refusal['native_execution_attempted']
            and refusal['plan_environment_sha256'] != refusal['runner_environment_sha256'],
            'recover the original prelaunch environment fingerprint refusal')
    interrupted = bank.previous.historic(PROFILE_PATH_STOP, 'dflash2_prefill_kv_stage_speed_v2')
    stopped = interrupted['profile_path_stop']
    profile = bank.P / PROFILE_SOURCE
    require(interrupted['stage'] == 'failed' and not interrupted['active']
            and stopped['scope_empty'] and stopped['direct_child_reaped'] and stopped['fresh_faults'] == []
            and stopped['actual'] != stopped['expected'] and bank.sha(profile) == stopped['sha256'],
            'recover the interrupted comparison and verify the identical local ranking profile')
    timing = bank.previous.historic(SERIALIZATION_STOP, 'dflash2_prefill_kv_stage_speed_v3')
    replay = timing['serialization_replay']
    require(timing['stage'] == 'failed' and not timing['active'] and timing['ownership_closed']
            and len(timing['processes']) == 2 and timing['processes'][0]['passed']
            and not timing['processes'][1]['passed'] and replay['original_failed']
            and replay['requests_matched'] == RUNS and replay['matched_native_work_after_JSON']
            and all(p['native_exit_code'] == 0 and p['fresh_faults'] == [] for p in timing['processes']),
            'recover the original timing stop and successful footer replay')
    base = generation.sync_baseline(historical=True)
    native = bank.previous.read(build.DFLASH2_KV_ROOT / 'record.json')
    build.context('dflash2-kv-stage').check(native)
    proof = bank.previous.historic(MODEL_PROOF, gate.KEY)
    native_hash = bank.signature(bank.builder.compact(native))
    require(proof['stage'] == 'closed' and proof['complete'] and proof['passed']
            and proof['ownership_closed'] and not proof['active']
            and proof['comparison'] == dict(matched=True, requested_prefix_H2D_bytes_removed=gate.EXPECTED_H2D_REDUCTION)
            and proof['plan']['binary'] == native['binaries']['strata']
            and proof['plan']['build_sha256'] == native_hash
            and proof['plan']['boot'] == base['boot'] and proof['plan']['driver'] == base['driver']
            and proof['plan']['runtime_manifest'] == base['runtime_manifest']
            and proof['plan']['drafter_assets'] == base['dflash2_assets']
            and list(proof['outcomes']) == list(gate.ARMS)
            and all(case['passed'] and case['native_exit_code'] == 0 and not case['fresh_faults']
                    for case in proof['outcomes'].values()),
            'require the closed same-executable 32K and live prefix comparison')
    row = dict(base, binary=native['binaries']['strata'], native_source_commit=native['native_source_commit'],
               native_build_sha256=native_hash, stage_model_evidence=MODEL_PROOF,
               preceding_attempts=[FINGERPRINT_REFUSAL, PROFILE_PATH_STOP, SERIALIZATION_STOP],
               ranking_profile=dict(path=str(profile), sha256=stopped['sha256'], old_path=stopped['actual']))
    if read == gate.SPLIT_READ:
        speed = bank.previous.historic(SPEED_64K, KEY)
        split = basis.validation.committed(bank.RESULT, (gate.SPLIT_KEY,))
        outcome, plan = split['outcome'], split['plan']
        require(speed['passed'] and speed['complete'] and speed['ownership_closed']
                and speed['decision']['repeated_warm_gain'] and speed['binary'] == row['binary']
                and split['stage'] == 'closed' and split['passed'] and split['complete']
                and split['ownership_closed'] and not split['active']
                and outcome['passed'] and outcome['native_exit_code'] == 0 and outcome['fresh_faults'] == []
                and all(plan[k] == row[k] for k in ('binary', 'native_source_commit', 'native_build_sha256',
                                                   'boot', 'driver', 'runtime_manifest'))
                and plan['drafter_assets'] == row['dflash2_assets']
                and plan['validation'] == row['native_validation_probe']
                and plan['split_reference'] == gate.SPLIT_REFERENCE
                and outcome['split']['store']['gpu_rows'] == read - 1
                and outcome['kv_stage']['reused_pages'] > 0,
                'require the same-ELF 64K gain and the completed 117K candidate trace')
        row.update(split_model_sha256=bank.signature(split), previous_speed=SPEED_64K)
    return row


def configuration(row, plan):
    cfg = generation.configuration(row, plan)
    args = cfg['args']
    profile = row['ranking_profile']
    require(args.count('--expert-profile') == 1, 'one explicit profile is required')
    index = args.index('--expert-profile') + 1
    require(args[index] == profile['old_path'], 'reference profile path changed')
    args[index] = profile['path']
    return cfg


def environment(row, plan, read=READ):
    if read == gate.SPLIT_READ:
        env = generation.v2.launch_environment(row, dict(plan, arm='candidate'), split=True)
        require(env['STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR'] == '1'
                and env['STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS'] == '128'
                and env[generation.v2.UPLOAD] == 'batched'
                and 'STRATA_COMMIT_SYNC' not in env and env.get('STRATA_SYCL_VERIFY_OVERLAP', '1') != '0',
                'require the batched 128-slot split route and independently synchronized commit')
        env.update(STRATA_SYCL_DECODE_Q6_DPAS_HEAD='1', STRATA_PREFILL_RELEASE_MTP_EXPERTS='0',
                   STRATA_COMMIT_SYNC='1', STRATA_WATCHDOG_S='240')
    else:
        env = generation.sync_environment(row, plan)
    return dict(env, **{SELECTOR: gate.ARMS[plan['arm']]})


def command(plan, read=READ):
    root = Path(plan['root'])
    return ['/bin/sh', str(bank.P / 'bench.sh'), '--config', str(root / 'config.json'),
            '--port', str(plan['port']), '--prefill-exact-tokens', str(read), '--think',
            '--runs', str(RUNS), '--profile-request-gates', '--workload-id', WORKLOAD_ID,
            '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')]


def request_route(text, plan, *, direct=False):
    read = plan['read']
    require('strata loan audit:' not in text, 'expected quiet prefill')
    if read == gate.SPLIT_READ:
        route = gate.split.split_route(text, read, direct=direct)
        store, slots = route['store'], route['prefill_slots']
    else:
        require('strata MoE split complete:' not in text, 'expected ordinary prefill')
        store = gate.model.restore.require_store(text, dict(direct=direct), plan)
        require(store['gpu_rows'] == read - 1 and store['host_rows'] == 0 and store['window_mode'] == 10,
                '64K must admit the ordinary all-GPU residual route')
        route, slots = {}, 384
        banks = re.findall(r'^strata MoE expert bank: requested_slots=384 effective_slots=384 staged=1 '
                           r'host_slab_bytes=\d+ device_bytes=1022361600 residual_gpu_rows=65536 ', text, re.M)
        require(len(banks) == 1, '384-slot bank differs')
    drain = re.findall(r'^strata MoE expert bank complete: uploads=(\d+) requested_H2D_bytes=(\d+) '
                       r'permanent_hits=(\d+) permanent_slots=(\d+) total_slots=(\d+) final_drain=checked$', text, re.M)
    jobs = re.findall(r'^strata MoE accumulator window: windows=48 dequants=(\d+) jobs=(\d+) '
                      rf'source_H2D_bytes=(\d+) GPU_resident_windows_only=1 operator_max=8192 maximum_rows={read - 1}$', text, re.M)
    tap = re.findall(r'^strata DFlash2: eager, prompt_tail_rows=2046 cold_prefix=0 vram_bytes=(\d+)$', text, re.M)
    header = re.findall(generation.clock.footer.HEADER, text, re.M)
    require(len(drain) == len(jobs) == len(tap) == len(header) == 1 and int(tap[0]) > 0,
            'DFlash2 tap, expert bank or request work differs')
    pages = KV.check_clean_request(text, 0, read - 1, 8192, reuse=plan['arm'] == 'retain',
                                   tap_layers=gate.TAPPED_QSA_LAYERS)
    pages['requested_prefix_H2D_bytes'] = pages['reloaded_pages'] * KV.PAGE * KV.CELL_BYTES
    bank_counts = dict(zip(('uploads', 'requested_H2D_bytes', 'permanent_hits', 'permanent_slots', 'total_slots'),
                          map(int, drain[0]), strict=True))
    moe_counts = dict(zip(('dequants', 'jobs', 'source_H2D_bytes'), map(int, jobs[0]), strict=True))
    require(slots <= bank_counts['total_slots'] <= (512 if read == gate.SPLIT_READ else slots)
            and bank_counts['permanent_slots'] <= bank_counts['total_slots']
            and bank_counts['uploads'] + bank_counts['permanent_hits'] == moe_counts['dequants'],
            'expert bank work does not account for every dequantization')
    return dict(route, store=store, bank=bank_counts, jobs=moe_counts, tap_vram_bytes=int(tap[0]), kv=pages,
                request_work=list(map(int, header[0])))


def validate_model_text(text, plan):
    profiles = generation.v2.continuation.standard.profiles
    profiles.check_profile_startup(text, 'original')
    require(text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
            and not re.search(r'(?m)^strata mtp: .*loaded|strata loan audit:', text)
            and not generation.suite.gen.draft_rows(text), 'unexpected target head, MTP layer or observer')


def verify_model_assets(row, config):
    # The retained target manifest also describes the explicit MTP control.
    # Recover its target file hashes from the original config and verify the
    # DFlash2 pack and vocabulary used by the actual native launch separately.
    comparison.shared.models.model_pins(row['configuration'], row['model_assets'], full=True)
    args = config['args']
    for flag in ('--draft-model', '--mtp-draft-vocab'):
        path = args[args.index(flag) + 1]
        require(bank.sha(path) == row['dflash2_assets'][path], 'DFlash2 asset changed')


def check_work(outcome, row, plan):
    read = plan['read']
    for route, sample in zip(outcome['routes'], outcome['samples'], strict=True):
        require(route['request_work'] == [read, 0, read, sample['generated_tokens'],
                                          sample['drafts_accepted'], sample['drafts_offered']],
                'native and client request work differs')
    require(outcome['clean_speed'] and outcome['native_identity'] not in
            [p['native_identity'] for p in row['processes']], 'require a fresh clean process')
    if row['processes']:
        def work(process):
            return [{k: v for k, v in route.items() if k != 'kv'} for route in process['routes']]
        require(work(outcome) == work(row['processes'][0]), 'work changed outside KV retention')
    outcome['arm'] = plan['arm']


def decision(processes, read=READ):
    first_and_warm, pp, ratios = {}, {}, []
    for p in processes:
        samples = p['samples']
        first_and_warm[p['index']] = dict(first_ms=samples[0]['prompt_ms'], first_ttft_s=samples[0]['ttft_s'],
            warm_ms=p['subsequent_median_ms'], warm_ttft_s=statistics.median(s['ttft_s'] for s in samples[1:]))
    for arm in gate.ARMS:
        pp[arm] = [read * 1000 / first_and_warm[p['index']]['warm_ms'] for p in processes if p['arm'] == arm]
    for a, b in zip(processes[::2], processes[1::2], strict=True):
        reload, retain = (a, b) if a['arm'] == 'reload' else (b, a)
        x, y = (first_and_warm[p['index']] for p in (reload, retain))
        ratios.append({metric: y[metric] / x[metric] for metric in x})
    return dict(process_times=list(first_and_warm.values()), warmed_PP=pp, retain_over_reload=ratios,
                repeated_warm_gain=all(p['warm_ms'] < 1 for p in ratios),
                repeated_1000={arm: all(v >= 1000 for v in rates) for arm, rates in pp.items()},
                full_context_qualified=False, sustained_decode_qualified=False, adopted=False)


def experiment(read=READ):
    require(read in (READ, gate.SPLIT_READ), 'select an audited DFlash2 input extent')
    split = read == gate.SPLIT_READ
    key = SPLIT_KEY if split else KEY
    root = SPLIT_ROOT if split else ROOT
    ctx = generation.experiment(synchronous_commit=True)
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = key, root, THIS, ORDER
    ctx.baseline = lambda: baseline(read)
    ctx.environment = lambda row, plan: environment(row, plan, read)
    ctx.command = lambda plan: command(plan, read)
    ctx.configuration = configuration
    placement = 'the batched 128-slot split' if split else 'the ordinary 384-slot bank'
    ctx.protocol = dict(order=list(ORDER), read=read, requests_per_process=RUNS,
        unit='Change only aligned KV retention on the untapped QSA layers.',
        acceptance=f'Require matched exact inputs and all-GPU residuals through {placement}, '
                   'identical work outside KV retention and all per-layer KV counts. Compare first use and '
                   'subsequent PP and client TTFT separately. A warmed speed benefit needs all three paired '
                   'prompt-time ratios below one. Keep any smaller or mixed effects in the comparison.',
        stop='Stop on first source, clock, owner, route, work, fault or budget failure. Keep all eligible '
             'samples. No speed-based exclusions. This short output does not qualify sustained decode.')
    old_pins, old_save, old_current = ctx.pins, ctx.save, ctx.current

    def pins(revision):
        paths = (THIS, gate.THIS, gate.split.THIS, build.__file__, PROFILE_SOURCE)
        relative = [str(Path(path).relative_to(bank.P)) if Path(path).is_absolute() else path for path in paths]
        sources = old_pins(revision)
        for path in relative:
            require((bank.P / path).read_bytes() == basis.validation.git('show', revision + ':' + path),
                    'changed DFlash2 timing input: ' + path)
            sources[path] = bank.sha(bank.P / path)
        return sources

    def normalized(row):
        for plan in row['plans']:
            env = ctx.environment(row, plan)
            del env[SELECTOR]
            yield {key: value.replace(plan['root'], '<process>') for key, value in env.items()}

    def save(row):
        if row['stage'] == 'prepared':
            for plan in row['plans']:
                plan.update(read=read, heads=HEADS)
            row['request_hashes'] = {}
        old_save(row)

    def current(row):
        require(all((p['read'], p['heads']) == (read, HEADS) for p in row['plans'])
                and len({bank.signature(env) for env in normalized(row)}) == 1,
                'planned input size or comparison environment differs')
        return old_current(row)

    ctx.pins, ctx.current, ctx.save = pins, current, save
    runner = SimpleNamespace(ROOT=root, ORDER=[(read, HEADS)] * len(ORDER), RUNS=RUNS,
        current=current, environment=ctx.environment, environment_fingerprint=generation.suite.signature,
        save=save, request_route=request_route,
        after_execution=check_work, decision=lambda rows: decision(rows, read), validate_model_text=validate_model_text,
        verify_model_assets=verify_model_assets)
    ctx.run = lambda index: comparison.run(index, runner)
    return ctx


if __name__ == '__main__':
    split = sys.argv[1:2] == ['split']
    if split:
        sys.argv.pop(1)
    generation.clock.main(experiment(gate.SPLIT_READ if split else READ))
