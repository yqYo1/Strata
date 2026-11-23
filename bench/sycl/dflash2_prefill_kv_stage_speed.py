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
KEY = 'dflash2_prefill_kv_stage_speed_v2'
ROOT = bank.B / KEY.replace('_', '-')
MODEL_PROOF = '91186794c8624121f245067433c387785dffd757'
PREVIOUS_KEY = 'dflash2_prefill_kv_stage_speed_v1'
READ, RUNS, HEADS = 65537, 3, 0
ORDER = ('reload', 'retain', 'retain', 'reload', 'reload', 'retain')
WORKLOAD_ID = 'native_gdn_20261018'
KV = generation.v2.state.kv
SELECTOR = generation.v2.state.KV


def baseline():
    previous = basis.validation.committed(bank.RESULT, (PREVIOUS_KEY,))
    refusal = previous['prelaunch_refusal']
    require(previous['stage'] == 'failed' and previous['ownership_closed'] and not previous['active']
            and not previous['processes'] and not refusal['native_execution_attempted']
            and refusal['plan_environment_sha256'] != refusal['runner_environment_sha256'],
            'recover the original prelaunch environment fingerprint refusal')
    base = generation.sync_baseline()
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
    return dict(base, binary=native['binaries']['strata'], native_source_commit=native['native_source_commit'],
                native_build_sha256=native_hash, stage_model_evidence=MODEL_PROOF,
                previous_prelaunch_refusal=bank.signature(refusal))


def environment(row, plan):
    env = generation.sync_environment(row, plan)
    return dict(env, **{SELECTOR: gate.ARMS[plan['arm']]})


def command(plan):
    root = Path(plan['root'])
    return ['/bin/sh', str(bank.P / 'bench.sh'), '--config', str(root / 'config.json'),
            '--port', str(plan['port']), '--prefill-exact-tokens', str(READ), '--think',
            '--runs', str(RUNS), '--profile-request-gates', '--workload-id', WORKLOAD_ID,
            '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')]


def request_route(text, plan):
    comparison.require('strata loan audit:' not in text and 'strata MoE split complete:' not in text,
                       'expected quiet ordinary prefill')
    store = gate.model.restore.require_store(text, dict(direct=False), plan)
    require(store['gpu_rows'] == READ - 1 and store['host_rows'] == 0 and store['window_mode'] == 10,
            '64K must admit the ordinary all-GPU residual route')
    banks = re.findall(r'^strata MoE expert bank: requested_slots=384 effective_slots=384 staged=1 '
                       r'host_slab_bytes=\d+ device_bytes=1022361600 residual_gpu_rows=65536 ', text, re.M)
    drain = re.findall(r'^strata MoE expert bank complete: uploads=(\d+) requested_H2D_bytes=(\d+) '
                       r'permanent_hits=(\d+) permanent_slots=\d+ total_slots=384 final_drain=checked$', text, re.M)
    jobs = re.findall(r'^strata MoE accumulator window: windows=48 dequants=(\d+) jobs=(\d+) '
                      r'source_H2D_bytes=(\d+) GPU_resident_windows_only=1 operator_max=8192 maximum_rows=65536$', text, re.M)
    tap = re.findall(r'^strata DFlash2: eager, prompt_tail_rows=2046 cold_prefix=0 vram_bytes=(\d+)$', text, re.M)
    header = re.findall(generation.clock.footer.HEADER, text, re.M)
    require(len(banks) == len(drain) == len(jobs) == len(tap) == len(header) == 1 and int(tap[0]) > 0,
            'DFlash2 tap, 384-slot bank or request work differs')
    pages = KV.check_clean_request(text, 0, READ - 1, 8192, reuse=plan['arm'] == 'retain',
                                   tap_layers=gate.TAPPED_QSA_LAYERS)
    pages['requested_prefix_H2D_bytes'] = pages['reloaded_pages'] * KV.PAGE * KV.CELL_BYTES
    return dict(store=store, bank=drain[0], jobs=jobs[0], tap_vram_bytes=int(tap[0]), kv=pages,
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
    for route, sample in zip(outcome['routes'], outcome['samples'], strict=True):
        require(route['request_work'] == [READ, 0, READ, sample['generated_tokens'],
                                          sample['drafts_accepted'], sample['drafts_offered']],
                'native and client request work differs')
    require(outcome['clean_speed'] and outcome['native_identity'] not in
            [p['native_identity'] for p in row['processes']], 'require a fresh clean process')
    if row['processes']:
        def work(process):
            return [{k: v for k, v in route.items() if k != 'kv'} for route in process['routes']]
        require(work(outcome) == work(row['processes'][0]), 'work changed outside KV retention')
    outcome['arm'] = plan['arm']


def decision(processes):
    first_and_warm, pp, ratios = {}, {}, []
    for p in processes:
        samples = p['samples']
        first_and_warm[p['index']] = dict(first_ms=samples[0]['prompt_ms'], first_ttft_s=samples[0]['ttft_s'],
            warm_ms=p['subsequent_median_ms'], warm_ttft_s=statistics.median(s['ttft_s'] for s in samples[1:]))
    for arm in gate.ARMS:
        pp[arm] = [READ * 1000 / first_and_warm[p['index']]['warm_ms'] for p in processes if p['arm'] == arm]
    for a, b in zip(processes[::2], processes[1::2], strict=True):
        reload, retain = (a, b) if a['arm'] == 'reload' else (b, a)
        x, y = (first_and_warm[p['index']] for p in (reload, retain))
        ratios.append({metric: y[metric] / x[metric] for metric in x})
    return dict(process_times=list(first_and_warm.values()), warmed_PP=pp, retain_over_reload=ratios,
                repeated_warm_gain=all(p['warm_ms'] < 1 for p in ratios),
                repeated_1000={arm: all(v >= 1000 for v in rates) for arm, rates in pp.items()},
                full_context_qualified=False, sustained_decode_qualified=False, adopted=False)


def experiment():
    ctx = generation.experiment(synchronous_commit=True)
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    ctx.baseline, ctx.environment, ctx.command = baseline, environment, command
    ctx.protocol = dict(order=list(ORDER), read=READ, requests_per_process=RUNS,
        unit='Change only aligned KV retention on the untapped QSA layers.',
        acceptance='Require matched exact inputs and route, ordinary all-GPU residuals, 384 staged slots, '
                   'identical work outside KV retention and all per-layer KV counts. Compare first use and '
                   'subsequent PP and client TTFT separately. A warmed speed benefit needs all three paired '
                   'prompt-time ratios below one. Keep any smaller or mixed effects in the comparison.',
        stop='Stop on first source, clock, owner, route, work, fault or budget failure. Keep all eligible '
             'samples. No speed-based exclusions. This short output does not qualify sustained decode.')
    old_pins, old_save, old_current = ctx.pins, ctx.save, ctx.current

    def pins(revision):
        paths = (THIS, gate.THIS, build.__file__)
        relative = [str(Path(path).relative_to(bank.P)) if Path(path).is_absolute() else path for path in paths]
        sources = old_pins(revision)
        for path in relative:
            require((bank.P / path).read_bytes() == basis.validation.git('show', revision + ':' + path),
                    'changed DFlash2 timing input: ' + path)
            sources[path] = bank.sha(bank.P / path)
        return sources

    def normalized(row):
        for plan in row['plans']:
            env = environment(row, plan)
            del env[SELECTOR]
            yield {key: value.replace(plan['root'], '<process>') for key, value in env.items()}

    def save(row):
        if row['stage'] == 'prepared':
            for plan in row['plans']:
                plan.update(read=READ, heads=HEADS)
            row['request_hashes'] = {}
        old_save(row)

    def current(row):
        require(all((p['read'], p['heads']) == (READ, HEADS) for p in row['plans'])
                and len({bank.signature(env) for env in normalized(row)}) == 1,
                'planned input size or comparison environment differs')
        return old_current(row)

    ctx.pins, ctx.current, ctx.save = pins, current, save
    runner = SimpleNamespace(ROOT=ROOT, ORDER=[(READ, HEADS)] * len(ORDER), RUNS=RUNS,
        current=current, environment=environment, environment_fingerprint=generation.suite.signature,
        save=save, request_route=request_route,
        after_execution=check_work, decision=decision, validate_model_text=validate_model_text,
        verify_model_assets=verify_model_assets)
    ctx.run = lambda index: comparison.run(index, runner)
    return ctx


if __name__ == '__main__':
    generation.clock.main(experiment())
