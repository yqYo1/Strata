"""Time aligned KV retention on the primary DFlash2 long-prefill route."""

from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_primary_context as physical
import dflash2_v2_generation as generation
import prefill_workspace_admission as workspace

bank, basis, require = physical.bank, physical.basis, physical.require
clock, checks = generation.clock, physical.checks
KEY = 'dflash2_primary_prefill_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('reload', 'retain', 'retain', 'reload', 'reload', 'retain')
READ, HEADS, RUNS, SLOTS, CHUNK = 122881, 1, 3, 32, 8192
KV = 'STRATA_SYCL_PREFILL_KEEP_KV_STAGE'
WORKLOAD = 'native_gdn_20261018'
PROTOCOL = dict(order=list(ORDER), read=READ, requests_per_process=RUNS,
    scope='Three counterbalanced process pairs. Change only aligned prefill KV retention before native exec.',
    work='Original bench.sh exact 122881-input bodies and 24 thinking-on outputs. Keep mode16, bank32, '
         'demand upload, BF16 residuals, direct attention and the DFlash2/decode settings fixed.',
    acceptance='Check actual GPU residency, window, direct destination and every QSA layer. Require equal '
               'expert and generation work. Report first use and subsequent-request medians independently. '
               'Record a recurring win when all three paired warm times improve. Record 1000 tok/s separately.',
    stop='Stop on the first input, route, resource admission, ownership, validation or fault failure. '
         'Keep every eligible sample.',
    limits='Clean prefill timing on the primary code. Whole-model direct tap parity, live RESTORE, sustained '
           'decode, task and agent qualification remain separate.')


def environment(row, plan):
    env = generation.sync_environment(row, dict(root=plan['root'], index=plan['index']))
    env.update(STRATA_SYCL_PREFILL_DIRECT_ATTENTION_STORE='1', STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS=str(SLOTS),
               STRATA_SYCL_PREFILL_MOE_WINDOW='16', STRATA_SYCL_PREFILL_SPLIT_EXPERT_UPLOAD='demand',
               STRATA_SYCL_PREFILL_SPLIT_SCRATCH_CHECK_ONCE='0')
    env[KV] = str(int(plan['arm'] == 'retain'))
    return env


def baseline():
    prior = bank.previous.historic(checks.primary.PREFILL, 'dflash2_snapshot_direct_prefill_v1')
    seed = bank.previous.read(checks.MEASURED)
    require(basis.compact(seed) == prior and prior['stage'] == 'closed' and prior['passed'] and prior['complete']
            and prior['ownership_closed'] and not prior['active'] and prior['decision']['all_warm_requests_1000']['snapshot'],
            'recover the repeated original 122881-input timing')
    for plan in prior['plans']:
        require(bank.signature(environment(seed, dict(plan, arm='retain'))) == plan['environment_sha256'],
                'reconstructed long-prefill settings differ from the measured run')
    native, probe = checks.build()
    envelope = checks.parameters(native, probe)
    full = basis.validation.committed(bank.RESULT, (physical.KEY,))
    require(full['stage'] == 'closed' and full['passed'] and full['complete'] and full['ownership_closed']
            and not full['active'] and full['outcomes']['full-copy']['final_slot_consumed']
            and full['native_build_sha256'] == envelope['build_sha256'], 'finish the exact primary physical-context check')
    path = bank.P / 'data/expert-profile.bin'
    require(seed['ranking_profile']['old_path'] == str(path) and bank.sha(path) == seed['ranking_profile']['sha256'],
            'the primary expert profile changed')
    require(all(bank.sha(p) == h for p, h in seed['dflash2_assets'].items()), 'DFlash2 inputs changed')
    basis.comparison.shared.models.model_pins(seed['configuration'], seed['model_assets'], full=False)
    contract, manifest = basis.core.frontend.python_manifest()
    fields = ('boot', 'driver', 'environment_template', 'native_validation_probe', 'configuration', 'model_assets',
              'runtime_manifest', 'template', 'ordinary_control', 'dflash2_assets', 'validated_dflash2_args')
    row = {k: seed[k] for k in fields}
    row.update(binary=native['binaries']['strata'], native_source_commit=native['native_source_commit'],
               native_build_sha256=envelope['build_sha256'], physical_context_sha256=bank.signature(full),
               python_contract=contract, python_selected=manifest['selected'], request_hashes=prior['request_hashes'],
               reference_evidence=checks.primary.PREFILL)
    return row


def command(plan):
    root = Path(plan['root'])
    return ['/bin/sh', str(bank.P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(plan['port']),
            '--prefill-exact-tokens', str(READ), '--think', '--runs', str(RUNS), '--profile-request-gates',
            '--workload-id', WORKLOAD, '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')]


def route(text, plan):
    rows = READ - 1
    layout = workspace.one(text, 'strata prefill: layer-major candidate, ')
    mode = workspace.one(text, 'strata MoE window configuration: ')
    bank_owner = workspace.one(text, 'strata MoE expert bank: ')
    work = workspace.one(text, 'strata MoE accumulator window: ')
    uploads = workspace.one(text, 'strata MoE expert bank complete: ')
    storage = workspace.one(text, 'strata prefill attention bank: ')
    admission = workspace.one(text, 'strata MoE accumulator admission: ')
    split_owner = workspace.one(text, 'strata MoE split admission: ')
    require(all(layout[k] == str(v) for k, v in dict(tokens=rows, chunk=CHUNK, residual_dtype='bf16',
                residual_element_bytes=2, residual_gpu_rows=rows, residual_gpu=rows * 20480, residual_host=0).items())
            and mode['requested_mode'] == mode['effective_mode'] == '16'
            and mode['maximum_rows'] == work['maximum_rows'] == str(rows) and work['windows'] == '48'
            and work['GPU_resident_windows_only'] == '1' and work['operator_max'] == str(CHUNK),
            'actual mode16 all-GPU geometry changed')
    require(admission['rows'] == admission['gpu_prefix_rows'] == split_owner['rows'] == str(rows)
            and admission['requested_mode'] == '16' and split_owner['active'] == '1'
            and split_owner['decode_changed'] == '0'
            and int(admission['fixed_including_headroom_bytes']) <= int(admission['free_gpu_bytes'])
            and 0 < int(bank_owner['device_bytes']) <= int(split_owner['max_single_allocation_bytes']),
            'actual memory admission or decode placement changed')
    require(bank_owner['requested_slots'] == bank_owner['effective_slots'] == str(SLOTS)
            and bank_owner['staged'] == '1' and bank_owner['residual_gpu_rows'] == str(rows)
            and bank_owner['general_map'] == 'private_scratch_never_published' and uploads['final_drain'] == 'checked'
            and int(uploads['uploads']) + int(uploads['permanent_hits']) == int(work['dequants']) > 0
            and 'strata MoE split expert upload: ' not in text, 'expert work or demand upload changed')
    arena = workspace.one(text, 'strata MoE long workspace complete: ')
    split = workspace.one(text, 'strata MoE split complete: ')
    require(arena['rows'] == split['rows'] == str(rows) and arena['guards'] == '3_checked'
            and split['guard'] == 'checked' and arena['final_drain'] == split['final_drain'] == 'checked'
            and arena['decode_changed'] == split['decode_changed'] == '0', 'long workspace lifecycle changed')
    pieces = str(48 * (rows // CHUNK))
    inputs = workspace.one(text, 'strata prefill attention input: ')
    require(inputs == dict(requested='0', eligible=str(46 * (rows // CHUNK)), borrowed='0',
            borrowed_bytes='0', requested_D2D_bytes=str(46 * rows * 20480)), 'attention reader changed')
    require(storage == dict(requested='1', pieces=pieces, direct=pieces, requested_D2D_bytes='0')
            and 'strata prefill heads: draft_reader=absent-dflash2' in text
            and 'strata prefill heads: target detached ' in text, 'direct writer or head retirement changed')
    expected = []
    for layer in range(3, 48, 4):
        reloaded, reused, _ = physical.kv.expected_prefix(range(0, rows, CHUNK), layer,
            reuse=plan['arm'] == 'retain' and layer not in physical.TAPPED_QSA_LAYERS)
        expected.append((layer, reloaded, reused))
    actual = [tuple(map(int, m.groups())) for line in text.splitlines() if (m := physical.kv.STAGE.fullmatch(line))]
    require(actual == expected and text.count('strata prefill KV stage:') == len(expected), 'actual KV reuse changed')
    tap, = re.findall(r'^strata DFlash2: eager, prompt_tail_rows=2046 cold_prefix=0 vram_bytes=([1-9]\d*)$', text, re.M)
    req, = re.findall(clock.footer.HEADER, text, re.M)
    return dict(kv=dict(reloaded_pages=sum(v[1] for v in actual), reused_pages=sum(v[2] for v in actual)),
                work=work, uploads=uploads, target_rows=rows, direct_pieces=int(pieces), tap_vram_bytes=int(tap),
                native_request=list(map(int, req)))


def after_execution(outcome, row, plan):
    require(outcome['clean_speed'] and outcome['native_identity'] not in
            [p['native_identity'] for p in row['processes']], 'expected fresh quiet process')
    for frame, sample in zip(outcome['routes'], outcome['samples'], strict=True):
        require(frame['native_request'] == [READ, 0, READ, 24, sample['drafts_accepted'], sample['drafts_offered']],
                'native and client decode handoff work differs')
    if row['processes']:
        for before, after in zip(row['processes'][0]['routes'], outcome['routes'], strict=True):
            require({k: v for k, v in after.items() if k != 'kv'} == {k: v for k, v in before.items() if k != 'kv'},
                    'work outside the KV stage differs')
    outcome['arm'] = plan['arm']


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER and all(p['passed'] for p in processes),
            'three complete matched process pairs required')
    results = []
    for p in processes:
        first, *warm = p['samples']
        results.append(dict(arm=p['arm'], first_ms=first['prompt_ms'], first_ttft_s=first['ttft_s'],
            warm_ms=statistics.median(s['prompt_ms'] for s in warm), warm_ttft_s=statistics.median(s['ttft_s'] for s in warm)))
    pairs = []
    for x, y in zip(results[::2], results[1::2], strict=True):
        reload, retain = (x, y) if x['arm'] == 'reload' else (y, x)
        pairs.append({k: retain[k] / reload[k] for k in ('first_ms', 'first_ttft_s', 'warm_ms', 'warm_ttft_s')})
    rates = {arm: [READ * 1000 / p['warm_ms'] for p in results if p['arm'] == arm] for arm in ('reload', 'retain')}
    return dict(process_times=results, retain_over_reload=pairs, warmed_process_PP=rates,
                median_warmed_PP={k: statistics.median(v) for k, v in rates.items()},
                all_pairs_faster=all(p['warm_ms'] < 1 for p in pairs),
                all_warm_requests_1000={arm: all(s['prompt_ms'] <= READ for p in processes if p['arm'] == arm
                    for s in p['samples'][1:]) for arm in rates}, adopted=False)


def experiment():
    ctx = clock.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol = baseline, PROTOCOL
    ctx.configuration, ctx.environment, ctx.command = generation.configuration, environment, command
    original_pins, original_current, original_save = ctx.pins, ctx.current, ctx.save

    def pins(revision):
        sources = original_pins(revision)
        for name in (THIS, generation.THIS, physical.THIS, checks.THIS, 'data/expert-profile.bin'):
            require((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed primary prefill input: ' + name)
            sources[name] = bank.sha(bank.P / name)
        return sources

    def save(row):
        if row['stage'] == 'prepared':
            for plan in row['plans']:
                plan.update(read=READ, heads=HEADS)
        original_save(row)

    def current(row):
        normalized = []
        for plan in row['plans']:
            require((plan['read'], plan['heads']) == (READ, HEADS), 'frozen extent changed')
            env = environment(row, plan)
            normalized.append({k: v.replace(plan['root'], '<process>') for k, v in env.items() if k != KV})
        require(all(v == normalized[0] for v in normalized), 'another setting differs between prefill arms')
        return original_current(row)

    def run(index):
        # Isolate this caller's DFlash2 asset contract from the legacy MTP runner.
        runner = basis.core.source.device.load_module('primary_prefill_timing', basis.comparison.__file__)
        row = bank.previous.read(ROOT / 'record.json')
        original_assets = runner.shared.models.model_pins
        runner.shared = SimpleNamespace(**vars(runner.shared))

        def model_pins(config, assets, full=False):
            original_assets(row['configuration'], assets, full=full)
            for flag in ('--draft-model', '--mtp-draft-vocab'):
                path = config['args'][config['args'].index(flag) + 1]
                require(bank.sha(path) == row['dflash2_assets'][path], 'actual DFlash2 asset changed')

        def validate_model(text, plan):
            require(text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
                    and not re.search(r'(?m)^strata mtp: .*loaded|strata loan audit:|strata DFlash2 tap payload:', text),
                    'unexpected decode head, MTP layer or observer')

        runner.shared.models = SimpleNamespace(model_pins=model_pins)
        runner.fingerprint = bank.signature
        policy = SimpleNamespace(ROOT=ROOT, ORDER=[(READ, HEADS)] * len(ORDER), RUNS=RUNS, current=current,
            native_context=lambda row, plan: row, environment=environment, save=save, request_route=route,
            after_execution=after_execution, validate_model_text=validate_model, decision=decision)
        runner.run(index, policy)

    ctx.pins, ctx.current, ctx.save, ctx.run = pins, current, save, run
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
