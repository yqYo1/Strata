"""Compare demand and batched mode16 copies in the same DFlash2 engine."""

from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_mode16_batch_route as upload
import dflash2_mode16_batch_build as candidate

curve, bank, require = upload.curve, upload.bank, upload.require
speed, basis, clock = curve.speed, curve.basis, curve.speed.clock
KEY = 'dflash2_mode16_batch_model_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
READ = upload.READ
ORDER = ('demand', 'batched', 'batched', 'demand', 'demand', 'batched')
UPLOAD = 'STRATA_SYCL_PREFILL_SPLIT_EXPERT_UPLOAD'
PLAN = dict(order=list(ORDER), read=READ, requests_per_process=speed.RUNS,
    scope='One coherent candidate binary. Change only demand versus batched issue on the all-GPU '
          'mode16 32-slot route with KV retention. Use the original three 102401-input bodies.',
    evidence='Check all 48 layer windows and the actual requested bytes, stationary expert work, '
             'repacked-bank drain, relocated scratch, direct stores, DFlash2 handoff and decode expert work.',
    decision='The warm prompt median must improve in at least two clean pairs with a paired median ratio '
             'below one. Report first use, warm time and TTFT separately. Output is limited to 24 tokens.',
    stop='Stop on changed source, allocation, placement, work, runtime, request, validation or ownership.')


def baseline():
    original = bank.previous.read(speed.ROOT / 'record.json')
    closed = bank.previous.historic(candidate.PREFILL, speed.KEY)
    length = bank.previous.historic(upload.CONTROL, curve.KEY)
    donor = length['processes'][3]
    require(basis.compact(original) == closed and closed['stage'] == 'closed'
            and closed['passed'] and closed['ownership_closed'] and not closed['active']
            and donor['read'] == READ and donor['passed'] and donor['complete']
            and donor['ownership_closed'] and donor['native_exit_code'] == 0 and not donor['fresh_faults']
            and donor['binary'] == original['binary'] and length['selected_arm'] == 'retain'
            and length['native_build_sha256'] == original['native_build_sha256']
            and len(donor['routes']) == len(donor['samples']) == speed.RUNS,
            'recover the closed primary speed control and all three original bodies')
    native = candidate.admitted_build()
    binary = native['binaries']['strata']
    fields = ('boot', 'driver', 'environment_template', 'native_validation_probe', 'configuration',
              'model_assets', 'runtime_manifest', 'template', 'ordinary_control', 'dflash2_assets',
              'validated_dflash2_args', 'python_contract', 'python_selected')
    row = {k: original[k] for k in fields}
    require(native['driver'] == row['driver'] and candidate.builder.guards.driver_guard() == row['driver']
            and original['boot'] == speed.checks.BOOT.read_text().strip()
            and all(bank.sha(p) == h for p, h in row['dflash2_assets'].items()),
            'model artifacts, driver or boot changed')
    runtime = row['runtime_manifest']
    providers = bank.previous.read(Path(runtime['path']))
    require(bank.sha(runtime['path']) == runtime['sha256'] and len(providers) == runtime['files']
            and all(bank.sha(p) == h for p, h in providers.items())
            and all(bank.sha(p['path']) == p['sha256'] for p in row['native_validation_probe'].values()),
            'qualified runtime or native validation probe changed')
    return dict(row, binary=binary, native_source_commit=native['native_source_commit'],
                native_build_sha256=bank.signature(candidate.builder.compact(native)), selected_arm='retain',
                request_hashes={str(READ): [item['request_sha256'] for item in donor['samples']]},
                prefill_work=[upload.comparable_work(frame) for frame in donor['routes']])


def command(plan):
    cmd = speed.command(plan)
    cmd[cmd.index('--prefill-exact-tokens') + 1] = str(READ)
    return cmd


def configuration(row, plan):
    owned = dict(row['configuration'], exe=str(bank.P / 'tools/sycl/engine_once.py'), cwd=str(bank.P))
    return speed.generation.configuration(dict(row, configuration=owned), plan)


def environment(row, plan):
    return dict(curve.environment(row, plan), **{UPLOAD: plan['arm']})


def after_execution(outcome, row, plan):
    require(outcome['clean_speed'] and outcome['request_routes_checked_before_continue'] == speed.RUNS,
            'incomplete clean model evidence')
    work = [upload.comparable_work(frame) for frame in outcome['routes']]
    require([{k: v for k, v in frame.items() if k != 'decode_experts'} for frame in work] == row['prefill_work']
            and all(work == [upload.comparable_work(frame) for frame in p['routes']]
                    for p in row['processes']), 'actual prefill or decode work changed')
    outcome['arm'] = plan['arm']


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and not p['fresh_faults']
                    and p['clean_speed'] and p['native_exit_code'] == 0 for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(ORDER), 'complete independent clean pairs required')
    clocks = []
    for p in processes:
        first, *warm = p['samples']
        clocks.append(dict(arm=p['arm'], first_ms=first['prompt_ms'], first_ttft_s=first['ttft_s'],
                           warm_ms=statistics.median(s['prompt_ms'] for s in warm),
                           warm_ttft_s=statistics.median(s['ttft_s'] for s in warm)))
    ratios = []
    for x, y in zip(clocks[::2], clocks[1::2], strict=True):
        demand, batched = (x, y) if x['arm'] == 'demand' else (y, x)
        ratios.append({k: batched[k] / demand[k] for k in ('first_ms', 'first_ttft_s', 'warm_ms', 'warm_ttft_s')})
    warm = [r['warm_ms'] for r in ratios]
    return dict(process_clocks=clocks, batched_over_demand=ratios,
                repeatable_warm_gain=sum(v < 1 for v in warm) >= 2 and statistics.median(warm) < 1,
                adopted=False)


def experiment():
    ctx = clock.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol = baseline, PLAN
    ctx.configuration, ctx.command, ctx.environment = configuration, command, environment
    original_save, original_current = ctx.save, ctx.current
    previous_pins = speed.experiment().pins

    def pins(revision):
        paths = previous_pins(revision)
        for name in (THIS, curve.THIS, candidate.THIS,
                     'bench/dflash2_mode16_batch_route.py', 'bench/sycl/dflash2_primary_foundation_proof.py'):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed batch comparison input: ' + name)
            paths[name] = bank.sha(path)
        return paths

    def save(row):
        if row['stage'] == 'prepared':
            for plan in row['plans']:
                plan.update(read=READ, heads=speed.HEADS, kv_arm='retain')
        original_save(row)

    def current(row):
        common = []
        for plan in row['plans']:
            require((plan['read'], plan['heads'], plan['kv_arm']) == (READ, speed.HEADS, 'retain'),
                    'the frozen upload mode or extent changed')
            common.append({k: v.replace(plan['root'], '<process>')
                           for k, v in environment(row, plan).items() if k != UPLOAD})
        require(all(env == common[0] for env in common), 'another immutable setting differs between arms')
        return original_current(row)

    def run(index):
        runner = basis.core.source.device.load_module('mode16_upload_timing', basis.comparison.__file__)
        row = bank.previous.read(ROOT / 'record.json')
        original_model_pins = runner.shared.models.model_pins
        runner.shared = SimpleNamespace(**vars(runner.shared))

        def model_pins(config, assets, full=False):
            original_model_pins(row['configuration'], assets, full=full)
            for flag in ('--draft-model', '--mtp-draft-vocab'):
                path = config['args'][config['args'].index(flag) + 1]
                require(bank.sha(path) == row['dflash2_assets'][path], 'actual DFlash2 asset changed')

        def validate_model(text, _plan):
            require(text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
                    and not re.search(r'(?m)^strata mtp: .*loaded|strata loan audit:|strata DFlash2 tap payload:', text),
                    'unexpected head, MTP layer or observer')

        runner.shared.models = SimpleNamespace(model_pins=model_pins)
        runner.fingerprint = bank.signature
        policy = SimpleNamespace(ROOT=ROOT, ORDER=[(READ, speed.HEADS)] * len(ORDER), RUNS=speed.RUNS,
            current=current, native_context=lambda r, _: r, environment=environment, save=save,
            request_route=upload.route, after_execution=after_execution, decision=decision,
            validate_model_text=validate_model)
        runner.run(index, policy)

    ctx.pins, ctx.save, ctx.current, ctx.run = pins, save, current, run
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
