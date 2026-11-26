"""Compare Xe client VRAM for gathered and indexed draft heads."""

from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_decode_indexed_head_memory as memory
import client_resources as resources


bank, basis, require, speed = memory.bank, memory.basis, memory.require, memory.speed
handoff = memory.handoff
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_indexed_head_drm_v1'
ROOT = bank.B / KEY.replace('_', '-')
MEMORY_RESULT = '6cae88023'
ARMS = ('indexed', 'gathered')
VRAM = ('drm-total-vram0', 'drm-resident-vram0')
PLAN = dict(memory.PLAN, order=list(ARMS),
    scope='Repeat the same two 126977-token bodies per arm in reverse process order. Add Xe client '
          'memory snapshots while the client waits at request begin and end. Use the same executable and stock expert map.',
    gates='Check both native identities and all request and memory gates. Bind four ordered fdinfo readings to '
          'each process. Retain individual client memory bytes. Shared or changed clients disqualify aggregate deltas.',
    decision='Compare Xe requested and resident VRAM separately with the SYCL free-memory observations. '
             'Keep application-owned weights, client residency, available memory and allocation admission distinct.',
    retention='Reuse the existing comparison generation. Retire this capture and review the executable after both arms.')


def mapped_plan(plan):
    return dict(plan, index=memory.ARMS.index(ARMS[plan['index']]))


def environment(row, plan):
    return memory.environment(row, mapped_plan(plan))


def route(text, plan):
    return memory.route(text, mapped_plan(plan))


def request_gate(index):
    request, gate = divmod(index, 2)
    return request, ('begin', 'end')[gate]


def take_snapshot(outcome, pci, request, stage):
    readings = outcome.setdefault('drm_boundaries', [])
    require((request, stage) == request_gate(len(readings)), 'out-of-order Xe request gate')
    identity = outcome['native_identity']
    sample = resources.snapshot(identity, pci)
    clients = [dict(client_id=c['client_id'], memory_bytes=c['memory_bytes']) for c in sample['clients']]
    clients.sort(key=lambda c: c['client_id'])
    require(clients and all(set(VRAM) <= c['memory_bytes'].keys() for c in clients), 'Xe client VRAM counters are missing')
    readings.append(dict(request=request, stage=stage, clients=clients,
                         snapshot_ns=sample['end_monotonic_ns'] - sample['begin_monotonic_ns']))


def finish_arm(outcome, row, plan):
    readings = outcome['drm_boundaries']
    require([(r['request'], r['stage']) for r in readings] == [request_gate(i) for i in range(2 * handoff.RUNS)],
            'incomplete Xe client sampling')
    for frame, sample in zip(outcome['routes'], outcome['samples'], strict=True):
        require(frame['head_arm'] == ARMS[plan['index']] and frame['native_request'] == [
                plan['read'], 0, plan['read'], 24, sample['drafts_accepted'], sample['drafts_offered']]
                and sample['drafts_offered'] > 0, 'the native draft and client work disagree')
    ids = {tuple(c['client_id'] for c in r['clients']) for r in readings}
    private = all(v == 0 for r in readings for c in r['clients'] for k, v in c['memory_bytes'].items()
                  if k.startswith('drm-shared-'))
    outcome.update(head_arm=ARMS[plan['index']], clean_speed=False, extra_bytes=row['extra_bytes'],
                   drm_aggregate_eligible=len(ids) == 1 and private)


def decide(processes):
    require(tuple(p['head_arm'] for p in processes) == ARMS and all(p['passed'] and p['ownership_closed']
            and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ARMS),
            'require both independently closed Xe memory arms')
    mapped, gathered = processes
    compared = memory.contrast(dict(processes=[gathered, mapped], extra_bytes=gathered['extra_bytes']))
    comparable = compared['matched_work'] and all(p['drm_aggregate_eligible'] for p in processes)
    totals = {p['head_arm']: [dict(request=r['request'], stage=r['stage'],
              bytes={k: sum(c['memory_bytes'][k] for c in r['clients']) for k in VRAM})
              for r in p['drm_boundaries']] for p in processes} if comparable else {}
    deltas = [dict(request=a['request'], stage=a['stage'],
                   bytes={k: a['bytes'][k] - b['bytes'][k] for k in VRAM})
              for a, b in zip(totals.get('indexed', ()), totals.get('gathered', ()), strict=True)]
    return dict(matched_work=compared['matched_work'], drm_aggregate_eligible=comparable,
                drm_vram=totals, indexed_minus_gathered_bytes=deltas, sycl_free_byte_deltas=compared['free_byte_deltas'],
                allocation_verified=False, clean_speed_qualified=False, adopted=False)


def experiment():
    ctx = memory.experiment()
    base, pins = ctx.baseline, ctx.pins
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.protocol, ctx.environment = PLAN, environment

    def baseline():
        row = base()
        prior = basis.validation.committed(bank.RESULT, (memory.KEY,))
        require(prior == bank.previous.historic(MEMORY_RESULT, memory.KEY)
                and handoff.retained.closed(prior, 'closed') and prior['binary'] == row['binary']
                and prior['native_build_sha256'] == row['native_build_sha256']
                and prior['request_hashes'] == row['fixed_bodies']
                and prior['decision']['headroom_comparison']['matched_work']
                and not prior['decision']['headroom_comparison']['eligible_for_allocation_probe'],
                'require the same-binary memory pair with the original headroom outcome')
        return dict(row, head_memory_reference_sha256=bank.signature(prior))

    def sources(revision):
        result = pins(revision)
        for name in (THIS, str(Path(resources.__file__).relative_to(bank.P))):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name), 'changed Xe measurement input: ' + name)
            result[name] = bank.sha(path)
        return result

    def run(index):
        row = bank.previous.read(ROOT / 'record.json')
        runner = basis.core.source.device.load_module('indexed_head_drm_runner', basis.comparison.__file__)
        original = runner.shared.models
        runner.shared = SimpleNamespace(**vars(runner.shared))
        pci = Path(row['driver']['pci']).name

        def model_pins(config, assets, full=False):
            original.model_pins(row['configuration'], assets, full=full)
            for flag in ('--draft-model', '--mtp-draft-vocab'):
                path = config['args'][config['args'].index(flag) + 1]
                require(bank.sha(path) == row['dflash2_assets'][path], 'DFlash2 asset changed')

        def validate_model(text, _):
            handoff.ranked.restore.check_loader(text, row['qualified_profiles']['stock']['path'], per_layer=False)
            require(text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
                    and not re.search(r'(?m)^strata mtp: .*loaded|strata loan audit:', text), 'unexpected native route')

        runner.shared.models = SimpleNamespace(model_pins=model_pins)
        runner.fingerprint = bank.signature
        policy = SimpleNamespace(ROOT=ROOT, ORDER=[(memory.READ, speed.HEADS)] * len(ARMS), RUNS=handoff.RUNS,
            current=ctx.current, native_context=lambda r, _: r, environment=environment, save=ctx.save,
            request_route=route, after_execution=finish_arm, decision=decide, validate_model_text=validate_model,
            observe_gate=lambda outcome, request, stage, _: take_snapshot(outcome, pci, request, stage))
        runner.run(index, policy)

    ctx.baseline, ctx.pins, ctx.run = baseline, sources, run
    return ctx


if __name__ == '__main__':
    speed.clock.main(experiment())
