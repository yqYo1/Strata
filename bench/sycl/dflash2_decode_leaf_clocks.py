"""Measure worker leaf clocks and logical weights on the DFlash2 code route."""

from collections import Counter
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_v2_decode_clocks as control
import adaptive_cpu_leaf_evidence as leaves

bank, basis, require = control.bank, control.basis, control.require
KEY = 'dflash2_decode_leaf_clocks_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
CLOCK_REFERENCE = 'e26315def2fb25b591ac6070b78236276ef2456d'
EXPERT_LAYOUT = 'tests/data/native_experts/iq3_s.txt'
ORDER = ('caller-before', 'worker', 'caller-after')
WORKER = 'STRATA_CPU_SERVICE'


def baseline():
    row = control.baseline()
    measured = bank.previous.historic(CLOCK_REFERENCE, control.KEY)
    native = bank.previous.read(control.generation.v2.upload.build.ROOT / 'record.json')
    require(measured['stage'] == 'closed' and measured['complete'] and measured['passed']
            and measured['ownership_closed'] and not measured['active']
            and measured['decision']['bounded_attribution']
            and all(row[name] == measured[name] for name in (
                'binary', 'native_source_commit', 'native_build_sha256', 'boot', 'driver', 'runtime_manifest'))
            and native['binaries']['strata'] == row['binary']
            and native['native_source_commit'] == row['native_source_commit']
            and '-DSTRATA_CPU_SERVICE_PROFILE:BOOL=ON' in native['recipe']['configure'],
            'recover the observed DFlash2 caller clocks and compiled worker observer')
    args = row['validated_dflash2_args']
    pack = Path(args[args.index('--pack') + 1])
    require((pack / 'native_experts.txt').read_bytes() == (bank.P / EXPERT_LAYOUT).read_bytes(),
            'the packed expert sizes must match the committed model manifest')
    return dict(row, decode_clock_reference=CLOCK_REFERENCE)


def environment(row, plan):
    env = control.environment(row, dict(plan, arm='clocks'))
    env[WORKER] = str(int(plan['arm'] == 'worker'))
    return env


def logical_weight_work(groups, target_rows):
    sizes, layers = {}, Counter()
    for gate, down, size in leaves.census.manifest_formats(bank.P / EXPERT_LAYOUT):
        pair = gate, down
        require(pair in leaves.service.TYPES and size > 0 and sizes.get(pair, size) == size,
                'each measured format pair must have one packed expert size')
        sizes[pair] = size
        layers[pair] += 1

    grouped_bytes = ungrouped_bytes = 0
    entries = Counter()
    for group in groups:
        pair = group['gu'], group['down']
        require(pair in sizes, 'worker format absent from the model manifest')
        jobs, width = group['jobs'], group['nt']
        grouped_bytes += jobs * sizes[pair]
        ungrouped_bytes += jobs * width * sizes[pair]
        entries[pair] += jobs * width
    require(all(count <= layers[pair] * target_rows * 10 for pair, count in entries.items()),
            'CPU format entries exceed all verified rows on the matching layers')
    return dict(logical_job_blob_bytes=grouped_bytes, logical_ungrouped_blob_bytes=ungrouped_bytes)


def routes(text, samples, arm):
    evidence = control.routes(text, samples, 'clocks')
    if arm != 'worker':
        require(not leaves.service.records(text), 'disabled worker observer emitted records')
        return evidence

    observed = leaves.extract(text, samples, width=4, workers=5)
    require(len(observed) == len(samples) and observed[-1]['request'] == len(samples),
            'worker records must bind every completed request')
    leaf = observed[-1]
    work = evidence['expert_copy']
    require((leaf['cpu_jobs'], leaf['cpu_entries']) == (work['cpu_unique_misses'], work['cpu_entries']),
            'worker service must conserve the checked long-request work')
    fields = ('calls', 'wall_ns', 'cpu_ns')
    by_op = {op: {key: sum(group['operations'][op][key] for group in leaf['format_nt'])
                  for key in fields} for op in leaves.service.OPS}
    workers = []
    for _, name, values in leaves.service.records(text):
        if name == 'CPU_SERVICE_THREAD' and int(values['request']) == leaf['request']:
            workers.append({key: int(values[key]) for key in ('slot', *fields)})
    require(len(workers) == leaf['thread_count'], 'worker coverage differs from the checked leaf sums')
    weight_work = logical_weight_work(leaf['format_nt'], evidence['clocks']['target_rows'])
    evidence['worker_service'] = dict(request=leaf['request'], operations=by_op, per_slot=workers,
        format_nt=leaf['format_nt'], cpu_jobs=leaf['cpu_jobs'], cpu_entries=leaf['cpu_entries'],
        **weight_work,
        aggregation='Worker time can overlap. Weight bytes charge one packed expert per job, or per token row without grouping.')
    return evidence


def decision(processes):
    samples = [record['evidence']['samples'][-1] for record in processes]
    require(tuple(record['arm'] for record in processes) == ORDER
            and len({(record['native_identity']['pid'], record['native_identity']['start_ticks'])
                     for record in processes}) == len(ORDER)
            and ['worker_service' in record['routes'] for record in processes] == [False, True, False],
            'three independent caller/worker/caller scopes required')
    work = [([sample[k] for k in control.WORK], record['routes']['expert_copy'],
             record['routes']['clocks']['windows'])
            for sample, record in zip(samples, processes, strict=True)]
    matched = work[0] == work[1] == work[2]
    observed = samples[1]['decode_ms']
    ratios = [observed / samples[i]['decode_ms'] for i in (0, 2)]
    weights = processes[1]['routes']['worker_service']['logical_job_blob_bytes']
    emitted = samples[1]['generated_tokens']
    return dict(matched_work=matched, worker_over_caller_wall=ratios,
                bounded_leaf_attribution=matched and all(1 / control.LIMIT <= x <= control.LIMIT for x in ratios),
                logical_expert_weight_bytes_per_output=weights / emitted,
                logical_expert_weight_GBps_at_70=weights * 70 / emitted / 1e9,
                decode_ms=[s['decode_ms'] for s in samples], physical_RAM_bytes='unknown',
                clean_speed_qualified=False)


def experiment():
    ctx = control.generation.experiment(synchronous_commit=True)
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.environment, ctx.command, ctx.routes = baseline, environment, control.command, routes
    ctx.evidence_reader.read_complete_suite = control.read_subset
    ctx.evidence_reader.decide = decision
    ctx.protocol = dict(order=list(ORDER), workload=control.WORKLOAD, overhead_factor_limit=control.LIMIT,
        unit='Switch the worker leaf clock before launch. All arms keep the qualified caller and rounded clocks.',
        acceptance='Use the original 33217-read regular-code body, thinking and exact source, assets and route. '
                   'Bind every native/request/copy record and all adaptive width-one-through-four leaf work '
                   'to independently reported expert jobs and entries. Check thread totals and the exact packed '
                   'weight sizes for the observed format mix.',
        decision='Require matched work, three fresh owners and both paired ratios within the previous 1.10 envelope. '
                 'Use the checked GU, hidden-quantization and Down aggregate clocks to select an experiment. '
                 'Run the fixed-cache format census separately when it is needed to confirm a specific kernel cell.',
        scope='The screen measures incremental worker-observer overhead. Retain the caller-clock controls as '
              'their own observations. Worker CPU totals overlap. Physical DRAM traffic needs its own measurement.',
        stop='Keep the original failed or excessive-overhead outcome. Stop on the first changed input, '
             'source, runtime, ownership, route, work, output or fault condition.')
    reference_pins = control.experiment().pins
    current = ctx.current

    def pins(revision):
        values = reference_pins(revision)
        paths = (THIS, EXPERT_LAYOUT, Path(leaves.__file__), Path(leaves.service.__file__), Path(leaves.census.__file__))
        for path in paths:
            name = str(path.relative_to(bank.P)) if isinstance(path, Path) else path
            actual = bank.P / name
            require(actual.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed worker attribution input: ' + name)
            values[name] = bank.sha(actual)
        return values

    def check(row):
        normalized = []
        for plan in row['plans']:
            settings = environment(row, plan)
            require(settings.pop(WORKER) == str(int(plan['arm'] == 'worker')), 'worker selector differs')
            normalized.append({k: v.replace(plan['root'], '<process>') for k, v in settings.items()})
        require(all(settings == normalized[0] for settings in normalized), 'settings differ outside worker observation')
        return current(row)

    ctx.pins, ctx.current = pins, check
    return ctx


if __name__ == '__main__':
    control.clock.main(experiment())
