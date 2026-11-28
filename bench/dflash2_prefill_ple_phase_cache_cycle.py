"""Screen cross-corpus cache reuse with the dedicated long-prefill cache."""

from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_prefill_ple_phase_cache as phase

bank, wait, geometry, corpus = phase.bank, phase.wait, phase.geometry, phase.corpus
capacity, validate, require = phase.capacity, phase.validate, phase.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_phase_cache_cycle_v1'
ROOT = bank.B / KEY.replace('_', '-')
CORPUS_ORDER = ('cxx', 'technical-prose')
REQUESTS = (*CORPUS_ORDER, *CORPUS_ORDER)
HASHES = [corpus.CORPORA[name][0] for name in REQUESTS]
FILES = {name: ROOT / (name + '.txt') for name in CORPUS_ORDER}
RUNS = len(REQUESTS)


def baseline():
    row = phase.baseline()
    proof = validate.committed(bank.RESULT, (phase.KEY,))
    require(wait.retained.closed(proof, 'closed') and proof['decision'] == capacity.decision(proof['processes'])
            and proof['decision']['warm_cache_mechanism_advances']
            and all(row[k] == proof[k] for k in ('binary', 'native_source_commit', 'native_build_sha256')),
            'complete same-build single-corpus validation before testing alternating corpora')
    row.pop('request_hashes')
    row.pop('frozen_work')
    return dict(row, single_corpus_proof_sha256=bank.signature(proof))


def after_execution(outcome, row, plan):
    require([sample['corpus_sha256'] for sample in outcome['samples']] == HASHES
            and all(sample['actual_read_tokens'] == wait.READ and sample['actual_reused_tokens'] == 0
                    for sample in outcome['samples']), 'actual alternating corpus order or native token extent differs')
    if row['processes']:
        expected_work = [[r['request_work'], r['bank_upload_bytes']] for r in row['processes'][0]['routes']]
    else:
        expected_work = [[r['native_request'], int(r['uploads']['requested_H2D_bytes'])] for r in outcome['routes']]
    phase.after_execution(outcome, dict(row, frozen_work=expected_work), plan)


def run(ctx, index):
    row = bank.previous.read(ctx.ROOT / 'record.json')
    basis = wait.basis
    runner = basis.core.source.device.load_module('phase_cache_cycle_storage', basis.comparison.__file__)
    original_models = runner.shared.models
    runner.shared = SimpleNamespace(**vars(runner.shared))
    observed = []

    def model_pins(config, assets, full=False):
        original_models.model_pins(row['configuration'], assets, full=full)
        for flag in ('--draft-model', '--mtp-draft-vocab'):
            path = config['args'][config['args'].index(flag) + 1]
            require(bank.sha(path) == row['dflash2_assets'][path], 'actual DFlash2 input changed')

    def validate_model(text, _):
        require(text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
                and not re.search(r'(?m)^strata mtp: .*loaded|strata loan audit:|strata DFlash2 tap payload:', text),
                'unexpected head, MTP layer or observer')

    def check_sample(sample, count, read, minimum):
        runner.check_sample(sample, count, read, minimum)
        digest = sample.get('corpus_sha256')
        require(len(observed) < RUNS and digest == count.get('corpus_sha256') == HASHES[len(observed)],
                'measured corpus or its token-count preflight differs from the frozen order')
        observed.append((sample['request_sha256'], digest))

    def finish(outcome, running, plan):
        require(len(observed) == RUNS, 'incomplete corpus identity checks')
        for sample, (request, digest) in zip(outcome['samples'], observed, strict=True):
            require(sample['request_sha256'] == request, 'timed request identity differs')
            sample['corpus_sha256'] = digest
        ctx.after_execution(outcome, running, plan)

    runner.shared.models = SimpleNamespace(model_pins=model_pins)
    runner.fingerprint = bank.signature
    policy = SimpleNamespace(ROOT=ctx.ROOT, ORDER=[(wait.READ, wait.speed.HEADS)] * len(ctx.ORDER),
        RUNS=RUNS, current=ctx.current, native_context=lambda data, _: data,
        environment=ctx.environment, save=ctx.save, request_route=ctx.request_route,
        observe_gate=geometry.attribution.observe_storage, check_sample=check_sample,
        after_execution=finish, decision=ctx.decision, validate_model_text=validate_model)
    runner.run(index, policy)


def decision(processes):
    require(tuple(p['arm'] for p in processes) == capacity.ORDER
            and all(p['complete'] and p['passed'] and p['ownership_closed']
                    and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == 4,
            'require four independently closed alternating-corpus processes')
    paired = {name: [] for name in CORPUS_ORDER}
    for a, b in ((processes[0], processes[1]), (processes[3], processes[2])):
        for i, name in enumerate(CORPUS_ORDER):
            slots = [i, i + len(CORPUS_ORDER)]
            runs = [[dict(prompt_ms=p['samples'][j]['prompt_ms'], ttft_s=p['samples'][j]['ttft_s'],
                          record_touches=p['routes'][j]['record_touches'],
                          ple_wait_ms=p['routes'][j]['ple_wait']['future_get_ms'])
                     for j in slots] for p in (a, b)]
            paired[name].append(dict(first_seen=[arm[0] for arm in runs], revisited=[arm[1] for arm in runs]))
    decision = {}
    for name, chunks in paired.items():
        visits = [p['revisited'] for p in chunks]
        decision[name] = dict(revisit_time_ratio=statistics.geometric_mean(
            large['prompt_ms'] / stock['prompt_ms'] for stock, large in visits),
            repeated_io_gain=all(large['record_touches'] < stock['record_touches'] for stock, large in visits),
            repeated_time_gain=all(large['prompt_ms'] < stock['prompt_ms'] and large['ttft_s'] < stock['ttft_s']
                                   for stock, large in visits))
    return dict(paired=paired, revisit_decision=decision,
                varied_prose_mechanism_advances=all(decision['technical-prose'][k]
                    for k in ('repeated_io_gain', 'repeated_time_gain')), adopted=False)


def experiment():
    ctx = geometry.experiment()
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, capacity.ORDER
    ctx.run = lambda index: run(ctx, index)
    base_command, base_save, base_current = ctx.command, ctx.save, ctx.current

    def save(row):
        if row['stage'] == 'prepared':
            row['request_hashes'] = {}
            for name, path in FILES.items():
                with path.open('xb') as target:
                    target.write(corpus.corpus_payload(name))
                path.chmod(0o400)
        base_save(row)

    def current(row):
        require(all(bank.sha(path) == corpus.CORPORA[name][0] for name, path in FILES.items()),
                'frozen alternating corpus bytes differ')
        return base_current(row)

    def command(plan):
        cmd = base_command(plan)
        cmd[cmd.index('--runs') + 1] = str(RUNS)
        for name in CORPUS_ORDER:
            cmd.extend(['--prefill-corpus', str(FILES[name])])
        return cmd

    ctx.baseline, ctx.save, ctx.current, ctx.command = baseline, save, current, command
    ctx.environment = phase.experiment().environment
    ctx.request_route, ctx.after_execution, ctx.decision = phase.route, after_execution, decision
    ctx.extra_sources = (*ctx.extra_sources, phase.THIS, capacity.THIS, corpus.THIS, THIS)
    ctx.selector = phase.SELECTOR
    ctx.protocol = dict(order=list(capacity.ORDER), read=wait.READ, requests_per_process=RUNS,
        corpus_order=list(REQUESTS), corpus_sha256=HASHES, corpus_revision=corpus.CORPUS_REVISION,
        corpus_paths={name: list(corpus.CORPORA[name][1]) for name in CORPUS_ORDER},
        scope='Use a fixed C++ and technical-prose cycle, each at 118785 fresh tokens with zero prompt reuse. '
              'Change only the dedicated prefill row-cache setting in two reverse-order process pairs.',
        decision='Report first encounters and revisits separately. Require repeated storage, PP and TTFT '
                 'improvement on technical-prose revisits before advancing. Report C++ and cold effects '
                 'individually before any broader adoption decision.',
        stop=geometry.PLAN['stop'])
    return ctx


if __name__ == '__main__':
    wait.speed.clock.main(experiment())
