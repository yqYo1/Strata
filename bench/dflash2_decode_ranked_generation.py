"""Compare one DFlash2 ranking change on the retained serving engine."""

from hashlib import sha256
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_decode_stock_global_restore as restore
import dflash2_primary_continuation as primary


bank, basis, require = primary.bank, primary.basis, primary.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
COMPARISONS = {'within': ('stock', 'within'), 'cross': ('within', 'cross')}
EXTENTS = ('short', 'near')


def restore_proof(row, key, order):
    proof = basis.validation.committed(bank.RESULT, (key,))
    require(primary.retained.closed(proof, 'closed')
            and all(proof[k] == row[k] for k in ('binary', 'native_source_commit', 'native_build_sha256'))
            and list(proof['outcomes']) == list(order) and proof['outcomes']['host']['passed'],
            'complete the same-engine stock and ranked RESTORE controls')
    return proof


def qualified_profile(arm, stock, ranked):
    if arm == 'stock':
        proof, tag, path = stock, 'global', restore.GLOBAL
        digest = proof['profile_sha256']
    else:
        proof, tag, path = ranked, arm, restore.RANKED_ROOT / (arm + '.bin')
        digest = proof['ranked_profile_sha256'][arm]
    require(bank.sha(path) == digest, 'the restored profile bytes changed')
    slots = restore.quota.masks(restore.quota.read_profile(path)[:restore.quota.EXPECTED_SLOTS])
    membership = sha256(b''.join(mask.to_bytes(64, 'little') for mask in slots)).hexdigest()
    for extent in EXTENTS:
        case = proof['outcomes'][f'{extent}-{tag}']
        require(case['passed'] and case['complete'] and case['ownership_closed']
                and case['native_exit_code'] == 0 and not case['fresh_faults']
                and len(case['decode_placement']) == 3
                and {p['membership_sha256'] for p in case['decode_placement']} == {membership},
                'required native slot and restore evidence differs')
    return dict(path=str(path), sha256=digest, membership_sha256=membership)


def configuration(row, plan):
    cfg = primary.speed.generation.configuration(row, plan)
    args = list(cfg['args'])
    for key, value in (('--expert-cache', '1256'), ('--expert-profile', str(restore.STOCK)),
                       ('--max-context', '262144'), ('--adapt-swaps', '0')):
        require(args.count(key) == 1 and args[args.index(key) + 1] == value, 'fixed serving option changed: ' + key)
    require(args.count('--expert-cache-per-layer') == 1, 'expected the primary per-layer seed configuration')
    args[args.index('--expert-cache') + 1] = '968'
    args[args.index('--expert-profile') + 1] = row['qualified_profiles'][plan['arm']]['path']
    args.remove('--expert-cache-per-layer')
    return dict(cfg, args=args)


def decision(processes, order):
    require(tuple(p['arm'] for p in processes) == order and all(p['passed'] and p['complete']
            and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(order), 'all six independent clean processes are required')
    ref = order[0]
    pairs = {name: [] for name in primary.WORKLOADS}
    for first, second in zip(processes[::2], processes[1::2], strict=True):
        old, new = (first, second) if first['arm'] == ref else (second, first)
        for name in pairs:
            x, y = [next(s for s in p['evidence']['samples'] if s.get('workload') == name) for p in (old, new)]
            require(x['request_sha256'] == y['request_sha256'] and x['read_tokens'] == y['read_tokens']
                    and x['reused_tokens'] == y['reused_tokens'] == 0
                    and min(x['generated_tokens'], y['generated_tokens']) >= 1024,
                    'paired read extent or sustained work changed')
            old_route, new_route = old['routes'][name], new['routes'][name]
            unpaired = ('expert_work', 'gpu_headroom_bytes')
            require({k: v for k, v in old_route.items() if k not in unpaired}
                    == {k: v for k, v in new_route.items() if k not in unpaired}, 'paired prefill dispatch changed')
            jobs = [r['expert_work']['cpu_unique_misses'] / s['generated_tokens']
                    for r, s in ((old_route, x), (new_route, y))]
            pairs[name].append(dict(tg_ratio=new['evidence']['decode_rates'][name] / old['evidence']['decode_rates'][name],
                                    prompt_time_ratio=y['prompt_ms'] / x['prompt_ms'], ttft_ratio=y['ttft_s'] / x['ttft_s'],
                                    cpu_jobs_per_output=jobs,
                                    drafts=[(s['drafts_accepted'], s['drafts_offered']) for s in (x, y)]))
    gains = {name: [p['tg_ratio'] for p in samples] for name, samples in pairs.items()}
    pp = all(statistics.median(p['prompt_time_ratio'] for p in samples) <= 1 for samples in pairs.values())
    tg = all(sum(value > 1 for value in values) >= 2 and statistics.geometric_mean(values) > 1
             for values in gains.values())
    return dict(pairs=pairs, prefill_preserved=pp, sustained_gain_screen=tg,
                eligible_for_qualification=pp and tg, adopted=False)


def experiment(kind):
    before, after = COMPARISONS[kind]
    order = (before, after, after, before, before, after)
    ctx = primary.experiment()
    base_pins = ctx.pins
    ctx.KEY = f'dflash2_decode_ranked_generation_{kind}_v1'
    ctx.ROOT, ctx.THIS, ctx.ORDER = bank.B / ctx.KEY.replace('_', '-'), THIS, order
    ctx.protocol = dict(primary.PLAN, order=list(order), comparison=kind,
        scope='Same primary engine, global loader and 1256 physical slots. Change only the ranked file. '
              'Run the same two 96K thinking bodies in three counterbalanced independent pairs.',
        acceptance='Keep identical matched fresh prompts and actual prefill route. Report PP, TTFT, TG, '
                   'generated and accepted draft counts and CPU jobs per output. Require TG gains in at '
                   'least two pairs and a geometric mean TG ratio above one on each workload, with '
                   'median prefill time maintained on both. Advance eligible candidates to task, agent, '
                   'weak-case and full-context qualification.')

    def baseline():
        row = primary.retained.baseline()
        stock = restore_proof(row, restore.KEY, restore.ORDER)
        ranked = restore_proof(row, restore.RANKED_KEY, restore.RANKED_ORDER)
        require(ranked['stock_control_sha256'] == bank.signature(stock), 'ranked runs used a different stock control')
        profiles = {arm: qualified_profile(arm, stock, ranked) for arm in (before, after)}
        return dict(row, qualified_profiles=profiles,
                    restore_evidence=[bank.signature(stock), bank.signature(ranked)])

    def pins(revision):
        paths = base_pins(revision)
        for name in (THIS, primary.THIS, restore.THIS, restore.quota.THIS, 'tools/make_profile.py'):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact ranked comparison input: ' + name)
            paths[name] = bank.sha(path)
        return paths

    def routes(text, samples, arm):
        row = bank.previous.read(ctx.ROOT / 'record.json')
        path = Path(row['qualified_profiles'][arm]['path'])
        return primary.routes(text, samples, 'dflash2',
                              profile_check=lambda log: restore.check_loader(log, path, per_layer=False))

    ctx.baseline, ctx.configuration, ctx.routes, ctx.pins = baseline, configuration, routes, pins
    ctx.evidence_reader.decide = lambda processes: decision(processes, order)
    return ctx


if __name__ == '__main__':
    kind = sys.argv.pop(1)
    require(kind in COMPARISONS, 'select within or cross before the planned stage')
    primary.clock.main(experiment(kind))
