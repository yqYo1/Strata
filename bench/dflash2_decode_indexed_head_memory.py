"""Compare gathered and indexed DFlash2 head residency on the same binary."""

from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_decode_indexed_head_fixture as fixture


indexed, handoff = fixture.indexed, fixture.handoff
bank, basis, require, speed = handoff.bank, handoff.basis, handoff.require, handoff.speed
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_indexed_head_memory_v1'
ROOT = bank.B / KEY.replace('_', '-')
READ = 126977
ARMS = ('gathered', 'indexed')
HEAD_BYTES = 223_227_900
GATHERED_DRAFT_BYTES = 845_776_220
HEAD_MARK = 'strata dflash2: indexed Q6_K head rows=106299 canonical_bytes=521472000 owned_head_bytes=0'
PLAN = dict(order=list(ARMS), read=READ, requests_per_process=handoff.RUNS,
    scope='Two independent processes on the same build and stock expert map. Change only the draft head '
          'selector. Reuse both fresh 126977-token bodies with 24 output tokens.',
    gates='Require mode16 and resident BF16 residuals, all four memory marks, matching draft work, '
          'and the actual head route and owned byte difference. Preserve normal provider, owner and fault checks.',
    decision='Advance to an allocation probe only if fixed work matches and each indexed decode memory '
             'sample covers the predeclared expert budget with positive measured gain over the gathered arm. '
             'No allocation or sustained performance claim.',
    retention='Keep both phase references and the indexed comparison build through its next matched check.')


def baseline(build):
    base = handoff.retained.baseline()
    native = bank.previous.read(build.ROOT / 'record.json')
    build.check(native)
    fixture_proof = basis.validation.committed(bank.RESULT, (fixture.KEY,))
    old = basis.validation.committed(bank.RESULT, (handoff.KEY,))
    identity = bank.signature(build.builder.compact(native))
    checked, = fixture_proof['results']
    test, = checked['fixtures']
    require(handoff.retained.closed(native, 'built') and native['overlay_commit'] == indexed.CANDIDATE
            and handoff.retained.closed(fixture_proof, 'indexed_passed')
            and fixture_proof['build_sha256'] == identity and fixture_proof['native_source_commit'] == indexed.CANDIDATE
            and fixture_proof['plans'][0]['commands'][0]['binary'] == native['binaries'][fixture.TARGET]
            and test['cases'] == 14 and test['ownership_closed'] and not checked['fresh_faults']
            and handoff.retained.closed(old, 'closed') and old['base_native_source_commit'] == base['native_source_commit'],
            'require the exact indexed fixture and stock graph evidence')
    stock = handoff.ranked.restore_proof(base, handoff.ranked.restore.KEY, handoff.ranked.restore.ORDER)
    profile = handoff.ranked.qualified_profile('stock', stock, None)
    bodies = old['request_hashes'][str(READ)]
    require(profile == old['qualified_profiles']['stock'] and len(bodies) == handoff.RUNS
            and len(set(bodies)) == handoff.RUNS, 'stock profile or paired request bodies differ')
    base['base_physical_context_sha256'] = base.pop('physical_context_sha256')
    return dict(base, binary=native['binaries']['strata'], native_source_commit=indexed.CANDIDATE,
                native_build_sha256=identity, base_native_source_commit=base['native_source_commit'],
                native_overlay_sha256=bank.signature(native['overlays']), native_context_qualified=False,
                qualified_profiles={'stock': profile}, stock_restore_sha256=bank.signature(stock),
                fixed_bodies={str(READ): bodies}, extra_bytes=old['decision']['replay_extra_bytes'],
                head_fixture_sha256=bank.signature(fixture_proof))


def environment(row, plan):
    return dict(handoff.environment(row, plan), STRATA_SYCL_DFLASH2_INDEXED_HEAD=str(plan['index']))


def route(text, plan):
    frame = handoff.request_route(text, plan)
    arm = ARMS[plan['index']]
    actual = [line for line in text.splitlines() if line.startswith('strata dflash2: indexed Q6_K head ')]
    match, = re.findall(r'^strata DFlash2: eager, prompt_tail_rows=2046 cold_prefix=0 vram_bytes=(\d+)$', text, re.M)
    owned = HEAD_BYTES if arm == 'gathered' else 0
    require(actual == ([] if arm == 'gathered' else [HEAD_MARK])
            and int(match) == GATHERED_DRAFT_BYTES - (HEAD_BYTES - owned),
            'DFlash2 head route, tail or allocated draft bytes differ')
    return dict(frame, head_arm=arm, head_owned_bytes=owned, drafter_owned_bytes=int(match))


def contrast(row):
    gathered, mapped = row['processes']
    require(gathered['read'] == mapped['read'] == READ and len(gathered['routes']) == len(mapped['routes']) == handoff.RUNS,
            'need a matched pair of complete requests for each head')
    work = ('work', 'uploads', 'kv', 'fixed_bytes', 'direct_pieces', 'native_request')
    matched = all(all(x[k] == y[k] for k in work)
                  for x, y in zip(gathered['routes'], mapped['routes'], strict=True))
    gains = [{phase: y['memory'][phase]['free_bytes'] - x['memory'][phase]['free_bytes']
              for phase in handoff.PHASES} for x, y in zip(gathered['routes'], mapped['routes'], strict=True)]
    available = min(min(y['max_single_allocation_bytes'], y['memory'][phase]['free_bytes'])
                    for y in mapped['routes'] for phase in handoff.PHASES[2:])
    margin = available - row['extra_bytes']
    return dict(matched_work=matched, free_byte_deltas=gains, min_indexed_decode_free_or_limit=available,
                extra_capacity_arithmetic_margin_bytes=margin,
                eligible_for_allocation_probe=matched and margin >= 0
                and all(gain[phase] > 0 for gain in gains for phase in handoff.PHASES[2:]))


def experiment():
    build = indexed.build_controller()
    observer = basis.core.source.device.load_module('indexed_head_memory_observer', handoff.__file__)
    observer.ROOT, observer.KEY, observer.THIS, observer.ORDER = ROOT, KEY, THIS, (READ,) * len(ARMS)
    observer.baseline = lambda _: baseline(build)
    observer.PLAN, observer.environment, observer.request_route = PLAN, environment, route
    ctx = observer.experiment()
    pins, run_trial = ctx.pins, ctx.run

    def sources(revision):
        result = pins(revision)
        for name in (handoff.THIS, indexed.THIS, fixture.THIS):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name), 'changed indexed head input: ' + name)
            result[name] = bank.sha(path)
        return result

    def run(index):
        run_trial(index)
        if index == len(ARMS) - 1:
            row = bank.previous.read(ROOT / 'record.json')
            require(handoff.retained.closed(row, 'closed'), 'close the indexed memory pair')
            row['decision']['headroom_comparison'] = contrast(row)
            ctx.save(row)

    ctx.pins, ctx.run = sources, run
    return ctx


if __name__ == '__main__':
    speed.clock.main(experiment())
