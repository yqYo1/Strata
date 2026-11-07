#!/usr/bin/env python3
"""Source-prepared96-copy admission on the unchanged payload successor.

Main-authored; repository LICENSE. Reuse the owned supervisor and ordered
reader from this project. No XeStrata input. Close both8/32cohorts first;
this diagnostic does not enable96-copy performance or inference adoption.
"""
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_adaptive_payload_entry as positive32
import check_adaptive_payload_failure as refusal32

SELF = Path(__file__).resolve()
KEY = 'adaptive_payload96_native_entry_v1'
REFUSAL_KEY = 'adaptive_payload96_record_failure_v1'
BUDGET = 96
EXPERT_CAP = 64 << 20
CYCLE_CAP = 256 << 20
HEADER_SHA256 = '8387ea06c7fed9afa89c86c2faf528b32abe2366bb2f65a4eb66d5d445ec220a'
require = positive32.require
# Keep qualification independent of later speed-controller revisions.
digest = positive32.budget.original.digest
signature = positive32.budget.original.width.signature


def reader(source=positive32.budget):
    spec = importlib.util.spec_from_file_location('strata_budget96_ordered_reader',
        source.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BUDGET = BUDGET
    return module


def audit(text):
    # Exact ordered96groups are required; partial groups remain visible.
    published = re.findall(r'^strata serve: ADAPT_AUDIT phase=published '
        r'generation=(\d+) swaps=96 identities=3 payload_match=1$', text, re.M)
    require(len(published) >= 2, 'at least two full96-copy publications required')
    result = reader().audit(text)
    result['full96_groups'] = len(published)
    result['refusal_seed_eligible'] = (published[:2] == ['1', '2'] and
        result['request_coverage'][0]['published'] >= 2)
    return result


def native_extent(text):
    lines = [line for line in text.splitlines() if line.strip()]
    require(lines and lines[0].startswith('# strata native experts v3:'),
        'native expert index version')
    header = re.search(r'\(n_expert 512, total (\d+);', lines[0])
    require(header is not None and len(header[1]) <= 20, 'native index geometry')
    total = int(header[1])
    require(0 < total < 2**64, 'native total extent')
    rows = [line.split() for line in lines[1:] if not line.startswith('#')]
    require(len(rows) == 48, 'all48native layers required')
    offset, maximum = 0, 0
    for layer, fields in enumerate(rows):
        require(len(fields) in (8, 9) and
            all(re.fullmatch(r'\d{1,20}', word) for word in fields[:8]),
            'native index row syntax')
        values = list(map(int, fields[:8]))
        require(all(value < 2**64 for value in values) and
            values[0] == layer and values[3] == offset and
            0 < values[4] <= EXPERT_CAP, 'native index order/extent')
        maximum = max(maximum, values[4])
        offset += values[4] * 512
        require(offset < 2**64, 'native arena extent overflow')
    require(offset == total and BUDGET * maximum <= CYCLE_CAP,
        'native arena or96-copy payload bound')
    return dict(layers=48, experts=512, arena_bytes=total,
        maximum_expert_bytes=maximum, maximum_group_bytes=BUDGET * maximum,
        expert_cap_bytes=EXPERT_CAP, cycle_cap_bytes=CYCLE_CAP)


def payload_bound(assets):
    paths = [name for name in assets if Path(name).name == 'native_experts.txt']
    require(len(paths) == 1, 'one pinned native expert index required')
    path = Path(paths[0])
    require(digest(path) == assets[str(path)]['sha256'],
        'native expert index bytes changed')
    return dict(native_extent(path.read_text()), path=str(path),
        sha256=assets[str(path)]['sha256'])


def fixture_group(generation, incoming, outgoing, count=BUDGET):
    prefix = 'strata serve: ADAPT_AUDIT '
    return [prefix + f'phase=prepared generation={generation} incoming={(i//24)*512+incoming+i%24} '
        f'outgoing={(i//24)*512+outgoing+i%24} slot={(i//24)*26+i%24}' for i in range(count)] + [
        prefix + f'phase=masked generation={generation} swaps={count} identities=3',
        prefix + f'phase=published generation={generation} swaps={count} identities=3 payload_match=1']


def fixture_index(blob):
    total = 48 * 512 * blob
    lines = [f'# strata native experts v3: fixture (n_expert 512, total {total}; fixture)']
    lines += [f'{layer} 18 20 {layer*512*blob} {blob} 0 0 0' for layer in range(48)]
    return '\n'.join(lines)


def controls():
    require(positive32.budget.BUDGET == 32, 'original32reader must stay unchanged')
    prefix = 'strata serve: ADAPT_AUDIT '
    footer = [
        prefix + 'phase=dispatch request=1 published=2 calls=48 incoming_vram_entries=3 outgoing_cpu_entries=1 post_swap_refills=0',
        prefix + 'phase=loan-after-positive completed=2 lent=1124 identities=3 payload_match=1',
        prefix + 'phase=dispatch request=2 published=2 calls=96 incoming_vram_entries=8 outgoing_cpu_entries=2 post_swap_refills=1']
    good = fixture_group(1, 100, 0) + fixture_group(2, 300, 100) + footer
    result = audit('\n'.join(good))
    require(result['swaps'] == 192 and result['full96_groups'] == 2, 'two full96groups')
    partial = good[:-1] + fixture_group(3, 400, 300, 9) + [
        good[-1].replace('published=2', 'published=3')]
    require(audit('\n'.join(partial))['swaps'] == 201, 'retain later partial group')
    across_requests = fixture_group(1, 100, 0) + [
        footer[0].replace('published=2', 'published=1'),
        footer[1].replace('completed=2', 'completed=1')
    ] + fixture_group(2, 300, 100) + [footer[2]]
    require(result['refusal_seed_eligible'] and
        not audit('\n'.join(across_requests))['refusal_seed_eligible'],
        'second-record refusal needs two full96groups in its first-request seed')
    bad = [good[:i] + good[i+1:] for i in
        (0, 96, 97, 98, 194, 195, 196, 197, 198)]
    for old, new in (
        ('slot=0', 'slot=1256'), ('incoming=100', 'incoming=24576'),
        ('incoming=100', 'incoming=512'), ('incoming=100', 'incoming=0'),
        ('generation=1', 'generation=2'),
        ('generation=1', 'generation=18446744073709551616'),
        ('generation=1', 'generation=-1')):
        bad.append([good[0].replace(old, new), *good[1:]])
    bad += [
        [good[0], good[1].replace('slot=1', 'slot=0'), *good[2:]],
        [good[0], good[1].replace('incoming=101', 'incoming=100'), *good[2:]],
        [*good[:96], good[97], good[96], *good[98:]],
        [*good, good[-1]], [good[-2], *good],
        [line.replace('swaps=96', 'swaps=95') for line in good],
        [line.replace('identities=3', 'identities=2') for line in good],
        [line.replace('outgoing_cpu_entries=1', 'outgoing_cpu_entries=0') for line in good],
        [line.replace('incoming_vram_entries=8', 'incoming_vram_entries=3') for line in good],
        [line.replace('post_swap_refills=1', 'post_swap_refills=2') for line in good],
        [*good[:-1], good[-1] + ' unknown=1'],
        fixture_group(1, 100, 0) + fixture_group(2, 300, 100, 95) + footer]
    for lines in bad:
        try:
            audit('\n'.join(lines))
        except RuntimeError:
            continue
        raise RuntimeError('corrupt96-copy audit accepted')
    at_cap = CYCLE_CAP // BUDGET
    require(native_extent(fixture_index(at_cap))['maximum_group_bytes'] <= CYCLE_CAP,
        'last fitting payload extent')
    invalid = [
        fixture_index(at_cap + 1), fixture_index(EXPERT_CAP + 1), fixture_index(0),
        fixture_index(256).replace('n_expert 512', 'n_expert 511'),
        fixture_index(256).replace('1 18 20 ', '0 18 20 ', 1),
        fixture_index(256).replace(' 256 0 0 0', ' -1 0 0 0', 1),
        '\n'.join(fixture_index(256).splitlines()[:-1])]
    for text in invalid:
        try:
            native_extent(text)
        except RuntimeError:
            continue
        raise RuntimeError('corrupt96payload extent accepted')
    require(positive32.budget.BUDGET == 32, '96controls changed the32reader')
    return dict(passed=True, GPU_executed=False, full96_groups=2,
        later_partial_group_retained=True, ordered_negative_cases=len(bad),
        payload_negative_cases=len(invalid), original32_reader_unchanged=True,
        first_request_refusal_seed_checked=True,
        record_failure=refusal_controls(),
        scope='CPU-only ordered/index controls;native96 remains unqualified.')


def refusal_audit(text, payload_limit=CYCLE_CAP):
    require(0 < payload_limit <= CYCLE_CAP, '96refusal payload limit')
    result = reader(refusal32.original).audit(text)
    require(result['first_H2D_requested_bytes'] <= payload_limit,
        '96refusal requested payload exceeds the pinned model bound')
    return result


def refusal_controls():
    prefix = 'strata serve: ADAPT_AUDIT '
    first, second = fixture_group(1, 100, 0), fixture_group(2, 300, 100)
    event1 = 'strata: ADAPT_RECORD_DIAGNOSTIC ordinal=1 injected=0 status=0'
    submit = f'strata serve: ADAPT_SUBMIT swaps=96 H2D_requested_bytes={CYCLE_CAP}'
    event2 = 'strata: ADAPT_RECORD_DIAGNOSTIC ordinal=2 injected=1 status=1'
    refused = 'strata serve: cannot record the adaptive refill event'
    good = first[:-1] + [event1, submit, first[-1]] + second[:-1] + [event2, refused]
    result = refusal_audit('\n'.join(good))
    require(result['first_swaps'] == result['queued_second_swaps'] == BUDGET and
        result['stale_event_not_published'], 'two full96groups and unpublished refusal')
    bad = [good[:i] + good[i+1:] for i in range(len(good))]
    bad += [
        [*good, prefix + 'phase=published generation=2 swaps=96 identities=3 payload_match=1'],
        [*good, prefix + 'phase=prepared generation=3 incoming=400 outgoing=300 slot=0'],
        [good[0], *good],
        [good[0].replace('incoming=100', 'incoming=512'), *good[1:]],
        [good[0].replace('slot=0', 'slot=1256'), *good[1:]],
        [good[0].replace('generation=1', 'generation=18446744073709551616'), *good[1:]],
        [line.replace('swaps=96', 'swaps=95') for line in good],
        [line.replace(event1, event1.replace('status=0', 'status=1')) for line in good],
        [line.replace(event2, event2.replace('status=1', 'status=0')) for line in good],
        [line.replace(event2, event2.replace('injected=1', 'injected=0')) for line in good],
        [line.replace(str(CYCLE_CAP), '0') for line in good],
        [line.replace(str(CYCLE_CAP), str(CYCLE_CAP + 1)) for line in good],
        [*good[:-1], good[-1] + ' unknown=1']]
    for lines in bad:
        try:
            refusal_audit('\n'.join(lines))
        except RuntimeError:
            continue
        raise RuntimeError('corrupt96second-record refusal accepted')
    try:
        refusal_audit('\n'.join(good), CYCLE_CAP - 1)
    except RuntimeError:
        pass
    else:
        raise RuntimeError('96refusal ignored its model-specific payload bound')
    require(refusal32.original.BUDGET == 32 and positive32.budget.BUDGET == 32,
        '96refusal controls changed original32readers')
    return dict(passed=True, GPU_executed=False, full_groups=2,
        negative_cases=len(bad) + 1, second_generation_published=False,
        scope='CPU-only ordered refusal;native expected-exit1 remains unqualified.')


def prerequisite_rows(ctx):
    data = ctx.base.committed(ctx.RESULT)
    for workload in ('regular_code', 'varied_prose'):
        row = data['adaptive_payload_generation_' + workload + '_v1']
        require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
            row['ownership_closed'] and not row['active'] and len(row['processes']) == 6 and
            all(p['passed'] and p['complete'] and p['ownership_closed'] and
                p['native_exit_code'] == 0 and not p['fresh_faults'] for p in row['processes']),
            'close both current8/32cohorts before96native work')
    require(data['adaptive_payload_generation_regular_code_v1']['decision']['repeated_TG_gain'],
        'current code TG gain is the96candidate basis;phase preservation remains separate')
    references = {}
    for old in (positive32.context(), refusal32.context()):
        path = old.ROOT / 'record.json'
        record = json.loads(path.read_text())
        require(data[old.KEY] == old.compact(record) and record['passed'] and
            record['complete'] and record['ownership_closed'] and not record['active'],
            'preserve actual successor32positive/refusal qualification')
        references[old.KEY] = dict(path=str(path), sha256=ctx.digest(path))
    return references


def context():
    ctx = positive32.context()
    ctx.ROOT, ctx.KEY, ctx.SELF = ctx.B / 'adaptive-payload96-entry-v1', KEY, SELF
    prior_configuration, prior_current = ctx.configuration, ctx.current

    def configuration():
        config, env, assets = prior_configuration()
        args = config['args']
        require(args[args.index('--adapt-swaps')+1] == '32', 'qualified32basis differs')
        args[args.index('--adapt-swaps')+1] = str(BUDGET)
        return config, env, assets

    def current(row, full=False):
        require(prerequisite_rows(ctx) == row['native32_basis'] and
            row['replacement_budget'] == BUDGET and
            row['host_qualified_header_sha256'] == HEADER_SHA256 and
            row['payload_bound'] == payload_bound(row['model_assets']),
            '96candidate basis/payload bound changed')
        return prior_current(row, full=full)

    ctx.configuration, ctx.current, ctx.audit, ctx.controls = configuration, current, audit, controls
    return ctx


def prepare(ctx):
    references = prerequisite_rows(ctx)
    proof = ctx.admission.build.host_proof()
    require(proof['generated'][ctx.admission.build.source.HEADER] == HEADER_SHA256,
        '96plan requires the unchanged qualified64/256MiB header')
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    try:
        row.update(native32_basis=references, replacement_budget=BUDGET,
            placement_profile=positive32.budget.profile_pin(ctx, row),
            payload_bound=payload_bound(row['model_assets']),
            successor_build_key=ctx.admission.build.KEY,
            host_qualified_header_sha256=HEADER_SHA256,
            predecessor_failed_key='adaptive_budget32_native_entry_v1')
        row['plan'] = (
            'One fresh owned unchanged e1b011 serving engine;synthetic1024/256 then32769/256. '
            'After both8/32cohorts close,change only budget32to96 at fixed8rounds;retain '
            '1256slots,cap4,half1,131072INT8resident,5workers,8192throughput BF16 HC/GDN16/MTP16,'
            '20ms spin and runtime safety. Commit source/config/15asset pins before exec. '
            'Require at least two full96-copy groups,checked completion/full-byte '
            'host-private-device publication,incoming GPU and outgoing CPU dispatch roles,'
            'occupied post-positive loan/refill,actual T1/T4 head windows,ordered two-request '
            'protocol,normalQUIT0,quiet validation,no fresh fault,unchanged final15assets/'
            'boot/driver and closed owners. Max layer blob and96-group bound must fit the '
            'unchanged64/256MiB header before exec. Any refusal/budget stop stays failed;'
            'no retry,narrower group,pin refresh,new build or runtime change. This positive '
            'entry does not qualify the still-required exact96second-record refusal.')
        row['limits'] = (
            'Synthetic96payload/map/dispatch/loan admission only. No clean speed,task '
            'quality,prefill preservation,agent,ordinary teardown,262144/RESTORE or adoption. '
            'Require a separate full96second-record refusal before any96speed comparison.')
    except BaseException as error:
        row.update(stage='failed96prepare', passed=False, complete=False, error=repr(error))
        raise
    finally:
        ctx.save(row)
    print('Prepared96positive entry;commit before native execution.', flush=True)


def refusal_context():
    source = refusal32.original.original
    spec = importlib.util.spec_from_file_location('strata_budget96_owned_refusal', source.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.positive = context()
    ctx.ROOT, ctx.KEY, ctx.SELF = ctx.B / 'adaptive-payload96-record-failure-v1', REFUSAL_KEY, SELF
    prior_current = ctx.current

    def normal_entry():
        record = json.loads((ctx.positive.ROOT / 'record.json').read_text())
        require(ctx.base.committed(ctx.RESULT)[KEY] == ctx.positive.compact(record) and
            record['stage'] == 'closed' and record['passed'] and record['complete'] and
            record['ownership_closed'] and not record['active'] and
            record['outcome']['native_exit_code'] == 0 and
            record['outcome']['adaptive']['maximum_group'] == BUDGET and
            record['outcome']['adaptive']['full96_groups'] >= 2 and
            record['outcome']['adaptive']['refusal_seed_eligible'] is True and
            record['outcome']['adaptive']['outgoing_CPU_observed'] and
            record['outcome']['adaptive']['post_positive_loan_groups'] > 0,
            'closed actual96positive/map/dispatch/loan entry required')
        return record

    def current(row, full=False):
        require(row['successor_positive_key'] == KEY and row['replacement_budget'] == BUDGET and
            row['host_qualified_header_sha256'] == HEADER_SHA256 and
            positive32.budget.profile_pin(ctx, row) == row['placement_profile'] and
            payload_bound(row['model_assets']) == row['payload_bound'],
            '96refusal positive/profile/payload binding changed')
        return prior_current(row, full=full)

    def audit_actual(text):
        return refusal_audit(text, normal_entry()['payload_bound']['maximum_group_bytes'])

    ctx.normal_entry, ctx.current = normal_entry, current
    ctx.audit, ctx.controls = audit_actual, refusal_controls
    return ctx


def prepare_refusal(ctx):
    prerequisite_rows(ctx.positive)
    ctx.prepare()
    row = json.loads((ctx.ROOT / 'record.json').read_text())
    try:
        row.update(successor_positive_key=KEY, replacement_budget=BUDGET,
            host_qualified_header_sha256=HEADER_SHA256,
            placement_profile=positive32.budget.profile_pin(ctx, row),
            payload_bound=payload_bound(row['model_assets']))
        row['plan'] = (
            'One fresh unchanged e1b011 engine after the actual96positive entry closes. '
            'Same96/every8,placement/precision/runtime safety and synthetic1024/max256,except '
            'private paths and pre-exec STRATA_SYCL_TEST_ADAPT_RECORD_FAILURE=1. Require '
            'two exactly96-copy prepared/masked groups,real first record0 and publication, '
            'real second null-record status1,second map unpublished,no later generation, '
            'exact ordered ERR/noDONE wire prefix with emitted tokens,and normal expected '
            'native1 without supervisor cleanup. Keep owners live until checked retirement. '
            'Require quiet validation,unchanged final15assets/boot/driver,no fresh fault '
            'and closed owners. Preserve the immutable positive96receipt. No retry, '
            'narrower group,hash refresh,new engine build or runtime safety change.')
        row['limits'] = (
            'Expected software record-refusal only;no GPU fault injection,successful '
            'generation,clean96speed,quality,prefill preservation,ordinary teardown, '
            'physical262144/RESTORE or adoption. Actual96positive and this expected-exit1 '
            'control must both close before a separately committed96speed plan.')
    except BaseException as error:
        row.update(stage='failed96refusal-prepare', passed=False, complete=False, error=repr(error))
        raise
    finally:
        ctx.save(row)
    print('Prepared96second-record refusal;commit before native execution.', flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    require(args in (['controls'], ['prepare'], ['run'], ['refusal-prepare'], ['refusal-run']),
        'use controls|prepare|run|refusal-prepare|refusal-run')
    ctx = refusal_context() if args[0].startswith('refusal-') else context()
    with (ctx.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        ctx.fcntl.flock(lock, ctx.fcntl.LOCK_EX | ctx.fcntl.LOCK_NB)
        ctx.base.guards.no_model_owners([])
        if args == ['controls']:
            revision = ctx.base.basis.build.git(ctx.P, 'rev-parse', 'HEAD').decode().strip()
            pins = ctx.base.controls(revision, controller=SELF)
            print(json.dumps(dict(controls(), source_commit=revision,
                source_count=len(pins), source_signature_sha256=signature(pins)),
                sort_keys=True))
        elif args == ['prepare']:
            prepare(ctx)
        elif args == ['refusal-prepare']:
            prepare_refusal(ctx)
        else:
            ctx.run()
