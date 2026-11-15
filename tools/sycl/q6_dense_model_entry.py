#!/usr/bin/env python3
"""Observe independent dense-DPAS/backing choices in the real model; Main only.

Reuse our maintained workload, supervision and runtime readers. These bounded
thinking-on handoffs qualify routing, not throughput or task quality.
Main-authored; repository LICENSE applies. No external implementation copied.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import build_q6_dense_model as build
import check_q6_dense_model as device
import q6_head_model_entry as previous

P, B, RESULT = build.P, build.B, build.primary.RESULT
ROOT, KEY = B / 'q6-dense-head-backing-model-v1', 'q6_dense_head_backing_model_v1'
THIS = 'tools/sycl/q6_dense_model_entry.py'
ARMS = ((0, 0), (1, 0), (0, 1), (1, 1))
SHAPES = ((6144, 2560), (2560, 6144), (2560, 10240), (2560, 12288))
IDENTITY = ('request', 'device', 'phase', 'tokens', 'capture_id')
require, sha, write, signature = build.require, build.sha, build.write, build.signature
owner, guards, validation = previous.owner, previous.guards, previous.validation


def native(expected=None):
    ctx = build.context(); n = json.loads((build.ROOT / 'record.json').read_text())
    require(validation.committed(RESULT, (build.KEY,)) == ctx.compact(n) and n['stage'] == 'built' and
        n['passed'] and n['complete'] and n['ownership_closed'] and not n['active'] and
        n['policy_check']['passed'], 'closed fresh native build/CPU policy required')
    before = n['admission_project']; project = ctx.source.pin(P, ctx.PREFIXES)
    a, b = before['source_sha256'], project['source_sha256']
    observers = {device.THIS, THIS}
    require(a == {p: hashlib.sha256(raw).hexdigest() for p, raw in
        ctx.source.blobs(P, before['revision'], a)} and set(b) - set(a) == observers and
        not set(a) - set(b) and all(b[p] == h for p, h in a.items()),
        'only the two committed device/model observers may extend the exact build admission')
    require(n['native_source_commit'] == n['project']['revision'] == n['actual_inputs']['source_commit'] and
        build.prerequisites() == n['prerequisites'], 'original native source/fallback proof changed')
    ctx.build.check_inputs(dict(n, project=project))
    require(all(sha(p) == h for p, h in (n['generated_inputs'] | n['configured_files']).items()),
        'generated/configured recipe changed')
    actual = n['actual_inputs']
    require(sha(actual['path']) == actual['sha256'] and actual['uncovered'] == 0, 'compiler census changed')
    census = json.loads(Path(actual['path']).read_text())
    require(all(sha(p) == h for field in ('compiled', 'generated', 'links') for p, h in census[field].items()) and
        all(sha(v['path']) == v['sha256'] and Path(v['path']).stat().st_size == v['bytes']
            for v in n['binaries'].values()), 'actual native inputs/five ELFs changed')
    prior = validation.committed(RESULT, (device.KEY,))
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete'] and prior['ownership_closed'] and
        not prior['active'] and prior['binaries'] == n['binaries'] and
        [v['mode'] for v in prior['results']] == list(device.STAGES) and
        all(v['passed'] and v['ownership_closed'] for v in prior['results']) and
        prior['native_observer_admission']['added_observers'] == {device.THIS: b[device.THIS]},
        'exact same-build eight-process device qualification required')
    proof = dict(native_source_commit=n['native_source_commit'], build_admission_source_commit=before['revision'],
        original_source_files=before['count'], observer_source_files=project['count'],
        observer_source_sha256=signature(b), added_observers={p: b[p] for p in sorted(observers)},
        exact_five_ELFs_census_recipe=True, native_recompiled=False, earlier_model_result_inherited=False)
    require(expected is None or proof == expected, 'model observer admission changed')
    n['model_observer_admission'] = proof
    return n


def context(admission=None):
    spec = importlib.util.spec_from_file_location('dense_model_entry_context', previous.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.build = ROOT, KEY, build.context()
    ctx.device = SimpleNamespace(KEY=device.KEY)
    ctx.native = lambda: native(admission)
    ctx.controller_pin = lambda: ctx.build.source.pin(P,
        ['tools/sycl', 'tools/strata_tokenizer.py', 'data/expert-profile.bin'])
    return ctx


def fields(line):
    pairs = [p.split('=', 1) for p in line.split()[2:]]
    require(all(len(p) == 2 for p in pairs), 'malformed route fields')
    value = dict(pairs)
    require(len(value) == len(pairs), 'duplicate route fields')
    return value


def routes(text, dense):
    require(type(dense) is int and dense in (0, 1), 'explicit dense choice')
    head = previous.routes(text, 2); base, _ = previous.route_tables(text, 2)
    legacy, selected = {key: {} for key in base}, {key: {} for key in base}
    for line in text.splitlines():
        if not line.startswith(('strata Q6_SHAPE ', 'strata Q6_DPAS_DENSE ')): continue
        f = fields(line); key = tuple(f.get(k) for k in IDENTITY)
        require(key in base, 'shape must join one complete live production capture')
        if line.startswith('strata Q6_DPAS_DENSE '):
            require(dense == 1 and set(f) == set(IDENTITY) | {'ni', 'no', 'nc', 'layout',
                'extra_weight_bytes', 'canonical_table_excludes_dpas', 'calls', 'complete'} and
                f['layout'] == 'canonical' and f['nc'] == '4' and f['extra_weight_bytes'] == '0' and
                f['canonical_table_excludes_dpas'] == f['complete'] == '1' and
                all(f[k].isdecimal() for k in ('ni', 'no', 'calls')), 'dense route schema/layout')
            shape = (int(f['ni']), int(f['no']))
            require(shape in SHAPES and shape not in selected[key], 'unexpected/duplicate dense shape')
            calls = int(f['calls'])
            require((calls > 0) == (key[2:4] == ('decode', '4')), 'dense route escaped decode-C4')
            selected[key][shape] = calls
        else:
            names = ('ni', 'no', 'nc', 'reference_calls', 'columns_calls', 'centered_calls', 'native_half_calls')
            require(set(f) == set(IDENTITY) | set(names) and all(f[k].isdecimal() for k in names),
                'legacy shape schema')
            shape = tuple(int(f[k]) for k in names[:3]); calls = tuple(int(f[k]) for k in names[3:])
            require(shape not in legacy[key] and shape[0] > 0 and shape[1] > 0 and
                1 <= shape[2] <= 4 and sum(calls) > 0 and calls[2:] == (0, 0), 'legacy shape policy/extent')
            require(key[2] == 'decode' or calls[1] == 0, 'columns escaped decode phase')
            legacy[key][shape] = calls
    normalized, dense_counts, small = {}, None, None
    for key in base:
        require(base[key].get('shapes', '').isdecimal() and int(base[key]['shapes']) == len(legacy[key]),
            'complete canonical shape count required')
        require(set(selected[key]) == (set(SHAPES) if dense else set()), 'all four dense rows required per capture')
        table = dict(legacy[key])
        if key[2:4] == ('decode', '4'):
            counts = {}
            for shape in SHAPES:
                cell = (*shape, 4)
                if dense:
                    require(cell not in table, 'dense cell counted again as canonical columns')
                    counts[f'{shape[0]}x{shape[1]}'] = selected[key][shape]
                else:
                    require(cell in table and table[cell][0] == 0 and table[cell][1] > 0,
                        'control must use production columns for each selected cell')
                    counts[f'{shape[0]}x{shape[1]}'] = table.pop(cell)[1]
            fallback = table.get((2560, 512, 4))
            require(fallback is not None and fallback[0] > 0 and fallback[1:] == (0, 0, 0),
                'losing small C4 cell must retain its generic route')
            require(dense_counts is None or dense_counts == counts, 'static dense calls changed across captures')
            require(small is None or small == fallback[0], 'small fallback changed across captures')
            dense_counts, small = counts, fallback[0]
        label = '/'.join(key[2:4])
        canonical = {'x'.join(map(str, cell)): list(calls) for cell, calls in sorted(table.items())}
        require(label not in normalized or normalized[label] == canonical,
            'static canonical calls changed within one phase/width')
        normalized[label] = canonical
    require(dense_counts and any(k.startswith('prompt/') for k in normalized), 'actual prompt and dense-C4 routes required')
    return dict(head=head, dense_enabled=dense, selected_C4_calls=dense_counts, generic_small_C4_calls=small,
        unchanged_routes_sha256=signature(normalized), phase_widths=sorted(normalized),
        phase_width_sha256={key: signature(value) for key, value in sorted(normalized.items())},
        scope='Static captured call counts;live replay extents are separate. Four selected decode-C4 cells excluded from isolation hash.')


def handoffs(text, backing):
    require(type(backing) is int and backing in (0, 1), 'explicit backing choice')
    size = 521472000; source = 'initial-RAM' if backing else 'owned-RAM'; d2h = 0 if backing else size
    target = re.findall(r'^strata prefill heads: target detached bytes=(\d+) source=(\S+) D2H=(\d+) H2D-on-restore=(\d+)$', text, re.M)
    require(target == [(str(size), source, str(d2h), str(size))] * 2, 'two actual target retire/restore transitions required')
    lines = text.splitlines()
    expected = {
        'strata prefill heads: draft detached ': ['strata prefill heads: draft detached bytes=0 retained-map-bytes=425196'] * 2,
        'strata native head packed:': [f'strata native head packed: ready=1 columns=1..4 weight_bytes={size} canonical_device_weight_bytes=0 extra_device_weight_bytes=0 permanent_host_weight_bytes={size if backing else 0} default_adopted=0'],
        'strata native head backing:': [f'strata native head backing: source=initial-upload layout=packed bytes={size} immutable=1 request_D2H_bytes=0 default_adopted=0'] if backing else [],
        'strata mtp: shared packed head ': ['strata mtp: shared packed head tokens=106299 subset_weight_bytes=0 subset_logit_bytes=425196 saved_device_bytes=222802704']}
    require(all([v for v in lines if v.startswith(prefix)] == value for prefix, value in expected.items()),
        'packed/backing/MTP ownership admission changed')
    return dict(transitions=2, target_freed_per_transition=size, target_D2H_per_transition=d2h,
        target_H2D_per_transition=size, permanent_host_weight_bytes=size if backing else 0,
        extra_device_weight_bytes=0, draft_freed_per_transition=0, retained_map_bytes=425196,
        scope='Declared actual allocation/copy requests;not physical bus traffic or bandwidth.')


def controls():
    old = previous.parser_controls()
    def sample(dense):
        out = []
        for request, capture, replay in ((1, 2, 10), (2, 2, 18), (3, 6, 10)):
            for phase, tokens in (('prompt', 4), ('decode', 1), ('decode', 4)):
                identity = f'request={request} device=0 phase={phase} tokens={tokens} capture_id={capture}'
                cells = [(2560, 512, (21, 0, 0, 0))]
                calls = (40, 14, 22, 5)
                active = phase == 'decode' and tokens == 4
                if not (dense and active):
                    cells += [(ni, no, (0, count, 0, 0) if phase == 'decode' else (count, 0, 0, 0))
                        for (ni, no), count in zip(SHAPES, calls)]
                out += ['strata Q6_ROUTE ' + identity + f' scope=live_capture_cumulative native_half_enabled=0 complete=1 shapes={len(cells)} replay_started={replay} replay_completed={replay}',
                    'strata Q6_PACKED_HEAD ' + identity + ' layout=sole-packed extra_weight_bytes=0 calls=1 complete=1']
                out += ['strata Q6_SHAPE ' + identity + f' ni={ni} no={no} nc={tokens} reference_calls={c[0]} columns_calls={c[1]} centered_calls={c[2]} native_half_calls={c[3]}' for ni, no, c in cells]
                if dense:
                    out += ['strata Q6_DPAS_DENSE ' + identity + f' ni={ni} no={no} nc=4 layout=canonical extra_weight_bytes=0 canonical_table_excludes_dpas=1 calls={count if active else 0} complete=1'
                        for (ni, no), count in zip(SHAPES, calls)]
        return '\n'.join(out)
    reference, candidate = routes(sample(0), 0), routes(sample(1), 1)
    require(all(reference[k] == candidate[k] for k in ('selected_C4_calls', 'generic_small_C4_calls', 'unchanged_routes_sha256')),
        'dense switch must preserve every other phase/width/shape')
    text = sample(1); marker = next(v for v in text.splitlines() if v.startswith('strata Q6_DPAS_DENSE '))
    active = next(v for v in text.splitlines() if v.startswith('strata Q6_DPAS_DENSE ') and 'phase=decode tokens=4' in v)
    bad = [text.replace(marker, '', 1), text + '\n' + marker,
        text.replace(marker, marker.replace('calls=0', 'calls=1'), 1),
        text.replace(active, active.replace('calls=40', 'calls=0'), 1),
        text.replace(marker, marker.replace('nc=4', 'nc=3'), 1),
        text.replace(marker, marker.replace('ni=6144', 'ni=512'), 1),
        text.replace(marker, marker.replace('layout=canonical', 'layout=packed'), 1),
        text.replace(marker, marker.replace('extra_weight_bytes=0', 'extra_weight_bytes=1'), 1),
        text.replace(marker, marker.replace('canonical_table_excludes_dpas=1', 'canonical_table_excludes_dpas=0'), 1),
        text.replace(marker, marker.replace('complete=1', 'complete=0'), 1),
        text.replace(marker, marker + ' calls=0', 1), text.replace('shapes=1', 'shapes=2', 1),
        text.replace('reference_calls=21', 'reference_calls=0', 1), text.replace('centered_calls=0', 'centered_calls=1', 1)]
    for value in bad:
        try: routes(value, 1)
        except RuntimeError: continue
        raise RuntimeError('dense parser accepted malformed/escaped/double-counted routing')
    try: routes(text, 0)
    except RuntimeError: pass
    else: raise RuntimeError('disabled dense choice accepted dense markers')
    for backing in (0, 1):
        size = 521472000
        text = '\n'.join([
            f'strata native head packed: ready=1 columns=1..4 weight_bytes={size} canonical_device_weight_bytes=0 extra_device_weight_bytes=0 permanent_host_weight_bytes={size if backing else 0} default_adopted=0',
            'strata mtp: shared packed head tokens=106299 subset_weight_bytes=0 subset_logit_bytes=425196 saved_device_bytes=222802704',
            *([f'strata native head backing: source=initial-upload layout=packed bytes={size} immutable=1 request_D2H_bytes=0 default_adopted=0'] if backing else []),
            *[v for _ in range(2) for v in (f'strata prefill heads: target detached bytes={size} source={"initial-RAM" if backing else "owned-RAM"} D2H={0 if backing else size} H2D-on-restore={size}',
                'strata prefill heads: draft detached bytes=0 retained-map-bytes=425196')]])
        handoffs(text, backing)
        for value, arm in ((text, 1 - backing), (text.replace(f'H2D-on-restore={size}', 'H2D-on-restore=0', 1), backing),
            (text.replace('subset_weight_bytes=0', 'subset_weight_bytes=1'), backing),
            (text.replace('extra_device_weight_bytes=0', 'extra_device_weight_bytes=1'), backing)):
            try: handoffs(value, arm)
            except RuntimeError: continue
            raise RuntimeError('backing parser accepted wrong policy/restore/owner')
    return dict(passed=True, prior_head_controls=old, dense_positive_arms=2, corrupt_dense_cases=len(bad) + 1,
        backing_positive_arms=2, corrupt_backing_cases=8, GPU_executed=False)


def prepare():
    require(not ROOT.exists() and KEY not in build.primary.public(), 'fresh model plan;no overwrite/retry')
    n = native(); ctx = context(n['model_observer_admission'])
    old = json.loads((previous.ROOT / 'record.json').read_text())
    require(validation.committed(RESULT, (previous.KEY,)) == previous.compact(old) and old['stage'] == 'closed' and
        old['passed'] and old['complete'] and old['ownership_closed'], 'committed existing workload/placement template required')
    prior = validation.committed(RESULT, (device.KEY,))
    require(old['probe'] == prior['native_validation_probe'] and old['runtime_manifest'] == prior['runtime_manifest'] and
        old['boot'] == prior['boot'] and old['driver'] == prior['driver'], 'current device/runtime template required')
    names = ('models', 'models_sha256', 'model_files', 'model_bytes', 'runtime_manifest', 'runtime_files',
        'probe', 'generator_runtime', 'tokenizer', 'requests', 'environment', 'argv', 'boot', 'driver', 'budgets')
    row = copy.deepcopy({k: old[k] for k in names})
    row['argv'][0] = n['binaries']['strata']['path']
    row['environment'].update(STRATA_SYCL_DECODE_Q6_DPAS_HEAD='2', STRATA_SYCL_DECODE_Q6_DPAS_DENSE='0',
        STRATA_PREFILL_HEAD_HOST_BACKING='0')
    require(row['environment']['STRATA_PREFILL_RELEASE_HEADS'] == '1' and
        row['requests'] == ctx.requests(Path(row['tokenizer'])), 'preserve the checked long/short thinking workload')
    row.update(stage='prepared', source_commit=ctx.build.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        controller=ctx.controller_pin(), native_source_commit=n['native_source_commit'],
        native_build=dict(binaries=n['binaries']), native_record=dict(path=str(build.ROOT / 'record.json'),
            sha256=sha(build.ROOT / 'record.json')), native_observer_admission=n['model_observer_admission'],
        device_evidence_sha256=signature(prior), arms=[dict(dense=d, backing=b) for d, b in ARMS],
        template=dict(key=previous.KEY, evidence_sha256=signature(previous.compact(old)),
            scope='Workload/placement/runtime template only;no earlier model outcome inherited.'),
        plan='Four fresh ordered processes:dense0/backing0,dense1/backing0,dense0/backing1,dense1/backing1;'
            'head2/spec4 fixed. In each:10197/82/10197thinking inputs,32outputs per request,two long head'
            ' retire/restore cycles,actual resident128K/1256cache/chunk8192 placement,immutable pre-exec flags,'
            'pre-main/maps proof,positive offered/accepted drafts,all live head and selected dense joins.'
            ' Selected four C4call counts must match the control;all commonly observed canonical phase/width'
            ' tables must match after excluding those four cells. Require shared prompt and decode-C4;'
            ' differing optional visited widths are recorded,not treated as output-equality failures.'
            ' Exact source/5ELFs/census/recipe and device evidence required. Full15asset hashes before preparation'
            ' and final closure,identity checks before/after each process. Commit every closed stage;stop first failure,no retry.',
        limits='Routing/handoff diagnosis only;not clean PP/TTFT/TG,task quality,physical262144/RESTORE,'
            'ordinary provider teardown or adoption. Independent matched bench.sh long/agent and strong/weak'
            'thinking generation remain required. Current physical limiting resource is unknown.',
        results=[], commands=[], passed=False, complete=False, active=False, ownership_closed=True,
        model_executed=False, speed_eligible=False, model_adoption_qualified=False,
        retention='Keep useful per-request extents,draft/route/transfer counts and outcomes. Retire successful'
            'diagnostic traces/caches after closure and verified source recovery. Keep a347/5fixtures and1db'
            'for imminent matched comparison;separate named phase/feature consumers and legacy holds unchanged.')
    ctx.assets(row, full=True)
    ROOT.mkdir(mode=0o700)
    for dense, backing in ARMS:
        arm = ROOT / f'dense{dense}-backing{backing}'; arm.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (arm / name).mkdir(mode=0o700)
    ctx.save(row); print('Prepared four independent real-model choices;commit before host controls.', flush=True)


def execute(stage):
    row = json.loads((ROOT / 'record.json').read_text()); ctx = context(row['native_observer_admission']); ctx.verify(row)
    if stage == 'host':
        require(row['stage'] == 'prepared' and not row['results'], 'ordered host stage;no retry')
        try: row['parser_controls'] = controls(); row['stage'] = 'host_passed'
        except BaseException as error: row.update(stage='failed', error=repr(error)); ctx.save(row); raise
        ctx.save(row); print('Dense/backing parser controls pass;commit before actual model.', flush=True); return
    index = int(stage); dense, backing = ARMS[index]; label = f'dense{dense}-backing{backing}'
    require(row['stage'] in ('host_passed', 'in_progress') and len(row['results']) == index and
        all(v['passed'] for v in row['results']), 'ordered fresh model process;no retry')
    scope = None; result = dict(dense=dense, backing=backing, passed=False, requests=[])
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if v.startswith('-- cursor: '))
    def persist():
        write(ROOT / (label + '-execution.json'), dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []))
    try:
        ctx.assets(row); guards.no_model_owners([])
        scope = owner.Owner(ROOT / ('owned-' + label), persist)
        arm = ROOT / label
        env = dict(row['environment'], TMPDIR=str(arm / 'tmp'), XDG_CACHE_HOME=str(arm / 'cache'),
            PYTHONPYCACHEPREFIX=str(arm / 'cache/python'), STRATA_MEASUREMENT_ROOT=str(arm),
            STRATA_SYCL_DECODE_Q6_DPAS_DENSE=str(dense), STRATA_PREFILL_HEAD_HOST_BACKING=str(backing),
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(arm / 'validation.log'))
        def interact(io, command):
            startup = []
            while True:
                line = io.line(480); startup.append(line); require(len(startup) <= 32, 'startup stdout cap')
                if line.startswith('READY '): break
            require(line == 'READY 131072 stop', 'actual context/READY differs')
            info = [v[5:] for v in startup if v.startswith('INFO ')]
            require(len(info) == 1, 'one actual INFO required')
            result['placement'] = dict(v.split('=', 1) for v in info[0].split()); previous.check_placement(result['placement'])
            actual = dict(v.split('=', 1) for v in Path('/proc', str(command['pid']), 'environ').read_text().split('\0') if '=' in v)
            require(actual == env, 'native environment changed')
            result.update(previous.model.runtime_evidence(dict(row, environment=env), command['pid'],
                (scope.output / (label + '.stderr')).read_text(errors='replace')))
            result['memory_at_READY'] = previous.memory_snapshot(command['pid'])
            print(label + ': READY with exact runtime/placement.', flush=True)
            for name in ('long', 'short', 'long'):
                request = row['requests'][name]
                parser = previous.protocol.RequestProtocol(request['prompt_tokens'], request['generated'])
                io.send(previous.model.wire(request['ids'], request['generated']))
                while parser.done is None: parser.feed(io.line(480))
                require(int(parser.done[7]) > 0 and int(parser.done[6]) > 0, 'actual positive MTP offers/acceptance required')
                result['requests'].append(dict(label=name, prompt_tokens=request['prompt_tokens'], generated=len(parser.tokens),
                    accepted=int(parser.done[6]), offered=int(parser.done[7]), output_ids_sha256=signature(parser.tokens), complete=True))
                row['model_executed'] = True; print(label + ': ' + name + ' complete.', flush=True)
            result['memory_after_requests'] = previous.memory_snapshot(command['pid'])
            io.send(b'QUIT\n'); io.exit(30)
        result['native_execution_attempted'] = True
        e, out, err = scope.run(label, row['argv'], env, P, interaction=interact, **row['budgets'])
        result['native_exit_code'] = e.get('exit_code')
        require(owner.closed(e) and e['exit_code'] == 0, 'owned actual model closure failed')
        text = err.read_text(errors='replace'); result['routes'] = routes(text, dense)
        result['handoffs'] = handoffs(text, backing)
        if index:
            reference = row['results'][0]['routes']
            require(all(result['routes'][k] == reference[k] for k in
                ('selected_C4_calls', 'generic_small_C4_calls')), 'selected or generic-small call counts changed')
            before, after = reference['phase_width_sha256'], result['routes']['phase_width_sha256']
            common = set(before) & set(after)
            require('decode/4' in common and any(k.startswith('prompt/') for k in common) and
                all(before[k] == after[k] for k in common), 'preserved phase/width/shape routing changed')
            result['route_isolation'] = dict(common=sorted(common), reference_only=sorted(set(before) - set(after)),
                candidate_only=sorted(set(after) - set(before)), passed=True,
                scope='Compare common static phase/width tables;MTP may visit different optional widths.')
        log = arm / 'validation.log'
        require(not re.search(r'UR_RESULT_ERROR_[A-Z_]+', text) and
            (not log.exists() or log.stat().st_size == 0), 'actual runtime/parameter-validation error')
        ctx.assets(row, full=index == len(ARMS) - 1)
        result['host_probe_accepted'] = True
    except BaseException as error:
        result.update(host_probe_accepted=False, error=repr(error))
    finally:
        if scope: validation.finish(result, scope, cursor, row['boot'])
        if result.get('driver_after') != row['driver']: result['passed'] = False
        row['results'].append(result); row['commands'] += scope.commands if scope else []
        row['active'] = bool(scope and scope.active)
        row['ownership_closed'] = not row['active'] and all(owner.closed(v) for v in row['commands'])
        row['stage'] = 'failed' if not result['passed'] else 'closed' if index == len(ARMS) - 1 else 'in_progress'
        row['complete'] = row['stage'] == 'closed'; row['passed'] = bool(row['complete'] and row['ownership_closed'])
        ctx.save(row)
    require(result['passed'], result.get('error', 'dense/backing real-model stage failed'))
    print(label + ' closed;commit before next stage. No speed/adoption claim.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', '0', '1', '2', '3'),
        'q6_dense_model_entry.py prepare|host|0|1|2|3')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
