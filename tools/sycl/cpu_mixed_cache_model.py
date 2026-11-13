#!/usr/bin/env python3
"""Owned mixed-cache model admission; repository LICENSE. No speed claim.

Reuse the owned coherent model transport/workloads, not an old native build.
All runtime choices are fixed in the native child's environment before exec.
"""
import copy
import fcntl
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_cpu_mixed_cache_model as device
import coherent_prefill_model as base
import cpu_work_census as census

P, B, RESULT = device.P, device.B, device.RESULT
ROOT, KEY = B / 'cpu-mixed-cache-model-v1', 'cpu_mixed_cache_model_v1'
THIS = 'tools/sycl/cpu_mixed_cache_model.py'
require, sha = device.require, device.sha
JOINT = re.compile(r'^strata cpu schedule: layer=(\d+) gu=(\d+) down=(\d+) jobs=(\d+) '
    r'nt1=(\d+) nt2=(\d+) nt3=(\d+) nt4=(\d+) other=(\d+) whole_calls=(\d+) '
    r'cache_policy=(small|mixed) bound_nt1=(\d+)$')
original_route = base.route


def native(expected=None, *, observers=()):
    extras = {THIS, *observers}
    require(all(p.startswith('tools/sycl/') and p.endswith('.py') for p in extras),
        'only explicitly declared model observers')
    n = device.native(observers={device.THIS, *extras})
    prior = base.original.validation.committed(RESULT, (device.KEY,))
    require(prior['stage'] == 'closed' and all(prior[k] for k in
        ('passed', 'complete', 'ownership_closed')) and not prior['active']
        and prior['binaries'] == n['binaries']
        and prior['native_source_commit'] == n['native_source_commit']
        and [r['mode'] for r in prior['results']] == list(device.STAGES)
        and all(r['passed'] and r['ownership_closed'] and r['native_exit_code'] == 0
            for r in prior['results']), 'all eleven same-build native stages first')
    proof = copy.deepcopy(n['observer_admission'])
    after = base.original.device.common.source.pin(P,
        base.original.device.common.PREFIXES)['source_sha256']
    require(base.original.device.common.signature(after) == proof['observer_source_sha256'],
        'actual observer/source snapshot')
    for name in extras:
        require(name in proof['declared_observers'] and name in after,
            'explicit new observer missing')
        del proof['declared_observers'][name]
        del after[name]
        proof['observer_source_files'] -= 1
    proof['observer_source_sha256'] = base.original.device.common.signature(after)
    require(proof == prior['native_observer_admission'],
        'recover exact qualified source, not a refreshed native proof')
    host = prior['results'][0]
    require(host['reader_cases'] == 54 and host['native_cases'] == 11
        and host['mixed_cache_host']['actual_dispatch_cases'] == 6
        and host['mixed_cache_host']['policy_refusals'] == 6
        and host['mixed_cache_host']['passed'] and host['physical_context_host']['passed']
        and host['policy_check']['passed'] and host['host_census_controls']['passed']
        and host['selected_head_IR']['same_as_current_reference']
        and host['selected_head_IR']['engine_fixture_identical'], 'host/IR/phase controls')
    require(all(r['GPU_executed'] and r['native_execution_attempted']
        and r['host_probe_accepted'] and not r['fresh_faults'] and r['boot_unchanged']
        and r['driver_after'] == prior['driver'] for r in prior['results'][1:]),
        'validated GPU closures without new faults')
    qsa = prior['results'][6]['checks']
    require(len(qsa) == 1 and sum(s.startswith('case cells=') for s in qsa[0]['cases']) == 11
        and any(s.startswith('case cells=262144 live_cells=262144 ')
            for s in qsa[0]['cases']), 'attention final-cell boundary')
    producer = prior['results'][8]['checks']
    require([v['flag'] for v in producer] == ['0', '1']
        and all(v['physical_cells'] == 262144 for v in producer)
        and producer[1]['bulk_calls'] == 33 and producer[1]['accepted_copy_commands'] == 136,
        'producer complete extent and overwrite')
    numeric = prior['results'][-1]['numerics']
    require(numeric['correctness_cases'] == 44 and numeric['max_half_ulp'] <= 1
        and numeric['BF16_capture_exact'] and numeric['guards'] and numeric['inputs_immutable'],
        'same-build GU numerical controls')
    proof = dict(n['observer_admission'], device_key=device.KEY,
        device_evidence_sha256=base.original.signature(prior), fresh_eleven_stage_controls=True,
        model_observer=THIS, additional_observers=sorted(set(observers)),
        original_partial_relink_or_model_result_inherited=False)
    require(expected is None or expected == proof, 'frozen model/native admission changed')
    return dict(n, model_observer_admission=proof)


def startup(text, formats, budget_mib):
    layers = sum(gu == 18 and down in (20, 42) for gu, down, *_ in formats)
    entries, budget = layers * 512, budget_mib << 20
    begin = re.findall(r'^CPU_COMPACT_GU_BEGIN budget_bytes=(\d+) layers=(\d+)$', text, re.M)
    prepared = re.findall(r'^CPU_COMPACT_GU_PREPARED phase=decode nt=1 entries=(\d+) '
        r'known_bytes=(\d+) budget_bytes=(\d+) setup_s=([0-9.]+)$', text, re.M)
    ready = re.findall(r'^CPU_COMPACT_GU_READY phase=decode nt=1 max_jobs=40 entries=(\d+) '
        r'known_bytes=(\d+) policy=mixed max_total_nt=40$', text, re.M)
    require(layers > 0 and len(begin) == len(prepared) == len(ready) == 1
        and not re.search(r'^CPU_COMPACT_GU_OFF\b', text, re.M), 'one admitted ready cache')
    ne, known, cap = map(census.integer, prepared[0][:3])
    setup = float(prepared[0][3])
    payload = entries * 2 * 6400 * 134
    require(tuple(map(census.integer, begin[0])) == (budget, layers) and ne == entries
        and cap == budget and tuple(map(census.integer, ready[0])) == (entries, known)
        and payload <= known <= min(budget, payload + entries*4096 + (1 << 20))
        and math.isfinite(setup) and setup >= 0, 'cache geometry/bytes/readiness mismatch')
    return dict(layers=layers, entries=entries, known_bytes=known, budget_bytes=budget, setup_s=setup)


def observe(text, formats, membership, prompts, budget_mib=16384):
    require(len(text) <= 64 << 20 and len(formats) == len(membership) == 48,
        'bounded complete target-model trace')
    setup = startup(text, formats, budget_mib)
    require(re.findall(r'^CPU_DECODE_SCHEDULE whole_mixed=(\d+) down=42 jobs=24,36 '
        r'nt=mixed_1_to_4 workers=5 host=1$', text, re.M) == ['0'], 'whole selector stays off')
    frames, current = [], None
    for line in text.splitlines():
        if line.startswith(census.PREFIX):
            kind, values = census.rows(line + '\n')[0]
            if kind == 'begin':
                require(current is None and len(frames) < len(prompts)
                    and values['phase'] == 'decode'
                    and census.integer(values['request']) == len(frames)+1
                    and census.integer(values['prompt_tokens']) == prompts[len(frames)],
                    'exact request framing')
                current = dict(prompt_tokens=prompts[len(frames)], windows=[], calls=0,
                    cpu_jobs=0, cpu_entries=0, bound_nt1=0, cache_callbacks=0, jobs_NT=[0]*8)
            elif kind == 'end':
                require(current is not None and current['windows']
                    and current['calls'] == 48*len(current['windows'])
                    and census.integer(values['callbacks']) == current['calls'], 'complete callback windows')
                frames.append(current); current = None
            continue
        window = re.fullmatch(r'strata trace: window (\d+) (\d+)', line)
        if window and current is not None:
            pos, width = map(int, window.groups())
            require(1 <= width <= 4 and current['calls'] == 48*len(current['windows']), 'window order')
            if current['windows']:
                before, extent = current['windows'][-1]
                require(1 <= pos-before <= extent, 'advancing target positions')
            current['windows'].append((pos, width))
        if not line.startswith('strata cpu schedule:'): continue
        m = JOINT.fullmatch(line)
        require(current is not None and m and current['windows'], 'bound callback syntax')
        layer, gu, down, jobs, *tail = map(census.integer, m.groups()[:10])
        nt, other, whole = tail[:4], tail[4], tail[5]
        rows = sum((i+1)*n for i, n in enumerate(nt)); width = current['windows'][-1][1]
        require(layer == current['calls'] % 48 and current['calls']//48 == len(current['windows'])-1
            and (gu, down) == formats[layer][:2] and sum(nt) == jobs
            and jobs <= 10*width and rows <= 10*width and not any(nt[width:])
            and other == whole == 0, 'actual format/NT/physical volume/row route')
        eligible = gu == 18 and down in (20, 42) and 0 < jobs <= 40
        bound = census.integer(m[12])
        require(m[11] == ('mixed' if eligible else 'small')
            and bound == (nt[0] if eligible else 0), 'actual cache source/NT1binding mismatch')
        current['calls'] += 1; current['cpu_jobs'] += jobs; current['cpu_entries'] += rows
        current['bound_nt1'] += bound; current['cache_callbacks'] += int(eligible)
        for i, count in enumerate(nt): current['jobs_NT'][i] += count
    require(current is None and len(frames) == len(prompts), 'complete model requests')
    clocks = re.findall(r'^strata decode timing: (\d+) windows,[^\n]* CPU ([0-9.]+)\]', text, re.M)
    require(len(clocks) == len(frames), 'independent pool clocks')
    bindings = []
    for frame, (windows, cpu_ms) in zip(frames, clocks):
        require(int(windows) == len(frame['windows']), 'trace/pool window counts')
        bindings.append(dict(prompt_tokens=frame['prompt_tokens'], windows=int(windows),
            rows=sum(t for _, t in frame['windows']), cpu_jobs=frame['cpu_jobs'],
            cpu_entries=frame['cpu_entries'], pool_ms=float(cpu_ms)*int(windows)))
    work = census.extract(text, bindings, formats, membership)
    require(all(frame['jobs_NT'] == w['jobs_NT'] for frame, w in zip(frames, work)),
        'independent per-NT callback/census counts differ')
    return dict(cache_setup=setup, requests=[dict(request=w['request'],
        prompt_tokens=frame['prompt_tokens'], windows=w['windows'], rows=w['rows'],
        cpu_jobs=w['cpu_jobs'], cpu_entries=w['cpu_entries'], vram_entries=w['vram_entries'],
        jobs_NT=w['jobs_NT'], pool_ms=w['pool_ms'], logical_job_blob_bytes=w['logical_job_blob_bytes'],
        membership_sha256=w['membership_sha256'], bound_nt1_jobs=frame['bound_nt1'],
        cached_GU_rows=640*frame['bound_nt1'], eligible_cache_callbacks=frame['cache_callbacks'])
        for w, frame in zip(work, frames)],
        limits='Target-verifier only;draft/refill/prefill excluded. Bindings are not direct kernel '
            'counts. Charged diagnostics do not qualify clean model speed or physical bus saturation.')


def controls():
    # Literal complete trace, independent of model output and the parser's
    # production calculations. All 48 layers support GU18 in this fixture.
    entries, budget = 24576, 65536 << 20
    known = entries*1715200 + entries*512 + 48*512*16 + 80
    header = ('CPU_COMPACT_GU_BEGIN budget_bytes=' + str(budget) + ' layers=48\n'
        'CPU_COMPACT_GU_PREPARED phase=decode nt=1 entries=' + str(entries)
        + ' known_bytes=' + str(known) + ' budget_bytes=' + str(budget) + ' setup_s=1.000000\n'
        'CPU_COMPACT_GU_READY phase=decode nt=1 max_jobs=40 entries=' + str(entries)
        + ' known_bytes=' + str(known) + ' policy=mixed max_total_nt=40\n'
        'CPU_DECODE_SCHEDULE whole_mixed=0 down=42 jobs=24,36 nt=mixed_1_to_4 workers=5 host=1\n')
    begin = ('strata cpu work: begin request=1 phase=decode prompt_tokens=32769 '
        'layers=48 experts=512 cache=restored_decode_snapshot_immutable\n')
    bits = '0000000003ffffff' + '0'*112
    cache = ''.join(f'strata cpu work: cache request=1 layer={l} bin=0 resident_bits={bits}\n'
        for l in range(48))
    joint = ''.join(f'strata cpu schedule: layer={l} gu=18 down=42 jobs={24 if l == 0 else 0} '
        f'nt1={20 if l == 0 else 0} nt2={4 if l == 0 else 0} nt3=0 nt4=0 other=0 whole_calls=0 '
        f'cache_policy={"mixed" if l == 0 else "small"} bound_nt1={20 if l == 0 else 0}\n'
        for l in range(48))
    bin_ = ('strata cpu work: bin request=1 bin=0 gu=18 down=42 H=2560 FF=640 blob_bytes=1715200 '
        'gu_act=0 down_act=0 gu_row=980 down_row=180 act_bytes=0 h_bytes=0 calls=48 pool_ns=1000000 '
        'CPU=28 VRAM=1892 PCIe=0 peer=0 jobs_NT=20,4,0,0,0,0,0,0\n')
    end = ('strata cpu work: end request=1 phase=decode valid=1 windows=1 rows=4 callbacks=48 k=10 '
        'cpu_jobs=24 CPU=28 VRAM=1892 PCIe=0 peer=0 pcie_jobs=0 error=none\n')
    text = header + begin + cache + 'strata trace: window 32768 4\n' + joint + bin_ + end \
        + 'strata decode timing: 1 windows, host [jobs 0.00 CPU 1.00]\n'
    args = ([(18, 42, 1715200)]*48, [(1 << 26)-1]*48, [32769])
    good = observe(text, *args, budget_mib=65536)
    require(good['requests'][0]['cpu_entries'] == 28
        and good['requests'][0]['bound_nt1_jobs'] == 20
        and good['requests'][0]['eligible_cache_callbacks'] == 1
        and good['cache_setup']['known_bytes'] == known, 'literal mixed/NT1 positive control')
    nt4 = text.replace('jobs=24 nt1=20 nt2=4 nt3=0 nt4=0',
        'jobs=10 nt1=0 nt2=0 nt3=0 nt4=10').replace('bound_nt1=20', 'bound_nt1=0') \
        .replace('CPU=28 VRAM=1892', 'CPU=40 VRAM=1880') \
        .replace('jobs_NT=20,4,0,0,0,0,0,0', 'jobs_NT=0,0,0,10,0,0,0,0') \
        .replace('cpu_jobs=24', 'cpu_jobs=10')
    good4 = observe(nt4, *args, budget_mib=65536)
    require(good4['requests'][0]['cpu_entries'] == 40
        and good4['requests'][0]['bound_nt1_jobs'] == 0
        and good4['requests'][0]['eligible_cache_callbacks'] == 1, 'NT4 stays native control')
    wrong = [
        text.replace('jobs=24 nt1=20 nt2=4', 'jobs=36 nt1=24 nt2=12'),
        text.replace('whole_calls=0', 'whole_calls=1', 1),
        text.replace('strata cpu schedule: layer=0 ', 'strata cpu schedule: layer=1 ', 1),
        text.replace(' callbacks=48 ', ' callbacks=47 '),
        text.replace(' k=10 ', ' k=9 '),
        text.replace('bound_nt1=20', 'bound_nt1=19'),
        text.replace('cache_policy=mixed', 'cache_policy=small', 1),
        text.replace('entries=24576', 'entries=24575'),
        text.replace('CPU_COMPACT_GU_READY ', 'MISSING_READY ', 1),
        text + 'CPU_COMPACT_GU_OFF native_fallback=1 reason=allocation\n',
        text.replace('max_total_nt=40', 'max_total_nt=39'),
        text.replace('known_bytes='+str(known), 'known_bytes='+str(budget+1)),
        text.replace('setup_s=1.000000', 'setup_s=nan'),
        header.splitlines()[0]+'\n'+text,
        text.replace('budget_bytes='+str(budget), 'budget_bytes='+str(budget-1)),
        text.replace('nt1=20 nt2=4 nt3=0 nt4=0', 'nt1=21 nt2=2 nt3=1 nt4=0')
            .replace('bound_nt1=20', 'bound_nt1=21'),
        nt4.replace('bound_nt1=0', 'bound_nt1=1', 1)]
    for i, candidate_text in enumerate(wrong):
        try:
            observe(candidate_text, *args, budget_mib=65536)
        except RuntimeError:
            continue
        raise RuntimeError('mixed-cache negative control escaped: '+str(i))
    return dict(literal_positive=2, rejected_controls=len(wrong), actual_model_or_GPU=False)


def route(text):
    value = original_route(text)
    row = json.loads((ROOT/'record.json').read_text()); argv = row['argv']
    require('--expert-cache-per-layer' in argv and argv[argv.index('--expert-cache')+1] == '1256',
        'retain exact GPU expert placement')
    formats = census.manifest_formats(device.candidate.cpu.inputs.METADATA)
    membership = census.expected_membership(argv[argv.index('--expert-profile')+1], 1256)
    value['CPU_mixed_cache_census'] = observe(text, formats, membership, [32769, 82, 32769])
    require(all(v['bound_nt1_jobs'] > 0 and v['eligible_cache_callbacks'] > 0
        for v in value['CPU_mixed_cache_census']['requests']), 'actual NT1 use in every request')
    return value


def memory_admission():
    available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
        if s.startswith('MemAvailable:'))
    require(available >= 80 << 30, '80GiB available for model/cache admission')


device.context = device.inherited.context
base.device = device
base.ROOT, base.KEY, base.THIS = ROOT, KEY, THIS
base.native, base.route = native, route


def prepare():
    memory_admission(); checked = controls(); base.prepare()
    row = json.loads((ROOT/'record.json').read_text())
    row['environment'].update(STRATA_CPU_COMPACT_GU_POLICY='mixed', STRATA_CPU_COMPACT_GU_MIB='16384',
        STRATA_CPU_DECODE_WHOLE='0', STRATA_CPU_WORK_CENSUS='1', STRATA_CPU_SERVICE='0',
        STRATA_IQ3_PAIR='1', STRATA_DECODE_TIMING='1', STRATA_TRACE='1')
    row['budgets']['rss_cap'] = 96 << 30
    row.pop('previous_preparation', None); row.pop('previous_control_failure', None)
    row['mixed_cache_parser_controls'] = checked
    row['plan'] = ('One bounded opt-in mixed/16384MiB route diagnostic,not a speed comparison. '
        'Keep thinking32769/82/32769inputs,32outputs each,spec4,5workers+host,static1256GPUslots, '
        '8192BF16/vLLM throughput prefill,logical262144/resident20480INT8KV and qualified safety. '
        'Require complete actual cache startup and per-callback source/NT1bindings,independent '
        'shape/membership/entry/pool-clock conservation,state handoff/restoration,quiet validation, '
        'normal0/closed ownership/no fresh fault and15full assets before/after. Record cache setup '
        'and peak owned RSS. Stop first failure,no retry/pin refresh. Commit matched repeated '
        'long/agent/thinking1K+ strong/weak and prefill-preservation comparisons only after route admission.')
    row['limits'] = ('Charged32-output route/state diagnostics are not clean PP/TTFT,sustainedTG, '
        'task quality,physical262144fill/tails/RESTORE,ordinary provider teardown or adoption. '
        'Binding counts are not direct kernel counters;known cache bytes exclude allocator/RSS.')
    row['retention'] = ('Retain compact cache/setup/RSS,source/state/conservation/fault evidence '
        'for the imminent model comparison. Retire successful raw output after closure. Keep14508 '
        'validated reference and this one temporary393d88candidate;no new native generation.')
    base.context().context(row['native_observer_admission']).save(row)


if __name__ == '__main__':
    require(sys.executable == base.PYTHON and len(sys.argv) == 2
        and sys.argv[1] in (*base.ACTIONS, 'controls'), 'qualified Python;prepare|host|run|controls')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.original.guards.no_model_owners([])
        if sys.argv[1] == 'controls': print(json.dumps(controls()))
        elif sys.argv[1] == 'prepare': prepare()
        else:
            if sys.argv[1] == 'run': memory_admission()
            base.context().execute(sys.argv[1])
