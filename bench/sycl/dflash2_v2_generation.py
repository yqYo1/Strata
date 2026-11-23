"""Measure the official suite with the B570 V2 DFlash2 serving route."""

from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_v2_model as model

v2, bank, basis, require = model.v2, model.bank, model.basis, model.require
clock, suite = v2.clock, v2.suite
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_v2_generation_v1'
ROOT = bank.B / KEY.replace('_', '-')
NATIVE_GATE = '6aa1c2b52f16c194d6869df56ed31a0ad99fefbe'
WORKLOAD_ID = 'dflash2_v2_generation_v1'
ORDER = ('dflash2',) * 3
WORKLOADS = suite.evidence_reader.SUSTAINED_WORKLOADS


def baseline():
    row = v2.baseline()
    gate = bank.previous.historic(NATIVE_GATE, model.KEY)
    plan, result = gate['plan'], gate['outcome']
    require(gate['complete'] and gate['passed'] and gate['ownership_closed'] and not gate['active']
            and result['native_exit_code'] == 0 and result['fresh_faults'] == []
            and result['states'][0] == result['states'][2] and result['placement']['gpu_rows'] == 32768
            and plan['binary'] == row['binary'] and plan['native_source_commit'] == row['native_source_commit']
            and plan['build_sha256'] == row['native_build_sha256'] and plan['boot'] == row['boot']
            and plan['driver'] == row['driver'] and plan['runtime_manifest'] == row['runtime_manifest']
            and all(bank.sha(p) == h for p, h in plan['drafter_assets'].items()),
            'require the exact DFlash2 V2 short/32K gate and draft assets')
    return dict(row, dflash2_gate=NATIVE_GATE, dflash2_assets=plan['drafter_assets'],
                validated_dflash2_args=plan['command'][2:])


def configuration(row, plan):
    base = suite.configuration(row, plan)
    cfg = dict(base, drafting='dflash2', user_args=['--spec', '4', '--prompt-cache', '0', '--prompt-cache-every', '0'])
    args = model.server.engine_args(cfg, exe=cfg['exe'], announce=False)
    expected = list(row['validated_dflash2_args'])
    expected[expected.index('--prompt-cache') + 1] = '0'
    require(args == expected and model.server.engine_args(dict(cfg, args=args), exe=cfg['exe'], announce=False) == args,
            'effective DFlash2 CLI changed outside the specified prompt-cache setting')
    return dict(cfg, args=args)


def environment(row, plan):
    env = v2.launch_environment(row, dict(plan, arm='candidate'), split=False)
    env.update(STRATA_SYCL_DECODE_Q6_DPAS_HEAD='1', STRATA_PREFILL_RELEASE_MTP_EXPERTS='0')
    require(not any(k in env for k in ('STRATA_TRACE', 'STRATA_STATE_HASH', 'STRATA_SNAPSHOT_VERIFY',
                                       'STRATA_DECODE_TIMING', 'STRATA_DRAFT_TIMING', 'UR_LOG_TRACING', 'LD_PRELOAD'))
            and env.get('STRATA_SYCL_CACHE_LOAN_AUDIT', '0') == '0', 'unexpected timing observer')
    return env


def routes(text, samples, arm):
    require(arm == 'dflash2' and text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
            and not any(marker in text for marker in ('strata trace:', 'strata loan audit:', 'strata decode timing:'))
            and not suite.gen.draft_rows(text), 'unexpected native head, drafter or observer')
    v2.continuation.standard.profiles.check_profile_startup(text, 'original')
    footer = clock.footer
    headers = list(re.finditer(footer.HEADER, text, re.M))
    copies = list(re.finditer(footer.COPY, text, re.M))
    require(len(headers) == len(copies) == len(samples), 'one checked header and expert footer per request')
    checked = []
    for i, (header, copy, sample) in enumerate(zip(headers, copies, samples, strict=True)):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        fields = ('prompt_tokens', 'reused_tokens', 'read_tokens', 'generated_tokens', 'drafts_accepted', 'drafts_offered')
        require(tuple(map(int, header.groups())) == tuple(sample[k] for k in fields)
                and header.end() <= copy.start() < end and sample['reused_tokens'] == 0,
                'actual native request or expert footer differs')
        frame = text[headers[i-1].end() if i else 0:header.start()]
        taps = re.findall(r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=([01]) vram_bytes=(\d+)$', frame, re.M)
        require(len(taps) == 1 and int(taps[0][0]) == min(sample['read_tokens'] - 1, 2046)
                and taps[0][1] == '0' and int(taps[0][2]) > 0, 'actual DFlash2 prompt tail differs')
        if sample['phase'] != 'prompt/32K' and not sample['phase'].startswith('long-generation/'):
            continue
        read = sample['read_tokens']
        store = model.restore.require_store(frame, dict(direct=False), dict(read=read))
        banks = re.findall(r'^strata MoE expert bank: requested_slots=(\d+) effective_slots=(\d+) '
                           r'staged=(\d+) host_slab_bytes=\d+ device_bytes=(\d+) ', frame, re.M)
        require(banks == [('384', '384', '1', '1022361600')] and store['gpu_rows'] == read - 1
                and store['window_mode'] == 10, 'expected ordinary all-GPU 384-slot residual route')
        kv = v2.state.kv.check_clean_request(frame, 0, read - 1, 8192, reuse=True)
        counts = dict(zip(footer.COPY_KEYS, map(int, copy.groups()), strict=True))
        require(all(counts[k] == 0 for k in footer.COPY_KEYS[:6])
                and 0 < counts['cpu_unique_misses'] <= counts['cpu_entries'], 'actual decode expert route differs')
        checked.append(dict(phase=sample['phase'], read=read, store=store, kv=kv, decode_work=counts,
                            tap_rows=int(taps[0][0]), dflash2_vram_bytes=int(taps[0][2])))
    require(len(checked) == 3, 'the nominal long and both sustained native routes are required')
    return checked


def decision(processes):
    require(len(processes) == len(ORDER) and all(p['passed'] and p['ownership_closed'] and p['complete']
               and p['native_exit_code'] == 0 and p['fresh_faults'] == [] for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER),
            'three independent complete and fault-free runs are required')
    workloads = {name: dict(requests=[s for p in processes for s in p['evidence']['samples']
                                      if s.get('workload') == name]) for name in WORKLOADS}
    rates = {name: [p['evidence']['decode_rates'][name] for p in processes]
             for name in (*WORKLOADS, 'agent_followup')}
    return dict(decode_tps=rates, decode_median={k: statistics.median(v) for k, v in rates.items()},
                observed_draft_classes=suite.official.d.client.long_generation.classify_pair(workloads, len(ORDER)),
                scope='Clean ordinary DFlash2 spec4 baseline. Match actual fixed-input work and keep individual requests. '
                      'The nominal 32K prompt is supplemental if it reads fewer than 32768 tokens.')


def experiment():
    ctx = clock.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.WORKLOAD_ID, ctx.ORDER = ROOT, KEY, THIS, WORKLOAD_ID, ORDER
    ctx.baseline, ctx.configuration, ctx.environment, ctx.routes = baseline, configuration, environment, routes
    ctx.evidence_reader.decide = decision
    ctx.protocol = dict(order=list(ORDER),
        execution='Three independent processes, each with the official long prompt, six-turn agent session and the '
                  'two v5 sustained thinking workloads. Canonical head mode1, Q4_0 DFlash2 spec4 confidence0.6, '
                  'ordinary 384-slot prefill bank and copied attention. Match the V2 DFlash2 gate with prompt cache off.',
        evidence='Require owned native closure, valid client TTFT, real route/tail and draft work, >=32768 fresh '
                 'sustained inputs and >=1024 generated tokens including thinking and a visible answer. Record '
                 'first-use and later requests by phase. Preserve each process and use observed draft acceptance.',
        stop='Stop at the first source, runtime, ownership, route, eligibility or budget failure. Keep original outcomes. '
             'There is no speed-based exclusion or automatic adoption.')

    def pins(revision):
        values = v2.experiment('official').pins(revision)
        values.update(bank.stage.source.pin(bank.P, [THIS, *model.SOURCES])['source_sha256'])
        for name in (THIS, *model.SOURCES):
            require((bank.P / name).read_bytes() == suite.gen.git('show', revision + ':' + name),
                    'changed DFlash2 benchmark input: ' + name)
        return values

    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
