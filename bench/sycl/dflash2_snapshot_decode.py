"""Check sustained DFlash2 decode across the target-only snapshot change."""

from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_decode_confidence as confidence
import dflash2_prefill_physical_context as physical
import dflash2_snapshot_direct_prefill as prefill

leaves = confidence.leaves
bank, basis, require, clock = confidence.bank, confidence.basis, confidence.require, confidence.clock
KEY = 'dflash2_snapshot_regular_code_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
CONFIDENCE_REFERENCE = 'ce3ef1a3a341d15b1a88bc83802018c960c93be7'
ORDER = ('incumbent', 'snapshot', 'snapshot', 'incumbent', 'incumbent', 'snapshot')

PROTOCOL = dict(order=list(ORDER), workload=leaves.WORKLOAD,
    only_factor='Choose the incumbent or target-only snapshot executable. Both use the measured 0.6 '
                'confidence, spec4, fixed stock placement, prompt cache off, and the same clean runtime.',
    evidence='Use four warmups and one sustained thinking-on regular-code request per fresh process. '
             'Require the original request bodies and match generated counts, accepted/offered drafts, '
             'actual verifier expert work, copy route and KV work. Keep PP, client TTFT and TG separately.',
    decision='Report every paired TG ratio and its direction. A uniformly slower result prevents decode '
             'adoption. Mixed directions require further evidence. The fixture covers one sustained code '
             'workload. Weak acceptance, agent sessions and adoption have their own decisions.',
    stop='Stop on the first source, model, work framing, route, owner, validation, budget or fault mismatch.')


def baseline():
    row = confidence.baseline()
    recent = bank.previous.historic(CONFIDENCE_REFERENCE, confidence.KEY)
    controls = [p for p in recent['processes'] if p['arm'] == 'threshold']
    require(recent['stage'] == 'closed' and recent['complete'] and recent['passed'] and not recent['active']
            and recent['ownership_closed'] and recent['binary'] == row['binary']
            and len(controls) == 2 and all(p['passed'] and p['native_exit_code'] == 0
                                          and not p['fresh_faults'] for p in controls),
            'recover the two clean bounded confidence controls')
    context = basis.validation.committed(bank.RESULT, (prefill.KEY,))
    selected, proof, _ = physical.snapshot_reference(row)
    require(context['stage'] == 'closed' and context['complete'] and context['passed']
            and not context['active'] and context['ownership_closed']
            and context['binary'] == row['binary'] and len(context['processes']) == 6
            and context['snapshot_reference']['binary'] == selected['binaries']['strata']
            and context['snapshot_reference']['native_build_sha256'] == proof
            and all(p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in context['processes']),
            'finish the fixed long-prefill comparison on both exact builds')
    capacity = basis.validation.committed(bank.RESULT, (physical.SNAPSHOT_KEY,))
    full = capacity['outcomes'].get('full-copy', {})
    require(capacity['stage'] == 'closed' and capacity['complete'] and capacity['passed']
            and capacity['ownership_closed'] and not capacity['active']
            and capacity['native_source_commit'] == selected['native_source_commit']
            and capacity['native_build_sha256'] == proof
            and capacity['binary'] == selected['binaries']['strata_full_context_diagnostic']
            and all(capacity[k] == row[k] for k in ('boot', 'driver', 'runtime_manifest', 'native_validation_probe'))
            and full.get('passed') and full.get('native_exit_code') == 0 and full.get('fresh_faults') == []
            and full.get('batched_rows') == 262143 and full.get('generated_ids') == 0
            and full.get('final_slot_consumed'), 'finish the exact snapshot physical context fixture')
    return dict(row, snapshot_arm=dict(binary=selected['binaries']['strata'],
        native_source_commit=selected['native_source_commit'], native_build_sha256=proof),
        confidence_evidence=CONFIDENCE_REFERENCE, direct_comparison_sha256=bank.signature(context),
        physical_context_sha256=bank.signature(capacity))


def native_context(row, plan):
    arm = plan['arm']
    require(arm == ORDER[plan['index']], 'planned executable arm changed')
    return dict(row, **row['snapshot_arm']) if arm == 'snapshot' else row


def configuration(row, plan):
    return confidence.configuration(row, dict(plan, arm='threshold'))


def routes(text, samples, _arm):
    return confidence.routes(text, samples, 'threshold')


def work(process):
    # This is a matched timing workload. Content hashes remain diagnostics.
    fields = tuple(k for k in leaves.WORK if not k.endswith('_sha256') or k == 'request_sha256')
    receipts = process['routes']
    return (tuple(tuple(s[k] for k in fields) for s in process['evidence']['samples']),
            receipts['window_work'], receipts['expert_work'], receipts['kv'], receipts['attention_store'])


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(ORDER)
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in processes), 'finish all six exact owned processes')
    result = []
    for left, right in zip(processes[::2], processes[1::2], strict=True):
        control, candidate = (left, right) if left['arm'] == 'incumbent' else (right, left)
        matched = work(control) == work(candidate)
        old, new = (p['evidence']['samples'][-1] for p in (control, candidate))
        result.append(dict(matched_work=matched, tg_ratio=old['decode_ms'] / new['decode_ms'],
            prompt_time_ratio=new['prompt_ms'] / old['prompt_ms'], ttft_ratio=new['ttft_s'] / old['ttft_s'],
            generation_hash_match=new['output_sha256'] == old['output_sha256']))
    tg = {arm: [p['evidence']['decode_rates'][leaves.WORKLOAD] for p in processes if p['arm'] == arm]
          for arm in ('incumbent', 'snapshot')}
    return dict(pairs=result, TG=tg, TG_median={k: statistics.median(v) for k, v in tg.items()},
                matched_work=all(v['matched_work'] for v in result),
                consistently_faster=all(v['matched_work'] and v['tg_ratio'] > 1 for v in result),
                consistently_slower=all(v['matched_work'] and v['tg_ratio'] < 1 for v in result),
                weak_acceptance_qualified=False, agent_session_qualified=False, adopted=False)


def experiment():
    ctx = leaves.experiment()
    prior_pins, run = ctx.pins, ctx.run
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment = baseline, configuration, confidence.environment
    ctx.routes, ctx.evidence_reader.decide = routes, decision
    ctx.protocol, ctx.native_context = PROTOCOL, native_context

    def pins(revision):
        pins = prior_pins(revision)
        for name in (THIS, confidence.THIS, physical.THIS, prefill.THIS,
                     prefill.snapshot.THIS, clock.THIS, clock.suite.THIS):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed target snapshot decode source: ' + name)
            pins[name] = bank.sha(path)
        return pins

    ctx.pins = pins
    ctx.run = lambda index: run(index, native_context=native_context)
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
