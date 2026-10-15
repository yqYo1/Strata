#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
from analyze_device_profile_v1 import parse

A = Path(__file__).resolve().parent


def main():
    path = A / 'model-device-profile-v1.json'
    d = json.loads(path.read_text())
    assert d['passed'] and d['complete'] and d['collected'] and not d['active']
    assert not d['fresh_faults'] and not d['driver_dump_exists'] and d['boot_unchanged']
    assert len(d['requests']) == 1
    req = d['requests'][0]
    assert req['reference_ids_equal'] and req['done']['generated'] == 128
    text = next(f['text'] for f in d['provider_files'] if Path(f['path']).name == 'device_timing.txt')
    result = parse(text)
    roles = {}
    for role, token in [
        ('prefill_selected_int8_attention', 'launch<1, false, false>(float const*, strata::kernels::QsaAttnPools'),
        ('prefill_GU_dequant', 'iq_dequant_gu_f16('),
        ('generic_f16_dequant_multiple_consumers', '::dequant<unsigned short>('),
        ('block_scores_shared_prefill_decode', '::qsa_block_scores('),
        ('GDN_recurrence_shared_prefill_decode', '::recurrence('),
        ('KV_ordered_resolver', 'kv_stream_resolve_with_copy_groups')]:
        rows = [k for k in result['kernels'] if token in k['kernel']]
        if role == 'KV_ordered_resolver': rows = [k for k in rows if 'lambda()#1' in k['kernel']]
        assert len(rows) == 1, (role, len(rows))
        roles[role] = rows[0]
    copies = [k for k in result['kernels'] if 'kv_stream_resolve_with_copy_groups' in k['kernel'] and 'nd_item<1>)#1' in k['kernel']]
    assert len(copies) == 1; roles['KV_copy'] = copies[0]
    windows = int(req['ledger']['coverage']['W'])
    assert windows == 55 and roles['KV_ordered_resolver']['calls'] == roles['KV_copy']['calls'] == windows * 12 == 660
    assert roles['prefill_selected_int8_attention']['calls'] == 12 * 9 == 108
    gu_cpu, down_cpu, quant_cpu = {}, {}, 0
    fs = req['format_service']
    for fmt in fs['formats']:
        counters = fs['per_pair'][str(fmt['pair'])]
        gu_cpu[fmt['gu']] = gu_cpu.get(fmt['gu'], 0) + counters['GU_cpu_ns']
        down_cpu[fmt['down']] = down_cpu.get(fmt['down'], 0) + counters['Down_cpu_ns']
        quant_cpu += counters['quant_cpu_ns']
    assert sum(gu_cpu.values()) + sum(down_cpu.values()) + quant_cpu == req['ledger']['cpu_service_sum_ns']
    result.update(passed=True, source_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), roles=roles,
        actual_input_tokens=d['actual_input_tokens'], generated=req['done']['generated'],
        same_reference_ids=True, decode_coverage=req['ledger']['coverage'],
        CPU_thread_service_ns=dict(total=req['ledger']['cpu_service_sum_ns'], GU_by_type=gu_cpu, Down_by_type=down_cpu, activation_quant=quant_cpu),
        scope='One instrumented actual model process; startup+prefill+decode aggregate. Source-specialized prompt kernel and660 decode resolver calls are identified. Shared kernels/copies are not split by phase; device sum is not busy-union or exclusive wall. Concurrent CPU service is not serial critical-path time.',
        next_priority=['Prefill selected attention44.066s, preceding QSA score-GEMM2.802s/GDN3.680s experiments in priority',
                       'Prefill native GU dequant22.413s and generic f16dequant12.073s; generic total is not all Down',
                       'Decode CPU GU recurrence/cache feasibility after positive streaming predecode repetitions',
                       'KV ordered resolver1.295s is measured but lower priority than above; copy0.318s'],
        model_performance_qualified=False, quality_qualified=False, adopted=False,
        physical_262144_qualified=False, restore_qualified=False)
    (A / 'model-device-profile-summary-v1.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, kernels=len(result['kernels']), device_service_seconds=result['total_device_service_ns']/1e9, same_reference_ids=True)))


if __name__ == '__main__': main()
