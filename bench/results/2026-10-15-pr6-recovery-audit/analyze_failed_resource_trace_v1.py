#!/usr/bin/env python3
"""Main-only offline analysis of one FAILED, closed profiler capture.
Repository LICENSE. No native/GPU execution, failure promotion or cleanup.
"""
from collections import Counter, defaultdict
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'main-vllm-resource-timeline-model-v3'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def artifact(p):
    p = Path(p)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))

def main():
    target = A / 'own-resource-failed-trace-analysis-v3.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not target.exists()
        receipt_path = A/'own-vllm-resource-timeline-model-v3.json'
        assert sha(receipt_path) == 'df26c8605a701fede97c1e50510b9c5ce3668d76b3f7245f695b3344ecaa208e'
        receipt = json.loads(receipt_path.read_text())
        assert receipt['status'] == 'FAIL' and not receipt['active'] and receipt['fresh_faults'] == [] and receipt['driver_dump_exists'] is False
        for key in ('final_PID_start_absence','wrapper_native_PID_start_absence'):
            assert receipt[key] and all(x['original_absent'] for x in receipt[key])
        sys.path.insert(0, str(A/'own-vllm-resource-timeline-v2'))
        import controller as c
        import timeline as t
        absence = c.absence(receipt['commands'])
        manifest_path = A/'own-vllm-resource-timeline-v2/manifest.json'
        assert sha(manifest_path) == receipt['manifest_sha256']
        manifest = json.loads(manifest_path.read_text())
        for name,pin in manifest['package_files'].items():
            c.check_pin(manifest_path.parent/name,pin)
        assert len(receipt['collector_sides']) == 1
        raw_pin = receipt['collector_sides'][0]
        raw_path = Path(raw_pin['path']);c.check_pin(raw_path,raw_pin)
        assert raw_path.parent == ROOT/'collector' and raw_pin['bytes'] <= 256 << 20
        events, summary = t.split(raw_path.read_text())
        printed = re.findall(r'Total Device Time for L0 backend \(ns\):\s*(\d+)', summary)
        assert len(printed) == 1
        scalar = int(printed[0]);observed = sum(e['end']-e['start'] for e in events)
        assert observed == scalar
        counts, payload, service, unknown = Counter(), Counter(), Counter(), Counter()
        full_names = Counter(e['name'] for e in events)
        for e in events:
            m = t.COPY.fullmatch(e['name'])
            if m:
                e['kind'] = 'copy';direction=m[1]+'2'+m[2]
                payload[direction] += int(m[3]);service[direction] += e['end']-e['start']
            elif t.SHAPE.search(e['name']):
                e['kind'] = 'compute'
            else:
                assert not ('MemoryCopy' in e['name'] or 'ImageCopy' in e['name'])
                e['kind'] = 'other';unknown[e['name']]+=1
            counts[e['kind']] += 1
        suite_path, = (ROOT/'stock-results').glob('*/results.json')
        suite=json.loads(suite_path.read_text());requests=suite['results']['prefill_only']
        assert suite['complete'] and not suite['results']['failures'] and requests['complete'] and requests['eligible_runs']==len(requests['requests'])==1
        req=requests['requests'][0]
        assert req['ok'] and req['eligible'] and req['stream_complete'] and req['actual_prompt_tokens']==req['actual_read_tokens']==39262 and req['actual_reused_tokens']==0 and req['generated_tokens']==24 and req['thinking_requested']
        stderr=(ROOT/'owned/engine.stderr').read_text()
        assert re.findall(r'^strata trace: layer-major layer (\d+) of (\d+): completed (\d+) chunks$',stderr,re.M)==[(str(i),'48','5') for i in range(48)]
        assert not c.ERROR.search(stderr)
        qsa=t.qsa_windows(events,suite)
        names=sorted(full_names);ids={n:i for i,n in enumerate(names)}
        samples_path=ROOT/'failed-timeline-samples-v3.json'
        with samples_path.open('x') as f:
            json.dump(dict(schema='strata-unitrace-text-individual-samples-v1',capture_status='FAIL',names=names,
                columns=['raw_line','name_index','thread','device','append_ns','submit_ns','start_ns','end_ns'],
                rows=[[e['line'],ids[e['name']],e['thread'],e['device'],e['append'],e['submit'],e['start'],e['end']] for e in events]),f,separators=(',',':'));f.write('\n')
        result=dict(schema='strata-failed-resource-trace-analysis-v1',status='FAILED_CAPTURE_ANALYZED_NOT_QUALIFIED',
            original_failure_status_preserved=True,collector_compatible=False,performance_adopted=False,GPU_executed=False,
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),analysis_source=artifact(__file__),invocation=sys.orig_argv,
            failed_receipt=artifact(receipt_path),raw=raw_pin,parser_source=artifact(Path(t.__file__)),individual_samples=artifact(samples_path),
            all_original_PID_start_absent=absence,rows=len(events),counts=dict(counts),
            unique_full_names=len(names),unique_compute_names=sum(bool(t.SHAPE.search(n)) for n in names),
            unique_compute_base_names=len({t.base_name(n) for n in names if t.SHAPE.search(n)}),
            total_reconciliation=dict(printed_device_total_ns=scalar,timeline_service_sum_ns=observed,difference_ns=observed-scalar,
                limitation='Scalar total only. Device rows/properties were not emitted before supervised shutdown timeout; no per-name/count/min/max reconciliation and no normal-native exit proof.'),
            requested_copy_payload_bytes_by_allocation_type=dict(payload),copy_service_ns_by_allocation_type=dict(service),
            requested_payload_divided_by_command_service_GBps={k:payload[k]/service[k] for k in payload},
            payload_limit='Requested bytes/service duration, decimal GB/s; M=UNKNOWN allocation. This is not physical PCIe/RAM/VRAM traffic, peak saturation or complete prefill attribution.',
            whole_native_command_envelope=t.measure(events,[min(e['append'] for e in events),max(e['end'] for e in events)],'All observed commands of FAILED native capture; startup/warmup/output/teardown included'),
            source_identified_QSA=qsa,unknown_or_other_categories=dict(unknown),
            benchmark_suite=artifact(suite_path),request=req,
            cpu_and_coarse_profile_lines=[s for s in stderr.splitlines() if s.startswith(('strata prefill cpu wait:','strata prefill cpu host:','strata lm profile: layer=-1'))],
            retention=dict(owner='Main',consumer='Main hardware-resource interpretation and future complete-profile comparison; distinct profiler-summary timeout mechanism',
                samples_byte_budget=96<<20,raw_byte_budget=256<<20,review_deadline_utc='2026-10-17T03:00:00Z',
                next_review='After independent extraction/hash checks, raw timeline can be replaced by this interned complete emitted-row table; retain relevant failure tail and original FAIL receipt. No cleanup here.'),
            limitations=['Full prefill boundaries unavailable','Full collector table/native normal exit unproved','No PMU/cache/SLM/XMX/physical bandwidth counters','Instrumented failed run is not a clean speed sample','No-observed-command gaps are not proof of hardware idle'])
        assert result['individual_samples']['bytes'] <= result['retention']['samples_byte_budget']
        with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(dict(status=result['status'],receipt=str(target),sha256=sha(target),rows=len(events),sample_bytes=result['individual_samples']['bytes'],original_status='FAIL')))

if __name__ == '__main__':
    main()
