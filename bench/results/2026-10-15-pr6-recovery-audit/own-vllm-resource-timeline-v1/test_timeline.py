"""Offline synthetic schema/interval fixtures only; MAIN runs after review.
Repository LICENSE. No copied PTI test fixture, GPU or process access.
"""
import copy
import tempfile
import unittest
from pathlib import Path
import timeline as t
import controller

COMPUTE = 'own_kernel[SIMD16 {2; 1; 1} {128; 1; 1}]'
COPY = 'zeCommandListAppendMemoryCopy(M2D)[128]'

def line(name, a, q, s, e):
    return f'Thread 7 Device 42 : "{name}" [ns] {a} (append) {q} (submit) {s} (start) {e} (end)\n'


def fixture():
    # Compute [10,30], copy [20,40], unknown/barrier [50,60].
    raw = line(COMPUTE,1,2,10,30)+line(COPY,2,3,20,40)+line('zeCommandListAppendBarrier',3,4,50,60)+'=== API Timing Summary ===\n'
    events, _ = t.split(raw)
    rows = [dict(name=e['name'],calls=1,time_ns=e['end']-e['start'],min_ns=e['end']-e['start'],max_ns=e['end']-e['start']) for e in events]
    return raw, events, dict(device_rows=rows,totals_ns=dict(Device=50),kernel_properties=[dict(name='own_kernel')])


class TimelineTests(unittest.TestCase):
    def test_full_summary_schema_and_malformed_tail(self):
        raw=(Path(__file__).parent/'fixtures/tiny-schema.log').read_text()
        for replacement in [None, raw.replace('50.000000','nan',1), raw.rsplit('128',1)[0], raw.rsplit('"own_kernel", JIT',1)[0], raw.replace('Time (ns), Time (%)','Time (s), Time (%)',1)]:
            with tempfile.TemporaryDirectory() as directory:
                collector=Path(directory)/'collector';collector.mkdir()
                (collector/'timing.99.log').write_text(raw if replacement is None else replacement)
                if replacement is None:
                    result=controller.parse_collection(collector,99)
                    self.assertEqual(result['resource_timeline']['zero_duration_row_count'],1)
                    self.assertEqual(result['totals_ns']['Device'],40)
                else:
                    with self.assertRaises((ValueError, RuntimeError)):
                        controller.parse_collection(collector,99)

    def test_overlaps_payload_unknown_and_samples(self):
        raw, events, summary = fixture()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'timing.1.log';p.write_text(raw)
            r=t.extract(events,summary,p,Path(directory)/'samples.json')
            m=r['whole_native_command_envelope']
            self.assertEqual((m['event_service_sum_ns'],m['busy_union_ns'],m['compute_union_ns'],m['copy_union_ns']),(50,40,20,20))
            self.assertEqual((m['compute_copy_intersection_ns'],m['copy_only_ns'],m['compute_only_ns'],m['neither_compute_nor_copy_ns'],m['no_observed_command_ns']),(10,10,10,29,19))
            self.assertEqual(r['requested_copy_payload_bytes_by_allocation_type'],{'M2D':128})
            self.assertEqual(r['unknown_or_other_categories'],{'zeCommandListAppendBarrier':1})
            self.assertEqual(r['reconciliation']['tolerance_ns'],0)
            self.assertTrue(r['individual_samples']['sha256'])

    def test_reject_bad_intervals_and_truncation(self):
        for raw in [line(COMPUTE,1,2,5,4),line(COMPUTE,-1,2,5,6),line(COMPUTE,3,2,5,6),line(COMPUTE,1,2,'nan',6),line(COMPUTE,1,2,5,'inf'),line(COMPUTE,1,2,5,6).rstrip()]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                t.split(raw if not raw.endswith('\n') else raw+'=== API Timing Summary ===\n')
        with self.assertRaises(ValueError):
            t.split('=== API Timing Summary ===\n'+line(COMPUTE,1,2,5,6))

    def test_zero_duration_barrier_is_preserved(self):
        # Existing qualified v6 summary has min_ns=0 for barriers. Zero is a
        # measured timer-resolution result, not negative or missing evidence.
        events,_=t.split(line('zeCommandListAppendBarrier',1,2,5,5)+'=== API Timing Summary ===\n')
        summary=dict(device_rows=[dict(name='zeCommandListAppendBarrier',calls=1,time_ns=0,min_ns=0,max_ns=0)],totals_ns=dict(Device=0))
        self.assertEqual(t.reconcile(events,summary)['rows'][0]['calls'],1)

    def test_no_heuristic_dedup(self):
        _, events, summary = fixture()
        with self.assertRaises(ValueError):
            t.reconcile(events+[copy.deepcopy(events[0])],summary)
        summary['device_rows'][0].update(calls=2,time_ns=40)
        summary['totals_ns']['Device']=70
        self.assertEqual(t.reconcile(events+[copy.deepcopy(events[0])],summary)['rows'][0]['difference_ns'],0)

    def test_reject_missing_or_unqualified_copy_payload(self):
        for name in ['zeCommandListAppendMemoryCopy(M2D)','zeCommandListAppendMemoryCopy(M2D)[0]','zeCommandListAppendMemoryCopyRegion(M2D)','zeCommandListAppendImageCopy(M2D)[128]']:
            raw, events, summary=fixture()
            events[1]['name']=name;summary['device_rows'][1]['name']=name
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'raw';p.write_text(raw)
                with self.subTest(name=name),self.assertRaises(ValueError):
                    t.extract(events,summary,p,Path(directory)/'samples.json')

    def test_exact_summary_reconciliation(self):
        _,events,summary=fixture()
        for field,value in [('calls',2),('time_ns',21),('min_ns',19),('max_ns',21),('name','missing')]:
            bad=copy.deepcopy(summary);bad['device_rows'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                t.reconcile(events,bad)

    def test_shape_sequence_and_no_full_prefill_claim(self):
        raw,events,summary=fixture()
        for e in events:e['kind']='other'
        qs=[]
        for i,n in enumerate([73]*12+([8192]*4+[6493])*12):
            qs.append(dict(line=i+10,thread=7,device=42,name=t.QSA_PREFIX+' rest[SIMD16 {'+str(n*2)+'; 1; 1} {128; 1; 1}]',append=100+i*10,submit=101+i*10,start=102+i*10,end=105+i*10,kind='compute'))
        suite=dict(results=dict(request_samples=[dict(phase='warm-up/chat',sample=dict(ok=True,stream_complete=True,prompt_tokens=74,read_tokens=74,reused_tokens=0)),dict(phase='prefill-only')]))
        r=t.qsa_windows(events+qs,suite)
        self.assertEqual((r['warm_dispatch_count'],r['long_dispatch_count']),(12,60))
        self.assertFalse(r['full_prefill_boundaries_proved'])
        qs[-1]['name']=qs[-1]['name'].replace('12986','12988')
        with self.assertRaises(ValueError):t.qsa_windows(events+qs,suite)

    def test_multidevice_rejected_and_window_clipping(self):
        raw,events,_=fixture()
        with self.assertRaises(ValueError):t.split(raw.replace('Device 42 : "'+COPY,'Device 43 : "'+COPY))
        for e in events:e['kind']='copy' if e['name']==COPY else 'compute'
        r=t.measure(events,[25,35],'clipped')
        self.assertEqual(r['compute_copy_intersection_ns'],5)
        self.assertEqual(r['event_service_sum_ns'],15)


if __name__ == '__main__':
    unittest.main()
