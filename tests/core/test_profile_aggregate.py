"""Collector table validation only; no GPU or native launch."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
from profile_aggregate import device_summary, fused_moe_service
from engine_once import profile_command

TABLE='''=== Device Timing Summary ===
Total Device Time for L0 backend (ns): 100
Kernel, Calls, Time (ns), Time (%), Average (ns), Min (ns), Max (ns)
ns::Kernel<float, 16>[SIMD32 {1; 1; 1} {256; 1; 1}], 2, 80, 80.0, 40, 30, 50
zeCommandListAppendMemoryCopy(M2D)[1164], 1, 20, 20.0, 20, 20, 20
=== Kernel Properties ===
'''


class AggregateTests(unittest.TestCase):
    def test_aggregate_records_graph_construction_without_a_timeline(self):
        profile=dict(mode='aggregate',launcher=dict(path='/collector'),session='private')
        native=['/native','--serve','--prefill','8192']
        self.assertEqual(profile_command(profile,native),
            ['/collector','--session','private','--device-timing','--verbose',*native])
        self.assertEqual(native,['/native','--serve','--prefill','8192'])

    def test_existing_trace_mode_keeps_its_exact_options(self):
        profile=dict(launcher=dict(path='/collector'),session='private')
        expected=['/collector','--session','private','--start-paused','--host-timing',
            '--device-timing','--chrome-kernel-logging','--chrome-call-logging',
            '--chrome-event-buffer-size','128','/native']
        self.assertEqual(profile_command(profile,['/native']),expected)
        self.assertEqual(profile_command(dict(profile,mode='trace'),['/native']),expected)

    def test_unknown_mode_fails_before_native_launch(self):
        with self.assertRaisesRegex(RuntimeError,'unsupported profile mode'):
            profile_command(dict(mode='typo'),['/native'])

    def test_template_comma_is_part_of_the_name(self):
        result=device_summary(TABLE)
        self.assertEqual(result['total_device_ns'],100)
        self.assertTrue(result['rows'][0]['name'].startswith('ns::Kernel<float, 16>'))
        self.assertEqual(result['rows'][0]['calls'],2)

    def test_rejects_lost_rows_duplicate_names_and_invalid_clocks(self):
        for text in (TABLE.replace('1, 20, 20.0, 20, 20, 20','1, 19, 20.0, 19, 19, 19'),
                     TABLE.replace('2, 80, 80.0, 40, 30, 50','2, 80, 80.0, 39, 30, 50'),
                     TABLE.replace('2, 80, 80.0, 40, 30, 50','2, 80, nan, 40, 30, 50'),
                     TABLE.replace('=== Kernel Properties ===',''),TABLE+TABLE,
                     TABLE.replace('zeCommandListAppendMemoryCopy(M2D)[1164]',
                        'ns::Kernel<float, 16>[SIMD32 {1; 1; 1} {256; 1; 1}]')):
            with self.subTest(text=text),self.assertRaises(ValueError):device_summary(text)

    def test_zero_barrier_minimum_preserves_positive_aggregate_and_sum(self):
        barrier=TABLE.replace('zeCommandListAppendMemoryCopy(M2D)[1164]',
            '"zeCommandListAppendBarrier"').replace(
            '1, 20, 20.0, 20, 20, 20','2, 20, 20.0, 10, 0, 20')
        result=device_summary(barrier)
        self.assertEqual(result['rows'][1]['min_ns'],0)
        self.assertEqual(result['rows'][1]['calls'],2)
        self.assertEqual(sum(row['total_ns'] for row in result['rows']),100)
        self.assertEqual(device_summary(barrier.replace('AppendBarrier',
            'AppendMemoryRangesBarrier'))['total_device_ns'],100)
        for bad in (barrier.replace('10, 0, 20','10, -1, 20'),
                    barrier.replace('AppendBarrier','UnrecognizedCommand'),
                    TABLE.replace('40, 30, 50','40, 0, 50'),
                    barrier.replace('2, 20, 20.0, 10, 0, 20','2, 0, 0.0, 0, 0, 0'),
                    barrier.replace('2, 20, 20.0, 10, 0, 20','2, 21, 20.0, 10, 0, 21')):
            with self.subTest(text=bad),self.assertRaises(ValueError):device_summary(bad)


class FusedFamilyTests(unittest.TestCase):
    def fixture(self):
        ns='strata::experimental::vllm_xe2_native16::'
        prefix=ns+'(anonymous namespace)::'
        hist=[[1,50],[2333,5],[2334,2],[8192,3]]
        rows=[]
        for cls,cols in [('GuSwiGLUKernel<('+ns+'Tile)1, false>',10),
                         ('DownScatterKernel<('+ns+'Tile)1, false, ('+ns+'ScatterLayout)2>',20)]:
            for group,calls in [(19,2),(64,3)]:
                rows.append(dict(name=prefix+cls+f'[SIMD16 {{1; {cols}; {group}}} {{128; 1; 1}}]',
                    calls=calls,total_ns=calls*100,average_ns=100,min_ns=90,max_ns=110))
        return dict(rows=rows,total_device_ns=1000),hist

    def test_full_fresh_shape_counts_and_projection_volume(self):
        table,hist=self.fixture();out=fused_moe_service(table,hist)
        for key,n,k in [('gu',1280,2560),('down',2560,640)]:
            r=out['families'][key]
            self.assertEqual(r['calls'],5)
            self.assertEqual(r['logical_rows'],2*2334+3*8192)
            self.assertEqual(r['projection_FLOPs'],2*(2*2334+3*8192)*n*k)
            self.assertGreater(sum(x['padded_rows'] for x in r['shapes']),r['logical_rows'])
        self.assertFalse(out['complete_model_coverage'])
        self.assertFalse(out['hardware_saturation_qualified'])

    def test_shared_dense_or_copy_rows_are_not_misclassified(self):
        table,hist=self.fixture();expected=fused_moe_service(table,hist)
        table['rows'].append(dict(name='DenseKernel<half, bf16>[SIMD16 {1; 10; 19} {128; 1; 1}]',
            calls=999,total_ns=99900,min_ns=99,max_ns=101))
        self.assertEqual(fused_moe_service(table,hist),expected)

    def test_wrong_capture_tile_mapping_namespace_or_geometry_fails(self):
        import copy
        for old,new in [('false','true'),('Tile)1','Tile)0'),('ScatterLayout)2','ScatterLayout)0'),
                        ('experimental::','other::'),('SIMD16','SIMD32'),('{128; 1; 1}','{64; 1; 1}'),
                        ('{1; 10; 19}','{1; 20; 19}')]:
            table,hist=self.fixture();table=copy.deepcopy(table)
            table['rows']=[dict(x,name=x['name'].replace(old,new)) for x in table['rows']]
            with self.subTest(old=old),self.assertRaises(ValueError):fused_moe_service(table,hist)

    def test_loss_duplication_or_count_preserving_shape_swap_fails(self):
        for variant in ('missing','duplicate','count','shape_swap'):
            table,hist=self.fixture()
            if variant=='missing':table['rows'].pop()
            if variant=='duplicate':table['rows'].append(dict(table['rows'][0]))
            if variant=='count':table['rows'][0]['calls']+=1
            if variant=='shape_swap':table['rows'][0]['calls'],table['rows'][1]['calls']=3,2
            with self.subTest(variant=variant),self.assertRaises(ValueError):fused_moe_service(table,hist)

    def test_unordered_duplicate_boolean_or_out_of_range_source_bins_fail(self):
        for hist in ([[2334,True]],[[True,2]],[[2334,0]],[[2334,1],[2334,2]],
                     [[8192,1],[2334,2]],[[8193,1]],[[2333,1]]):
            table,_=self.fixture()
            with self.subTest(hist=hist),self.assertRaises(ValueError):fused_moe_service(table,hist)


if __name__=='__main__':unittest.main()
