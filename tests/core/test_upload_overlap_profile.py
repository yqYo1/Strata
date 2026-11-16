"""Host-only source/ownership/count/dependency controls; repository LICENSE."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import profile_prefill_upload_overlap as target


class ReachedSourcePin(Exception):pass


def events(arm):
    out=[dict(name='DenseKernel',begin=0,end=1000),dict(name='DenseKernel',begin=10000,end=20000),
         dict(name='zeCommandListAppendMemoryCopy(H2D)[1971200]',begin=1000,end=5000)]
    for i in range(512):
        t=100000+i*30000
        out += [dict(name='zeCommandListAppendMemoryCopy(H2D)[1971200]',begin=t,end=t+10000),
                dict(name='DenseKernel',begin=t+(10000 if arm=='serial' else 2000),
                     end=t+(20000 if arm=='serial' else 12000))]
    return out


class UploadOverlapProfile(unittest.TestCase):
    def test_observed_device_only_markers_are_admitted(self):
        # Minimal normalized schema from the real2163876device-only trace.
        # PTI chromelogger.h emits GPUside t/s markers;no host API endpoints.
        document={'traceEvents':[
            dict(ph='M',name='process_name',pid=42,args=dict(name='HOST<test>')),
            dict(ph='M',name='process_name',pid=7,args=dict(name='DEVICE<test> 0:5:0:0 #0')),
            dict(ph='t',pid=7,tid=7,cat='Flow_H2D_3_42',id=3,ts=10),
            dict(ph='s',pid=7,tid=7,cat='Flow_D2H_3_42',id=3,ts=10),
            dict(ph='X',pid=7,tid=7,cat='gpu_op',name='DenseKernel',ts=10,dur=2,args=dict(id='3'))]}
        self.assertEqual(len(target.trace_events(document,42,0,20000)),1)

    def test_fixed_counts_and_overlap_match_known_schedule(self):
        for arm,overlap in [('serial',0),('overlap',512*8000)]:
            with self.subTest(arm=arm):
                result=target.benchmark_attribution(events(arm),arm)
                self.assertEqual(result['timed_copy_calls'],512)
                self.assertEqual(result['timed_dense_calls'],512)
                self.assertEqual(result['copy_kernel_overlap_ns'],overlap)
                self.assertEqual(result['requested_timed_H2D_bytes'],1009254400)

    def test_missing_extra_or_wrong_work_refuses_attribution(self):
        for change in ['missing_copy','extra_kernel','wrong_bytes','bad_arm']:
            with self.subTest(change=change):
                data=events('serial');arm='serial'
                if change=='missing_copy':data.pop(3)
                elif change=='extra_kernel':data.append(dict(name='DenseKernel',begin=20000000,end=20010000))
                elif change=='wrong_bytes':data[3]['name']='zeCommandListAppendMemoryCopy(H2D)[1971201]'
                else:arm='unspecified'
                with self.assertRaises(RuntimeError):target.benchmark_attribution(data,arm)

    def test_readiness_or_one_lookahead_violation_is_rejected(self):
        for change in ['serial_read_before_ready','overwrite_before_previous_compute']:
            with self.subTest(change=change):
                data=events('serial')
                if change=='serial_read_before_ready':data[4].update(begin=100001,end=110001)
                else:data[5].update(begin=110000,end=120000)
                with self.assertRaises(RuntimeError):target.benchmark_attribution(data,'serial')

    def test_trace_requires_native_origin_clock_and_device_identity(self):
        base={'traceEvents':[
            dict(ph='M',name='process_name',pid=42,args=dict(name='HOST<test>')),
            dict(ph='M',name='process_name',pid=7,args=dict(name='DEVICE<[0] 0:5:0:0 #1>')),
            dict(ph='t',pid=7,tid=7,cat='Flow_H2D_3_42',id=3,ts=10),
            dict(ph='s',pid=7,tid=7,cat='Flow_D2H_3_42',id=3,ts=10),
            dict(ph='X',pid=7,tid=7,cat='gpu_op',name='DenseKernel',ts=10,dur=2,args=dict(id=3))]}
        self.assertEqual(len(target.trace_events(base,42,0,20000)),1)
        for change in ['foreign_source','outside_clock','wrong_device','duplicate_id','missing_marker','wrong_lane','duplicate_marker','MPI_metadata','host_endpoint_only']:
            with self.subTest(change=change):
                data=copy.deepcopy(base)
                if change=='foreign_source':data['traceEvents'][2]['cat']='Flow_H2D_3_43'
                elif change=='outside_clock':data['traceEvents'][4]['ts']=30
                elif change=='wrong_device':data['traceEvents'][1]['args']['name']='DEVICE<other>'
                elif change=='duplicate_id':data['traceEvents'].append(copy.deepcopy(data['traceEvents'][4]))
                elif change=='missing_marker':data['traceEvents'].pop(2)
                elif change=='wrong_lane':data['traceEvents'][2]['tid']=8
                elif change=='duplicate_marker':data['traceEvents'].append(copy.deepcopy(data['traceEvents'][2]))
                elif change=='MPI_metadata':data['traceEvents'][0]['args']['name']='RANK 1 HOST<test>'
                else:data['traceEvents'][2].update(ph='s',pid=42,tid=42)
                with self.assertRaises(RuntimeError):target.trace_events(data,42,0,20000)

    def test_typed_plan_refuses_changes_before_source_admission(self):
        row=dict(project=dict(count=1,revision='a'*40,source_sha256={'source.py':'b'*64}),native_proof={},tools_proof={},
            ownership_closed=True,cases=[dict(label='probe',argv=['/guarded/image'],environment=dict(EnableDirectSubmission='0'))])
        public=target.compact(row)
        def call(value,error):
            with patch.object(target.ur_validation,'committed',return_value=public), \
                 patch.object(target.component.original.coherent.build.common.source,'pin',side_effect=ReachedSourcePin) as pin:
                with self.assertRaises(error):target.check(value)
                if error is RuntimeError:pin.assert_not_called()
                else:pin.assert_called_once()
        call(copy.deepcopy(row),ReachedSourcePin)
        for change in ['argv','safety','typed_owner','proof']:
            with self.subTest(change=change):
                value=copy.deepcopy(row)
                if change=='argv':value['cases'][0]['argv']=['/other/image']
                elif change=='safety':value['cases'][0]['environment']['EnableDirectSubmission']='1'
                elif change=='typed_owner':value['ownership_closed']=1
                else:value['native_proof']={'other':True}
                call(value,RuntimeError)


if __name__=='__main__':unittest.main()
