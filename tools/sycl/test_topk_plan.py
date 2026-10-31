"""Host admission controls for the real prefill selector entry,not speed claims."""
import copy
import unittest

from test_topk_early import EarlyCompletion
import topk_early as probe


def parsed():
    return {name:[] for name in ('PROPERTY','CASE','ADMISSION','ROUTE','POLICY','SAMPLE')}


def numeric_cases():
    geometries=[(n,n,min(3,n),2051,p,0) for n in probe.ENDPOINTS for p in range(8)]
    geometries.append((65,128,14,13,6,1))
    return [dict(zip(('cells','capacity','queries','cap','pattern','invalid'),map(str,g)),
        oracle='PASS',inputs='PASS',guards='PASS',repeats='2',apis='3') for g in geometries]


class PlanControls(unittest.TestCase):
    def test_missing_duplicate_wrong_extent_and_missing_api_checks_refused(self):
        probe.case_checks(numeric_cases())
        for change in range(7):
            cases=copy.deepcopy(numeric_cases())
            if change==0:cases.pop()
            if change==1:cases[-1]=copy.deepcopy(cases[-2])
            if change==2:cases[-2]['cells']='262143'
            if change==3:cases[0]['guards']='FAIL'
            if change==4:cases[0].pop('inputs')
            if change==5:cases[0]['apis']='1'
            if change==6:cases[0]['repeats']='1'
            with self.assertRaises(RuntimeError):probe.case_checks(cases)

    def test_invalid_policy_cannot_be_reported_as_disabled(self):
        case=dict(kind='policy',argv=['--policy','invalid'])
        result=parsed();result['POLICY']=[dict(expected='invalid',invalid='1',enabled='0',PASS='1')]
        probe.check_result(case,result)
        result['POLICY'][0]['invalid']='0'
        with self.assertRaises(RuntimeError):probe.check_result(case,result)

    def test_wrong_dispatch_and_missing_actual_kernel_refused(self):
        case=dict(kind='smoke',early=True);result=parsed()
        result['ROUTE']=[dict(early='1',entry='qsa_block_topk_prefill_early')]
        result['CASE']=[dict(cells='17',capacity='32',queries='3',cap='5',pattern='0',invalid='0',
            oracle='PASS',inputs='PASS',guards='PASS',repeats='2',apis='3')]
        trace='urEnqueueKernelLaunch\nurKernelCreate kernel=submit_topk_prefill_early'
        probe.check_result(case,result,trace)
        for change in range(3):
            bad=copy.deepcopy(result);badtrace=trace
            if change==0:bad['ROUTE'][0].update(early='0',entry='qsa_block_topk_prefill')
            if change==1:badtrace='urEnqueueKernelLaunch\nurKernelCreate kernel=submit_topk'
            if change==2:badtrace='urKernelCreate kernel=submit_topk_prefill_early'
            with self.assertRaises(RuntimeError):probe.check_result(case,bad,badtrace)
        case['early']=False;result['ROUTE']=[dict(early='0',entry='qsa_block_topk_prefill')]
        with self.assertRaises(RuntimeError):probe.check_result(case,result,trace)

    def test_preexec_cases_do_not_mutate_the_base_environment(self):
        base={probe.ENV_FLAG:'1','STRATA_SYCL_SELECT_RADIX4':'1','STRATA_SYCL_SELECT_CYCLIC':'0'}
        before=copy.deepcopy(base);cases=probe.planned_cases('/pinned/libze_loader.so.1')
        self.assertEqual(len(cases),16)
        missing=next(c for c in cases if c.get('label')=='missing-radix')
        child=probe.case_environment(base,missing)
        self.assertNotIn('STRATA_SYCL_SELECT_RADIX4',child)
        self.assertEqual(base,before)
        unset=next(c for c in cases if c.get('label')=='unset')
        self.assertNotIn(probe.ENV_FLAG,probe.case_environment(base,unset))

    def test_timing_data_cannot_qualify_as_correctness(self):
        result=parsed();result['SAMPLE']=[dict(wall_s='0.01')]
        with self.assertRaises(RuntimeError):probe.check_result(dict(kind='properties'),result)


if __name__=='__main__':unittest.main()
