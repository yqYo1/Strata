"""Host controls for complete paired samples and prospective performance gates."""
import copy
import unittest

from test_topk_early import EarlyCompletion
import topk_early as probe


def cohort():
    result=[]
    for process in range(3):
        samples=[]
        for scenario in range(6):
            for rep in [-1,0,1,2]:
                for arm in [0,1]:
                    # First timing pair must remain visible but outside the
                    # subsequent median;all samples still pass timing admission.
                    ratio=2. if rep==-1 else .98
                    samples.append(dict(process=str(process),scenario=str(scenario),repetition=str(rep),
                        arm=str(arm),cells=str((32768,65536,117488)[scenario//2]),pattern=str(scenario%2),
                        queries='256',capacity='131072',cap='2051',batch='1024',checkpoint='64',
                        wall_s=str(1.1*(ratio if arm else 1)),service_s=str(ratio if arm else 1),output='PASS'))
        result.append(dict(kind='compare',argv=['--compare',str(process)],process=dict(pid=100+process),
            samples=samples,timers=[dict(samples='1000',p99_ns='100',resolution_ns='1')],
            controls=[dict(repetition=str(i),wall_s='.00001') for i in [-1,0,1,2,3,4]]))
    return result


class PlanControls(unittest.TestCase):
    def test_first_pair_and_all_metrics(self):
        result=probe.decision(cohort())
        self.assertTrue(result['mechanism_screen_passed'])
        for value in result['metrics'].values():self.assertAlmostEqual(value['geometric_mean_ratio'],.98)

    def test_one_regressing_group_vetoes_gain(self):
        records=cohort()
        for sample in records[0]['samples']:
            if sample['scenario']=='0' and sample['arm']=='1' and int(sample['repetition'])>=0:
                sample.update(wall_s='1.122',service_s='1.02')
        result=probe.decision(records)
        self.assertFalse(result['mechanism_screen_passed'])
        self.assertLess(result['metrics']['wall_s']['geometric_mean_ratio'],.99)

    def test_missing_duplicate_pid_and_ineligible_samples(self):
        for mutation in range(5):
            records=copy.deepcopy(cohort())
            if mutation==0:records[0]['samples'].pop()
            if mutation==1:records[0]['samples'][-1]=copy.deepcopy(records[0]['samples'][-2])
            if mutation==2:records[1]['process']['pid']=records[0]['process']['pid']
            if mutation==3:records[0]['samples'][0]['service_s']='.001'
            if mutation==4:records[0]['samples'][0]['wall_s']='nan'
            with self.assertRaises(RuntimeError):probe.decision(records)


if __name__=='__main__':unittest.main()
