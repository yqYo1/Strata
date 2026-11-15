#!/usr/bin/env python3
"""Controller controls: launcher identity and matched immutable settings."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import compare_coherent_generations as comparison


class GenerationBindingTests(unittest.TestCase):
    def fixture(self):
        row=dict(configuration=dict(exe=str(comparison.P/'tools/sycl/engine_once.py'),
            args=['--spec','4','--pcie-frac','0','--adapt-swaps','0']),native_builds={})
        for arm in ('reference','candidate'):
            row['native_builds'][arm]=dict(binary=dict(path='/'+arm,sha256=comparison.SHAS[arm]),
                native_source_commit=comparison.SOURCES[arm],native_validation_probe={'pinned':True},boot='boot')
        plans=[dict(index=i,read=32769,kernel=arm,heads=1,root='/case/'+str(i))
            for i,arm in enumerate(('reference','candidate'))]
        return row,plans

    def test_native_plan_selects_generation_without_bypassing_wrapper(self):
        row,plans=self.fixture();before=copy.deepcopy(row)
        for p in plans:
            arm=p['kernel'];native=comparison.native_context(row,p)
            self.assertEqual(comparison.launch_plan(row,p)['binary']['path'],'/'+arm)
            self.assertEqual(native['native_source_commit'],comparison.SOURCES[arm])
            self.assertEqual(comparison.configuration(row,p)['exe'],row['configuration']['exe'])
        self.assertEqual(row,before)
        for bad in ('native','shape','wrapper'):
            data=copy.deepcopy(row);plan=copy.deepcopy(plans[1])
            if bad=='native':data['native_builds']['candidate']['binary']['sha256']=comparison.SHAS['reference']
            elif bad=='shape':plan['read']=65537
            else:data['configuration']['exe']='/candidate'
            with self.subTest(case=bad),self.assertRaises(RuntimeError):
                comparison.launch_plan(data,plan);comparison.configuration(data,plan)

    def test_arms_keep_identical_environment_and_reject_hidden_setting_changes(self):
        row,plans=self.fixture()
        def environment(data,plan):
            return dict(SAFETY='fixed',CACHE=plan['root']+'/cache',
                STRATA_SYCL_DECODE_KV_COALESCED='1')
        with mock.patch.object(comparison.kv.base,'environment',side_effect=environment):
            self.assertEqual(comparison.invariant(row,plans[0]),comparison.invariant(row,plans[1]))
            for p in plans:
                env=comparison.environment(row,p)
                self.assertEqual(env['STRATA_SYCL_DECODE_KV_COALESCED'],'0')
                self.assertEqual(env['STRATA_SYCL_KV_COPY_GROUPS'],'0')
            changed=copy.deepcopy(row);changed['configuration']['extra']='unexpected'
            self.assertNotEqual(comparison.invariant(row,plans[0]),comparison.invariant(changed,plans[1]))
            wrong=copy.deepcopy(plans[1]);wrong['kernel']='reference'
            with self.assertRaises(RuntimeError):comparison.invariant(row,wrong)


if __name__=='__main__':unittest.main()
