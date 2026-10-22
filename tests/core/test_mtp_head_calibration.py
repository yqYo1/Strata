#!/usr/bin/env python3
"""Refuse incomplete head clock/numerical/timing evidence without a GPU."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import mtp_head_calibration as h


class EvidenceTests(unittest.TestCase):
    def test_each_child_has_real_private_temp_and_cache_before_exec(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(h,'ROOT',Path(tmp)/'head'):
            old=h.ROOT/'jobs'/'build'
            env=dict(STRATA_MEASUREMENT_ROOT=str(old),TMPDIR=str(old/'tmp'),XDG_CACHE_HOME=str(old/'cache'))
            expected=dict(env)
            plan=dict(environment={'build':env})
            actual=h.child_environment(plan,'build','configure')
            self.assertEqual(plan['environment']['build'],expected)
            for key in env:self.assertTrue(Path(actual[key]).is_dir())
            self.assertEqual(Path(actual['TMPDIR']),h.ROOT/'jobs'/'build-configure'/'tmp')
            self.assertNotEqual(h.child_environment(plan,'build','compile')['TMPDIR'],actual['TMPDIR'])
            plan['environment']['build']['TMPDIR']=str(Path(tmp)/'escape')
            with self.assertRaises(RuntimeError):h.child_environment(plan,'build','escape')

    def fixture(self,profile=1,mode='measure'):
        rows=106299 if mode=='measure' else 17
        valid=dict(kind='validation',graph_direct_bits_equal=True,all_guards=True,inputs_unchanged=True)
        lines=[dict(kind='reference',rows=rows,type=14,N=2560,NT=1,weight_bytes=rows*2100,
            map_unique=True,Q8_1_bytes_equal=True,max_scaled_error=1e-7),valid]
        if profile:
            lines += [dict(kind='envelope',rep=i,submit=100+i*100,start=90+i*100,end=150+i*100,
                resolution=2,wall_s=.001,query_s=1e-6) for i in range(64 if mode=='measure' else 2)]
        if mode=='measure':
            lines += [dict(kind='timer',query_s=1e-8,fastest_replay_s=.001)]
            lines += [dict(kind='sample',rep=i,replays=1000,wall_s=1.3,cpu_s=1.2) for i in range(5)]
        lines += [dict(kind='device',name='Intel Arc B570 Graphics',global_bytes=10<<30),copy.deepcopy(valid),dict(kind='closed',profile=profile,passed=True,checked_retirement=True)]
        return rows,lines

    def test_all_modes_and_clock_domain_inversion(self):
        # submit/start inversion is retained; ordered positive envelopes are valid.
        for profile in (0,1):
            for mode in ('check','measure'):
                rows,lines=self.fixture(profile,mode)
                self.assertTrue(h.parse(lines,rows,profile,mode)['validated'])

    def test_incomplete_or_invalid_evidence_rejected(self):
        for variant in ('Q8','type','rows','reference','nonfinite_reference','guard','bits','owner',
                        'extra','missing_clock','clock_order','resolution','slow_query','bad_query',
                        'sample_count','short','nonfinite','zero_replays','timer','device'):
            rows,lines=self.fixture()
            if variant=='Q8':lines[0]['Q8_1_bytes_equal']=False
            if variant=='type':lines[0]['type']=13
            if variant=='rows':lines[0]['rows']-=1
            if variant=='reference':lines[0]['max_scaled_error']=4e-6
            if variant=='nonfinite_reference':lines[0]['max_scaled_error']=float('nan')
            if variant=='guard':lines[1]['all_guards']=False
            if variant=='bits':lines[1]['graph_direct_bits_equal']=False
            if variant=='owner':lines[-1]['checked_retirement']=False
            if variant=='extra':lines.insert(0,dict(kind='unknown'))
            if variant=='missing_clock':lines.pop(2)
            if variant=='clock_order':lines[3]['start']=91
            if variant=='resolution':lines[2]['resolution']=61
            if variant=='slow_query':lines[2]['query_s']=1
            if variant=='bad_query':lines[2]['query_s']=float('nan')
            if variant=='sample_count':lines.pop(-4)
            if variant=='short':lines[-4]['wall_s']=.99
            if variant=='nonfinite':lines[-4]['cpu_s']=float('nan')
            if variant=='zero_replays':lines[-4]['replays']=0
            if variant=='timer':lines[66]['query_s']=.1
            if variant=='device':lines[-3]['name']='another GPU'
            with self.subTest(variant=variant),self.assertRaises(RuntimeError):h.parse(lines,rows,1,'measure')


if __name__=='__main__':unittest.main()
