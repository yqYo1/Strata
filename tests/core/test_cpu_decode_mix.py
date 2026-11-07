#!/usr/bin/env python3
"""Refuse incomplete or mislabeled current-work calibration evidence; no GPU."""
import copy
import contextlib
import json
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import cpu_decode_mix as mix



class EvidenceTests(unittest.TestCase):
    def fixture(self,arm='lut',group=10):
        row=dict(blob_bytes=2176000);count=mix.stream_count(row)
        lines=[dict(kind='lut_validation',experts=count,activation_cases=4,comparisons=4*count,native_bits_equal=True,finite_guards=True),
            dict(kind='lut_work',arm=arm,group=group,NT=1,experts=count,host_cpu=0,affinity_verified=True,
                slab_bytes=524288,accumulator_bytes=group*640*64,raw_blob_bytes=count*row['blob_bytes'])]
        if arm!='check':
            for stream in (False,True):
                lines.append(dict(kind='lut_timer',stream=stream,query_s=1e-8,fastest_call_s=.001))
                for rep in range(5):
                    rows=10*group*640
                    lines.append(dict(kind='lut_sample',stream=stream,rep=rep,calls=10,wall_s=1.8 if arm=='raw' else 1.2,
                        cpu_s=1.0,output_rows=rows,slabs=100 if arm=='lut' else 0,
                        block_rows=10*rows if arm=='lut' else 0,native_pair_rows=rows if arm=='raw' else 0))
                lines.append(dict(kind='lut_result',stream=stream,native_bits_equal=True,finite_guards=True))
        lines.append(dict(kind='closed',passed=True));return row,lines

    def test_lut_and_native_cases(self):
        for group in (1,10):
            for arm in ('raw','lut','check'):
                row,lines=self.fixture(arm,group)
                with patch.object(mix.base,'machine',return_value={'physical_cpus':[0]}):
                    v=mix.parse_lut(lines,row,group,arm);self.assertTrue(v['numerical_qualified'])
                    self.assertEqual(len(v.get('samples',[])),0 if arm=='check' else 10)

    def test_incomplete_or_false_evidence_is_refused(self):
        variants=('missing','extra','short','nan','badcpu','zero_calls','boolean_count','wrong_slabs','wrong_rows',
                  'fallback','wrong_NT','wrong_affinity','last_bits','timer','closure','boolean_closure')
        for variant in variants:
            row,lines=self.fixture()
            if variant=='missing':lines.pop(4)
            if variant=='extra':lines.insert(4,dict(kind='unknown'))
            if variant=='short':lines[3]['wall_s']=.99
            if variant=='nan':lines[3]['wall_s']=float('nan')
            if variant=='badcpu':lines[3]['cpu_s']=3
            if variant=='zero_calls':lines[3]['calls']=0
            if variant=='boolean_count':lines[3]['calls']=True
            if variant=='wrong_slabs':lines[3]['slabs']-=1
            if variant=='wrong_rows':lines[3]['block_rows']+=1
            if variant=='fallback':lines[3]['native_pair_rows']=640
            if variant=='wrong_NT':lines[1]['NT']=2
            if variant=='wrong_affinity':lines[1]['affinity_verified']=False
            if variant=='last_bits':lines[8]['native_bits_equal']=False
            if variant=='timer':lines[2]['query_s']=.1
            if variant=='closure':lines[-1]['passed']=False
            if variant=='boolean_closure':lines[-1]['passed']=1
            with self.subTest(variant=variant),patch.object(mix.base,'machine',return_value={'physical_cpus':[0]}):
                with self.assertRaises(RuntimeError):mix.parse_lut(lines,row,10,'lut')

    def test_control_mapping_and_typed_counts(self):
        good=dict(kind='lut_controls',table_entries=2097152,scaled_chunks=4194304,activation_edges=8,refusals=14,guards=True)
        self.assertEqual(mix.parse_controls([good,dict(kind='closed',passed=True)]),good)
        for key in good:
            bad=copy.deepcopy(good);bad[key]=False if type(good[key]) is bool else -1
            with self.subTest(key=key),self.assertRaises(RuntimeError):mix.parse_controls([bad,dict(kind='closed',passed=True)])
        for key in ('activation_edges','refusals'):
            bad=copy.deepcopy(good);bad[key]=float(bad[key])
            with self.assertRaises(RuntimeError):mix.parse_controls([bad,dict(kind='closed',passed=True)])

    def children(self):
        values=[]
        for case in mix.screen_order():
            row,lines=self.fixture(case[4],case[2])
            with patch.object(mix.base,'machine',return_value={'physical_cpus':[0]}):validation=mix.parse_lut(lines,row,case[2],case[4])
            values.append(dict(case=case,validation=validation,exit_code=0))
        return values

    def test_negative_control_and_upper_reuse_have_distinct_roles(self):
        cells=mix.screen_decision(self.children())['cells']
        self.assertTrue(all(c['repeated_gain_qualified'] for c in cells))
        self.assertTrue(all(c['followup_eligible']==(c['group']==10) for c in cells))

    def test_favorable_mean_does_not_erase_unstable_pairs(self):
        children=self.children()
        for c in children:
            if c['case']==[18,20,10,1,'lut']:
                for s in c['validation']['samples']:s['wall_s']=3.6
        d=mix.screen_decision(children);self.assertFalse(d['performance_qualified'])
        upper=[c for c in d['cells'] if c['group']==10]
        self.assertTrue(all(c['geomean']>1 and not c['repeated_gain_qualified'] for c in upper))

    def test_missing_duplicate_and_reordered_pairs_are_refused(self):
        for mode in ('missing','duplicate','reordered','boolean','failed'):
            c=self.children()
            if mode=='missing':c.pop()
            if mode=='duplicate':c[-1]=c[0]
            if mode=='reordered':c.reverse()
            if mode=='boolean':c[0]['case'][2]=True
            if mode=='failed':c[0]['exit_code']=1
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):mix.screen_decision(c)

    def resource(self):
        cells=[dict(gu=gu,down=dn,nt=nt,jobs=nt) for gu,dn in sorted(mix.base.PAIRS)
               for nt in range(1,5)]
        processes=[]
        for i,enabled in enumerate(mix.LEAF_ORDER):
            leaf=dict(format_nt_jobs=copy.deepcopy(cells),cpu_jobs=sum(c['jobs'] for c in cells))
            processes.append(dict(index=i,passed=True,complete=True,ownership_closed=True,
                native_exit_code=0,fresh_faults=[],resources=dict(passed=True,leaf_enabled=enabled,
                    completed_requests=5,warmup_prefix=4,work_signature_sha256='same',
                    long_request=leaf if enabled else None)))
        return dict(stage='closed',passed=True,complete=True,ownership_closed=True,active=False,
                    processes=processes,decision=dict(leaf_observer_cost_qualified=False))

    def test_current_work_binding_and_refusals(self):
        # A charged observer's cost verdict is independent of exact job counts.
        for variant in ('valid','incomplete','native_exit','fresh_fault','prefix','role',
                        'missing_bin','duplicate_bin','boolean_jobs','changed_jobs','work'):
            value=self.resource();first=value['processes'][1]['resources']['long_request']
            if variant=='incomplete':value['complete']=False
            if variant=='native_exit':value['processes'][0]['native_exit_code']=1
            if variant=='fresh_fault':value['processes'][0]['fresh_faults']=['fault']
            if variant=='prefix':value['processes'][0]['resources']['warmup_prefix']=3
            if variant=='role':value['processes'][0]['resources']['leaf_enabled']=True
            if variant=='missing_bin':first['format_nt_jobs'].pop();first['cpu_jobs']-=4
            if variant=='duplicate_bin':first['format_nt_jobs'].append(first['format_nt_jobs'][0])
            if variant=='boolean_jobs':first['format_nt_jobs'][0]['jobs']=True
            if variant=='changed_jobs':first['format_nt_jobs'][0]['jobs']+=1;first['cpu_jobs']+=1
            if variant=='work':value['processes'][0]['resources']['work_signature_sha256']='different'
            with self.subTest(variant=variant):
                if variant=='valid':
                    self.assertEqual(set(mix.observed_histograms(value)),set(mix.base.PAIRS))
                else:
                    with self.assertRaises(RuntimeError):mix.observed_histograms(value)

    def test_owner_setup_failure_is_terminal_before_any_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);planpath=root/'plan.json';planpath.write_text('{}')
            plan=dict(source=dict(revision='source'),calibration=dict(commit='calibration'),budgets=dict(stage_seconds=10))
            with patch.object(mix,'ROOT',root),patch.object(mix,'PLAN',planpath),\
                 patch.object(mix,'RESULT',root/'result.json'),patch.object(mix,'load',return_value=plan),\
                 patch.object(mix.gen,'committed',return_value={}),\
                 patch.object(mix.owner,'Owner',side_effect=RuntimeError('setup failed')) as constructor:
                self.assertEqual(mix.run('build'),1)
                failed=json.loads((root/'build.json').read_text())
                self.assertFalse(failed['passed']);self.assertFalse(failed['active'])
                self.assertTrue(failed['ownership_closed']);self.assertEqual(failed['commands'],[])
                self.assertIn('setup failed',failed['error'])
                with self.assertRaises(RuntimeError):mix.run('build')
                self.assertEqual(constructor.call_count,1)

    def test_actual_compile_and_link_input_refusals(self):
        for variant in ('valid','unknown_header','changed_header','missing_producer','wrong_C_compiler',
                        'fast_math','wrong_layout','associative_math','unknown_link'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'run';project=Path(tmp)/'project';source=Path(tmp)/'source'
                (root/'build').mkdir(parents=True);(root/'owned').mkdir()
                required=[source/'src/kernels/cpu'/n for n in mix.CPU_UNITS]
                required += [project/'bench/sycl/cpu_decode_mix/main.cpp']
                cfile=Path(tmp)/'ggml-cpu/arch/x86/quants.c'
                paths=required+[cfile]+[Path(tmp)/('header'+str(i)+'.h') for i in range(100)]
                allowed={str(p.resolve()):'sha' for p in paths}
                units=[dict(file=str(p),command=' '.join([str(mix.COMPILER/'bin/icpx'),'-O3','-DNDEBUG',
                    '-std=c++20','-march=znver3','-mtune=znver3','-DSTRATA_CPU_SERVICE_PROFILE=1'])) for p in required]
                units.append(dict(file=str(cfile),command=' '.join([str(mix.COMPILER/'bin/icx'),'-O3','-DNDEBUG',
                    '-march=native','-fno-associative-math'])))
                library=Path(tmp)/'frozen.a';library.write_bytes(b'fixture');allowed[str(library)]='sha'
                if variant=='unknown_header':paths.append(Path(tmp)/'unplanned.h')
                if variant=='changed_header':allowed[str(paths[-1])]='different'
                if variant=='missing_producer':units.pop(0)
                if variant=='wrong_C_compiler':units[-1]['command']=units[-1]['command'].replace(str(mix.COMPILER/'bin/icx'),'/usr/bin/cc')
                if variant=='fast_math':units[0]['command']+=' -ffast-math'
                if variant=='wrong_layout':units[0]['command']=units[0]['command'].replace(' -DSTRATA_CPU_SERVICE_PROFILE=1','')
                if variant=='associative_math':units[-1]['command']=units[-1]['command'].replace(' -fno-associative-math','')
                if variant=='unknown_link':allowed.pop(str(library))
                (root/'build/compile_commands.json').write_text(json.dumps(units))
                (root/'owned/build-compile.stdout').write_text(str(library)+'\n')
                output='\n'.join('    '+str(p) for p in paths)
                with contextlib.ExitStack() as stack:
                    for name,value in (('ROOT',root),('P',project),('CPU_SOURCE',source)):
                        stack.enter_context(patch.object(mix,name,value))
                    stack.enter_context(patch.object(mix,'allowed_inputs',return_value=allowed))
                    stack.enter_context(patch.object(mix.base,'digest',return_value='sha'))
                    if variant=='valid':self.assertEqual(mix.attest_build({},output)['compiled_inputs']['uncovered'],0)
                    else:
                        with self.assertRaises(RuntimeError):mix.attest_build({},output)


if __name__=='__main__':unittest.main()
