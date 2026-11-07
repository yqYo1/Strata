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
    def fixture(self):
        row=dict(blob_bytes=2<<20,observed_jobs_NT=[1,1,1,1])
        valid=dict(kind='validation',distinct_experts=66,NT_mask=30,pool_serial_bits_equal=True,mixed_NT_bits_equal=True,
            inactive_outputs_untouched=True,canaries=True)
        lines=[dict(kind='work',workers=5,host_works=True,group=6,batches=100,
            unique_stream_experts=66,blob_bytes=2<<20,jobs_NT=[150]*4,phase='decode'),
            valid,dict(kind='placement',workers=5,host_cpu=0,worker_cpus=[1,2,3,4,5],actual_affinity_verified=True)]
        for stream in (False,True):
            lines.append(dict(kind='timer',stream=stream,query_s=1e-8,fastest_pass_s=.2))
            lines.extend(dict(kind='sample',stream=stream,rep=i,passes=7,wall_s=1.4,cpu_s=7,
                gu_s=.9,quant_s=.1,down_s=.3) for i in range(5))
            lines.append(copy.deepcopy(valid))
        lines.append(dict(kind='closed',passed=True))
        return row,lines

    def test_known_histograms_and_integer_conservation(self):
        self.assertEqual(mix.expected_histogram([1,1,1,1],6),[150]*4)
        self.assertEqual(mix.expected_histogram([0,0,0,7],28),[0,0,0,2800])
        self.assertEqual(mix.expected_histogram([1,1,1,0],28),[934,933,933,0])

    def test_compact_evidence_is_complete_and_typed(self):
        controls=dict(kind='compact_controls',codec_cases=524288,integer_chunks=524288,
            encoded_values=134217728,FMA_cases=16,partial_ranges=64,GU_refusals=10,
            encode_refusals=2,finite_canaries=True)
        row=dict(experts=84,blob_bytes=2176000)
        model=dict(kind='compact_validation',format=18,nt=1,activation_cases=4,experts=84,
            native_pair_bits_equal=True,finite=True,canaries=True,
            compact_GU_bytes_per_expert=1715200,raw_plus_compact_bytes_per_expert=3891200)
        for expected,fixture in ((controls,None),(model,row)):
            for variant in ('valid','missing','extra','count','boolean','guard','closure'):
                value=[copy.deepcopy(expected),dict(kind='closed',passed=True)]
                if variant=='missing':value.pop(0)
                if variant=='extra':value[0]['unknown']=0
                if variant=='count':value[0]['codec_cases' if fixture is None else 'experts']-=1
                if variant=='boolean':value[0]['finite_canaries' if fixture is None else 'finite']=1
                if variant=='guard':value[0]['finite_canaries' if fixture is None else 'canaries']=False
                if variant=='closure':value[-1]['passed']=False
                with self.subTest(model=fixture is not None,variant=variant):
                    if variant=='valid':self.assertEqual(mix.parse_compact(value,fixture),expected)
                    else:
                        with self.assertRaises(RuntimeError):mix.parse_compact(value,fixture)

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

    def test_intermediate_worker_evidence_and_histogram_types(self):
        for workers in mix.WORKERS:
            row,lines=self.fixture();lines[0].update(workers=workers,host_works=workers==5)
            lines[2].update(workers=workers,worker_cpus=list(range(1,workers+1)))
            with self.subTest(workers=workers),patch.object(mix.base,'machine',return_value={'physical_cpus':list(range(6))}):
                self.assertTrue(mix.parse(lines,row,workers,6,'measure')['validated'])
        for hist in ([0]*4,[True,0,0,0],[-1,1,1,1],[1.0,1,1,1],[1,2,3]):
            with self.subTest(hist=hist),self.assertRaises(RuntimeError):mix.expected_histogram(hist,6)
        row,lines=self.fixture()
        for workers,group in ((True,6),(4,6),(5,7)):
            with self.subTest(workers=workers,group=group),self.assertRaises(RuntimeError):
                mix.parse(lines,row,workers,group,'measure')

    def test_complete_matrix_rejects_duplicate_missing_and_boolean_cells(self):
        order=[list(x) for x in sorted(mix.matrix())];mix.validate_order(order)
        mix.validate_order(list(reversed(order)))
        for variant in ('missing','duplicate','boolean','unsupported_worker'):
            value=copy.deepcopy(order)
            if variant=='missing':value.pop()
            if variant=='duplicate':value[-1]=value[0]
            if variant=='boolean':value[0][2]=True
            if variant=='unsupported_worker':value[0][2]=4
            with self.subTest(variant=variant),self.assertRaises(RuntimeError):mix.validate_order(value)

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

    def test_full_evidence_and_refusals(self):
        for variant in ('valid','phase','histogram','affinity','numerical','mixed_NT','lost_sample','extra',
                        'short','zero_passes','nonfinite','overcount','timer','closure'):
            row,lines=self.fixture()
            if variant=='phase':lines[0]['phase']='prefill'
            if variant=='histogram':lines[0]['jobs_NT']=[151,149,150,150]
            if variant=='affinity':lines[2]['actual_affinity_verified']=False
            if variant=='numerical':lines[1]['pool_serial_bits_equal']=False
            if variant=='mixed_NT':lines[1]['mixed_NT_bits_equal']=False
            if variant=='lost_sample':lines.pop(4)
            if variant=='extra':lines.insert(0,dict(kind='unknown'))
            if variant=='short':lines[4]['wall_s']=.9
            if variant=='zero_passes':lines[4]['passes']=0
            if variant=='nonfinite':lines[4]['cpu_s']=float('nan')
            if variant=='overcount':lines[4]['gu_s']=2
            if variant=='timer':lines[3]['query_s']=.1
            if variant=='closure':lines[-1]['passed']=False
            with self.subTest(variant=variant),patch.object(mix.base,'machine',return_value={'physical_cpus':list(range(6))}):
                if variant=='valid':self.assertEqual(len(mix.parse(lines,row,5,6,'measure')['samples']),10)
                else:
                    with self.assertRaises(RuntimeError):mix.parse(lines,row,5,6,'measure')


if __name__=='__main__':unittest.main()
