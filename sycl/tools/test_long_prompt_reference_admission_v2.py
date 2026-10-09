"""Independent pure admission/fake-resource fixtures; no project/GPU imports."""
import unittest
import long_prompt_reference_admission_v2 as admit


class AdmissionFixtures(unittest.TestCase):
    def test_reviewed_head_exception_literal(self):
        expected={'src/kernels/cpu/native_expert.cpp':'a'*64,'sycl/src/program/generate.cpp':'b'*64}
        got=admit.validate_closure('1eb89482a4afd20277ae0405780ed4f8eb98eb20','23268953314d12588fd3f496414a46bd426a7306','',expected,dict(expected),['serve/server.py','docs/DETAILS.md'])
        self.assertEqual(got['compiled_input_count'],2)
        self.assertEqual(got['changed_noncompiled_files'],['docs/DETAILS.md','serve/server.py'])
        self.assertEqual(got['build_commit'],'1eb89482a4afd20277ae0405780ed4f8eb98eb20')
        self.assertEqual(got['current_clean_HEAD'],'23268953314d12588fd3f496414a46bd426a7306')

    def test_changed_compiled_input_reject(self):
        expected={'src/kernels/cpu/native_expert.cpp':'a'*64}
        with self.assertRaisesRegex(ValueError,'changed compiled-input'):
            admit.validate_closure(admit.BUILD_COMMIT,admit.REVIEWED_HEAD,'',expected,{'src/kernels/cpu/native_expert.cpp':'b'*64},[])
        with self.assertRaisesRegex(ValueError,'changed closure path'):
            admit.validate_closure(admit.BUILD_COMMIT,admit.REVIEWED_HEAD,'',expected,expected,['src/kernels/cpu/native_expert.cpp'])
        with self.assertRaisesRegex(ValueError,'closure keys'):
            admit.validate_closure(admit.BUILD_COMMIT,admit.REVIEWED_HEAD,'',expected,{},[])

    def test_head_dirty_path_pin_negatives(self):
        expected={'src/kernels/cpu/native_expert.cpp':'a'*64}
        for head,status,branch in (('0'*40,'','reviewed'),(admit.REVIEWED_HEAD,' M serve/server.py','dirty')):
            with self.assertRaisesRegex(ValueError,branch): admit.validate_closure(admit.BUILD_COMMIT,head,status,expected,expected,[])
        with self.assertRaisesRegex(ValueError,'source SHA256'):
            admit.validate_closure(admit.BUILD_COMMIT,admit.REVIEWED_HEAD,'',{'src/a.cpp':'missing'}, {'src/a.cpp':'missing'},[])
        for path in ('/src/a.cpp','../src/a.cpp','src/../a.cpp','src//a.cpp'):
            with self.assertRaisesRegex(ValueError,'relative source path'):
                admit.validate_closure(admit.BUILD_COMMIT,admit.REVIEWED_HEAD,'',{path:'a'*64},{path:'a'*64},[])

    def test_conservative_headers_and_cmake_literal(self):
        tracked=['src/kernels/cpu/native_expert.cpp','include/expert.hpp','sycl/include/dpct/device.hpp','sycl/CMakeLists.txt','docs/README.md','serve/server.py','src/unused.cpp']
        got=admit.conservative_inputs(tracked,['src/kernels/cpu/native_expert.cpp'])
        self.assertEqual(got,['include/expert.hpp','src/kernels/cpu/native_expert.cpp','src/unused.cpp','sycl/CMakeLists.txt','sycl/include/dpct/device.hpp'])
        with self.assertRaisesRegex(ValueError,'absent from build commit'): admit.conservative_inputs(tracked,['src/missing.cpp'])

    def test_actual_compile_native_path_and_root(self):
        def command(path): return dict(directory='/build',command='pinned literal compile command',file=path)
        valid=[command('/project/src/kernels/cpu/native_expert.cpp'),command('/ggml/ggml/src/ggml.c')]
        self.assertEqual(admit.compile_paths(valid,'/project','/ggml','/build'),dict(project=['src/kernels/cpu/native_expert.cpp'],ggml=['ggml/src/ggml.c']))
        with self.assertRaisesRegex(ValueError,'production native source path'):
            admit.compile_paths([command('/project/sycl/src/kernels/cpu/native_expert.cpp')],'/project','/ggml','/build')
        with self.assertRaisesRegex(ValueError,'outside pinned roots'):
            admit.compile_paths(valid+[command('/other/a.cpp')],'/project','/ggml','/build')
        invalid=dict(valid[0],directory='/wrong')
        with self.assertRaisesRegex(ValueError,'compile directory'): admit.compile_paths([invalid],'/project','/ggml','/build')

    def test_setup_failure_closes_all_resources_and_never_math(self):
        observed=[]
        class Fake:
            def __init__(self,name,fail=False): self.name=name;self.fail=fail
            def close(self):
                observed.append(self.name)
                if self.fail: raise OSError('literal close failure')
        errors=admit.close_setup_resources([Fake('raw',True),None,Fake('events'),Fake('lock')])
        self.assertEqual(observed,['raw','events','lock'])
        self.assertEqual(len(errors),1)
        result=admit.failed_setup(RuntimeError('literal initial save failure'))
        for key in ('active','completed','healthy','math_gate_passed','reference_established','protocol_boundary_gate_passed','text_budget_gate_passed','performance_eligible','adopted','full_lifecycle_passed'):
            self.assertIs(result[key],False)
        self.assertIn('initial save failure',result['error'])

    def test_collector_source_guard_retains_rules_and_protected_setup(self):
        # Source guard only; does not execute/import collector or prove actual I/O.
        from pathlib import Path
        source=Path(__file__).with_name('long_prompt_reference_v2.py').read_text()
        self.assertIn("RULES_SHA='f6c6e4fe4765b51563bef3914446213fd6fab9e1c0832d2b895f1f4ea4079229'",source)
        self.assertIn("out=BASE/'owned-long-prompt-baseline-v0141-reference-r2'",source)
        self.assertIn("source_closure(build)",source)
        self.assertIn("try:\n        require(shutil.disk_usage(BASE).free>",source)
        self.assertIn("out.mkdir(mode=0o700);(out/'probes').mkdir(mode=0o700)",source)
        self.assertIn("finally: lock.close()",source)
        self.assertIn("compare_states(result['prefill_state'],seen[key]['prefill_state'],n-1)",source)
        self.assertIn("rules.validate_request(result,n)",source)
        self.assertIn("rules.repeat_checks(result,seen[key],state_cmp)",source)
        self.assertNotIn("args += ['--pool-tasks'",source)


if __name__=='__main__': unittest.main()
