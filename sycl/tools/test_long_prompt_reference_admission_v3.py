"""Independent reachable-target fixtures, including the retained real build metadata."""
import hashlib
from pathlib import Path
import unittest
import long_prompt_reference_admission_v3 as admission


class TargetClosure(unittest.TestCase):
    def test_compile_and_link_rules_are_distinct(self):
        commands = '\n'.join([
            '/compiler/icpx -O3 -o native.o -c /project/src/kernels/cpu/native_expert.cpp',
            '/compiler/icpx -O3 -o gen.o -c /project/sycl/src/program/generate.cpp',
            '/compiler/icx -O3 -o ggml.o -c /ggml/ggml/src/ggml.c',
            ': && /compiler/icpx native.o gen.o ggml.o -o strata && :',
        ])
        result = admission.target_commands_paths(commands, '/project', '/ggml')
        self.assertEqual(result, dict(sources=dict(project=['src/kernels/cpu/native_expert.cpp', 'sycl/src/program/generate.cpp'], ggml=['ggml/src/ggml.c']), compile_commands=3, unique_sources=3))

    def test_duplicate_source_still_has_two_compile_rules(self):
        line = '/compiler/icpx -o x.o -c /project/src/kernels/cpu/native_expert.cpp'
        result = admission.target_commands_paths(line+'\n'+line.replace('x.o', 'y.o'), '/project', '/ggml')
        self.assertEqual((result['compile_commands'], result['unique_sources']), (2, 1))

    def test_invalid_compile_sources_refused(self):
        valid = '/compiler/icpx -c /project/src/kernels/cpu/native_expert.cpp\n'
        for line, message in [
            ('/compiler/other -c /project/src/a.cpp', 'compile rule executable'),
            ('/compiler/icpx -c /other/a.cpp', 'outside pinned roots'),
            ('/compiler/icpx -c /project/src/../a.cpp', 'absolute target source path'),
            ('/compiler/icpx -c', 'source argument absent'),
            ('/compiler/icpx -c /project/a.cpp -c /project/b.cpp', 'compile rule executable'),
        ]:
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, message):
                admission.target_commands_paths(valid+line, '/project', '/ggml')
        with self.assertRaisesRegex(ValueError, 'actual target native source path'):
            admission.target_commands_paths('/compiler/icx -c /ggml/ggml/src/ggml.c', '/project', '/ggml')

    def test_actual_target_includes_native_missing_from_sparse_compdb(self):
        base = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
        data = (base/'baseline-869-source-equivalence-provenance-v1/target-commands.stdout').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), 'fadfe6129713809443e877df18a9d7ccc8b90c310d15c9ff34b0bb2888996779')
        project = '/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-v0.1.41-20261009'
        result = admission.target_commands_paths(data.decode(), project, '/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
        self.assertEqual((result['compile_commands'], result['unique_sources']), (115, 114))
        self.assertIn('src/kernels/cpu/native_expert.cpp', result['sources']['project'])
        self.assertIn('sycl/src/program/generate.cpp', result['sources']['project'])
        self.assertNotIn('sycl/src/kernels/cpu/native_expert.cpp', result['sources']['project'])

    def test_collector_binds_target_and_keeps_numerical_rules(self):
        source = Path(__file__).with_name('long_prompt_reference_v3.py').read_text()
        self.assertIn("RULES_SHA='f6c6e4fe4765b51563bef3914446213fd6fab9e1c0832d2b895f1f4ea4079229'", source)
        self.assertIn("owned-long-prompt-baseline-v0141-reference-r3", source)
        self.assertIn("admission.target_commands_paths(target_commands", source)
        self.assertNotIn("admission.compile_paths(commands", source)

    def test_dependency_closure_includes_arbitrary_suffix_and_normalizes_paths(self):
        text='obj: #deps 4, deps mtime 1 (VALID)\n    /project/src/../src/rows.inl\n    /ggml/ggml/include/../include/quant.unknown\n    /project/build/ggml/src/ggml-version.h\n    /usr/include/stdlib.h\n'
        value=admission.recorded_dependency_paths(text,'/project','/ggml','/project/build')
        self.assertEqual(value,dict(sources=dict(project=['src/rows.inl'],ggml=['ggml/include/quant.unknown'],generated=['ggml/src/ggml-version.h']),object_blocks=1,external_unique_paths=1))
        for malformed in (text.replace('#deps 4','#deps 5'),text.replace('(VALID)','(STALE)'),text.replace('/usr/include/stdlib.h','relative.h')):
            with self.subTest(text=malformed), self.assertRaises(ValueError):
                admission.recorded_dependency_paths(malformed,'/project','/ggml','/project/build')

    def test_changed_inline_dependency_is_rejected(self):
        paths=admission.conservative_inputs(['src/native.cpp','src/rows.inl','src/rows.ipp','src/rows.tpp'],['src/native.cpp'])
        self.assertEqual(len(paths),4)
        expected={p:'a'*64 for p in paths};actual=dict(expected);actual['src/rows.inl']='b'*64
        with self.assertRaisesRegex(ValueError,'changed compiled-input source: src/rows.inl'):
            admission.validate_closure(admission.BUILD_COMMIT,admission.REVIEWED_HEAD,'',expected,actual,['src/rows.inl'])

    def test_actual_recorded_object_dependency_superset(self):
        base=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
        data=(base/'baseline-869-source-equivalence-provenance-v1/target-deps.stdout').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),'e1585f1b8c79cc5e1c900b5edb6e96183032e290d38316c5825c268b4088021f')
        project='/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-v0.1.41-20261009'
        value=admission.recorded_dependency_paths(data.decode(),project,'/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned',project+'/build-sycl-integrated-20261009')
        sources=value['sources']
        self.assertEqual((len(sources['project']),len(sources['ggml'])),(241,52))
        self.assertEqual(sources['generated'],['ggml/src/ggml-version.h'])
        self.assertIn('src/kernels/cpu/iq_avx2_rows.inl',sources['project'])
        self.assertIn('src/kernels/cpu/q2_avx2_rows.inl',sources['project'])


if __name__ == '__main__':
    unittest.main()
