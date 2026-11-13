"""Host controls for current-runtime provenance and combined observer admission."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_current_decode as profile


class CurrentProfileTest(unittest.TestCase):
    def sources(self):
        sources = {'include/owned.hpp': 'original',
            'tools/sycl/profile_current_decode.py': 'controller',
            'tools/sycl/test_profile_current_decode.py': 'tests'}
        proof = dict(observer_source_files=3, observer_source_sha256=profile.control.signature(sources),
            declared_observers={p: h for p, h in sources.items() if p.endswith('.py')},
            additional_observers=sorted(p for p in sources if p.endswith('.py')),
            native_source_commit='393', class_coherence={'all_objects_from_fresh_build': True})
        return proof, sources

    def cases(self):
        return [dict(label=label, samples=[dict(request_sha256=str(i) * 64, generated_tokens=24,
            prompt_ms=100., decode_ms=200., ttft_s=1.) for i in (1, 2)]) for label in profile.ORDER]

    def test_projection_preserves_original_source_and_class_proof(self):
        proof, sources = self.sources(); original = copy.deepcopy(proof)
        projected = profile.observer_projection(proof, sources)
        self.assertEqual(projected['observer_source_files'], 1)
        self.assertEqual(projected['observer_source_sha256'], profile.control.signature({'include/owned.hpp': 'original'}))
        self.assertEqual(projected['declared_observers'], {})
        self.assertEqual(projected['native_source_commit'], '393')
        self.assertEqual(projected['class_coherence'], original['class_coherence'])
        self.assertEqual(proof, original)

    def test_projection_refuses_incomplete_or_changed_observer_evidence(self):
        proof, sources = self.sources()
        for mutate in (lambda p: p.update(observer_source_files=2),
                       lambda p: p.update(observer_source_sha256='wrong'),
                       lambda p: p['declared_observers'].pop('tools/sycl/profile_current_decode.py'),
                       lambda p: p['additional_observers'].pop()):
            wrong = copy.deepcopy(proof); mutate(wrong)
            with self.assertRaises(RuntimeError): profile.observer_projection(wrong, sources)

    def test_each_observer_phase_has_its_own_gate(self):
        cases = self.cases()
        self.assertTrue(profile.observer_decision(cases)['observer_screen_passed'])
        cases[1]['samples'][0]['decode_ms'] = 302.
        result = profile.observer_decision(cases)
        self.assertFalse(result['observer_screen_passed'])
        self.assertTrue(result['requests'][0]['phases']['prompt_ms']['screen_passed'])
        self.assertFalse(result['requests'][0]['phases']['decode_ms']['screen_passed'])
        self.assertTrue(result['requests'][1]['observer_screen_passed'])
        self.assertFalse(result['clean_speed_qualified'])
        self.assertFalse(result['sustained_TG_qualified'])

    def test_changed_work_or_nonfinite_time_cannot_qualify_observation(self):
        cases = self.cases(); cases[1]['samples'][1]['generated_tokens'] = 23
        self.assertFalse(profile.observer_decision(cases)['observer_screen_passed'])
        for field, value in (('request_sha256', 'different'), ('decode_ms', float('nan'))):
            wrong = self.cases(); wrong[1]['samples'][0][field] = value
            with self.assertRaises(RuntimeError): profile.observer_decision(wrong)


if __name__ == '__main__': unittest.main()
