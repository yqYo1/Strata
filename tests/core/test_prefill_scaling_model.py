"""Admission checks for source-owned length characterization;no GPU or model."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import prefill_scaling_model as model


class ScalingAdmission(unittest.TestCase):
    def fixture(self):
        samples=[]
        lines=['strata serve: prompt 74 tokens = 0 reused + 74 read, 24 generated']
        for minimum,run in model.request_order():
            n=minimum*5//4+17
            samples.append(dict(actual_prompt_tokens=n,actual_read_tokens=n,generated_tokens=24,
                                requested_min_tokens=minimum,run=run))
            if n-1>8192:
                lines.extend([f'strata prefill: layer-major candidate, tokens={n-1} chunk=8192 residual_dtype=bf16',
                              'strata MoE window configuration: requested_mode=10 effective_mode=10'])
            lines.append(f'strata serve: prompt {n} tokens = 0 reused + {n} read, 24 generated')
        return '\n'.join(lines)+'\n',samples

    def test_actual_native_routes_and_counts_bind_to_every_request(self):
        text,samples=self.fixture()
        rows=model.route_evidence(text,samples)
        self.assertEqual(len(rows),19)
        self.assertEqual(rows[0]['run'],-1)
        self.assertEqual(sum(r['route']=='non-layer-major short' for r in rows),6)
        self.assertEqual(sum(r['route']=='layer-major BF16 mode10' for r in rows),13)

    def test_api_count_relabeling_is_rejected(self):
        text,samples=self.fixture()
        samples[0]['actual_read_tokens']+=1
        with self.assertRaisesRegex(RuntimeError,'native/API'):
            model.route_evidence(text,samples)

    def test_missing_duplicate_or_wrong_native_route_is_rejected(self):
        text,samples=self.fixture()
        for wrong in (text.replace('chunk=8192','chunk=4096',1),
                      text.replace('residual_dtype=bf16','residual_dtype=f32',1),
                      text.replace('effective_mode=10','effective_mode=2',1),
                      text.replace('strata prefill: layer-major candidate','unrelated',1)):
            with self.subTest(wrong=wrong[:80]),self.assertRaises(RuntimeError):
                model.route_evidence(wrong,samples)
        with self.assertRaisesRegex(RuntimeError,'footer count'):
            model.route_evidence(text+text.splitlines()[-1]+'\n',samples)


if __name__=='__main__':
    unittest.main()
