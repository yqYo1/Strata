"""Reject incomplete/ambiguous callback-route discovery, without GPU calls."""
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import prefill_route as route


def fixture():
    # The warmup may legitimately use a token/graph fallback. It cannot supply
    # evidence for the long request, whose markers follow its completed footer.
    rows = ['strata serve: DRAFT_PREFILL path=token mode=2 cells=7',
            'strata serve: prompt 8 tokens = 0 reused + 8 read, 1 generated',
            'strata prefill: layer-major candidate, tokens=117488 chunk=8192 '
            'residual_dtype=bf16 residual_element_bytes=2 residual_total=2406154240 '
            'residual_gpu_rows=117488 residual_gpu=2406154240 residual_host=0 residual_staging=0 layer_cache=1022361600',
            'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=81920']
    rows.extend('strata serve: DRAFT_PREFILL path=batched mode=2 cells=' + str(n) for n in [8192] * 14 + [2800])
    rows.extend(['strata prefill dense: phase=restored residency=Full activations=48 full_bytes=1',
                 'strata serve: prompt 117489 tokens = 0 reused + 117489 read, 24 generated'])
    return '\n'.join(rows) + '\n'


class RouteEvidence(unittest.TestCase):
    def test_complete_long_route_excludes_warmup(self):
        result = route.analyze(fixture(), dict(read_tokens=117489, reused_tokens=0,
            prompt_tokens=117489, generated_tokens=24))
        self.assertEqual(result['batch_count'], 15)
        self.assertEqual(result['draft_cells'], 117488)
        self.assertEqual(result['final_layer_callback_graph_fallbacks'], 0)
        self.assertFalse(result['library_internal_graphs_qualified'])

    def reject(self, text):
        with self.assertRaises(RuntimeError):
            route.analyze(text, dict(read_tokens=117489, reused_tokens=0,
                prompt_tokens=117489, generated_tokens=24))

    def test_missing_duplicate_wrong_extent_and_fallback(self):
        original = 'strata serve: DRAFT_PREFILL path=batched mode=2 cells=2800\n'
        for replacement in ('', original * 2, original.replace('2800', '2799'),
                            original.replace('batched', 'token'), original.replace('mode=2', 'mode=0')):
            with self.subTest(replacement=replacement): self.reject(fixture().replace(original, replacement))

    def test_trailing_callback_is_not_bound_to_finished_request(self):
        self.reject(fixture() + 'strata serve: DRAFT_PREFILL path=batched mode=2 cells=64\n')

    def test_missing_or_duplicate_restore(self):
        line = 'strata prefill dense: phase=restored residency=Full activations=48 full_bytes=1\n'
        for replacement in ('', line * 2, line.replace('activations=48', 'activations=47')):
            with self.subTest(replacement=replacement): self.reject(fixture().replace(line, replacement))

    def test_cpu_residual_or_mode_fallback(self):
        self.reject(fixture().replace('residual_gpu_rows=117488', 'residual_gpu_rows=8192'))
        self.reject(fixture().replace('effective_mode=10', 'effective_mode=2'))

    def test_readback_and_timer_observers_rejected(self):
        for marker in ('strata draft kv: 117488 cells', 'strata serve: SNAPSHOT_VERIFY result=PASS',
                       'strata serve: STATE_HASH value=1'):
            with self.subTest(marker=marker): self.reject(fixture() + marker + '\n')

    def test_native_output_count_and_prefix_reuse_differ(self):
        self.reject(fixture().replace('24 generated', '23 generated'))
        self.reject(fixture().replace('0 reused + 117489', '1 reused + 117488'))


if __name__ == '__main__': unittest.main()
