"""Reject unsafe paired-QSA metadata and trace evidence without a GPU.

Authored for Strata; repository LICENSE applies. In particular, preserving only
the multiset of local sizes must not hide a changed argument binding.
"""
import copy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import qsa_head_native as native


def properties():
    return [dict(name='_ZTS' + marker + mode, arguments=str(arguments), wg='128')
            for marker in ['17qsa_head_original', '18qsa_head_candidate']
            for mode, arguments in [('NORMAL', 18), ('STATE', 19)]]


def trace(props, changed=None, omit=None, repeated=None):
    result = []
    local = list(enumerate([16384, 1024, 64, 64, 64, 256, 64, 2112], 6))
    for number, prop in enumerate(props):
        handle = hex(256 + number)
        result.append(f"<--- urKernelCreate(.name = 0x1 ({prop['name']}), .phKernel = 0x1 ({handle})) -> UR_RESULT_SUCCESS")
        if number == omit:
            continue
        variants = [local if number != changed else [(i, size) for i, size in reversed(local)]]
        if repeated == number:
            variants.append([(i, local[1][1] if i == local[0][0] else local[0][1] if i == local[1][0] else size) for i, size in local])
        for signature in variants:
            args = '; '.join(f'UR_EXP_KERNEL_ARG_TYPE_LOCAL, .index = {i}, .size = {size}' for i, size in signature)
            result.append(f"<--- urEnqueueKernelLaunchWithArgsExp(.hKernel = {handle}, .numArgs = {prop['arguments']}, .pLocalWorkSize = 0x1 (128), {args}) -> UR_RESULT_SUCCESS")
    return '\n'.join(result)


class PairedAdmission(unittest.TestCase):
    def setUp(self):
        self.props = properties()

    def rejected(self, text, props=None):
        with self.assertRaises(RuntimeError):
            native.dispatches(text, self.props if props is None else props)

    def test_valid_paired_normal_and_state(self):
        proof = native.dispatches(trace(self.props), self.props)
        self.assertTrue(proof['paired_signatures_equal'])
        self.assertEqual(list(proof['counts'].values()), [1, 1, 1, 1])
        self.assertEqual(len(proof['indexed_local_arguments']), 4)

    def test_trace_array_order_does_not_change_index_binding(self):
        self.assertTrue(native.dispatches(trace(self.props, changed=2), self.props)['paired_signatures_equal'])

    def test_pair_argument_count_mismatch(self):
        changed = copy.deepcopy(self.props)
        changed[2]['arguments'] = '20'
        self.rejected(trace(changed), changed)

    def test_missing_paired_property(self):
        self.rejected(trace(self.props), self.props[:-1])

    def test_duplicate_paired_property(self):
        self.rejected(trace(self.props), self.props + [self.props[0]])

    def test_rebinding_same_multiset_is_rejected(self):
        # Both sizes remain present, but a different local argument gets them.
        text = trace(self.props)
        line = text.splitlines()[5]
        changed = line.replace('.index = 6, .size = 16384', '.index = 6, .size = 1024').replace('.index = 7, .size = 1024', '.index = 7, .size = 16384')
        self.rejected(text.replace(line, changed))

    def test_local_signature_cannot_change_between_launches(self):
        self.rejected(trace(self.props, repeated=2))

    def test_duplicate_local_argument_index(self):
        self.rejected(trace(self.props).replace('.index = 7, .size = 1024', '.index = 6, .size = 1024'))

    def test_out_of_range_argument_index(self):
        self.rejected(trace(self.props).replace('.index = 6, .size = 16384', '.index = 99, .size = 16384'))

    def test_actual_count_and_workgroup_must_match_properties(self):
        for text in [trace(self.props).replace('.numArgs = 18,', '.numArgs = 20,'),
                     trace(self.props).replace('(128)', '(256)')]:
            self.rejected(text)

    def test_missing_state_enqueue(self):
        self.rejected(trace(self.props, omit=3))


if __name__ == '__main__':
    unittest.main()
