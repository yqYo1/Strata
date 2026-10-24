#!/usr/bin/env python3
"""CPU source/receipt refusal checks for the private MTP control; no GPU."""
import unittest
import mtp_resident_control as m


class SourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = {n: raw for n, raw in m.build.blobs(m.P, m.BASE,
            ['src/core/mtp.cpp', 'src/program/generate.cpp'])}
        cls.generated = {n: m.transformed(n, raw) for n, raw in cls.original.items()}

    def test_binary_and_unrelated_source_pass_through(self):
        for name in ('src/prefill/prefill.cpp', 'include/strata/core/mtp.hpp', 'fixture.bin'):
            self.assertEqual(m.transformed(name, b'\xff\x00\x80'), b'\xff\x00\x80')

    def test_changed_source_requires_exact_unique_anchors(self):
        for name, raw in self.original.items():
            anchor = m.OBSERVER if name.endswith('mtp.cpp') else m.GUARD_ANCHOR
            for bad in (raw.replace(anchor.encode(), b''), raw + anchor.encode()):
                with self.subTest(name=name), self.assertRaises(RuntimeError): m.transformed(name, bad)

    def test_only_declared_mtp_edits_and_no_prefill_or_first_draft_edit(self):
        original = self.original['src/core/mtp.cpp'].decode()
        changed = self.generated['src/core/mtp.cpp'].decode()
        restored = changed.replace(m.CAP_ZERO + m.OBSERVER + m.CAP_ANCHOR, m.CAP_ANCHOR, 1)
        anchor = '                       float* probs, float min_p, int* n_drafts) {\n'
        self.assertEqual(restored.count(anchor), 1)
        restored = restored.replace(anchor, anchor + m.OBSERVER, 1)
        self.assertEqual(restored, original)
        self.assertEqual(changed.split('bool MtpDrafter::draft(')[0], original.split('bool MtpDrafter::draft(')[0])
        self.assertEqual(changed.split('bool MtpDrafter::draft_first(')[1], original.split('bool MtpDrafter::draft_first(')[1])

    def test_program_has_only_scope_guard_and_after_timing_summary(self):
        original = self.original['src/program/generate.cpp'].decode()
        changed = self.generated['src/program/generate.cpp'].decode()
        self.assertEqual(changed.replace(m.GUARD, '', 1).replace(m.FOOTER, '', 1), original)
        self.assertLess(changed.index(m.GUARD), changed.index(m.GUARD_ANCHOR))
        self.assertLess(changed.index('"strata serve: prompt %'), changed.index(m.FOOTER))

    def test_cap_zero_cannot_bypass_owners_bounds_or_open_observer(self):
        changed = self.generated['src/core/mtp.cpp'].decode().split('bool MtpDrafter::draft(')[1]
        checks = [changed.index(x) for x in ('const OnDevice on_device', 'head_detached_',
            '!experts_', 'T < 1 || T > max_t_ || a < 0 || a >= T', 'if (max_drafts_ == 0)',
            'graph_observer_.window', 'capture_round(T, cp, err)')]
        self.assertEqual(checks, sorted(checks))
        # Restoring the original suffix proves normal proposal math and its
        # synchronization are copied verbatim, rather than a second formula.
        self.assertEqual(changed.split(m.CAP_ANCHOR)[1],
            self.original['src/core/mtp.cpp'].decode().split('bool MtpDrafter::draft(')[1].split(m.CAP_ANCHOR)[1])

    def test_original_selected_identity_and_no_public_native_file_mutation(self):
        self.assertEqual(m.BASE, '6e40afc1beddd059ee518617f6c1efce1da14aca')
        for name, raw in self.original.items():
            self.assertNotEqual(self.generated[name], raw)
        self.assertEqual(m.ROOT.parent, m.B)
        self.assertFalse(m.ROOT.is_relative_to(m.P))


if __name__ == '__main__': unittest.main()
