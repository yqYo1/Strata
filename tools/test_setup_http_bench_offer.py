# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The setup HTTP benchmark offer only displays an explicit attach command, with no engine/model work."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup as S  # noqa: E402


class Offer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "tools").mkdir()
        (self.root / "tools" / "strata_bench.py").write_text("# fixture, never executed\n")
        self.cfg = self.root / "config with spaces.json"

    def call(self, answer="n", yes=False, tty=True):
        with mock.patch.object(S, "ROOT", self.root), mock.patch.object(S, "STATE", self.root / "state"), \
                mock.patch.object(S.sys.stdin, "isatty", return_value=tty), \
                mock.patch.object(S, "ask", return_value=answer) as ask, \
                mock.patch.object(S, "say") as say:
            result = S.offer_http_bench(self.cfg, 8095, yes)
            return result, ask.call_args_list, say.call_args_list

    def test_default_no_and_noninteractive_are_quiet(self):
        result, asks, said = self.call()
        self.assertFalse(result)
        self.assertEqual(asks[0].args[2], "n")
        self.assertEqual(said, [])
        for yes, tty in ((True, True), (False, False)):
            result, asks, said = self.call("y", yes, tty)
            self.assertFalse(result)
            self.assertEqual((asks, said), ([], []))

    def test_acceptance_only_shows_quoted_attach_command(self):
        result, _, said = self.call("y")
        self.assertTrue(result)
        text = "\n".join(c.args[0] for c in said)
        self.assertIn("--url http://127.0.0.1:8095", text)
        self.assertIn("'" + str(self.cfg) + "'", text)
        self.assertIn("xestrata --bench", text)
        self.assertIn("COMMUNITY_BENCHMARKS.md", text)


if __name__ == "__main__":
    unittest.main()
