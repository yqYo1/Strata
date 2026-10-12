# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream tool-order compatibility and prompt-cache prefix regressions."""
import unittest


class ToolOrder(unittest.TestCase):
    """A tool set that comes back in another order is put back in the order it was first seen (RealScaniX, #1624): the
    template renders the tool list before the system prompt, so a client whose MCP tools register in a varying order
    would move the shared prefix back thousands of tokens on every turn and have the conversation read again.  The
    first request of a set is left exactly as sent."""

    def names(self, tools, wrapper="function"):
        from serve.frontend import _tool_list
        return [t.get(wrapper, t)["name"] for t in _tool_list(tools, wrapper)]

    def test_openai_shape_keeps_the_first_order(self):
        a = [{"type": "function", "function": {"name": n}} for n in ("t1624_read", "t1624_rbash", "t1624_ctx7")]
        self.assertEqual(self.names(a), ["t1624_read", "t1624_rbash", "t1624_ctx7"])      # the first sight: as sent
        self.assertEqual(self.names(list(reversed(a))), ["t1624_read", "t1624_rbash", "t1624_ctx7"])
        self.assertEqual(self.names(a[1:] + a[:1]), ["t1624_read", "t1624_rbash", "t1624_ctx7"])

    def test_anthropic_shape_and_a_different_set(self):
        a = [{"name": "t1624_read", "input_schema": {}}, {"name": "t1624_edit", "input_schema": {}}]
        self.assertEqual(self.names(a, None), ["t1624_read", "t1624_edit"])
        self.assertEqual(self.names(a[::-1], None), ["t1624_read", "t1624_edit"])
        more = a[::-1] + [{"name": "t1624_new", "input_schema": {}}]       # another set: as sent, then remembered
        self.assertEqual(self.names(more, None), ["t1624_edit", "t1624_read", "t1624_new"])

    def test_duplicate_names_and_one_tool_are_left_alone(self):
        d = [{"name": "t1624_x", "input_schema": {}}, {"name": "t1624_x", "input_schema": {"a": 1}}]
        self.assertEqual(self.names(d, None), ["t1624_x", "t1624_x"])
        one = [{"name": "t1624_only", "input_schema": {}}]
        self.assertEqual(self.names(one, None), ["t1624_only"])


if __name__ == "__main__":
    unittest.main()
