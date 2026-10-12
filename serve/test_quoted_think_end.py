# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Quoted reasoning tags survive arbitrary stream boundaries; the real end still closes reasoning."""
import unittest

from serve.frontend import OutputParser


def parsed(parts):
    parser = OutputParser(thinking=True)
    events = []
    for part in parts:
        events.extend(parser.feed(part))
    events.extend(parser.finish("stop"))
    return tuple("".join(event.text for event in events if event.kind == kind)
                 for kind in ("reasoning", "content"))


class QuotedThinkEnd(unittest.TestCase):
    def test_quoted_tags_at_every_stream_boundary(self):
        for quote in ('"', "'", "`"):
            reasoning = f"Here is {quote}</think>{quote} in prose."
            text = reasoning + "</think>\n\nAnswer."
            for split in range(len(text) + 1):
                with self.subTest(quote=quote, split=split):
                    self.assertEqual(parsed([text[:split], text[split:]]), (reasoning, "Answer."))
            self.assertEqual(parsed(list(text)), (reasoning, "Answer."))

    def test_real_end_followed_by_newline_after_a_quote(self):
        reasoning = "The last character is a quote `"
        text = reasoning + "</think>\n\nAnswer."
        self.assertEqual(parsed(list(text)), (reasoning, "Answer."))

    def test_held_real_end_at_eof_is_consumed(self):
        reasoning = "The last character is a quote `"
        self.assertEqual(parsed(list(reasoning + "</think>")), (reasoning, ""))


if __name__ == "__main__":
    unittest.main()
