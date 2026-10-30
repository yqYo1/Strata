"""Streaming-boundary and scan-cost checks for the Niko46c152 parser port.

Main-authored tests, repository LICENSE. Run:
    python -m unittest tools.test_frontend_stream_scan
"""
import json
import random
import unittest
from unittest import mock

from serve import frontend as f


TOOL = {"name": "run", "parameters": {"type": "object", "properties": {
    "command": {"type": "string"}}}}


def call(value):
    return (f.CALL_START + "<function=run><parameter=command>" + value +
            f.PARAM_END + f.FUNC_END + f.CALL_END)


def events(chunks, stream_tools=False):
    parser = f.OutputParser(thinking=True, tools=[TOOL], stream_tools=stream_tools)
    result = []
    for chunk in chunks:
        result.extend(parser.feed(chunk))
    result.extend(parser.finish())
    return result


class HoldBoundary(unittest.TestCase):
    def test_proper_prefix_suffix_including_unicode_and_overlapping_tags(self):
        parser = f.OutputParser()
        tags = (f.THINK_END, f.CALL_START, "<tool", "<", "", "🙂終")
        samples = ["", "plain🙂", "<" * 40, "</think>", "<tool_call>", "🙂"]
        samples += ["前🙂" + tag[:i] for tag in tags for i in range(len(tag) + 1)]
        rng = random.Random(460152)
        samples += ["".join(rng.choices("<>/think_🙂終", k=rng.randrange(50))) for _ in range(300)]
        for text in samples:
            expected = max((n for tag in tags for n in range(1, len(tag))
                            if text.endswith(tag[:n])), default=0)
            self.assertEqual(parser._hold(text, tags), expected, repr(text))


class StreamBoundary(unittest.TestCase):
    def test_split_tags_literal_closers_and_two_calls_keep_values_and_streamed_json(self):
        values = ["a🙂 </tool_call> quoted </function> and </parameter> literal\nz", "次🦊"]
        text = "思考🙂</think>\n\n前文\n\n" + call(values[0]) + "\n\n" + call(values[1]) + "\n後続"
        partitions = [[text], list(text)] + [[text[:i], text[i:]] for i in range(len(text) + 1)]
        for stream in (False, True):
            for chunks in partitions:
                got = events(chunks, stream)
                self.assertEqual("".join(e.text for e in got if e.kind == "reasoning"), "思考🙂")
                self.assertEqual("".join(e.text for e in got if e.kind == "content"), "前文後続")
                calls = [e.call for e in got if e.kind == "tool_call"]
                self.assertEqual([(c.name, c.arguments) for c in calls],
                                 [("run", {"command": v}) for v in values])
                if stream:
                    self.assertEqual([e.call.id for e in got if e.kind == "tool_start"],
                                     [c.id for c in calls])
                    for c in calls:
                        payload = "".join(e.text for e in got if e.kind == "tool_args" and e.call.id == c.id)
                        self.assertEqual(json.loads(payload), c.arguments)

    def test_partial_thinking_and_call_openers_are_held_until_finish(self):
        parser = f.OutputParser()
        self.assertEqual(parser.feed("考え</thi"), [f.Event("reasoning", "考え")])
        self.assertEqual(parser.finish(), [f.Event("reasoning", "</thi")])
        parser = f.OutputParser(thinking=False)
        self.assertEqual(parser.feed("答え\n<tool_"), [f.Event("content", "答え")])
        self.assertEqual(parser.finish(), [f.Event("content", "\n<tool_")])

    def test_long_incomplete_call_avoids_rewalking_its_growing_body(self):
        parser = f.OutputParser(thinking=False, tools=[TOOL])
        with mock.patch.object(f, "call_end", wraps=f.call_end) as walk:
            self.assertEqual(parser.feed(f.CALL_START + "<function=run><parameter=command>"), [])
            for _ in range(200):
                self.assertEqual(parser.feed("ordinary🙂" * 20), [])
            self.assertEqual(walk.call_count, 0)
            got = parser.feed(f.PARAM_END + f.FUNC_END + f.CALL_END)
            self.assertEqual(walk.call_count, 1)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].call.arguments, {"command": "ordinary🙂" * 4000})

    def test_literal_close_candidate_keeps_structural_disambiguation(self):
        parser = f.OutputParser(thinking=False, tools=[TOOL])
        value = "literal </tool_call> followed by more data🙂"
        prefix = f.CALL_START + "<function=run><parameter=command>"
        self.assertEqual(parser.feed(prefix + value), [])
        self.assertEqual(parser.feed(f.PARAM_END + f.FUNC_END), [])
        result = parser.feed(f.CALL_END)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].call.arguments, {"command": value})


if __name__ == "__main__":
    unittest.main()
