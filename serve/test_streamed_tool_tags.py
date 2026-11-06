"""Prospective Niko3bbf tool-tag behavior; Main-authored, repository LICENSE."""
import json
import unittest

from serve.frontend import OutputParser


SCHEMAS = [{"name": "write_file", "parameters": {"properties": {
    "path": {"type": "string"}, "content": {"type": "string"}}}},
    {"name": "read", "parameters": {"properties": {"path": {"type": "string"}}}}]
LITERAL = "Closing syntax:\n</parameter>\n</function>\n</tool_call>\nThese are file contents."


def wrapped(name, **arguments):
    return ("<tool_call><function=" + name + ">" + "".join(
        "<parameter=" + key + ">\n" + value + "\n</parameter>" for key, value in arguments.items()) +
        "</function></tool_call>")


class StreamedToolTags(unittest.TestCase):
    def parse(self, pieces, stream):
        parser = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=stream)
        events = []
        for piece in pieces:
            events.extend(parser.feed(piece))
        events.extend(parser.finish())
        return events

    def assert_case(self, text, expected, content):
        cuts = [[text], list(text)] + [[text[:i], text[i:]] for i in range(len(text)+1)]
        for stream in (False, True):
            for pieces in cuts:
                with self.subTest(stream=stream, lengths=[len(p) for p in pieces[:2]]):
                    events = self.parse(pieces, stream)
                    calls = [e.call for e in events if e.kind == "tool_call"]
                    self.assertEqual([(c.name, c.arguments) for c in calls], expected)
                    self.assertEqual("".join(e.text for e in events if e.kind == "content"), content)
                    if stream:
                        self.assertEqual([e.call.id for e in events if e.kind == "tool_start"],
                                         [c.id for c in calls])
                        for call in calls:
                            value = "".join(e.text for e in events if e.kind == "tool_args" and
                                            e.call.id == call.id)
                            self.assertEqual(json.loads(value), call.arguments)

    def test_literal_closing_sequence_in_file(self):
        args = {"path": "TOOLS.md", "content": LITERAL}
        self.assert_case("Saving the file.\n\n" + wrapped("write_file", **args),
                         [("write_file", args)], "Saving the file.")

    def test_literal_closing_sequence_without_prose(self):
        args = {"path": "TOOLS.md", "content": LITERAL}
        self.assert_case(wrapped("write_file", **args), [("write_file", args)], "")

    def test_prose_after_final_call(self):
        self.assert_case(wrapped("read", path="a.txt") + "\n\nRead complete.",
                         [("read", {"path": "a.txt"})], "Read complete.")

    def test_two_calls_with_prose_between(self):
        self.assert_case(wrapped("read", path="a.txt") + "\n\nNext file:\n\n" +
                         wrapped("read", path="b.txt"),
                         [("read", {"path": "a.txt"}), ("read", {"path": "b.txt"})], "Next file:")

    def test_tool_starts_and_arguments_stream_before_final_input(self):
        parser = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=True)
        events = parser.feed("<tool_call><function=write_file><parameter=content>\npartial")
        self.assertEqual(len([e for e in events if e.kind == "tool_start"]), 1)
        self.assertTrue(any(e.kind == "tool_args" and "partial" in e.text for e in events))
        self.assertFalse([e for e in events if e.kind == "tool_call"])
        self.assertFalse([e for e in parser.finish() if e.kind == "tool_call"])

    def test_lookahead_is_bounded(self):
        parser = OutputParser(thinking=False, tools=SCHEMAS)
        self.assertEqual(parser.feed(wrapped("read", path="a.txt")), [])
        events = parser.feed("x" * 1501)
        self.assertEqual(len([e for e in events if e.kind == "tool_call"]), 1)


if __name__ == "__main__":
    unittest.main()
