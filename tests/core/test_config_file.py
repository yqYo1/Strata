"""Config cases from Niko1221/Strata ffa80d73 JoinedEngineArgs, plus error checks.

Copyright (c) 2026 Niko1221 and the Strata contributors. Repository LICENSE (MIT).
"""
import contextlib
import io
import pathlib
import tempfile
import unittest

from serve.config_file import parse_config_text, read_config, split_joined_args


class ConfigFileTests(unittest.TestCase):
    def split(self, args):
        config = {"args": list(args)}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            split_joined_args(config)
        return config["args"], output.getvalue()

    def test_joined_engine_options(self):
        args, notice = self.split(["--spec", "4", "--ple-io mmap", "--kv=int8"])
        self.assertEqual(args, ["--spec", "4", "--ple-io", "mmap", "--kv", "int8"])
        self.assertIn("two elements", notice)

    def test_literal_values_and_paths_stay_whole(self):
        for args in (["--system-prompt", "--be brief please"],
                     ["--native", "/data/My Models/a b.gguf", "--vision"],
                     ["--ctx", "4096", "--spec", "4"]):
            with self.subTest(args=args):
                self.assertEqual(self.split(args), (args, ""))

    def test_no_argument_list(self):
        for config in ({}, {"args": None}, {"args": "--spec 4"}):
            original = dict(config)
            split_joined_args(config)
            self.assertEqual(config, original)

    def test_invalid_json_reports_file_and_location(self):
        with self.assertRaises(SystemExit) as caught:
            parse_config_text('{\n"args": ["--spec" "4"]}', "run.json")
        message = str(caught.exception)
        self.assertIn("run.json", message)
        self.assertIn("line 2, column", message)
        self.assertIn("each option and each value", message)
        self.assertTrue(caught.exception.__suppress_context__)

    def test_notepad_bom_and_missing_comma(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "run.json"
            path.write_text('\ufeff{"args": ["--ple-io", "mmap"]}', encoding="utf-8")
            self.assertEqual(read_config(path), {"args": ["--ple-io", "mmap"]})
            path.write_text('{"args": ["--ple-io" "mmap"]}', encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                read_config(path)
            self.assertIn(str(path), str(caught.exception))


if __name__ == "__main__":
    unittest.main()
