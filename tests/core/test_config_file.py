"""Config cases from Niko1221/Strata ffa80d73 JoinedEngineArgs, plus error checks.

Copyright (c) 2026 Niko1221 and the Strata contributors. Repository LICENSE (MIT).
"""
import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

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


class ConfigServerTests(unittest.TestCase):
    """Main-authored entrypoint checks; stop before any native engine starts."""

    def test_main_repairs_args_before_engine_construction(self):
        from serve import server

        class EngineReached(Exception):
            pass

        expected = ["--spec", "4", "--ple-io", "mmap", "--kv", "int8",
                    "--native", "/data/My Models/a b.gguf", "--system-prompt", "--be brief please"]
        joined = ["--spec", "4", "--ple-io mmap", "--kv=int8", *expected[6:]]
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            (root / "vocab.json").write_text('{"<|im_end|>": 0}', encoding="utf-8")
            (root / "merges.txt").write_text("", encoding="utf-8")
            (root / "token_type.json").write_text("[3]", encoding="utf-8")
            path = root / "run.json"
            tokenizer = SimpleNamespace(Tokenizer=lambda *args: server.ByteTokenizer())
            for args in (joined, expected):
                with self.subTest(args=args):
                    cfg = {"exe": sys.executable, "args": args, "cwd": folder, "tokenizer": folder}
                    path.write_text("\ufeff" + json.dumps(cfg), encoding="utf-8")
                    output = io.StringIO()
                    with mock.patch.object(server, "Server") as listener, \
                         mock.patch.object(server, "StrataEngine", side_effect=EngineReached) as engine, \
                         mock.patch.object(server, "hub_from_config", return_value=None), \
                         mock.patch.dict(sys.modules, {"strata_tokenizer": tokenizer}), \
                         mock.patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(path)]), \
                         contextlib.redirect_stdout(output), self.assertRaises(EngineReached):
                        server.main()
                    self.assertEqual(engine.call_args.args[1], expected)
                    self.assertEqual(listener.call_count, 1)  # only the mocked port check
                    self.assertEqual("two elements" in output.getvalue(), args is joined)

    def test_bad_config_stops_before_port_or_engine(self):
        from serve import server

        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "run.json"
            path.write_text('{\n"args": ["--spec" "4"]}', encoding="utf-8")
            with mock.patch.object(server, "Server") as listener, \
                 mock.patch.object(server, "StrataEngine") as engine, \
                 mock.patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(path)]), \
                 self.assertRaises(SystemExit) as caught:
                server.main()
            self.assertIn(f"{path} is not valid JSON", str(caught.exception))
            self.assertIn("line 2, column", str(caught.exception))
            listener.assert_not_called()
            engine.assert_not_called()

    def test_bad_config_cli_reports_location_without_traceback(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "run.json"
            path.write_text('{\n"args": ["--spec" "4"]}', encoding="utf-8")
            for entry in (["-m", "serve.server"], [str(root / "serve/server.py")]):
                with self.subTest(entry=entry):
                    result = subprocess.run([sys.executable, "-B", *entry, "--engine", "strata", "--config", str(path)],
                                            cwd=root, capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(str(path), result.stderr)
                    self.assertIn("line 2, column", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
