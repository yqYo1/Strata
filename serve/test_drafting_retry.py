"""GUI-FB3 item 2(b): an Auto start that picked DFlash2 and ended before the engine was ready is started once more with the MTP, in one quiet line. A start the person
named DFlash2 for fails with the engine's own reason. A real process and a real pipe (a fake engine program), like serve/test_fatal_recovery.py."""
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from serve.server import EngineDied, StrataEngine
from strata_control import drafting
from serve import server

FAKE = r'''
import sys
from pathlib import Path
args = sys.argv[1:]
Path(args[args.index('--starts') + 1]).open('a').write(' '.join(args) + '\n')
if '--draft' in args and '--die-on-draft' in args:
    sys.stderr.write('strata generate: --draft dflash2 drafts without the MTP layer and needs --draft-model PACK, --spec 2 or more and --mtp DIR (for DIR/draft_vocab.bin)\n')
    sys.stderr.flush()
    raise SystemExit(2)
if '--die-short' in args and ('--draft' in args or '--mtp' in args):
    sys.stderr.write('ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected'+chr(10))
    sys.stderr.write('strata generate: layer split, CUDA1 expert cache: no room'+chr(10))
    sys.stderr.flush()
    raise SystemExit(1)
if '--die-always' in args:
    sys.stderr.write('strata generate: something else is wrong\n')
    sys.stderr.flush()
    raise SystemExit(2)
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
'''


class AutoRetriesWithTheMtp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        py = self.root / "native.py"
        if os.name == "nt":
            py.write_text(FAKE)
            self.script = self.root / "native.cmd"
            self.script.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            self.script = py
            py.write_text("#!" + sys.executable + "\n" + FAKE)
            py.chmod(0o700)
        self.starts = self.root / "starts.txt"
        self.log = str(self.root / "engine.log")
        self.first = ["--pack", "/d/packs/g", "--starts", str(self.starts), "--die-on-draft", "--draft", "dflash2", "--draft-model", "/p", "--draft-fallback", "none",
                      "--mtp-draft-vocab", "/d/mtp/rt/draft_vocab.bin"]
        self.mtp = ["--pack", "/d/packs/g", "--starts", str(self.starts), "--die-on-draft", "--mtp", "/d/mtp/rt", "--spec", "4"]
        drafting.take_note()
        self.addCleanup(drafting.take_note)

    def started(self):
        return self.starts.read_text().splitlines() if self.starts.exists() else []

    def test_auto_that_picked_dflash2_starts_again_with_the_mtp_and_says_so_quietly(self):
        engine = StrataEngine(str(self.script), self.first, log=self.log, retry_args=self.mtp)
        self.addCleanup(engine.close)
        self.assertTrue(engine.alive())
        starts = self.started()
        self.assertEqual(len(starts), 2, "one start with DFlash2, one more with the MTP, never a third")
        self.assertIn("--draft dflash2", starts[0])
        self.assertNotIn("--draft", starts[1])
        self.assertEqual(engine.spawn[1], self.mtp, "a later restart of this engine uses the arguments that worked")
        self.assertEqual(engine.info["drafting"], "mtp")
        self.assertEqual(engine.info["drafting_note"], "Drafting: MTP (DFlash2 didn't start)", "the one quiet line the Models screen shows")
        text = Path(self.log).read_text(encoding="utf-8")
        self.assertIn("starting it again with the MTP", text)
        self.assertIn("needs --draft-model PACK", text, "the engine's own reason stays in the log")

    def test_a_start_with_no_retry_arguments_fails_with_the_engines_reason(self):
        """Named by the person (or nothing to retry with): a real error that says why."""
        with self.assertRaises(EngineDied) as caught:
            StrataEngine(str(self.script), self.first, log=self.log)
        self.assertIn("exited before it was ready", str(caught.exception))
        self.assertIn("needs --draft-model PACK", str(caught.exception))
        self.assertEqual(len(self.started()), 1)

    def test_the_retry_is_one_start_only(self):
        both = [*self.first, "--die-always"]
        retry = [*self.mtp, "--die-always"]
        with self.assertRaises(EngineDied) as caught:
            StrataEngine(str(self.script), both, log=self.log, retry_args=retry)
        self.assertIn("something else is wrong", str(caught.exception), "the second start's reason, not the first one's")
        self.assertEqual(len(self.started()), 2)

    def test_a_start_that_works_is_not_started_twice(self):
        ok = ["--pack", "/d/packs/g", "--starts", str(self.starts)]
        engine = StrataEngine(str(self.script), ok, log=self.log, retry_args=self.mtp)
        self.addCleanup(engine.close)
        self.assertEqual(len(self.started()), 1)
        self.assertEqual(engine.info.get("drafting_note", ""), "")

    def test_a_start_that_ran_out_of_memory_is_repeated_with_no_drafter_never_with_the_mtp(self):
        """2 small cards on a layer split: the engine says a card has no room.  The old retry swapped DFlash2 for the MTP, which takes MORE memory from a card that is short."""
        first = [a for a in self.first if a != "--die-on-draft"] + ["--die-short"]
        mtp = [a for a in self.mtp if a != "--die-on-draft"] + ["--die-short"]
        nodraft = ["--pack", "/d/packs/g", "--starts", str(self.starts), "--die-short"]
        engine = StrataEngine(str(self.script), first, log=self.log, retry_args=mtp, retry_vram_args=nodraft)
        self.addCleanup(engine.close)
        self.assertTrue(engine.alive())
        starts = self.started()
        self.assertEqual(len(starts), 2, "one start with DFlash2, one without a drafter, never the MTP in between")
        self.assertIn("--draft dflash2", starts[0])
        self.assertNotIn("--mtp", starts[1])
        self.assertNotIn("--draft", starts[1])
        self.assertEqual(engine.spawn[1], nodraft)
        self.assertEqual(engine.info["drafting"], "off")
        self.assertEqual(engine.info["drafting_note"], "Drafting: off (the cards were short of memory)")
        text = Path(self.log).read_text(encoding="utf-8")
        self.assertIn("short of memory", text)
        self.assertIn("expert cache: no room", text, "the engine's own reason stays in the log")

    def test_a_memory_stop_without_a_no_drafter_retry_fails_with_the_engines_reason_and_does_not_try_the_mtp(self):
        first = [a for a in self.first if a != "--die-on-draft"] + ["--die-short"]
        with self.assertRaises(EngineDied) as caught:
            StrataEngine(str(self.script), first, log=self.log, retry_args=[*self.mtp, "--die-short"])
        self.assertIn("expert cache: no room", str(caught.exception))
        self.assertEqual(len(self.started()), 1)

    def test_the_same_arguments_are_not_a_retry(self):
        with self.assertRaises(EngineDied):
            StrataEngine(str(self.script), self.first, log=self.log, retry_args=list(self.first))
        self.assertEqual(len(self.started()), 1)


class SyclVisionDrafting(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        for obj, name, value in [
            (drafting, "detect_hardware", ("sycl", ["arc_b"])),
            (drafting, "vocab_path", Path("/unused/draft_vocab.bin")),
            (drafting, "drafter_ready", True),
            (drafting, "pack_ready", True),
            (drafting, "pack_path", Path("/unused/dflash2.pack")),
            (drafting, "tuned_applies", False),
            (server, "drafting_card_class", "other"),
            (server, "drafting_vram_mib", None),
        ]:
            stack.enter_context(mock.patch.object(obj, name, return_value=value))
        stack.enter_context(redirect_stdout(StringIO()))
        drafting.take_note()
        self.addCleanup(drafting.take_note)

    def config(self, mode, vision):
        cfg = dict(model_name="qwen3.8-flash-next-iq3_s", drafting=mode,
                   args=["--native", "/unused/pack", "--mtp", "/unused/mtp", "--spec", "4", "--prefill", "8192"])
        if vision == "config":
            cfg["vision"] = {}  # Even an empty section enables --vision in engine_args.
        elif vision == "argument":
            cfg["args"].append("--vision")
        return cfg

    def test_auto_selects_an_available_drafter_before_adding_vision(self):
        for vision in ("config", "argument"):
            for mtp_here in (True, False):
                with self.subTest(vision=vision, mtp_here=mtp_here), mock.patch.object(drafting, "mtp_ready", return_value=mtp_here):
                    args = server.engine_args(self.config("auto", vision), announce=False)
                    self.assertIn("--vision", args)
                    self.assertNotIn("--draft", args)
                    self.assertEqual("--mtp" in args, mtp_here)

    def test_named_dflash2_refuses_images_during_argument_selection(self):
        for vision in ("config", "argument"):
            with self.subTest(vision=vision):
                with self.assertRaisesRegex(ValueError, "text-only"):
                    server.engine_args(self.config("dflash2", vision), announce=False)

    def test_named_images_fail_before_the_vision_reader_or_model_starts(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for name, value in [("vocab.json", {"a": 0, "b": 1, "ab": 2, "<|im_end|>": 3}),
                            ("token_type.json", [1, 1, 1, 3]),
                            ("tokenizer.json", {"pre": "qwen35"})]:
            (root / name).write_text(json.dumps(value))
        (root / "merges.txt").write_text("a b")
        cfg = self.config("dflash2", "config")
        cfg.update(exe="/unused/strata", tokenizer=str(root))
        cfg["vision"] = {"exe": "/unused/vision", "gpu": True}
        path = root / "config.json"
        path.write_text(json.dumps(cfg))
        with ExitStack() as stack:
            stack.enter_context(mock.patch("strata_control.private_files.adopt_user_defaults"))
            stack.enter_context(mock.patch.object(server.lifeline, "install_stop_handlers"))
            for name in ("apply_saved_tuning", "with_component_engine", "apply_vision_defaults"):
                stack.enter_context(mock.patch.object(server, name, side_effect=lambda value: value))
            stack.enter_context(mock.patch.object(server, "hub_from_config", return_value=None))
            stack.enter_context(mock.patch.object(server, "card_roles_lines", return_value=[]))
            stack.enter_context(mock.patch.object(server, "Server"))
            vision = stack.enter_context(mock.patch.object(server, "Vision"))
            engine = stack.enter_context(mock.patch.object(server, "StrataEngine"))
            with self.assertRaisesRegex(SystemExit, "SYCL engine is text-only"):
                server._main(["--engine", "strata", "--config", str(path), "--port", "0", "--offline"])
            vision.assert_not_called()
            engine.assert_not_called()

    def test_explicit_mtp_off_and_legacy_configs_keep_their_selection(self):
        for vision in ("config", "argument"):
            for mode in ("mtp", "off", None):
                with self.subTest(vision=vision, mode=mode), mock.patch.object(drafting, "mtp_ready", return_value=True):
                    cfg = self.config(mode, vision)
                    if mode is None:
                        del cfg["drafting"]
                    args = server.engine_args(cfg, announce=False)
                    self.assertIn("--vision", args)
                    self.assertNotIn("--draft", args)
                    self.assertEqual("--mtp" in args, mode != "off")

    def test_text_auto_keeps_dflash2(self):
        with mock.patch.object(drafting, "mtp_ready", return_value=True):
            args = server.engine_args(self.config("auto", None), announce=False)
        self.assertEqual(args[args.index("--draft") + 1], "dflash2")
        self.assertNotIn("--mtp", args)

    def test_transparent_measurement_launcher_keeps_the_named_drafter(self):
        launcher = server.ROOT / "tools/sycl/engine_once.py"
        self.assertIsNone(drafting.engine_form(launcher))
        with mock.patch.object(drafting, "mtp_ready", return_value=True):
            args = server.engine_args(self.config("dflash2", None), exe=str(launcher), announce=False)
        self.assertEqual(args[args.index("--draft") + 1], "dflash2")
        self.assertIn("--mtp-draft-vocab", args)
        self.assertNotIn("--mtp", args)

    def test_inspectable_native_without_drafter_still_uses_mtp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        native = root / "engine"
        native.write_bytes(b"\x7fELFno drafter flags in this native fixture")
        self.assertEqual(drafting.engine_form(native), drafting.FORM_NONE)
        with mock.patch.object(drafting, "mtp_ready", return_value=True):
            args = server.engine_args(self.config("auto", None), exe=str(native), announce=False)
        self.assertNotIn("--draft", args)
        self.assertIn("--mtp", args)

    def test_supported_family_keeps_dflash2_with_the_existing_display_alias(self):
        cfg = self.config("dflash2", None)
        cfg.update(model_name="qwen3.8-flash-next-iq3_s-primary", family="qwen")
        args = server.engine_args(cfg, exe=str(server.ROOT / "tools/sycl/engine_once.py"), announce=False)
        self.assertEqual(args[args.index("--draft") + 1], "dflash2")
        self.assertNotIn("--mtp", args)

    def test_unsupported_family_does_not_borrow_a_supported_display_name(self):
        cfg = self.config("dflash2", None)
        cfg["family"] = "coder"
        args = server.engine_args(cfg, announce=False)
        self.assertNotIn("--draft", args)
        self.assertIn("--mtp", args)


if __name__ == "__main__":
    unittest.main()
