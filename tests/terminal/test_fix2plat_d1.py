"""D1 (0.2.16 retest, LIN): offline, the DFlash2 drafter only turned on once BOTH q4_0.pack and q4_0-selq4.pack were copied in.

The smaller prepared file (selq4) is a complete drafter for an engine that reads it, so it alone is enough now; and when nothing is there,
the message names the exact file(s) and sizes to copy in.  Synthetic small files with matching pins (the real ones are 522 and 340 MB).

    python -m unittest tests.terminal.test_fix2plat_d1
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import drafting, drafting_fetch


class OnlyTheSmallFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "data"
        (self.data / drafting.FOLDER).mkdir(parents=True)
        blob = bytes(range(256)) * 6000                                   # 1.5 MB: more than the 1 MB a file must have to count as here
        self.small = drafting.converted_path(self.data)
        self.small.write_bytes(blob)
        pins = mock.patch.multiple(drafting, CONVERTED_BYTES=len(blob), CONVERTED_SHA256=hashlib.sha256(blob).hexdigest())
        pins.start()
        self.addCleanup(pins.stop)
        self.stock = drafting.target_path(self.data)

    def reads(self, yes: bool):
        patch = mock.patch.object(drafting, "engine_has_selector_q4", return_value=yes)
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_small_file_alone_is_a_ready_drafter_for_an_engine_that_reads_it(self):
        self.reads(True)
        self.assertFalse(self.stock.exists())
        self.assertTrue(drafting.pack_ready(self.data))
        self.assertEqual(drafting.pack_path(self.data), self.small)
        self.assertTrue(drafting.drafter_ready(self.data, "x", "cuda"))

    def test_an_engine_that_cannot_read_it_still_needs_the_stock_file(self):
        self.reads(False)
        self.assertFalse(drafting.pack_ready(self.data))
        self.assertEqual(drafting.pack_path(self.data), self.stock)

    def test_a_damaged_small_file_is_not_a_drafter(self):
        self.reads(True)
        self.small.write_bytes(b"x" * 1_600_000)
        self.assertFalse(drafting.pack_ready(self.data))

    def test_plan_and_launch_use_the_explicit_engine_when_only_the_small_pack_exists(self):
        self.reads(True)
        (self.small.parent / drafting.VOCAB_NAME).write_bytes(b"\1\0\0\0" * 8)
        for capable in (False, True):
            with self.subTest(capable=capable):
                engine = self.data / f"engine-{capable}"
                engine.write_bytes(drafting._MARK_FILE + b"\0" + (drafting._MARK_SELQ4 if capable else b""))
                plan = drafting_fetch.plan(self.data, engine=engine)
                self.assertEqual(plan["ready"], capable)
                self.assertEqual(Path(plan["pack"]), self.small if capable else self.stock)
                cfg = {"model_name": "qwen3.8-flash-next-iq3_s", "drafting": "dflash2", "args": []}
                decision = drafting.apply(cfg, data_dir=self.data, engine=engine, backend="sycl")
                self.assertEqual(decision.drafter == "dflash2", capable)
                if capable:
                    self.assertEqual(cfg["args"][cfg["args"].index("--draft-model") + 1], str(self.small))
                else:
                    self.assertNotIn("--draft-model", cfg["args"])

    def test_the_stock_file_alone_is_still_a_drafter_and_wins_without_an_engine(self):
        self.reads(True)
        self.small.unlink()
        self.stock.write_bytes(b"\0" * 1_600_000)
        self.assertTrue(drafting.pack_ready(self.data))
        self.assertEqual(drafting.pack_path(self.data), self.stock)

    def test_fetch_offline_with_only_the_small_file_is_ready_not_an_error(self):
        self.reads(True)
        with mock.patch.object(drafting_fetch, "online", return_value=False):
            got = drafting_fetch.ensure(self.data)
        self.assertEqual(got, self.small)


class NamesTheFiles(unittest.TestCase):
    def test_fetch_reports_the_same_offline_file_in_text_and_json(self):
        from tests.terminal.launch_support import run_cli

        with tempfile.TemporaryDirectory() as folder, mock.patch("strata_cli.drafting_cmd._hardware", return_value=("sycl", [])), \
                mock.patch.object(drafting, "engine_has_selector_q4", return_value=False), \
                mock.patch.object(drafting_fetch, "_present", return_value={}), mock.patch.object(drafting_fetch, "online", return_value=False):
            args = ["drafting", "fetch", "--family", "gyro", "--mode", "dflash2", "--data-dir", folder]
            code, text, _ = run_cli([*args, "--output", "json"])
            self.assertEqual(code, 5)
            error = json.loads(text)["error"]
            self.assertEqual(error["code"], "drafting_failed")
            self.assertIn(drafting.PACK_NAME, error["help"])
            self.assertNotIn(drafting.CONVERTED_NAME, error["help"])
            code, _, stderr = run_cli(args)
            self.assertEqual(code, 5)
            self.assertIn(error["help"], stderr)

    def test_nothing_there_offline_names_both_files_with_sizes(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(drafting_fetch, "online", return_value=False), \
                mock.patch.object(drafting, "engine_has_selector_q4", return_value=True):
            with self.assertRaises(drafting_fetch.OfflineError) as caught:
                drafting_fetch.ensure(d)
        self.assertEqual(str(caught.exception), "There is no internet connection, so the DFlash2 drafter can't be downloaded now.")   # the app's own text, unchanged
        note = caught.exception.files_note
        self.assertIn("qwen3.8-flash-next-dflash2.q4_0.pack (522 MB)", note)
        self.assertIn("qwen3.8-flash-next-dflash2.q4_0-selq4.pack (340 MB)", note)
        self.assertIn("one of these files", note)
        self.assertIn(str(Path(d) / "dflash2"), note)
        self.assertNotIn("sha", note.lower())

    def test_an_engine_that_cannot_read_the_small_file_is_not_told_to_copy_it(self):
        note = drafting_fetch.offline_files_note("/data", engine_reads_q4=False)
        self.assertIn("this file: qwen3.8-flash-next-dflash2.q4_0.pack (522 MB)", note)
        self.assertNotIn("selq4", note)

    def test_a_set_with_a_tuned_drafter_names_its_own_file(self):
        note = drafting_fetch.offline_files_note("/data", tuned=True, variant="selq4")
        self.assertIn("qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack (340 MB)", note)
        note = drafting_fetch.offline_files_note("/data", tuned=True, variant="bf16sel")
        self.assertIn("qwen3.8-flash-next-dflash2.strata1.q4_0-bf16sel.pack (522 MB)", note)


if __name__ == "__main__":
    unittest.main()
