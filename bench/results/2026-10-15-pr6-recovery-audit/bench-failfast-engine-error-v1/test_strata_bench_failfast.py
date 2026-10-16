"""Host-only fail-fast integration cases. Proposal: NOT_RUN."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import strata_bench as b
from test_benchmark_evidence import raw


class FailFastTests(unittest.TestCase):
    def run_main(self, directory, replies, extra=()):
        machine = {"model": "fixture", "context": 262144, "quant": "q"}
        with patch.object(b, "is_strata", return_value=True), \
             patch.object(b, "collect_machine", return_value=machine), \
             patch.object(b.Suite, "agent_files", return_value=[]), \
             patch.object(b, "chat", side_effect=replies) as chat:
            status = b.main(["--url", "http://unused", "--out", directory,
                             "--fail-fast-engine-error", "--quiet", *extra])
        result_dir = next(Path(directory).iterdir())
        result = json.loads((result_dir / "results.json").read_text())
        rows = [json.loads(line) for line in (result_dir / "requests.jsonl").read_text().splitlines()]
        return status, result_dir, result, rows, chat.call_count

    def check_failure(self, returned, results, failed_phase):
        status, directory, result, rows, count = returned
        self.assertEqual(status, 1)
        self.assertEqual(count, results)
        self.assertEqual(len([r for r in rows if r["kind"] == "result"]), results)
        self.assertEqual(len([r for r in rows if r["kind"] == "start"]), results)
        self.assertFalse(any(r["kind"] == "interrupted" for r in rows))
        proof = b.verify_request_journal(directory)
        self.assertTrue(proof["valid"])
        self.assertEqual(proof["pending"], [])
        state = json.loads((directory / "journal-state.json").read_text())
        self.assertEqual(state["terminal"], "failed")
        self.assertFalse(state["complete"])
        self.assertFalse(state["qualified"])
        self.assertEqual(result["terminal"], "failed")
        self.assertFalse(result["complete"])
        self.assertFalse(result["qualified"])
        self.assertEqual(result["terminal_failure"]["phase"], failed_phase)
        self.assertEqual(len(result["results"]["request_samples"]), results)
        self.assertEqual(result["comparison_identity"]["model"], "fixture")
        self.assertIn("calibration", result["comparison_identity"])
        self.assertTrue(result["params"]["fail_fast_engine_error"])
        self.assertIn("--fail-fast-engine-error", result["params"]["command"])

    def test_calibration_failure_stops_before_second_calibration(self):
        failed = raw(); failed["err"] = "engine exited"
        with tempfile.TemporaryDirectory() as td:
            returned = self.run_main(td, [raw(), failed])
            self.check_failure(returned, 2, "calibration")
            self.assertEqual(returned[2]["results"]["request_samples"][0]["sample"]["generated_tokens"], 1500)

    def test_third_32k_failure_preserves_two_prior_samples(self):
        failed = raw(); failed["err"] = "watchdog native exit"
        def scenarios(suite):
            suite.calibrate()
            for run in range(3):
                suite.request("prompt/32K", run, None, [], 24)
            self.fail("follow-up phase reached")
        with tempfile.TemporaryDirectory() as td, patch.object(b.Suite, "run", scenarios):
            returned = self.run_main(td, [raw(), raw(), raw(), raw(), failed])
            self.check_failure(returned, 5, "prompt/32K")
            samples = returned[2]["results"]["request_samples"]
            self.assertEqual([r["sample"]["generated_tokens"] for r in samples[2:4]], [1500, 1500])
            self.assertEqual(samples[-1]["sample"]["err"], "watchdog native exit")

    def test_default_continues_after_returned_error(self):
        failed = raw(); failed["err"] = "engine exited"
        suite = b.Suite("unused", "", 262144, 1, False)
        with patch.object(b, "chat", side_effect=[failed, raw()]) as chat:
            suite.request("chat", 0, None, [], 24)
            suite.request("chat", 1, None, [], 24)
        self.assertEqual(chat.call_count, 2)
        self.assertNotIn("terminal_failure", suite.out)

    def test_longgen_eligibility_failure_does_not_stop(self):
        suite = b.Suite("unused", "", 262144, 1, False, log=lambda _: None,
                        fail_fast_engine_error=True)
        short = raw(); short["usage"]["completion_tokens"] = 24
        with patch.object(b, "chat", return_value=short) as chat:
            suite.run_long_generation()
        self.assertEqual(chat.call_count, 2)
        self.assertNotIn("terminal_failure", suite.out)
        self.assertTrue(suite.out["failures"])

    def test_interrupts_have_one_terminal_row_and_original_exception(self):
        for returned_interrupt in (False, True):
            for error in (KeyboardInterrupt(), SystemExit(7), ConnectionResetError("stream")):
                with self.subTest(returned=returned_interrupt, error=type(error).__name__), tempfile.TemporaryDirectory() as td:
                    journal = b.RequestJournal(Path(td), lambda x: x)
                    suite = b.Suite("unused", "", 262144, 1, False, journal=journal,
                                    fail_fast_engine_error=True)
                    response = raw(); response["_interrupted"] = error
                    options = {"return_value": response} if returned_interrupt else {"side_effect": error}
                    with patch.object(b, "chat", **options), self.assertRaises(type(error)) as raised:
                        suite.request("chat", 0, None, [], 24)
                    self.assertIs(raised.exception, error)
                    rows = [json.loads(line) for line in (Path(td)/"requests.jsonl").read_text().splitlines()]
                    self.assertEqual([r["kind"] for r in rows], ["start", "result" if returned_interrupt else "interrupted"])
                    self.assertTrue(b.verify_request_journal(td)["valid"])
                    self.assertNotIn("terminal_failure", suite.out)

    def test_failed_digest_without_error_and_guard_against_caught_failure(self):
        failed = raw(); failed["first"] = None
        suite = b.Suite("unused", "", 262144, 1, False, log=lambda _: None,
                        fail_fast_engine_error=True)
        with patch.object(b, "chat", return_value=failed) as chat:
            for _ in range(2):
                with self.assertRaises(b.EngineRequestError):
                    suite.request("chat", 0, None, [], 24)
        self.assertEqual(chat.call_count, 1)

    def test_concurrent_option_rejected_before_server_access(self):
        with patch.object(b, "is_strata") as server, self.assertRaises(SystemExit):
            b.main(["--fail-fast-engine-error", "--clients", "2"])
        server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
