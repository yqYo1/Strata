import unittest

from dflash2_width_admission import audit_head, audit_requests, audit_windows


class WidthAdmissionTests(unittest.TestCase):
    def test_terminal_prefix_at_each_context_clamp(self):
        # Geometry only. Native GEN admission reserves room for a reply.
        for width in (4, 5):
            for rows in range(1, width + 1):
                for accepted in range(rows):
                    for emitted in range(2, accepted + 3):
                        for finish in ("length", "stop"):
                            with self.subTest(width=width, rows=rows, accepted=accepted,
                                              emitted=emitted, finish=finish):
                                windows = [(81, 1), (82, rows)]
                                response = dict(prompt=82, generated=emitted, finish=finish,
                                                offered=rows - 1, accepted=accepted)
                                result = audit_windows(windows, response, prompt=82, width=width,
                                                       context=82 + rows,
                                                       output_limit=emitted + (finish == "stop"))
                                self.assertEqual(result["commit_end"], 83 + accepted)
                                self.assertEqual(result["committed_after_last_output"],
                                                 accepted + 2 - emitted)

    def test_trace_reconciles_early_commit_and_truncated_last_window(self):
        text = ("strata trace: request 82 0\n"
                "strata trace: window 81 1\n"
                "strata trace: request 82 0\n"
                "strata trace: window 81 1\n"
                "strata trace: window 82 5\n"
                "strata trace: window 84 5\n")
        requests = [([0] * 82, 1), ([0] * 82, 4)]
        responses = [dict(prompt=82, generated=1, finish="length", offered=0, accepted=0),
                     dict(prompt=82, generated=4, finish="length", offered=8, accepted=4)]
        first, second = audit_requests(text, requests, responses, width=5)
        self.assertEqual(first["width_counts"], {1: 1})
        self.assertEqual(second["accepted_prefix_counts"], {0: 1, 1: 1, 3: 1})
        self.assertEqual(second["committed_after_last_output"], 3)

        with self.assertRaisesRegex(ValueError, "unexpected DFlash2 window"):
            audit_requests(text, requests, responses, width=4)
        with self.assertRaisesRegex(ValueError, "unowned native window"):
            audit_requests(text.split("\n", 1)[1], requests, responses, width=5)

    def test_rejects_inconsistent_draft_or_output_extent(self):
        windows = [(81, 1), (82, 5), (84, 5)]
        reference = dict(prompt=82, generated=4, finish="length", offered=8, accepted=4)
        for changes, error in (
            (dict(offered=7), "draft counters"),
            (dict(accepted=0), "accepted prefix"),
            (dict(accepted=6), "accepted prefix"),
            (dict(generated=3), "output extent"),
            (dict(generated=5), "output extent"),
            (dict(finish="cancel"), "output extent"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, error):
                audit_windows(windows, dict(reference, **changes), prompt=82, output_limit=4, width=5)


    def test_head_route_joins_the_actual_replays(self):
        for width in (4, 5):
            lines = []
            counts = {1: 1, width: 9}
            for capture, (rows, replays) in enumerate(counts.items(), start=1):
                identity = f"request=1 device=0 phase=decode tokens={rows} capture_id={capture}"
                lines.append(f"strata Q6_ROUTE {identity} scope=live_capture_cumulative native_half_enabled=0 "
                             f"complete=1 shapes=7 cache_hits={replays - 1} "
                             f"replay_started={replays} replay_completed={replays}")
                lines.append(f"strata Q6_DPAS_HEAD {identity} layout=canonical extra_weight_bytes=0 "
                             f"canonical_table_excludes_dpas=1 calls={int(rows == 4)} complete=1")
                if rows != 4:
                    lines.append(f"strata Q6_SHAPE {identity} ni=2560 no=248320 nc={rows} "
                                 f"reference_calls={int(rows == 5)} columns_calls={int(rows == 1)} "
                                 "centered_calls=0 native_half_calls=0")
            text = "\n".join(lines)
            with self.subTest(width=width):
                expected = "dpas" if width == 4 else "q6-reference"
                self.assertEqual(audit_head(text, counts)[-1]["route"], expected)
                with self.assertRaisesRegex(ValueError, "capture route or replay"):
                    audit_head(text, {1: 1, width: 8})
                cap_head = f"strata Q6_DPAS_HEAD request=1 device=0 phase=decode tokens={width} "
                with self.assertRaisesRegex(ValueError, "must cover every observed width"):
                    audit_head(text.replace(cap_head, "ignored "), counts)
                with self.assertRaisesRegex(ValueError, "head fallback differs"):
                    audit_head(text.replace("reference_calls=0 columns_calls=1",
                                            "reference_calls=1 columns_calls=0"), counts)
                with self.assertRaisesRegex(ValueError, "sole-packed"):
                    audit_head(text + "\nstrata Q6_PACKED_HEAD unexpected", counts)


if __name__ == "__main__":
    unittest.main()
