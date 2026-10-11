# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Config thinking defaults and legacy migration: mock-only, no engine subprocess."""
import json
import unittest
import urllib.error
import urllib.request
from unittest import mock

from serve import server as S
from serve.test_server import ByteTokenizer, CTX


class ConfigEffort(unittest.TestCase):
    def svc(self, effort=None, shared=None):
        tok = ByteTokenizer()
        svc = S.Service(S.MockEngine(tok, "ok", max_context=CTX), tok,
                        S.ChatTemplate(S.ROOT / "serve/test_chat_template.jinja"))
        svc.config_effort = effort
        svc.shared = shared or {}
        return svc

    def test_precedence_and_default_identity(self):
        req = {"messages": []}
        self.assertIs(self.svc().with_shared(req, "openai"), req)
        svc = self.svc("low")
        self.assertEqual(svc.with_shared({}, "openai")["reasoning_effort"], "low")
        self.assertEqual(svc.with_shared({"reasoning_effort": "none"}, "openai")["reasoning_effort"], "none")
        self.assertEqual(
            self.svc("low", {"reasoning_effort": "high"}).with_shared({}, "openai")["reasoning_effort"], "high")
        for req in ({"reasoning": {"effort": "high"}}, {"chat_template_kwargs": {"enable_thinking": False}},
                    {"chat_template_kwargs": {"reasoning_effort": "medium"}}):
            self.assertNotIn("reasoning_effort", svc.with_shared(req, "openai"))
        self.assertEqual(svc.with_shared({}, "anthropic")["output_config"], {"effort": "low"})
        self.assertEqual(self.svc("none").with_shared({}, "anthropic")["thinking"], {"type": "disabled"})
        req = {"thinking": {"type": "disabled"}}
        self.assertEqual(svc.with_shared(req, "anthropic"), req)

    def test_migration_and_key_priority(self):
        for args in (["--ctx", "4096", "--reasoning-effort", "low"], ["--reasoning-effort=low", "--ctx", "4096"]):
            cfg = {"args": args}
            with mock.patch("builtins.print"):
                S.lift_reasoning_effort_arg(cfg)
            self.assertEqual(cfg, {"args": ["--ctx", "4096"], "reasoning_effort": "low"})
        cfg = {"args": ["--reasoning-effort", "low"], "reasoning_effort": "none"}
        with mock.patch("builtins.print"):
            S.lift_reasoning_effort_arg(cfg)
        self.assertEqual(cfg, {"args": [], "reasoning_effort": "none"})

    def test_invalid_migration_does_not_mutate_args(self):
        for args in (["--reasoning-effort"], ["--reasoning-effort", "--ctx", "4096"],
                     ["--reasoning-effort=bogus"], ["--reasoning-effort="], ["--reasoning-effort", True],
                     ["--reasoning-effort", None], ["--reasoning-effort", ""]):
            cfg = {"args": args[:]}
            with self.assertRaises(ValueError):
                S.lift_reasoning_effort_arg(cfg)
            self.assertEqual(cfg, {"args": args})
        for value in (True, 2, "bogus", {}, []):
            with self.assertRaises(ValueError):
                S.clean_shared_defaults({"reasoning_effort": value})

    def test_http_routes_receive_config_and_request_effort(self):
        svc = self.svc("low")
        seen = []

        def stop_before_generation(messages, tools, kwargs, *args, **extra):
            seen.append(kwargs)
            raise ValueError("fixture stops before engine generation")

        svc.prepare = stop_before_generation
        svc.start_telemetry = lambda: None
        httpd = S.serve(svc, port=0)
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            for path, body, override in (
                ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]},
                 {"reasoning_effort": "high"}),
                ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
                 {"output_config": {"effort": "high"}}),
                ("/v1/responses", {"input": "hi"}, {"reasoning": {"effort": "high"}}),
            ):
                for fields, expected in (({}, "low"), (override, "xhigh")):
                    req = urllib.request.Request(base + path, data=json.dumps({**body, **fields}).encode(),
                                                 headers={"Content-Type": "application/json"})
                    with self.assertRaises(urllib.error.HTTPError) as err:
                        urllib.request.urlopen(req, timeout=10)
                    self.assertEqual(err.exception.code, 400)
                    self.assertEqual(seen[-1].get("reasoning_effort"), expected)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
