# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request

from serve.logit_bias import engine_key, normalize, validate_request
from serve.server import (ByteTokenizer, MockEngine, Service, StrataEngine, clean_shared_defaults,
                          sampling_defaults_from_config, serve)
from serve.frontend import ChatTemplate


class LogitBias(unittest.TestCase):
    def test_forms_and_transport(self):
        self.assertEqual(normalize({"7": -100, "9": 2.5}, 10), normalize([[7, False], [9, 2.5]], 10))
        self.assertEqual(engine_key([[9, 2.5], [7, False]]), " logit_bias=7:-100,9:2.5")
        self.assertEqual(StrataEngine.sampling_keys({"logit_bias": {"7": -100}}), " logit_bias=7:-100")
        self.assertEqual(StrataEngine.projection_key({"logit_bias": {"7": -100}}), " logit_bias=7:-100")
        self.assertNotEqual(StrataEngine.projection_key({"logit_bias": {"7": -100}}),
                            StrataEngine.projection_key({"logit_bias": {"8": -100}}))
        for value in [None, {}, []]:
            self.assertEqual(engine_key(value), "")

    def test_transport_preserves_ban_boundary(self):
        for value in [-99.9999999, -99.99999, -0.000000123456789, 99.9999999]:
            with self.subTest(value=value):
                native = engine_key({"1": value}).split("=", 1)[1]
                self.assertEqual(float(native.split(":", 1)[1]), value)
                self.assertNotEqual(native, "1:-100")
        self.assertEqual(engine_key({"1": -100}), " logit_bias=1:-100")

    def test_effective_defaults_and_explicit_clear(self):
        service = SimpleNamespace(
            sampling_defaults={"logit_bias": {"1": -100}}, shared={},
            engine=SimpleNamespace(info={}), tok=SimpleNamespace(tokens=list(range(10))))
        with self.assertRaises(ValueError):
            Service.validate_logit_bias(service, {})
        with self.assertRaises(ValueError):
            Service.validate_logit_bias(service, {"logit_bias": None})
        self.assertEqual(Service.validate_logit_bias(service, {"logit_bias": {}}), {})
        service.engine.info = {"logit_bias": 1}
        self.assertEqual(Service.validate_logit_bias(service, {}), {1: -100})
        service.shared = {"logit_bias": {"2": 3}}
        self.assertEqual(Service.validate_logit_bias(service, {}), {2: 3})
        self.assertEqual(Service.validate_logit_bias(service, {"logit_bias": {"3": 4}}), {3: 4})
        with self.assertRaises(ValueError):
            Service.validate_logit_bias(service, {"logit_bias": {"10": 1}})
        service.engine.batch = 2
        with self.assertRaises(ValueError):
            Service.validate_logit_bias(service, {})
        self.assertEqual(Service.validate_logit_bias(service, {"logit_bias": []}), {})

    def test_invalid(self):
        bad = [
            True,
            1,
            "x",
            [1],
            [[1]],
            [[1, 0, 2]],
            {"-1": 0},
            {"1.0": 0},
            {"10": 0},
            {"1": False},
            [[1, True]],
            [[1.0, 0]],
            [[True, 0]],
            [[1, 0], ["1", 2]],
            {"1": 101},
            {"1": -101},
            {"1": float("nan")},
            {"1": float("inf")},
            {"1": 10**1000},
        ]
        for value in bad:
            with self.subTest(value=str(value)[:50]), self.assertRaises(ValueError):
                normalize(value, 10)
        with self.assertRaises(ValueError):
            normalize({str(i): -100 for i in range(10)}, 10)

    def test_large_list(self):
        entries = [[i, False] for i in range(103215)]
        self.assertEqual(len(normalize(entries, 248320)), 103215)
        self.assertEqual(engine_key(entries).count(":-100"), 103215)

    def test_real_config_and_shared_default_helpers(self):
        self.assertEqual(sampling_defaults_from_config({"sampling": {"logit_bias": [[1, False]]}}),
                         {"logit_bias": {1: -100}})
        self.assertEqual(clean_shared_defaults({"logit_bias": {"2": 3}}), {"logit_bias": {2: 3}})
        self.assertEqual(clean_shared_defaults({"logit_bias": {}}), {"logit_bias": {}})
        with self.assertRaises(SystemExit):
            sampling_defaults_from_config({"sampling": {"logit_bias": [[True, 1]]}})
        with self.assertRaises(ValueError):
            clean_shared_defaults({"logit_bias": [[1, True]]})

    def test_capability_and_batch(self):
        for engine in [SimpleNamespace(info={}), SimpleNamespace(info={"logit_bias": 1}, batch=2)]:
            with self.assertRaises(ValueError):
                validate_request({"logit_bias": {"1": -100}}, engine, 10)
            self.assertEqual(validate_request({}, engine, 10), {})
        self.assertEqual(
            validate_request({"logit_bias": {"1": -100}}, SimpleNamespace(info={"logit_bias": 1}), 10), {1: -100}
        )

    def test_http_rejects_before_stream(self):
        class CountEngine(MockEngine):
            def generate(self, *args, **kwargs):
                self.calls += 1
                yield from super().generate(*args, **kwargs)

        tok = ByteTokenizer()
        engine = CountEngine(tok, "ok", max_context=4096)
        engine.calls = 0
        service = Service(engine, tok, ChatTemplate(Path(__file__).parent / "test_chat_template.jinja"))
        service.start_telemetry = lambda: None
        httpd = serve(service, port=0)
        try:
            for bias in [True, {"1": -100}, [[1, True]]]:
                data = json.dumps(
                    {
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 2,
                        "stream": True,
                        "logit_bias": bias,
                    }
                ).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(req, timeout=10)
                with caught.exception as response:
                    self.assertEqual(response.code, 400)
                    self.assertIn("logit_bias", json.loads(response.read())["error"]["message"])
            self.assertEqual(engine.calls, 0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_responses_rejects_before_stream_and_without_generation(self):
        class CountEngine(MockEngine):
            def generate(self, *args, **kwargs):
                self.calls += 1
                yield from super().generate(*args, **kwargs)

        tok = ByteTokenizer()
        engine = CountEngine(tok, "ok", max_context=4096)
        engine.calls = 0
        service = Service(engine, tok, ChatTemplate(Path(__file__).with_name("test_chat_template.jinja")))
        service.start_telemetry = lambda: None
        httpd = serve(service, port=0)
        try:
            for bias in [True, {"1": -100}, [[1, True]]]:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{httpd.server_address[1]}/v1/responses",
                    data=json.dumps({"input": "hi", "max_output_tokens": 2,
                                     "stream": True, "logit_bias": bias}).encode(),
                    headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(req, timeout=10)
                with caught.exception as response:
                    self.assertEqual(response.code, 400)
                    error = json.loads(response.read())["error"]
                    self.assertEqual(error["param"], "logit_bias")
            self.assertEqual(engine.calls, 0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_anthropic_unsupported_defaults_reject_before_stream(self):
        class CountEngine(MockEngine):
            def generate(self, *args, **kwargs):
                self.calls += 1
                yield from super().generate(*args, **kwargs)

        tok = ByteTokenizer()
        engine = CountEngine(tok, "ok", max_context=4096)
        engine.calls = 0
        service = Service(engine, tok, ChatTemplate(Path(__file__).with_name("test_chat_template.jinja")))
        service.start_telemetry = lambda: None
        httpd = serve(service, port=0)
        try:
            variants = [({"logit_bias": {"1": -100}}, {}, {}),
                        ({}, {"logit_bias": {"2": 3}}, {}),
                        ({}, {}, {"logit_bias": {"3": -2}})]
            for defaults, shared, fields in variants:
                service.sampling_defaults, service.shared = defaults, shared
                for stream in [False, True]:
                    with self.subTest(defaults=defaults, shared=shared, fields=fields, stream=stream):
                        req = urllib.request.Request(
                            f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages",
                            data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                             "max_tokens": 2, "stream": stream, **fields}).encode(),
                            headers={"Content-Type": "application/json"})
                        try:
                            response = urllib.request.urlopen(req, timeout=10)
                        except urllib.error.HTTPError as error:
                            response = error
                        with response:
                            body = response.read()
                            self.assertEqual(response.code, 400, body)
                            self.assertIn("application/json", response.headers["Content-Type"])
                            self.assertNotIn(b"message_start", body)
                            self.assertIn("logit_bias", json.loads(body)["error"]["message"])
            self.assertEqual(engine.calls, 0)
            for fields in [{"logit_bias": {}}, {"logit_bias": []}, {}]:
                if not fields:
                    service.sampling_defaults, service.shared = {}, {}
                req = urllib.request.Request(
                    f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages",
                    data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                     "max_tokens": 2, "stream": True, **fields}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    self.assertEqual(response.code, 200)
                    self.assertIn(b"event: message_start", response.read())
            self.assertEqual(engine.calls, 3)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_supported_http_request_transport_changes_and_clear(self):
        class CaptureEngine(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.keys.append(StrataEngine.sampling_keys(sampling))
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        engine = CaptureEngine(tok, "ok", max_context=4096)
        engine.info, engine.keys = {"logit_bias": 1}, []
        service = Service(engine, tok, ChatTemplate(Path(__file__).with_name("test_chat_template.jinja")))
        service.start_telemetry = lambda: None
        httpd = serve(service, port=0)
        try:
            # A supported CPU fake checks API transport, not GPU numerics or model quality.
            for fields in [{"logit_bias": {"1": -100}}, {}, {"logit_bias": {"2": 2.5}},
                           {"logit_bias": {}}, {"logit_bias": [[3, False]]}]:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                    data=json.dumps({"messages": [{"role": "user", "content": "hi"}],
                                     "max_tokens": 2, **fields}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    self.assertEqual(response.status, 200)
                    json.loads(response.read())
            self.assertEqual(len(engine.keys), 5)
            self.assertEqual([key.split(" logit_bias=", 1)[-1] if " logit_bias=" in key else ""
                              for key in engine.keys], ["1:-100", "", "2:2.5", "", "3:-100"])
        finally:
            httpd.shutdown()
            httpd.server_close()
