"""Unread SSE clients release the engine turn across all three APIs; CPU/mock only.

Regression for Niko1221/Strata c08fd2ecc87327125dfb76de57f64a5377707242.
Main-authored tests; repository LICENSE. No model files, GPU or telemetry thread.
"""
import contextlib
import http.client
import json
from pathlib import Path
import socket
import threading
import time
import unittest
from unittest import mock

import serve.server as server
from serve.frontend import ChatTemplate


class ObservedEngine(server.MockEngine):
    def __init__(self, tok):
        super().__init__(tok, "</think>\n\n" + " ".join(map(str, range(20000))),
                         max_context=500000)
        self.entered = threading.Event()
        self.finished = threading.Event()
        self.was_cancelled = False

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.entered.set()
        try:
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)
        finally:
            self.was_cancelled = cancel.is_set()
            self.finished.set()


class StreamWriteTimeout(unittest.TestCase):
    @contextlib.contextmanager
    def running(self, timeout_s):
        tok = server.ByteTokenizer()
        engine = ObservedEngine(tok)
        svc = server.Service(engine, tok,
            ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        base = server.make_handler(svc)

        class SmallSendBuffer(base):
            def setup(self):
                super().setup()
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)

        with mock.patch.object(server, "STREAM_WRITE_TIMEOUT_S", timeout_s):
            httpd = server.Server(("127.0.0.1", 0), SmallSendBuffer)
            thread = threading.Thread(target=httpd.serve_forever,
                                      kwargs={"poll_interval": 0.02}, daemon=True)
            thread.start()
            try:
                yield svc, engine, httpd.server_address[1]
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(3)
                self.assertFalse(thread.is_alive(), "owned HTTP listener remains live")

    @staticmethod
    def body(path, stream, limit):
        req = dict(model="qwen3.8-flash-next", stream=stream)
        if path == "/v1/responses":
            req.update(input="go", max_output_tokens=limit)
        else:
            req.update(messages=[dict(role="user", content="go")], max_tokens=limit)
        return json.dumps(req).encode()

    def unread_client(self, port, path):
        raw = socket.socket()
        try:
            raw.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            raw.settimeout(3)
            raw.connect(("127.0.0.1", port))
            body = self.body(path, True, 200000)
            raw.sendall(f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                        + body)
        except BaseException:
            raw.close()
            raise
        return raw

    def wait_released(self, svc):
        end = time.monotonic() + 3
        while svc.fifo.locked() and time.monotonic() < end:
            time.sleep(0.01)
        self.assertFalse(svc.fifo.locked(), "the unread client still owns the engine turn")

    def next_request(self, port, path):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            conn.request("POST", path, self.body(path, False, 20),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            body = response.read()
            self.assertEqual(response.status, 200, body[:500])
            self.assertIsInstance(json.loads(body), dict)
        finally:
            conn.close()

    def check_api(self, path):
        with self.running(0.25) as (svc, engine, port):
            raw = self.unread_client(port, path)
            try:
                self.assertTrue(engine.entered.wait(3), "stream never reached the mock engine")
                self.assertTrue(engine.finished.wait(5), "blocked write did not cancel generation")
                self.assertTrue(engine.was_cancelled, "stream ended without a disconnect")
                self.wait_released(svc)
                self.next_request(port, path)
            finally:
                raw.close()
                self.assertTrue(engine.finished.wait(3))
                self.wait_released(svc)

    def test_chat(self):
        self.check_api("/v1/chat/completions")

    def test_responses(self):
        self.check_api("/v1/responses")

    def test_messages(self):
        self.check_api("/v1/messages")

    def test_zero_keeps_turn_until_disconnect(self):
        path = "/v1/chat/completions"
        with self.running(0) as (svc, engine, port):
            raw = self.unread_client(port, path)
            try:
                self.assertTrue(engine.entered.wait(3))
                self.assertFalse(engine.finished.wait(0.6), "zero unexpectedly limits writes")
                self.assertTrue(svc.fifo.locked())
            finally:
                raw.close()
                self.assertTrue(engine.finished.wait(3), "disconnect did not stop the mock engine")
                self.wait_released(svc)
            self.next_request(port, path)


if __name__ == "__main__":
    unittest.main()
