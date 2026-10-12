# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""#1527: READY failures finish lazy-load/restart requests with 503 and release startup state.

Fake CPU engine processes behind the real HTTP request path; no GPU, model or downloads.
"""
import json
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.server import ByteTokenizer, ChatTemplate, Service, StrataEngine, serve

REAL_POPEN = subprocess.Popen
SILENT = "import time\ntime.sleep(3600)\n"
HANG_ON_RESTART = r'''
import sys
import time
from pathlib import Path
hang = Path(sys.argv[sys.argv.index('--hang') + 1])
if hang.exists():
    time.sleep(3600)
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if line.startswith(('GEN', 'BGEN')):
        print('T 90\nDONE 1 1 0 1 stop', flush=True)
'''


class ReadyTimeout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patch in (mock.patch.object(server, "ENGINE_READY_S", 0.5),
                      mock.patch.object(StrataEngine, "RESTART_RETRY_S", 0.0)):
            patch.start()
            self.addCleanup(patch.stop)

    def service(self, name, code, extra_args=()):
        script = self.root / (name + ".py")
        script.write_text(code)
        # Same fake-process route as test_restart_waiters: invoke the current Python directly, no shell wrapper.
        patch = mock.patch.object(server.subprocess, "Popen",
                                  lambda cmd, **kw: REAL_POPEN([sys.executable, str(script), *cmd[1:]], **kw))
        patch.start()
        self.addCleanup(patch.stop)
        engine = StrataEngine(str(script), list(extra_args), lazy=True)
        self.addCleanup(engine.close)
        return Service(engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "test_chat_template.jinja"))

    def http(self, svc):
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(self, base):
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.read()

    def assert503(self, base):
        before = time.monotonic()
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.post(base)
        error = raised.exception
        try:
            self.assertEqual(error.code, 503)
            body = error.read().decode()
            self.assertIn("did not report READY", body)
            self.assertIn("next request restarts it", body)
        finally:
            error.close()
        self.assertLess(time.monotonic() - before, 60)

    def test_ready_timeout_on_lazy_load_answers_the_request(self):
        svc = self.service("silent", SILENT)
        self.assert503(self.http(svc))
        self.assertFalse(svc.engine.starting)
        self.assertFalse(svc.fifo.locked())
        self.assertEqual(svc.status["queued"], 0)

    def test_ready_timeout_on_restart_answers_the_request(self):
        hang = self.root / "hang"
        svc = self.service("hang_on_restart", HANG_ON_RESTART, ("--hang", str(hang)))
        base = self.http(svc)
        self.assertEqual(self.post(base)[0], 200)
        svc.engine.proc.kill()
        svc.engine.proc.wait(timeout=10)
        hang.touch()
        self.assert503(base)
        self.assertFalse(svc.engine.starting)
        self.assertFalse(svc.fifo.locked())
        self.assertEqual(svc.status["queued"], 0)


if __name__ == "__main__":
    unittest.main()
