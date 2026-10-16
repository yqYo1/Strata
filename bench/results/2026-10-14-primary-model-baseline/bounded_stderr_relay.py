"""Drain native stderr completely, retaining a bounded prefix, API census and diagnostics."""
import collections
import hashlib
import json
import re
import sys
import threading
from pathlib import Path


class Relay:
    def __init__(self, stream, root, prefix_cap=2 << 20, selected_cap=2 << 20):
        self.stream, self.root = stream, Path(root)
        self.prefix_cap, self.selected_cap = prefix_cap, selected_cap
        self.summary = dict(passed=False, total_bytes=0, total_lines=0, prefix_bytes=0,
            selected_bytes=0, prefix_cap=prefix_cap, selected_cap=selected_cap, oversized_lines=0,
            filtered_trace=True, full_history_retained=False, API_counts={}, error_counts={})
        self.api, self.errors = collections.Counter(), collections.Counter()
        self.hash = hashlib.sha256()
        self.tail = bytearray()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.api_boundaries = 0

    def start(self):
        self.thread.start()

    def close(self):
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('stderr drain did not reach EOF')

    def line(self, data):
        self.summary['total_lines'] += 1
        text = data.decode(errors='replace')
        if '---> ur' in text:
            match = re.search(r'---> (ur\w+)', text)
            if match:
                self.api[match.group(1)] += 1
        tokens = re.findall(r'(?:UR|ZE)_RESULT_ERROR_\w+', text)
        self.errors.update(tokens)
        warning_or_failure = any(s in text for s in ['[ERROR]', '[WARNING]', ' ERROR (',
            'stack smashing', 'strata host diagnostic:', 'terminate called', 'what():']) or bool(tokens)
        boundary = any(s in text for s in ['zeModuleCreate', 'zeModuleBuildLog', 'urProgramBuild', 'urProgramCompile'])
        phase = text.startswith(('strata ', '  expert pool', '  verify window', '  memory:', '  2 s later:'))
        selected = phase or warning_or_failure or (boundary and self.api_boundaries < 1024)
        if boundary:
            self.api_boundaries += 1
        if selected and self.summary['prefix_bytes'] >= self.prefix_cap:
            if self.summary['selected_bytes'] + len(data) <= self.selected_cap:
                sys.stderr.buffer.write(data); sys.stderr.buffer.flush()
                self.summary['selected_bytes'] += len(data)
            else:
                self.summary['selected_cap_reached'] = True

    def run(self):
        pending = bytearray()
        try:
            while True:
                data = self.stream.read1(65536)
                if not data:
                    break
                self.hash.update(data); self.summary['total_bytes'] += len(data)
                self.tail.extend(data)
                if len(self.tail) > 256 << 10:
                    del self.tail[:len(self.tail)-(256 << 10)]
                keep = min(len(data), self.prefix_cap-self.summary['prefix_bytes'])
                if keep > 0:
                    sys.stderr.buffer.write(data[:keep]); sys.stderr.buffer.flush()
                    self.summary['prefix_bytes'] += keep
                pending.extend(data)
                while b'\n' in pending:
                    line, _, rest = pending.partition(b'\n')
                    pending[:] = rest
                    self.line(line + b'\n')
                if len(pending) > 64 << 10:
                    self.summary['oversized_lines'] += 1
                    self.line(bytes(pending[:64 << 10]))
                    pending.clear()
            if pending:
                self.line(bytes(pending))
            self.summary['passed'] = True
        except BaseException as error:
            self.summary['error'] = repr(error)
            # Even a retention error must not leave a blocked producer pipe.
            while self.stream.read1(65536):
                pass
        finally:
            self.stream.close()
            self.summary.update(full_stream_sha256=self.hash.hexdigest(), API_counts=dict(self.api),
                                error_counts=dict(self.errors), API_boundary_lines=self.api_boundaries)
            (self.root / 'owned/engine-stderr-tail.txt').write_bytes(self.tail)
            (self.root / 'stderr-relay-summary.json').write_text(json.dumps(self.summary, indent=2) + '\n')
