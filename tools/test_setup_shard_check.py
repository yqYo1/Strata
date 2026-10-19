"""Niko e88ee39d shard checks, with copied-file and excess-data regressions.

GGUF fixture writer and original two checks come from the upstream MIT source.
No GPU, model download or persistent fixture files are needed.
"""
import contextlib
import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gguf(path, tensors=(), pad_data=0, alignment=32):
    """A GGUF v3 file with one string key and the given F32 tensors."""
    kv = _str("general.name") + struct.pack("<I", 8) + _str("t")
    if alignment != 32:
        kv += _str("general.alignment") + struct.pack("<II", 4, alignment)
    tinfo = b""
    off = 0
    for name, n in tensors:
        tinfo += _str(name) + struct.pack("<I", 1) + struct.pack("<Q", n) + struct.pack("<I", 0) + struct.pack("<Q", off)
        off += ((n * 4 + alignment - 1) // alignment) * alignment
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1 + int(alignment != 32)) + kv + tinfo
    body = head
    if tensors:
        body += b"\0" * ((alignment - len(head) % alignment) % alignment)
        body += b"\0" * pad_data
    path.write_bytes(body)
    return len(head)


class ShardCheck(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="strata-shard-test-")
        self.addCleanup(temporary.cleanup)
        self.d = Path(temporary.name)

    def assert_refused(self, path):
        self.assertFalse(setup.whole_shard(path))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                setup.check_shards([path])

    def test_metadata_only_shard_without_padding_is_whole(self):
        p = self.d / "m-00001-of-00006.gguf"
        n = gguf(p)
        self.assertNotEqual(n % 32, 0, "exercise an unaligned metadata-only header")
        setup.check_shards([p])
        self.assertTrue(setup.whole_shard(p))

    def test_truncated_tensor_data_is_still_refused(self):
        p = self.d / "t-00002-of-00006.gguf"
        gguf(p, tensors=[("w", 64)], pad_data=100)  # needs 256 bytes
        self.assert_refused(p)

    def test_complete_tensor_and_small_padding_are_whole(self):
        p = self.d / "w.gguf"
        for padding in (0, 63):
            with self.subTest(padding=padding):
                gguf(p, tensors=[("w", 64)], pad_data=256 + padding)
                self.assertTrue(setup.whole_shard(p))
                setup.check_shards([p])

    def test_one_tensor_extra_payload_is_refused(self):
        p = self.d / "w.gguf"
        gguf(p, tensors=[("w", 64)], pad_data=256 + 64)
        self.assert_refused(p)

    def test_declared_large_alignment_padding_is_whole(self):
        p = self.d / "aligned.gguf"
        gguf(p, tensors=[("w", 1)], pad_data=128, alignment=128)
        self.assertTrue(setup.whole_shard(p))
        setup.check_shards([p])

    def test_extra_payload_exceeding_declared_alignment_is_refused(self):
        p = self.d / "aligned.gguf"
        gguf(p, tensors=[("w", 1)], pad_data=4 + 128, alignment=128)
        self.assert_refused(p)

    def test_incomplete_header_is_refused(self):
        p = self.d / "bad.gguf"
        p.write_bytes(b"GGUF")
        self.assert_refused(p)


if __name__ == "__main__":
    unittest.main()
