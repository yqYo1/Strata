"""SOURCE ONLY independent fake byte-source fixtures; root runs serially."""
import unittest
from bounded_engine_protocol_v1 import BoundedProtocolLines, ProtocolFramingError


class FakeSource:
    """Explicit chunks rather than a real process, PTY, socket or helper I/O."""
    def __init__(self,*chunks): self.chunks=list(chunks)
    def deliver(self,framer):
        for chunk in self.chunks: framer.feed(chunk)


class ProtocolFixtures(unittest.TestCase):
    def reject(self,framer,chunk,message):
        before=framer.state();queued=framer.queued_tail();pending=framer.pending_tail()
        with self.assertRaisesRegex(ProtocolFramingError,'^'+message+'$'): framer.feed(chunk)
        after=framer.state()
        self.assertEqual((after.received_bytes,after.consumed_bytes,after.buffered_bytes),
                         (before.received_bytes,before.consumed_bytes,before.buffered_bytes))
        self.assertEqual(framer.queued_tail(),queued);self.assertEqual(framer.pending_tail(),pending)
        self.assertIsNotNone(after.error)
    def conserved(self,framer):
        s=framer.state()
        self.assertEqual(s.received_bytes-s.consumed_bytes,s.buffered_bytes)
        self.assertLessEqual(s.buffered_bytes,framer.buffer_limit)
        self.assertEqual(s.has_unconsumed,bool(s.buffered_bytes))
    def test_exact_line_bound_and_split_lf(self):
        f=BoundedProtocolLines();FakeSource(b'x'*65535,b'\n').deliver(f)
        self.assertEqual(f.pop_line(),b'x'*65535+b'\n');self.conserved(f)
        self.assertFalse(f.finish().has_unconsumed)
        f=BoundedProtocolLines();f.feed(b'x'*65535+b'\n')
        self.assertEqual(len(f.pop_line()),65536)
    def test_no_lf_limit_and_complete_overlimit(self):
        f=BoundedProtocolLines();self.reject(f,b'x'*65536,'unterminated line cannot fit LF within line limit')
        f=BoundedProtocolLines();f.feed(b'x'*65535)
        self.reject(f,b'x\n','complete line exceeds configured line limit including LF')
    def test_chunk_and_buffer_budget_atomicity(self):
        f=BoundedProtocolLines();self.reject(f,b'x'*65537,'chunk exceeds configured limit')
        f=BoundedProtocolLines(line_limit=8,chunk_limit=8,buffer_limit=8)
        f.feed(b'A\nB\nC\n')
        self.reject(f,b'D\nE\n','unconsumed bytes exceed configured buffer budget')
        self.assertEqual(f.queued_tail(),(b'A\n',b'B\n',b'C\n'))
        self.assertEqual(f.pop_line(),b'A\n');self.conserved(f)
        # A framing/budget error is latched, even if consumption frees capacity.
        with self.assertRaisesRegex(ProtocolFramingError,'^framer failed: '): f.feed(b'D\n')
    def test_consumption_frees_budget_without_discard(self):
        f=BoundedProtocolLines(line_limit=8,chunk_limit=8,buffer_limit=8)
        f.feed(b'A\nB\nC\n');self.assertEqual(f.pop_line(),b'A\n')
        f.feed(b'D\nE\n');self.conserved(f)
        self.assertEqual([f.pop_line() for _ in range(4)],[b'B\n',b'C\n',b'D\n',b'E\n'])
        self.assertIsNone(f.pop_line());self.assertFalse(f.state().has_unconsumed)
    def test_burst_and_split_content_no_loss(self):
        f=BoundedProtocolLines();FakeSource(b'RE',b'ADY 1\nT 42\nLP ',b'42:0.0',b'\nDONE 1\n').deliver(f)
        self.assertEqual(f.queued_tail(),(b'READY 1\n',b'T 42\n',b'LP 42:0.0\n',b'DONE 1\n'))
        self.assertEqual([f.pop_line() for _ in range(4)],list((b'READY 1\n',b'T 42\n',b'LP 42:0.0\n',b'DONE 1\n')))
        self.conserved(f)
    def test_done_boundary_complete_and_partial_tail_exposed(self):
        f=BoundedProtocolLines();f.feed(b'DONE 64\nPOST 7\npart')
        self.assertEqual(f.pop_line(),b'DONE 64\n')
        self.assertTrue(f.state().has_unconsumed)
        self.assertEqual(f.queued_tail(),(b'POST 7\n',));self.assertEqual(f.pending_tail(),b'part')
        # Caller decides whether POST is legal here and assigns it before a new
        # request/QUIT. The framer does not silently reject or reattribute it.
        self.assertEqual(f.pop_line(),b'POST 7\n');f.feed(b'ial\n')
        self.assertEqual(f.pop_line(),b'partial\n');self.conserved(f)
        self.assertFalse(f.state().has_unconsumed)
    def test_caller_explicit_boundary_consumption(self):
        f=BoundedProtocolLines();f.feed(b'DONE A\nHEALTH ok\n')
        self.assertEqual(f.pop_line(),b'DONE A\n')
        self.assertTrue(f.state().has_unconsumed) # not ready for unconditional new GEN attribution
        observed=[]
        while (line:=f.pop_line()) is not None: observed.append(line)
        self.assertEqual(observed,[b'HEALTH ok\n'])
        self.assertFalse(f.state().has_unconsumed)
        f.feed(b'RESUME B\nDONE B\n');self.assertEqual(f.pop_line(),b'RESUME B\n')
        self.assertEqual(f.pop_line(),b'DONE B\n');self.conserved(f)
    def test_eof_preserves_complete_queue_and_unterminated_tail(self):
        f=BoundedProtocolLines();f.feed(b'DONE\nTAIL\n');state=f.finish()
        self.assertTrue(state.eof);self.assertTrue(state.has_unconsumed)
        self.assertEqual([f.pop_line(),f.pop_line()],[b'DONE\n',b'TAIL\n'])
        self.assertFalse(f.finish().has_unconsumed)
        f=BoundedProtocolLines();f.feed(b'DONE\nunterminated')
        with self.assertRaisesRegex(ProtocolFramingError,'^EOF with unterminated protocol line$'): f.finish()
        self.assertTrue(f.state().eof);self.assertEqual(f.pending_tail(),b'unterminated')
        self.assertEqual(f.pop_line(),b'DONE\n');self.assertTrue(f.state().has_unconsumed)
    def test_empty_chunk_is_not_eof_and_feed_after_eof(self):
        f=BoundedProtocolLines();f.feed(b'');self.assertFalse(f.state().eof)
        self.assertIsNone(f.pop_line());f.finish()
        with self.assertRaisesRegex(ProtocolFramingError,'^feed after EOF$'): f.feed(b'A\n')
    def test_cr_policy_preserves_or_explicitly_rejects(self):
        f=BoundedProtocolLines();FakeSource(b'A\r',b'\nB\rC\n').deliver(f)
        self.assertEqual([f.pop_line(),f.pop_line()],[b'A\r\n',b'B\rC\n'])
        f=BoundedProtocolLines(cr_policy='reject');f.feed(b'GOOD\n')
        self.reject(f,b'A\r\n','CR forbidden by configured policy')
        self.assertEqual(f.pop_line(),b'GOOD\n')
    def test_validation_atomic_even_with_good_prefix_bad_suffix(self):
        f=BoundedProtocolLines(line_limit=8,chunk_limit=16,buffer_limit=32);f.feed(b'old')
        self.reject(f,b'\nGOOD\n12345678','unterminated line cannot fit LF within line limit')
        self.assertEqual(f.pending_tail(),b'old');self.assertEqual(f.queued_tail(),())
    def test_empty_lines_and_raw_bytes_no_decoding(self):
        f=BoundedProtocolLines();f.feed(b'\n\x00\xff\n')
        self.assertEqual([f.pop_line(),f.pop_line()],[b'\n',b'\x00\xff\n']);self.conserved(f)
    def test_configuration_and_input_types(self):
        for args in (dict(line_limit=65537),dict(chunk_limit=65537),dict(buffer_limit=0),
                     dict(line_limit=True),dict(line_limit=8,buffer_limit=7),dict(cr_policy='strip')):
            with self.assertRaises(ValueError): BoundedProtocolLines(**args)
        f=BoundedProtocolLines();self.reject(f,bytearray(b'A\n'),'chunk must be bytes')


if __name__=='__main__': unittest.main()
