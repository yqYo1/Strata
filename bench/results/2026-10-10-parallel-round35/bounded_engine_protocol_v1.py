"""Pure CPU byte framing; no engine/controller policy or I/O.

Lines include their LF and are never stripped or decoded. Default CR policy
preserves CR as content (including CRLF); optional 'reject' forbids every CR.
Caller validates message semantics and consumes/classifies trailing output.
"""
from collections import deque
from dataclasses import dataclass


class ProtocolFramingError(ValueError):
    pass


@dataclass(frozen=True)
class BufferState:
    received_bytes: int
    consumed_bytes: int
    queued_lines: int
    queued_bytes: int
    pending_bytes: int
    eof: bool
    error: str | None

    @property
    def buffered_bytes(self):
        return self.queued_bytes + self.pending_bytes

    @property
    def has_unconsumed(self):
        return self.buffered_bytes != 0


class BoundedProtocolLines:
    """Single-caller bounded LF splitter with an explicit unread tail.

    feed() is atomic: rejected chunks are not accepted partially. All previously
    accepted bytes remain accessible. Framing errors latch; no further feed is
    allowed. pop_line()/state/tail inspection remain available for diagnostics.
    Empty feed is a no-op, not EOF. finish() alone marks EOF. No reset/discard API.
    """
    def __init__(self, *, line_limit=65536, chunk_limit=65536,
                 buffer_limit=131072, cr_policy='preserve'):
        for name,value in (('line_limit',line_limit),('chunk_limit',chunk_limit),('buffer_limit',buffer_limit)):
            if type(value) is not int or not 1 <= value <= 131072:
                raise ValueError(name+' must be an integer in 1..131072')
        if line_limit > 65536:
            raise ValueError('line_limit exceeds canonical 65536 bytes including LF')
        if chunk_limit > 65536:
            raise ValueError('chunk_limit exceeds 65536 bytes')
        if buffer_limit < line_limit:
            raise ValueError('buffer_limit must cover one maximum line')
        if cr_policy not in ('preserve','reject'):
            raise ValueError('cr_policy must be preserve or reject')
        self.line_limit=line_limit; self.chunk_limit=chunk_limit
        self.buffer_limit=buffer_limit; self.cr_policy=cr_policy
        self._queue=deque(); self._pending=bytearray(); self._queued_bytes=0
        self._received=0; self._consumed=0; self._eof=False; self._error=None

    def state(self):
        return BufferState(self._received,self._consumed,len(self._queue),self._queued_bytes,
                           len(self._pending),self._eof,self._error)

    def queued_tail(self):
        """Immutable complete lines, in order; inspection consumes nothing."""
        return tuple(self._queue)

    def pending_tail(self):
        """Immutable bounded copy of the unfinished line; consumes nothing."""
        return bytes(self._pending)

    def _fail(self,message):
        self._error=message
        raise ProtocolFramingError(message)

    def feed(self,chunk):
        if self._error is not None:
            raise ProtocolFramingError('framer failed: '+self._error)
        if self._eof:
            self._fail('feed after EOF')
        if type(chunk) is not bytes:
            self._fail('chunk must be bytes')
        if len(chunk)>self.chunk_limit:
            self._fail('chunk exceeds configured limit')
        if self.state().buffered_bytes+len(chunk)>self.buffer_limit:
            self._fail('unconsumed bytes exceed configured buffer budget')
        if self.cr_policy=='reject' and b'\r' in chunk:
            self._fail('CR forbidden by configured policy')
        # Validate lengths before copying or mutating any accepted state. No
        # pending+whole-chunk concatenation is made, even for newline bursts.
        start=0; unfinished=len(self._pending)
        while True:
            newline=chunk.find(b'\n',start)
            if newline<0:
                unfinished+=len(chunk)-start
                if unfinished>=self.line_limit:
                    self._fail('unterminated line cannot fit LF within line limit')
                break
            if unfinished+newline-start+1>self.line_limit:
                self._fail('complete line exceeds configured line limit including LF')
            unfinished=0; start=newline+1
        # The complete chunk is now admitted. Copy only one validated line or
        # bounded partial segment at a time; queue payload+pending stays capped.
        start=0
        while True:
            newline=chunk.find(b'\n',start)
            if newline<0:
                self._pending.extend(chunk[start:])
                break
            self._pending.extend(chunk[start:newline+1])
            line=bytes(self._pending);self._pending.clear()
            self._queue.append(line);self._queued_bytes+=len(line)
            start=newline+1
        self._received+=len(chunk)

    def pop_line(self):
        """Return the next raw LF-terminated bytes, or None; never a partial."""
        if not self._queue:
            return None
        line=self._queue.popleft()
        self._queued_bytes-=len(line);self._consumed+=len(line)
        return line

    def finish(self):
        """Declare EOF. Complete queued tails survive; unfinished tails error."""
        if self._error is not None:
            raise ProtocolFramingError('framer failed: '+self._error)
        self._eof=True
        if self._pending:
            self._fail('EOF with unterminated protocol line')
        return self.state()
