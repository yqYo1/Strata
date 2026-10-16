"""Main's executable scheduling/memory contract, not an inference implementation.

The first candidate groups GPU-resident residual pieces only. Causal operators
stay at 8192 rows; an expert's count may reach 16384, but each GEMM job stays
bounded by 8192 rows. No runtime, queue or device is created by this module.
"""
from dataclasses import dataclass

CHUNK = 8192
WINDOW = 2 * CHUNK
CONTEXT = 262144
N, HC, TOPK, FF = 2560, 4, 10, 640
D = N * HC
REGION = 1180698624
EMB_PAYLOAD = CHUNK * N * 4


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


@dataclass(frozen=True)
class Piece:
    row: int
    count: int
    position: int


@dataclass(frozen=True)
class Window:
    pieces: tuple
    gpu: bool
    grouped: bool

    @property
    def rows(self):
        return sum(p.count for p in self.pieces)


def schedule(rows, gpu_rows, position=0):
    require(0 < rows <= CONTEXT and 0 <= position <= CONTEXT - rows,
            'request outside physical context')
    require(0 <= gpu_rows <= rows and (gpu_rows == rows or gpu_rows % CHUNK == 0),
            'residual boundary must match the existing chunk plan')
    pieces = [Piece(r, min(CHUNK, rows-r), position+r)
              for r in range(0, rows, CHUNK)]
    result = []
    i = 0
    while i < len(pieces):
        p = pieces[i]
        gpu = p.row < gpu_rows
        pair = (gpu and i+1 < len(pieces) and
                pieces[i+1].row + pieces[i+1].count <= gpu_rows)
        group = tuple(pieces[i:i+2] if pair else pieces[i:i+1])
        result.append(Window(group, gpu, bool(pair)))
        i += len(group)
    return tuple(result)


def expert_jobs(expert_rows, window_rows):
    require(0 < window_rows <= WINDOW and 0 <= expert_rows <= window_rows,
            'expert count outside window')
    # Copy host_src[offset+begin:offset+begin+count] to device_src[0:count].
    # The complete host_src owner remains immutable until the window drains.
    return tuple((r, min(CHUNK, expert_rows-r))
                 for r in range(0, expert_rows, CHUNK))


@dataclass(frozen=True)
class Span:
    offset: int
    payload: int
    reserved: int

    @property
    def end(self):
        return self.offset + self.reserved


def carve(payloads, capacity):
    spans = {}
    offset = 0
    for name, payload in payloads:
        require(payload > 0 and offset % 256 == 0, 'invalid carve geometry')
        reserved = (payload + 256 + 255) // 256 * 256
        require(offset + reserved <= capacity, 'carve exceeds owning extent')
        spans[name] = Span(offset, payload, reserved)
        offset += reserved
    return spans, offset


def memory_plan():
    # New typed views within the existing region. Leave all legacy m.* fields
    # unchanged. Dm hosts preparation temporaries BEFORE expert outputs exist.
    spans, used = carve([
        ('down_bf16', WINDOW * TOPK * N * 2),
        ('shared_bf16', WINDOW * N * 2),
        ('weights_f32', WINDOW * TOPK * 4),
        ('ids_then_slots_i32', WINDOW * TOPK * 4),
        ('shared_gate_f32', WINDOW * 4),
        ('injection_f32', WINDOW * HC * 4),
        ('gather_f16', CHUNK * N * 2),
        ('gate_up_bf16', CHUNK * 2 * FF * 2),
        ('hidden_f16', CHUNK * FF * 2),
        ('expert_src_i32', CHUNK * 4),
    ], REGION)
    prep, prep_used = carve([
        ('router_logits_f32', CHUNK * 512 * 4),
        ('shared_gate_bf16', CHUNK * FF * 2),
        ('shared_up_bf16', CHUNK * FF * 2),
        ('shared_hidden_f16', CHUNK * FF * 2),
    ], spans['down_bf16'].payload)
    require(WINDOW * N * 2 == EMB_PAYLOAD,
            'window mixed/BO plane exceeds old embedding backing')
    return dict(region=spans, region_used=used, region_capacity=REGION,
                prep=prep, prep_used=prep_used,
                emb_mixed_then_bo_bytes=EMB_PAYLOAD)


class Lifetime:
    """Abstract submission/order oracle; it never models tensor arithmetic.

    Region release requires a positive checked retirement. A failure invalidates
    the request; it does not roll session tensors back. The bank itself is an
    outer request owner, distinct from FP32 m.R and m.region.
    """
    def __init__(self, window, final_layer=False):
        require(window.grouped and window.gpu, 'GPU window required')
        self.window = window
        self.final_layer = final_layer
        self.bank = ['input'] * len(window.pieces)
        self.prepared = 0
        self.stage = 'attention'
        self.next_attention = 0
        self.next_write = 0
        self.next_emit = 0
        self.region_live = True
        self.mixed_live = False
        self.bo_live = False
        self.invalid = False
        self.history = []

    def step(self, action, piece=None):
        require(not self.invalid, 'request already invalid')
        self.history.append((action, piece))
        if action == 'attention_store':
            require(self.stage == 'attention' and piece == self.next_attention,
                    'causal pieces must stay ordered')
            require(self.bank[piece] == 'input', 'input was already consumed')
            self.bank[piece] = 'post_attention'
            self.next_attention += 1
            if self.next_attention == len(self.bank):
                self.stage = 'prepare'
        elif action == 'prepare':
            require(self.stage == 'prepare' and piece == self.prepared,
                    'finish both causal pieces before borrowing embedding plane')
            require(all(s == 'post_attention' for s in self.bank),
                    'post-attention rows must remain in their own bank')
            self.prepared += 1
            self.mixed_live = True
        elif action == 'route_readback':
            require(self.stage == 'prepare' and self.prepared == len(self.bank),
                    'router IDs are incomplete')
            self.stage = 'experts'
        elif action == 'combine':
            require(self.stage == 'experts' and self.mixed_live,
                    'all expert submissions must precede mixed-to-BO reuse')
            # Same in-order queue: combine follows the final expert read.
            self.mixed_live = False
            self.bo_live = True
            self.stage = 'write'
        elif action == 'write_final':
            require(self.stage == 'write' and self.bo_live and
                    piece == self.next_write, 'invalid HC final-write order')
            require(self.bank[piece] == 'post_attention', 'lost HC input')
            self.bank[piece] = 'final'
            self.next_write += 1
        elif action == 'checked_drain':
            require(self.stage == 'write' and self.next_write == len(self.bank),
                    'both writes must precede callbacks or region reuse')
            self.region_live = self.bo_live = False
            self.stage = 'emit'
        elif action == 'emit':
            require(self.stage == 'emit' and not self.region_live and
                    not self.bo_live and piece == self.next_emit,
                    'callback can overwrite live MoE or injection data')
            require(self.bank[piece] == 'final', 'cannot publish post-attention rows')
            self.next_emit += 1
            # Final model callback may expand the BANK slice into FP32 m.R,
            # then MTP may carve m.region. Neither holds another live slice.
            if self.next_emit == len(self.bank):
                self.stage = 'done'
        else:
            raise ValueError('unknown action')

    def cancel(self, checked_retirement):
        self.invalid = True
        if not checked_retirement:
            # A real controller must abort with owners live on drain failure.
            raise RuntimeError('retirement failed; retain owners until process exit')
        self.region_live = False


def source_upload_span(global_offset, job_rows, total_routes):
    require(0 <= global_offset <= total_routes and
            0 < job_rows <= CHUNK and job_rows <= total_routes-global_offset,
            'host source-map bounds')
    return dict(host_offset=global_offset, device_offset=0, rows=job_rows,
                device_capacity=CHUNK)
