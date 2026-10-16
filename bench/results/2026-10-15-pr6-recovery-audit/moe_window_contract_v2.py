"""Extend the frozen ownership oracle with exact expert/job coverage.

This is a host contract, not C++/SYCL inference or a device qualification.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True
BASE = Path(__file__).with_name('moe_window_contract_v1.py')
if hashlib.sha256(BASE.read_bytes()).hexdigest() != 'e85ec84e890b2d6bdb2823b8da02ae2f1091e4e4af89b2d52e6da244bbc7f64a':
    raise RuntimeError('Frozen v1 scheduling/memory contract changed.')
_spec = importlib.util.spec_from_file_location('frozen_moe_window_contract_v1', BASE)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)
require = base.require


class Lifetime(base.Lifetime):
    def __init__(self, window, final_layer=False):
        super().__init__(window, final_layer)
        self.jobs = ()
        self.next_job = 0
        self.offsets = ()
        self.dequantized = set()
        self.rows_submitted = 0
        self.host_source_live = False
        self.emissions = []

    def step(self, action, piece=None, *, counts=None):
        require(not self.invalid, 'request already invalid')
        if action == 'route_readback':
            require(counts is not None and len(counts) == 512,
                    'all expert counts are required')
            require(all(type(c) is int and 0 <= c <= self.window.rows for c in counts),
                    'expert count outside unique top-k window bound')
            require(sum(counts) == self.window.rows * 10, 'route conservation')
            super().step(action, piece)
            offsets = []; total = 0; jobs = []
            for e, count in enumerate(counts):
                offsets.append(total)
                for begin, rows in base.expert_jobs(count, self.window.rows):
                    jobs.append((e,begin,rows))
                total += count
            self.offsets = tuple(offsets)
            self.jobs = tuple(jobs)
            self.host_source_live = True
        elif action == 'dequantize':
            require(self.stage == 'experts' and self.next_job < len(self.jobs),
                    'no outstanding expert job')
            require(piece == self.jobs[self.next_job][0] and piece not in self.dequantized,
                    'dequantize each active expert once before its first job')
            self.dequantized.add(piece)
            self.history.append((action,piece))
        elif action == 'expert_job':
            require(self.stage == 'experts' and self.next_job < len(self.jobs) and
                    piece == self.jobs[self.next_job], 'missing, repeated or unordered expert job')
            e, begin, rows = piece
            require(e in self.dequantized and self.host_source_live and self.mixed_live,
                    'expert operands are not live')
            source = base.source_upload_span(self.offsets[e]+begin, rows,
                                             self.window.rows*10)
            require(source['device_offset']==0, 'source list must be rebased')
            self.next_job += 1
            self.rows_submitted += rows
            self.history.append((action,piece))
            return dict(**source,down_output_row=self.offsets[e]+begin,
                        tile='compact' if rows<128 else 'wide')
        elif action == 'combine':
            require(self.next_job == len(self.jobs) and self.jobs and
                    self.rows_submitted == self.window.rows*10,
                    'combine before all expert jobs have been submitted')
            super().step(action,piece)
        elif action == 'reuse_host_source':
            require(not self.host_source_live,
                    'H2D source owner must remain immutable until checked drain')
        elif action == 'checked_drain':
            super().step(action,piece)
            self.host_source_live = False
        elif action == 'emit':
            super().step(action,piece)
            p = self.window.pieces[piece]
            self.emissions.append(dict(row=p.row,count=p.count,position=p.position,
                operation='FP32_callback_from_bank' if self.final_layer else 'internal_bank_completion'))
        elif action == 'change_queue':
            raise ValueError('one immutable in-order queue is required')
        else:
            super().step(action,piece)

    def cancel(self, checked_retirement):
        super().cancel(checked_retirement)
        self.host_source_live = False
