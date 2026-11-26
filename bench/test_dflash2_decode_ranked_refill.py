"""Host controls for the ranked-refill cancellation and timing readers."""

from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import dflash2_decode_ranked_refill_restore as restore
import dflash2_decode_ranked_refill_speed as speed


DIGESTS = dict(stock='9e0b070ebe90ab9b', blend='68eea56314d4990b', masked='9377e60e8ad8ea63')
KV_STAGE = 'strata prefill KV stage: layer=3 reloaded_pages=256 reused_pages=512'
PUBLISH = 'strata DFlash2 decode refill: published first=132 slots=1124 payload_bytes=2321664000'
STARTUP = 'strata DFlash2 decode refill: available slots=1256 first_changed=132'


class CancelPeer:
    def __init__(self, work, diagnostic, info, *, late_stop=False, bad_mask=False):
        self.work = iter(work)
        self.diagnostic = diagnostic
        self.lines = deque(('INFO ' + ' '.join(f'{k}={v}' for k, v in info.items()), 'READY 262144 stop'))
        self.pending_stop = None
        self.index, self.stops = 0, []
        self.late_stop, self.bad_mask = late_stop, bad_mask
        self.quit = False
        self.exit_watch = self

    def poll(self, *_):
        return False

    def line(self, _):
        return self.lines.popleft()

    def finish(self, n, *, cancelled=False):
        if cancelled:
            self.lines.extend((f'PP 8192 {n} 1 1', 'REUSED 0',
                               f'DONE 0 {n} 1 1 cancel 0 0 0 0 0 0 0 0 8192'))
        else:
            self.lines.extend((f'PP {n-1} {n} 1 1', 'REUSED 0', 'T 17',
                               f'DONE 1 {n} 1 1 length 0 0 0 0 0 0 0 0 {n}'))

    def send(self, sent):
        if sent == b'QUIT\n':
            self.quit = True
        elif sent == b'STOP\n':
            if self.pending_stop is None:
                raise AssertionError('unexpected or duplicate STOP')
            self.stops.append(self.index)
            self.finish(self.pending_stop, cancelled=not self.late_stop)
            self.pending_stop = None
        else:
            payload, request = next(self.work)
            if sent != payload or self.pending_stop is not None:
                raise AssertionError('request bytes or order changed')
            self.index += 1
            n = request['prompt']
            if request['cancelled']:
                digest = DIGESTS['blend' if self.bad_mask else 'masked']
                phase = (f'strata DFlash2 expert phase: request={self.index} phase=prefill-batched '
                         f'from=0 to={n-1} first_masked=132 entries=24576 map={digest}')
                with self.diagnostic.open('a') as f:
                    f.write(phase + '\n' + KV_STAGE + '\n')
                self.pending_stop = n
            else:
                self.finish(n)

    def exit(self, _):
        if not self.quit or self.lines or self.pending_stop is not None:
            raise AssertionError('unsettled protocol at QUIT')


class RankedRefillReaders(unittest.TestCase):
    def simulated_cancel(self, **failure):
        with TemporaryDirectory(prefix='strata-refill-reader-') as directory:
            root = Path(directory)
            (root / 'owned').mkdir()
            diagnostic = root / 'owned' / 'cancel-ranked.stderr'
            diagnostic.touch()
            ctx = restore.experiment()
            work = list(restore.cancel_workloads(ctx))
            case = dict(root=directory, name='cancel-ranked', masked_digest=DIGESTS['masked'],
                        requests=[request for _, request in work])
            peer = CancelPeer(work, diagnostic, ctx.INFO, **failure)
            results = restore.interact_cancel(ctx, peer, case, 4096)
            return results, peer.stops

    def test_cancel_and_recover_in_both_placements(self):
        results, stops = self.simulated_cancel()
        self.assertEqual(stops, [1, 4])
        self.assertEqual([row['read'] for row in results], [8192, 33, 32769, 8192, 33])
        self.assertEqual([row['stop_after'] for row in results], [KV_STAGE, None, None, KV_STAGE, None])
        self.assertEqual([row['finish'] for row in results], ['cancel', 'length', 'length', 'cancel', 'length'])

    def test_wrong_mask_never_triggers_a_stop(self):
        with self.assertRaisesRegex(RuntimeError, 'unexpected expert phase'):
            self.simulated_cancel(bad_mask=True)

    def test_a_late_stop_cannot_qualify_prefill_cancellation(self):
        with self.assertRaisesRegex(RuntimeError, 'did not cancel before prompt completion'):
            self.simulated_cancel(late_stop=True)

    def test_cancellation_keeps_the_preexisting_profile(self):
        results, _ = self.simulated_cancel()
        trace = restore.expected_trace(results, DIGESTS, ranked=True)
        phases = [line for line in trace if line.startswith(restore.PHASE)]
        self.assertTrue(phases[1].endswith('map=' + DIGESTS['stock']))
        self.assertTrue(phases[2].endswith('map=' + DIGESTS['stock']))
        self.assertTrue(phases[5].endswith('map=' + DIGESTS['blend']))
        self.assertTrue(phases[7].endswith('map=' + DIGESTS['blend']))
        self.assertTrue(phases[8].endswith('map=' + DIGESTS['blend']))
        self.assertEqual(trace.count(PUBLISH), 1)

    def test_publication_is_scoped_to_the_expected_timed_requests(self):
        header = 'strata serve: prompt 96204 tokens = 0 reused + 96204 read in 100000 ms, '
        header += '2048 generated in 100000 ms, drafts accepted 1024 of 1100, 0 checkpoints'
        samples = [{}, {'workload': 'regular-code'}, {'workload': 'varied-prose'}]
        stock = [header] * 3
        ranked = [STARTUP, header, PUBLISH, header, PUBLISH, header]
        self.assertEqual(speed.check_publication('\n'.join(stock), samples, False),
                         {'regular-code': 0, 'varied-prose': 0})
        self.assertEqual(speed.check_publication('\n'.join(ranked), samples, True),
                         {'regular-code': 2321664000, 'varied-prose': 2321664000})
        for wrong in (ranked[:-2] + [header], [STARTUP, PUBLISH, *ranked[1:]], [*ranked, PUBLISH],
                      [line.replace('2321664000', '2321663999') for line in ranked], ranked[1:]):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(RuntimeError, 'profile|refill'):
                speed.check_publication('\n'.join(wrong), samples, True)


if __name__ == '__main__':
    unittest.main()
