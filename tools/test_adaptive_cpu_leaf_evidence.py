"""Main-authored CPU-only checks of completed native leaf/footer evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import adaptive_cpu_leaf_evidence as leaf


def fixture(request=1, groups=((18, 20, 1, 2), (22, 42, 4, 3))):
    sample = dict(prompt_tokens=33207 + request, reused_tokens=0,
                  read_tokens=33207 + request, generated_tokens=64,
                  drafts_accepted=32, drafts_offered=48)
    lines = ['strata CPU_SERVICE request=' + str(request) + ' ' +
             ' '.join(k + '=' + v for k, v in leaf.service.HEADER.items())]
    totals = dict(calls=0, wall_ns=0, cpu_ns=0)
    for gu, down, nt, jobs in groups:
        for op, rows, calls, cpu in (('GU', 640, 2, 11), ('HQ', 640, 1, 7), ('DOWN', 2560, 4, 13)):
            value = dict(calls=jobs * calls, wall_ns=jobs * (cpu + 2), cpu_ns=jobs * cpu)
            lines.append(f'strata CPU_SERVICE_BIN request={request} gu={gu} down={down} nt={nt} op={op} '
                         f'calls={value["calls"]} rows={jobs * rows} token_rows={jobs * rows * nt} '
                         f'wall_ns={value["wall_ns"]} cpu_ns={value["cpu_ns"]}')
            for k in totals:
                totals[k] += value[k]
    if groups:
        for slot, tid, remainder in ((0, 111, False), (64, 222, True)):
            value = {k: v - v // 2 if remainder else v // 2 for k, v in totals.items()}
            lines.append(f'strata CPU_SERVICE_THREAD request={request} slot={slot} tid={tid} ' +
                         ' '.join(k + '=' + str(v) for k, v in value.items()))
    lines.append(f'strata CPU_SERVICE_END request={request} complete=1 '
                 f'bins={len(groups) * 3} threads={2 if groups else 0}')
    lines.append(f'strata serve: prompt {sample["prompt_tokens"]} tokens = 0 reused + '
                 f'{sample["read_tokens"]} read in 100 ms, 64 generated in 200 ms, '
                 'drafts accepted 32 of 48, 0 checkpoints')
    counters = dict.fromkeys(leaf.footer.COPY_KEYS, 0)
    counters.update(cpu_unique_misses=sum(g[3] for g in groups),
                    cpu_entries=sum(g[2] * g[3] for g in groups))
    lines.append('strata serve: SYCL_EXPERT_COPY ' +
                 ' '.join(k + '=' + str(v) for k, v in counters.items()))
    return '\n'.join(lines) + '\n', sample


class LeafEvidenceTests(unittest.TestCase):
    def test_dynamic_requests_conserve_native_jobs_and_entries(self):
        a, sa = fixture()
        b, sb = fixture(2, ((21, 20, 2, 1),))
        rows = leaf.extract(a + b, [sa, sb], 4)
        self.assertEqual([(r['cpu_jobs'], r['cpu_entries']) for r in rows], [(5, 14), (1, 2)])
        self.assertEqual(rows[0]['cpu_ns'], 155)
        self.assertEqual(rows[0]['wall_ns'], 185)
        self.assertEqual(rows[0]['thread_count'], 2)

    def test_cap2_and_zero_job_requests(self):
        text, sample = fixture(groups=((18, 20, 1, 2), (21, 20, 2, 3)))
        self.assertEqual(leaf.extract(text, [sample], 2)[0]['cpu_entries'], 8)
        text, sample = fixture(groups=())
        row = leaf.extract(text, [sample], 2)[0]
        self.assertEqual((row['cpu_jobs'], row['cpu_entries'], row['thread_count']), (0, 0, 0))

    def test_corrupt_native_evidence_is_rejected(self):
        text, sample = fixture()
        replacements = [
            ('rows=1280', 'rows=1281'),
            ('op=HQ calls=2', 'op=HQ calls=3'),
            ('cpu_unique_misses=5', 'cpu_unique_misses=6'),
            ('cpu_entries=14', 'cpu_entries=15'),
            ('wall_ns=26 cpu_ns=22', 'wall_ns=21 cpu_ns=22'),
            ('op=GU', 'op=UNKNOWN'),
            ('nt=4', 'nt=5'),
            ('calls=4 rows=', 'calls=4 calls=4 rows='),
            ('request=1 valid=1', 'request=2 valid=1'),
            ('bins=6 threads=2', 'bins=5 threads=2'),
            ('submitted_bytes=0', 'submitted_bytes=1'),
            ('slot=64', 'slot=63'),
            ('tid=222', 'tid=111'),
            ('strata serve: prompt ', 'strata serve: prompt broken '),
            ('cpu_entries=14', 'cpu_entries=18446744073709551616'),
        ]
        for old, new in replacements:
            with self.subTest(mutation=new):
                self.assertIn(old, text)
                with self.assertRaises(RuntimeError):
                    leaf.extract(text.replace(old, new, 1), [sample], 4)
        lines = text.splitlines(keepends=True)
        corruptions = [text + lines[0], ''.join(lines[:1] + lines[2:]),
                       ''.join(lines[:2] + [lines[1]] + lines[2:]),
                       ''.join(lines[:1] + [lines[2], lines[1]] + lines[3:]),
                       ''.join(line for line in lines if 'CPU_SERVICE_END' not in line),
                       ''.join(lines[:-2] + [lines[-1], lines[-2]]),
                       'strata cpu work: begin request=1\n' + text]
        for changed in corruptions:
            with self.subTest(framing=changed[-160:]):
                with self.assertRaises(RuntimeError):
                    leaf.extract(changed, [sample], 4)

    def test_thread_owners_must_remain_stable_across_requests(self):
        a, sa = fixture()
        b, sb = fixture(2)
        for changed in (b.replace('tid=111', 'tid=333'), b.replace('tid=222', 'tid=111')):
            with self.assertRaises(RuntimeError):
                leaf.extract(a + changed, [sa, sb], 4)

    def test_overflow_and_incompatible_scope_fail_closed(self):
        text, sample = fixture()
        lines = text.splitlines()
        for i in (1, 2):
            words = lines[i].split()
            words = [f'{w.split("=")[0]}={leaf.MAX}' if w.startswith(('wall_ns=', 'cpu_ns=')) else w
                     for w in words]
            lines[i] = ' '.join(words)
        with self.assertRaises(RuntimeError):
            leaf.extract('\n'.join(lines), [sample], 4)
        for width, workers in ((1, 5), (True, 5), (4, 0), (4, 65), (2, 5)):
            with self.assertRaises(RuntimeError):
                leaf.extract(text, [sample], width, workers)
        for key, value in (('read_tokens', -1), ('generated_tokens', True), ('drafts_accepted', 49)):
            bad = copy.deepcopy(sample)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                leaf.extract(text, [bad], 4)


if __name__ == '__main__':
    unittest.main()
