"""Independent literal CPU fixtures; not actual closed-r6 CLI evidence."""
import unittest
import static_cache_quota_trace as trace
import static_cache_quota as dp


def pair(l,e,n): return dict(layer=l,expert=e,entries=n)


class TraceFixtures(unittest.TestCase):
    def test_rank_not_count_order_and_missing_zero(self):
        ranked=[(1,2),(0,2),(1,0),(0,0),(0,1),(1,1)]
        curves=trace.prefix_curves(ranked,[pair(0,0,100),pair(0,2,3),pair(1,0,7)],2,3,3)
        self.assertEqual(curves,[[0,3,103,103],[0,0,7,7]])
        self.assertEqual(dp.score_prefix_quota(curves,[1,1]),3)
        self.assertEqual(dp.score_prefix_quota(curves,[2,1]),103)

    def test_greedy_failure_literal(self):
        curves=trace.prefix_curves([(0,1),(1,0),(0,0),(1,1)],
            [pair(0,1,10),pair(1,0,9),pair(1,1,100)],2,2,2)
        self.assertEqual(curves,[[0,10,10],[0,9,109]])
        result=dp.select_prefix_quotas(curves,2,[1,1])
        self.assertEqual((result['quotas'],result['score'],result['baseline_score']),([0,2],109,19))

    def test_production_truncation_and_default(self):
        ranked=[(l,e) for e in range(512) for l in range(48)]
        curves=trace.prefix_curves(ranked,[pair(0,127,11),pair(0,128,999),pair(47,33,7),pair(47,34,888),pair(1,0,2)])
        self.assertEqual(len(curves),48)
        self.assertTrue(all(len(c)==129 for c in curves))
        self.assertEqual(curves[0][-1],11)
        self.assertEqual(dp.score_prefix_quota(curves,trace.DEFAULT),9)
        q=[0]*48;q[0]=128
        self.assertEqual(dp.score_prefix_quota(curves,q),11)

    def test_rank_negatives(self):
        for ranks,branch in (([(0,0),(0,0)],'duplicate profile'),
                             ([(False,0),(0,1)],'profile pair bounds'),
                             ([(0,2),(0,1)],'profile pair bounds'),
                             ([(0,0)],'complete rank geometry')):
            with self.subTest(branch=branch),self.assertRaisesRegex(ValueError,branch):
                trace.prefix_curves(ranks,[],1,2,2)
        for geometry in ((True,2,2),(49,2,2),(1,513,2),(1,2,129),(1,2,0)):
            with self.subTest(geometry=geometry),self.assertRaisesRegex(ValueError,'curve geometry'):
                trace.prefix_curves([],[],*geometry)

    def test_count_negatives(self):
        ranked=[(0,0),(0,1)]
        cases=[([pair(0,0,True)],'uint64 entry'),([pair(0,0,-1)],'uint64 entry'),
               ([pair(0,0,2**64)],'uint64 entry'),([pair(False,0,1)],'count pair bounds'),
               ([pair(0,2,1)],'count pair bounds'),([pair(0,0,1),pair(0,0,2)],'duplicate count'),
               ([pair(0,0,2**64-1),pair(0,1,1)],'prefix overflow')]
        for counts,branch in cases:
            with self.subTest(branch=branch),self.assertRaisesRegex(ValueError,branch):
                trace.prefix_curves(ranked,counts,1,2,2)

    def test_cleanup_explicit_false_only(self):
        good=dict(forced=False,inferior_survived=False,gdb_survived=False)
        trace.cleanup_gate(dict(cleanup=good))
        for value in (None,{},dict(forced=False),dict(good,forced=True),dict(good,gdb_survived=0)):
            with self.subTest(value=value),self.assertRaises(ValueError):
                trace.cleanup_gate(dict(cleanup=value))
        with self.assertRaises(ValueError): trace.cleanup_gate({})

    def test_curve_hash_literal(self):
        import hashlib
        expected=hashlib.sha256(b'[[0,10,10],[0,9,109]]\n').hexdigest()
        self.assertEqual(trace.curves_hash([[0,10,10],[0,9,109]]),expected)
        self.assertNotEqual(trace.curves_hash([[0,9,109],[0,10,10]]),expected)


if __name__=='__main__': unittest.main()
