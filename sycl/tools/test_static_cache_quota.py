"""Independent exhaustive tiny oracle; SOURCE ONLY, root owns execution."""
import itertools
import unittest
from static_cache_quota import select_prefix_quotas, score_prefix_quota


def exhaustive(curves,budget,default):
    options=[]
    for q in itertools.product(*(range(len(curve)) for curve in curves)):
        if sum(q)==budget:
            score=sum(curve[n] for curve,n in zip(curves,q))
            distance=sum(abs(n-d) for n,d in zip(q,default))
            options.append((-score,distance,q))
    best=min(options)
    return -best[0],best[1],list(best[2])


class PrefixQuotaFixtures(unittest.TestCase):
    def check_oracle(self,curves,budget,default):
        score,distance,quota=exhaustive(curves,budget,default)
        result=select_prefix_quotas(curves,budget,default)
        self.assertEqual((result['score'],result['l1_distance_from_default'],result['quotas']),
                         (score,distance,quota))
        self.assertEqual(sum(result['quotas']),budget)
        self.assertEqual(result['baseline_gain'],score-sum(curve[d] for curve,d in zip(curves,default)))
    def test_literal_greedy_failure(self):
        result=select_prefix_quotas([[0,10,10],[0,9,109]],2,[1,1])
        self.assertEqual((result['score'],result['baseline_score'],result['baseline_gain'],result['quotas']),
                         (109,19,90,[0,2]))
        self.check_oracle([[0,10,10],[0,9,109]],2,[1,1])
    def test_tiny_independent_exhaustive_grid(self):
        curves_options=([0],[0,0],[0,1],[0,0,3],[0,4,4],[0,2,5])
        for curves in itertools.product(curves_options,repeat=3):
            for default in itertools.product(*(range(len(curve)) for curve in curves)):
                budget=sum(default)
                self.check_oracle(curves,budget,list(default))
    def test_uniform_default_and_lex_ties(self):
        self.assertEqual(select_prefix_quotas([[0,1,2]]*3,3,[1,1,1])['quotas'],[1,1,1])
        # Score requires two slots in layers0/1. Defaults put both in zero-score
        # layer2; [0,2,0],[1,1,0],[2,0,0] tie in score and L1; lex picks first.
        curves=[[0,1,2],[0,1,2],[0,0,0]]
        result=select_prefix_quotas(curves,2,[0,0,2])
        self.assertEqual((result['score'],result['l1_distance_from_default'],result['quotas']),(2,4,[0,2,0]))
        self.check_oracle(curves,2,[0,0,2])
    def test_zero_and_full_generic_budget(self):
        self.assertEqual(select_prefix_quotas([[0,4],[0,9]],0,[0,0])['quotas'],[0,0])
        self.assertEqual(select_prefix_quotas([[0,4],[0,0,9]],3,[1,2])['quotas'],[1,2])
        self.assertEqual(score_prefix_quota([[0,4],[0,0,9]],[1,2]),13)
    def test_maximal_production_geometry_hand_oracle(self):
        # Only final layer has positive marginal counts, always3. All128 slots
        # must go there for score384; every other layer has zero score.
        curves=[[0]*257 for unused in range(47)]+[[3*q for q in range(257)]]
        result=select_prefix_quotas(curves,128,[2]*47+[34])
        self.assertEqual(result['quotas'],[0]*47+[128])
        self.assertEqual((result['score'],result['baseline_score'],result['baseline_gain'],
                          result['l1_distance_from_default']),(384,102,282,188))
    def test_all_zero_full_production_default_tie(self):
        default=[2]*47+[34]
        self.assertEqual(select_prefix_quotas([[0]*257]*48,128,default)['quotas'],default)
    def test_malformed_geometry_defaults_nonmonotone_and_booleans(self):
        cases=[([],0,[]),([[0]]*49,0,[0]*49),([[0]*258],0,[0]),([[0,2,1]],1,[1]),
               ([[1,2]],1,[1]),([[0]],1,[1]),([[0,1]],1,[0]),([[0,1]],1,[]),
               ([[0,1]],129,[1]),([[0,1]],-1,[0]),([[0,1]],True,[1]),
               ([[0,True]],1,[1]),([[0,1]],1,[True]),([[0,-1]],1,[1]),
               ([[0,1.0]],1,[1]),([[0,1]],1,[1.0]),([[0]],0,[False]),
               ([[0,1]],1,(x for x in [1]))]
        for curves,budget,default in cases:
            with self.subTest(curves=curves,budget=budget,default=default),self.assertRaises(ValueError):
                select_prefix_quotas(curves,budget,default)
    def test_uint64_individual_and_additive_overflow(self):
        maximum=(1<<64)-1
        self.assertEqual(select_prefix_quotas([[0,maximum]],1,[1])['score'],maximum)
        with self.assertRaises(ValueError): select_prefix_quotas([[0,maximum+1]],1,[1])
        with self.assertRaisesRegex(ValueError,'uint64 additive score overflow'):
            select_prefix_quotas([[0,maximum],[0,1]],2,[1,1])
        # Baseline fits; another feasible exact-budget objective overflows.
        with self.assertRaisesRegex(ValueError,'uint64 additive score overflow'):
            select_prefix_quotas([[0,maximum],[0,1,1]],2,[0,2])


if __name__=='__main__': unittest.main()
