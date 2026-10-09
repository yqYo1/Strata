"""Exact offline additive prefix-count oracle; no engine or file access.

One uniform MAXBLOB slot buys one additional within-layer profile prefix entry.
Nondecreasing prefix scores need not have diminishing marginal counts. This is
not packed-byte allocation, service cost, traffic, LRU or a live cache policy.
"""
U64_MAX = (1 << 64) - 1
MAX_LAYERS = 48
MAX_LAYER_SLOTS = 256
MAX_BUDGET = 128


def _integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(label+' must be an integer in bounds (bool forbidden)')


def _add(left, right):
    if right > U64_MAX-left:
        raise ValueError('uint64 additive score overflow')
    return left+right


def _validate(curves, budget, default):
    _integer(budget,0,MAX_BUDGET,'total budget')
    if type(curves) not in (list,tuple) or not 1 <= len(curves) <= MAX_LAYERS:
        raise ValueError('layer geometry must contain 1..48 curves')
    if type(default) not in (list,tuple) or len(default)!=len(curves):
        raise ValueError('default quota geometry')
    caps=[]
    for layer,curve in enumerate(curves):
        if type(curve) not in (list,tuple) or not 1 <= len(curve) <= MAX_LAYER_SLOTS+1:
            raise ValueError('prefix geometry must contain 1..257 entries')
        prior=0
        for score in curve:
            _integer(score,0,U64_MAX,'prefix score')
            if score<prior: raise ValueError('prefix scores must be nondecreasing')
            prior=score
        if curve[0]!=0: raise ValueError('prefix entry zero must equal zero')
        cap=len(curve)-1;caps.append(cap)
        _integer(default[layer],0,cap,'default quota')
    if sum(default)!=budget: raise ValueError('default quotas must sum to exact budget')
    return caps


def score_prefix_quota(curves, quota):
    """Checked additive captured-count score of a feasible uniform-slot quota."""
    if type(quota) not in (list,tuple): raise ValueError('default quota geometry')
    for value in quota: _integer(value,0,MAX_LAYER_SLOTS,'default quota')
    budget=sum(quota)
    _validate(curves,budget,quota)
    score=0
    for curve,q in zip(curves,quota): score=_add(score,curve[q])
    return score


def select_prefix_quotas(curves, budget, default):
    """Exact score max, then minimum default L1, then lexicographic quota.

    DP considers extendable exact-budget states only. Overflow of any considered
    feasible score rejects the objective, instead of saturating/wrapping it.
    O(L*K^2) transitions and O(L*K) states/backpointers. Prefix lexicographic ranks
    make tie comparisons constant-size; ranking adds O(L*K*log K) work.
    """
    caps=_validate(curves,budget,default)
    baseline=score_prefix_quota(curves,default)
    # State = score, L1 distance, prefix lex rank, previous slots, this quota.
    stages=[{0:(0,0,0,None,None)}]
    remaining=sum(caps)
    for layer,(curve,cap) in enumerate(zip(curves,caps)):
        remaining-=cap;candidates={}
        for spent,state in stages[-1].items():
            low=max(0,budget-spent-remaining)
            for q in range(low,min(cap,budget-spent)+1):
                total=spent+q
                candidate=(_add(state[0],curve[q]),state[1]+abs(q-default[layer]),state[2],q,spent)
                previous=candidates.get(total)
                # Higher score dominates; lower distance and lex(parent,q) break ties.
                key=(-candidate[0],candidate[1],candidate[2],candidate[3])
                if previous is None or key<(-previous[0],previous[1],previous[2],previous[3]):
                    candidates[total]=candidate
        ranked=sorted(candidates.items(),key=lambda item:(item[1][2],item[1][3]))
        current={}
        for rank,(total,candidate) in enumerate(ranked):
            score,distance,parent_rank,q,spent=candidate
            current[total]=(score,distance,rank,spent,q)
        stages.append(current)
    if budget not in stages[-1]: raise ValueError('exact budget infeasible')
    winner=stages[-1][budget];quotas=[];spent=budget
    for stage in reversed(stages[1:]):
        state=stage[spent];quotas.append(state[4]);spent=state[3]
    quotas.reverse()
    return {'score':winner[0],'baseline_score':baseline,'baseline_gain':winner[0]-baseline,
            'quotas':quotas,'total_slots':budget,'l1_distance_from_default':winner[1],
            'tie_rule':'maximum score; minimum total L1 from default; lexicographically smallest quota',
            'objective_scope':'one predeclared additive captured count; uniform MAXBLOB slots only'}
