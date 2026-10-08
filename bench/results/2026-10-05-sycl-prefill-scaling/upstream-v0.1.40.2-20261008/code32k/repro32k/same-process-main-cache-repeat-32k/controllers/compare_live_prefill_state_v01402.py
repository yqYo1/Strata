"""Compare every live byte while retaining hashes of rounded-page padding."""
import hashlib

def padding_ranges(part_index, size, upto, *, heads=2, head_dim=256, page=4):
    if upto <= 0 or heads != 2 or head_dim != 256 or page != 4:
        raise ValueError('comparison is pinned to positive B570/Qwen int8 geometry')
    if part_index < 6:
        return []
    kind=(part_index-6)%5
    if kind==4:
        expected=(upto//page+1)*128*4
        if size!=expected:raise ValueError('unexpected pooled/spare-row payload size')
        return []  # The moving indexer spare row is live and must match.
    cells=((upto+page-1)//page)*page
    row=head_dim if kind in [0,1] else (head_dim//64)*2
    if size!=cells*heads*row:raise ValueError('unexpected int8 payload size')
    if upto==cells:return []
    valid_in_last=upto%page
    base=(cells-page)*heads*page*row
    return [(base+(head*page+valid_in_last)*row,
             base+(head+1)*page*row) for head in range(heads)]

def hash_live(path, offset, size, ignored):
    digest=hashlib.sha256();cursor=0
    with open(path,'rb') as stream:
        for left,right in [*ignored,(size,size)]:
            if not 0<=cursor<=left<=right<=size:raise ValueError('invalid padding range')
            stream.seek(offset+cursor)
            remaining=left-cursor
            while remaining:
                data=stream.read(min(1048576,remaining))
                if not data:raise ValueError('truncated state payload')
                digest.update(data);remaining-=len(data)
            cursor=right
    return digest.hexdigest()

def compare_states(current, baseline, upto):
    a=current['parts'];b=baseline['parts']
    if len(a)!=66 or len(b)!=66:raise ValueError('expected all66 state parts')
    result={'different_raw_parts':[],'different_live_parts':[],
            'padding_only_different_parts':[],'part_evidence':[]}
    for p,q in zip(a,b):
        i=p['index']
        if i!=q['index'] or p['bytes']!=q['bytes']:raise ValueError('state metadata differs')
        ranges=padding_ranges(i,p['bytes'],upto)
        raw_equal=p['sha256']==q['sha256']
        live_equal=raw_equal
        detail={'part':i,'bytes':p['bytes'],'padding_ranges':ranges,'raw_equal':raw_equal}
        if not raw_equal:
            result['different_raw_parts'].append(i)
            if ranges:
                ah=hash_live(current['file'],p['offset'],p['bytes'],ranges)
                bh=hash_live(baseline['file'],q['offset'],q['bytes'],ranges)
                detail.update(live_sha256=ah,baseline_live_sha256=bh);live_equal=ah==bh
            if live_equal:result['padding_only_different_parts'].append(i)
            else:result['different_live_parts'].append(i)
        detail['live_equal']=live_equal;result['part_evidence'].append(detail)
    return result
