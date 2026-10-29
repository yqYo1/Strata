"""Project the committed pair-prefill change onto exact selected6e40 source.

Own MIT constructor. The live tree also contains unselected native observers
and proposals. They are NOT imported into this model comparison. Every actual
change comes from the named integration commit,which also changes the real
feature/sycl tree;the constructor is not a second source implementation.
"""
import difflib
import hashlib
from pathlib import Path
import subprocess

P=Path(__file__).resolve().parents[2]
BASE='6e40afc1beddd059ee518617f6c1efce1da14aca'
INTEGRATION='4e6dbe25a'
MODIFIED=['cmake/sycl_backend.cmake','include/strata/prefill/qsa_native16.hpp',
          'src/prefill/sycl/qsa_native16.cpp','src/prefill/prefill.cpp','src/prefill/layer_major.inc']
ADDED=['include/strata/kernels/qsa_pair_union.hpp','src/kernels/sycl/qsa_pair_union.cpp',
       'include/strata/kernels/qsa_pair_attention.hpp','src/kernels/sycl/qsa_pair_attention.cpp',
       'include/strata/kernels/qsa_pair_prefix.hpp','src/kernels/sycl/qsa_pair_prefix.cpp']
CONFIGURATION=['tools/sycl/'+n+'.cpp' for n in ['host_transfer_probe','kv_copy_groups_host',
    'kv_copy_groups_bench','gemm_orientation_bench','projection_bench','moe_bench','gr_bench',
    'expert_contract','gemm_device_probe']]


def git(*args):
    return subprocess.check_output(['git',*args],cwd=P,timeout=30)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def project(base,parent,integrated):
    """Apply only changed spans;unchanged unique anchors locate them in BASE.

    Context may overlap but edited spans may not. No fuzz,untracked source or
    nearest-revision fallback. Ambiguity/conflict is an error. Exact inverse
    reconstruction proves that other selected source bytes stayed unchanged.
    """
    before=parent.decode().splitlines(keepends=True);after=integrated.decode().splitlines(keepends=True)
    original=base.decode();changes=[]
    opcodes=difflib.SequenceMatcher(None,before,after,autojunk=False).get_opcodes()
    for k,(tag,a,b,c,d) in enumerate(opcodes):
        if tag=='equal':continue
        left=opcodes[k-1][2]-opcodes[k-1][1] if k and opcodes[k-1][0]=='equal' else 0
        right=opcodes[k+1][2]-opcodes[k+1][1] if k+1<len(opcodes) and opcodes[k+1][0]=='equal' else 0
        found=None
        for radius in range(13):
            for pre in range(min(left,radius)+1):
                post=radius-pre
                if post>right:continue
                anchor=''.join(before[a-pre:b+post]);prefix=''.join(before[a-pre:a])
                if anchor and original.count(anchor)==1 and parent.decode().count(anchor)==1:
                    found=original.index(anchor)+len(prefix);break
            if found is not None:break
        if found is None:raise ValueError('pair-prefill edit has no exact unique selected-source anchor')
        old=''.join(before[a:b]);new=''.join(after[c:d])
        if original[found:found+len(old)]!=old:raise ValueError('selected preimage differs')
        changes.append((found,old,new))
    changes.sort(key=lambda r:r[0])
    if any(a+len(old)>b for (a,old,_),(b,_,_) in zip(changes,changes[1:])):
        raise ValueError('projected pair-prefill changes overlap')
    result=original
    for offset,old,new in reversed(changes):result=result[:offset]+new+result[offset+len(old):]
    delta=0;inverse=[]
    for offset,old,new in changes:
        inverse.append((offset+delta,new,old));delta+=len(new)-len(old)
    check=result
    for offset,new,old in reversed(inverse):
        if check[offset:offset+len(new)]!=new:raise ValueError('projected postimage differs')
        check=check[:offset]+old+check[offset+len(new):]
    if check!=original:raise ValueError('pair projection changed other selected bytes')
    return result.encode(),len(changes)


def overlays():
    """Exact current production additions plus five bounded selected overlays."""
    result={};proof={}
    for name in MODIFIED:
        base=git('show',BASE+':'+name);parent=git('show',INTEGRATION+'^:'+name)
        integrated=git('show',INTEGRATION+':'+name)
        if (P/name).read_bytes()!=integrated:raise ValueError('commit/review a fresh integration before changing '+name)
        raw,count=project(base,parent,integrated);result[name]=raw
        proof[name]=dict(base_sha256=sha(base),parent_sha256=sha(parent),
            integration_sha256=sha(integrated),projected_sha256=sha(raw),changed_spans=count,
            inverse_recovers_original=True)
    for name in ADDED:
        raw=git('show',INTEGRATION+':'+name)
        if raw!=(P/name).read_bytes():raise ValueError('qualified pair source changed: '+name)
        result[name]=raw;proof[name]=dict(source_commit=INTEGRATION,sha256=sha(raw))
    return result,proof
