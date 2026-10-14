#!/usr/bin/env python3
"""Materialize reviewed diagnostic diffs in memory; never write product files."""
import difflib
import hashlib
import json
import re
from pathlib import Path
from check_gemm_orientation import P, D, sha


def main():
    original=D/'layer-major-boundary-audit-v1/source.diff'
    assert sha(original)=='f651698b8b0f1efe72393d5dc136b9e2a9272079f7446a6203cb1d2d548bdb89'
    lines=original.read_text().splitlines(True)
    base={};modified={};i=0
    while i<len(lines):
        assert lines[i].startswith('--- ')
        old=lines[i][4:].strip();i+=1
        assert lines[i].startswith('+++ b/')
        name=lines[i][6:].strip();i+=1
        source='' if old=='/dev/null' else (P/name).read_text()
        before=source.splitlines(True);after=[];cursor=0
        while i<len(lines) and not lines[i].startswith('--- '):
            match=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',lines[i]);assert match
            start=max(0,int(match[1])-1);i+=1
            removed=[];added=[]
            while i<len(lines) and not lines[i].startswith(('@@ ','--- ')):
                line=lines[i];assert line[0] in ' +-'
                if line[0]!='+':removed.append(line[1:])
                if line[0]!='-':added.append(line[1:])
                i+=1
            assert start>=cursor and before[start:start+len(removed)]==removed
            after.extend(before[cursor:start]);after.extend(added);cursor=start+len(removed)
        after.extend(before[cursor:]);base[name]=source;modified[name]=''.join(after)
    header='include/strata/prefill/layer_major_boundary_audit.hpp'
    s=modified[header]
    old='if (n <= 0 || pos < 0 || width != 10240 || layers != 48 || chunk <= 0)'
    new='if (n <= 0 || n > 262144 || pos < 0 || pos > 262144 || n > 262144-pos ||\n            width != 10240 || layers != 48 || chunk <= 0 || chunk > 8192)'
    assert s.count(old)==1;s=s.replace(old,new)
    old='''                for (int64_t i = 0; i < width_; ++i)
                    e[i] = half_ ? strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(scratch_[size_t(i)])) : scratch_[size_t(i)];'''
    new='''                if (half_) for (int64_t i = 0; i < width_; ++i)
                    e[i] = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(scratch_[size_t(i)]));
                else std::memcpy(e, scratch_.data(), size_t(width_)*4);'''
    assert s.count(old)==1;s=s.replace(old,new);modified[header]=s
    name='src/prefill/layer_major.inc';s=modified[name]
    old='''    if (audit_switch && std::strcmp(audit_switch, "1") == 0)
        audit.reset(new detail::LayerMajorBoundaryAudit(n, pos0, D, g.n_layers, m.T,
            plan.dtype == detail::LayerMajorResidualDtype::fp16));'''
    new='''    if (audit_switch && std::strcmp(audit_switch, "1") == 0) {
        // Even the parser's maximum64 rows, metadata and scratch fit below4MiB.
        // Account this optional host owner before allocation, retaining8GiB.
        if (!core::conversation_memory_admit(core::conversation_available_memory(),
                detail::layer_major_add(host_need, 4ull << 20), host_headroom)) {
            err = "prefill boundary audit: diagnostic storage exceeds RAM headroom"; return false;
        }
        audit.reset(new detail::LayerMajorBoundaryAudit(n, pos0, D, g.n_layers, m.T,
            plan.dtype == detail::LayerMajorResidualDtype::fp16));
    }'''
    assert s.count(old)==1;s=s.replace(old,new);modified[name]=s
    patch=''.join(''.join(difflib.unified_diff(base[n].splitlines(True),modified[n].splitlines(True),
        fromfile='a/'+n if base[n] else '/dev/null',tofile='b/'+n)) for n in modified)
    out=D/'layer-major-boundary-audit-v2';assert not out.exists();out.mkdir(mode=0o700)
    (out/'source.diff').write_text(patch)
    pins=dict(status='PREPARED_UNAPPLIED_UNBUILT_UNTESTED',original_diff_sha256=sha(original),
        recipe_sha256=sha(Path(__file__)),source_diff_sha256=hashlib.sha256(patch.encode()).hexdigest(),
        files={n:dict(base_sha256=hashlib.sha256(base[n].encode()).hexdigest(),
                      proposed_sha256=hashlib.sha256(modified[n].encode()).hexdigest(),new_file=not bool(base[n])) for n in modified},
        changes='Main geometry/overflow guards, FP32 memcpy preserves bits, optional4MiB host admission. '
                'Original proposal and failed numerical gate unchanged. No product mutation or measurement.')
    (out/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    print(json.dumps(dict(prepared=True,product_changed=False,source_diff_sha256=pins['source_diff_sha256'])),flush=True)


if __name__=='__main__':
    main()
