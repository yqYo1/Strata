"""Prepare an isolated source candidate; no compilation or GPU execution."""
from pathlib import Path
import datetime,difflib,hashlib,json,re
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
original=root/'sycl/src/kernels/cuda/kv_stream.dp.cpp'
out=root/'build-sycl-kv-stream-safe-v2-20261008/source';assert not out.exists();out.mkdir(parents=True)
receipt=base/'kv-stream-safe-v01402-source-candidate-v2';receipt.mkdir(mode=0o700)
text=original.read_text();changed=text
before='''                m.slot_stamp[sl] = epoch;
                m.slot_ref[sl] = 1;'''
after='''                // Queries can hit the same page from different work-items.
                // Publish identical hit metadata through atomic stores too.
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                    m.slot_stamp[sl]).store(epoch);
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                    m.slot_ref[sl]).store(1);'''
assert changed.count(before)==1;changed=changed.replace(before,after)
before='''    const unsigned long long* u = reinterpret_cast<const unsigned long long*>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];'''
after='''    // The protocol packs 64-bit counters into this int32_t byte copy.
    // memcpy reads their representation without type-punning or alignment assumptions.
    static_assert(sizeof(r.misses) == 8 && sizeof(r.lookups) == 8 && sizeof(r.calls) == 8);
    std::memcpy(&r.misses, c + 4, sizeof(r.misses));
    std::memcpy(&r.lookups, c + 6, sizeof(r.lookups));
    std::memcpy(&r.calls, c + 8, sizeof(r.calls));'''
assert changed.count(before)==1;changed=changed.replace(before,after)
assert '#include <cstring>' not in changed;changed=changed.replace('#include <cstdlib>','#include <cstdlib>\n#include <cstring>')
property_decl=re.compile(r'        auto exp_props = sycl::ext::oneapi::experimental::properties\{\n            sycl::ext::oneapi::experimental::use_root_sync\};\n\n')
changed,n=property_decl.subn('',changed);assert n==4,n
assert changed.count('exp_props,')==4
changed,nargs=re.subn(r'\bexp_props,\s*', '', changed);assert nargs==4,nargs
assert 'exp_props' not in changed and 'use_root_sync' not in changed
assert '[[sycl::reqd_sub_group_size(32)]]' in changed and 'constexpr int RT = 1024;' in changed
p=out/'kv_stream.dp.cpp';p.write_text(changed)
patch=''.join(difflib.unified_diff(text.splitlines(keepends=True),changed.splitlines(keepends=True),fromfile=str(original),tofile=str(p)))
(receipt/'candidate.diff').write_text(patch)
# A legal sorted selection whose shared hit belongs to two different lanes.
selections=[[16,32],[4,16]];writes={}
for query,ids in enumerate(selections):
    assert ids==sorted(ids)
    for lane,cell in enumerate(ids):
        page=cell//4
        if lane>0 and ids[lane-1]//4==page:continue
        writes.setdefault(page,[]).append({'query':query,'lane':lane})
conflicts={page:owners for page,owners in writes.items() if len({o['lane'] for o in owners})>1}
assert conflicts=={4:[{'query':0,'lane':0},{'query':1,'lane':1}]}
record={'active':False,'source_prepared':True,'compiled':False,'gpu_tested':False,'adopted':False,
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_source':str(original),'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'candidate_source':str(p),'candidate_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'scope':'Private source-only fixes before exercising existing KV streaming for full context. No existing binary, production source/default or runtime is changed. Performance inputs remain >=32768; full262144 gates mandatory.',
        'changes':['Use relaxed device-scope global atomic stores for hit stamps/reference bits that different query lanes can share. The following full work-group barrier separates later victim scanning.',
                   'Read packed64-bit host counters with memcpy instead of aliasing an int32_t array as unsigned long long.',
                   'Remove four unused root-sync properties from reset/resolve/copy/ring launches. Preserve kernel bodies/arithmetic, WG1024 resolve and required SG32, and queue/order/ownership.'],
        'sorted_hit_counterexample':{'cell_ids':selections,'conflicting_page_owners':conflicts,'scope':'Static possible competing same-value stores across queries without an inter-query barrier. Not a claim this caused previous hangs: earlier32K records used KV mode0.'},
        'next_gates':['Pin actual compiled source/archive/build command; compile only the candidate KV translation unit against accepted e82fc5 objects in a new private output.',
                      'Audit host-USM lifetime, arena cleanup, ring restore, staging/clipping and snapshot identity reconstruction before model use.',
                      'First bounded owned normal-MTP32K run with max-context262144, kv-resident32768 and flushed UR/LevelZero validation. All live/raw main state, complete first head,64IDs/logprobs and MTP counts must match hard control; record true physical residency and bytes.',
                      'Do not count diagnostics/captures as speed. Then matched clean32K-or-longer and full262144 occupancy/repeat/restore/clipped-tail/refusal/later-valid. No unsupported cache switch or global change.']}
(receipt/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'source_prepared':True,'compiled':False,'gpu_tested':False,'candidate_sha256':record['candidate_sha256'],'hit_conflicts':conflicts,'root_properties_removed':n},indent=2))
