"""Retain offline cache review and explicitly limited event-pool evidence."""
from pathlib import Path
import datetime,hashlib,json,shutil
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro=parent/'code32k/repro32k';out=repro/'same-process-main-cache-repeat-32k'
def digest(p):
    with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def copy(p,relative):
    q=out/relative;assert not q.exists();q.parent.mkdir(parents=True,exist_ok=True);assert p.stat().st_size<20*1024**2;shutil.copy2(p,q)
v2=json.loads((base/'main-cache-repeat32k-v01402-resource-analysis-v2-events/record.json').read_text())
v3=json.loads((base/'main-cache-repeat32k-v01402-resource-analysis-v3-event-pools/record.json').read_text())
assert not v2['active'] and not v2['passed'] and 'zeEventDestroy' in v2['error']
assert v3['passed'] and not v3['active']
assert all(p['live_native']['event_pool']['objects']==0 for p in v3['memory_phases'])
assert sum(c.get('zeEventPoolCreate',0) for c in v3['counts'].values())==0
for source,relative in [('main-cache-repeat32k-v01402-resource-analysis-v2-events','analysis/native-v2-incomplete-events'),
                        ('main-cache-repeat32k-v01402-resource-analysis-v3-event-pools','analysis/native-v3-event-pools'),
                        ('ur-v2-cache-source-review-20261008','source-review/runtime-cache')]:
    for p in sorted((base/source).rglob('*')):
        if p.is_file():copy(p,relative+'/'+str(p.relative_to(base/source)))
for n in ['analyze_main_cache_repeat32k_v01402_resources_v2_events.py','analyze_main_cache_repeat32k_v01402_resources_v3_event_pools.py',Path(__file__).name]:copy(base/n,'controllers/'+n)
old={}
for name in ['full-context-eager-verifier-serve','full-context-verifier-lazy-restore-serve']:
    p=base/name/'record.json';r=json.loads(p.read_text());e=r['environment']
    assert not r['active'] and not r['new_fault_messages']
    assert e['STRATA_PREFILL_RELEASE_CACHE']=='1' and e['STRATA_PREFILL_RELEASE_DRAFT']=='1'
    old[name]={'record_sha256':digest(p),'binary_sha256':r['binary_sha256'],
               'error':r['error'],'main_release_was_enabled':True,'mtp_release_was_enabled':True,
               'requests':[(x.get('name'),x.get('input_tokens'),x.get('lazy_first_request_comparison')) for x in r['requests']]}
review={'active':False,'passed':True,'gpu_access':False,'controller_sha256':digest(Path(__file__)),
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Offline source/API scope review. No new runtime, environment, GPU run or production adoption.',
        'native_v2_limitation':'Tracking only zeEventCreate misses counter-event extension creation. The unknown zeEventDestroy is a parser coverage failure, not evidence of an unowned event, GPU fault or code bug. No complete event-handle balance is claimed.',
        'native_v3_scope':'Track successful native standard EventPool create/destroy separately; there are zero such creations/objects at all24 markers. This does not assert no counter events/internal allocations.',
        'official_source_revision':'95ed2199718412903ef28d5c9c1c51af2efe15b2',
        'official_source_version_warning':'Current source is pinned and retained but not proven identical to installed oneAPI2026.1.1/UR0.12.0.',
        'source_facts':['Regular command-list release returns native handles to a per-context descriptor cache; borrow resets a regular list. addCommandList has a TODO for a size limit.',
                        'The examined v2 event providers retain freelist events and borrowed event pools. The reviewed v2 cache/provider files do not read UR_L0_DISABLE_EVENTS_CACHING; an installed-library string alone does not prove it affects this path.',
                        'Current provider_counter uses the core counter-event API when available, with a legacy zexCounterBasedEventCreate2 fallback. The recorded run resolves that extension; standard pool counts cannot cover its creation.'],
        'external_report':{'url':'https://github.com/intel/llvm/issues/23073','scope':'Windows A770 host-process private memory under graph finalization, not Linux B570 VRAM; no source-match/root-cause/mitigation claim for this host.'},
        'older_full_context':old,
        'capacity_limitation':'Both older full-context attempts already enabled main-cache and MTP release. Adding the same release cannot be claimed to solve their capacity/math failures. The eager case produces4IDs but differs in logprobs/whole head; the lazy case aborts. They use older distinct binaries and are not gold controls for the integrated candidate.',
        'next_work':'Obtain source matching the installed UR adapter and evaluate a narrowly scoped private regular-command-list cache limit or graph footprint reduction, with bounded logged32K correctness and matched quiet32K comparisons before full262144 gates. No unsupported cache switch is adopted.'}
(out/'source-review/runtime-cache-conclusions.json').write_text(json.dumps(review,indent=2)+'\n')
p=out/'README.md';p.write_text(p.read_text()+'''

The [offline runtime-cache follow-up](source-review/runtime-cache-conclusions.json)
retains official Intel UR source at95ed2199, with explicit installed-version
uncertainty. Its regular-list deleter returns a native list to a per-context
cache; borrowing resets it, and addCommandList has a TODO for a size limit.
This is consistent with the measured distinction between released UR buffers
and retained native objects, but does not attribute their resident size.

The [event-pool-only count](analysis/native-v3-event-pools/record.json) has zero
successful standard zeEventPoolCreate calls. The intermediate event-handle
parser stops at a destroy whose extension creation it did not cover; this is
an offline coverage failure, not a GPU/code fault. Counter-event extensions
are a separate path, so neither complete event balance nor no internal event
allocation is claimed. The [Windows A770 report](https://github.com/intel/llvm/issues/23073)
measures host private memory; it does not establish this B570 VRAM cause or a
cache-disable remedy. The examined v2 cache/provider source does not read that
switch, and an installed-library string is insufficient to adopt it.

Both older full-context attempts already enabled main and MTP cache release.
The eager attempt fails whole-head/logprob equality after four outputs; the
lazy attempt aborts. Their distinct older binaries remain negative evidence,
not integrated-candidate gold controls. Main release alone therefore cannot
be presented as resolving the full-context failure. Reducing retained native
graph/list resources remains an implementation candidate, not a measured fix.
''')
for d in [out,repro,parent/'code32k',parent]:
    p=d/'manifest.json';m=json.loads(p.read_text());m.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Retain pinned runtime-cache source and limited event-pool counts; distinguish older full-context math and allocation failures.',
    files={str(f.relative_to(d)):{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(d.rglob('*')) if f.is_file() and f!=p});p.write_text(json.dumps(m,indent=2)+'\n')
    for k,v in m['files'].items():
        f=d/k;assert f.stat().st_size==v['bytes'] and digest(f)==v['sha256']
    print(d.name,len(m['files']),flush=True)
