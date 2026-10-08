from pathlib import Path
import ast, datetime, hashlib, json

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
source=base/'run_owned_compact_layer_v01402_code32k.py'
target=base/'run_owned_layer_gpu_release_v01402_code32k.py'
audit=base/'layer-gpu-release-v01402-source-review'
assert not target.exists() and not audit.exists()
text=source.read_text()
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace("mode in ['compact2-layout1','layer2-ram-layout1']", "mode=='layer2-gpu-release-layout1'")
replace("if mode=='layer2-ram-layout1':\n    compact_gate=json.loads((base/'compact2-layout1-v01402-state-sequence/record.json').read_text())\n    assert compact_gate['passed'] and not compact_gate['active']", "layer_gate=json.loads((base/'compact-layer-v01402-state-sequence/record.json').read_text())\nassert layer_gate['passed'] and not layer_gate['active']")
replace("+list(base.glob('owned-layer2-ram-layout1-v01402-code32k-*/record.json')):", "+list(base.glob('owned-layer2-ram-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-layer2-gpu-release-layout1-v01402-code32k-*/record.json')):")
replace("if mode=='layer2-ram-layout1':env.update(STRATA_PREFILL_LAYER_MAJOR='2')", "env.update(STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_DRAFT_VERIFY='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='32767',STRATA_PREFILL_LAYER_MAJOR_R_INPLACE='1')")
replace("['STRATA_PREFILL_ATTN_BATCH','STRATA_GDN_KEYHEAD','STRATA_GDN_KEYHEAD_TUNED','STRATA_PREFILL_RELEASE_CACHE','STRATA_PREFILL_RELEASE_DRAFT','STRATA_PREFILL_LAYER_MAJOR_R_GPU','STRATA_PREFILL_LAYER_MAJOR_R_INPLACE','STRATA_PROMPT_ATTN_XMX']", "['STRATA_PREFILL_ATTN_BATCH','STRATA_GDN_KEYHEAD','STRATA_GDN_KEYHEAD_TUNED','STRATA_PREFILL_RELEASE_CACHE','STRATA_PROMPT_ATTN_XMX']")
start=text.index("    record['scope']='Private32K processing-order")
end=text.index('\n',start)
text=text[:start]+"    record['scope']='Private32K layer-major experiment after three compact and three RAM-row full-state/head controls. Keep all32767 batched residual rows on GPU; reuse the8192-row original residual scratch prefix and allocate only the remaining rows. Suspend MTP decode-only experts/head from virtual-memory backing, retaining its K/V/dense state; retire graphs, restore exact immutable RAM bytes at the same VAs and wait before decode. Main expert cache remains backed. Payload verification is enabled before/after this captured diagnostic/state step and excluded from later timing. Same e82fc5 executable and ordered model arithmetic. Match all66state parts,248320 head floats,64IDs and every logprob to the completed baseline; reject timing on any mismatch. No full256K/adoption/hang-prevention claim.'"+text[end:]
replace("record['source_review_sha256']=hashlib.sha256((base/'compact-layer-v01402-source-review/record.json').read_bytes()).hexdigest()", "record['source_review_sha256']=hashlib.sha256((base/'layer-gpu-release-v01402-source-review/record.json').read_bytes()).hexdigest()")
ast.parse(text)
target.write_text(text)
audit.mkdir(mode=0o700)
review={
    'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope':'Source/lifetime review for all-GPU residuals and MTP decode-only release. Not full-context memory-budget or runtime UB proof.',
    'binary_sha256':'e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323',
    'parent_source_review_sha256':hashlib.sha256((base/'compact-layer-v01402-source-review/record.json').read_bytes()).hexdigest(),
    'added_environment':{'STRATA_PREFILL_RELEASE_DRAFT':'1','STRATA_PREFILL_DRAFT_VERIFY':'1','STRATA_PREFILL_LAYER_MAJOR_R_GPU':'32767','STRATA_PREFILL_LAYER_MAJOR_R_INPLACE':'1'},
    'review':[
        'Exactly32767 batched rows precede the held-out input token for the fixed32768-token request. All rows fit one of the four original chunks, including8191 tail. Inplace reuses original residual scratch for the8192 first rows; remaining24575 rows receive a distinct allocation. Partial GPU prefixes are chunk-rounded, though none is used here.',
        'For all-GPU rows, hand_in points to device data and run_impl chooses its matching GPU row. Inplace R is the same row, so input upload/output download are skipped; embedding and child scratch stay separate. No memcpy is submitted with overlapping source/destination for the same rows.',
        'Last-layer MTP inputs are copied into stable Rin before replay. Graphs do not capture the changing R_rows address. Prefill::draft_kv binds stable drafter K/V, and its queued work is ordered/drained before temporary residual storage is reclaimed.',
        'MTP release verifies complete immutable host sources, waits all registered device queues, discards backend graphs before physical unmap, and marks partially successful suspension as suspended. Expert/head VAs remain reserved; dense weights, persistent state and KV remain backed.',
        'Restoration remaps full expert/head backing, checks same virtual addresses, copies immutable bytes, waits, optionally checks every payload byte and then clears suspended state. Decode recaptures graphs against live backing. The temporary layer cache/device residual allocation are freed before MTP restore.',
        'Main cache release is absent in this step; snapshot-vs-RAM and partial main-cache release remain separate experiments. Required device residual size is checked against free VRAM, and all first-use/state runs compare complete persistent state/head/output before timing.',
        'An older full262144 repeated-prefill budget failure can mask its primary error with destructor restoration failure. No claim that this32K experiment fixes that path; full occupancy/repeats/restore/clipped tail/refusal/later-valid remain pending.'
    ],
    'parent_controller_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'controller_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'preparer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
}
(audit/'record.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'controller':str(target),'review':str(audit/'record.json')},indent=2))
