"""Keep the immutable terminal receipt and correct one inherited hash label separately."""
from pathlib import Path
import ast, datetime, hashlib, json

base=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=base/'archive_qsa_full256k_and_gemm_phase_preparation_v01402_v1.py'
run=base/'owned-qsa-reduce12-sg32-v01402-full256k-diagnostic-r1'
r=json.loads((run/'record.json').read_text());assert not r['active'] and r['healthy'] and r['math_gate_passed']
qsa_path=base/'qsa-reduce12-matched-quiet32k-v01402-sequence-v1/record.json'
poll_path=base/'poll-matched-quiet32k-v01402-sequence-v1/record.json'
assert r['qsa_matched_quiet32k_sequence_sha256']==sha(poll_path)!=sha(qsa_path)
qsa=json.loads(qsa_path.read_text());assert qsa['passed'] and not qsa['active']
assert [(x['mode'],x['repetition']) for x in qsa['steps']]==[('dd5-control',1),('qsa-reduce12',1),('qsa-reduce12',2),('dd5-control',2)]
for x in qsa['steps']:
    p=Path(x['receipt']);assert sha(p)==x['receipt_sha256']
    d=json.loads(p.read_text());assert not d['active'] and d['healthy'] and d['math_gate_passed']
controller=base/'run_owned_qsa_reduce12_full256k_v01402_v1.py'
c=controller.read_text()
assert c.index("quiet_path=base/'qsa-reduce12-matched-quiet32k-v01402-sequence-v1/record.json'")<c.index("out = base/f'owned-qsa-reduce12")
assert "quiet_terminal=json.loads(quiet_path.read_text())" in c
assert "quiet_path=base/'poll-matched-quiet32k-v01402-sequence-v1/record.json'" in c
assert "record['qsa_matched_quiet32k_sequence_sha256']=hashlib.sha256(quiet_path.read_bytes()).hexdigest()" in c
fix={'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'raw_terminal_receipt_sha256':sha(run/'record.json'),'executed_controller_sha256':sha(controller),
     'raw_field':'qsa_matched_quiet32k_sequence_sha256','raw_field_sha256':r['qsa_matched_quiet32k_sequence_sha256'],
     'raw_field_actually_identifies':str(poll_path),'actual_QSA_matched_sequence':str(qsa_path),
     'actual_QSA_matched_sequence_sha256':sha(qsa_path),'all_four_QSA_matched_jobs_terminal_and_receipt_hashes_validated':True,
     'scope':'Metadata-only inherited variable reuse: the correct QSA sequence was guarded before output-directory/model creation, but a later DD5 polling prerequisite reused quiet_path before the QSA-labelled digest assignment. Raw executed source/receipt retained unchanged; physical/numerical gates unaffected. No GPU rerun.'}
side=run/'actual-quiet-sequence-identity-v1.json';assert not side.exists();side.write_text(json.dumps(fix,indent=2)+'\n')
negative={'active':False,'CPU_only':True,'GPU_launched':False,'archive_directory_created':False,
          'rejected_archiver_sha256':sha(parent),'assertion':'raw QSA-labelled digest differs from actual QSA matched sequence',
          'raw_field_sha256':r['qsa_matched_quiet32k_sequence_sha256'],'expected_QSA_sequence_sha256':sha(qsa_path),
          'scope':'Archiver v1 rejected the metadata mismatch before directory creation or copy. Preserve raw receipt plus correction sidecar, rather than weakening physical qualification.'}
p=base/'qsa-full256k-archive-preflight-rejection-v1.json';assert not p.exists();p.write_text(json.dumps(negative,indent=2)+'\n')
s=parent.read_text()
old="assert r['qsa_matched_quiet32k_sequence_sha256']==hashlib.sha256((base/'qsa-reduce12-matched-quiet32k-v01402-sequence-v1/record.json').read_bytes()).hexdigest()"
assert s.count(old)==1
s=s.replace(old,"""identity=json.loads((run/'actual-quiet-sequence-identity-v1.json').read_text())
assert identity['passed'] and identity['raw_terminal_receipt_sha256']==hashlib.sha256((run/'record.json').read_bytes()).hexdigest()
assert identity['raw_field_sha256']==r['qsa_matched_quiet32k_sequence_sha256']
assert identity['actual_QSA_matched_sequence_sha256']==hashlib.sha256((base/'qsa-reduce12-matched-quiet32k-v01402-sequence-v1/record.json').read_bytes()).hexdigest()""")
s=s.replace("'existing-prefill-profiler-feasibility-v01402-source-review-v1.json',", "'existing-prefill-profiler-feasibility-v01402-source-review-v1.json','qsa-full256k-archive-preflight-rejection-v1.json',")
s=s.replace("'run_owned_dd5_phase32k_v01402_v2.py',Path(__file__).name]", "'run_owned_dd5_phase32k_v01402_v2.py','archive_qsa_full256k_and_gemm_phase_preparation_v01402_v1.py','prepare_archive_qsa_full256k_and_gemm_phase_preparation_v01402_v2.py',Path(__file__).name]")
s=s.replace("'qualified_dd5_full256k_receipt_sha256':r['qualified_dd5_full256k_receipt_sha256'],", "'qualified_dd5_full256k_receipt_sha256':r['qualified_dd5_full256k_receipt_sha256'],'QSA_quiet_sequence_identity_correction_sha256':sha(run/'actual-quiet-sequence-identity-v1.json'),")
s=s.replace('The main and production binaries remain unchanged.', 'The main and production binaries remain unchanged. The raw full-run receipt\'s QSA-labelled quiet-sequence digest accidentally identifies an earlier DD5 polling prerequisite because the executed controller reused a path variable. The correct completed QSA sequence was independently guarded before model creation. The raw receipt and executed source remain unchanged, with an [identity correction](terminal-qsa-full256k-r1/actual-quiet-sequence-identity-v1.json) specifying both digests and validating the four actual QSA sequence jobs. Archiver v1 rejected this mismatch before copying; its CPU rejection record is retained.')
ast.parse(s)
target=base/'archive_qsa_full256k_and_gemm_phase_preparation_v01402_v2.py'
assert not target.exists();target.write_text(s)
print(json.dumps({'prepared':True,'sidecar_sha256':sha(side),'controller_sha256':sha(target),'raw_terminal_receipt_unchanged':sha(run/'record.json')==fix['raw_terminal_receipt_sha256'],'GPU_rerun':False}))
