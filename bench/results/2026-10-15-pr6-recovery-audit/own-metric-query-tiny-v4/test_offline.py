"""Main-run offline source/status fixtures. SOURCE ONLY, UNRUN by worker.
Repository LICENSE. No compiler, runtime, device or subprocess use.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import unittest
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
sys.path.insert(0,str(D))
from status_reader import reconcile
from synthetic_fixtures import fixture,encode,resequence

class Status(unittest.TestCase):
    def setUp(self):
        self.rows,self.pid,self.query,self.group,self.catalog=fixture()
        self.codes=json.loads((D/'result_codes.json').read_text())
    def read(self,text=None):
        return reconcile(encode(self.rows) if text is None else text,self.pid,self.query,
                         self.group,self.catalog,time.monotonic()+10,self.codes)
    def test_valid_exact_reconciliation(self):
        result=self.read()
        self.assertTrue(result['proved']); self.assertEqual(result['samples'],2)
        self.assertEqual(result['instances'],[11,12]); self.assertEqual(result['pending_polls'],2)
    def test_failure_with_exit_zero_and_csv_still_holds(self):
        row=next(r for r in self.rows if r.get('site')=='data-read' and r['phase']=='result')
        row.update(code=0x78000003,hex='0x78000003',name='ZE_RESULT_ERROR_UNSUPPORTED_FEATURE',data=None)
        with self.assertRaises(ValueError): self.read()
    def test_missing_result_even_with_complete_csv(self):
        self.rows.remove(next(r for r in self.rows if r.get('site')=='data-read' and r['phase']=='result'))
        resequence(self.rows)
        with self.assertRaises(ValueError): self.read()
    def test_missing_whole_calculation_with_repaired_sequences(self):
        call=next(r['call'] for r in self.rows if r.get('site')=='calculate-read')
        self.rows=[r for r in self.rows if r.get('call')!=call]
        for r in self.rows:
            if r.get('call',0)>call: r['call']-=1
        self.rows[-1]['calls']-=1; resequence(self.rows)
        with self.assertRaises(ValueError): self.read()
    def test_truncated_line(self):
        with self.assertRaises(ValueError): self.read(encode(self.rows)[:-7])
    def test_missing_close(self):
        with self.assertRaises(ValueError): self.read(encode(self.rows[:-1]))
    def test_exact_symbolic_numeric_agreement(self):
        next(r for r in self.rows if r['phase']=='result')['name']='ZE_RESULT_NOT_READY'
        with self.assertRaises(ValueError): self.read()
    def test_missing_sample(self):
        self.query['rows'].append(dict(global_instance_id=13))
        with self.assertRaises(ValueError): self.read()
    def test_partial_multisample_calculation(self):
        next(r for r in self.rows if r.get('site')=='calculate-read' and r['phase']=='result')['data']['samples']=2
        with self.assertRaises(ValueError): self.read()
    def test_wrong_catalog(self):
        self.catalog[5]['result_type']=2
        with self.assertRaises(ValueError): self.read()
    def test_interleaved_status(self):
        with self.assertRaises(ValueError): self.read('partial native line '+encode(self.rows))

# Exact unified-diff applicator in memory: no fuzzy matching, Git, or file edits.
def exact_apply(before,patch):
    lines=before.splitlines(True); out=[]; cursor=0
    chunks=patch.splitlines(True); i=2
    while i<len(chunks) and chunks[i].startswith('@@'):
        m=re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n',chunks[i])
        if not m: raise ValueError('bad hunk')
        start=int(m[1])-1; old_count=int(m[2] or 1); new_count=int(m[4] or 1)
        if start<cursor: raise ValueError('overlap')
        out.extend(lines[cursor:start]); cursor=start; old_used=new_used=0; i+=1
        while i<len(chunks) and chunks[i][:1] in (' ','+','-') and not chunks[i].startswith(('--- ','+++ ')):
            flag,text=chunks[i][0],chunks[i][1:]
            if flag in (' ','-'):
                if cursor>=len(lines) or lines[cursor]!=text: raise ValueError('exact context mismatch')
                cursor+=1; old_used+=1
            if flag in (' ','+'): out.append(text); new_used+=1
            i+=1
        if (old_used,new_used)!=(old_count,new_count): raise ValueError('hunk count')
    out.extend(lines[cursor:]); return ''.join(out)

class Source(unittest.TestCase):
    def test_exact_patch_and_every_selected_metric_call_wrapped(self):
        path=Path('/home/yayoi/ghq/github.com/intel/pti-gpu/tools/unitrace/src/levelzero/ze_collector.h')
        raw=path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),'6ee0130e2b03918a5e4bb5d310dacb94e5ba06bf67c21a23d3134dc406aa9122')
        patch=(D/'pti-query-status.patch').read_text()
        patched=exact_apply(raw.decode(),patch)
        expected=json.loads((D/'patched-source-sha256.json').read_text())
        self.assertEqual(hashlib.sha256(patched.encode()).hexdigest(),expected['tools/unitrace/src/levelzero/ze_collector.h'])
        self.assertIsNone(re.search(r'ZE_FUNC\(zet(?:Metric|ContextActivateMetric|CommandListAppendMetric)\w*\)\(',patched))
        sites=json.loads((D/'call-sites.json').read_text())
        for site in sites:
            self.assertIn('OwnMetricAudit::call("'+site['site']+'", "'+site['api']+'"',patched)
        self.assertIn('OwnMetricAudit::safe_finalize();',patched)
        self.assertIn('OwnMetricAudit::safe_query(OwnMetricAudit::id(query));',patched)
        self.assertIn('OwnMetricAudit::completed(',patched)
    def test_standalone_imports_do_not_launch_or_import_device_packages(self):
        for name in ('status_reader','collector_identity','main_controller','synthetic_fixtures'):
            spec=importlib.util.spec_from_file_location('offline_'+name,D/(name+'.py'))
            module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertFalse(any(n=='torch' or n.startswith(('dpctl','pyze','numba')) for n in sys.modules))
    def test_no_environment_write_in_new_collector_header(self):
        header=(D/'metric_query_audit.h').read_text()
        self.assertNotRegex(header,r'\b(?:setenv|unsetenv|putenv)\s*\(')
        self.assertIn('std::_Exit(125)',header)
        self.assertIn('missing_loader_symbol',header)

if __name__=='__main__': unittest.main()
