"""Host-only launch/evidence refusal contracts; no model or speed measurements."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import prefill_recovery as subject


class Recovery(unittest.TestCase):
    def test_observers_do_not_change_current_algorithms(self):
        setup=subject.setup_reader.pilot(); original=copy.deepcopy(setup)
        checked=subject.environment(setup,subject.ROOT/'checked',True)
        clean=subject.environment(setup,subject.ROOT/'clean-0',False)
        differences={k for k in checked.keys()|clean.keys() if checked.get(k)!=clean.get(k)}
        self.assertEqual(differences,{'TMPDIR','XDG_CACHE_HOME','STRATA_MEASUREMENT_ROOT',
            'STRATA_TRACE','STRATA_SNAPSHOT_VERIFY','STRATA_SYCL_CACHE_LOAN_AUDIT',
            'UR_ENABLE_LAYERS','UR_LOG_VALIDATION','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'})
        self.assertEqual(setup,original)
        for k in ('STRATA_IQ2_PAIR','STRATA_MTP_BATCH_F16','STRATA_IQ_SINGLE_EXACT'):
            self.assertEqual(clean[k],setup['environment'][k])

    def test_changed_decode_setting_is_refused(self):
        setup=subject.setup_reader.pilot(); setup['environment']['STRATA_IQ2_PAIR']='0'
        with self.assertRaises(RuntimeError): subject.environment(setup,subject.ROOT/'clean-0',False)

    def rejected_run(self,fault_failure=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); run=root/'checked'; (run/'bench'/'one').mkdir(parents=True)
            samples=[{'prompt_ms':123456.7,'actual_read_tokens':117489,'failed_sample_retained':True}]
            (run/'bench'/'one'/'results.json').write_text(json.dumps({'results':{'prefill_only':{'requests':samples}}}))
            row={'stage':'collecting','bank':{'passed':True},'processes':[],
                'process_plans':[{'mode':'checked','root':str(run),'command':['fake'],'environment':{}}]}
            (root/'frozen.json').write_text(json.dumps(row)); result=root/'result.json'
            class Owner:
                active=None
                commands=[]
                def __init__(self,*args): pass
                def run(self,*args,**kwargs): return {'exit_code':0},None,None
            fault=dict(fresh_faults=[],boot_unchanged=True,fault_probe_successful=True)
            with patch.object(subject,'ROOT',root),patch.object(subject,'RESULT',result),\
                 patch.object(subject.gen,'committed',return_value=copy.deepcopy(row)),\
                 patch.object(subject,'current',return_value={}),\
                 patch.object(subject.supervision,'Owner',Owner),\
                 patch.object(subject.supervision,'closed',return_value=True),\
                 patch.object(subject.history,'official',side_effect=RuntimeError('route sample refused')),\
                 patch.object(subject.faults,'start_fault_scope',return_value=('boot','cursor')),\
                 patch.object(subject.faults,'fault_evidence',side_effect=RuntimeError('journal unavailable') if fault_failure else None,return_value=fault):
                self.assertEqual(subject.execute('run'),1)
            saved=json.loads(result.read_text())
            self.assertEqual(saved['stage'],'closed')
            self.assertEqual(saved['processes'][0]['requests'],samples)
            self.assertTrue(saved['processes'][0]['ownership_closed'])
            self.assertFalse(saved['processes'][0]['passed'])
            if fault_failure: self.assertIn('fault_probe_error',saved['processes'][0])

    def test_validation_failure_keeps_original_samples(self): self.rejected_run()
    def test_fault_probe_failure_still_closes_and_keeps_samples(self): self.rejected_run(True)


if __name__=='__main__': unittest.main()
