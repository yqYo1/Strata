"""Main serial observer fixtures and actual compiled caller linkage proof."""
import fcntl
import importlib.util
import json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
A = B / 'xe-native-dequant-batch-census-observer-source-v2'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('qualified_owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
HARNESS = '''import importlib.util, json, sys, unittest
from collections import Counter
from pathlib import Path
a, ledger = map(Path, sys.argv[1:])
sys.path.insert(0, str(a))
s = importlib.util.spec_from_file_location('observer_fixtures', a/'selfcheck.py')
v = importlib.util.module_from_spec(s); s.loader.exec_module(v)
v.Selfcheck.ledger = staticmethod(v.load_ledger(ledger))
class Results(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.subcases=[]
    def addSubTest(self,test,subtest,error):
        self.subcases.append(dict(method=test._testMethodName,parameters=str(subtest),passed=error is None))
        super().addSubTest(test,subtest,error)
r=unittest.TextTestRunner(verbosity=2,resultclass=Results).run(unittest.defaultTestLoader.loadTestsFromTestCase(v.Selfcheck))
counts=Counter(x['method'] for x in r.subcases)
assert r.wasSuccessful() and r.testsRun==6 and len(r.subcases)==47
assert counts==dict(test_all_context_corruptions=20,test_rejected_structures=20,test_rejected_independent_pins=7)
print(json.dumps(dict(passed=True,unittest_methods=r.testsRun,positive_cases=3,negative_cases=len(r.subcases),total_cases=50,negative_subcases=r.subcases)))
'''

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B/'xe-native-dequant-batch-census-observer-cpu-root-v2'
    m.require(not out.exists(), 'immutable output')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             controller_sha256=m.sha(__file__), owner_sha256=m.sha(O),
             scope='Actual V2 parser adversarial fixtures and archive undefined references; no GPU/model or performance proof',
             gpu_executed=False, model_executed=False, adopted=False,
             source_pins={}, build_proofs={}, asan_proofs={}, linkage={},
             raw_log_budget_bytes=64 << 20, wall_budget_seconds=180)
    def save():
        (out/'record.json').write_text(json.dumps(r, indent=2)+'\n')
    owner = m.Owner(out/'commands', save)
    r['commands'] = owner.commands
    save()
    def run(label, argv):
        entry, so, _ = owner.run(label, argv, ENV, W, wall=60, cpu=60,
                                text_cap=64<<20, total_cap=64<<20, rss_cap=2<<30)
        m.require(m.closed(entry) and entry['exit_code'] == 0, 'normal closed '+label)
        return so.read_text()
    try:
        review_path = B/'xe-native-dequant-batch-census-observer-main-review-v2.json'
        review = json.loads(review_path.read_text())
        m.require(review['full_report_parser_patch_selfcheck_manifest_read'] and review['all_input_output_pins_verified'], 'Main full source review')
        m.require(all(m.sha(Path(p)) == h for p,h in review['source_artifact_pins'].items()), 'exact reviewed source artifact')
        r['review_sha256'] = m.sha(review_path)
        r['source_pins'] = {str(A/n):m.sha(A/n) for n in ('batch_census.py','selfcheck.py')}
        ledger = B/'xe-source-llvm-census32k-v3/transfer_ledger.py'
        m.require(m.sha(ledger) == '7e5e37467208c229eeda3c7367fa16f506c11fe502ad93b28bbcc70fa4e9ab6d', 'qualified independent H2D parser')
        r['source_pins'][str(ledger)] = m.sha(ledger)
        for mode in ('free','contrib-llvm'):
            for suffix,key in [('full-build-root-v21','build_proofs'),('asan-full-build-root-v18','asan_proofs')]:
                proof = B/('xe-e32-'+mode+'-'+suffix)/'record.json'
                p = json.loads(proof.read_text())
                m.require(p['passed'] and p['complete'] and not p['active'] and all(m.closed(c) for c in p['commands']), 'closed actual '+mode+' '+suffix)
                m.require(all(m.sha(W/n) == h for n,h in p['source_pins'].items()), 'current source matches '+mode)
                m.require(m.sha(Path(p['binary']['path'])) == p['binary']['sha256'], 'current binary matches '+mode+' '+suffix)
                r[key][mode] = m.sha(proof)
            archive = W/'build'/mode/'libstrata_prefill.a'
            m.require(archive.is_file(), 'actual prefill archive '+mode)
            archive_hash = m.sha(archive)
            symbols = run('prefill-undefined-symbols-'+mode, ['/usr/bin/nm','-C','-u',str(archive)])
            names = ['strata::kernels::iq_dequant_gu_group_f16(', 'strata::kernels::iq_dequant_down_group_f16(']
            selected = [line for line in symbols.splitlines() if any(name in line for name in names)]
            m.require(len(selected) == 2 and all(any(name in line for line in selected) for name in names), 'both group APIs retained in actual caller '+mode)
            m.require(m.sha(archive) == archive_hash, 'archive stable '+mode)
            r['linkage'][mode] = dict(path=str(archive), sha256=archive_hash, undefined_group_references=selected)
        text = run('observer-adversarial-fixtures', ['/usr/bin/python3','-c',HARNESS,str(A),str(ledger)])
        r['fixtures'] = json.loads(text)
        m.require(r['fixtures']['passed'] and r['fixtures']['total_cases'] == 50, 'actual complete fixtures')
        r['source_stable'] = all(m.sha(Path(p)) == h for p,h in r['source_pins'].items())
        m.require(r['source_stable'], 'observer inputs unchanged')
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed','error','source_stable','linkage')}))
    raise SystemExit(0 if r['passed'] else 1)
