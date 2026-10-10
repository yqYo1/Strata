"""Host-only classification of an immutable failed kernel interval."""
import fcntl
import hashlib
import json
from pathlib import Path
import re

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
SOURCE=B/'postfault-status-health-root-r2/record.json'
PIN='7d83b26c2bf260004e83d1c3bab81fe6b73d67a419b763cea458a26d25937ea6'
PATTERNS={
    'G2H_timeout':r'Timed out wait for G2H',
    'GuC_PC_query_failure':r'GuC PC query task state failed',
    'job_timeout_check':r'Check job timeout',
    'schedule_disable_failure':r'Schedule disable failed to respond',
    'reset_attempt':r'trying reset',
    'reset_queued':r'\breset queued\b',
    'reset_started':r'\breset started\b',
    'reset_done':r'\breset done\b',
    'timedout_job':r'Timedout job',
    'CAT_error':r'memory.*CAT error',
    'fault_response':r'Fault response',
}

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==PIN
    d=json.loads(SOURCE.read_text())
    assert d['active'] is False and d['passed'] is False and d['completed'] is True
    assert d['status_success'] is True and d['integer_exact_words']==16384
    rows=d['kernel_device_entries'];assert len(rows)==199
    counts={name:sum(bool(re.search(pattern,r['MESSAGE'],re.I)) for r in rows) for name,pattern in PATTERNS.items()}
    assert counts==dict(G2H_timeout=1,GuC_PC_query_failure=1,job_timeout_check=30,
                       schedule_disable_failure=1,reset_attempt=32,reset_queued=32,
                       reset_started=32,reset_done=32,timedout_job=31,CAT_error=0,fault_response=0)
    jobs=[r for r in rows if re.search(PATTERNS['timedout_job'],r['MESSAGE'],re.I)]
    assert all('guc_id=0' in r['MESSAGE'] and 'in no process [-1]' in r['MESSAGE'] for r in jobs)
    result=dict(original_receipt=dict(path=str(SOURCE),sha256=PIN,status='FAILED unchanged'),
                classifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                gpu_executed=False,model_executed=False,reset_by_controller=False,
                data_path_completed_exact=True,kernel_health_passed=False,
                kernel_entries=199,classification_patterns=PATTERNS,counts=counts,
                first_timestamp=rows[0]['__REALTIME_TIMESTAMP'],last_timestamp=rows[-1]['__REALTIME_TIMESTAMP'],
                dump_unchanged=d['dump_unchanged'],boot_unchanged=d['boot_unchanged'],
                attribution='no-process/guc_id0 do not establish old or new process ownership',
                classification_limit='32 logged reset sequences; not an independent hardware reset counter',
                original_regex_limit='enumerated only GuC PC query failure; complete original199entries also contain resets/job timeouts',
                decision='No further GPU/model probe; CPU/source work continues; no reset/rebind/service/dumpclear')
    out=B/'postfault-status-health-root-r2/kernel-classification.json'
    with out.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,derived_record=str(out),counts=counts)))
