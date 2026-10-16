#!/usr/bin/env python3
"""Compare all journaled request inputs only; no native closure or speed claim."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
sys.path.insert(0,str(PROJECT/'tools'))
from strata_bench import verify_request_journal

MAX_BYTES=8<<20
MAX_ROWS=2048
MAX_STATE_BYTES=64<<10
MAX_REPORT_BYTES=4<<20
FIELDS=('phase','run','turn','max_tokens','thinking','request_sha256')
HEX=re.compile(r'[0-9a-f]{64}\Z')


def require(ok,message):
    if not ok:raise ValueError(message)


def snapshot(path,limit):
    with path.open('rb') as stream:data=stream.read(limit+1)
    require(len(data)<=limit,'input byte budget exceeded: '+str(path))
    return data


def digest(data):return hashlib.sha256(data).hexdigest()


def read_suite(path,record):
    path=Path(path)
    directory=path.parent if path.name=='requests.jsonl' else path
    journal=directory/'requests.jsonl';statepath=directory/'journal-state.json'
    record.update(directory=str(directory),journal=str(journal),state_path=str(statepath))
    data=snapshot(journal,MAX_BYTES);state=snapshot(statepath,MAX_STATE_BYTES)
    record.update(journal_sha256=digest(data),journal_bytes=len(data),state_sha256=digest(state),state_bytes=len(state))
    lines=data.splitlines()
    require(len(lines)<=MAX_ROWS,'journal row budget exceeded')
    # Verify the byte-bounded exact snapshots with the existing whole-file
    # verifier. Concurrent growth of the source cannot bypass the read budget.
    # These are copies of original journal/state, never fabricated completion.
    with tempfile.TemporaryDirectory(prefix='bench-request-identity-') as tmp:
        snap=Path(tmp)
        (snap/'requests.jsonl').write_bytes(data)
        (snap/'journal-state.json').write_bytes(state)
        record['verification']=verify_request_journal(snap)
    record['verification_of_bounded_snapshot']=True
    require(snapshot(journal,MAX_BYTES)==data and snapshot(statepath,MAX_STATE_BYTES)==state,
            'journal/state changed during validation')
    verify=record['verification']
    require(verify['valid'] and verify['complete'] and not verify['pending'],'journal chain/completion/pending gate')
    terminal=json.loads(state)
    require(terminal.get('terminal')=='complete' and terminal.get('complete') is True and not terminal.get('pending'),
            'terminal journal state is incomplete')
    rows=[json.loads(line) for line in lines]
    require(rows and rows[0].get('kind')=='identity' and sum(r.get('kind')=='identity' for r in rows)==1,
            'missing/duplicate/out-of-order suite identity')
    require(all(r.get('kind') in ('identity','start','result') for r in rows),'unexpected/interrupted journal kind')
    starts=[];seen=set()
    for row in rows:
        if row.get('kind')!='start':continue
        require(all(k in row for k in FIELDS),'request parameters/hash missing')
        item={k:row[k] for k in FIELDS}
        require(isinstance(item['phase'],str) and 0<len(item['phase'])<=256,'invalid request phase')
        require(all(v is None or (type(v) is int and abs(v)<=2**63-1) for v in (item['run'],item['turn'])),'invalid run/turn key')
        require(type(item['max_tokens']) is int and 1<=item['max_tokens']<=2**63-1 and type(item['thinking']) is bool,
                'invalid request parameters')
        require(isinstance(item['request_sha256'],str) and HEX.fullmatch(item['request_sha256']),'invalid actual request hash')
        key=(item['phase'],item['run'],item['turn'])
        if key in seen:record.setdefault('duplicate_request_keys',[]).append(list(key))
        seen.add(key);starts.append(dict(journal_index=row['index'],**item))
    require(starts,'journal has no request starts')
    record['start_count']=len(starts)
    return starts


def compare(left,right):
    report=dict(schema='bench-request-input-identity-v1',status='FAIL',inputs_match=False,
                claim='All journaled request inputs only; no source/configuration/native closure/quality/performance/adoption approval.',
                limits=dict(journal_bytes_each=MAX_BYTES,rows_each=MAX_ROWS,state_bytes_each=MAX_STATE_BYTES,report_bytes=MAX_REPORT_BYTES),
                verifier_source_sha256=digest((PROJECT/'tools/strata_bench.py').read_bytes()),
                left={},right={},samples=[],errors=[])
    sides=[]
    for name,path in [('left',left),('right',right)]:
        try:sides.append(read_suite(path,report[name]))
        except (OSError,ValueError,KeyError,TypeError,UnicodeError) as error:
            report['errors'].append(dict(side=name,error=repr(error)));sides.append(None)
    if any(side is None for side in sides):return report
    a,b=sides
    for name in ('left','right'):
        if report[name].get('duplicate_request_keys'):report['errors'].append(dict(side=name,error='duplicate request key',keys=report[name]['duplicate_request_keys']))
    for i in range(max(len(a),len(b))):
        one=a[i] if i<len(a) else None;two=b[i] if i<len(b) else None
        differences={k:dict(left=one.get(k) if one else None,right=two.get(k) if two else None)
                     for k in FIELDS if one is None or two is None or one[k]!=two[k]}
        report['samples'].append(dict(position=i,left=one,right=two,differences=differences,match=not differences))
    if len(a)!=len(b):report['errors'].append(dict(error='missing/extra request starts',left_count=len(a),right_count=len(b)))
    if any(not row['match'] for row in report['samples']):report['errors'].append(dict(error='ordered request input mismatch'))
    if not report['errors']:report.update(status='REQUEST_INPUTS_MATCH',inputs_match=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left');parser.add_argument('right');parser.add_argument('output')
    args=parser.parse_args(argv)
    target=Path(args.output)
    # Reserve before reads, so existing reports are never overwritten on failure.
    with target.open('x',encoding='utf-8') as stream:
        report=compare(args.left,args.right)
        encoded=json.dumps(report,indent=2,ensure_ascii=False)+'\n'
        require(len(encoded.encode())<=MAX_REPORT_BYTES,'report byte budget exceeded')
        stream.write(encoded)
    print(report['status'])
    return 0 if report['inputs_match'] else 1


if __name__=='__main__':sys.exit(main())
