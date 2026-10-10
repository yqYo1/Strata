import hashlib, importlib.util, json, re, struct, subprocess, sys
from pathlib import Path
B=Path(__file__).parent
binary=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir()
spec=importlib.util.spec_from_file_location('census',B/'ple_offline_census_v2.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
checks=[]
def checked(name,condition):
    assert condition,name;checks.append(name)
def run(argv,expected=0,needle=None):
    r=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    assert r.returncode==expected,(argv,r.returncode,r.stderr.decode()[:500])
    if needle:assert needle in r.stderr.decode() and not r.stdout
    return json.loads(r.stdout) if r.stdout else None
gold=(Path(c.DEFAULT_ROOT)/'src/kernels/ple_oracle_vectors.inc').read_text()
count=0
for n in range(6):
    values=[]
    for what in ('tokens','prev','rows'):
        body=re.search(r'case'+str(n)+'_'+what+r'\[\d+\]\s*=\s*\{(.*?)\};',gold,re.S).group(1)
        values.append(list(map(int,re.findall(r'-?\d+',body))))
    tokens,prev,rows=values
    for i,t in enumerate(tokens):
        checked('python_external_golden_%d_%d'%(n,i),c.independent_rows([prev[2*i],prev[2*i+1],t],2,(-1,-1))==rows[16*i:16*i+16]);count+=16
checked('all304_independent_golden_rows',count==304)
tiny=out/'tiny.txt';tiny.write_text('0 7 11 248044 19 23 -17 29 0 -2147483648 2147483647\n')
def census(input_,rows,prefix,extra=()):
    return ['/usr/bin/python3',str(B/'ple_offline_census_v2.py'),'--input',str(input_),'--fixture','tiny','--prefix',str(prefix),'--oracle-rows',str(rows),'--oracle-bin',str(binary),*extra]
for n,prev in enumerate(((7,11),(-17,248044))):
    rows=out/f'tiny-{n}.rows';args=['--prev',*map(str,prev)]
    r=run([str(binary),str(tiny),'11',str(rows),*args]);s=run(census(tiny,rows,11,args))
    checked('negative_own_token_EOS_null_wrap_history_'+str(n),r['row_count']==s['all_oracle_rows_compared']==176)
    if n==0:
        clean=rows.read_bytes()
        for label,offset in [('first',0),('last',len(clean)-4)]:
            bad=bytearray(clean);old=struct.unpack_from('<I',bad,offset)[0];struct.pack_into('<I',bad,offset,(old+1)%c.NROWS)
            p=out/(label+'.badrows');p.write_bytes(bad);run(census(tiny,p,11),1,'oracle mismatch');checks.append(label+'_in_range_corruption_rejected')
        for label,data in [('short',clean[:-1]),('long',clean+b'\0')]:
            p=out/(label+'.badrows');p.write_bytes(data);run(census(tiny,p,11),1,'exact size mismatch');checks.append(label+'_size_rejected')
        run([str(binary),str(tiny),'11',str(rows)],1,'existing files refused');checks.append('existing_output_refused')
edge=[(i*397)%248044 for i in range(8208)];edge[8189:8196]=[11,0,248044,19,-17,2147483647,23]
ep=out/'edge.txt';ep.write_text(' '.join(map(str,edge))+'\n');streams=[]
for prefix in (8191,8192,8193,8208):
    rows=out/f'edge-{prefix}.rows';r=run([str(binary),str(ep),str(prefix),str(rows)]);s=run(census(ep,rows,prefix))
    checked('cross8192_history_allrows_'+str(prefix),r['row_count']==s['all_oracle_rows_compared']==prefix*16)
    streams.append(rows.read_bytes())
checked('chunk_prefix_identity',all(x==streams[-1][:len(x)] for x in streams))
# Addresses independently derived from the 192-byte start and 90-byte row stride.
g=c.geometry([42,43,44]+[42]*13,0,1)
checked('dedup_straddler_overlap_kept',g['first_page_dedup_jobs']==2 and g['jobs_8k']==1 and g['baseline_job_page_bytes_sum_with_overlap']==12288 and g['distinct_page_union_bytes']==8192 and g['row_request_page_straddlers']==1)
g=c.geometry([1454]*16,0,1)
checked('record_boundary_two_records',g['first_page_dedup_jobs']==1 and g['jobs_8k']==1 and g['baseline_jobs_assumed_records_touched']==2 and g['row_request_assumed_record_straddlers']==16)
g=c.geometry([c.NROWS-1]*16,0,1)
checked('EOF_geometry_832_not_IO_pass',g['baseline_jobs_extending_beyond_admitted_eof']==1 and g['baseline_job_bytes_beyond_eof_geometry_only']==832)
for rows in ([c.NROWS]*16,[-1]*16):
    try:c.geometry(rows,0,1)
    except ValueError:checks.append('out_of_table_geometry_rejected')
    else:raise AssertionError('invalid row admitted')
for label,value in [('plus','+1'),('overflow','2147483648'),('malformed','1.0')]:
    p=out/(label+'.tokens');p.write_text(value+'\n');run([str(binary),str(p),'1',str(out/(label+'.rows'))],1,'invalid integer');checks.append(label+'_parser_rejected')
print(json.dumps(dict(passed=True,checks=checks,checks_count=len(checks),external_golden_rows=304,scope='CPU offline equation/history/geometry qualification only; no GPU/shard/actualcache/IO/model context result.'),separators=(',',':')))
