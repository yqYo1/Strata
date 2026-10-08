"""Enumerate actual linked TU dependencies before a uniform DPCT-header build."""
from pathlib import Path
import datetime,hashlib,json,re,shlex,subprocess,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build=root/'build-sycl-event-ack-registered-copy-v3-20261008/build'
header=build.parent/'source/sycl/include/dpct/device.hpp'
out=base/'dpct-profile-definition-linked-tu-audit-v2';out.mkdir(mode=0o700)
record=dict(active=True,passed=False,gpu_tested=False,adopted=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Actual strata-linked Ninja command/dependency inventory; no compiler or GPU workload. Correct literal Ninja deps delimiter; retain the prior failed parser separately.')
try:
    cs=subprocess.check_output(['/usr/bin/ninja','-t','commands','strata'],cwd=build,text=True,timeout=30).splitlines();rows=[]
    for command in cs:
        v=shlex.split(command)
        if '-c' in v and '-o' in v and Path(v[v.index('-c')+1]).is_file():
            p=Path(v[v.index('-c')+1]);obj=v[v.index('-o')+1];text=p.read_text();defines='DPCT_PROFILING_ENABLED' in ' '.join(x for x in v if x.startswith('-D')) or any(x.strip().startswith('#define DPCT_PROFILING_ENABLED') for x in text.splitlines())
            rows.append(dict(source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),object=obj,profiling_macro_defined_in_source_or_flags=defines,argv=v))
    argv=['/usr/bin/ninja','-t','deps']+[x['object'] for x in rows];data=subprocess.check_output(argv,cwd=build,text=True,timeout=30);(out/'ninja-deps.txt').write_text(data);current=None;deps={}
    for line in data.splitlines():
        if line and not line.startswith(' '):
            assert ': #deps ' in line,line
            current=line.split(': #deps ',1)[0];deps[current]=[]
        elif current and line.startswith('    '):deps[current].append(line.strip())
    selected=[]
    for row in rows:
        assert row['object'] in deps,row['object']
        ds=deps[row['object']];row['device_header_in_actual_dependencies']=str(header) in ds;row['codepin_header_in_actual_dependencies']=any('/dpct/codepin/' in x for x in ds)
        if row['device_header_in_actual_dependencies'] and not row['profiling_macro_defined_in_source_or_flags']:selected.append(row)
    assert any(x['source'].endswith('/conversation_state.cpp') for x in selected)
    assert not any(x['codepin_header_in_actual_dependencies'] for x in rows)
    review=json.loads((base/'dpct-profile-definition-v01402-source-review-v3/record.json').read_text());assert review['passed'] and not review['active']
    record.update(passed=True,ninja_dependency_command=argv,ninja_dependency_sha256=hashlib.sha256((out/'ninja-deps.txt').read_bytes()).hexdigest(),device_header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),source_review_sha256=hashlib.sha256((base/'dpct-profile-definition-v01402-source-review-v3/record.json').read_bytes()).hexdigest(),linked_compile_commands=len(rows),actual_device_header_users=sum(x['device_header_in_actual_dependencies'] for x in rows),required_unprofiled_tus=selected,codepin_not_in_actual_linked_dependencies=True)
except BaseException as e:record['error']=repr(e);raise
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record.get(k) for k in ['passed','error','linked_compile_commands','actual_device_header_users']}|{'required_objects':[x['object'] for x in record.get('required_unprofiled_tus',[])]},indent=2))
