"""Stream a completed UR/Level Zero log; no runtime/GPU interaction."""
from pathlib import Path
import collections
import hashlib
import json
import math
import re
import sys

base = Path(__file__).parent
case = sys.argv[1]
directory = base / case
terminal = json.loads((directory/'record.json').read_text())
assert not terminal['active']
assert not terminal['cleanup']['inferior_survived'] and not terminal['cleanup']['gdb_survived']
source = directory/'debugger/inferior.stderr'
create = re.compile(r'SUCCESS .* in zeKernelCreate\(.*pKernelName="([^"]+)".*phKernel=(0x[0-9a-f]+)\)')
launch = re.compile(r'SUCCESS .* in zeCommandListAppendLaunchKernelWithArguments\(.*hKernel=(0x[0-9a-f]+), groupCounts=\{groupCountX=(\d+), groupCountY=(\d+), groupCountZ=(\d+)\}, groupSizes=\{groupSizeX=(\d+), groupSizeY=(\d+), groupSizeZ=(\d+)\}')
begin = re.compile(r'^\s*---> (urEnqueueKernelLaunch(?:WithArgsExp)?)\s*$')
end = re.compile(r'^\s*<--- (urEnqueueKernelLaunch(?:WithArgsExp)?)\(')
native_names = {}
pending = None
rows = {}
unmatched = []
counts = collections.Counter()
with source.open('r', errors='replace') as stream:
    for number, line in enumerate(stream,1):
        if ' in zeKernelCreate(' in line:
            matched = create.search(line)
            if matched:
                native_names[matched[2]] = matched[1]
        if '   ---> urEnqueueKernelLaunch' in line:
            matched = begin.match(line)
            if matched:
                assert pending is None, ('nested enqueue',number)
                pending = {'method':matched[1],'line':number,'launches':[]}
        if ' in zeCommandListAppendLaunchKernelWithArguments(' in line:
            matched = launch.search(line)
            if matched:
                item = dict(kernel=native_names.get(matched[1],'unknown:'+matched[1]),
                            native_groups=list(map(int,matched.group(2,3,4))),
                            native_local=list(map(int,matched.group(5,6,7))),line=number)
                counts['native_launches'] += 1
                if pending:
                    pending['launches'].append(item)
                else:
                    unmatched.append(item)
        if '   <--- urEnqueueKernelLaunch' in line:
            matched = end.match(line)
            if not matched:
                continue
            assert pending and matched[1] == pending['method'], ('enqueue mismatch',number)
            cooperative = 'UR_KERNEL_LAUNCH_FLAG_COOPERATIVE' in line
            successful = '-> UR_RESULT_SUCCESS;' in line
            counts['successful_UR_enqueues' if successful else 'failed_UR_enqueues'] += 1
            if cooperative:
                counts['cooperative_UR_enqueues'] += 1
            assert len(pending['launches'])==1 or not successful, ('native count',number,len(pending['launches']))
            for item in pending['launches']:
                key = (item['kernel'],tuple(item['native_local']),cooperative)
                row = rows.setdefault(key,dict(kernel=item['kernel'],native_local=item['native_local'],
                                               sycl_local=list(reversed(item['native_local'])),cooperative=cooperative,
                                               calls=0,failed_UR_calls=0,max_total_groups=0,shapes={},first_line=item['line']))
                groups = math.prod(item['native_groups'])
                row['calls'] += int(successful)
                row['failed_UR_calls'] += int(not successful)
                row['max_total_groups'] = max(row['max_total_groups'],groups)
                shape = tuple(item['native_groups'])
                row['shapes'][shape] = row['shapes'].get(shape,0)+1
            pending = None
assert pending is None and not unmatched
result_rows=[]
for row in rows.values():
    row['shapes']=[dict(native_groups=list(shape),calls=count) for shape,count in sorted(row['shapes'].items())]
    result_rows.append(row)
assert all(not row['kernel'].startswith('unknown:') for row in result_rows)
mapping=json.loads((base/'observed-cooperative-source-targets.json').read_text())
selected={c for r in mapping['rows'] for c in r['source_candidates'][0]['classes']}
remaining=[r for r in result_rows if r['cooperative']]
assert not any(any(str(len(c))+c in r['kernel'] for c in selected) for r in remaining)
result=dict(scope='Terminal diagnostic log association of UR enqueue entry/return and native kernel-name/complete-shape records; successful APIs do not alone prove device completion. Only reviewed selected kernel flags are asserted absent; remaining flags require resource/source review.',
            model_completed=terminal['completed'],model_healthy=terminal['healthy'],
            case=case,record_sha256=hashlib.sha256((directory/'record.json').read_bytes()).hexdigest(),
            log_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),log_bytes=source.stat().st_size,
            controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),counts=dict(counts),
            distinct_combinations=len(result_rows),remaining_cooperative_combinations=len(remaining),
            all_41_reviewed_classes_absent_from_cooperative_enqueues=True,unmatched_native_launches=0,
            rows=sorted(result_rows,key=lambda r:(not r['cooperative'],-r['max_total_groups'],r['kernel'])))
path=base/(case+'-launch-flags.json');path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
print(json.dumps([dict(kernel=r['kernel'],local=r['sycl_local'],groups=r['max_total_groups']) for r in remaining],indent=2))
