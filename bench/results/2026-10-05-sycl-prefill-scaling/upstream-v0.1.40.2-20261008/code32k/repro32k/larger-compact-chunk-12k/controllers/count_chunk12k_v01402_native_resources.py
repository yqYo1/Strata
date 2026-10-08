"""Count native resource API results in the completed diagnostic, offline."""
from pathlib import Path
import collections, datetime, hashlib, json, re
base=Path(__file__).parent
job=base/'owned-main-vmm-full-ram-chunk12k-v01402-code32k-diagnostic-r1'
record=json.loads((job/'record.json').read_text())
assert record['completed'] and not record['active'] and record['exit_code']==0
out=base/'main-cache-chunk12k-v01402-native-resources'
out.mkdir(mode=0o700)
apis=['zeModuleCreate','zeModuleDestroy','zeKernelCreate','zeKernelDestroy',
      'zeCommandListCreate','zeCommandListCreateImmediate','zeCommandListDestroy',
      'zePhysicalMemCreate','zePhysicalMemDestroy','zeVirtualMemMap','zeVirtualMemUnmap']
counter=collections.defaultdict(collections.Counter)
sizes=collections.defaultdict(collections.Counter)
examples={};phase='startup';previous_clock=None
result=re.compile(rb'SUCCESS \(ZE_RESULT_SUCCESS\) in ('+b'|'.join(a.encode() for a in apis)+rb')\(')
size_pattern=re.compile(rb'(?:inputSize|size)=(\d+)')
with (job/'debugger/inferior.stderr').open('rb') as stream:
    for number,line in enumerate(stream,1):
        if line.startswith(b'strata prefill memory:'):
            phase=line.decode().split(': ',1)[1].split(';',1)[0]
        elif b'SUCCESS (ZE_RESULT_SUCCESS)' in line:
            m=result.search(line)
            if not m:continue
            api=m[1].decode();counter[phase][api]+=1
            s=size_pattern.search(line)
            if s:sizes[phase][api]+=int(s[1])
            examples.setdefault(phase,{}).setdefault(api,{'line':number,'raw':line.decode().rstrip()})
analysis={'active':False,'passed':True,'gpu_access':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope':'Successful LevelZero resource API results counted by preceding phase marker, from one32K debug/validation run. Sizes are requested API inputs, not resident memory; no timing, leak or isolated root-cause conclusion.',
          'counts':{p:dict(c) for p,c in counter.items()},
          'requested_sizes':{p:dict(c) for p,c in sizes.items()},'first_example':examples}
(out/'record.json').write_text(json.dumps(analysis,indent=2)+'\n')
print(json.dumps({'counts':analysis['counts'],'requested_sizes':analysis['requested_sizes']},indent=2))
