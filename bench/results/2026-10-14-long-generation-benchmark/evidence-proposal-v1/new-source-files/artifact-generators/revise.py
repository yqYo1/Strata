from pathlib import Path
import hashlib,json,difflib
P=Path('/home/yayoi/orca/workspaces/Strata/sycl');D=P/'bench/results/2026-10-14-long-generation-benchmark/evidence-proposal-v1';S=D/'new-source-files'
p=S/'tools/strata_bench.py';s=p.read_text().replace('        journal = RequestJournal(directory, red)','''        journal = RequestJournal(directory, red)
        journal.append("identity", "suite", source=client_fingerprint(),
                       model=pre.get("model"), quant=pre.get("quant"), context=pre.get("context"),
                       thinking=a.think, long_generation_thinking=long_generation.THINKING,
                       workload_version=long_generation.WORKLOAD_VERSION, requested_runs=max(1, a.runs))''')
s=s.replace('"draft counts remain recorded and are ineligible.",','"draft counts remain recorded. Speed eligibility uses normal generation/usage/body/clock gates; "\n              "agreement eligibility additionally requires valid draft counts. Zero-offer generations can have "\n              "throughput but have undefined agreement.",')
s=s.replace('"| Workload | Run | Generated | Decode tok/s | Accepted / offered | Acceptance | Eligible |",','"| Workload | Run | Generated | Decode tok/s | Accepted / offered | Acceptance | Speed eligible | Agreement eligible |",')
s=s.replace('"| --- | ---: | ---: | ---: | --- | ---: | --- |"]','"| --- | ---: | ---: | ---: | --- | ---: | --- | --- |"]')
s=s.replace("f\"{row['drafts_accepted']} / {row['drafts_offered']} | {rate_text} | {row['eligible']} |\")","f\"{row['drafts_accepted']} / {row['drafts_offered']} | {rate_text} | {row['speed_eligible']} | {row['eligible']} |\")")
p.write_text(s)
identity=json.loads((D/'identity.json').read_text());patches=[]
for n in identity['proposed_sources']:
    before=(P/n).read_text() if n in identity['bases'] else ''
    after=(S/n).read_text();identity['proposed_sources'][n]=hashlib.sha256(after.encode()).hexdigest()
    patches.append(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+n if before else '/dev/null',tofile='b/'+n)))
proposal=''.join(patches);(D/'proposal.patch').write_text(proposal);identity['proposal_sha256']=hashlib.sha256(proposal.encode()).hexdigest();(D/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
