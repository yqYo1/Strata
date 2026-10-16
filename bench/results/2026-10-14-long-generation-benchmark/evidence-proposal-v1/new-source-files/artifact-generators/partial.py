from pathlib import Path
import hashlib,json,difflib
P=Path('/home/yayoi/orca/workspaces/Strata/sycl');D=P/'bench/results/2026-10-14-long-generation-benchmark/evidence-proposal-v1';S=D/'new-source-files';p=S/'tools/strata_bench.py';s=p.read_text()
s=s.replace('    reasoning, finish_reason, done_received = [], None, False','    reasoning, finish_reason, done_received = [], None, False\n    interrupted = None')
s=s.replace('    except urllib.error.HTTPError as e:\n        err =', '    except (KeyboardInterrupt, SystemExit) as e:\n        interrupted = e\n        err = repr(e)[:300]\n    except urllib.error.HTTPError as e:\n        err =')
s=s.replace('"done_received": done_received, "request_sha256":','"done_received": done_received, "_interrupted": interrupted, "request_sha256":')
s=s.replace('            self.journal.append("result", phase, run, turn, sample=row)\n        return raw','            self.journal.append("result", phase, run, turn, sample=row)\n        if raw.get("_interrupted") is not None:\n            raise raw["_interrupted"]\n        return raw')
p.write_text(s)
identity=json.loads((D/'identity.json').read_text());patches=[]
for n in identity['proposed_sources']:
    before=(P/n).read_text() if n in identity['bases'] else '';after=(S/n).read_text();identity['proposed_sources'][n]=hashlib.sha256(after.encode()).hexdigest()
    patches.append(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+n if before else '/dev/null',tofile='b/'+n)))
proposal=''.join(patches);(D/'proposal.patch').write_text(proposal);identity['proposal_sha256']=hashlib.sha256(proposal.encode()).hexdigest();(D/'identity.json').write_text(json.dumps(identity,indent=2)+'\n');print(identity['proposal_sha256'])
