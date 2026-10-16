#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare diff and fingerprints from packaged immutable bases/candidate bytes only."""
from pathlib import Path
import hashlib, difflib, json
p=Path(__file__).resolve().parent
sha=lambda b: hashlib.sha256(b).hexdigest()
base={str(f.relative_to(p/'base')):f.read_bytes() for f in sorted((p/'base').rglob('*')) if f.is_file()}
files={str(f.relative_to(p/'files')):f.read_bytes() for f in sorted((p/'files').rglob('*')) if f.is_file()}
diff=''
for name,data in files.items():
 old=base.get(name,b'')
 diff+=''.join(difflib.unified_diff(old.decode().splitlines(True),data.decode().splitlines(True),fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
(p/'source.diff').write_text(diff)
m={'base_commit':'617647659f21487ff3bf61fa1fd4ba87ba2520bf','license':'root MIT','status':'SOURCE ONLY; implementation/build/test/runtime/GPU validation UNRUN','origin':'Own current MIT prompt_attention.cpp and prompt_k_cellmajor.cpp; independent fixture improvements; v1 production bytes retained unchanged; no XeStrata source/helper/test/archive read or used','base_sha256':{n:sha(b) for n,b in base.items()},'candidate_sha256':{n:sha(b) for n,b in files.items()},'patch_sha256':sha(diff.encode())}
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
