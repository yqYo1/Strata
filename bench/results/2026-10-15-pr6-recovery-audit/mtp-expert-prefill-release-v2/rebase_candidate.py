#!/usr/bin/env python3
"""Main review artifact only: merge original proposal without touching product."""
import difflib
import hashlib
import json
import re
import subprocess
from pathlib import Path

O=Path(__file__).parent
P=O.parents[3]
OLD=O.parent/'mtp-expert-prefill-release-v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert (P/'AGENTS.md').is_file()
    prior=O/'rebase-v1.json';failed=json.loads(prior.read_text())
    assert not failed['complete'] and not (O/'rebase-v3.json').exists()
    original=json.loads((OLD/'baseproposedSHA.json').read_text())
    assert sha(OLD/'source.diff')==original['source_diff_sha256']
    r=dict(complete=False,applied=False,compiled=False,GPU_executed=False,
        failed_preparation_receipt_sha256=sha(prior),original_patch_sha256=sha(OLD/'source.diff'),
        base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
        recipe_sha256=sha(Path(__file__)),files={},merge_commands=[],
        preparation_error='Initial script selected bench as its root; failed before product mutation/replay. Corrected root to actual sycl worktree.',
        main_adjustments=['Resolve only shared <limits>/<memory> insertion conflict.',
            'Fail-stop recapture exceptions before unwinding expert/graph/session owners; default-off rethrows unchanged.',
            'Record actual free/total VRAM at four prompt detach/restore boundaries.',
            'Retain675MiB immutable original host source; restore from RAM without repeated file IO/SHA scans. Host admission counts it before load.'],
        initial_preparation_receipt_sha256=sha(O/'rebase-v2.json'),
        owned_ram_fragment_sha256=sha(O/'owned_ram_upload.fragment'))
    for name,identity in original['files'].items():
        source=OLD/'files'/name;assert sha(source)==identity['proposed_sha256']
        target=O/'files'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if identity['base_sha256'] is None:
            assert not (P/name).exists();text=source.read_text();pin=None
        else:
            base=OLD/'base'/name;assert sha(base)==identity['base_sha256']
            command=['git','merge-file','-p',str(P/name),str(base),str(source)]
            merged=subprocess.run(command,capture_output=True,text=True,timeout=30)
            r['merge_commands'].append(dict(path=name,argv=command,exit_code=merged.returncode,stderr=merged.stderr))
            text=merged.stdout;pin=sha(P/name)
            if merged.returncode:
                assert name=='src/core/mtp.cpp' and merged.returncode==1
                pattern=r'<<<<<<<[^\n]+\n=======\n#include <memory>\n>>>>>>>[^\n]+\n'
                text,count=re.subn(pattern,'#include <memory>\n',text)
                assert count==1
            assert not re.search(r'^<<<<<<<|^=======|^>>>>>>>',text,re.M)
        if name=='include/strata/core/mtp.hpp':
            anchor='    std::array<uint8_t, 32> expert_source_sha256_{};\n'
            assert text.count(anchor)==1
            text=text.replace(anchor,anchor+'    std::vector<uint8_t> expert_host_source_; // immutable opt-in RAM restore owner\n',1)
        if name=='src/core/mtp.cpp':
            begin=text.index('// Used only by the opt-in loader/restore.')
            end=text.index('bool MtpDrafter::detach_prefill_experts',begin)
            text=text[:begin]+(O/'owned_ram_upload.fragment').read_text()+'\n'+text[end:]
            for signature,next_anchor in [
                ('bool MtpDrafter::capture_round(int T, bool coupled, std::string& err)', '// Chain step'),
                ('bool MtpDrafter::capture_step(int j, bool coupled, std::string& err)', 'void MtpDrafter::kv_restore')]:
                start=text.index(signature+' {');end=text.index(next_anchor,start)
                body=text[start:end]
                assert body.count('return captured;\n}')==1
                body=body.replace(signature+' {',signature+' try {',1)
                body=body.replace('return captured;\n}',
                    'return captured;\n} catch (...) {\n'
                    '    if (expert_recapture_timing_)\n'
                    '        expert_failstop("expert recapture threw before checked ownership close", cudaErrorUnknown);\n'
                    '    throw;\n}',1)
                text=text[:start]+body+text[end:]
        if name=='src/program/generate.cpp':
            anchor='    strata::core::MtpPrefillExpertScope release_scope;\n'
            assert text.count(anchor)==1
            telemetry='''    auto memory_mark = [](const char* phase) {
        size_t available = 0, total = 0;
        const auto status = cudaMemGetInfo(&available, &total);
        if (status != cudaSuccess || available > total) {
            std::fprintf(stderr, "strata: MTP expert memory telemetry failed at %s; stopping with owners live\\n", phase);
            std::fflush(stderr); std::_Exit(EXIT_FAILURE);
        }
        std::fprintf(stderr, "strata mtp: prompt expert phase=%s free_bytes=%llu total_bytes=%llu\\n", phase,
                     (unsigned long long)available, (unsigned long long)total);
    };
    memory_mark("before-detach");
'''
            text=text.replace(anchor,telemetry+anchor,1)
            anchor='    if (!release_scope.open(mtp, err)) return false;\n'
            assert text.count(anchor)==1
            text=text.replace(anchor,anchor+'    memory_mark("after-detach");\n',1)
            anchor='    auto restore = [&]() {\n        std::string restore_error;'
            assert text.count(anchor)==1
            text=text.replace(anchor,'    auto restore = [&]() {\n        memory_mark("after-prompt-before-restore");\n        std::string restore_error;',1)
            anchor='            std::fflush(stderr); std::_Exit(EXIT_FAILURE);\n        }\n    };\n    bool ok = false;'
            assert text.count(anchor)==1
            text=text.replace(anchor,'            std::fflush(stderr); std::_Exit(EXIT_FAILURE);\n        }\n        memory_mark("after-restore");\n    };\n    bool ok = false;',1)
        target.write_text(text)
        r['files'][name]=dict(base_sha256=pin,proposed_sha256=sha(target))
    patch=''.join(''.join(difflib.unified_diff(
        (P/n).read_text().splitlines(True) if i['base_sha256'] else [],
        (O/'files'/n).read_text().splitlines(True),
        fromfile='a/'+n if i['base_sha256'] else '/dev/null',tofile='b/'+n)) for n,i in r['files'].items())
    (O/'source.diff').write_text(patch)
    r.update(complete=True,source_diff_sha256=sha(O/'source.diff'))
    (O/'rebase-v3.json').write_text(json.dumps(r,indent=2)+'\n')
    assert sha(prior)==r['failed_preparation_receipt_sha256']
    print(json.dumps(dict(complete=True,applied=False,files=len(r['files']),source_diff_sha256=r['source_diff_sha256'])))


if __name__=='__main__':main()
