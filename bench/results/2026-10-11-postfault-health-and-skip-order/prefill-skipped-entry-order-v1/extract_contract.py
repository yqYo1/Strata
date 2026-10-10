"""Root-run extraction of pinned ACTUAL old/new release_to lambdas; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
OLD=B/'prefill-gemm-service-only-v1/prefill.cpp'
NEW=B/'prefill-skipped-entry-order-v1/prefill.cpp'
PINS=['1cf7e911a555e8611c1368a4d18d4ba81c95027bb8de77f89ad65ac64b05fc3c',
      'e0685a85bc391b6ef916c33ea3f959680088832b8a9a7d706668acba3c47f8d0']
ADDED='                                    wait_issued(k);   // finish issuer slot reads before publishing skipped-entry reuse\n'
ANCHOR='                            auto release_to = [&](int32_t e_stop) {'
END='\n                            };'

def require(v,s):
    if not v: raise RuntimeError(s)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True,type=Path)
    out=p.parse_args().output
    require(out.is_absolute() and out.parent.resolve().is_relative_to(B)
            and not out.exists(),'new private extraction directory')
    texts=[]
    for path,pin in zip((OLD,NEW),PINS):
        with path.open('rb') as f: data=f.read(1<<20)
        require(len(data)<1<<20 and hashlib.sha256(data).hexdigest()==pin,'immutable full source pin')
        texts.append(data.decode('utf-8'))
    require(texts[1].count(ADDED)==1 and texts[1].replace(ADDED,'',1)==texts[0],'only one wait line added')
    blocks=[]
    for text in texts:
        require(text.count(ANCHOR)==1,'unique primary release_to anchor')
        start=text.index(ANCHOR); stop=text.index(END,start)+len(END)
        block=text[start:stop]
        require(block.count('dpct::sync_barrier(m.used[sl], m.cs);')==1
                and block.count('consumed = ++k;')==1,'exact skipped entry body')
        blocks.append(block)
    require(blocks[1].replace(ADDED,'',1)==blocks[0],'actual extracted body delta')
    out.mkdir(mode=0o700)
    for name,block in zip(('old_release.inc','new_release.inc'),blocks):
        with (out/name).open('x') as f: f.write(block+'\n')
    receipt=dict(source_sha256=PINS,extracted_sha256=[hashlib.sha256((s+'\n').encode()).hexdigest() for s in blocks],
                 scope='actual pinned primary lambda, GPU event call stubbed and slot metadata atomic stand-ins')
    with (out/'extraction.json').open('x') as f: f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__': main()
