import hashlib, json, subprocess, sys
from pathlib import Path

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0,str(root/'tools'))
from strata_tokenizer import Tokenizer
out = Path('/tmp/strata-upstream-arc-256k-context-fixture')
out.mkdir(exist_ok=True)
pack = Path.home()/'.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/tokenizer'
vocab = json.loads((pack/'vocab.json').read_text()); tokens = [None]*len(vocab)
for text,index in vocab.items(): tokens[index] = text
tok = Tokenizer(tokens,(pack/'merges.txt').read_text().splitlines(),json.loads((pack/'token_type.json').read_text()))
rev = 'e274e17'; parts = []; sources = []
first = ['src/program/generate.cpp','serve/server.py','src/prefill/prefill.cpp']
def add(name,start=0,end=None):
    text = subprocess.check_output(['git','show',rev+':'+name],cwd=root,text=True,stderr=subprocess.PIPE)[start:end]
    sources.append(dict(path=name,start_character=start,characters=len(text),excerpt_sha256=hashlib.sha256(text.encode()).hexdigest()))
    parts.append('\nFile '+name+':\n```\n'+text+'\n```\n')
for name in first: add(name,end=22000)
for name in first: add(name,start=22000)
for name in ['src/engine/engine.cpp','src/kernels/qsa.cu','src/kernels/gdn.cu','src/kernels/cpu/expert.cpp']:
    try: add(name)
    except subprocess.CalledProcessError: pass
prompt = '<|im_start|>user\nReview the following inference-engine code. Explain how prompt batching, expert streaming, and the server interact. Identify changes that could improve long-context coding-agent workloads.\n'+''.join(parts)+'<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n'
ids = tok.encode(prompt,parse_special=True)
assert tok.decode(ids)==prompt
old = Path('/tmp/strata-upstream-arc-long-context-fixture/coding-context-64k-tokens.txt')
prefix = [int(x) for x in old.read_text().split()]
assert ids[:len(prefix)]==prefix
assert len(ids)==281668 and len(ids)>262144
stored = ids[:262145]
path = out/'coding-context-256k-tokens.txt'
path.write_text(' '.join(map(str,stored))+'\n')
(out/'fixture.json').write_text(json.dumps(dict(kind='prefixes of one extended code-review prompt',
    source_revision=rev,sources=sources,full_prompt_tokens=len(ids),stored_tokens=len(stored),
    tokenizer_hashes={n:hashlib.sha256((pack/n).read_bytes()).hexdigest() for n in ['vocab.json','merges.txt','token_type.json']},
    fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),previous_fixture_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
    previous_prefix_bit_identical=True,note='The 64K and 256K fixtures are prefixes of the same repository-code token stream; prefixes are diagnostic causal inputs, not separately completed chat turns.'),indent=2)+'\n')
print('stored',len(stored),'full',len(ids),'characters',len(prompt),flush=True)
