import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,'tools');sys.path.insert(0,'.')
import strata_tokenizer as ST
from serve.frontend import ChatTemplate
root=Path.home()/'.local/share/strata-sycl';cfg=json.loads((root/'serve-config-iq3_s.json').read_text());tp=Path(cfg['tokenizer'])
vocab=json.loads((tp/'vocab.json').read_text());v=[None]*len(vocab)
for t,i in vocab.items():v[i]=t
tok=ST.Tokenizer(v,(tp/'merges.txt').read_text().split('\n'),json.loads((tp/'token_type.json').read_text()))
meta=[]
for label,records in [('4k',240),('8k',480)]:
 text='Remember this fact: the secret color is blue.\n'+''.join(f'Record {i}: the routine check passed and the clock was repaired.\n' for i in range(1,records+1))+'\nWhat is the secret color? Reply with only the color word.'
 prompt=ChatTemplate(tp/'chat_template.jinja').render([dict(role='user',content=text)],enable_thinking=False)
 ids=tok.encode(prompt,parse_special=True)
 path=Path(f'/tmp/strata-sycl-goal-prefill-scale-{label}-tokens.txt');path.write_text(' '.join(map(str,ids))+'\n')
 meta.append(dict(label=label,records=records,tokens=len(ids),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),tokens_file=str(path),text=text))
 print(label,len(ids),flush=True)
Path('/tmp/strata-sycl-goal-prefill-scale-fixtures.json').write_text(json.dumps(meta,indent=2)+'\n')
