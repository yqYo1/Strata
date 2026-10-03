import json,time,urllib.request
from pathlib import Path
base='http://127.0.0.1:18085';result={'base_url':base,'config':'/home/yayoi/.local/share/strata-sycl/serve-config.json','checks':[]}
def get(path):
 with urllib.request.urlopen(base+path,timeout=120) as r:return r.status,r.read()
def post(path,body):
 req=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 return urllib.request.urlopen(req,timeout=180)
prompt='Write a 300-word short story about an engineer repairing a clock on a rainy evening. Include a clear ending.'
body={'model':'qwen3.8-flash-next-sycl','messages':[{'role':'user','content':prompt}],'max_tokens':8,'temperature':0,'reasoning_effort':'none'}
status,models=get('/v1/models');assert status==200;result['models']=json.loads(models)
status,html=get('/');assert status==200 and b'<html' in html.lower();result['web_ui_status']=status
answers=[]
for i in range(2):
 start=time.monotonic()
 with post('/v1/chat/completions',body) as r:data=json.load(r);assert r.status==200
 answers.append(data['choices'][0]['message']);result['checks'].append({'kind':'openai_nonstream','request':i+1,'seconds':time.monotonic()-start,'response':data})
 assert data['usage']['completion_tokens']==8
assert answers[0]==answers[1]
with post('/v1/chat/completions',dict(body,stream=True,max_tokens=128)) as r:
 first=None
 while True:
  line=r.readline().decode().strip()
  if not line.startswith('data: ') or line=='data: [DONE]':continue
  data=json.loads(line[6:]);delta=data['choices'][0]['delta']
  if delta.get('content'):
   first=delta['content'];break
result['checks'].append({'kind':'cancel_stream_after_content','first_content':first})
start=time.monotonic()
with post('/v1/chat/completions',body) as r:data=json.load(r)
assert data['choices'][0]['message']==answers[0];result['checks'].append({'kind':'openai_after_cancel','seconds':time.monotonic()-start,'response':data})
anth={'model':body['model'],'messages':body['messages'],'max_tokens':8,'temperature':0,'thinking':{'type':'disabled'}}
with post('/v1/messages',anth) as r:data=json.load(r);assert r.status==200
assert data['type']=='message' and data['usage']['output_tokens']==8
result['checks'].append({'kind':'anthropic_nonstream','response':data})
with post('/v1/chat/completions',dict(body,stream=True)) as r:
 chunks=[];done=False
 for raw in r:
  line=raw.decode().strip()
  if line=='data: [DONE]':done=True;break
  if line.startswith('data: '):chunks.append(json.loads(line[6:]))
assert done
content=''.join(c['choices'][0]['delta'].get('content','') for c in chunks if c.get('choices'))
assert content==answers[0]['content'];result['checks'].append({'kind':'openai_complete_stream','chunks':chunks,'done':done,'content_equal':True})
Path('/tmp/strata-sycl-mtp-floor-http-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'checks':[c['kind'] for c in result['checks']],'repeat_equal':True,'cancel_recovery_equal':True,'complete_stream_equal':True,'answer':answers[0]},ensure_ascii=False),flush=True)
