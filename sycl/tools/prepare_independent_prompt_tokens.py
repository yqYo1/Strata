"""Host-only whole-document train/validation prompt preparation, never inference.

Manifest order is the predeclared selection rule. Stop at the first whole-document
prefix meeting the minimum; reject overage, never truncate/pad/repeat. No scores.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types

TOKENIZER_SHA='9b0327e25aa62754406a924db6cc137e6edb1d93c7af76a4f3d14a8920ea72c8'
FRONTEND_SHA='095bbac7d49b19d815d2634457d14d216ba9bd4e98998a4e3b4288f20a25e923'
ASSETS=('vocab.json','merges.txt','token_type.json','tokenizer.json','chat_template.jinja')
DOC_CAP=2*1024**2
CORPUS_CAP=8*1024**2
ASSET_CAP=32*1024**2
ASSETS_CAP=64*1024**2
OUTPUT_CAP=32*1024**2
MANIFEST_CAP=128*1024
MIN_TOKENS=32768
MAX_TOKENS=65536


def require(ok,message):
    if not ok: raise ValueError(message)


def digest(data): return hashlib.sha256(data).hexdigest()


def sha(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'expected SHA256 required')
    return value


def read_verified(path,expected,cap):
    sha(expected)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_nlink==1 and 0<=before.st_size<=cap,'owned regular bounded file required')
        data=bytearray()
        while len(data)<before.st_size:
            part=os.read(fd,min(65536,before.st_size-len(data)))
            require(part,'short read');data.extend(part)
        require(not os.read(fd,1),'file grew')
        after=os.fstat(fd)
        require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_uid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')),'file changed')
        require(digest(data)==expected,'SHA256 mismatch')
        return bytes(data)
    finally: os.close(fd)


def json_bytes(value): return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8')


def parse_json(data):
    def unique(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate JSON key');result[k]=v
        return result
    return json.loads(data.decode('utf-8'),object_pairs_hook=unique)


def load_source(name,path,expected):
    data=read_verified(path,expected,256*1024)
    module=types.ModuleType(name);module.__file__=str(path)
    sys.modules[name]=module
    try: exec(compile(data,str(path),'exec'),module.__dict__)
    except BaseException:
        del sys.modules[name];raise
    return module


def documents(manifest,reader=read_verified):
    require(type(manifest) is dict and manifest.get('schema')=='independent-documents-v1','manifest schema')
    require(manifest.get('selection_rule')=='manifest-order-first-complete-prefix','fixed selection rule')
    require(type(manifest.get('model_identity')) is str and bool(manifest['model_identity']),'declared model identity')
    corpora=manifest.get('corpora')
    require(type(corpora) is dict and set(corpora)=={'train','validation'},'two named corpora required')
    output={};ids=set();hashes=set();total=0
    for role in ('train','validation'):
        entries=corpora[role]
        require(type(entries) is list and 1<=len(entries)<=64,'bounded document list')
        docs=[]
        for entry in entries:
            require(type(entry) is dict and set(entry)=={'path','source_id','sha256','version'},'document fields')
            for key in ('path','source_id','version'):
                require(type(entry[key]) is str and 0<len(entry[key])<=4096,'document metadata')
            require(entry['source_id'] not in ids,'duplicate/overlapping source ID')
            expected=sha(entry['sha256'])
            require(expected not in hashes,'duplicate/overlapping document hash')
            ids.add(entry['source_id']);hashes.add(expected)
            data=reader(entry['path'],expected,DOC_CAP)
            require(len(data)<=DOC_CAP and digest(data)==expected,'document bounds/hash')
            total+=len(data);require(total<=CORPUS_CAP,'total corpus budget')
            text=data.decode('utf-8')
            require(text.strip(),'empty document')
            docs.append((entry,text))
        output[role]=docs
    return output


def checked_ids(ids,vocab_size,maximum=MAX_TOKENS):
    require(type(ids) is list and len(ids)<=maximum,'token array budget/type')
    require(all(type(i) is int and 0<=i<vocab_size for i in ids),'token ID bounds/type')
    return ids


def select_prompt(docs,render,encode,vocab_size,prefix='',suffix='',minimum=MIN_TOKENS,maximum=MAX_TOKENS):
    require(type(minimum) is int and type(maximum) is int and 1<=minimum<=maximum<=MAX_TOKENS,'token bounds')
    parts=[];selected=[]
    for entry,text in docs:
        parts.append(text);selected.append(entry)
        body='\n\n'.join(parts) # complete UTF8 documents, explicit natural separator
        rendered=render(prefix+body+suffix)
        require(type(rendered) is str and len(rendered.encode('utf-8'))<=CORPUS_CAP+16384,'rendered text budget')
        ids=checked_ids(encode(rendered),vocab_size)
        require(len(ids)<=maximum,'whole-document overage')
        if len(ids)>=minimum:
            return dict(selected_documents=selected,rendered=rendered,ids=ids)
    raise ValueError('insufficient complete documents')


def overlap_gate(train,validation):
    """Fixed 16-ID shingle containment gate; not statistical independence proof."""
    require(type(train) is list and type(validation) is list and 16<=len(train)<=MAX_TOKENS and 16<=len(validation)<=MAX_TOKENS,'overlap length/type bound')
    require(all(type(i) is int and 0<=i<248320 for i in train+validation),'overlap ID type/bounds')
    a={tuple(train[i:i+16]) for i in range(len(train)-15)}
    b={tuple(validation[i:i+16]) for i in range(len(validation)-15)}
    shared=len(a & b);denominator=min(len(a),len(b))
    require(shared*5<denominator*4,'near-overlap token shingles (>=80%)')
    common=0
    for x,y in zip(train,validation):
        if x!=y: break
        common+=1
    return dict(shingle_width=16,train_unique_shingles=len(a),validation_unique_shingles=len(b),shared_shingles=shared,containment_denominator=denominator,common_prefix_ids=common,matching_coordinates=sum(x==y for x,y in zip(train,validation)),compared_coordinates=min(len(train),len(validation)),rule='reject >=80% unique 16-ID shingle containment; diagnostic only, not generalization proof')


def runtime(source_root,asset_dir,asset_hashes):
    require(type(asset_hashes) is dict and set(asset_hashes)==set(ASSETS),'all tokenizer asset hashes required')
    assets={name:read_verified(Path(asset_dir)/name,sha(asset_hashes[name]),ASSET_CAP) for name in ASSETS}
    require(sum(map(len,assets.values()))<=ASSETS_CAP,'total asset budget')
    cfg=parse_json(assets['tokenizer.json']);vocab=parse_json(assets['vocab.json']);types_=parse_json(assets['token_type.json'])
    require(type(cfg) is dict and cfg.get('model')=='gpt2' and cfg.get('pre')=='qwen35' and cfg.get('add_bos_token') is False,'tokenizer config/no BOS')
    require(type(vocab) is dict and len(vocab)==248320 and cfg.get('vocab_size')==248320,'vocabulary geometry')
    tokens=[None]*len(vocab)
    for text,index in vocab.items():
        require(type(text) is str and type(index) is int and 0<=index<len(tokens) and tokens[index] is None,'vocabulary ID inverse')
        tokens[index]=text
    require(all(type(t) is str for t in tokens),'vocabulary completeness')
    merges=assets['merges.txt'].decode('utf-8').split('\n')
    require(len(merges)==247587 and cfg.get('n_merges')==247587 and len(set(merges))==len(merges),'merge geometry/duplicates')
    require(type(types_) is list and len(types_)==len(tokens) and all(type(t) is int and 1<=t<=6 for t in types_),'token type geometry')
    specials=cfg.get('special_ids')
    require(type(specials) is dict and all(type(k) is str and type(v) is int and 0<=v<len(tokens) for k,v in specials.items()),'special ID config')
    tkmod=load_source('frozen_prompt_tokenizer',Path(source_root)/'tools/strata_tokenizer.py',TOKENIZER_SHA)
    frontend=load_source('frozen_prompt_frontend',Path(source_root)/'serve/frontend.py',FRONTEND_SHA)
    require(cfg.get('pre_pattern')==tkmod.QWEN35_PATTERN,'pre pattern identity')
    tok=tkmod.Tokenizer(tokens,merges,types_)
    # Constructor rereads the path; compare its source and reverify exact asset.
    template=frontend.ChatTemplate(Path(asset_dir)/'chat_template.jinja')
    require(template.source.encode('utf-8')==assets['chat_template.jinja'],'template constructor source changed')
    read_verified(Path(asset_dir)/'chat_template.jinja',asset_hashes['chat_template.jinja'],ASSET_CAP)
    return tok,template,cfg


def write_bundle(output,payloads,writer=None):
    """New private directory, exclusive files, completion manifest written last.

    A failed write leaves an incomplete private directory, never a pass receipt;
    no cleanup/overwrite. Caller must inspect/remove it under their ownership.
    """
    require(type(payloads) is dict and 'record.json' in payloads,'bundle record required')
    require(all(type(k) is str and '/' not in k and k not in ('.','..') and type(v) is bytes for k,v in payloads.items()),'bundle names/types')
    require(sum(len(v) for v in payloads.values())<=OUTPUT_CAP,'total output budget')
    os.mkdir(output,0o700)
    for name in sorted(k for k in payloads if k!='record.json')+['record.json']:
        fd=os.open(Path(output)/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            data=payloads[name]
            if writer is not None: writer(fd,data)
            else:
                view=memoryview(data)
                while view:
                    n=os.write(fd,view);require(n>0,'short output write');view=view[n:]
            require(os.fstat(fd).st_size==len(data),'incomplete output write')
            os.fsync(fd)
        finally: os.close(fd)


def prepare(manifest_data,manifest_sha,source_root,asset_dir,output,self_sha):
    require(digest(manifest_data)==sha(manifest_sha),'manifest SHA256 mismatch')
    manifest=parse_json(manifest_data);docs=documents(manifest)
    wrapper=manifest.get('wrapper')
    require(type(wrapper) is dict and set(wrapper)=={'user_prefix','user_suffix'},'explicit common wrapper')
    require(all(type(v) is str and len(v.encode('utf-8'))<=4096 for v in wrapper.values()),'wrapper budget')
    require(manifest.get('render_options')=={'enable_thinking':True},'fixed production render options')
    tok,template,cfg=runtime(source_root,asset_dir,manifest.get('tokenizer_asset_sha256'))
    # Simple user text only: forbid model special literals so Service.encode_prompt
    # marking/plain-span branch cannot differ. No tools/effort_end/images/server.
    for text in list(wrapper.values())+[text for role in docs.values() for _,text in role]:
        require(not any(literal in text for literal in tok.special_tokens),'special literal in source text unsupported')
    render=lambda body: template.render([{'role':'user','content':body}],tools=None,add_generation_prompt=True,enable_thinking=True)
    encode=lambda text: tok.encode(text,parse_special=True)
    selected={role:select_prompt(docs[role],render,encode,len(tok.tokens),wrapper['user_prefix'],wrapper['user_suffix']) for role in ('train','validation')}
    overlap=overlap_gate(selected['train']['ids'],selected['validation']['ids'])
    empty_wrapper=render(wrapper['user_prefix']+wrapper['user_suffix'])
    payloads={'manifest.json':manifest_data}
    summaries={}
    for role,result in selected.items():
        text=result['rendered'].encode('utf-8');ids=(' '.join(map(str,result['ids']))+'\n').encode('ascii')
        payloads[role+'.rendered.txt']=text;payloads[role+'.tokens.txt']=ids
        summaries[role]=dict(selected_documents=result['selected_documents'],document_count=len(result['selected_documents']),token_count=len(result['ids']),rendered_text_sha256=digest(text),token_ids_sha256=digest(ids),token_serialization='ASCII decimal IDs separated by spaces, final LF',rendered_path=role+'.rendered.txt',tokens_path=role+'.tokens.txt')
    record=dict(schema='independent-prompt-preparation-v1',completed=True,scope='host input fixture preparation only',manifest_sha256=manifest_sha,self_sha256=self_sha,source_sha256={'tools/strata_tokenizer.py':TOKENIZER_SHA,'serve/frontend.py':FRONTEND_SHA},source_root=str(source_root),tokenizer_dir=str(asset_dir),tokenizer_asset_sha256=manifest['tokenizer_asset_sha256'],model_identity=manifest['model_identity'],tokenizer_config=cfg,selection_rule=manifest['selection_rule'],render_options=manifest['render_options'],parse_special=True,add_generation_prompt=True,add_bos_token=False,separator='two LF between whole documents',minimum_tokens=MIN_TOKENS,maximum_tokens=MAX_TOKENS,prompts=summaries,overlap=overlap,common_wrapper=dict(rendered_sha256=digest(empty_wrapper.encode('utf-8')),token_count=len(encode(empty_wrapper)),note='empty-body wrapper count is not a proven prefix span; BPE boundaries can change'),GPU_used=False,inference_run=False,performance_eligible=False,adopted=False,full_lifecycle_passed=False,limitations=['Declared source/document disjointness plus fixed token near-overlap gate do not certify statistical independence or generalization.','No validation scores, timing, policy selection, inference or model payload access.','Simple one-user-message thinking-enabled production template path only; special literals/tools/images/effort-end unsupported.','Actual host pack tokenization and future model correctness/lifecycle require separate root qualification.'])
    payloads['record.json']=json_bytes(record)
    write_bundle(output,payloads)
    return record


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','source-root','tokenizer-dir','output'): cli.add_argument('--'+name,type=Path,required=True)
    cli.add_argument('--manifest-sha256',required=True);cli.add_argument('--self-sha256',required=True)
    args=cli.parse_args()
    read_verified(__file__,args.self_sha256,256*1024)
    data=read_verified(args.manifest,args.manifest_sha256,MANIFEST_CAP)
    record=prepare(data,args.manifest_sha256,args.source_root,args.tokenizer_dir,args.output,args.self_sha256)
    print(json.dumps(dict(completed=True,output=str(args.output),record_sha256=digest(json_bytes(record))),sort_keys=True))


if __name__=='__main__': main()
