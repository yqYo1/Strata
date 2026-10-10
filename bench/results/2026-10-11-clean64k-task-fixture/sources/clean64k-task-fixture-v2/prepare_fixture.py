"""Root-owned CPU fixture preparation; never invokes a model or GPU.

Historical corpus prefix plus verified historical task/chat suffix. This is
not a newly rendered natural conversation or independent quality dataset.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
META=B/'coding-review-32k-fixture.json'
META_SHA='2df376e5f5e6cab14e5caee90acd7acf316ab401194c83f828b9182668834dc6'
OLD=B/'coding-review-32k-tokens.txt'
OLD_SHA='137fa1157c697295238d2fd487c09f9df60ab9cfee421e8c7e0b743b240af449'
SOURCE=B/'full-context-copy-off/coding-context-256k-tokens.txt'
SOURCE_SHA='cc29e4427bcacb21c9f7df2d1f1a7d41897fce5bd76a8ca43d8c3c34d914dd2a'
HISTORICAL_TOKENIZER_SHA='0a1d68e33f8563f2f3791d00e2ad33147eac6cb4b53460026e102c3554524703'
TOKENIZER=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-events-20261010/tools/strata_tokenizer.py')
TOKENIZER_SHA='9b0327e25aa62754406a924db6cc137e6edb1d93c7af76a4f3d14a8920ea72c8'
ASSETS=Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/tokenizer')
PINS={
    'vocab.json':('4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4',32<<20),
    'merges.txt':('4ba698529766c7a2f5052083ea10cdf2485577f5237a332ee291054eb11e3332',16<<20),
    'token_type.json':('5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b',4<<20),
    'tokenizer.json':('c0bd435efbc6c3b28964b47999afe16c4ad268c7c7c4e8cb3af410796a6f30bf',1<<20),
    'chat_template.jinja':('12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce',2<<20),
}


def require(value,why):
    if not value: raise RuntimeError(why)


def digest(data): return hashlib.sha256(data).hexdigest()


def pinned(path,expected,limit):
    require(path.is_file() and not path.is_symlink(),'regular pinned input required')
    with path.open('rb') as stream: data=stream.read(limit+1)
    require(len(data)<=limit,'input size bound')
    require(digest(data)==expected,'input identity mismatch: '+path.name)
    return data


def ids(data,lo,hi):
    words=data.decode('ascii').split()
    require(lo<=len(words)<=hi,'input token count')
    require(all(re.fullmatch('[0-9]{1,6}',v) for v in words),'canonical integer domain')
    values=list(map(int,words))
    require(all(0<=v<248320 for v in values),'token vocabulary domain')
    return values


def serial(values): return (' '.join(map(str,values))+'\n').encode('ascii')


def prepare(output):
    # All admission and tokenization precedes any output creation.
    rawmeta=pinned(META,META_SHA,1<<16); meta=json.loads(rawmeta)
    require(meta.get('tokens')==32768 and meta.get('sha256')==OLD_SHA
            and meta.get('source_sha256')==SOURCE_SHA,'historical metadata identities')
    require(meta.get('source')==str(SOURCE) and meta.get('file')==str(OLD),'historical exact paths')
    require(meta.get('tokenizer_source_sha256')==HISTORICAL_TOKENIZER_SHA
            and meta.get('suffix_roundtrip') is True,'historical suffix attestation')
    suffix=meta.get('suffix_ids'); text=meta.get('suffix_text')
    require(isinstance(text,str) and 0<len(text.encode())<=4096,'bounded historical suffix data')
    require(isinstance(suffix,list) and len(suffix)==49
            and all(type(v) is int and 0<=v<248320 for v in suffix),'exact49 suffix IDs')
    old=ids(pinned(OLD,OLD_SHA,1<<20),32768,32768)
    # Historical corpus contains262145 IDs; only constructed65536 enter GEN.
    source=ids(pinned(SOURCE,SOURCE_SHA,4<<20),262145,262145)
    require(old==source[:32719]+suffix and old[-49:]==suffix,'original32K body and full tail equality')
    code=pinned(TOKENIZER,TOKENIZER_SHA,1<<20)
    asset={name:pinned(ASSETS/name,sha,limit) for name,(sha,limit) in PINS.items()}
    cfg=json.loads(asset['tokenizer.json'])
    require(cfg.get('model')=='gpt2' and cfg.get('pre')=='qwen35'
            and cfg.get('vocab_size')==248320 and cfg.get('n_merges')==247587
            and cfg.get('add_bos_token') is False,'pack tokenizer metadata')
    # Import only exact admitted host implementation. No GGUF reader/model load.
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('fixture_verified_tokenizer',TOKENIZER)
    require(spec is not None and spec.loader is not None,'tokenizer import API')
    module=importlib.util.module_from_spec(spec)
    # Execute the admitted bytes, not a second file read after SHA admission.
    exec(compile(code,str(TOKENIZER),'exec'),module.__dict__)
    vocab=json.loads(asset['vocab.json']); types=json.loads(asset['token_type.json'])
    require(isinstance(vocab,dict) and len(vocab)==248320,'full vocabulary')
    tokens=[None]*248320
    for token,index in vocab.items():
        require(isinstance(token,str) and type(index) is int and 0<=index<248320
                and tokens[index] is None,'unique vocabulary indices')
        tokens[index]=token
    require(all(v is not None for v in tokens),'complete vocabulary indices')
    require(isinstance(types,list) and len(types)==248320 and all(type(v) is int for v in types),'token type geometry')
    merges=asset['merges.txt'].decode('utf-8').split('\n')
    require(len(merges)==247587 and cfg.get('pre_pattern')==module.QWEN35_PATTERN,'exact merge/pattern contract')
    # Same actual host Tokenizer constructor used by serve/server.py.
    tokenizer=module.Tokenizer(tokens,merges,types)
    require(tokenizer.encode(text,parse_special=True)==suffix,'actual host suffix encode differs')
    require(tokenizer.decode(suffix,errors='strict')==text,'actual host suffix decode differs')
    body=source[:65487]; fixture=body+suffix
    require(len(body)==65487 and len(fixture)==65536 and fixture[-49:]==old[-49:],'exact constructed geometry/tail')
    bodybytes=serial(body); fixturebytes=serial(fixture)
    manifest=dict(schema='clean64k-historical-task-fixture-v1',passed=True,active=False,
        builder_sha256=digest(Path(__file__).read_bytes()),
        source=dict(path=str(SOURCE),sha256=SOURCE_SHA,tokens=len(source)),
        historical32k=dict(path=str(OLD),sha256=OLD_SHA,tokens=32768),
        historical_metadata=dict(path=str(META),sha256=META_SHA),
        historical_tokenizer_source_sha256=HISTORICAL_TOKENIZER_SHA,
        actual_tokenizer=dict(path=str(TOKENIZER),sha256=TOKENIZER_SHA,
            api='Tokenizer(tokens, merges, token_types); encode(parse_special=True); decode(errors=strict)',
            pre=cfg['pre'],vocab_size=248320,n_merges=247587,special_ids=cfg['special_ids']),
        assets={name:dict(path=str(ASSETS/name),sha256=pin[0],bytes=len(asset[name])) for name,pin in PINS.items()},
        template_rendered=False,template_status='pinned asset recorded; historical suffix reused after exact API verification',
        body=dict(tokens=65487,serialization='ASCII decimal IDs separated by spaces, final LF',sha256=digest(bodybytes)),
        fixture=dict(path=str(output/'coding-review-65536-tokens.txt'),tokens=65536,sha256=digest(fixturebytes),
                     first_ids=fixture[:8],last_ids=fixture[-8:]),
        suffix=dict(tokens=49,text_sha256=digest(text.encode('utf-8')),ids_sha256=digest(serial(suffix)),
                    encode_exact=True,decode_exact=True,original32k_tail_exact=True),
        token_domain=dict(minimum=min(fixture),maximum=max(fixture),vocabulary_size=248320,all_valid=True),
        scope='historical corpus prefix plus verified49-token task/chat tail; no model inference',
        limitations=['Body prefix may cut source code; suffix closes code fence and request.',
                     'Historical corpus lineage overlaps32K; not independent quality data or natural new conversation.',
                     'Current host tokenizer differs from historical recorded source; exact suffix encode/decode must agree.',
                     'No canonical chat re-rendering of full body; only historical task/chat boundary restored.',
                     'No causal early-EOS repair proof or guaranteed64-token decode; root must evaluate baseline first.',
                     'No full262144 lifecycle or GPU qualification.'],
        gpu_executed=False,model_executed=False,environment_logged=False)
    output.mkdir(mode=0o700) # Exclusive new directory, never replace existing fixtures.
    with (output/'body-prefix-65487-tokens.txt').open('xb') as f: f.write(bodybytes)
    with (output/'coding-review-65536-tokens.txt').open('xb') as f: f.write(fixturebytes)
    with (output/'manifest.json').open('x') as f: f.write(json.dumps(manifest,indent=2)+'\n')
    return dict(passed=True,manifest=str(output/'manifest.json'),fixture=manifest['fixture'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    options=parser.parse_args()
    require(options.output.is_absolute() and options.output.parent.resolve()==B.resolve()
            and not options.output.exists() and len(str(options.output))<4096,'new direct child output under B required')
    # Existing shared lock only, no creation/replacement; held over preparation.
    with (B/'owned-v0141-measurement.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=prepare(options.output)
    print(json.dumps(result))
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as error:
        # No suffix text, token corpus, environment, repr or arbitrary traceback.
        print(json.dumps(dict(passed=False,error_type=type(error).__name__,
                              error='fixture preparation failed closed; no inference admission')),
              file=sys.stderr)
        raise SystemExit(1)
