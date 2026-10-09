"""Pure variable-length numerical-reference rules; no engine or file access."""
import array
import math
import re

VOCAB=248320
INPUTS={
 'train':(36004,'ddeded38b2f8e2fd10e735a9f6d382434049566bd8d891534fa7f6aa84c24796'),
 'validation':(37462,'1c7eb5878f521890c5f7e511bea7d1f73146b61a784d6c86169c16713bd36d3c')}
NUMBER=r'[0-9]+(?:\.[0-9]+)?'
SIGNED=r'-?[0-9]+(?:\.[0-9]+)?'


def require(ok,message):
    if not ok: raise ValueError(message)


def integer(text):
    require(type(text) is str and re.fullmatch(r'0|[1-9][0-9]{0,19}',text),'canonical integer')
    value=int(text);require(value<2**64,'integer overflow');return value


def expected_progress(n):
    require(type(n) is int and 32768<=n<=65536,'input length bounds')
    return list(range(8192,n-1,8192))+[n-1]


def input_identity(key,count,sha):
    require(key in INPUTS and type(count) is int and (count,sha)==INPUTS[key],'frozen input key/count/hash')


def validate_request(request,n):
    lines=request['protocol'];require(type(lines) is list and 1<=len(lines)<=140,'protocol cardinality')
    pp=expected_progress(n)
    require(lines[0]=='RESUME 0','fresh RESUME exactly once first')
    for at,(line,endpoint) in enumerate(zip(lines[1:],pp)):
        words=line.split()
        require(len(words)==5 and words[0]=='PP' and integer(words[1])==endpoint and integer(words[2])==n,'exact prefill progress')
        times=[float(x) for x in words[3:]]
        require(all(math.isfinite(x) and x>0 for x in times),'finite positive PP times')
    cursor=1+len(pp)
    require(len(lines)>cursor and lines[cursor]=='REUSED 0','fresh REUSED after exact PP')
    fields=lines[-1].split()
    require(len(fields)==16 and fields[0]=='DONE','exact16 DONE fields')
    integers={i:integer(fields[i]) for i in (1,2,6,7,8,9,10,11,12,14,15)}
    generated=integers[1]
    require(1<=generated<=64 and integers[2]==n and fields[5] in ('length','stop'),'DONE count/prompt/finish')
    require(fields[5]!='length' or generated==64,'length finish needs requested64')
    times=[float(fields[i]) for i in (3,4,13)]
    require(all(math.isfinite(x) for x in times) and times[0]>0 and times[1]>0 and times[2]>=0,'finite DONE times')
    require(integers[6]<=integers[7] and integers[8]==0 and integers[9]<=integers[10] and integers[14]==n and integers[15]==0,'DONE MTP/fresh/offload counters')
    output=lines[cursor+1:-1]
    require(len(output)==generated*2,'T/LP pair cardinality')
    ids=[];lps=[]
    for index in range(generated):
        t,lp=output[index*2:index*2+2]
        require(t.startswith('T '),'ordered T/LP')
        token=integer(t[2:]);require(token<VOCAB,'output ID bound');ids.append(token)
        words=lp.split();require(len(words)==7 and words[0]=='LP','all5 LP candidates')
        require(re.fullmatch(SIGNED,words[1]),'LP numeric syntax')
        values=[float(words[1])];candidate_ids=[]
        for item in words[2:]:
            pieces=item.split(':');require(len(pieces)==2 and re.fullmatch(SIGNED,pieces[1]),'LP candidate syntax')
            i=integer(pieces[0]);require(i<VOCAB,'LP ID bound');candidate_ids.append(i);values.append(float(pieces[1]))
        require(len(set(candidate_ids))==5 and candidate_ids[0]==token,'greedy LP candidate identity')
        require(all(math.isfinite(v) and v<=0 for v in values),'finite logprobs')
        require(values[0]==values[1],'selected LP value identity');lps.append(lp)
    require(request['ids']==ids and request['logprobs']==lps,'stored IDs/LP equal raw protocol')
    return dict(passed=True,finish_reason=fields[5],mtp_counts=[integers[6],integers[7]],generated=generated,prompt_tokens=n,prompt_ms=times[0],decode_ms=times[1],progress=pp)


def repeat_checks(actual,reference,live_comparison):
    return dict(ids_equal=actual['ids']==reference['ids'],all_LP_equal=actual['logprobs']==reference['logprobs'],mtp_equal=actual['mtp_counts']==reference['mtp_counts'],finish_equal=actual['finish_reason']==reference['finish_reason'],first_head_equal=actual['first_head']['sha256']==reference['first_head']['sha256'],all66_live_state_parts_equal=len(live_comparison['part_evidence'])==66 and not live_comparison['different_live_parts'])


def state_sizes(upto):
    require(type(upto) is int and 32767<=upto<=65535,'state upto bounds')
    cells=((upto+3)//4)*4
    return [8,117669888,368640,18432,6144,48]+[cells*512,cells*512,cells*16,cells*16,(upto//4+1)*512]*12


def finite_head(blob):
    require(type(blob) is bytes and len(blob)==VOCAB*4,'firsthead exact bytes')
    values=array.array('f');values.frombytes(blob)
    require(len(values)==VOCAB and all(math.isfinite(v) for v in values),'firsthead finite floats')
    return dict(floats=VOCAB,finite=True)
