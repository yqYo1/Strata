#!/usr/bin/env python3
"""Main ordinary-host tests of exact extracted CLI admission; never initializes SYCL."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
C = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-pr6-prefill-restore')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root = B / 'main-pr6-harness-admission-v2'
    target = A / 'own-pr6-harness-admission-v2.json'
    source = C / 'validation/pr6-restore/harness/prefill_api.cpp'
    assert not root.exists() and not target.exists()
    assert sha(source) == 'c6d9684f87a86ea1da3a947eb2130dd88baadc2cdbbb101321d10da10153bd46'
    text = source.read_text()
    number = text[text.index('int64_t number('):text.index('\nstd::string quoted(')]
    parser = text[text.index('struct Options {'):text.index('\nstd::map<std::string, std::string> environment(')]
    prefix = '''// Exact source extraction; repository LICENSE. Host CLI checks only.
#include <charconv>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
'''
    suffix = '''
int main(int argc,char** argv){try{const auto o=parse(argc,argv);
std::cout<<"{\\"accepted\\":true,\\"count\\":"<<o.count<<",\\"small\\":"
<<(o.validation_small_input?"true":"false")<<"}\\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\\n";return 2;}}
'''
    fixture = A / 'pr6_harness_admission_v2.cpp'
    assert not fixture.exists()
    fixture.write_text(prefix + number + '\n' + parser + suffix)
    owner_source = A.parents[2] / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert sha(owner_source) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('pr6_host_admission_owner', owner_source)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        inputs = root / 'inputs'; inputs.mkdir(); (inputs / 'native_experts.txt').write_text('')
        for name in ('tokens.bin','ple.gguf','shard.gguf'):
            (inputs / name).write_bytes(b'')
        record = dict(schema='strata-pr6-harness-host-admission-v2', passed=False, complete=False,
            active=False, commands=[], source=str(source), source_sha256=sha(source),
            fixture=str(fixture), fixture_sha256=sha(fixture),
            exact_extracted_number_sha256=hashlib.sha256(number.encode()).hexdigest(),
            exact_extracted_parser_sha256=hashlib.sha256(parser.encode()).hexdigest(),
            GPU_executed=False, legacy_engine_built=False, samples=[],
            scope='Exact CLI parser and integer admission only. Empty private files satisfy pathname checks; they are never read as model or benchmark data. No kernel, model, provider, physical256K or historical600 qualification.')
        owner = None
        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner = m.Owner(root/'owned',persist); record['commands']=owner.commands
        env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
        def run(label,argv):
            c,out,err=owner.run(label,argv,env,A.parents[2],wall=30,cpu=30,rss_cap=1<<30,text_cap=1<<20,file_cap=4<<20,total_cap=16<<20)
            assert m.closed(c)
            return c,out.read_text(),err.read_text()
        try:
            binary=root/'host-admission'
            c,_,_=run('host-build',['/usr/bin/g++','-std=c++20','-O2','-Wall','-Wextra','-Werror',str(fixture),'-o',str(binary)])
            assert c['exit_code']==0
            record['binary_sha256']=sha(binary)
            base={'--model-root':str(inputs),'--shard':str(inputs/'shard.gguf'),'--tokens-file':str(inputs/'tokens.bin'),
                '--ple-file':str(inputs/'ple.gguf'),'--token-count':'32768','--context':'65536','--chunk':'8192',
                '--first-chunk':'0','--residual-gpu-tokens':'0','--ring-slots':'8','--device':'0','--kv':'int8','--logit-id':'0'}
            cases=[('long_min',{},[],True),('full_context',{'--token-count':'262144','--context':'262144'},[],True),
                ('short_unflagged',{'--token-count':'32767'},[],False),
                ('single',{'--token-count':'1','--context':'8192'},['--validation-small-input'],True),
                ('odd',{'--token-count':'17','--context':'8192'},['--validation-small-input'],True),
                ('short_max',{'--token-count':'32767'},['--validation-small-input'],True),
                ('long_flagged',{},['--validation-small-input'],False),
                ('zero',{'--token-count':'0'},['--validation-small-input'],False),
                ('context_below_chunk',{'--token-count':'1','--context':'8191'},['--validation-small-input'],False),
                ('context_too_short',{'--token-count':'65537'},[],False),
                ('context_overflow',{'--context':'262145'},[],False),
                ('count_overflow',{'--token-count':'262145','--context':'262145'},[],False),
                ('int64_overflow',{'--token-count':'9223372036854775808'},[],False),
                ('trailing_integer',{'--token-count':'32768x'},[],False),
                ('duplicate_flag',{'--token-count':'1'},['--validation-small-input','--validation-small-input'],False),
                ('unknown_option',{},['--unrecognized','1'],False)]
            for label,updates,tail,accepted in cases:
                opts={**base,**updates}; argv=[str(binary),*[x for pair in opts.items() for x in pair],*tail]
                c,out,err=run(label,argv)
                assert c['exit_code']==(0 if accepted else 2),(label,c,out,err)
                if accepted:
                    result=json.loads(out);assert result==dict(accepted=True,count=int(opts['--token-count']),small='--validation-small-input' in tail)
                    assert not err
                else:
                    assert not out and err.strip()
                    result=dict(accepted=False,error=err.strip())
                record['samples'].append(dict(case=label,expected_accepted=accepted,result=result,passed=True))
            assert sha(source)==record['source_sha256']
            record.update(passed=True,complete=True)
        except BaseException as error:
            record['error']=repr(error)
        finally:
            record['ownership_closed']=owner.active is None and all(m.closed(c) for c in owner.commands)
            for c in owner.commands:
                for old in c['owners']+[c['direct_identity']]:
                    now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
            persist();target.write_bytes((root/'record.json').read_bytes())
        print(json.dumps(dict(receipt=str(target),passed=record['passed'],checks=len(record['samples']),GPU_executed=False,error=record.get('error'))))
        return 0 if record['passed'] and record['ownership_closed'] else 1


if __name__=='__main__':
    sys.exit(main())
