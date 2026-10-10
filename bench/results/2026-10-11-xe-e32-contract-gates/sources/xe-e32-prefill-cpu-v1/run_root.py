import fcntl, importlib.util, json
from pathlib import Path

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE=Path(__file__).resolve().parent
WX=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
OWNER=B/'xestrata-clean-64k-comparison-v4/direct_owner.py'
spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.require(m.sha(OWNER)=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686','owner pin')
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')

def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=B/'xe-e32-prefill-cpu-root-v1';m.require(not out.exists(),'immutable new output');out.mkdir()
        record=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),gpu_executed=False,
                    full262144_runtime_qualified=False,scope='Production publication helper and actual Prefill::run entry guard with CPU stand-ins; plain and ASan/UBSan.')
        def save(): (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=m.Owner(out/'commands',save);record['commands']=owner.commands
        def cmd(label,argv,wall=60,env=ENV):
            e,so,se=owner.run(label,argv,env,HERE,wall=wall,text_cap=2<<20,rss_cap=4<<30,cpu=wall,total_cap=8<<20)
            m.require(m.closed(e) and e['exit_code']==0,'closed normal '+label)
            return so.read_text()
        save()
        try:
            files=['src/prefill/prefill.cpp','include/strata/prefill/publication.hpp','tools/test_prefill_publication.cpp']
            record['source_pins']={f:m.sha(WX/f) for f in files}
            text=(WX/files[0]).read_text();entry=text.split('bool Prefill::run(',1)[1].split('    const core::OnDevice on_device(m.device);',1)[0]
            m.require('publication::valid_span(n, pos0, ss.max_cells)' in entry,'guard precedes device selection and first enqueue')
            source='''#include "strata/prefill/publication.hpp"
#include <cassert>
#include <limits>
#include <iostream>
namespace strata::prefill {
namespace core { struct SessionState { int64_t max_cells = 262144; }; }
struct Impl { core::SessionState* ss; };
struct Prefill { Impl* impl_; bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err); };
bool Prefill::run('''+entry+'''    return true; // stand-in for the GPU body after the actual guarded entry
} catch (const std::exception& e) { err = e.what(); return false; }
}
int main() {
    using namespace strata::prefill;
    core::SessionState ss; Impl impl{&ss}; Prefill pf{&impl};
    int64_t tok=1; std::string err;
    assert(pf.run(&tok,262144,0,err) && err.empty());
    assert(pf.run(&tok,1,262143,err) && err.empty());
    assert(pf.run(nullptr,0,262144,err) && err.empty());
    for (const auto& p : {std::pair<int64_t,int64_t>{1,262144},{2,262143},{0,262145},{-1,0},{1,-1},{2,std::numeric_limits<int64_t>::max()-1}}) {
        assert(!pf.run(&tok,p.first,p.second,err));
        assert(err.find("physical context capacity")!=std::string::npos);
    }
    assert(!pf.run(nullptr,1,0,err) && err=="prefill: missing input tokens");
    std::cout << "actual prefill entry: final cell, full span, empty end, invalid/overflow/null inputs passed before GPU body\\n";
}
'''
            extract=out/'prefill-entry-exact.cpp';extract.write_text(source);record['actual_entry_extract_sha256']=m.sha(extract)
            for mode in ['plain','asan-ubsan']:
                options=['-O2'] if mode=='plain' else ['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
                for label,src in [('publication',WX/files[2]),('entry',extract)]:
                    exe=out/(label+'-'+mode)
                    cmd('compile-'+label+'-'+mode,['/usr/bin/g++','-std=c++20','-Wall','-Wextra','-Werror','-pthread',*options,'-I'+str(WX/'include'),str(src),'-o',str(exe)])
                    env=dict(ENV,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1') if mode!='plain' else ENV
                    output=cmd('run-'+label+'-'+mode,[str(exe)],wall=20,env=env)
                    record.setdefault('results',[]).append(dict(case=label,mode=mode,stdout=output,binary_sha256=m.sha(exe)))
            record['source_stable']=all(m.sha(WX/f)==h for f,h in record['source_pins'].items());m.require(record['source_stable'],'tested source stable')
            record['complete']=True
        except BaseException as exc: record['error']=repr(exc)
        finally:
            record['active']=owner.active is not None;record['passed']=record['complete'] and not record['active'] and not record.get('error');record['finished_utc']=m.utc();save()
        print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],error=record.get('error'),results=record.get('results'))))
        return 0 if record['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
