"""Compile the actual changed SYCL caller and replace only its archive member."""
import fcntl
import importlib.util
import json
from pathlib import Path
import shlex
import shutil

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE=Path(__file__).resolve().parent
OWNER=B/'xestrata-clean-64k-comparison-v4/direct_owner.py'
spec=importlib.util.spec_from_file_location('qualified_owner',OWNER)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
require=m.require;sha=m.sha
require(sha(OWNER)=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686','owner pin')

def pin(path):
    path=Path(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'prefill-skip-order-linked-root-r2';require(not out.exists(),'new output');out.mkdir(mode=0o700)
    record=dict(active=True,complete=False,passed=False,gpu_executed=False,model_executed=False,adopted=False,
                full262144_qualified=False,controller_sha256=sha(__file__),started_utc=m.utc(),
                scope='one skipped-entry issuer publication wait; actual SYCL TU and link only')
    def save():
        temp=out/'record.json.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(out/'record.json')
    owner=m.Owner(out/'commands',save);record['commands']=owner.commands
    def run(label,args,env,wall=10):
        e,so,se=owner.run(label,args,env,HERE,wall=wall,text_cap=8<<20,rss_cap=6<<30,
                          total_cap=256<<20,cpu=wall,file_cap=64<<20)
        require(m.closed(e) and e['exit_code']==0,'normal owned build command '+label)
        return so
    save()
    try:
        source=HERE/'prefill.cpp';require(sha(source)=='e0685a85bc391b6ef916c33ea3f959680088832b8a9a7d706668acba3c47f8d0','changed source pin')
        cpu=B/'prefill-skipped-entry-order-cpu-root-r1/record.json'
        check=json.loads(cpu.read_text());require(check['passed'] and not check['active'] and check['complete'],'CPU contract gate')
        record['CPU_qualification']=pin(cpu);record['source']=pin(source)
        tu_path=B/'prefill-gemm-only-tu-root-v1/record.json'
        require(sha(tu_path)=='23abeff86c8b6727dea7618db2e6d36bd13d0adf9008481219fa8525278c3428','original TU recipe pin')
        prior_tu=json.loads(tu_path.read_text());require(prior_tu['passed'] and not prior_tu['active'],'original TU closed')
        link_path=B/'prefill-gemm-only-linked-root-v1/record.json'
        prior_link=json.loads(link_path.read_text());require(prior_link['passed'] and not prior_link['active'],'prior link closed')
        record['prior_TU_receipt']=pin(tu_path);record['prior_link_receipt']=pin(link_path)
        cached=prior_link['cached_dependencies_before']
        for path,p in cached.items():require(sha(path)==p['sha256'],'cached dependency unchanged '+path)
        record['cached_dependencies_before']=cached
        for path,p in prior_tu['quoted_relative_include_files'].items():require(sha(path)==p['sha256'],'quoted header pin')
        for name,field in [('prefill_route_census.hpp','header'),('prefill_service_ledger.hpp','service_header')]:
            header=B/'prefill-gemm-only-tu-root-v1/include/strata'/name
            require(sha(header)==prior_tu[field]['sha256'],'actual injected header pin')
        env=dict(prior_tu['environment']);record['environment']=env
        compiler=Path(prior_tu['effective_command'][0]);record['compiler']=pin(compiler)
        obj=out/'prefill.cpp.o';dep=out/'prefill.cpp.d'
        argv=list(prior_tu['effective_command'])
        argv[argv.index('-o')+1]=str(obj);argv[-1]=str(source)
        argv+=['-MMD','-MF',str(dep)]
        record['effective_compile']=argv
        run('compile-SYCL-prefill',argv,env,180)
        record['object']=pin(obj)
        dependency_text=dep.read_text().replace('\\\n',' ')
        paths=shlex.split(dependency_text.split(':',1)[1]);require(len(paths)<2000,'dependency count cap')
        record['actual_included_dependencies']={path:pin(path) for path in sorted(set(paths))}
        archive=out/'libstrata_prefill.a'
        old_archive=B/'prefill-gemm-only-linked-root-v1/libstrata_prefill.a'
        require(sha(old_archive)==prior_link['archive_identity']['sha256'],'prior archive pin')
        record['prior_archive']=pin(old_archive);shutil.copyfile(old_archive,archive)
        before=run('archive-list',['/usr/bin/ar','t',str(archive)],env).read_text().splitlines()
        require(before.count('prefill.cpp.o')==1,'unique changed member')
        member_pins={}
        for i,name in enumerate(before):member_pins[name]=pin(run('member-before-'+str(i),['/usr/bin/ar','p',str(archive),name],env))
        run('replace',['/usr/bin/ar','r',str(archive),str(obj)],env)
        run('index',['/usr/bin/ar','s',str(archive)],env)
        after=run('archive-list-after',['/usr/bin/ar','t',str(archive)],env).read_text().splitlines()
        require(after==before,'archive member order/count')
        comparisons=[]
        for i,name in enumerate(after):
            p=pin(run('member-after-'+str(i),['/usr/bin/ar','p',str(archive),name],env))
            expected=record['object']['sha256'] if name=='prefill.cpp.o' else member_pins[name]['sha256']
            require(p['sha256']==expected,'exact replaced/unchanged member')
            comparisons.append(dict(member=name,before=member_pins[name],after=p,replaced=name=='prefill.cpp.o'))
        record['member_comparison']=comparisons
        link=list(prior_link['effective_link']);binary=out/'strata'
        link[link.index('-o')+1]=str(binary)
        require(link.count(str(old_archive))==1,'one prior archive link input')
        link[link.index(str(old_archive))]=str(archive)
        record['effective_link']=link
        run('link',link,env,180)
        record['binary']=pin(binary);record['archive']=pin(archive)
        for path,p in cached.items():require(sha(path)==p['sha256'],'cached dependency still unchanged')
        for path,p in record['actual_included_dependencies'].items():require(sha(path)==p['sha256'],'included dependency stable')
        require(sha(source)==record['source']['sha256'],'source stable')
        record.update(complete=True,passed=True,linked=True,all_other_members_unchanged=True,cached_inputs_unchanged=True)
    except BaseException as exc:record['error']=repr(exc)
    finally:
        record['active']=owner.active is not None;record['finished_utc']=m.utc();save()
    print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],error=record.get('error'))))
    raise SystemExit(0 if record['passed'] else 1)
