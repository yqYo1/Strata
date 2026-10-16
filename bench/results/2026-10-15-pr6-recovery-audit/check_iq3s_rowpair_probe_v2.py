#!/usr/bin/env python3
"""Main-owned CPU-only candidate selection, serialized after model gates."""
import fcntl
import importlib.util
import json
import os
import statistics
from pathlib import Path
from check_gemm_orientation import P,B,D,OWNER,sha

O=B/'main-primary-iq3s-rowpair-probe-root-v2'
C=D/'iq3s-nt1-rowpair-v2'

def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paired=json.loads((D/'boundary-audit-long-pair-v1.json').read_text());assert paired['complete']
        for mode in ['long-fixed','long-auto']:
            closed=json.loads((D/('boundary-audit-'+mode+'-v1.json')).read_text())
            assert closed['complete'] and not closed['active'] and not closed['fresh_faults']
            assert all(c['direct_child_reaped'] and c['session_empty'] and c['observation_complete'] for c in closed['commands'])
        seed=json.loads((D/'boundary-service-build-v1.json').read_text())
        assert seed['passed'] and seed['complete'] and not seed['active']
        assert all(sha(P/n)==h for n,h in seed['source_pins'].items())
        native=B/'main-primary-native-build-root-v1/record.json';nr=json.loads(native.read_text())
        assert nr['passed'] and nr['complete'] and not nr['active'] and not nr['GPU_executed']
        compiler=Path(nr['compiler']['path']);assert sha(compiler)==nr['compiler']['sha256']
        assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
        spec=importlib.util.spec_from_file_location('owner',OWNER)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        for mode in ['long-fixed','long-auto']:
            for c in json.loads((D/('boundary-audit-'+mode+'-v1.json')).read_text())['commands']:
                for i in c['owners']:
                    now=m.identity(i['pid'])
                    assert now is None or now['start_ticks']!=i['start_ticks'] or now['state']=='Z'
        pins=json.loads((C/'pins.json').read_text());manifest=json.loads((C/'manifest.json').read_text())
        assert all(sha(Path(n))==h for n,h in pins['source_sha256'].items())
        assert all(sha(C/n)==h for n,h in manifest['artifact_sha256'].items())
        assert sha(C/'source.diff')=='910fceecee32adde7d3fa6e34f0b54665ea1b3bb4f895ff652bd9103f702b6eb'
        assert not O.exists();O.mkdir(mode=0o700)
        build=Path(seed['build_path']);dep=Path(pins['configured_dependency'])
        archives=[build/'ggml/src/libggml-cpu.a',build/'ggml/src/libggml-base.a']
        env=dict(seed['environment']);assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        core=min(os.sched_getaffinity(0))
        topology={}
        for n in ['topology/thread_siblings_list','cpufreq/scaling_governor','cpufreq/scaling_cur_freq']:
            path=Path('/sys/devices/system/cpu')/('cpu'+str(core))/n
            if path.exists():topology[n]=path.read_text().strip()
        compile_commands=build/'compile_commands.json'
        dependency_commands=[c for c in json.loads(compile_commands.read_text()) if c['file'].startswith(str(dep)+'/')]
        r=dict(active=True,passed=False,complete=False,GPU_executed=False,product_applied=False,commands=[],environment=env,
            recipe_sha256=sha(Path(__file__)),proposal_manifest_sha256=sha(C/'manifest.json'),
            seed_build_receipt_sha256=sha(D/'boundary-service-build-v1.json'),native_build_receipt_sha256=sha(native),
            compiler=nr['compiler'],dependency_revision=nr['dependency_revision'],dependency_source_pins=pins['source_sha256'],
            archive_sha256={str(p):sha(p) for p in archives},dependency_compile_commands=dependency_commands,
            compile_commands_sha256=sha(compile_commands),core=core,topology_before=topology,source_pins=seed['source_pins'],
            performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
            scope='Synthetic IQ3_S nt1/2560/640 CPU component only. OriginalGGML/exact/candidate, same archive. Existing archive source/build receipt and current hashes are recorded; sanitizer does not instrument prebuilt archive internals.')
        def persist(): (O/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=m.Owner(O/'owned',persist);r['commands']=owner.commands;persist()
        def run(label,args,wall=180):
            e,out,err=owner.run(label,args,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),
                text_cap=64<<20,rss_cap=8<<30,total_cap=512<<20,file_cap=64<<20)
            assert m.closed(e) and e['exit_code']==0,(label,e.get('error'),err.read_text()[-5000:],out.read_text()[-3000:])
            return out,err
        try:
            r['correctness']={}
            for sanitized in [False,True]:
                label='asan-ubsan' if sanitized else 'clean';folder=O/label
                args=['/usr/bin/python3',str(C/'build_probe.py'),'--compiler',str(compiler),
                    '--ggml-root',str(dep),'--cpu-archive',str(archives[0]),'--base-archive',str(archives[1]),'--out-dir',str(folder)]
                if sanitized:args.append('--sanitize')
                run('build-'+label,args,600)
                exe=folder/'fixture';r.setdefault('binaries',{})[label]=sha(exe)
                out,err=run('correctness-'+label,['/usr/bin/taskset','-c',str(core),str(exe),'--correctness'],180)
                rows=[json.loads(s) for s in out.read_text().splitlines()]
                assert rows[-1]['kind']=='correctness_close'
                assert rows[-1]['cases']==353 and rows[-1]['pass'] and rows[-1]['criterion']=='bitwise'
                cases=[x for x in rows if x['kind']=='correctness_case'];assert len(cases)==344
                assert all(x['bitwise_pass'] and x['guards'] for x in cases)
                r['correctness'][label]=dict(rows=rows,stdout_sha256=sha(out),stderr=err.read_text(),binary_sha256=sha(exe))
                print('CPU rowpair correctness PASS '+label,flush=True);persist()
            r['timings']={}
            for regime,banks,inner in [('hot',1,8),('stream',64,2)]:
                exe=O/'clean/fixture';assert sha(exe)==r['binaries']['clean']
                out,err=run('timing-'+regime,['/usr/bin/taskset','-c',str(core),str(exe),'--timing',regime,str(banks),'12',str(inner),'--correctness-approved-by-main'],600)
                rows=[json.loads(s) for s in out.read_text().splitlines()]
                assert rows[-1]['kind']=='timing_close' and rows[-1]['complete'] and rows[-1]['samples']==144
                samples=[x for x in rows if x['kind']=='timing_sample'];assert len(samples)==144
                assert all(not x['migration_observed'] and x['cpu_start']==x['cpu_end']==core and x['jobs']==banks*inner and x['output_rows']==banks*inner*640 for x in samples)
                medians={arm:statistics.median(x['ns_per_output_row'] for x in samples if x['arm']==arm) for arm in ['ggml','existing_exact','rowpair']}
                r['timings'][regime]=dict(rows=rows,medians_ns_per_output_row=medians,
                    candidate_over_ggml=medians['rowpair']/medians['ggml'],candidate_over_exact=medians['rowpair']/medians['existing_exact'],stderr=err.read_text())
                print('CPU rowpair '+regime+' '+json.dumps(r['timings'][regime]['medians_ns_per_output_row']),flush=True);persist()
            assert all(sha(P/n)==h for n,h in seed['source_pins'].items())
            assert all(sha(Path(p))==h for p,h in r['archive_sha256'].items())
            r.update(passed=True,complete=True)
        except BaseException as error:
            r['error']=repr(error);raise
        finally:
            r['active']=False;persist();(D/'iq3s-rowpair-probe-v2.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=True,product_applied=False,timings={n:v['medians_ns_per_output_row'] for n,v in r['timings'].items()})),flush=True)

if __name__=='__main__':main()
