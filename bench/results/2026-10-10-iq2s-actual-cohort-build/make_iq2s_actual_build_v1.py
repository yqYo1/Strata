from pathlib import Path
import ast
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');p=B/'build_native_service_calibration_cpu_v4.py';s=p.read_text();source=ast.parse((B/'archive_iq2s_real_cohort_source_v1.py').read_text());pins=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='files' for t in n.targets))
s=s.replace('diag-sycl-native-role-plan-v0141-20261010','perf-sycl-iq2s-nt1-index-spread-20261010').replace("out = B/'native-service-calibration-cpu-build-v4'","out = B/'iq2s-actual-cohort-cpu-build-v1'").replace("build = W/'build-native-service-calibration-v1'","build = W/'build-iq2s-actual-cohort-release-v1'").replace("assert build.is_dir(), 'existing qualified build for metadata-only rebuild'","assert not build.exists(), 'fresh source-only build'")
a=s.index('PINS =');z=s.index('\n\ndef sha',a);s=s[:a]+'PINS = '+repr(pins)+s[z:]
s=s.replace("'-DGGML_SOURCE_DIR='+str(G/'ggml')])","'-DGGML_SOURCE_DIR='+str(G/'ggml'),'-DSTRATA_CALIBRATION_IQ2S_INDEX_CHECK=ON'])")
s=s.replace("'--parallel','6'","'--parallel','4'").replace("len(project)==11, 'eleven harness/production translation units'","len(project)==12, 'twelve harness/production translation units'")
s=s.replace("record.update(complete=True,passed=True,binary=",'''# Compare existing baseline full compile flags; exclude only new private macro.
            import shlex
            baseline=B/'native-service-calibration-cpu-build-v4/record.json'
            old=json.loads(baseline.read_text());assert old['passed'] and old['complete'] and not old['active'] and not old['cleanup'] and not old['survivors']
            oldcc=Path(old['compile_commands']['path']);assert sha(oldcc)==old['compile_commands']['sha256']
            oldroot=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-role-plan-v0141-20261010');oldbuild=oldcc.parent
            def canon(c,root,directory):
                return [v.replace(str(directory),'{BUILD}').replace(str(root),'{ROOT}') for v in shlex.split(c['command']) if v!='-DSTRATA_CALIBRATION_IQ2S_INDEX_CHECK=1']
            before={c['file'].replace(str(oldroot),'{ROOT}'):canon(c,oldroot,oldbuild) for c in json.loads(oldcc.read_text())}
            after={c['file'].replace(str(W),'{ROOT}'):canon(c,W,build) for c in commands}
            common=set(before)&set(after);diff=[n for n in sorted(common) if before[n]!=after[n]]
            record['baseline_compile_flag_comparison']=dict(receipt_sha256=sha(baseline),common_units=len(common),mismatches=diff,new_units=sorted(set(after)-set(before)),missing_units=sorted(set(before)-set(after)),only_excluded_flag='-DSTRATA_CALIBRATION_IQ2S_INDEX_CHECK=1')
            assert common and not diff and not(set(before)-set(after)), 'baseline flags changed'
            helper=[c for c in project if c['file'].endswith('/iq2s_index_dot.cpp')];assert len(helper)==1
            assert all(v in helper[0]['command'] for v in ('-mavx2','-mfma','-mf16c','-fp-model=precise'))
            record.update(complete=True,passed=True,binary=''')
s=s.replace("out.mkdir(mode=0o700)","assert subprocess.check_output(['/usr/bin/git','-C',str(W),'rev-parse','HEAD'],text=True).strip()=='d68516f4baae2f9ddd80d95494e6332532d4017a'\n        assert not subprocess.check_output(['/usr/bin/git','-C',str(W),'status','--porcelain'],text=True)\n        out.mkdir(mode=0o700)")
out=B/'build_iq2s_actual_cohort_cpu_v1.py';assert not out.exists();ast.parse(s);out.write_text(s);print(out)
