from pathlib import Path
import fcntl,hashlib,json
B=Path(__file__).parent;Q=B/'xestrata-diagnostic-contract-root-v3'
with (B/'owned-v0141-measurement.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=Q/'run_first_diagnostic_v7.py';text=old.read_text()
    assert hashlib.sha256(old.read_bytes()).hexdigest()=='09c10203e075814af9bf95b3f0a7c646e364dcd9fa76dc89a3d57d71dff06bd6'
    text=text.replace("[8192, 16384, 24576, 32767]","[4096, 8192, 12288, 16384, 20480, 24576, 28672, 32767]")
    text=text.replace("    return root, binary, [str(binary)] + args, tokens, baseline, profile","    args[args.index('--prefill') + 1] = '4096'\n    return root, binary, [str(binary)] + args, tokens, baseline, profile")
    text=text.replace("expected_physical_expert_slots=144,","expected_physical_expert_slots=144, prefill_chunk=4096,\n                  prior_8k_admission='source-supported hostUSM cache PASS; READY50MiB<512 reserve; no GEN launched',")
    text=text.replace('fixed8K complete PP positions','fixed4K complete PP positions')
    new=Q/'run_first_diagnostic_v8.py';assert not new.exists();compile(text,str(new),'exec');new.write_text(text)
    t=(B/'qualify_xestrata_diagnostic_v7_cpu.py').read_text().replace("OUT=B/'xestrata-diagnostic-v7-cpu-root-v3'","OUT=B/'xestrata-diagnostic-v8-cpu-root-v4'").replace("source=Q/'run_first_diagnostic_v7.py'","source=Q/'run_first_diagnostic_v8.py'")
    t=t.replace("    assert ast.dump(get(v4,'validate'))==ast.dump(get(v5,'validate'))","    assert '4096, 8192, 12288, 16384, 20480, 24576, 28672, 32767' in source.read_text()\n    baseline=copy.deepcopy(baseline)\n    baseline['requests'][0]['protocol']=[x for x in baseline['requests'][0]['protocol'] if not x.startswith('PP ')]\n    baseline['requests'][0]['protocol'][:0]=[f'PP {p} 32768 100 100.0' for p in [4096,8192,12288,16384,20480,24576,28672,32767]]")
    t=t.replace("    tests=ns['tests']","    tests=ns['tests'];tests[0]['case']='synthetic4K-PP-geometry-plus-existing-finite-LP-template'")
    dest=B/'qualify_xestrata_diagnostic_v8_cpu.py';assert not dest.exists();compile(t,str(dest),'exec');dest.write_text(t)
    print(json.dumps(dict(prepared=True,controller=str(new),sha256=hashlib.sha256(new.read_bytes()).hexdigest())))
