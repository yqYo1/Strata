from pathlib import Path
import ast,json
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
s=(B/'run_gdn_quad_quiet_gates_v1.py').read_text();build=ast.parse((B/'build_gdn_host_event_cpu_v1.py').read_text())
pins=next(ast.literal_eval(n.value) for n in build.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets))
s=s.replace('"""Root-owned supervised host probes and initial diagnostic GPU differential; no model/performance."""','"""Root-owned diagnostic host/event attribution; instrumented synthetic component, no speed/adoption claim."""')
s=s.replace("HEAD='fdf4b3c52a68f8d925afb3a6519be4ce0b74156f'","HEAD='cd353f30a47dfd42818aace6f464bd227084e779'")
s=s.replace("BUILD=W/'build-sycl-gdn-quad-quiet-v1'","BUILD=W/'build-sycl-gdn-host-event-v1'").replace("OUT=B/'gdn-quad-quiet-gates-v1'","OUT=B/'gdn-host-event-runtime-v1'")
a=s.index('PINS=');z=s.index('\ndef sha',a);s=s[:a]+'PINS='+repr(pins)+s[z:]
s=s.replace("qualified_path=B/'gdn-quad-quiet-cpu-build-v1/record.json'","qualified_path=B/'gdn-host-event-cpu-build-v1/record.json'")
s=s.replace("scope='Requalified short diagnostic plus two fresh counterbalanced >=32768 quiet synthetic component service processes; no model/full-lifecycle/default-selector or kernel-only timing'","scope='Short traced correctness followed by two fresh counterbalanced 32768 event-capture diagnostic processes; host/device clocks nonadditive; instrumented synthetic component only, no clean timing/model/full-lifecycle/adoption'")
s=s.replace("sha(B/'research-20261009/gdn-quad-quiet-prefix-independent-review-round102.txt')","sha(B/'research-20261009/gdn-host-event-final-source-review-round107.txt')")
s=s.replace("[('timing-invalid-length','32767'),('timing-dirty-environment','32768')]","[('diagnostic-invalid-length','32767'),('diagnostic-dirty-environment','32768')]").replace("args=['--timing-prefix',total","args=['--diagnostic-prefix',total")
s=s.replace("'SYCL async error:' not in error","'SYCL async error:' not in error and 'asynchronous SYCL error:' not in error and 'PROFILE_QUERY_FAILED' not in error")
a=s.index("   r['quiet_environment']=dict(env)");z=s.index("   journal=run('kernel-journal'",a)
s=s[:a]+'''   r['diagnostic_environment_without_API_layers']=dict(env);r['diagnostic_component_samples']=[];save()
   assert not any(k in env for k in ('STRATA_TRACE','UR_ENABLE_LAYERS','ZE_ENABLE_VALIDATION_LAYER','ZEL_ENABLE_LOADER_LOGGING','LD_PRELOAD'))
   for order in ('legacy-first','quad-first'):
    r['active_gpu_stage']='diagnostic32768 '+order;save()
    log=run('diagnostic-'+order,[binary,'--diagnostic-prefix','32768','--chunk','2048','--order',order,'--samples','1'],wall=600)
    rows=list(csv.reader(log.read_text().splitlines()));events=[x for x in rows if x[0]=='DIAGNOSTIC'];prefixes=[x for x in rows if x[0]=='TIMING']
    assert len(rows)==106 and len(events)==96 and len(prefixes)==6 and len(rows[1])==21
    assert rows[0][0]=='DIAGNOSTIC_META' and 'capture_compiled=true' in rows[0] and 'host_device_not_additive=true' in rows[0]
    assert rows[-1]==['DIAGNOSTIC_SUMMARY','rows=96','recorded_pairs=1','exact_paired_chunks=true','profile_available=true','normal_buffers_released=true','host_device_not_additive=true','model=false','full_lifecycle=false','adopted=false','pass']
    identities={tuple(x[14:18]) for x in prefixes};assert len(identities)==1
    aggregated=[]
    for phase,pair,reverse in [('warmup',1,False),('warmup',2,True),('sample',1,False)]:
     expected_order=('quad-first' if order=='legacy-first' else 'legacy-first') if reverse else order
     for arm in ('legacy','quad'):
      ev=[x for x in events if x[1]==phase and x[2]==str(pair) and x[5]==arm]
      pr=[x for x in prefixes if x[1]=='diagnostic_'+phase and x[2]==str(pair) and x[5]==arm]
      assert len(ev)==16 and len(pr)==1;pr=pr[0]
      assert [int(x[7]) for x in ev]==list(range(0,32768,2048))
      expected_position='1' if (arm=='quad')==(expected_order=='quad-first') else '2'
      for x in ev:
       assert len(x)==21 and x[3:5]==[expected_order,expected_position] and x[8]=='2048'
       assert x[6]==('GdnRecQuadPipelineSG32' if arm=='quad' else 'GdnLegacyPipelineReference')
       assert math.isfinite(float(x[9])) and float(x[9])>0
       ns=[int(v) for v in x[10:]];assert all(v>=0 for v in ns)
       assert ns[7]==ns[6]-ns[5] and ns[10]==ns[9]-ns[8]
       assert ns[6]>=ns[5] and ns[9]>=ns[8]
       if arm=='legacy':assert ns[2]==0
      assert len(pr)==24 and pr[3:6]==[expected_order,expected_position,arm]
      assert pr[6:9]==['32768','2048','16'] and pr[10:12]==['32768','16'] and pr[13]=='16' and pr[-1]=='pass'
      assert pr[12]==('16' if arm=='quad' else '0') and pr[18]==('32' if arm=='quad' else '0')
      assert math.isfinite(float(pr[9])) and float(pr[9])>0
      assert abs(sum(float(x[9]) for x in ev)-float(pr[9]))<2e-8,'prefix service sum rounding'
      aggregated.append(dict(phase=phase,pair=pair,order=expected_order,arm=arm,position=int(expected_position),service_seconds=float(pr[9]),prevalidation_ns=sum(int(x[10]) for x in ev),queue_lookup_ns=sum(int(x[11]) for x in ev),admission_ns=sum(int(x[12]) for x in ev),recurrence_submit_host_ns=sum(int(x[13]) for x in ev),norm_submit_host_ns=sum(int(x[14]) for x in ev),recurrence_device_ns=sum(int(x[17]) for x in ev),norm_device_ns=sum(int(x[20]) for x in ev)))
    err=(OUT/('diagnostic-'+order+'.stderr')).read_text()
    assert all(x not in err for x in ('FAIL-STOP','SYCL async error:','asynchronous SYCL error:','PROFILE_QUERY_FAILED','DIAGNOSTIC_DENIED','DIAGNOSTIC_UNAVAILABLE'))
    r['diagnostic_component_samples'].append(dict(starting_order=order,aggregate_rows=aggregated,repeat_digests=list(identities)[0],per_chunk_rows=96,normal_buffers_released=True));save()
   assert r['diagnostic_component_samples'][0]['repeat_digests']==r['diagnostic_component_samples'][1]['repeat_digests']
'''+s[z:]
s=s.replace("r.update(component_timing_validated=True,component_definition='host steady recurrence/norm completion service sum, per-chunk admission included; transfers/checks excluded; interleaved matching chunks',active_gpu_stage=None,complete=True,passed=True)","r.update(diagnostic_event_capture_validated=True,clean_component_timing_validated=False,component_definition='Instrumented host service plus separate backend recurrence/norm timestamps; per-chunk admission observed. Asynchronous intervals overlap: do not add/subtract host and device values; transfers/checks excluded.',active_gpu_stage=None,complete=True,passed=True)")
out=B/'run_gdn_host_event_runtime_v1.py';assert not out.exists();ast.parse(s);out.write_text(s)
print(out)
