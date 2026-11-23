"""Reusable owned prefill workspace,dispatch and upload admission readers.

Main-authored, repository LICENSE. Device/workspace readers are preserved from
prefill_window_model.py at570f6903e; eager-route reader is preserved from the
owned eager_fullfit_bench.py at5f1a95453. No XeStrata inputs. Closed comparison
command modes live in Git. Retain these readers for the next same-image
input-length curve and its source/workspace/dispatch admission.
"""
import hashlib
import json
from pathlib import Path
import re
import coherent_model_basis as capacity
import gu_small_bench as dispatch_reader
import prefill_window_device as device
import qsa16_stream_bench as core

P,B,RESULT=capacity.P,capacity.B,capacity.RESULT
require,sha=capacity.require,capacity.sha
STRIDE,ROW=2662400,20480
RETIRED_CONTROL='tools/sycl/prefill_window_model.py'


def historical_controls(pins,revision):
    """Recover a closed control without retaining its obsolete CLI in the tree.

    This exception names only the retired campaign,not changed native inputs or
    live qualification readers. Current campaign controls are pinned separately.
    Original strict build/device admission remains unchanged.
    """
    recovered={p:hashlib.sha256(raw).hexdigest()
        for p,raw in device.build.source.blobs(P,revision,pins)}
    require(recovered==pins and RETIRED_CONTROL in pins and not (P/RETIRED_CONTROL).exists()
        and not device.build.source.git(P,'ls-tree','HEAD','--',RETIRED_CONTROL),
        'exact archived closed controller and committed retirement required')
    live={p:h for p,h in pins.items() if p!=RETIRED_CONTROL}
    require(device.build.source.pin(P,list(live))['source_sha256']==live,
        'non-retired historical control or native input changed')
    return dict(revision=revision,recovered_files=len(pins),unchanged_live_files=len(live),
        retired_control=dict(path=RETIRED_CONTROL,sha256=pins[RETIRED_CONTROL]),
        native_recompiled=False,new_GPU_qualification=False)


def device_gate():
    # Reuse the exact closed build and component receipts. The old all-tools
    # reader also froze an unused campaign CLI;recover that single deleted
    # controller from Git rather than refreshing a receipt or restoring it.
    build=device.build
    native=json.loads((build.ROOT/'record.json').read_text())
    require(build.ur_validation.committed(RESULT,(build.KEY,))==build.compact(native)
        and native['stage']=='built' and native['passed'] and native['complete']
        and native['ownership_closed'] and not native['active'],'closed original candidate build required')
    build_controls=historical_controls(native['controller_sources'],native['source_commit'])
    original,_=build.original.verify_original(expected=native['foundation'])
    for pin in original['dependencies']:
        require(all(sha(Path(pin['root'])/p)==h for p,h in pin['source_sha256'].items()),
            'original committed dependency changed')
    require(all(sha(build.ROOT/'source'/p)==h for p,h in native['source_files'].items())
        and build.guards.driver_guard()==native['driver']
        and build.bank_gate()==native['bank_gate']
        and build.decode_reference()==native['retained_decode_reference'],
        'native composition,platform or independent baseline changed')
    require(all(sha(p)==h for name in ('compiled','generated','links','objects','produced')
        for p,h in native['actual_inputs'][name].items()) and len(native['binaries'])==8
        and all(sha(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['bytes']
            for v in native['binaries'].values()),'exact compiler/link inputs or candidate fixtures changed')
    value=json.loads((device.ROOT/'record.json').read_text())
    require(build.ur_validation.committed(RESULT,(device.KEY,))==device.compact(value),
        'original committed component receipt changed')
    device_controls=historical_controls(value['sources'],value['source_commit'])
    require(sha(build.ROOT/'record.json')==value['native_record_sha256']
        and native['binaries']==value['binaries'] and build.guards.driver_guard()==value['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==value['boot']
        and value['fixture_environment']==device.FIXTURE_ENV
        and build.ownership.artifact_bytes(device.ROOT)<=value['whole_bytes'],
        'same-build component identity,fixture contract,platform or budget changed')
    require(value['stage']=='closed' and not value['active'] and value['passed'] and value['complete']
        and value['ownership_closed'] and [v['mode'] for v in value['results']]==device.STAGES
        and all(v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults']
            and v['boot_unchanged'] for v in value['results'])
        and not value['results'][0]['GPU_executed']
        and all(v['GPU_executed'] and v['native_exit_code']==0 for v in value['results'][1:]),
        'all seven closed same-build component stages required')
    return native,dict(key=device.KEY,source_commit=value['source_commit'],
        record_sha256=sha(device.ROOT/'record.json'),binaries=value['binaries'],
        historical_controller_reuse=dict(build=build_controls,device=device_controls),
        scope='Component address/math/closure only;actual allocation,model/task/phase/physical-context gates not inherited.')


def records(text,prefix):
    result=[]
    for line in text.splitlines():
        if line.startswith(prefix):
            pairs=re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)',line.split(';',1)[0])
            require(len({k for k,v in pairs})==len(pairs),'duplicate '+prefix);result.append(dict(pairs))
    return result


def one(text,prefix):
    values=records(text,prefix);require(len(values)==1,'missing/duplicate '+prefix);return values[0]


def workspace(rows,split,relocated):
    # Independent byte census of the pinned owned planner;no runtime allocation.
    primary=65536 if split else rows
    payloads=[primary*10240,*([rows*40]*4),rows*16,8192*5120,8192*2560,
        8192*1280,8192*5120,8192*5120,rows*4,8192*2048,*([8192*1280]*3),4]
    reserved=[(x+255)//256*256+256 for x in payloads]
    tail=(rows-primary)*10240;extra=(tail+255)//256*256+256 if tail else 0
    if relocated:extra+=reserved[6]+reserved[12]
    return dict(region=sum(reserved)-(reserved[6]+reserved[12] if relocated else 0),
        tail_payload=tail,extra=extra,mixed_primary=min(rows,65536)*5120,mixed_tail=max(0,rows-65536)*5120)


def eager_route(text,plan):
    require(plan['upload'] in ('jit','eager'),'unplanned full-fit upload policy')
    matches=re.findall(r'^strata prefill fullfit upload: policy=eager layer=(\d+) first_row=(\d+) '
        r'logical_experts=(\d+) accepted_copy_commands=(\d+) requested_H2D_bytes=(\d+) '
        r'publication=accepted_same_queue native_completion=unchecked_until_outer_drain$',text,re.M)
    records=[tuple(map(int,v)) for v in matches]
    require(text.count('strata prefill fullfit upload:')==len(records),'malformed/unknown upload route')
    if plan['upload']=='jit':require(not records,'JIT arm unexpectedly used eager upload')
    else:
        require(len(records)==24 and len({v[0] for v in records})==24 and all(0<=l<48 and first==0
            and 0<n<=512 and commands==n and size>0 and size%n==0 and
            size//n in [1510400,1971200,1715200,1868800] for l,first,n,commands,size in records),
            'actual24full-fit eager layers/individual copies differ')
    return dict(policy=plan['upload'],fullfit_layers=len(records),accepted_copy_commands=sum(v[3] for v in records),
        requested_fullfit_H2D_bytes=sum(v[4] for v in records),native_completion='outer checked bank drain',
        physical_bus_traffic_or_overlap_measured=False)


def request_route(text,plan,*,prefix_page_size=16):
    require(prefix_page_size in (4,16),'unsupported BF16 prefix page size')
    result=core.request_route(text,plan);n=plan['read']-1
    split=n>90112 and (plan['window']==16 or n<=117488)
    relocated=split and bool(plan['window']==16 or plan['relocate'])
    maximum=n if split else min(n,81920);layout=workspace(maximum,split,relocated)
    expected_windows=48*((n+maximum-1)//maximum)
    main=one(text,'strata prefill: layer-major candidate, ')
    bank=one(text,'strata MoE expert bank: ');admission=one(text,'strata MoE accumulator admission: ')
    active=one(text,'strata MoE split admission: ');window=one(text,'strata MoE window configuration: ')
    work=one(text,'strata MoE accumulator window: ');mixed=one(text,'strata MoE mixed banks: ')
    source=one(text,'strata prefill expert source: ');end=one(text,'strata MoE expert bank complete: ')
    cache=plan['bank']*STRIDE;limit=int(bank['allocation_limit_bytes']);arena=int(admission['shared_arena_bytes'])
    effective=16 if plan['window']==16 and split else 10
    require(main['tokens']==str(n) and main['chunk']=='8192' and main['residual_dtype']=='bf16'
        and main['residual_element_bytes']=='2' and int(main['residual_gpu_rows'])==n
        and int(main['residual_total'])==int(main['residual_gpu'])==n*ROW
        and main['residual_host']==main['residual_staging']=='0' and int(main['layer_cache'])==cache,
        'actual residual placement differs;no automatic recalibration')
    require(bank['requested_slots']==bank['effective_slots']==str(plan['bank']) and bank['staged']=='1'
        and bank['host_slab_bytes']=='1363148800' and int(bank['device_bytes'])==cache
        and int(bank['residual_gpu_rows'])==n and bank['general_map']=='private_scratch_never_published',
        'requested bank/private-map contract differs')
    require(active['active']==str(int(split)) and int(active['rows'])==n
        and int(active['primary_rows'])==(65536 if split else 0)
        and int(active['tail_payload_bytes'])==layout['tail_payload']
        and int(active['extra_allocation_bytes'])==layout['extra']
        and active['prefill_slots']==str(plan['bank']) and active['decode_changed']=='0'
        and int(active['max_single_allocation_bytes'])==limit,'actual accumulator split/extra owner differs')
    extra_rows=records(text,'strata MoE long workspace: ')
    require(len(extra_rows)==int(plan['window']==16 or plan['relocate']), 'long-workspace admission count differs')
    if extra_rows:
        extra=extra_rows[0]
        require(extra==dict(requested_window=str(plan['window']),relocated_requested=str(plan['relocate']),
            relocated_active=str(int(relocated)),effective_window=str(effective),extra_bytes=str(layout['extra']),
            admission='admitted' if split else 'ordinary_fallback',decode_changed='0'),'long-workspace admission differs')
    require(window['requested_mode']==str(plan['window']) and window['effective_mode']==str(effective)
        and int(window['maximum_rows'])==int(work['maximum_rows'])==maximum and window['mixed_fp16_bytes']=='0'
        and int(work['windows'])==expected_windows and work['operator_max']=='8192'
        and work['GPU_resident_windows_only']=='1' and int(work['source_H2D_bytes'])==48*n*40,
        'actual window/map work differs')
    require(mixed['storage']=='scratch' and mixed['extra_allocation_bytes']=='0'
        and int(mixed['primary_bytes'])==layout['mixed_primary'] and int(mixed['tail_bytes'])==layout['mixed_tail']
        and int(mixed['tail_offset'])==layout['region'] and layout['region']+layout['mixed_tail']<=arena
        and mixed['phase']=='FFN_only_after_attention_drain','actual mixed producer/consumer layout differs')
    padded_prefix_rows = ((n + prefix_page_size - 1) // prefix_page_size) * prefix_page_size
    prefix = padded_prefix_rows * 2056 + 256
    fixed = cache + prefix + 536870912 + n * ROW + layout['extra']
    require(admission['requested_mode']==str(plan['window']) and int(admission['rows'])==maximum
        and int(admission['gpu_prefix_rows'])==n and int(admission['mixed_fp16_bytes'])==maximum*5120
        and int(admission['fixed_including_headroom_bytes'])==fixed<=int(admission['free_gpu_bytes'])
        and max(cache,n*ROW,layout['extra'],prefix)<=limit,'observed total/max-allocation admission differs')
    require(source['requested_view']==source['effective_view']=='1' and source['layers']=='48'
        and source['owned_slab_bytes']=='0' and int(source['borrowed_layer_span_bytes'])==50292326400
        and source['host_copy']=='eliminated' and source['source_owner']=='caller_through_checked_retirement',
        'immutable all-layer source view differs')
    require(end['final_drain']=='checked' and 0<=int(end['permanent_slots'])<=int(end['total_slots'])<=512
        and int(end['total_slots'])>=plan['bank']
        and int(end['uploads'])+int(end['permanent_hits'])==int(work['dequants'])
        and min(int(end['requested_H2D_bytes']),int(end['permanent_hits']))>=0,
        'expert bank accepted-upload/retirement conservation differs')
    completes=records(text,'strata MoE split complete: ')
    require(len(completes)==int(split),'split completion count differs')
    if split:
        require(completes[0]==dict(rows=str(n),primary_rows='65536',tail_bytes=str(layout['tail_payload']),
            guard='checked',final_drain='checked',decode_changed='0'),'FP32tail guard/drain differs')
    ends=records(text,'strata MoE long workspace complete: ')
    require(len(ends)==int(relocated),'relocated completion count differs')
    if relocated:
        require(ends[0]==dict(rows=str(n),extra_bytes=str(layout['extra']),guards='3_checked',
            final_drain='checked',decode_changed='0'),'relocated job-plane guards/drain differ')
    require('softmax=parallel-exp-ascending-sum' in text and 'softmax=subgroup' not in text,
        'independent attention selection changed')
    fullfit=eager_route(text,plan)
    require(re.findall(r'^strata mtp: prompt expert phase=(\S+) ',text,re.M)==
        ['before-detach','after-detach','after-prompt-before-restore','after-restore']
        and re.findall(r'^strata mtp: experts restored bytes=(\d+),',text,re.M)==['707788800'],
        'actual decode MTP handoff differs')
    if plan['trace_progress']:
        packed=records(text,'strata packed expert bank: ');require(len(packed)==48,'48actual packed layer views required')
        for layer,g in enumerate(packed):
            stride=int(g['stride']);count=min(512,cache//stride)
            require(int(g['layer'])==layer and stride>0 and stride%256==0 and int(g['slots'])==count
                and int(g['used_bytes'])==count*stride<=cache and int(g['backing_bytes'])==cache
                and g['full_fit']==str(int(count==512)),'actual packed layer geometry differs')
    result.update(layout=dict(**layout,shared_arena_bytes=arena,requested_bank=plan['bank'],window_rows=maximum,
        windows=expected_windows,split=split,relocated=relocated,free_gpu_bytes=int(admission['free_gpu_bytes']),
        fixed_including_headroom_bytes=fixed,allocation_limit_bytes=limit,actual_admission=True),
        actual_GU_dispatch=dispatch_reader.dispatch(text,'large'),
        uploads=int(end['uploads']),permanent_hits=int(end['permanent_hits']),
        requested_bank_H2D_bytes=int(end['requested_H2D_bytes']),requested_source_map_H2D_bytes=int(work['source_H2D_bytes']))
    result['fullfit_upload']=fullfit
    return result
