#!/usr/bin/env python3
"""Shared packed-bank source qualification and actual private-view evidence.

Closed117K model/clean launchers are recoverable at6213f0325d807b5eaf319bbf8096c9337f5402bd.
Current consumers are the long-generation comparison and offline refusal tests.
"""
import hashlib
import json
import re
import sys

sys.dont_write_bytecode = True
import cache_long as admission

q, gen, guards = admission.q, admission.gen, admission.guards
P, B, A = admission.P, admission.B, admission.A
BANK = A/'moe-expert-bank-current.json'
ORIGINAL_BUILD = 'b62a63744'


def qualification():
    full, count = q.current_build()
    bank = gen.committed(BANK)['packed_gpu']
    old_raw = gen.git('show', ORIGINAL_BUILD+':'+str(q.BUILD.relative_to(P)))
    old = json.loads(old_raw)
    guards.require(bank['passed'] and bank['complete'] and bank['ownership_closed'] and
        not bank['fresh_faults'] and bank['boot_unchanged'] and bank['mode']=='packed-validate' and
        len(bank['checks'])==2 and bank['build_source_commit']==old['source_commit'] and
        bank['fixture']==old['fixtures']['sycl_expert_bank_test'] and
        bank['provider_fixture']==old['fixtures']['file_expert_source_test'], 'passed original packed component required')
    guards.require(all(full[k]==old[k] for k in ('binary','fixtures','compiler','configure_argv','recipe_reference')) and
        [{k:s[k] for k in ('root','revision','count','aggregate_sha256')} for s in full['dependencies']]==old['inputs'][1:] and
        not gen.git('diff','--name-only',old['source_commit'],full['source_commit'],'--',
                    'src','include','cmake','CMakeLists.txt','third_party','bench/sycl'),
        'packed component native/recipe/fixture identity changed')
    proof = dict(original_inputs_recovered=count, source_build_sha256=gen.common.digest(q.BUILD_RECORD),
        shared_component_sha256=gen.common.digest(q.COMPONENT), packed_component_sha256=gen.common.digest(BANK),
        original_packed_build=dict(commit=ORIGINAL_BUILD,path=str(q.BUILD.relative_to(P)),sha256=hashlib.sha256(old_raw).hexdigest()),
        native_origin_source_commit=old['source_commit'], controller_only_native_reuse=True)
    updates = q.host_gate_test_updates(full)
    if updates:
        proof['controller_only_host_test_updates'] = updates
    return full, proof


def geometry(log, arm):
    rows = admission.records(log,'strata packed expert bank',48 if arm else 0)
    census = {}
    for expected, (_, row) in enumerate(rows):
        guards.require(set(row)=={'layer','slots','stride','used_bytes','backing_bytes','full_fit'}, 'unexpected geometry fields')
        layer,slots,stride,used,backing,full = [admission.integer(row,k) for k in
            ('layer','slots','stride','used_bytes','backing_bytes','full_fit')]
        guards.require(layer==expected and stride>0 and stride%256==0 and backing==1022361600 and
            slots==min(512,backing//stride) and 0<slots<=512 and used==slots*stride<=backing and
            full==int(slots==512), 'actual private geometry differs from checked extent')
        census[str(stride)]=census.get(str(stride),0)+1
    complete, = admission.records(log,'strata MoE expert bank complete',1)
    row=complete[1]
    guards.require(not rows or rows[-1][0]<complete[0], 'layout appeared after final drain')
    guards.require(set(row)=={'uploads','requested_H2D_bytes','permanent_hits','permanent_slots','total_slots','final_drain'} and
        row['final_drain']=='checked', 'missing completed expert uploads')
    counters={k:admission.integer(row,k) for k in row if k!='final_drain'}
    guards.require(counters['uploads']>0 and counters['requested_H2D_bytes']>0 and
        counters['permanent_slots']<=counters['total_slots']<=512, 'invalid expert upload totals')
    if arm:
        guards.require(counters['total_slots']==int(rows[-1][1]['slots']), 'final private layout differs')
    else:
        guards.require(counters['total_slots']==384 and counters['permanent_slots']<=383, 'baseline bank changed')
    return dict(actual_private_layers=len(rows),stride_layers=census,requested_work=counters,
                physical_PCIe_traffic_measured=False)
