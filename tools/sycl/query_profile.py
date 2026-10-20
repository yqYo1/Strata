#!/usr/bin/env python3
"""Shared same-build diagnostic admission and production-query evidence.

Closed tiny/long query launchers are archived at55265f9e3. The current bounded
cache model check lives in cache_borrow.py; early scheduling stays unqualified."""
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import cpu_capacity as assets
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
from build_engine import blobs
from check_verifier_retirement import DELTA

P, B, A = gen.P, gen.B, gen.A
BUILD = A / 'native-gdn-engine-build.json'
COMPONENT = A / 'decode-query-profile-component-current.json'
BUILD_RECORD = B / 'main-native-gdn-engine/record.json'
COPY_KEYS = ('submitted_blobs','submitted_bytes','completed_blobs','completed_bytes',
             'host_view_groups','gpu_miss_groups','cpu_unique_misses','cpu_entries')


def runtime_assets_unchanged(record):
    # Admit only these new diagnostic modes and the exact assets already used
    # by the same-build normal51+observer fixture; never relabel a mode to pass
    # generation_engine's independently scoped admission check.
    guards.require(record['mode'] in ('default-route-query-profile-check',
        'default-route-query-profile-long','cache-borrow-check'), 'unknown query diagnostic mode')
    expected = {n:dict(qualified_sha256=v[0],candidate_sha256=v[1]) for n,v in DELTA.items()}
    component = gen.committed(COMPONENT)
    def identities(delta):
        # Older model receipts also carry package/version timestamps. Compare
        # the complete asset set and both immutable hashes, not those labels.
        return {n:{k:item[k] for k in ('qualified_sha256','candidate_sha256')}
            for n,item in delta.items()}
    guards.require(component['passed'] and component['complete'] and component['ownership_closed'] and
        not component['active'] and not component['fresh_faults'] and
        identities(record['runtime_asset_delta']) == identities(component['runtime_asset_delta']) == expected,
        'query diagnostic assets differ from the same-build component admission')
    settings = gen.committed(gen.SETTINGS)
    guards.require(set(expected) <= set(settings['runtime_elfs']), 'unknown runtime asset')
    for name,item in settings['runtime_elfs'].items():
        proposed = expected.get(name)
        if proposed:
            guards.require(proposed['qualified_sha256'] == item['sha256'], 'original runtime baseline changed')
        sha = proposed['candidate_sha256'] if proposed else item['sha256']
        guards.require(gen.common.digest(name) == sha, 'runtime asset changed: '+name)


def current_build():
    compact, component = gen.committed(BUILD), gen.committed(COMPONENT)
    full = json.loads(BUILD_RECORD.read_text())
    guards.require(full['passed'] and full['complete'] and full['ownership_closed'] and
        not full['active'] and all(full[k] == compact[k] for k in
        ('source_commit','binary','fixtures','compiler','configure_argv')), 'closed committed build required')
    guards.require(component['passed'] and component['complete'] and component['ownership_closed'] and
        not component['active'] and not component['fresh_faults'] and component['binary']==full['binary'] and
        component['fixtures']==full['fixtures'] and component['source_commit']==full['source_commit'] and
        len(component['checks'])==11 and 'cases=17 native_variable_stride=PASS' in component['checks'][0]['cases'][0]
        and len(component['checks'][1]['cases'])==10 and len(component['checks'][2]['cases'])==8
        and len(component['checks'][3]['cases'])==16 and len(component['checks'][4]['cases'])==1
        and len(component['checks'][5]['cases'])==9 and component.get('residency_publication_contract_passed') is True
        and len(component['checks'][6]['cases'])==10 and component.get('cache_tail_contract_passed') is True
        and len(component['checks'][7]['cases'])==22 and component.get('scratch_layout_contract_passed') is True
        and len(component['checks'][8]['cases'])==1 and len(component['checks'][9]['cases'])==15
        and len(component['checks'][10]['cases'])==1 and component['checks'][10].get('exit_code')==1
        and component.get('prefill_device_layout_contract_passed') is True
        and component.get('prefill_reset_retirement_contract_passed') is True
        and component.get('grouped_copy_projection_contract_passed') is True
        and component.get('query_profile_contract_passed') is True
        and component.get('mode')=='normal-route-query-profile-components',
        'matching passed normal51+observer1+residency9+tail10+scratch22+device17 required before model preparation')
    count = 0
    for source in [full['project'], *full['dependencies']]:
        recovered = {n:hashlib.sha256(data).hexdigest() for n,data in
            blobs(Path(source['root']),source['revision'],source['source_sha256'])}
        guards.require(recovered==source['source_sha256'], 'original build inputs not recoverable')
        for name, expected in recovered.items():
            guards.require(gen.common.digest(Path(source['root'])/name)==expected, 'native build input changed: '+name)
        count += len(recovered)
    for item in [full['binary'],*full['fixtures'].values()]:
        guards.require(gen.common.digest(item['path'])==item['sha256'], 'compiled artifact changed')
    return full, count


QUERY_KINDS = ('resident_launch','resident_pool_return','pcie_launch','pcie_pool_return')
QUERY_KEYS = ('calls','ready','pending','errors','clock_errors','wall_ns','max_wall_ns')


def query_evidence(log, requests):
    pattern = (r'^strata serve: SYCL_EVENT_QUERY scope=decode wrapper_wall=1 kind=([a-z_]+) '
        + r' '.join(k+r'=(\d+)' for k in QUERY_KEYS)+r'$')
    rows = re.findall(pattern,log,re.M)
    guards.require(len(rows)==requests*4 and log.count('SYCL_EVENT_QUERY')==len(rows),
        'missing/ambiguous query records')
    groups = []
    for offset in range(0,len(rows),4):
        selected=rows[offset:offset+4]
        guards.require(tuple(row[0] for row in selected)==QUERY_KINDS,'query category order differs')
        group={row[0]:dict(zip(QUERY_KEYS,map(int,row[1:]))) for row in selected}
        for kind,q in group.items():
            guards.require(q['calls']==q['ready']+q['pending']+q['errors'] and
                q['errors']==q['clock_errors']==0 and q['max_wall_ns']<=q['wall_ns'],
                'query status/time conservation failed')
            if kind.startswith('pcie_'):
                guards.require(not any(q.values()),'default-off early query route executed')
            else:
                guards.require(q['calls']>0 and q['wall_ns']>0,'resident query observer not reached')
        groups.append(group)
    return groups
