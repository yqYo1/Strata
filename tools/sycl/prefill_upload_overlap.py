#!/usr/bin/env python3
"""Source-capability/ownership evidence for current actual-weight plans.

Closed copied-bank runner retired to source0a3295b89619cdc9cb4c6c4b2333043325fc8695.
Main-authored; repository LICENSE. No native execution entrypoint.
"""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import coherent_original_reuse as original
import engine_once
import measurement_guards as guards
import measurement_owner as owner_api
import upload_event_lane as event
import ur_validation

P, B = original.P, original.B
RESULT = event.RESULT
SAFETY = event.SOURCE
HEADER = event.HEADER
MAPPED = 'include/strata/prefill/mapped_host_source.hpp'
PAGES = 'tests/sycl/prefill_host_pages.hpp'
PREFIXES = ['tools/sycl', SAFETY, HEADER, MAPPED, PAGES, 'AGENTS.md',
            'docs/ARTIFACT_RETENTION.md', 'docs/BENCHMARKS.md', 'tests/core/test_upload_event_lane_plan.py']
BUILD = dict(wall=900, cpu=1800, rss_cap=8<<30, text_cap=32<<20,
             file_cap=256<<20, total_cap=512<<20)
RUN = dict(wall=120, cpu=240, rss_cap=4<<30, text_cap=32<<20,
           file_cap=32<<20, total_cap=128<<20)
require, sha, fingerprint = guards.require, owner_api.sha, original.fingerprint
git, write = event.git, event.write


def parent_evidence():
    evidence = '3ee710401f9cb8323757161dd53a17dce6689094'
    overlap = json.loads(git(P, 'show', evidence+':'+str(RESULT.relative_to(P))))['prefill_upload_overlap_v1']
    safety=['runtime-'+name for name in ['metadata','forward','reverse','permanent','cancel','cancel-throw','refuse']]
    require(overlap['stage']=='closed' and overlap['ownership_closed'] and not overlap['active']
            and len(overlap['results'])==23
            and [v['label'] for v in overlap['results'][:7]]==safety
            and all(v['passed'] is True and v['native_exit_code']==0 and v['ownership_closed'] is True
                    for v in overlap['results'])
            and overlap['analysis']['1024']['repeated_component_wall_gain'] is False
            and overlap['analysis']['4096']['repeated_component_wall_gain'] is True,
            'preserve original Runtime/E2qualification and both shape outcomes')
    return dict(evidence_commit=evidence, native_source_commit=overlap['native_source_commit'],
                original_result_sha256=fingerprint(overlap), execution_input=False,
                new_source_or_performance_qualification=False)


def capability_gate(value, archived):
    require(all(value.get(k) is True and archived.get(k) is True
                for k in ['passed','complete','ownership_closed'])
        and value.get('stage')==archived.get('stage')=='closed'
        and value.get('active') is False and archived.get('active') is False
        and fingerprint(value['capabilities'])==fingerprint(archived['capabilities']),
        'exact closed original capability evidence required')
    cap=value['capabilities']
    require(type(cap['api_version']) is int and cap['api_version']>=65552
        and type(cap['external_sysmem_mapping_version']) is int and cap['external_sysmem_mapping_version']>=65536
        and cap['read_only_mapping_reported_eligible'] is True
        and cap['mapping_or_transfer_attempted'] is False,
        'standard external mapping plus read-only API prerequisite required')
    return cap


def source_capability():
    key='prefill_host_source_capabilities_v2'
    value=ur_validation.committed(RESULT,(key,)); archive=value['archive']
    archived=json.loads(git(P,'show',archive['commit']+':'+archive['path']))[key]
    cap=capability_gate(value,archived)
    asset=value['queried_runtime_provider']
    require(sha(asset['path'])==asset['sha256'] and sha(value['header']['path'])==value['header']['sha256'],
            'queried driver/header changed before source-adapter qualification')
    return dict(evidence=archive,capabilities=cap,driver=asset,header=value['header'],
                mapping_or_transfer_qualified=False)


def property_gate(value, archived):
    require(all(value.get(k) is True and archived.get(k) is True
                for k in ['passed','complete','ownership_closed'])
        and value.get('stage')==archived.get('stage')=='closed'
        and value.get('active') is False and archived.get('active') is False
        and fingerprint(value['properties'])==fingerprint(archived['properties']),
        'exact closed original property observation required')
    prop=value['properties']
    require(prop['kind']=='mapping-properties' and type(prop['bytes']) is int and prop['bytes']==4096
        and prop['read_only'] is True and prop['same_address'] is True
        and prop['allocation_and_unimport_executed'] is True and prop['original_heap_unchanged'] is True
        and prop['explicit_native_context'] is True and prop['production_runtime_context_qualified'] is False
        and prop['source_admission_or_transfer_qualified'] is False
        and type(prop['application_commands']) is int and prop['application_commands']==0,
        'zero-command read-only heap lifecycle prerequisite required')
    for when in ['before','during','after']:
        require(all(type(prop[when][k]) is int for k in ['type','allocation_id','page_bytes']),
                'typed native allocation properties required')
    require(prop['before']==prop['after']==dict(type=0,allocation_id=0,page_bytes=0)
        and prop['during']['type']==1 and prop['during']['allocation_id']>0
        and prop['during']['page_bytes']==4096
        and type(prop['address_range']['result']) is int and prop['address_range']['result']==0
        and prop['address_range']['same_base'] is True
        and type(prop['address_range']['bytes']) is int and prop['address_range']['bytes']==4096,
        'pinned HOST property,positive ID,range and unimport removal required')
    return prop


def property_evidence():
    key='prefill_host_mapping_properties_v1'
    value=ur_validation.committed(RESULT,(key,));archive=value['archive']
    require(archive['commit']=='37017156e3d27a1f9a594a834bec320eeab350c5',
            'pin original property observation,not later metadata')
    archived=json.loads(git(P,'show',archive['commit']+':'+archive['path']))[key]
    prop=property_gate(value,archived);capability=source_capability()
    require(archived['project']['revision']==value['native_source_commit']=='b777f52f8ddfc60b71166cce9a68afb952c71dde'
        and fingerprint(value['queried_provider'])==fingerprint(capability['driver'])
        and archived['environment']['ZE_ENABLE_ALT_DRIVERS']==capability['driver']['path']
        and capability['capabilities']['api_version']==65553,
        'exact source,installed provider and API1.17 prerequisite required')
    return dict(evidence=archive,properties=prop,driver=capability['driver'],
                native_source_commit=value['native_source_commit'],new_source_admission_qualified=False)
