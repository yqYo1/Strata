"""Existing independently produced full catalog identity contract. Repository LICENSE.
Pure receipt checks; no execution, enrollment, approval generation or path discovery.
The caller pins all input bytes and validates the enum controller package separately.
"""

def need(value, message):
    if not value:
        raise ValueError('HOLD_CATALOG_IDENTITY: ' + message)


def same_artifact(actual, expected):
    """Existing compact evidence may omit realpath; never omit path/hash/size."""
    return (type(actual) is dict and
            all(actual.get(k) == expected[k] for k in ('path', 'bytes', 'sha256')) and
            ('realpath' not in actual or actual['realpath'] == expected['realpath']))


def validate(cat, build, enumeration, review, evidence, contract, boot_id,
             compiled_source_sha256, enum_package_sha256):
    """Separate enum controller identity from its unchanged v1 compiled source.

    Exact immutable receipt bytes are checked outside this function, including
    Main's binary/source review. Owner.closed and PID/start absence stay with
    the original controller. No query package equality or rebuild is requested.
    """
    need(contract['schema'] == 'strata-catalog-input-contract-v3' and
         set(cat) == set(contract['artifacts']) == {'receipt', 'build_receipt', 'binary', 'stdout'},
         'exact existing catalog identities')
    for key, expected in contract['artifacts'].items():
        need(cat[key] == expected and same_artifact(evidence[key], expected),
             'existing actual catalog artifact: ' + key)
    enum_sha = contract['enum_controller_package']['sha256']
    source = contract['compiled_catalog_source']
    need(enum_package_sha256 == enum_sha and compiled_source_sha256 == source['sha256'],
         'independent enum-controller package and exact compiled-source identity')
    need(review['schema'] == 'strata-main-host-helper-binary-review-v1' and
         review['reviewer'] == 'Main' and review['approved'] is True and
         review['source_sha256'] == source['sha256'] and
         review['build_receipt'] == cat['build_receipt'] and review['binary'] == cat['binary'],
         'existing Main source/build/binary review')
    need(evidence['schema'] == 'strata-metric-full-catalog-evidence-v2' and
         evidence['reviewer'] == 'Main' and evidence['status'] == 'ENUMERATION_PASS' and
         evidence['collection_qualified'] is False, 'existing enumeration-only evidence')
    need(build['reviewed_package_sha256'] == enumeration['reviewed_package_sha256'] == enum_sha and
         build['boot_id'] == enumeration['boot_id'] == boot_id and
         build['mode'] == 'build' and enumeration['mode'] == 'enumerate' and
         build['binary_candidate'] == cat['binary'], 'actual enum package/boot/build bindings')
    for record in (build, enumeration):
        need(record['passed'] is True and record['complete'] is True and record['active'] is False and
             record['ownership_closed'] is True and record['no_gpu_commands_authored'] is True and
             record['collection_qualified'] is False and record['fault_window']['complete'] is True and
             record['fault_window']['boot_id'] == boot_id and not record['fault_window']['fresh_faults'],
             'normal completed fault-free enumeration/build; no collection claim')
    builds = [c for c in build['commands'] if c['label'] == 'host-build']
    need(len(builds) == 1 and builds[0]['argv'] == contract['build_argv'] and
         builds[0]['argv'].count(source['path']) == 1 and
         builds[0]['environment'] == contract['build_environment'], 'exact actual compiler source argv/environment')
    native = [c for c in enumeration['commands'] if c['label'] == 'native-enumeration']
    need(len(native) == 1 and native[0]['argv'] == contract['native_argv'] and
         native[0]['environment'] == contract['native_environment'] == enumeration['native_snapshot']['environment'],
         'actual immutable native enumeration argv/environment')
    out = native[0]['logs']['native-enumeration.stdout']
    need(all(out[k] == cat['stdout'][k] for k in ('sha256', 'bytes')), 'actual stdout size/hash')
    need(enumeration['enumeration_summary'] == evidence['summary'] and
         enumeration['enumeration_summary']['status'] == 'ENUMERATION_PASS', 'actual complete enumeration summary')
    return dict(bound=True, enum_controller_package=contract['enum_controller_package'],
                compiled_catalog_source=source, binary_review=contract['binary_review'],
                catalog_evidence=contract['catalog_evidence'], artifacts=cat,
                limitation='Existing enum build/controller/source identities only; query collection remains separately gated')
