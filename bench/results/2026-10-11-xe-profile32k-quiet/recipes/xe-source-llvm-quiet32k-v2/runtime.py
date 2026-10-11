"""Exact quiet-run runtime identities; optional diagnostic layers are checked if mapped."""
from pathlib import Path

CORE = {'libsycl.so.9.0.0-0', 'libur_loader.so.0.12.0',
        'libur_adapter_level_zero_v2.so.0.12.0', 'libumf.so.1.1.0',
        'libze_loader.so.1.32.0', 'libze_intel_gpu.so.1.17.39395'}
OPTIONAL = {'libze_tracing_layer.so.1.32.0', 'libze_validation_layer.so.1.32.0'}


def validate_runtime(observed, qualified):
    expected = {x['path']: x['sha256'] for x in qualified}
    names = {Path(p).name: p for p in expected}
    if len(expected) != 8 or len(names) != 8 or set(names) != CORE | OPTIONAL:
        raise ValueError('exact qualified diagnostic runtime set')
    for name in CORE:
        path = names[name]
        if observed.get(path, {}).get('sha256') != expected[path]:
            raise ValueError('missing or changed core runtime: ' + name)
    for path, identity in observed.items():
        if path not in expected or identity.get('path') != path or identity.get('sha256') != expected[path]:
            raise ValueError('unqualified actual mapped runtime: ' + path)
    return {'required_core': sorted(names[n] for n in CORE),
            'optional_mapped': sorted(p for p in observed if Path(p).name in OPTIONAL),
            'optional_not_mapped': sorted(names[n] for n in OPTIONAL if names[n] not in observed),
            'all_observed_exactly_qualified': True,
            'scope': 'Six exact production libraries required; diagnostic layers optional only in quiet model, any mapped layer still requires exact path/hash. No completion or GPU health inferred.'}


def selftest():
    base = '/fixture/'
    qualified = [{'path': base+n, 'sha256': n} for n in sorted(CORE | OPTIONAL)]
    full = {x['path']: dict(x) for x in qualified}
    core = {p:x for p,x in full.items() if Path(p).name in CORE}
    seven = {p:x for p,x in full.items() if Path(p).name != 'libze_validation_layer.so.1.32.0'}
    for fixture in [full, core, seven]:
        validate_runtime(fixture, qualified)
    negatives=[]
    for name in sorted(CORE):
        wrong=dict(full); del wrong[base+name]; negatives.append(wrong)
    for name in ['libsycl.so.9.0.0-0', *sorted(OPTIONAL)]:
        wrong=dict(full); wrong[base+name]=dict(path=base+name, sha256='wrong');negatives.append(wrong)
    wrong=dict(full); wrong['/unqualified/libsycl.so.9.0.0-0']=dict(path='/unqualified/libsycl.so.9.0.0-0',sha256='libsycl.so.9.0.0-0');negatives.append(wrong)
    for fixture in negatives:
        try: validate_runtime(fixture, qualified)
        except ValueError: continue
        raise AssertionError('accepted invalid actual mapping')
    return dict(positive=3,negative=len(negatives))


if __name__ == '__main__':
    print(selftest())
