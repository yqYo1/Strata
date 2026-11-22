"""Rebuild the measured KV/gather engine with the 128-slot upload selector."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_large_gather_model_source as gather
import prefill_split_upload_build as upload
from official_model_sources import merge_delta

bank, builder, require = upload.bank, upload.builder, upload.require
KEY = 'prefill_kv_upload_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PARENT_EVIDENCE = '83ae2717d51f5ab54f50787cbcae5acf1a945cff'
UPLOAD_EVIDENCE = '982211f5c91b4251ff89b8ad93f1f369a46fbad0'
UPLOAD_BUILD = 'e4dfad786347e5adf0f913d69df50454e3decbaa'
SCOPE = 'd92f63fa7e92e31324d27062f8ac5f1702a71bab'
OVERLAYS = ((upload.UPLOAD, (upload.UNIT, upload.WINDOW, upload.LAYER)),
            (upload.HELPER, (upload.WINDOW, upload.HEADER, upload.FIXTURE)),
            (SCOPE, (upload.WINDOW,)))


def apply_upload(source):
    original = bank.stage.hashed(source)
    for revision, paths in OVERLAYS:
        for path in paths:
            before = source[path]
            source[path] = merge_delta(before, path, revision + '^', revision)
            require(merge_delta(source[path], path, revision, revision + '^') == before,
                    'upload would replace unrelated source: ' + path)
    changed = {path: h for path, h in bank.stage.hashed(source).items() if original.get(path) != h}
    require(set(changed) == {upload.UNIT, upload.WINDOW, upload.LAYER, upload.HEADER, upload.FIXTURE},
            'unexpected native change in upload projection')
    return changed


def project(expected=None):
    native, foundation, source, recipe, original = gather.build_inputs(expected)
    parent = bank.previous.historic(PARENT_EVIDENCE, gather.KEY)
    require(parent['stage'] == 'built' and parent['passed'] and parent['complete']
            and parent['ownership_closed'] and not parent['active'] and parent['host_control']['passed']
            and parent['native_source_commit'] == gather.DISPATCH_COUNTS
            and parent['projection'] == original and parent['recipe'] == recipe
            and parent['external_inputs'] == native['external_inputs']
            and parent['source_files'] == dict(files=len(source), sha256=original['source_sha256']),
            'recover the exact measured KV/gather source and recipe')
    earlier = bank.previous.historic(UPLOAD_EVIDENCE, 'prefill_split_upload_native_v1')
    timing = bank.previous.historic(UPLOAD_EVIDENCE, 'prefill_split_upload_speed_v1')
    require(earlier['native_source_commit'] == UPLOAD_BUILD and earlier['passed']
            and earlier['recipe']['targets'] == recipe['targets']
            and earlier['binaries']['strata'] == timing['binary']
            and timing['complete'] and timing['ownership_closed'] and timing['passed']
            and timing['decision']['repeated_warm_gain']['batched/demand'],
            'recover the earlier same-queue scheduling evidence')
    changed = apply_upload(source)
    old, new = str(gather.MODEL_ROOT), str(ROOT)
    require(recipe['configure'][2:5] == [old + '/source', '-B', old + '/build'],
            'unexpected parent configure paths')
    recipe = dict(configure=[arg.replace(old, new) for arg in recipe['configure']],
                  environment={k: v.replace(old, new) for k, v in recipe['environment'].items()},
                  targets=recipe['targets'])
    proof = dict(parent_evidence=PARENT_EVIDENCE, parent_source=parent['native_source_commit'],
                 parent_binary=parent['binaries']['strata'], parent_source_files=parent['source_files'],
                 upload_evidence=UPLOAD_EVIDENCE, overlay_commits=[rev for rev, _ in OVERLAYS],
                 overlays=changed, object_count=earlier['projection']['object_count'],
                 object_set_sha256=earlier['projection']['object_set_sha256'])
    return native, foundation, source, recipe, proof


def context():
    ctx = bank.basis.core.source.device.load_module('kv_upload_build', upload.__file__)
    ctx.ROOT, ctx.KEY, ctx.project = ROOT, KEY, project
    return ctx


def prepare(ctx):
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT),
            'inspect the existing KV/upload generation first')
    require(not upload.ROOT.exists(), 'verify retirement of the previous upload build')
    native, foundation, source, recipe, proof = project()
    modules = (gather, gather.fixture, gather.kv, upload, upload.boundary, upload.boundary.stock)
    controls = [THIS, *(str(Path(m.__file__).resolve().relative_to(bank.P)) for m in modules)]
    pins = bank.stage.source.pin(bank.P, [*controls, 'tools/sycl', *bank.basis.POLICIES])
    row = dict(stage='prepared', source_commit=pins['revision'], native_source_commit=pins['revision'],
               controller_sources=pins['source_sha256'], source_files=bank.stage.hashed(source),
               foundation=foundation, projection=proof, recipe=recipe, external_inputs=native['external_inputs'],
               build_tools={name: bank.sha(name) for name in ('/usr/bin/cmake', '/usr/bin/ninja')},
               driver=builder.guards.driver_guard(), retained_decode_reference=upload.decode_reference(recover=True),
               active=False, passed=False, complete=False, ownership_closed=True, commands=[],
               scope='Compare demand and batched issue in the original 128-slot split workspace. '
                     'Keep KV reuse, scalar gather and attention copy fixed.',
               retention='Keep the measured KV/gather and independent decode references. '
                         'Review the third generation on comparison closure or failed admission.')
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, content in source.items():
        path = ROOT / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    ctx.check(row, committed=False)
    ctx.save(row)


def main(action):
    require(action in ('projection', 'prepare', 'configure', 'compile', 'host'), 'select one build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        ctx = context()
        builder.ROOT, builder.KEY, builder.check, builder.save = ROOT, KEY, ctx.check, ctx.save
        if action == 'projection':
            _, _, source, _, proof = project()
            print({'source_files': len(source), 'projection': proof})
        elif action == 'prepare':
            prepare(ctx)
        elif action == 'host':
            bank.stage.host(root_path=ROOT, check_build=ctx.check, save_record=ctx.save)
        else:
            row = bank.previous.read(ROOT / 'record.json')
            builder.TARGETS = row['recipe']['targets']
            builder.execute(action)


if __name__ == '__main__':
    action, = sys.argv[1:]
    main(action)
