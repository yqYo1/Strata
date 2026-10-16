#!/usr/bin/env python3
"""Main-only bounded inspection of the exact integrated6354 native16 executable."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
SPEC = A / 'own-vllm-xe2-qualification-v1/manifest.json'
MANIFEST = '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def need(v, reason):
    if not v:
        raise RuntimeError(reason)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['extract', 'disassemble'])
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--prior-sha256', required=True)
    ap.add_argument('--image', type=Path)
    ap.add_argument('--image-sha256')
    ap.add_argument('--ocloc', type=Path)
    args = ap.parse_args()
    need(sha(SPEC) == MANIFEST, 'reviewed R816 manifest changed')
    meta = json.loads(SPEC.read_text())
    ownerpath = Path(meta['direct_owner']['path'])
    need(sha(ownerpath) == meta['direct_owner']['sha256'], 'Owner changed')
    spec = importlib.util.spec_from_file_location('isa_owner', ownerpath)
    owner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner_module)
    controller = A / 'own-vllm-xe2-qualification-v1/main_controller.py'
    need(sha(controller) == meta['package_files']['main_controller.py']['sha256'], 'R816 helper changed')
    spec = importlib.util.spec_from_file_location('isa_helpers', controller)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    root = args.root.absolute()
    need(root.parent == B and re.fullmatch(r'main-vllm-integrated-isa-[A-Za-z0-9_-]{1,64}', root.name), 'private root scope')
    need(not root.exists() and not root.is_symlink(), 'fresh root required')
    selfhash = sha(__file__)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        record = dict(schema='strata-integrated-native16-isa-probe-v1', mode=args.mode, passed=False, complete=False, active=False,
                      boot_id=helpers.boot(), commands=[], fresh_faults=[],
                      driver_dump_exists=False, pins={}, fault_windows=[],
                      GPU_executed=False, offline_tool_may_query_device=args.mode == 'disassemble',
                      source_sha256=selfhash, invocation=sys.argv,
                      ISA_inference='UNASSESSED; Main must inspect actual instructions and match runtime symbols')
        receipt = root / 'record.json'

        def save():
            receipt.write_text(json.dumps(record, indent=2) + '\n')

        owned = owner_module.Owner(root / 'owned', save)
        record['commands'] = owned.commands
        save()

        def remember(p):
            p = Path(p).resolve()
            value = helpers.pin(p)
            record['pins'][str(p)] = value
            return value

        def verify():
            need(sha(__file__) == selfhash and sha(SPEC) == MANIFEST, 'inspection source drift')
            need(helpers.boot() == record['boot_id'], 'boot drift')
            for p, value in record['pins'].items():
                helpers.pinned(p, value)

        env = dict(meta['execution_environment'])
        env.update(XDG_CACHE_HOME=str(root / 'cache'), TMPDIR=str(root / 'tmp'))
        (root / 'cache').mkdir(); (root / 'tmp').mkdir()

        def run(label, argv):
            need(time.monotonic() < deadline, 'whole inspection deadline')
            remember(argv[0]); verify()
            command, out, err = owned.run(label, list(map(str, argv)), env, root,
                wall=min(90, deadline-time.monotonic()), cpu=90, rss_cap=1 << 30, text_cap=8 << 20,
                total_cap=256 << 20, file_cap=64 << 20)
            verify()
            need(owner_module.closed(command) and command.get('exit_code') == 0,
                 'inspection command failed: ' + label)
            return out, err

        def journal(label, cursor=None):
            argv = ['/usr/bin/journalctl', '-k', '-b', '--no-pager', '-o', 'json', '--show-cursor']
            argv += ['--after-cursor', cursor] if cursor else ['-n', '0']
            out, err = run(label, argv)
            body = out.read_text()
            match = re.search(r'^-- cursor: (\S+)\s*$', body, re.M)
            rows = [json.loads(line) for line in body.splitlines()
                    if line and not line.startswith('-- cursor: ') and line != '-- No entries --']
            need(match or (cursor and not rows), 'fresh journal cursor absent')
            faults = [r for r in rows if helpers.FAULT.search(str(r.get('MESSAGE', '')))]
            record['fresh_faults'].extend(faults)
            record['fault_windows'].append(dict(label=label, after=cursor,
                end=match[1] if match else cursor, entries=len(rows), faults=faults))
            helpers.driver_guard()
            need(not faults, 'fresh GPU fault')
            return match[1] if match else cursor

        cursor = None
        deadline = time.monotonic() + 600
        try:
            prior, record['prior'] = helpers.receipt_gate(owner_module, args.prior,
                args.prior_sha256, record['boot_id'], qualification=True)
            buildpath = A / 'own-vllm-engine-build-v2.json'
            need(sha(buildpath) == '990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639', 'current immutable build')
            build = json.loads(buildpath.read_text())
            need(build['passed'] and build['complete'] and not build['active'], 'current build qualified')
            need(prior.get('schema') == record['schema'] or prior.get('binary_sha256') == build['binary_sha256'], 'same integrated predecessor')
            record['products'] = build['products']
            remember(buildpath)
            if args.mode == 'extract':
                need(prior.get('binary_sha256') == build['binary_sha256'], 'current native hardware prior')
            remember(args.prior); remember(controller); remember(SPEC); remember(ownerpath)
            for group in ['external_pins', 'loaded_library_pins']:
                for p, value in meta[group].items():
                    helpers.pinned(p, value)
                    record['pins'][p] = value
            record['driver_before'] = helpers.driver_guard()
            cursor = journal('journal-before')
            if args.mode == 'extract':
                product = build['products']['sycl_vllm_native16_test']
                helpers.pinned(product['path'], product)
                remember(product['path'])
                record['input'] = product
                run('host-elf', ['/usr/bin/readelf', '-h', '-S', '-W', product['path']])
                tool = Path(meta['compiler']).parent / 'clang-offload-extract'
                run('extract-images', [tool, product['path'], '--output=' + str(root / 'image')])
                images = sorted(root.glob('image.*'))
                need(0 < len(images) <= 32, 'bounded nonempty extracted images')
                record['images'] = []
                for p in images:
                    need(p.is_file() and not p.is_symlink() and p.stat().st_size <= 64 << 20,
                         'bounded regular image')
                    with p.open('rb') as f:
                        prefix = f.read(64)
                    record['images'].append(dict(path=str(p), **remember(p), prefix_hex=prefix.hex()))
                    if prefix.startswith(b'\x7fELF'):
                        run('image-elf-' + str(len(record['images'])), ['/usr/bin/readelf', '-h', '-S', '-W', p])
            else:
                need(args.image and args.image_sha256 and args.ocloc, 'explicit image/hash/ocloc required')
                image = args.image.resolve()
                need(image.is_relative_to(B) and image.parent.name.startswith('main-vllm-integrated-isa-') and
                     sha(image) == args.image_sha256 and image.stat().st_size <= 64 << 20, 'exact prior extracted image')
                need(prior.get('schema') == record['schema'] and prior.get('mode') == 'extract' and
                     any(i['path'] == str(image) and i['sha256'] == args.image_sha256 for i in prior.get('images', [])),
                     'disassembly must consume the immediately preceding qualified extraction')
                need(args.ocloc.is_absolute() and args.ocloc.is_file(), 'existing explicit ocloc')
                remember(image)
                # Decoder -device takes a hardware family prefix, not ocloc's
                # compile-time product-config acronym bmg-g21-a0. The ELF note
                # retains the exact producer product configuration separately.
                run('disassemble', [args.ocloc, 'disasm', '-file', image, '-device', 'bmg', '-dump', root / 'isa'])
                files = sorted(p for p in (root / 'isa').rglob('*') if p.is_file())
                need(0 < len(files) <= 1024, 'bounded disassembly artifact census')
                record['disassembly_files'] = [dict(path=str(p), **remember(p)) for p in files]
            cursor = journal('journal-after', cursor)
            verify()
            record['passed'] = True
        except BaseException as exc:
            record['error'] = repr(exc)
        finally:
            if cursor and owned.active is None and record['fault_windows'][-1]['label'] != 'journal-after':
                try:
                    journal('journal-after-failure', cursor)
                except BaseException as exc:
                    record['final_fault_check_error'] = repr(exc)
            record['active'] = owned.active is not None
            record['ownership_closed'] = bool(owned.commands) and not record['active'] and all(owner_module.closed(c) for c in owned.commands)
            record['driver_dump_exists'] = (helpers.DEVICE / 'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*')))
            record['private_bytes'] = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            record['complete'] = record['ownership_closed'] and helpers.boot() == record['boot_id'] and not record.get('final_fault_check_error')
            record['passed'] &= record['complete'] and record['private_bytes'] <= 256 << 20 and not record['fresh_faults'] and not record['driver_dump_exists']
            record['finished_utc'] = owner_module.utc()
            record['retention'] = dict(owner='Main', consumer='actual integrated6354 native16 ISA/symbol/type inspection',
                byte_budget=256 << 20, review='retire duplicate raw disassembly/images after compact instruction/provenance proof is verified')
            save()
        print(json.dumps(dict(passed=record['passed'], complete=record['complete'], receipt=str(receipt), sha256=sha(receipt))))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
