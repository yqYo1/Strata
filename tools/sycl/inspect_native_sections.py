#!/usr/bin/env python3
"""Hash embedded ELF64 native text/metadata without loading a GPU runtime.

Main-only bounded artifact inspection. A matching text hash proves equal code
bytes, not execution, quality, timing, occupancy or complete ABI equivalence.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys


def inspect(path, extract_to=None):
    path = Path(path)
    if not 0 < path.stat().st_size <= 256 << 20:
        raise ValueError('expected an artifact of at most256MiB')
    data = path.read_bytes()
    images = []
    start = 0
    while True:
        offset = data.find(b'\x7fELF', start)
        if offset < 0:
            break
        start = offset + 4
        if data[offset + 4:offset + 6] != b'\x02\x01' or offset + 64 > len(data):
            continue
        header = struct.unpack_from('<HHIQQQIHHHHHH', data, offset + 16)
        shoff, entsize, count, strings = header[5], header[10], header[11], header[12]
        if entsize != 64 or not 0 < count <= 4096 or strings >= count:
            continue
        if shoff > len(data) - offset - count * 64:
            continue
        sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset + shoff + i * 64) for i in range(count)]
        if any(s[1] != 8 and s[4] + s[5] > len(data) - offset for s in sections):
            continue
        names = sections[strings]
        if names[5] > 1 << 20:
            continue
        table = data[offset + names[4]:offset + names[4] + names[5]]
        found = {}
        for section in sections:
            if section[0] >= len(table) or section[1] == 8:
                continue
            end = table.find(b'\0', section[0])
            if end < 0:
                continue
            name = table[section[0]:end].decode('utf-8', errors='strict')
            if name == '.ze_info' or name.startswith('.text.'):
                blob = data[offset + section[4]:offset + section[4] + section[5]]
                found[name] = dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        if '.ze_info' in found:
            image = dict(offset=offset, machine=header[1], sections=found)
            if extract_to is not None:
                end = max(shoff + count * 64, 64,
                          *(s[4] + s[5] for s in sections if s[1] != 8))
                blob = data[offset:offset + end]
                if len(images) >= 16 or len(blob) > 32 << 20:
                    raise ValueError('native extraction budget exceeded')
                destination = Path(extract_to) / ('image-' + str(len(images)) + '.bin')
                with destination.open('xb') as stream:
                    stream.write(blob)
                image['extracted'] = dict(path=str(destination), bytes=len(blob),
                    sha256=hashlib.sha256(blob).hexdigest())
            images.append(image)
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), images=images)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('use inspect_native_sections.py ARTIFACT [ARTIFACT ...]')
    print(json.dumps([inspect(p) for p in sys.argv[1:]], indent=2))
