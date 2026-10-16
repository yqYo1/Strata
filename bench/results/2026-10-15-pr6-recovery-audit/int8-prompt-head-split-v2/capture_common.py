#!/usr/bin/env python3
"""Shared capture I/O for the unexecuted v2 controller/comparator proposals."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

FULL_CASES = [f'{kind}_tail_{w}' for w in (0,1,31,32,33,63,64,65,2051)
              for kind in ('sparse','dense')] + [
    'all_missing','zero_scales','cap262144_logical262143','large_query_tile0',
    'large_query_tile1','subnormal_query','fp16_subnormal_scales','zero_query',
    'negative_width','width_gt_cap']
HEX = re.compile(r'[0-9a-f]{64}\Z')
NAME = re.compile(r'[a-z0-9_.-]+\Z')
MAX_JSON = 1 << 20

def require(ok, message):
    if not ok:
        raise ValueError(message)

def parts(path):
    value = os.fspath(path)
    require(value.startswith('/') and not value.endswith('/'), 'absolute path without trailing slash required')
    result = value[1:].split('/')
    require(result and all(p and p not in ('.','..') for p in result), 'unsafe path component')
    return result

def open_dir(path, private=False):
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for item in parts(path):
            child = os.open(item, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        s = os.fstat(fd)
        if private:
            require(s.st_uid == os.geteuid() and not (s.st_mode & 0o077), 'capture directory must be owned/private')
        return fd
    except BaseException:
        os.close(fd)
        raise

def new_dir(path):
    p = Path(path)
    items = parts(p)
    # The root itself cannot be a user-created output path.
    require(len(items) >= 2, 'output needs an owned non-root parent')
    parent = open_dir(p.parent)
    try:
        s = os.fstat(parent)
        require(s.st_uid == os.geteuid() and not (s.st_mode & 0o022), 'output parent must be owned and not group/other writable')
        os.mkdir(items[-1], 0o700, dir_fd=parent)  # EEXIST is fatal, including symlinks.
        os.fsync(parent)
    finally:
        os.close(parent)
    return open_dir(p, private=True)

def read_bytes(root_fd, name, limit):
    require(NAME.fullmatch(name), 'unsafe capture basename')
    fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=root_fd)
    try:
        s = os.fstat(fd)
        require(stat.S_ISREG(s.st_mode) and s.st_uid == os.geteuid(), 'capture must be an owned regular file')
        require(s.st_size <= limit, 'capture size exceeds bound')
        data = bytearray()
        while True:
            piece = os.read(fd, min(65536, limit + 1 - len(data)))
            if not piece:
                break
            data.extend(piece)
            require(len(data) <= limit, 'capture grew beyond bound')
        require(len(data) == s.st_size, 'capture changed during read')
        return bytes(data)
    finally:
        os.close(fd)

def _pairs(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, 'duplicate JSON field')
        result[k] = v
    return result

def read_json(root_fd, name):
    return json.loads(read_bytes(root_fd, name, MAX_JSON), object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON value')))

def write_json(path, value):
    p = Path(path)
    parts(p)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    require(len(data) <= MAX_JSON, 'receipt exceeds 1 MiB')
    parent = open_dir(p.parent)
    fd = -1
    try:
        s = os.fstat(parent)
        require(s.st_uid == os.geteuid() and not (s.st_mode & 0o022), 'receipt parent must be owned/nonwritable by others')
        fd = os.open(p.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
        offset = 0
        while offset < len(data):
            n = os.write(fd, data[offset:])
            require(n > 0, 'short receipt write')
            offset += n
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.fsync(parent)
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent)

def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        while data := source.read(1 << 20):
            h.update(data)
    return h.hexdigest()

def sha(data):
    return hashlib.sha256(data).hexdigest()
