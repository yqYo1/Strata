"""Read-only full model input identity, run as a supervised CPU child."""
from pathlib import Path
import hashlib
import json
import sys


def identity(path):
    s = path.stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


result = []
for argument in sys.argv[1:]:
    path = Path(argument).resolve(strict=True)
    before = identity(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert identity(path) == before, 'Model input changed while hashing'
    result.append(dict(path=str(path), sha256=digest, stat=before))
    print('HASHED ' + str(path), flush=True)
print('IDENTITIES ' + json.dumps(result), flush=True)
