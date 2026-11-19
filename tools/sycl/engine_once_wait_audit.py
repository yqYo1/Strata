#!/usr/bin/env python3
"""The existing owned native launch with a private loader audit."""

import hashlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import engine_once


original_environment = engine_once.native_environment


def native_environment(plan, parent):
    env = original_environment(plan, parent)
    if 'LD_AUDIT' in env:
        raise RuntimeError('the loader observer must be private to the native child')
    audit = plan.get('queue_wait_audit')
    if audit is None:
        return env
    path = Path(audit['library']['path'])
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != audit['library']['sha256']:
            raise RuntimeError('the pinned queue wait observer changed')
    if path.stat().st_size != audit['library']['bytes'] or not plan.get('native_validation_probe'):
        raise RuntimeError('the observer requires its pinned file and private validation')
    return dict(env, LD_AUDIT=str(path),
                STRATA_WAIT_AUDIT_RETURN=audit['return_pc'], STRATA_WAIT_AUDIT_PARENT=str(audit['parent_offset']))


if __name__ == '__main__':
    engine_once.native_environment = native_environment
    sys.exit(engine_once.main())
