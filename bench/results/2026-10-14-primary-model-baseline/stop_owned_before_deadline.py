#!/usr/bin/env python3
"""STOP only the identified, still-active v5 test before its hard wall limit."""
import datetime
import hashlib
import json
import os
import stat
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
O = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-primary-model-diagnostic-root-v5')
ENGINE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-primary-native-build-root-v1/build/strata')
NATIVE_PID = 1501474


def identity(pid):
    text = Path(f'/proc/{pid}/stat').read_text()
    fields = text[text.rfind(')') + 2:].split()
    return dict(pid=pid, ppid=int(fields[1]), sid=int(fields[3]), start_ticks=int(fields[19]))


def main():
    record = json.loads((O / 'record.json').read_text())
    result = json.loads((O / 'result.json').read_text())
    assert record['active'] and result['ready'] and not result['passed']
    assert hashlib.sha256(ENGINE.read_bytes()).hexdigest() == record['binary_sha256']
    server = identity(result['server_pid'])
    child = identity(NATIVE_PID)
    wrapper = identity(child['ppid'])
    assert child['sid'] == wrapper['sid'] == server['sid']
    assert wrapper['ppid'] == server['pid']
    assert Path(f'/proc/{NATIVE_PID}/exe').resolve() == ENGINE
    output = O / 'control-action.json'
    assert not output.exists()
    fd = os.open(f'/proc/{NATIVE_PID}/fd/0', os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        assert stat.S_ISFIFO(os.fstat(fd).st_mode)
        assert identity(NATIVE_PID) == child and identity(server['pid']) == server
        action = dict(action='STOP on owned engine stdin', original_wall_limit_seconds=1200,
            reason='steady progress, but full-validation diagnostic cannot complete remaining chunks within original wall budget',
            final_status_requirement='incomplete, not model/performance/256K pass', server_identity=server,
            engine_identity=child, wrapper_identity=wrapper, engine_sha256=record['binary_sha256'],
            executed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            fd_inode=os.fstat(fd).st_ino, message='STOP\n', submitted=False)
        output.write_text(json.dumps(action, indent=2) + '\n')
        assert os.write(fd, b'STOP\n') == 5
        action['submitted'] = True
        output.write_text(json.dumps(action, indent=2) + '\n')
    finally:
        os.close(fd)
    print(json.dumps(dict(STOP_submitted=True, only_owned_test=True)), flush=True)


if __name__ == '__main__':
    main()
