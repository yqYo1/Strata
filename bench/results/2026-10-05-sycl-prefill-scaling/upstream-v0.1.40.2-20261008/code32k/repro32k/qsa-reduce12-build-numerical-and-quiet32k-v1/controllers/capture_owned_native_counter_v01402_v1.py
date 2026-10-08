"""Read native counter fields only in an already stopped, owned inferior.

Field names follow NEO26.31.39395.14 source, not guessed byte offsets.
No inferior function calls, attachments, resume or event/runtime changes.
"""
from pathlib import Path
import datetime, hashlib, json, re, sys, time

def console_text(path):
    text = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('~"'):
            text.append(json.loads(line[1:]))
    return ''.join(text)

def event_frames(text):
    found = []
    for thread in re.finditer(r'^Thread (\d+) .*?\n(.*?)(?=^Thread \d+ |\Z)', text, re.M|re.S):
        stack = thread[2]
        matches = list(re.finditer(r'^#(\d+)\s+[^\n]*L0::EventImp<[^\n]*>::queryCounterBasedEventStatus[^\n]*', stack, re.M))
        if not matches:
            continue
        frame = matches[0]
        address = re.search(r'\bthis=(0x[0-9a-fA-F]+)', frame[0])
        poll = re.search(r'pollAddress=(0x[0-9a-fA-F]+)', stack)
        found.append({'thread': int(thread[1]), 'frame': int(frame[1]),
                      'stack_this': address[1] if address else None,
                      'stack_poll_address': poll[1] if poll else None, 'stack': stack})
    return found

def value_of(result):
    match = re.search(r'\bvalue=("(?:[^"\\]|\\.)*")', result)
    if not match:
        raise ValueError('GDB did not return a quoted value')
    return int(json.loads(match[1]), 0)

def capture(g, snapshot, output, *, budget_seconds=25):
    # Import the same ownership helper that constructed this process.
    from owned_gdb import process_identity
    assert g.inferior and g.debugger_identity and g.exit_code is None and g.exit_signal is None
    assert g.stops and g.stops[-1] != 'resumed'
    for identity in [g.inferior, g.debugger_identity]:
        now = process_identity(identity['pid'])
        assert now and now['start_ticks'] == identity['start_ticks']
    snapshot_path = Path(snapshot['path'])
    assert snapshot_path.resolve().parent == g.output.resolve() and not snapshot['resumed']
    assert snapshot_path.stat().st_size <= 4*1024**2
    output = Path(output)
    assert not output.exists()
    begin = time.monotonic()
    record = {'active': False, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Read-only expected/native host-counter fields at owned failure stop. First host partition only; not a dependency, lifetime, GPU completion or stall-fix proof.',
              'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inferior': g.inferior, 'debugger': g.debugger_identity, 'stop': g.stops[-1],
              'snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
              'inferior_function_calls': False, 'resumed': False, 'events': [], 'errors': []}
    def command(text):
        left = budget_seconds-(time.monotonic()-begin)
        if left <= 0:
            raise TimeoutError('bounded native field capture budget')
        return g.command(text, seconds=min(2, left))
    threads = command('-thread-info')
    selected = re.search(r'current-thread-id="(\d+)"', threads)
    selected_id = selected[1] if selected else None
    try:
        command('-gdb-set may-call-functions off')
        frames = event_frames(console_text(snapshot_path))
        record['matching_frames'] = len(frames)
        for frame in frames[:8]:
            row = dict(frame)
            row['raw_commands'] = []
            record['events'].append(row)
            try:
                command('-thread-select '+str(frame['thread']))
                command('-stack-select-frame '+str(frame['frame']))
                prefix = 'this->inOrderExecHelper.'
                data = prefix+'sharableEventDataHelper.eventDataPtr'
                expressions = [('event_this', 'this'), ('data_pointer', data),
                               ('data_assigned', prefix+'dataAssigned'),
                               ('base_host_address', prefix+'baseHostCpuAddress'),
                               ('heapfull_profiling', 'this->heapfullCbEventWithProfiling')]
                for name, expression in expressions:
                    result = command('-data-evaluate-expression '+json.dumps('(unsigned long long)('+expression+')'))
                    row['raw_commands'].append({'name': name, 'expression': expression, 'result': result})
                    row[name] = value_of(result)
                assert row['data_pointer'] and row['base_host_address'] and row['data_assigned']
                for name, member in [('expected_counter', 'counterValue'), ('counter_offset', 'counterOffset'),
                                     ('host_partitions', 'hostPartitions'), ('device_partitions', 'devicePartitions')]:
                    expression = data+'->'+member
                    result = command('-data-evaluate-expression '+json.dumps('(unsigned long long)('+expression+')'))
                    row['raw_commands'].append({'name': name, 'expression': expression, 'result': result})
                    row[name] = value_of(result)
                assert 0 <= row['counter_offset'] <= 2**32-1 and 1 <= row['host_partitions'] <= 64
                address = row['base_host_address']+row['counter_offset']
                assert 4096 <= address < 2**64-8
                row['first_host_counter_address'] = hex(address)
                result = command('-data-read-memory-bytes '+hex(address)+' 8')
                row['raw_commands'].append({'name': 'first_host_counter_memory', 'result': result})
                values = re.findall(r'\bcontents="([0-9a-fA-F]{16})"', result)
                assert len(values) == 1 and sys.byteorder == 'little'
                row['first_host_counter'] = int.from_bytes(bytes.fromhex(values[0]), 'little')
                row['first_counter_ge_expected'] = row['first_host_counter'] >= row['expected_counter']
                row['all_host_partitions_read'] = row['host_partitions'] == 1
                row['plain_counter_branch'] = not bool(row['heapfull_profiling'])
                if frame['stack_this']:
                    assert int(frame['stack_this'], 16) == row['event_this']
                row['captured'] = True
            except BaseException as error:
                row['captured'] = False
                row['error'] = repr(error)
    except BaseException as error:
        record['errors'].append(repr(error))
    finally:
        if selected_id:
            try:
                command('-thread-select '+selected_id)
            except BaseException as error:
                record['errors'].append(repr(error))
        record['elapsed_seconds'] = time.monotonic()-begin
        record['all_matching_fields_captured'] = bool(record.get('matching_frames')) and len(record['events']) == record['matching_frames'] and all(r.get('captured') for r in record['events']) and not record['errors']
        output.write_text(json.dumps(record, indent=2)+'\n')
    return record
