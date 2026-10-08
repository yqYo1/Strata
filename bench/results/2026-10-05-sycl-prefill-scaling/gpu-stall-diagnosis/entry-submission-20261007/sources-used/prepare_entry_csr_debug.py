"""Freeze a second bounded full-context diagnostic with read-only CSR inspection."""
from pathlib import Path

base = Path(__file__).parent
source = (base/'run_full_entry_capacity.py').read_text()
def change(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)

change("out = base / 'full-context-entry-capacity'", "out = base / 'full-context-entry-csr'")
change("7200-second capacity deadline; original tuning; L0 API entries/error results without successful argument dumps", "300-second diagnostic deadline; read-only CSR inspection; original tuning; L0 API entries/error results without successful argument dumps")
change("'diagnostic_deadline_seconds': 7200, 'log_limit_bytes': 8 * 1024**3", "'diagnostic_deadline_seconds': 300, 'log_limit_bytes': 2 * 1024**3")
change("health = json.loads((base / 'api-entry-profile-health/record.json').read_text())", "health = json.loads((base / 'post-entry-stall-health/record.json').read_text())")
change("assert health['healthy'] and health['boot_id'] == record['boot_id']", "assert health['healthy'] and health['boot_id'] == record['boot_id']\n    owned = json.loads((base/'full-context-entry-capacity/record.json').read_text())\n    assert not owned['active'] and not owned['new_fault_messages']\n    assert not owned['cleanup']['inferior_survived'] and not owned['cleanup']['gdb_survived']\n    for key in ['inferior', 'debugger']:\n        assert not Path('/proc',str(owned[key]['pid'])).exists()")
change("record['capacity_deadline_seconds'] = 7200", "record['capacity_deadline_seconds'] = 7200\n    record['diagnostic_only_deadline_seconds'] = 300")
change("g.run()", "g.command('-gdb-set may-call-functions off')\n    g.run()")
change("signature = (current['minor_faults'], current['major_faults'], current['io']['rchar'], current['io']['read_bytes'], json.dumps([v['busy_cycles'] for v in current['gpu']], sort_keys=True))", "signature = (total_log, current['minor_faults'], current['major_faults'], current['io']['rchar'], current['io']['read_bytes'])")
change("if current['gpu'] and record['unchanged_progress_seconds'] >= 60 and snapshots < 2 and time.monotonic() - last_snapshot >= 30:", "if current['gpu'] and record['unchanged_progress_seconds'] >= 30 and snapshots < 2 and time.monotonic() - last_snapshot >= 15:")
change("record['snapshots'].append(g.snapshot('unchanged-progress-' + str(snapshots)))\n                    snapshots += 1; last_snapshot = time.monotonic()", "snap = g.snapshot('unchanged-api-and-io-' + str(snapshots), resume=False)\n                    if snap:\n                        start = g.raw.tell()\n                        g.command('-interpreter-exec console ' + json.dumps('source ' + str(base/'read-csr.gdb')))\n                        g.raw.flush()\n                        dump = out/'debugger'/('csr-' + str(snapshots) + '.mi.txt')\n                        with (out/'debugger/gdb-mi.stdout').open('rb') as source, dump.open('wb') as dest:\n                            source.seek(start)\n                            import shutil\n                            shutil.copyfileobj(source, dest)\n                        snap['read_only_csr_dump'] = str(dump)\n                        if snapshots == 0 and 'signal-name=\"SIGINT\"' in snap['stop']:\n                            g.command('-exec-continue --all')\n                            snap['resumed'] = True\n                            g.stops.append('resumed')\n                    record['snapshots'].append(snap)\n                    snapshots += 1; last_snapshot = time.monotonic()\n                    if snapshots == 2:\n                        record['stop_reason'] = 'Two read-only snapshots after unchanged API log and I/O; bounded investigation, not a full-context pass'\n                        break")
change("[Path(__file__), src, root/'sycl/tools/owned_gdb.py', root/'sycl/src/program/generate.cpp']", "[Path(__file__), base/'read-csr.gdb', src, root/'sycl/tools/owned_gdb.py', root/'sycl/src/program/generate.cpp']")
target = base/'run_full_entry_csr.py'
assert not target.exists()
target.write_text(source)
gdb = base/'read-csr.gdb'
assert not gdb.exists()
gdb.write_text('''set may-call-functions off
python
import gdb
thread = next((x for x in gdb.selected_inferior().threads() if x.num == 1), None)
if thread is None:
    print('NO MAIN THREAD')
else:
    thread.switch()
    frame = gdb.newest_frame()
    csr = None
    ioctl = None
    while frame is not None:
        name = frame.name() or ''
        if 'NEO::IoctlHelperXe::execBuffer' in name:
            ioctl = frame
        if 'NEO::DrmCommandStreamReceiver<' in name and 'flushInternal' in name:
            csr = frame
        frame = frame.older()
    if ioctl:
        ioctl.select()
        print('SUBMISSION FRAME:', ioctl.name())
        for expression in ['counterValue', 'completionGpuAddress', 'execBuffer']:
            try:
                gdb.execute('p ' + expression)
            except gdb.error as error:
                print('UNAVAILABLE', expression, error)
    if csr:
        csr.select()
        print('CSR FRAME:', csr.name())
        for expression in ['this', 'this->taskCount', 'this->latestSentTaskCount', 'this->latestFlushedTaskCount', 'this->tagAddress', '*this->tagAddress', 'this->completionFenceValuePointer', '*this->completionFenceValuePointer', 'this->completionFenceValue']:
            print('READ:', expression)
            try:
                gdb.execute('p ' + expression)
            except gdb.error as error:
                print('UNAVAILABLE', expression, error)
    else:
        print('NO CSR FRAME: preserve backtrace; do not infer zero counters')
end
''')
compile(source, str(target), 'exec')
print(target)
