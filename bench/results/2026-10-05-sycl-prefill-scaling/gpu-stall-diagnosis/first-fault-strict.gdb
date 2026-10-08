set startup-with-shell off
set pagination off
set confirm off
set print frame-arguments scalars
python
inferior_exit = [None]
def record_exit(event):
    inferior_exit[0] = getattr(event, 'exit_code', None)
gdb.events.exited.connect(record_exit)
end
run
python
if gdb.selected_inferior().pid:
    gdb.execute('thread apply all bt 12')
    gdb.execute('info registers')
    gdb.execute('info sharedlibrary')
    gdb.execute('quit 1')
elif inferior_exit[0] is None:
    gdb.execute('quit 1')
else:
    code = inferior_exit[0]
    gdb.execute('quit %d' % (code if 0 <= code <= 255 else 1))
end
