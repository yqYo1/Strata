set startup-with-shell off
set pagination off
set print frame-arguments scalars
run
python
if gdb.selected_inferior().pid:
    gdb.execute('thread apply all bt 12')
    gdb.execute('info registers')
    gdb.execute('info sharedlibrary')
end
