set may-call-functions off
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
    wait = None
    while frame is not None:
        name = frame.name() or ''
        if 'NEO::IoctlHelperXe::execBuffer' in name:
            ioctl = frame
        if 'NEO::DrmCommandStreamReceiver<' in name and 'flushInternal' in name:
            csr = frame
        if 'NEO::CommandStreamReceiver::baseWaitFunction' in name:
            wait = frame
            if csr is None:
                csr = frame
        elif csr is None and any('NEO::CommandStreamReceiver::' + method in name for method in ['waitForCompletionWithTimeout', 'downloadTagAllocation', 'downloadAllocation']):
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
    if wait:
        wait.select()
        print('COMPLETION WAIT FRAME:', wait.name())
        for expression in ['taskCountToWait', 'pollAddress', '*pollAddress', 'params']:
            print('READ:', expression)
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
