"""Main-supplied request bytes/response parser/shutdown remain outside proposal.

Use this function as Owner.run(..., interaction=...) after Main's preflight gate.
It never starts a process and never changes os.environ or product source.
"""
from owned_stack_probe import StackProbe, StallCaptured, require


def with_probe(module, debugger_owner, native_io, native_entry, record, persist,
               cwd, *, await_ready, run_requests, finish_stalled,
               shutdown_payload, silence_seconds=240):
    """Callbacks execute in the same direct parent controller, without threads.

    await_ready(io): Main's existing bounded READY parser + preload-grant check.
    run_requests(io, gate): Main's exact token-sequence protocol, calling gate()
        before each request. Sends are also guarded after the first stall.
    finish_stalled(io): Main's bounded QUIT/STOP parser or watchdog-exit observer;
        must complete native exit. It cannot submit another model request.
    shutdown_payload: exact predeclared allowed shutdown bytes, or None for
        watch-original-watchdog mode. All other sends are denied after stall.
    """
    require(not record.get('probe'), 'one probe per native diagnostic process')
    await_ready(native_io)
    probe_record = record.setdefault('probe', {})
    probe = StackProbe(module, debugger_owner, native_io, native_entry,
                       probe_record, persist, cwd, silence_seconds=silence_seconds,
                       shutdown_payload=shutdown_payload)
    try:
        try:
            run_requests(native_io, probe.require_requests_allowed)
        except StallCaptured:
            require(probe.requests_stopped, 'stall did not close request gate')
            require(probe_record['captures'][0]['tracer_clear'], 'unsafe post-capture state')
            probe_record['main_selected_post_stall_handler'] = True
            persist()
            finish_stalled(native_io)
    finally:
        # Before native Owner's final scan/reap; an exception goes directly to
        # that Owner's existing pinned-pidfd cleanup. Never adopt or signal here.
        probe.close()
