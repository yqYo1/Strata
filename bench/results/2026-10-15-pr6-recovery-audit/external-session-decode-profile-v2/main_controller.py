"""Unexecuted v2 integration candidate. Import from Main's serial owned runner.

Main imports the controller. The explicit --runtime-child mode only preserves
the launcher's injection and records its own native child's actual exit status.
Use the unchanged pinned direct_owner.Owner for both launch and control commands.
"""
from pathlib import Path
import os
import re
import stat
import hashlib
import json
import select
import subprocess
import sys
import time

MIB = 1 << 20
MAX_MARKERS = 16
MAX_MARKER_BYTES = 512
MAX_CHANNEL_BYTES = 8192
PHASES = ("START", "ACTIVE", "DRAINED", "PAUSED", "FINISHED")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def raw_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def prepare_channels(directory):
    """Main calls before model spawn. New private directory; no overwrite."""
    directory = Path(directory)
    require(directory.is_absolute(), "absolute private channel directory")
    directory.mkdir(mode=0o700, exist_ok=False)
    for name in ("markers", "acks"):
        os.mkfifo(directory / name, 0o600)
    return directory


def open_fifo(directory_fd, name):
    fd = os.open(name, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
                 dir_fd=directory_fd)
    v = os.fstat(fd)
    require(stat.S_ISFIFO(v.st_mode) and stat.S_IMODE(v.st_mode) == 0o600
            and v.st_uid == os.getuid() and v.st_nlink == 1, "private exclusive FIFO")
    return fd


class Budget:
    """Finite limits with polling overshoot, explicitly NOT an aggregate hard quota.

    The existing Owner supplies RLIMIT_FSIZE=16MiB on every child and RSS88GiB.
    Successful completion additionally validates file names, counts and final bytes.
    No files are deleted, truncated, rotated or compressed by this candidate.
    """
    def __init__(self, job, provider_dir):
        self.job = Path(job)
        self.provider_dir = Path(provider_dir)
        self.observed = {}
        self.last_check = 0.0
        self.last_success = time.monotonic()
        self.max_observation_gap = 0.0

    def check(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_check < .02:
            return
        self.last_check = now
        sizes = []
        provider_files = []
        for p in self.job.rglob("*"):
            v = p.lstat()
            require(not stat.S_ISLNK(v.st_mode), "job symlink rejected")
            if stat.S_ISREG(v.st_mode):
                sizes.append((p, v.st_size))
                if self.provider_dir in p.parents:
                    provider_files.append((p, v.st_size))
        text = sum(n for p, n in sizes if p.suffix in (".stdout", ".stderr", ".txt"))
        aggregate = sum(n for _, n in provider_files)
        total = sum(n for _, n in sizes)
        self.max_observation_gap = max(self.max_observation_gap, now-self.last_success)
        self.observed = dict(raw_ns=raw_ns(), text_bytes=text,
                             provider_bytes=aggregate, job_bytes=total,
                             provider_file_count=len(provider_files),
                             observation_gap_s=now-self.last_success,
                             max_observation_gap_s=self.max_observation_gap)
        # Leave reserve for owner persistence and finite termination logs.
        require(text < 48*MIB, "text soft threshold: incomplete")
        require(aggregate < 48*MIB, "provider aggregate soft threshold: incomplete")
        require(total < 192*MIB, "job soft threshold: incomplete")
        require(len(provider_files) <= 8, "provider file-count threshold: incomplete")
        require(all(n <= 16*MIB for _, n in provider_files), "file cap: incomplete")
        for p, _ in provider_files:
            require(p.name in ("run_config.json", "device_timing.txt"),
                    "unexpected provider output: incomplete")
        self.last_success = now


class Controller:
    """Bounded FIFO state machine; control(action, native_poll) is injected.

    identify(pid) MUST verify pid/startticks against the native-wrapper receipt,
    binary hash and the pinned pidfd owned session. Do not adopt from markers.
    persist(records) writes an owner-budgeted receipt; never stdout/protocol.
    """
    def __init__(self, directory, control, identify, persist, budget):
        self.directory = Path(directory)
        self.control = control
        self.identify = identify
        self.persist = persist
        self.budget = budget
        self.records = []
        self.pending = bytearray()
        self.received = 0
        self.phase = 0
        self.key = None
        self.native_identity = None
        self.busy = False
        self.incomplete = True
        self.channel_fds = []
        dfd = os.open(directory, os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            v = os.fstat(dfd)
            require(stat.S_IMODE(v.st_mode) == 0o700 and v.st_uid == os.getuid(),
                    "private channel directory")
            self.markers = open_fifo(dfd, "markers")
            self.channel_fds.append(self.markers)
            self.acks = open_fifo(dfd, "acks")
            self.channel_fds.append(self.acks)
        except BaseException:
            self.close()
            raise
        finally:
            os.close(dfd)

    def close(self):
        for fd in self.channel_fds:
            os.close(fd)
        self.channel_fds.clear()

    def install(self, io):
        """Inside Owner.run(interaction=...). Keep original pinned owner poll."""
        original = io.poll

        def combined():
            original()
            self.poll(original)

        io.poll = combined
        return original

    def poll(self, native_poll):
        self.budget.check()
        if self.busy:
            return
        try:
            data = os.read(self.markers, 4096)
        except BlockingIOError:
            return
        self.received += len(data)
        require(self.received <= MAX_CHANNEL_BYTES, "channel byte cap")
        self.pending.extend(data)
        require(len(self.pending) <= MAX_CHANNEL_BYTES, "marker pending cap")
        while b"\n" in self.pending:
            line, _, tail = self.pending.partition(b"\n")
            self.pending[:] = tail
            require(len(line) < MAX_MARKER_BYTES, "marker record cap")
            self.handle(line.decode("ascii"), native_poll)
        require(len(self.pending) < MAX_MARKER_BYTES, "unterminated marker cap")

    def handle(self, line, native_poll):
        require(len(self.records) < MAX_MARKERS, "marker count cap")
        words = line.split()
        require(words and words[0] == "PROFILE2", "marker version")
        fields = {}
        for w in words[1:]:
            k, v = w.split("=", 1)
            require(k not in fields, "duplicate marker field")
            fields[k] = v
        self.records.append(dict(received_raw_ns=raw_ns(), source=fields))
        self.persist(self.records)
        require(fields.get("phase") not in ("FAILED", "MISSED"),
                "source failed or missed target: incomplete")
        require(self.phase < len(PHASES) and fields["phase"] == PHASES[self.phase],
                "out-of-order or repeated source phase")
        require(fields["request"] == "1" and fields["decode"] == "8"
                and fields["T"] == "4" and fields["layers"] == "48"
                and fields["work"] == "normal_mtp4", "selected actual work")
        key = tuple(fields[k] for k in ("pid", "tid", "request", "decode", "verifier",
                                        "window", "T", "pos", "compute", "copy"))
        if self.key is None:
            self.key = key
            self.native_identity = self.identify(int(fields["pid"]))
            require(self.native_identity, "native identity not independently pinned")
        require(self.key == key, "window identity changed")
        require(int(fields["raw_ns"]) > 0, "source RAW clock missing")
        if fields["phase"] in ("START", "DRAINED"):
            action = "resume" if fields["phase"] == "START" else "pause"
            before = raw_ns()
            self.busy = True
            try:
                receipt = self.control(action, native_poll)
            finally:
                self.busy = False
            after = raw_ns()
            require(receipt.get("success") is True, "session control failed")
            require(after-before < 3_000_000_000, "session control exceeded 3s")
            self.records[-1]["control"] = dict(action=action, begin_raw_ns=before,
                                                end_raw_ns=after, receipt=receipt)
            self.budget.check(force=True)
            native_poll()
            ack = f'ACK {fields["phase"]} {fields["pid"]} {fields["window"]}\n'.encode()
            require(len(ack) < 128 and os.write(self.acks, ack) == len(ack), "ack write")
            self.records[-1]["ack_sent_raw_ns"] = raw_ns()
            self.persist(self.records)
        self.phase += 1
        # This only confirms the source handshake. Provider callback coverage and
        # actual model/native exit/fault gates remain separate and default false.
        self.incomplete = self.phase != len(PHASES)


def owned_cli_control(control_owner, unitrace, session, safe_env, cwd, budget,
                      model_closed=lambda: False):
    """Return control callback; use a distinct Owner instance for CLI children.

    Their leaders stay unreaped through their final scan; the native owner's
    original poll continues while CLI runs. No Popen.poll/wait/communicate here.
    safe_env is Main's explicit non-secret exec whitelist, without LD_PRELOAD,
    PTI/UNITRACE or XPTI injection; no inherited os.environ snapshot.
    """
    require(re.fullmatch(r"[A-Za-z0-9]{16,64}", session), "unique alphanumeric session")
    require(not any(k.startswith(("UNITRACE_", "PTI_", "XPTI_")) or k == "LD_PRELOAD"
                    for k in safe_env), "control CLI environment must be uninstrumented")
    calls = 0
    released = False

    def control(action, native_poll):
        nonlocal calls, released
        if action == "stop":
            require(not released and model_closed(), "session release requires closed model owner")
            released = True
        else:
            require(action in ("resume", "pause") and calls < 2 and not released,
                    "two capture control calls only")
            require((calls == 0 and action == "resume") or (calls == 1 and action == "pause"),
                    "control order")
            calls += 1

        def interaction(ci, entry):
            ci.proc.stdin.close()
            deadline = time.monotonic()+2.5
            while not ci.exit_watch.poll(0) or not ci.eof:
                native_poll()
                budget.check()
                ci.poll()
                require(time.monotonic() < deadline, "control CLI 2.5s deadline")
                ci.read()
                ci.pending.clear()
                # A bounded select keeps both owner polls live without busy spin.
                ci.exit_watch.poll(5)

        entry, control_stdout, control_stderr = control_owner.run(
            "session-"+action, [str(unitrace), "--"+action, session],
            env=dict(safe_env), cwd=str(cwd), interaction=interaction,
            wall=3, text_cap=64<<10, rss_cap=512<<20,
            total_cap=256*MIB, cpu=3, file_cap=16*MIB)
        success = (entry.get("exit_code") == 0 and entry.get("direct_child_reaped")
                   and entry.get("session_empty") and entry.get("observation_complete")
                   and not entry.get("error") and not entry.get("errors")
                   and not entry.get("cleanup"))
        # A stopped session makes resume/pause return without changing state.
        # Require the documented state-write confirmation as well as exit 0.
        output = control_stdout.read_bytes() + control_stderr.read_bytes()
        require(len(output) <= 64<<10, "control log budget")
        state = {"resume": "resumed", "pause": "paused", "stop": "stopped"}[action]
        expected = f'[INFO] Session {session} is {state}'
        confirmed = expected.encode() in output
        return dict(success=bool(success and confirmed), label=entry["label"],
                    state_write_confirmation=confirmed,
                    pid=entry.get("pid"), exit_code=entry.get("exit_code"))

    return control


def native_identity(pid):
    # Explicit PID only. No process discovery or ownership adoption here.
    v = Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, state=v[0], ppid=int(v[1]), sid=int(v[3]), start_ticks=int(v[19]))


def exclusive_json(path, value):
    data = (json.dumps(value, indent=2)+"\n").encode()
    require(len(data) <= 16<<10, "runtime receipt budget")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            require(n > 0, "runtime receipt write")
            view = view[n:]
    finally:
        os.close(fd)


def runtime_child(job, native_argv):
    """Main-owned session descendant, not an independent session owner.

    Environment is inherited intact without enumeration, logging or mutation.
    Outer unchanged Owner supervises the launcher/wrapper/native and deadlines.
    No signals or numeric PID adoption. Native remains unreaped until its final
    identity check; unitrace leader remains the outer unreaped session anchor.
    """
    root = Path(job)
    v = root.lstat()
    require(root.is_absolute() and stat.S_ISDIR(v.st_mode) and v.st_uid == os.getuid()
            and stat.S_IMODE(v.st_mode) == 0o700, "private existing job")
    require(native_argv and Path(native_argv[0]).is_absolute(), "absolute native executable")
    with Path(native_argv[0]).open("rb") as stream:
        binary_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    entry = dict(binary=native_argv[0], binary_sha256=binary_sha,
                 wrapper=native_identity(os.getpid()), begin_raw_ns=raw_ns(),
                 native_child_reaped=False)
    child = subprocess.Popen(native_argv)  # inherit stdin/stdout/stderr and frozen injected env
    fd = os.pidfd_open(child.pid)
    try:
        entry["native"] = native_identity(child.pid)
        require(entry["native"]["sid"] == entry["wrapper"]["sid"], "native left owned session")
        exclusive_json(root/"runtime-start.json", entry)
        watch = select.poll()
        watch.register(fd, select.POLLIN)
        while not watch.poll(100):
            pass  # outer Owner supplies the finite 1800s start-to-reap budget
        final = native_identity(child.pid)
        require(final["start_ticks"] == entry["native"]["start_ticks"]
                and final["sid"] == entry["native"]["sid"], "unreaped native identity changed")
        entry["native_before_reap"] = final
        code = child.wait()  # only after pidfd death and final identity check
        entry.update(exit_code=code, native_child_reaped=True, end_raw_ns=raw_ns())
        exclusive_json(root/"runtime-exit.json", entry)
        return code if code >= 0 else 128-code
    finally:
        os.close(fd)


if __name__ == "__main__":
    require(len(sys.argv) >= 4 and sys.argv[1] == "--runtime-child",
            "only Main's explicit --runtime-child job executable [args...] mode is supported")
    sys.exit(runtime_child(sys.argv[2], sys.argv[3:]))
