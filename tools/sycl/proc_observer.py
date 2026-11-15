"""Bounded /proc sampling across an owned child's exit; repository MIT.

An inaccessible /proc file is not proof of process exit. The caller retains
its unreaped child and pidfd, keeps supervising output/resources, and requires
positive exit plus its usual provider, validation and result checks afterward.
"""
import math
import time


class ProcObserver:
    def __init__(self, grace_seconds=2.0, clock=time.monotonic):
        if not math.isfinite(grace_seconds) or not 0 < grace_seconds <= 5:
            raise ValueError('finite /proc exit grace in (0,5] required')
        self.grace_seconds = grace_seconds
        self.clock = clock
        self.unavailable_since = None
        self.successful_samples = 0
        self.permission_denials = 0
        self.missing_samples = 0
        self.exit_confirmed_after_unavailable = False

    def sample(self, observe, exited):
        """Try a snapshot; tolerate a short gap, never infer exit from it."""
        if exited():
            # An unreaped zombie can expose an empty readable maps file. It is
            # no longer a live provider snapshot; do not parse it as one.
            self.exit_confirmed_after_unavailable |= self.unavailable_since is not None
            return False
        try:
            observe()
        except (PermissionError, FileNotFoundError, ProcessLookupError) as error:
            if isinstance(error, PermissionError):
                self.permission_denials += 1
            else:
                self.missing_samples += 1
            if exited():
                self.exit_confirmed_after_unavailable = True
                return False
            now = self.clock()
            if self.unavailable_since is None:
                self.unavailable_since = now
            if now - self.unavailable_since >= self.grace_seconds:
                raise RuntimeError('/proc unavailable while owned child remains live') from error
            return False
        # Validation errors raised by observe are deliberately not caught.
        self.successful_samples += 1
        self.unavailable_since = None
        return True

    def summary(self):
        return dict(successful_samples=self.successful_samples,
                    permission_denials=self.permission_denials,
                    missing_samples=self.missing_samples,
                    exit_confirmed_after_unavailable=self.exit_confirmed_after_unavailable,
                    grace_seconds=self.grace_seconds)
