"""Process-scoped, crash-released single writer lock for library mutations."""
from contextlib import contextmanager
from functools import wraps
import os

from .common import LibraryError


@contextmanager
def writer_lock(cat):
    if getattr(cat, "_writer_depth", 0):
        yield
        return
    path = cat.settings.home / "writer.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise LibraryError("writer_busy", "Another library writer is active") from None
        cat._writer_depth = 1
        try:
            yield
        finally:
            cat._writer_depth = 0
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def single_writer(func):
    @wraps(func)
    def wrapped(cat, *args, **kwargs):
        with writer_lock(cat):
            return func(cat, *args, **kwargs)
    return wrapped
