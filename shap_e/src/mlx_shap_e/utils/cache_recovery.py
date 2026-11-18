from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


def with_corrupt_cache_retry(
    fn: Callable[[], T],
    *,
    on_retry: Callable[[Path], None] | None = None,
    max_attempts: int = 2,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    attempt = 1
    while True:
        try:
            return fn()
        except RuntimeError as exc:
            msg = str(exc)
            if "should have hash" not in msg or "The file " not in msg:
                raise
            start = msg.find("The file ") + len("The file ")
            end = msg.find(" should have hash", start)
            if start < 0 or end < 0:
                raise
            bad_path = Path(msg[start:end]).expanduser()
            if not bad_path.exists():
                raise
            if on_retry is not None:
                on_retry(bad_path)
            bad_path.unlink()
            attempt += 1
            if attempt > max_attempts:
                raise
