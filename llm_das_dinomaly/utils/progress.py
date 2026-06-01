from __future__ import annotations

import sys
import time
from typing import Optional, TextIO


class ProgressBar:
    """Small dependency-free terminal progress bar."""

    def __init__(
        self,
        total: int,
        *,
        label: str,
        enabled: bool = True,
        stream: Optional[TextIO] = None,
        width: int = 28,
        min_interval_seconds: float = 5.0,
    ) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.width = width
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.current = 0
        self.started_at = time.time()
        self._last_render_at = 0.0
        self._last_len = 0
        self._is_tty = _stream_is_tty(self.stream)
        if self.enabled:
            self.render(force=True)

    def update(self, step: int = 1, *, suffix: str = "") -> None:
        self.current = min(self.total, self.current + step)
        if self.enabled:
            self.render(suffix=suffix)

    def close(self, *, suffix: str = "done") -> None:
        self.current = self.total
        if self.enabled:
            self.render(suffix=suffix, force=True)
            if self._is_tty:
                self.stream.write("\n")
            self.stream.flush()

    def render(self, *, suffix: str = "", force: bool = False) -> None:
        now = time.time()
        if (
            not self._is_tty
            and not force
            and self.current < self.total
            and now - self._last_render_at < self.min_interval_seconds
        ):
            return
        elapsed = max(1e-6, time.time() - self.started_at)
        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self.current / elapsed
        remaining = (self.total - self.current) / rate if rate > 0 else 0.0
        message = (
            f"{self.label} [{bar}] {self.current}/{self.total} "
            f"{ratio * 100:5.1f}% elapsed={_fmt_time(elapsed)} eta={_fmt_time(remaining)}"
        )
        if suffix:
            message += f" {suffix}"
        if self._is_tty:
            padding = " " * max(0, self._last_len - len(message))
            self.stream.write("\r" + message + padding)
            self._last_len = len(message)
        else:
            self.stream.write(message + "\n")
        self.stream.flush()
        self._last_render_at = now


def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


def _stream_is_tty(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(isatty()) if callable(isatty) else False
