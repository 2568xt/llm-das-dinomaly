from __future__ import annotations

from io import StringIO

from llm_das_dinomaly.utils import ProgressBar


def test_progress_bar_renders_and_closes():
    stream = StringIO()
    bar = ProgressBar(2, label="work", stream=stream)
    bar.update()
    bar.update(suffix="item=2")
    bar.close()
    output = stream.getvalue()
    assert "work" in output
    assert "2/2" in output
    assert output.endswith("\n")


def test_progress_bar_can_be_disabled():
    stream = StringIO()
    bar = ProgressBar(1, label="quiet", enabled=False, stream=stream)
    bar.update()
    bar.close()
    assert stream.getvalue() == ""


def test_progress_bar_non_tty_uses_tail_friendly_lines():
    stream = StringIO()
    bar = ProgressBar(3, label="log", stream=stream, min_interval_seconds=60)
    bar.update(suffix="first")
    bar.update()
    bar.update(suffix="done")
    bar.close()
    output = stream.getvalue()
    assert "\r" not in output
    assert "log" in output
    assert "3/3" in output
    assert output.endswith("\n")
