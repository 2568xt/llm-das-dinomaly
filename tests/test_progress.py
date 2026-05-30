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
