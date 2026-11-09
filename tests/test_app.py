"""Streamlit smoke tests in standalone mode with the Himalayas fetch replaced by synthetic data."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import himalayas
from tests.synthetic import make_raw_jobs

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture
def fake_fetch(monkeypatch):
    def fetch(**kwargs):
        jobs = make_raw_jobs()
        report = himalayas.FetchReport(pages=8, jobs=len(jobs), stopped_because="no more pages")
        return jobs, report

    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.setattr(himalayas, "fetch_jobs", fetch)
    _clear_caches()
    yield
    _clear_caches()


def _clear_caches():
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()


def test_every_tab_renders_without_exceptions(fake_fetch):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == [
        "📊 Market overview",
        "🔎 Job search",
        "💰 Salary insights",
        "🤖 Salary predictor",
        "🏷️ Category classifier",
        "🧭 Am I a fit?",
        "⚙️ About & pipeline",
    ]
    labels = [m.label for m in at.metric]
    for expected in ["Live jobs", "Predicted salary", "Accuracy", "Market match", "Rows in last fetch"]:
        assert expected in labels
    assert any("Source: [Himalayas]" in c.value for c in at.caption)


def test_search_filters_and_fit_inputs(fake_fetch):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_input(key="search_q").input("devops").run()
    assert not at.exception
    assert any("matching jobs" in m.value for m in at.markdown)
    at.toggle[0].set_value(True).run()
    assert not at.exception and len(at.dataframe) >= 1
    at.text_area[0].input("Kubernetes, Terraform, AWS").run()
    assert not at.exception


def test_friendly_error_when_api_is_down(monkeypatch):
    def broken(**kwargs):
        raise himalayas.FetchError("Could not download any jobs from Himalayas (HTTP 503).")

    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.setattr(himalayas, "fetch_jobs", broken)
    _clear_caches()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "could not reach the Himalayas jobs API" in at.error[0].value
    assert not at.tabs
