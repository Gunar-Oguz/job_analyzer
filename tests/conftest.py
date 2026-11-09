"""Shared fixtures. Nothing in the test suite touches the network."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import requests

from core import ml
from core.etl import clean_jobs
from tests.synthetic import NOW, make_raw_jobs

NOW_DT = datetime.fromtimestamp(NOW, tz=UTC)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any test tries a real HTTP request."""

    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not use the network")

    monkeypatch.setattr(requests.Session, "request", blocked)
    monkeypatch.setattr(requests, "request", blocked)


@pytest.fixture(scope="session")
def raw_jobs() -> list[dict]:
    return make_raw_jobs()


@pytest.fixture(scope="session")
def jobs(raw_jobs):
    frame, _ = clean_jobs(raw_jobs, now=NOW_DT)
    return frame


@pytest.fixture(scope="session")
def models(jobs) -> ml.Models:
    return ml.train_all(jobs)
