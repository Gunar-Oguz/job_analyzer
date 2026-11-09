import pytest
import requests

from core import himalayas
from tests.synthetic import fake_page, make_raw_jobs


class FakeResponse:
    def __init__(self, status: int = 200, payload: dict | None = None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    """Serves synthetic pages by cursor. ``script`` overrides responses per call number."""

    def __init__(self, jobs, script=None):
        self.jobs = jobs
        self.script = script or {}
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params), "headers": headers})
        action = self.script.get(len(self.calls))
        if isinstance(action, Exception):
            raise action
        if isinstance(action, int):
            return FakeResponse(action)
        start = int(params.get("cursor") or 0)
        return FakeResponse(200, fake_page(self.jobs, start))


class FakeClock:
    def __init__(self, step: float = 0.0):
        self.now = 0.0
        self.step = step  # time that passes on every clock read (simulates slow requests)
        self.sleeps = []

    def __call__(self):
        self.now += self.step
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def run(session, clock=None, **kwargs):
    clock = clock or FakeClock()
    return himalayas.fetch_jobs(session=session, sleep=clock.sleep, clock=clock, **kwargs)


def test_paginates_with_next_cursor_until_the_end():
    session = FakeSession(make_raw_jobs(50))
    jobs, report = run(session, max_pages=10)
    assert len(jobs) == 50
    assert report.pages == 3 and report.stopped_because == "no more pages"
    assert [c["params"].get("cursor") for c in session.calls] == [None, "20", "40"]
    assert all(c["params"]["limit"] == 20 for c in session.calls)
    assert session.calls[0]["headers"]["User-Agent"] == himalayas.USER_AGENT


def test_page_limit_and_polite_pause():
    clock = FakeClock()
    jobs, report = run(FakeSession(make_raw_jobs(100)), clock=clock, max_pages=2, pause=0.7)
    assert len(jobs) == 40 and report.stopped_because == "page limit"
    assert clock.sleeps == [0.7, 0.7]


def test_retries_then_succeeds():
    session = FakeSession(make_raw_jobs(20), script={1: 503, 2: requests.ConnectionError("boom")})
    clock = FakeClock()
    jobs, report = run(session, clock=clock, max_pages=5)
    assert len(jobs) == 20
    assert report.retries == 2
    assert clock.sleeps[:2] == [1.0, 2.0]  # exponential backoff


def test_partial_failure_keeps_jobs_already_fetched():
    # page 1 succeeds, page 2 fails on every attempt (1 try + 2 retries)
    session = FakeSession(make_raw_jobs(100), script={2: 500, 3: 500, 4: 500})
    jobs, report = run(session, max_pages=5, max_retries=2)
    assert len(jobs) == 20
    assert report.stopped_because == "request failed"
    assert report.errors


def test_non_retryable_error_is_not_retried():
    session = FakeSession(make_raw_jobs(100), script={2: 404})
    jobs, report = run(session, max_pages=5)
    assert len(jobs) == 20 and report.retries == 0 and len(session.calls) == 2


def test_total_failure_raises():
    session = FakeSession(make_raw_jobs(20), script={i: 503 for i in range(1, 10)})
    with pytest.raises(himalayas.FetchError):
        run(session, max_pages=5, max_retries=3)


def test_time_limit_stops_and_keeps_data():
    clock = FakeClock(step=4.0)  # every clock read advances 4 seconds
    jobs, report = run(FakeSession(make_raw_jobs(200)), clock=clock, max_pages=10, time_limit=20)
    assert 0 < len(jobs) < 200
    assert report.stopped_because == "time limit"


def test_backoff_is_capped():
    assert himalayas.backoff_seconds(0) == 1
    assert himalayas.backoff_seconds(3) == 8
    assert himalayas.backoff_seconds(10) == 30
