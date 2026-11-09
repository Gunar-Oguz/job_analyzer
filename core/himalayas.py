"""A small, polite client for the public Himalayas jobs API (no API key needed).

API docs: https://himalayas.app/api

The browse endpoint returns at most 20 jobs per request and is paged with a
``nextCursor`` token. Unauthenticated callers share one public allowance, so
this client:

* sends a descriptive User-Agent,
* pauses between requests,
* retries failed requests with capped exponential backoff,
* stops at an overall time limit, and
* keeps whatever it already downloaded if a later page fails.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

BROWSE_URL = "https://himalayas.app/jobs/api"
SOURCE_URL = "https://himalayas.app"
USER_AGENT = "job-analyzer-portfolio/1.0 (github.com/Gunar-Oguz/job_analyzer)"
PAGE_SIZE = 20  # the API maximum
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """Raised when not a single job could be downloaded."""


@dataclass
class FetchReport:
    """What happened during one fetch; shown on the pipeline tab."""

    pages: int = 0
    jobs: int = 0
    retries: int = 0
    elapsed_seconds: float = 0.0
    stopped_because: str = ""
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "pages": self.pages,
            "jobs": self.jobs,
            "retries": self.retries,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "stopped_because": self.stopped_because,
            "errors": self.errors,
        }


def backoff_seconds(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff (1s, 2s, 4s, ...) capped at ``cap`` seconds.

    We deliberately ignore ``Retry-After``: a server could ask us to wait for
    minutes, and a demo app should rather show partial data than hang.
    """
    return min(cap, base * (2**attempt))


def _get_page(
    session: requests.Session,
    params: dict,
    deadline: float,
    report: FetchReport,
    max_retries: int,
    timeout: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> dict:
    """GET one page with retries. Raises the last error if every attempt fails."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(BROWSE_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if response.status_code in RETRYABLE_STATUS:
                raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            retryable = not isinstance(error, requests.HTTPError) or (
                error.response is not None and error.response.status_code in RETRYABLE_STATUS
            )
            wait = backoff_seconds(attempt)
            if not retryable or attempt == max_retries or clock() + wait >= deadline:
                break
            report.retries += 1
            logger.warning("Himalayas request failed (%s); retrying in %.0fs", error, wait)
            sleep(wait)
    raise last_error  # type: ignore[misc]


def fetch_jobs(
    max_pages: int = 75,
    pause: float = 0.5,
    time_limit: float = 75.0,
    max_retries: int = 3,
    timeout: float = 15.0,
    session: requests.Session | None = None,
    on_page: Callable[[int, int], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list[dict], FetchReport]:
    """Download up to ``max_pages * 20`` of the newest jobs.

    Returns ``(jobs, report)``. If a page fails after all retries, or the
    time limit is reached, the jobs fetched so far are returned. Raises
    :class:`FetchError` only when no job at all could be downloaded.

    ``session``, ``sleep`` and ``clock`` can be replaced in tests so that no
    real network call or real waiting happens.
    """
    session = session or requests.Session()
    report = FetchReport()
    start = clock()
    deadline = start + time_limit
    jobs: list[dict] = []
    cursor: str | None = None

    while True:
        if report.pages >= max_pages:
            report.stopped_because = "page limit"
            break
        if clock() >= deadline:
            report.stopped_because = "time limit"
            break
        params = {"limit": PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        try:
            payload = _get_page(session, params, deadline, report, max_retries, timeout, sleep, clock)
        except (requests.RequestException, ValueError) as error:
            report.errors.append(str(error))
            report.stopped_because = "request failed"
            break
        page_jobs = payload.get("jobs") or []
        jobs.extend(page_jobs)
        report.pages += 1
        if on_page:
            on_page(report.pages, len(jobs))
        cursor = payload.get("nextCursor")
        if not page_jobs or not cursor:
            report.stopped_because = "no more pages"
            break
        sleep(pause)

    report.jobs = len(jobs)
    report.elapsed_seconds = clock() - start
    logger.info("Fetched %d jobs from %d pages in %.1fs", len(jobs), report.pages, report.elapsed_seconds)
    if not jobs:
        detail = report.errors[-1] if report.errors else report.stopped_because
        raise FetchError(f"Could not download any jobs from Himalayas ({detail}).")
    return jobs, report
