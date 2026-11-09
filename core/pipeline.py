"""Fetch + clean in one call, shared by the standalone app and the API."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from core import himalayas
from core.etl import clean_jobs


def load_live_jobs(max_pages: int = 75, pause: float = 0.5, time_limit: float = 75.0) -> tuple[pd.DataFrame, dict]:
    """Download the newest jobs from Himalayas and clean them.

    Returns the clean frame and a report with the fetch statistics, the
    count for every cleaning rule and the fetch time (UTC, ISO format).
    Raises :class:`core.himalayas.FetchError` if nothing could be downloaded.
    """
    raw_jobs, fetch_report = himalayas.fetch_jobs(max_pages=max_pages, pause=pause, time_limit=time_limit)
    frame, clean_report = clean_jobs(raw_jobs)
    report = {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": himalayas.SOURCE_URL,
        "fetch": fetch_report.as_dict(),
        "cleaning": clean_report,
    }
    return frame, report
