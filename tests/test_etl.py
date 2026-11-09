import pandas as pd
import pytest

from core.etl import (
    annual_usd_salary,
    clean_html,
    clean_jobs,
    drop_expired,
    frame_to_records,
    records_to_frame,
)
from tests.conftest import NOW_DT
from tests.synthetic import NOW, make_raw_jobs


def test_clean_html_strips_tags_entities_and_scripts():
    raw = "<div><h3>Role</h3><p>Python&amp;SQL</p><script>alert(1)</script><ul><li>One</li><li>Two</li></ul></div>"
    assert clean_html(raw) == "Role Python&SQL One Two"
    assert clean_html(None) == ""


@pytest.mark.parametrize(
    ("low", "high", "currency", "period", "expected"),
    [
        (100_000, 150_000, "USD", "annual", (100_000, 150_000, "ok")),
        (50, 60, "USD", "hourly", (104_000, 124_800, "ok")),
        (5_000, 6_000, "USD", "monthly", (60_000, 72_000, "ok")),
        (90_000, None, "USD", "annual", (90_000, 90_000, "ok")),
        (150_000, 100_000, "usd", "yearly", (100_000, 150_000, "ok")),
        (80_000, 90_000, "EUR", "annual", (None, None, "non_usd")),
        (1_000, 1_600, "USD", "annual", (None, None, "out_of_range")),
        (900_000, 2_000_000, "USD", "annual", (None, None, "out_of_range")),
        (10, 20, "USD", "fortnightly", (None, None, "unknown_period")),
        (None, None, None, "annual", (None, None, "missing")),
        (0, "", "USD", "annual", (None, None, "missing")),
    ],
)
def test_annual_usd_salary(low, high, currency, period, expected):
    assert annual_usd_salary(low, high, currency, period) == expected


def test_clean_jobs_applies_every_rule_and_reports_counts():
    raw = make_raw_jobs(10)
    raw.append(dict(raw[0]))  # duplicate guid
    raw.append({**raw[1], "guid": "https://jobs.example.org/expired", "expiryDate": NOW - 60})
    raw.append({**raw[2], "guid": "https://jobs.example.org/no-title", "title": "  "})
    raw.append({**raw[3], "guid": "https://jobs.example.org/no-link", "applicationLink": ""})
    frame, report = clean_jobs(raw, now=NOW_DT)

    assert report["raw_rows"] == 14
    assert report["dropped_duplicates"] == 1
    assert report["dropped_expired"] == 1
    assert report["dropped_missing_title_or_link"] == 2
    assert report["clean_rows"] == len(frame) == 10
    salary_counts = sum(report[k] for k in report if k.startswith("salary_"))
    assert salary_counts == 10


def test_clean_jobs_schema_and_values(raw_jobs):
    frame, _ = clean_jobs(raw_jobs[:5], now=NOW_DT)
    first = frame.iloc[0]
    assert first["apply_url"] == raw_jobs[0]["applicationLink"]  # passed through unchanged
    assert first["source_url"] == raw_jobs[0]["guid"]
    assert "<" not in first["description"]
    assert first["category"] == raw_jobs[0]["parentCategories"][0]
    assert isinstance(first["pub_date"], pd.Timestamp)
    assert frame["salary_usd"].dtype == float
    assert frame.iloc[0]["is_worldwide"]  # first synthetic job has no country restriction
    assert "Python" in first["skills"]


def test_clean_jobs_empty_input():
    frame, report = clean_jobs([])
    assert frame.empty and report["clean_rows"] == 0
    assert "salary_usd" in frame.columns


def test_drop_expired(jobs):
    later = NOW_DT + pd.Timedelta(days=365)
    assert drop_expired(jobs, now=NOW_DT).shape[0] == len(jobs)
    assert drop_expired(jobs, now=later).empty


def test_records_round_trip(jobs):
    back = records_to_frame(frame_to_records(jobs))
    assert len(back) == len(jobs)
    assert back["skills"].tolist() == jobs["skills"].tolist()
    assert back["salary_usd"].isna().sum() == jobs["salary_usd"].isna().sum()
    assert str(back["pub_date"].dt.tz) == "UTC"
