"""Turn raw Himalayas job dictionaries into one clean pandas DataFrame.

The main entry point is :func:`clean_jobs`. It returns the DataFrame and a
small report that says how many rows each cleaning rule removed, so the UI
can show the pipeline honestly.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from core.skills import extract_skills

UNCATEGORIZED = "Uncategorized"  # Himalayas jobs without a parent category
MIN_ANNUAL_SALARY = 10_000
MAX_ANNUAL_SALARY = 1_000_000

# Multipliers that turn a salary for one period into an annual salary.
PERIOD_TO_ANNUAL = {
    "annual": 1,
    "annually": 1,
    "yearly": 1,
    "year": 1,
    "monthly": 12,
    "month": 12,
    "weekly": 52,
    "week": 52,
    "daily": 260,
    "day": 260,
    "hourly": 2080,
    "hour": 2080,
}

LIST_COLUMNS = ["seniority_levels", "categories", "parent_categories", "locations", "timezones", "skills"]
DATE_COLUMNS = ["pub_date", "expiry_date"]
NUMERIC_COLUMNS = ["salary_min", "salary_max", "salary_usd_min", "salary_usd_max", "salary_usd"]

COLUMNS = [
    "id",
    "title",
    "company",
    "company_logo",
    "excerpt",
    "description",
    "employment_type",
    "seniority",
    "seniority_levels",
    "category",
    "parent_categories",
    "categories",
    "locations",
    "is_worldwide",
    "timezones",
    "currency",
    "salary_period",
    "salary_min",
    "salary_max",
    "salary_usd_min",
    "salary_usd_max",
    "salary_usd",
    "pub_date",
    "expiry_date",
    "apply_url",
    "source_url",
    "skills",
]

_BLOCK_TAGS = re.compile(r"</?(?:p|div|br|li|ul|ol|h[1-6]|tr|table|section)\b[^>]*>", re.IGNORECASE)
_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_ANY_TAG = re.compile(r"<[^>]+>")


def clean_html(text: str | None) -> str:
    """Strip HTML tags and entities and collapse whitespace.

    Block-level tags become spaces so that words from neighbouring
    paragraphs or list items do not stick together.
    """
    if not text:
        return ""
    text = _SCRIPT_STYLE.sub(" ", text)
    text = _BLOCK_TAGS.sub(" ", text)
    text = _ANY_TAG.sub("", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def humanise_category(slug: str) -> str:
    """Turn a Himalayas category slug such as "Data-Analyst" into "Data Analyst"."""
    return slug.replace("-", " ").strip()


def job_id(guid: str) -> str:
    """Return a short, stable id for a job (Himalayas ``guid`` values are long URLs)."""
    return hashlib.sha1(guid.encode("utf-8")).hexdigest()[:12]


def to_number(value: Any) -> float | None:
    """Convert a salary value to a positive float, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def annual_usd_salary(min_salary: Any, max_salary: Any, currency: str | None, period: str | None) -> tuple:
    """Return ``(min, max, status)`` as annual USD.

    ``status`` is one of ``"ok"``, ``"missing"``, ``"non_usd"``,
    ``"unknown_period"`` or ``"out_of_range"``. Only ``"ok"`` comes with numbers.
    Non-USD salaries are not converted, because we do not want to guess
    exchange rates; they are simply left out of the salary statistics.
    """
    low, high = to_number(min_salary), to_number(max_salary)
    if low is None and high is None:
        return None, None, "missing"
    if (currency or "").upper() != "USD":
        return None, None, "non_usd"
    factor = PERIOD_TO_ANNUAL.get((period or "annual").strip().lower())
    if factor is None:
        return None, None, "unknown_period"
    low = low if low is not None else high
    high = high if high is not None else low
    if low > high:
        low, high = high, low
    low, high = low * factor, high * factor
    if low < MIN_ANNUAL_SALARY or high > MAX_ANNUAL_SALARY:
        return None, None, "out_of_range"
    return low, high, "ok"


def _epoch_to_datetime(value: Any) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(int(value), unit="s", tz="UTC")
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def transform_job(raw: dict) -> dict:
    """Map one raw Himalayas job onto our flat schema (no filtering here)."""
    description = clean_html(raw.get("description"))
    excerpt = clean_html(raw.get("excerpt"))
    title = (raw.get("title") or "").strip()
    seniority_levels = [str(s) for s in _as_list(raw.get("seniority"))]
    parents = [str(c) for c in _as_list(raw.get("parentCategories"))]
    locations = [str(c) for c in _as_list(raw.get("locationRestrictions"))]
    salary_low, salary_high, _ = annual_usd_salary(
        raw.get("minSalary"), raw.get("maxSalary"), raw.get("currency"), raw.get("salaryPeriod")
    )
    guid = raw.get("guid") or raw.get("applicationLink") or ""
    return {
        "id": job_id(guid),
        "title": title,
        "company": (raw.get("companyName") or "Unknown").strip(),
        "company_logo": raw.get("companyLogo") or None,
        "excerpt": excerpt,
        "description": description,
        "employment_type": raw.get("employmentType") or "Unknown",
        "seniority": seniority_levels[0] if seniority_levels else "Unknown",
        "seniority_levels": seniority_levels,
        "category": parents[0] if parents else UNCATEGORIZED,
        "parent_categories": parents,
        "categories": [humanise_category(str(c)) for c in _as_list(raw.get("categories"))],
        "locations": locations,
        "is_worldwide": not locations,
        "timezones": _as_list(raw.get("timezoneRestrictions")),
        "currency": raw.get("currency"),
        "salary_period": raw.get("salaryPeriod"),
        "salary_min": to_number(raw.get("minSalary")),
        "salary_max": to_number(raw.get("maxSalary")),
        "salary_usd_min": salary_low,
        "salary_usd_max": salary_high,
        "salary_usd": (salary_low + salary_high) / 2 if salary_low is not None else None,
        "pub_date": _epoch_to_datetime(raw.get("pubDate")),
        "expiry_date": _epoch_to_datetime(raw.get("expiryDate")),
        # Himalayas terms: the application link must be used unchanged.
        "apply_url": raw.get("applicationLink") or "",
        "source_url": raw.get("guid") or "",
        "skills": extract_skills(title, description or excerpt),
    }


def empty_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the clean schema."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in COLUMNS})


def clean_jobs(raw_jobs: list[dict], now: datetime | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean a list of raw Himalayas jobs.

    Rules, in order:

    1. drop rows without a title or an application link,
    2. drop duplicates (same ``guid``),
    3. drop expired jobs (``expiryDate`` before ``now``),
    4. keep the job but blank its USD salary if the salary is non-USD, has an
       unknown period, or is outside 10k-1M per year after conversion.

    Returns the clean DataFrame and a report with a count for every rule.
    """
    now = now or datetime.now(UTC)
    now_ts = pd.Timestamp(now)
    report = {
        "raw_rows": len(raw_jobs),
        "dropped_missing_title_or_link": 0,
        "dropped_duplicates": 0,
        "dropped_expired": 0,
        "salary_missing": 0,
        "salary_non_usd": 0,
        "salary_unknown_period": 0,
        "salary_out_of_range": 0,
        "salary_usable": 0,
        "clean_rows": 0,
    }
    seen: set[str] = set()
    rows = []
    for raw in raw_jobs:
        if not (raw.get("title") or "").strip() or not raw.get("applicationLink"):
            report["dropped_missing_title_or_link"] += 1
            continue
        guid = raw.get("guid") or raw["applicationLink"]
        if guid in seen:
            report["dropped_duplicates"] += 1
            continue
        seen.add(guid)
        expiry = _epoch_to_datetime(raw.get("expiryDate"))
        if expiry is not None and expiry < now_ts:
            report["dropped_expired"] += 1
            continue
        status = annual_usd_salary(
            raw.get("minSalary"), raw.get("maxSalary"), raw.get("currency"), raw.get("salaryPeriod")
        )[2]
        key = "salary_usable" if status == "ok" else f"salary_{status}"
        report[key] += 1
        rows.append(transform_job(raw))

    frame = _coerce_types(pd.DataFrame(rows, columns=COLUMNS)) if rows else empty_frame()
    report["clean_rows"] = len(frame)
    return frame, report


def drop_expired(frame: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Remove rows whose ``expiry_date`` is in the past (used for stored data)."""
    if frame.empty:
        return frame
    now_ts = pd.Timestamp(now or datetime.now(UTC))
    expiry = pd.to_datetime(frame["expiry_date"], utc=True)
    return frame[expiry.isna() | (expiry >= now_ts)].reset_index(drop=True)


def frame_to_records(frame: pd.DataFrame, columns: list[str] | None = None) -> list[dict]:
    """Serialise a clean frame to JSON-safe dicts (NaN becomes None, dates become ISO strings)."""
    subset = frame[columns] if columns else frame
    return json.loads(subset.to_json(orient="records", date_format="iso"))


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Inverse of :func:`frame_to_records`, used by the API client and the database layer."""
    if not records:
        return empty_frame()
    frame = pd.DataFrame.from_records(records)
    for column in COLUMNS:
        if column not in frame:
            frame[column] = None
    frame = _coerce_types(frame)
    frame["is_worldwide"] = frame["locations"].apply(lambda v: not v)
    return frame[COLUMNS + [c for c in frame.columns if c not in COLUMNS]]


def _coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Give numeric, date and list columns consistent dtypes."""
    for column in DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    for column in LIST_COLUMNS:
        frame[column] = frame[column].apply(lambda v: list(v) if isinstance(v, (list, tuple)) else [])
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    frame["is_worldwide"] = frame["is_worldwide"].astype(bool)
    return frame
