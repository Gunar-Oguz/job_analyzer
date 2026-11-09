"""Pure pandas functions that answer questions about a clean jobs DataFrame.

Nothing here knows about Streamlit, FastAPI or the database, so the same
functions power the standalone app, the API and the tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

import pandas as pd

WORLDWIDE = "Worldwide"


def overview(df: pd.DataFrame) -> dict:
    """Headline numbers for the market overview."""
    total = len(df)
    if total == 0:
        return {"jobs": 0, "companies": 0, "pct_with_salary": 0.0, "pct_worldwide": 0.0, "median_salary_usd": None}
    salaries = df["salary_usd"].dropna()
    return {
        "jobs": total,
        "companies": int(df["company"].nunique()),
        "pct_with_salary": round(100 * len(salaries) / total, 1),
        "pct_worldwide": round(100 * float(df["is_worldwide"].mean()), 1),
        "median_salary_usd": float(salaries.median()) if len(salaries) else None,
    }


def value_counts(df: pd.DataFrame, column: str, n: int = 10) -> pd.DataFrame:
    """Count values of a scalar or list column. Returns columns ``[column, "jobs", "share"]``."""
    if df.empty:
        return pd.DataFrame(columns=[column, "jobs", "share"])
    series = df[column].explode().dropna() if df[column].map(type).eq(list).any() else df[column].dropna()
    counts = series.value_counts().head(n).rename_axis(column).reset_index(name="jobs")
    counts["share"] = (counts["jobs"] / len(df)).round(4)
    return counts


def skill_demand(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Most requested skills: number and share of jobs that mention each skill."""
    return value_counts(df, "skills", n).rename(columns={"skills": "skill"})


def location_options(df: pd.DataFrame) -> list[str]:
    """All countries that appear in location restrictions, plus "Worldwide"."""
    countries = sorted({c for locations in df["locations"] for c in locations}) if not df.empty else []
    return [WORLDWIDE] + countries


def posts_per_day(df: pd.DataFrame, days: int = 30, now: datetime | None = None) -> pd.DataFrame:
    """Number of jobs published per day over the last ``days`` days (missing days are 0)."""
    end = pd.Timestamp(now or datetime.now(UTC)).tz_convert("UTC").normalize()
    index = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    if df.empty:
        return pd.DataFrame({"date": index, "jobs": 0})
    dates = pd.to_datetime(df["pub_date"], utc=True).dt.normalize()
    counts = dates.value_counts().reindex(index, fill_value=0)
    return pd.DataFrame({"date": index, "jobs": counts.to_numpy()})


def posting_timeline(df: pd.DataFrame, days: int = 30, now: datetime | None = None) -> tuple[pd.DataFrame, str]:
    """Jobs published over time, per day, or per hour when the sample spans less than 3 days.

    The Himalayas feed is sorted newest first, so a sample of the newest
    ~1,500 jobs may cover only a day or two. Returns ``(frame, "day" | "hour")``.
    """
    if df.empty or df["pub_date"].isna().all():
        return posts_per_day(df, days, now), "day"
    dates = pd.to_datetime(df["pub_date"], utc=True).dropna()
    if dates.max() - dates.min() >= pd.Timedelta(days=3):
        return posts_per_day(df, days, now), "day"
    hours = dates.dt.floor("h")
    index = pd.date_range(hours.min(), hours.max(), freq="h")
    counts = hours.value_counts().reindex(index, fill_value=0)
    return pd.DataFrame({"date": index, "jobs": counts.to_numpy()}), "hour"


def filter_jobs(
    df: pd.DataFrame,
    query: str = "",
    categories: Iterable[str] = (),
    seniorities: Iterable[str] = (),
    employment_types: Iterable[str] = (),
    location: str | None = None,
    min_salary: float | None = None,
    has_salary: bool = False,
) -> pd.DataFrame:
    """Filter jobs the way the search tab does. Empty filters keep everything.

    ``query`` matches title, company, excerpt or skills (case-insensitive,
    every word must appear). ``location`` keeps jobs open in that country,
    including worldwide jobs; "Worldwide" keeps only worldwide jobs.
    """
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if query:
        haystack = (
            df["title"].fillna("")
            + " "
            + df["company"].fillna("")
            + " "
            + df["excerpt"].fillna("")
            + " "
            + df["skills"].apply(" ".join)
        ).str.lower()
        for word in query.lower().split():
            mask &= haystack.str.contains(word, regex=False)
    categories, seniorities, employment_types = set(categories), set(seniorities), set(employment_types)
    if categories:
        mask &= df["parent_categories"].apply(lambda cats: bool(categories & set(cats))) | df["category"].isin(
            categories
        )
    if seniorities:
        mask &= df["seniority_levels"].apply(lambda levels: bool(seniorities & set(levels)))
    if employment_types:
        mask &= df["employment_type"].isin(employment_types)
    if location == WORLDWIDE:
        mask &= df["is_worldwide"]
    elif location:
        mask &= df["is_worldwide"] | df["locations"].apply(lambda locs: location in locs)
    if has_salary or min_salary:
        mask &= df["salary_usd"].notna()
    if min_salary:
        mask &= df["salary_usd_max"] >= min_salary
    return df[mask].sort_values("pub_date", ascending=False)


def salary_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Only rows with a usable annual USD salary."""
    return df[df["salary_usd"].notna()] if not df.empty else df


def salary_summary(df: pd.DataFrame, by: str, min_count: int = 3) -> pd.DataFrame:
    """Median, 25th and 75th percentile of annual USD salary per group."""
    salaries = salary_frame(df)
    columns = [by, "jobs", "p25", "median", "p75"]
    if salaries.empty:
        return pd.DataFrame(columns=columns)
    grouped = salaries.groupby(by)["salary_usd"]
    summary = pd.DataFrame(
        {
            "jobs": grouped.size(),
            "p25": grouped.quantile(0.25),
            "median": grouped.median(),
            "p75": grouped.quantile(0.75),
        }
    ).reset_index()
    summary = summary[summary["jobs"] >= min_count]
    return summary.sort_values("median", ascending=False).reset_index(drop=True)[columns]


def salary_overall(df: pd.DataFrame) -> dict:
    """p25 / median / p75 over all usable salaries."""
    salaries = salary_frame(df)["salary_usd"] if not df.empty else pd.Series(dtype=float)
    if salaries.empty:
        return {"jobs": 0, "p25": None, "median": None, "p75": None}
    return {
        "jobs": int(len(salaries)),
        "p25": float(salaries.quantile(0.25)),
        "median": float(salaries.median()),
        "p75": float(salaries.quantile(0.75)),
    }


def top_paying_skills(df: pd.DataFrame, min_count: int = 5, n: int = 15) -> pd.DataFrame:
    """Median salary of jobs that mention each skill (skills with fewer than ``min_count`` jobs are hidden)."""
    salaries = salary_frame(df)
    if salaries.empty:
        return pd.DataFrame(columns=["skill", "jobs", "median"])
    exploded = salaries[["skills", "salary_usd"]].explode("skills").dropna(subset=["skills"])
    grouped = exploded.groupby("skills")["salary_usd"].agg(jobs="size", median="median").reset_index()
    grouped = grouped.rename(columns={"skills": "skill"})
    grouped = grouped[grouped["jobs"] >= min_count]
    return grouped.sort_values("median", ascending=False).head(n).reset_index(drop=True)


def jobs_in_category(df: pd.DataFrame, category: str | None) -> pd.DataFrame:
    """Jobs whose parent categories include ``category`` (all jobs if None)."""
    if not category or df.empty:
        return df
    return df[df["parent_categories"].apply(lambda cats: category in cats) | (df["category"] == category)]


def skill_gap(df: pd.DataFrame, user_skills: Iterable[str], category: str | None, top_n: int = 15) -> dict:
    """Compare a user's skills with the most demanded skills of a category.

    ``match_score`` is demand-weighted coverage: the share of the top-N
    skills' total demand that the user already covers. A skill wanted by 60%
    of jobs counts more than one wanted by 5%.
    """
    user = set(user_skills)
    jobs = jobs_in_category(df, category)
    demand = skill_demand(jobs, top_n)
    if demand.empty:
        return {"jobs": len(jobs), "match_score": 0.0, "skills": demand.assign(have=pd.Series(dtype=bool))}
    demand["have"] = demand["skill"].isin(user)
    score = demand.loc[demand["have"], "share"].sum() / demand["share"].sum()
    return {"jobs": len(jobs), "match_score": round(100 * float(score), 1), "skills": demand}


def best_matching_jobs(df: pd.DataFrame, user_skills: Iterable[str], category: str | None = None, n: int = 10):
    """Rank live jobs by how much of their skill list the user already has.

    ``jaccard`` = |user ∩ job| / |user ∪ job| and ``coverage`` = |user ∩ job| /
    |job skills|. We sort by Jaccard first (coverage alone would favour jobs
    that list a single skill), then coverage, then the number of shared
    skills. Jobs that share no skill with the user are left out.
    """
    user = set(user_skills)
    jobs = jobs_in_category(df, category)
    jobs = jobs[jobs["skills"].apply(len) > 0] if not jobs.empty else jobs
    if jobs.empty or not user:
        return jobs.head(0).assign(coverage=[], jaccard=[], matched=[], missing=[])
    matched = jobs["skills"].apply(lambda skills: [s for s in skills if s in user])
    missing = jobs["skills"].apply(lambda skills: [s for s in skills if s not in user])
    sizes = jobs["skills"].apply(len)
    shared = matched.apply(len)
    union = jobs["skills"].apply(lambda skills: len(user | set(skills)))
    ranked = jobs.assign(
        matched=matched,
        missing=missing,
        shared=shared,
        coverage=(shared / sizes).round(3),
        jaccard=(shared / union).round(3),
    )
    ranked = ranked[ranked["shared"] > 0]
    return ranked.sort_values(["jaccard", "coverage", "shared"], ascending=False).head(n)
