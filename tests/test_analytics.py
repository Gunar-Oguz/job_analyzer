import pandas as pd

from core import analytics
from tests.conftest import NOW_DT


def test_overview(jobs):
    stats = analytics.overview(jobs)
    assert stats["jobs"] == len(jobs)
    assert stats["companies"] == 40
    assert 60 <= stats["pct_with_salary"] <= 70
    assert stats["pct_worldwide"] == 20.0
    assert analytics.overview(jobs.head(0))["jobs"] == 0


def test_value_counts_scalar_and_list_columns(jobs):
    categories = analytics.value_counts(jobs, "category")
    assert categories.iloc[0]["category"] == "Developer"
    assert categories["jobs"].sum() == len(jobs)
    skills = analytics.skill_demand(jobs, 5)
    assert list(skills.columns) == ["skill", "jobs", "share"]
    assert skills["jobs"].is_monotonic_decreasing


def test_posts_per_day_fills_missing_days(jobs):
    per_day = analytics.posts_per_day(jobs, days=30, now=NOW_DT)
    assert len(per_day) == 30
    assert per_day["jobs"].sum() == len(jobs)  # synthetic jobs are all within the last 21 days
    frame, unit = analytics.posting_timeline(jobs, now=NOW_DT)
    assert unit == "day" and len(frame) == 30


def test_posting_timeline_uses_hours_for_short_samples(jobs):
    recent = jobs.assign(pub_date=pd.Timestamp(NOW_DT) - pd.to_timedelta(jobs.index % 10, unit="h"))
    frame, unit = analytics.posting_timeline(recent, now=NOW_DT)
    assert unit == "hour"
    assert frame["jobs"].sum() == len(jobs)


def test_filter_jobs(jobs):
    assert len(analytics.filter_jobs(jobs)) == len(jobs)
    found = analytics.filter_jobs(jobs, query="data scientist")
    assert len(found) and found["title"].str.contains("Data Scientist").all()
    senior_ds = analytics.filter_jobs(jobs, categories=["Data Science"], seniorities=["Senior"])
    assert set(senior_ds["category"]) == {"Data Science"} and set(senior_ds["seniority"]) == {"Senior"}
    germany = analytics.filter_jobs(jobs, location="Germany")
    assert all(row.is_worldwide or "Germany" in row.locations for row in germany.itertuples())
    worldwide = analytics.filter_jobs(jobs, location=analytics.WORLDWIDE)
    assert worldwide["is_worldwide"].all()
    rich = analytics.filter_jobs(jobs, min_salary=150_000)
    assert (rich["salary_usd_max"] >= 150_000).all()
    assert analytics.filter_jobs(jobs, has_salary=True)["salary_usd"].notna().all()
    assert analytics.filter_jobs(jobs, employment_types=["Part Time"])["employment_type"].eq("Part Time").all()


def test_salary_summary_and_overall(jobs):
    summary = analytics.salary_summary(jobs, "seniority")
    assert summary["median"].is_monotonic_decreasing and summary["jobs"].ge(3).all()
    assert (summary["p25"] <= summary["median"]).all() and (summary["median"] <= summary["p75"]).all()
    overall = analytics.salary_overall(jobs)
    assert overall["p25"] < overall["median"] < overall["p75"]


def test_top_paying_skills_respects_min_count(jobs):
    paying = analytics.top_paying_skills(jobs, min_count=10)
    assert (paying["jobs"] >= 10).all()
    assert paying["median"].is_monotonic_decreasing
    assert analytics.top_paying_skills(jobs, min_count=10_000).empty


def test_skill_gap(jobs):
    gap = analytics.skill_gap(jobs, ["Python", "SQL"], "Data Science", top_n=6)
    skills = gap["skills"]
    assert skills["share"].is_monotonic_decreasing  # ordered by demand
    assert set(skills.loc[skills["have"], "skill"]) == {"Python", "SQL"}
    assert 0 < gap["match_score"] < 100
    assert analytics.skill_gap(jobs, [], "Data Science")["match_score"] == 0
    all_skills = skills["skill"].tolist()
    assert analytics.skill_gap(jobs, all_skills, "Data Science", top_n=6)["match_score"] == 100


def test_best_matching_jobs(jobs):
    best = analytics.best_matching_jobs(jobs, ["Kubernetes", "Terraform", "AWS", "Linux", "CI/CD"], n=5)
    assert len(best) == 5
    assert best["title"].str.contains("DevOps").all()
    assert best["jaccard"].is_monotonic_decreasing
    assert analytics.best_matching_jobs(jobs, []).empty


def test_location_options(jobs):
    options = analytics.location_options(jobs)
    assert options[0] == analytics.WORLDWIDE and "Germany" in options
