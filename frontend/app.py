"""Streamlit UI for Job Market Analyzer.

Two modes:

* **Standalone** (default, used on Streamlit Community Cloud): fetch live
  jobs from Himalayas, clean them and train the models inside this process.
* **API** (when ``BACKEND_URL`` is set, e.g. in docker compose): read jobs
  and model predictions from the FastAPI backend.
"""

from __future__ import annotations

import html
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # so "core" imports work however the app is started
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from core import analytics, ml, pipeline  # noqa: E402
from core.etl import UNCATEGORIZED  # noqa: E402
from core.himalayas import SOURCE_URL, FetchError  # noqa: E402
from core.skills import ALL_SKILLS, parse_user_skills  # noqa: E402
from frontend import api_client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("job_analyzer.app")
REPO_URL = "https://github.com/Gunar-Oguz/job_analyzer"
CACHE_TTL_SECONDS = 12 * 60 * 60
STANDALONE_PAGES = int(os.getenv("HIMALAYAS_PAGES", "75"))
SENIORITY_ORDER = ["Entry-level", "Mid-level", "Senior", "Manager", "Director", "Executive"]
ACCENT = "#2563eb"
PLOT_TEMPLATE = "plotly_white"
CARDS_PER_PAGE = 15


def backend_url() -> str | None:
    url = os.getenv("BACKEND_URL", "").strip()
    return url.rstrip("/") or None


# ------------------------------------------------------------------ data layer


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_standalone_data(max_pages: int) -> tuple[pd.DataFrame, dict]:
    return pipeline.load_live_jobs(max_pages=max_pages)


@st.cache_resource(show_spinner=False, max_entries=2)
def train_standalone_models(fetched_at: str, max_pages: int) -> ml.Models:
    """Train once per dataset; ``fetched_at`` is the cache key."""
    jobs, _ = load_standalone_data(max_pages)
    models = ml.train_all(jobs)
    for name, model in (("salary", models.salary), ("category", models.category)):
        if model is not None:
            logger.info("Trained %s model: %s", name, model.metrics)
    return models


@st.cache_data(ttl=300, show_spinner=False)
def load_api_data(base_url: str) -> tuple[pd.DataFrame, dict]:
    return api_client.load_jobs(base_url)


def load_everything() -> tuple[pd.DataFrame, dict, ml.Models, str]:
    """Return ``(jobs, report, models, mode)`` or stop the script with a friendly message."""
    base_url = backend_url()
    if base_url:
        try:
            jobs, report = load_api_data(base_url)
            models = api_client.load_models(base_url)
        except api_client.BackendError as error:
            st.error(f"The backend at `{base_url}` is not reachable yet. Please try again in a moment.\n\n{error}")
            st.stop()
        return jobs, report, models, "api"

    try:
        with st.spinner(
            f"Downloading about {STANDALONE_PAGES * 20:,} of the newest remote jobs from Himalayas. "
            "The first visit takes about a minute; after that the data is cached for 12 hours."
        ):
            jobs, report = load_standalone_data(STANDALONE_PAGES)
        with st.spinner("Training the salary and category models on today's data..."):
            models = train_standalone_models(report["fetched_at"], STANDALONE_PAGES)
    except FetchError as error:
        st.error(
            "We could not reach the Himalayas jobs API right now, so there is no live data to show. "
            "This is usually temporary. Please try again in a few minutes."
        )
        st.caption(f"Details: {error}")
        if st.button("Try again"):
            load_standalone_data.clear()
            st.rerun()
        st.stop()
    return jobs, report, models, "standalone"


# --------------------------------------------------------------------- helpers


def money(value: float | None) -> str:
    return "n/a" if value is None or pd.isna(value) else f"${value:,.0f}"


def salary_range(job: pd.Series) -> str:
    if pd.notna(job.get("salary_usd_min")):
        low, high = job["salary_usd_min"], job["salary_usd_max"]
        return money(low) if low == high else f"{money(low)} - {money(high)} / year"
    if pd.notna(job.get("salary_min")) and job.get("currency"):
        period = job.get("salary_period") or ""
        return f"{job['currency']} {job['salary_min']:,.0f} - {job['salary_max']:,.0f} ({period})"
    return "Salary not listed"


def location_text(locations: list[str]) -> str:
    if not locations:
        return "Worldwide"
    if len(locations) > 4:
        return ", ".join(locations[:4]) + f" +{len(locations) - 4} more"
    return ", ".join(locations)


def posted_text(when) -> str:
    if when is None or pd.isna(when):
        return ""
    hours = (datetime.now(UTC) - when.to_pydatetime()).total_seconds() / 3600
    if hours < 1:
        return "just now"
    if hours < 48:
        return f"{hours:.0f}h ago"
    return f"{hours / 24:.0f}d ago"


def badges(skills: list[str], color: str = "blue") -> str:
    return " ".join(f":{color}-badge[{skill}]" for skill in skills)


def bar(frame: pd.DataFrame, x: str, y: str, title: str, horizontal: bool = True, fmt: str | None = None):
    fig = px.bar(
        frame,
        x=x if not horizontal else y,
        y=y if not horizontal else x,
        orientation="h" if horizontal else "v",
        title=title,
        template=PLOT_TEMPLATE,
        color_discrete_sequence=[ACCENT],
        text_auto=fmt or True,
    )
    fig.update_traces(textangle=0, textposition="outside", cliponaxis=False)
    if horizontal:
        fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=max(320, 28 * len(frame) + 80))
    return fig


def source_note() -> None:
    st.caption(f"Source: [Himalayas]({SOURCE_URL}) remote jobs API. Apply links go to the original posting.")


# ------------------------------------------------------------------------ tabs


def tab_overview(jobs: pd.DataFrame) -> None:
    stats = analytics.overview(jobs)
    cols = st.columns(5)
    cols[0].metric("Live jobs", f"{stats['jobs']:,}", border=True)
    cols[1].metric("Companies hiring", f"{stats['companies']:,}", border=True)
    cols[2].metric("Jobs with salary", f"{stats['pct_with_salary']}%", border=True)
    cols[3].metric("Open worldwide", f"{stats['pct_worldwide']}%", border=True, help="No country restriction at all.")
    cols[4].metric("Median salary (USD)", money(stats["median_salary_usd"]), border=True)

    left, right = st.columns(2)
    categories = analytics.value_counts(jobs, "category", 13)
    categories = categories[categories["category"] != UNCATEGORIZED].head(12)
    left.plotly_chart(bar(categories, "category", "jobs", "Top categories"))
    uncategorized = int((jobs["category"] == UNCATEGORIZED).sum())
    left.caption(f"{uncategorized:,} jobs have no Himalayas category and are not shown.")
    right.plotly_chart(bar(analytics.skill_demand(jobs, 15), "skill", "jobs", "Most requested skills"))

    left, right = st.columns(2)
    seniority = analytics.value_counts(jobs, "seniority")
    fig = px.pie(seniority, names="seniority", values="jobs", hole=0.55, title="Seniority mix", template=PLOT_TEMPLATE)
    fig.update_traces(textinfo="percent+label", showlegend=False)
    left.plotly_chart(fig)
    right.plotly_chart(
        bar(analytics.value_counts(jobs, "employment_type"), "employment_type", "jobs", "Employment type")
    )

    timeline, unit = analytics.posting_timeline(jobs)
    title = "Jobs posted per day (last 30 days)" if unit == "day" else "Jobs posted per hour"
    fig = px.bar(timeline, x="date", y="jobs", title=title, template=PLOT_TEMPLATE, color_discrete_sequence=[ACCENT])
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title=None)
    st.plotly_chart(fig)
    if unit == "hour":
        st.caption(
            "The Himalayas feed is sorted newest first, so this sample covers only the last hours or days. "
            "That is why the chart is per hour."
        )


def tab_search(jobs: pd.DataFrame) -> None:
    query = st.text_input("Keyword", placeholder="e.g. data analyst, python, Stripe", key="search_q")
    c1, c2, c3 = st.columns(3)
    categories = c1.multiselect("Category", analytics.value_counts(jobs, "category", 100)["category"].tolist())
    present = set(jobs["seniority"])
    seniorities = c2.multiselect("Seniority", [s for s in SENIORITY_ORDER if s in present])
    employment = c3.multiselect("Employment type", analytics.value_counts(jobs, "employment_type")["employment_type"])
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    location = c1.selectbox(
        "Open to applicants in", ["Any location"] + analytics.location_options(jobs), help="Includes worldwide jobs."
    )
    min_salary = c2.slider("Minimum salary (USD / year)", 0, 300_000, 0, step=10_000)
    has_salary = c3.checkbox("Has salary")
    as_table = c4.toggle("Table view")

    found = analytics.filter_jobs(
        jobs,
        query,
        categories,
        seniorities,
        employment,
        None if location == "Any location" else location,
        min_salary or None,
        has_salary,
    )
    st.markdown(f"**{len(found):,}** matching jobs")
    source_note()
    if found.empty:
        st.info("No job matches these filters. Try removing one.")
        return

    if as_table:
        table = found.assign(
            salary=found.apply(salary_range, axis=1),
            location=found["locations"].apply(location_text),
            skills_list=found["skills"].apply(", ".join),
        )
        st.dataframe(
            table[["title", "company", "seniority", "salary", "location", "pub_date", "skills_list", "apply_url"]],
            hide_index=True,
            column_config={
                "apply_url": st.column_config.LinkColumn("Apply", display_text="Apply"),
                "pub_date": st.column_config.DatetimeColumn("Posted", format="MMM D, HH:mm"),
                "skills_list": "Skills",
            },
        )
        return

    pages = max(1, -(-len(found) // CARDS_PER_PAGE))
    page = st.number_input("Page", 1, pages, 1, key="search_page") if pages > 1 else 1
    for _, job in found.iloc[(page - 1) * CARDS_PER_PAGE : page * CARDS_PER_PAGE].iterrows():
        job_card(job)
    st.caption(f"Page {page} of {pages}")


def job_card(job: pd.Series, extra: str = "") -> None:
    with st.container(border=True):
        logo, body, actions = st.columns([1, 8, 2], vertical_alignment="center")
        if job.get("company_logo"):
            logo.image(job["company_logo"], width=48)
        body.markdown(f"**{html.escape(job['title'])}**  \n{html.escape(job['company'])}")
        details = [job["seniority"], job["employment_type"], location_text(job["locations"]), salary_range(job)]
        body.caption(" · ".join(html.escape(str(d)) for d in details) + f" · {posted_text(job['pub_date'])}")
        if job["skills"]:
            body.markdown(badges(job["skills"][:8]))
        if extra:
            body.markdown(extra)
        actions.link_button("Apply", job["apply_url"], type="primary", width="stretch")
        actions.markdown(f"[via Himalayas]({job['source_url'] or SOURCE_URL})")


def tab_salary(jobs: pd.DataFrame) -> None:
    salaries = analytics.salary_frame(jobs)
    overall = analytics.salary_overall(jobs)
    st.caption(
        "Only jobs that publish a salary in US dollars are used. Hourly pay is multiplied by 2,080 hours and "
        "monthly pay by 12; values below $10k or above $1M per year are dropped as impossible."
    )
    if overall["jobs"] < 5:
        st.info("Not enough jobs with a USD salary in this sample to show salary statistics.")
        return
    cols = st.columns(4)
    cols[0].metric("Jobs with USD salary", f"{overall['jobs']:,}", border=True)
    cols[1].metric("25th percentile", money(overall["p25"]), border=True)
    cols[2].metric("Median", money(overall["median"]), border=True)
    cols[3].metric("75th percentile", money(overall["p75"]), border=True)

    present = [s for s in SENIORITY_ORDER if s in set(salaries["seniority"])]
    fig = px.box(
        salaries,
        x="seniority",
        y="salary_usd",
        category_orders={"seniority": present},
        points="outliers",
        title="Annual salary by seniority (USD)",
        template=PLOT_TEMPLATE,
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(yaxis_title="USD / year", xaxis_title=None)
    st.plotly_chart(fig)

    by_category = analytics.salary_summary(jobs, "category", min_count=5)
    top = by_category["category"].head(10).tolist()
    fig = px.box(
        salaries[salaries["category"].isin(top)],
        y="category",
        x="salary_usd",
        orientation="h",
        category_orders={"category": top},
        title="Annual salary by category (categories with 5+ salaries)",
        template=PLOT_TEMPLATE,
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(xaxis_title="USD / year", yaxis_title=None, height=max(350, 40 * len(top) + 100))
    st.plotly_chart(fig)

    left, right = st.columns(2)
    fig = px.histogram(
        salaries, x="salary_usd", nbins=40, title="Salary distribution", template=PLOT_TEMPLATE,
        color_discrete_sequence=[ACCENT],
    )  # fmt: skip
    fig.update_layout(xaxis_title="USD / year", yaxis_title="Jobs")
    left.plotly_chart(fig)
    min_count = right.slider("Minimum jobs per skill", 3, 30, 5, key="pay_min_count")
    paying = analytics.top_paying_skills(jobs, min_count=min_count, n=15)
    if paying.empty:
        right.info("No skill has enough salaried jobs. Lower the minimum.")
    else:
        right.plotly_chart(bar(paying, "skill", "median", "Top-paying skills (median salary)", fmt="$.3s"))

    st.dataframe(
        analytics.salary_summary(jobs, "seniority", min_count=1),
        hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="$%d") for c in ("p25", "median", "p75")},
    )


def tab_predictor(jobs: pd.DataFrame, models: ml.Models) -> None:
    model = models.salary
    if model is None:
        st.info(f"The salary model is not available: {models.problems.get('salary', 'no data')}")
        return
    m = model.metrics
    st.markdown(
        "A Ridge regression on **log salary** with TF-IDF title words, one-hot seniority / category / location / "
        f"employment type and skill flags. Trained just now on **{m['rows']:,}** jobs with a USD salary."
    )
    with st.form("salary_form"):
        title = st.text_input("Job title", "Senior Data Scientist")
        c1, c2 = st.columns(2)
        seniority = _select(c1, "Seniority", model.options["seniority"], "Senior")
        category = _select(c2, "Category", model.options["category"], "Data Science")
        location = _select(c1, "Location", model.options["location"], "United States")
        employment = _select(c2, "Employment type", model.options["employment_type"], "Full Time")
        skills = st.multiselect("Skills", ALL_SKILLS, default=["Python", "SQL", "Machine Learning"])
        st.form_submit_button("Predict salary", type="primary")
    result = model.predict(title, seniority, category, location, skills, employment)
    cols = st.columns(3)
    cols[0].metric("Predicted salary", money(result["predicted"]), border=True)
    cols[1].metric("Typical range (low)", money(result["low"]), border=True)
    cols[2].metric("Typical range (high)", money(result["high"]), border=True)
    st.caption("The range covers the 10th to 90th percentile of the model's errors on held-out jobs.")

    st.subheader("How good is the model?")
    cols = st.columns(5)
    cols[0].metric("Test MAE", money(m["mae_usd"]), help=f"Held-out test split of {m['test_rows']} jobs.")
    cols[1].metric("Baseline MAE", money(m["baseline_mae_usd"]), help="Always predicting the training median.")
    cols[2].metric("Test R² (USD)", f"{m['r2']:.2f}")
    cols[3].metric("Test R² (log)", f"{m['r2_log']:.2f}")
    cols[4].metric("5-fold CV MAE", money(m["cv_mae_usd"]), help=f"Cross-validated R² (USD): {m['cv_r2']:.2f}")
    st.caption(
        f"Train / test rows: {m['train_rows']} / {m['test_rows']}. Salary depends on many things a posting does "
        "not say (company, equity, exact city), so treat the number as a rough guide. "
        "The metrics change every day because the model is retrained on fresh data."
    )


def _index(options: list[str], wanted: str) -> int:
    return options.index(wanted) if wanted in options else 0


def _select(container, label: str, options: list[str], default: str) -> str:
    return container.selectbox(label, options, index=_index(options, default))


def tab_classifier(models: ml.Models) -> None:
    model = models.category
    if model is None:
        st.info(f"The category model is not available: {models.problems.get('category', 'no data')}")
        return
    m = model.metrics
    st.markdown(
        "A logistic regression on TF-IDF features of the title and description predicts the main Himalayas "
        f"category. It knows **{m['classes']}** categories and was trained on **{m['rows']:,}** jobs."
    )
    title = st.text_input("Job title", "Senior Backend Engineer (Go, Kubernetes)", key="clf_title")
    description = st.text_area("Description (optional)", "", height=100, key="clf_desc")
    if title.strip():
        predictions = model.predict(title, description)
        for p in predictions:
            st.progress(p["probability"], text=f"{p['category']}: {p['probability']:.0%}")
    st.subheader("How good is the model?")
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{m['accuracy']:.0%}", help=f"Held-out test split of {m['test_rows']} jobs.")
    cols[1].metric("Top-3 accuracy", f"{m['top3_accuracy']:.0%}")
    cols[2].metric("Macro F1", f"{m['macro_f1']:.2f}")
    cols[3].metric(
        "Baseline accuracy", f"{m['baseline_accuracy']:.0%}", help="Always guessing the most common category."
    )
    cols[4].metric(
        "Any-label accuracy", f"{m['any_label_accuracy']:.0%}", help="Prediction is one of the job's categories."
    )
    st.caption(
        "Many jobs belong to more than one category (for example Sales and Customer Service), "
        "so the strict accuracy is a lower bound."
    )


def tab_fit(jobs: pd.DataFrame) -> None:
    st.markdown("Paste your skills or a short resume, pick a target category, and compare yourself with live demand.")
    text = st.text_area(
        "Your skills or resume", "Python, SQL, pandas, Tableau, statistics, A/B testing, Excel", height=110
    )
    c1, c2 = st.columns([2, 1])
    options = ["All categories"] + analytics.value_counts(jobs, "category", 100)["category"].tolist()
    target = c1.selectbox("Target category", options, index=_index(options, "Data Science"))
    top_n = c2.slider("Compare with the top N skills", 5, 30, 15)
    category = None if target == "All categories" else target

    user_skills = parse_user_skills(text)
    st.markdown("**Skills we recognised:** " + (badges(user_skills, "green") if user_skills else "none yet"))
    gap = analytics.skill_gap(jobs, user_skills, category, top_n)
    if gap["skills"].empty:
        st.info("No jobs with recognised skills in this category.")
        return
    skills = gap["skills"]
    cols = st.columns(3)
    cols[0].metric(
        "Market match", f"{gap['match_score']:.0f}%", border=True, help="Demand-weighted share of top skills you have."
    )
    cols[1].metric("Top skills you have", f"{int(skills['have'].sum())} / {len(skills)}", border=True)
    cols[2].metric("Jobs in scope", f"{gap['jobs']:,}", border=True)
    missing = skills.loc[~skills["have"], "skill"].head(3).tolist()
    if missing:
        st.success("Learn next (most demanded skills you are missing): " + ", ".join(missing))

    chart = skills.assign(status=skills["have"].map({True: "You have it", False: "Missing"}))
    fig = px.bar(
        chart,
        x="share",
        y="skill",
        color="status",
        orientation="h",
        title="Most demanded skills in this category (share of jobs)",
        template=PLOT_TEMPLATE,
        color_discrete_map={"You have it": "#16a34a", "Missing": "#cbd5e1"},
        category_orders={"skill": chart["skill"].tolist()},
        text_auto=".0%",
    )
    fig.update_yaxes(title=None)
    fig.update_xaxes(tickformat=".0%", title=None)
    fig.update_layout(height=max(350, 26 * len(chart) + 100), legend_title=None)
    st.plotly_chart(fig)

    st.subheader("Best matching live jobs")
    best = analytics.best_matching_jobs(jobs, user_skills, category, 8)
    if best.empty:
        st.info("No live job shares a skill with you in this category yet.")
    for _, job in best.iterrows():
        extra = f"Skill match **{job['jaccard']:.0%}** (Jaccard)."
        if job["missing"]:
            extra += " Missing: " + ", ".join(job["missing"][:6])
        job_card(job, extra)


def tab_about(jobs: pd.DataFrame, report: dict, models: ml.Models, mode: str) -> None:
    fetch = report.get("fetch", {})
    cleaning = report.get("cleaning", {})
    cols = st.columns(4)
    mode_help = "Everything runs in this process." if mode == "standalone" else "FastAPI backend with PostgreSQL."
    cols[0].metric("Mode", "Standalone" if mode == "standalone" else "API", border=True, help=mode_help)
    cols[1].metric("Last fetch (UTC)", _short_time(report.get("fetched_at")), border=True)
    cols[2].metric("Rows in last fetch", f"{fetch.get('jobs', 0):,}", border=True)
    cols[3].metric("Fetch time", f"{fetch.get('elapsed_seconds', 0):.0f}s", border=True)

    st.subheader("Pipeline")
    st.markdown(
        "1. **Fetch** the newest jobs from the public Himalayas API, 20 per request, following `nextCursor`, "
        "with a pause between requests, capped retries and an overall time limit.\n"
        "2. **Clean**: strip HTML, drop rows without a title or apply link, drop duplicates and expired jobs, "
        "convert USD salaries to a yearly amount and drop impossible values.\n"
        "3. **Extract skills** with word-boundary regular expressions over a dictionary of "
        f"{len(ALL_SKILLS)} skills.\n"
        "4. **Analyse** with plain pandas functions and **train** two scikit-learn models on the same data."
    )
    if cleaning:
        rules = pd.DataFrame(
            [
                ("Raw rows from the API", cleaning.get("raw_rows", 0)),
                ("Dropped: missing title or apply link", cleaning.get("dropped_missing_title_or_link", 0)),
                ("Dropped: duplicates", cleaning.get("dropped_duplicates", 0)),
                ("Dropped: expired", cleaning.get("dropped_expired", 0)),
                ("Clean rows", cleaning.get("clean_rows", 0)),
                ("Salary usable (USD, yearly, 10k-1M)", cleaning.get("salary_usable", 0)),
                ("Salary not listed", cleaning.get("salary_missing", 0)),
                ("Salary ignored: not USD", cleaning.get("salary_non_usd", 0)),
                ("Salary ignored: out of range", cleaning.get("salary_out_of_range", 0)),
                ("Salary ignored: unknown period", cleaning.get("salary_unknown_period", 0)),
            ],
            columns=["Step", "Rows"],
        )
        st.dataframe(rules, hide_index=True)
    if fetch:
        st.caption(
            f"Pages: {fetch.get('pages')} · retries: {fetch.get('retries')} · "
            f"stopped because: {fetch.get('stopped_because')}"
        )

    st.subheader("Models")
    left, right = st.columns(2)
    if models.salary:
        left.markdown("**Salary model** (Ridge, log target)")
        left.dataframe(_metrics_table(models.salary.metrics), hide_index=True)
    if models.category:
        right.markdown("**Category model** (logistic regression)")
        right.dataframe(_metrics_table(models.category.metrics), hide_index=True)
    for name, problem in models.problems.items():
        st.warning(f"{name}: {problem}")

    st.subheader("Architecture")
    st.markdown(
        "- `core/`: pure Python (fetch, ETL, skills, analytics, ML). No web framework imports.\n"
        "- `frontend/app.py`: this Streamlit app. Without `BACKEND_URL` it runs everything in-process.\n"
        "- `backend/`: FastAPI service with PostgreSQL storage, used by the Docker Compose stack.\n\n"
        f"Code: [{REPO_URL}]({REPO_URL}) · Data: [Himalayas]({SOURCE_URL}). "
        "Job data is fetched at runtime and never stored in the repository."
    )


def _short_time(iso: str | None) -> str:
    return datetime.fromisoformat(iso).strftime("%b %d, %H:%M") if iso else "n/a"


def _metrics_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame({"metric": list(metrics), "value": [str(v) for v in metrics.values()]})


# ------------------------------------------------------------------------ main


def main() -> None:
    st.set_page_config(page_title="Job Market Analyzer", page_icon="💼", layout="wide")
    st.title("💼 Job Market Analyzer")
    st.markdown(f"Live analytics and ML on remote job postings. Source: [Himalayas]({SOURCE_URL}) · [Code]({REPO_URL})")

    jobs, report, models, mode = load_everything()
    if mode == "api":
        with st.sidebar:
            st.header("Backend")
            st.caption(backend_url())
            if st.button("Refresh data from Himalayas", type="primary"):
                with st.spinner("The backend is fetching new jobs..."):
                    try:
                        api_client.refresh(backend_url())
                    except api_client.BackendError as error:
                        st.error(str(error))
                load_api_data.clear()
                st.rerun()
    if jobs.empty:
        st.warning("There are no jobs yet. In API mode, press 'Refresh data from Himalayas' in the sidebar.")
        st.stop()
    if report.get("fetched_at"):
        st.caption(f"{len(jobs):,} live jobs · data fetched {report['fetched_at'].replace('T', ' ')[:16]} UTC")

    tabs = st.tabs(
        [
            "📊 Market overview",
            "🔎 Job search",
            "💰 Salary insights",
            "🤖 Salary predictor",
            "🏷️ Category classifier",
            "🧭 Am I a fit?",
            "⚙️ About & pipeline",
        ]
    )
    with tabs[0]:
        tab_overview(jobs)
    with tabs[1]:
        tab_search(jobs)
    with tabs[2]:
        tab_salary(jobs)
    with tabs[3]:
        tab_predictor(jobs, models)
    with tabs[4]:
        tab_classifier(models)
    with tabs[5]:
        tab_fit(jobs)
    with tabs[6]:
        tab_about(jobs, report, models, mode)

    st.divider()
    st.caption(
        f"Job data from [Himalayas]({SOURCE_URL}), shown only in this app and refreshed from the live API. "
        f"Built by Gulnar Albushova · [GitHub]({REPO_URL})"
    )


if __name__ == "__main__":
    main()
