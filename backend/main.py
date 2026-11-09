"""FastAPI backend for Job Market Analyzer.

Run locally:      uvicorn backend.main:app --reload
Interactive docs: http://localhost:8000/docs

All logic lives in the ``core`` package; this module only wires it to HTTP
and to a store (PostgreSQL in Docker, memory otherwise).
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from backend.storage import make_store
from core import analytics, ml
from core.etl import empty_frame, frame_to_records
from core.himalayas import SOURCE_URL, FetchError
from core.pipeline import load_live_jobs
from core.skills import ALL_SKILLS, parse_user_skills

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("job_analyzer.api")

# Columns returned by /jobs. The long description is left out to keep responses small.
LIST_FIELDS = [
    "id", "title", "company", "company_logo", "excerpt", "employment_type", "seniority", "seniority_levels",
    "category", "parent_categories", "categories", "locations", "is_worldwide", "timezones", "currency",
    "salary_period", "salary_min", "salary_max", "salary_usd_min", "salary_usd_max", "salary_usd",
    "pub_date", "expiry_date", "apply_url", "source_url", "skills",
]  # fmt: skip


class AppState:
    """The current data and models, swapped atomically after each refresh."""

    def __init__(self, store, loader: Callable[..., tuple[pd.DataFrame, dict]]) -> None:
        self.store = store
        self.loader = loader
        self.jobs: pd.DataFrame = empty_frame()
        self.models = ml.Models(salary=None, category=None, problems={"all": "No data yet. POST /refresh."})
        self.report: dict | None = None
        self.refreshing = threading.Lock()

    def reload(self) -> None:
        """Load jobs from the store and retrain the models."""
        jobs = self.store.load()
        models = ml.train_all(jobs) if len(jobs) else self.models
        self.jobs, self.models = jobs, models
        self.report = self.store.last_refresh()

    def refresh(self, max_pages: int) -> dict:
        """Fetch from Himalayas, save, reload. Only one refresh runs at a time."""
        if not self.refreshing.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A refresh is already running.")
        try:
            frame, report = self.loader(max_pages=max_pages)
            report["saved"] = self.store.save(frame, report)
            self.reload()
            return report
        finally:
            self.refreshing.release()


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


class SalaryRequest(BaseModel):
    title: str = Field(..., min_length=2, examples=["Senior Data Scientist"])
    seniority: str = Field("Mid-level", examples=["Senior"])
    category: str = Field("Developer", examples=["Data Science"])
    location: str = Field("United States", examples=["United States"])
    employment_type: str = Field("Full Time", examples=["Full Time"])
    skills: list[str] = Field(default_factory=list, examples=[["Python", "SQL", "Machine Learning"]])


class ClassifyRequest(BaseModel):
    title: str = Field(..., min_length=2, examples=["Senior Backend Engineer (Go)"])
    description: str = ""


class FitRequest(BaseModel):
    skills_text: str = Field(..., examples=["Python, SQL, pandas, Tableau, A/B testing"])
    category: str | None = Field(None, examples=["Data Science"])
    top_n: int = Field(15, ge=5, le=40)
    jobs: int = Field(10, ge=1, le=50)


def create_app(store=None, loader: Callable[..., tuple[pd.DataFrame, dict]] = load_live_jobs) -> FastAPI:
    """Build the app. Tests pass a memory store and a fake loader."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = AppState(store or make_store(), loader)
        state.store.init()
        state.reload()
        app.state.data = state
        logger.info("Started with %d jobs (%s store)", len(state.jobs), state.store.name)
        if state.jobs.empty and _env_flag("REFRESH_ON_STARTUP"):
            pages = int(os.getenv("REFRESH_PAGES", "50"))
            threading.Thread(target=_background_refresh, args=(state, pages), daemon=True).start()
        yield

    app = FastAPI(
        title="Job Market Analyzer API",
        version="2.0.0",
        description=(
            "Live remote-job market analytics and ML. Job data comes from the public "
            f"[Himalayas]({SOURCE_URL}) jobs API; apply links are passed through unchanged."
        ),
        lifespan=lifespan,
    )

    def state() -> AppState:
        return app.state.data

    @app.get("/", tags=["meta"])
    def root() -> dict:
        return {"name": "Job Market Analyzer API", "docs": "/docs", "source": SOURCE_URL}

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        s = state()
        return {
            "status": "ok",
            "jobs": len(s.jobs),
            "store": s.store.name,
            "refreshing": s.refreshing.locked(),
            "models_ready": {"salary": s.models.salary is not None, "category": s.models.category is not None},
        }

    @app.post("/refresh", tags=["pipeline"])
    def refresh(max_pages: Annotated[int, Query(ge=1, le=100)] = 50) -> dict:
        """Pull the newest jobs from Himalayas (20 per page), clean them, store them and retrain the models."""
        try:
            return state().refresh(max_pages)
        except FetchError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/pipeline", tags=["pipeline"])
    def pipeline() -> dict:
        """Last refresh report: fetch statistics and rows removed by each cleaning rule."""
        s = state()
        return {"store": s.store.name, "jobs": len(s.jobs), "last_refresh": s.report}

    @app.get("/jobs", tags=["jobs"])
    def list_jobs(
        q: str = "",
        category: Annotated[list[str] | None, Query()] = None,
        seniority: Annotated[list[str] | None, Query()] = None,
        employment_type: Annotated[list[str] | None, Query()] = None,
        location: str | None = None,
        min_salary: Annotated[float | None, Query(ge=0)] = None,
        has_salary: bool = False,
        limit: Annotated[int, Query(ge=1, le=5000)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict:
        """Search live jobs. List filters accept several values (`?category=Sales&category=Marketing`)."""
        found = analytics.filter_jobs(
            state().jobs, q, category or (), seniority or (), employment_type or (), location, min_salary, has_salary
        )
        page = found.iloc[offset : offset + limit]
        return {"total": len(found), "count": len(page), "jobs": frame_to_records(page, LIST_FIELDS)}

    @app.get("/jobs/{job_id}", tags=["jobs"])
    def get_job(job_id: str) -> dict:
        """One job, including its cleaned description."""
        jobs = state().jobs
        match = jobs[jobs["id"] == job_id]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return frame_to_records(match)[0]

    @app.get("/stats/overview", tags=["analytics"])
    def stats_overview() -> dict:
        jobs = state().jobs
        return {
            **analytics.overview(jobs),
            "categories": frame_to_records(analytics.value_counts(jobs, "category", 15)),
            "seniority": frame_to_records(analytics.value_counts(jobs, "seniority")),
            "employment_type": frame_to_records(analytics.value_counts(jobs, "employment_type")),
        }

    @app.get("/skills/top", tags=["analytics"])
    def top_skills(n: Annotated[int, Query(ge=1, le=len(ALL_SKILLS))] = 20, category: str | None = None) -> dict:
        jobs = analytics.jobs_in_category(state().jobs, category)
        return {"jobs": len(jobs), "skills": frame_to_records(analytics.skill_demand(jobs, n))}

    @app.get("/salaries/summary", tags=["analytics"])
    def salary_summary(by: Annotated[str, Query(pattern="^(seniority|category|employment_type)$")] = "seniority"):
        """p25 / median / p75 of annual USD salary, overall and per group."""
        jobs = state().jobs
        return {
            "overall": analytics.salary_overall(jobs),
            "groups": frame_to_records(analytics.salary_summary(jobs, by)),
        }

    @app.get("/salaries/top-skills", tags=["analytics"])
    def top_paying_skills(min_count: Annotated[int, Query(ge=1)] = 5, n: Annotated[int, Query(ge=1, le=50)] = 15):
        return {"skills": frame_to_records(analytics.top_paying_skills(state().jobs, min_count, n))}

    @app.post("/fit", tags=["analytics"])
    def fit(request: FitRequest) -> dict:
        """Compare a skill list (or short resume) with what the market asks for."""
        jobs = state().jobs
        user_skills = parse_user_skills(request.skills_text)
        gap = analytics.skill_gap(jobs, user_skills, request.category, request.top_n)
        best = analytics.best_matching_jobs(jobs, user_skills, request.category, request.jobs)
        columns = ["id", "title", "company", "apply_url", "source_url", "coverage", "jaccard", "matched", "missing"]
        return {
            "your_skills": user_skills,
            "jobs_in_scope": gap["jobs"],
            "match_score": gap["match_score"],
            "skills": frame_to_records(gap["skills"]),
            "best_jobs": frame_to_records(best, columns) if len(best) else [],
        }

    @app.get("/ml/metrics", tags=["ml"])
    def ml_metrics() -> dict:
        """Held-out test metrics of both models, plus the options the salary model knows."""
        models = state().models
        return {
            "salary": models.salary.metrics if models.salary else None,
            "salary_options": models.salary.options if models.salary else None,
            "category": models.category.metrics if models.category else None,
            "categories": models.category.classes if models.category else None,
            "problems": models.problems,
        }

    @app.post("/ml/predict-salary", tags=["ml"])
    def predict_salary(request: SalaryRequest) -> dict:
        model = state().models.salary
        if model is None:
            raise HTTPException(status_code=503, detail="Salary model is not trained yet. POST /refresh first.")
        skills = parse_user_skills("\n".join(request.skills))
        result = model.predict(
            request.title, request.seniority, request.category, request.location, skills, request.employment_type
        )
        return {**result, "skills_used": skills, "metrics": model.metrics}

    @app.post("/ml/classify", tags=["ml"])
    def classify(request: ClassifyRequest) -> dict:
        model = state().models.category
        if model is None:
            raise HTTPException(status_code=503, detail="Category model is not trained yet. POST /refresh first.")
        return {"predictions": model.predict(request.title, request.description), "metrics": model.metrics}

    return app


def _background_refresh(state: AppState, pages: int) -> None:
    try:
        state.refresh(pages)
    except Exception:  # noqa: BLE001 - a failed startup refresh must not kill the server
        logger.exception("Startup refresh failed; POST /refresh to try again")


app = create_app()
