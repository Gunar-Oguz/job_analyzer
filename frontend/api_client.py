"""Thin HTTP client for the FastAPI backend (used when BACKEND_URL is set).

The remote model classes have the same ``metrics`` / ``options`` /
``predict`` interface as the local models in ``core.ml``, so the UI code
does not care which mode it runs in.
"""

from __future__ import annotations

import pandas as pd
import requests

from core.etl import records_to_frame
from core.ml import Models

TIMEOUT = 30


class BackendError(RuntimeError):
    """The backend could not be reached or returned an error."""


def _request(method: str, url: str, **kwargs) -> dict:
    try:
        response = requests.request(method, url, timeout=kwargs.pop("timeout", TIMEOUT), **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        raise BackendError(f"Backend request failed: {error}") from error


def load_jobs(base_url: str) -> tuple[pd.DataFrame, dict]:
    """All live jobs plus the pipeline report."""
    jobs = _request("GET", f"{base_url}/jobs", params={"limit": 5000})
    pipeline = _request("GET", f"{base_url}/pipeline")
    report = dict(pipeline.get("last_refresh") or {})
    report["store"] = pipeline.get("store")
    return records_to_frame(jobs["jobs"]), report


def refresh(base_url: str, max_pages: int = 50) -> dict:
    return _request("POST", f"{base_url}/refresh", params={"max_pages": max_pages}, timeout=180)


class RemoteSalaryModel:
    def __init__(self, base_url: str, metrics: dict, options: dict) -> None:
        self.base_url, self.metrics, self.options = base_url, metrics, options

    def predict(self, title, seniority, category, location="United States", skills=None, employment_type="Full Time"):
        body = {
            "title": title,
            "seniority": seniority,
            "category": category,
            "location": location,
            "employment_type": employment_type,
            "skills": list(skills or []),
        }
        result = _request("POST", f"{self.base_url}/ml/predict-salary", json=body)
        return {key: result[key] for key in ("predicted", "low", "high")}


class RemoteCategoryModel:
    def __init__(self, base_url: str, metrics: dict, classes: list[str]) -> None:
        self.base_url, self.metrics, self.classes = base_url, metrics, classes

    def predict(self, title: str, description: str = "", top_k: int = 3) -> list[dict]:
        body = {"title": title, "description": description}
        return _request("POST", f"{self.base_url}/ml/classify", json=body)["predictions"][:top_k]


def load_models(base_url: str) -> Models:
    info = _request("GET", f"{base_url}/ml/metrics")
    salary = RemoteSalaryModel(base_url, info["salary"], info["salary_options"]) if info.get("salary") else None
    category = RemoteCategoryModel(base_url, info["category"], info["categories"]) if info.get("category") else None
    return Models(salary=salary, category=category, problems=info.get("problems") or {})
