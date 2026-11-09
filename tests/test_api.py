"""FastAPI endpoints, with an in-memory store and a fake Himalayas loader."""

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.storage import MemoryStore
from core.etl import clean_jobs
from core.himalayas import FetchError
from tests.synthetic import make_raw_jobs


def fake_loader(max_pages: int = 50):
    frame, cleaning = clean_jobs(make_raw_jobs())
    return frame, {"fetched_at": "2026-01-01T00:00:00+00:00", "fetch": {"pages": max_pages}, "cleaning": cleaning}


def failing_loader(max_pages: int = 50):
    raise FetchError("Himalayas is down")


@pytest.fixture
def client():
    with TestClient(create_app(store=MemoryStore(), loader=fake_loader)) as test_client:
        test_client.post("/refresh", params={"max_pages": 8})
        yield test_client


def test_health_and_root():
    with TestClient(create_app(store=MemoryStore(), loader=fake_loader)) as c:
        health = c.get("/health").json()
        assert health["status"] == "ok" and health["jobs"] == 0 and health["store"] == "memory"
        assert c.get("/").json()["docs"] == "/docs"
        assert c.post("/ml/classify", json={"title": "Engineer"}).status_code == 503


def test_refresh_loads_data_and_trains_models(client):
    health = client.get("/health").json()
    assert health["jobs"] == 160
    assert health["models_ready"] == {"salary": True, "category": True}
    pipeline = client.get("/pipeline").json()
    assert pipeline["last_refresh"]["cleaning"]["clean_rows"] == 160
    assert pipeline["last_refresh"]["fetch"]["pages"] == 8


def test_refresh_failure_returns_503():
    with TestClient(create_app(store=MemoryStore(), loader=failing_loader)) as c:
        response = c.post("/refresh")
        assert response.status_code == 503 and "down" in response.json()["detail"]


def test_jobs_search_and_filters(client):
    body = client.get("/jobs", params={"limit": 5}).json()
    assert body["total"] == 160 and body["count"] == 5
    job = body["jobs"][0]
    assert job["apply_url"].startswith("https://jobs.example.com/") and "description" not in job
    filtered = client.get(
        "/jobs", params={"category": ["Data Science", "Sales"], "seniority": "Senior", "has_salary": True}
    ).json()
    assert filtered["total"] > 0
    assert all(j["category"] in {"Data Science", "Sales"} and j["salary_usd"] for j in filtered["jobs"])
    assert client.get("/jobs", params={"limit": 0}).status_code == 422


def test_job_detail_and_404(client):
    job_id = client.get("/jobs", params={"limit": 1}).json()["jobs"][0]["id"]
    detail = client.get(f"/jobs/{job_id}").json()
    assert detail["id"] == job_id and detail["description"]
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_analytics_endpoints(client):
    overview = client.get("/stats/overview").json()
    assert overview["jobs"] == 160 and overview["categories"]
    skills = client.get("/skills/top", params={"n": 5, "category": "Developer"}).json()
    assert len(skills["skills"]) == 5
    salaries = client.get("/salaries/summary", params={"by": "category"}).json()
    assert salaries["overall"]["median"] > 0 and salaries["groups"]
    assert client.get("/salaries/summary", params={"by": "title; DROP TABLE jobs"}).status_code == 422
    assert client.get("/salaries/top-skills", params={"min_count": 3}).json()["skills"]


def test_fit_endpoint(client):
    body = client.post("/fit", json={"skills_text": "python, sql, statistics", "category": "Data Science"}).json()
    assert body["your_skills"] == ["Python", "SQL", "Statistics"]
    assert 0 < body["match_score"] <= 100
    assert body["best_jobs"] and body["best_jobs"][0]["apply_url"]


def test_ml_endpoints(client):
    metrics = client.get("/ml/metrics").json()
    assert metrics["salary"]["rows"] > 0 and metrics["category"]["accuracy"] > 0
    salary = client.post(
        "/ml/predict-salary",
        json={"title": "Senior Backend Engineer", "seniority": "Senior", "category": "Developer", "skills": ["python"]},
    ).json()
    assert salary["low"] <= salary["predicted"] <= salary["high"]
    assert salary["skills_used"] == ["Python"]
    classified = client.post("/ml/classify", json={"title": "Account Executive, B2B sales"}).json()
    assert classified["predictions"][0]["category"] == "Sales"
    assert client.post("/ml/predict-salary", json={"title": ""}).status_code == 422
