import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    """Test root endpoint returns welcome message"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_jobs():
    """Test jobs listing endpoint"""
    response = client.get("/jobs?limit=5")
    assert response.status_code == 200
    assert "jobs" in response.json()
    assert "count" in response.json()

def test_jobs_search():
    """Test job search with filters"""
    response = client.get("/jobs/search?keyword=data&location=us&limit=5")
    assert response.status_code == 200
    assert "jobs" in response.json()

def test_skills_top():
    """Test top skills endpoint"""
    response = client.get("/skills/top?limit=10")
    assert response.status_code == 200
    assert "skills" in response.json()

def test_salary_stats():
    """Test salary statistics endpoint"""
    response = client.get("/salaries/stats?keyword=data")
    assert response.status_code == 200

def test_top_companies():
    """Test top companies endpoint"""
    response = client.get("/companies/hiring?limit=5")
    assert response.status_code == 200
    assert "companies" in response.json()
