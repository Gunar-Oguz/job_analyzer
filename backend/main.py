from fastapi import FastAPI
from fetch_jobs import fetch_jobs_from_adzuna
from database import save_jobs_to_db, get_jobs_from_db, create_tables
from etl import process_jobs

app = FastAPI(title="Job Analyzer API")

@app.get("/")
def read_root():
    return {"message": "Job Analyzer API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/jobs")
def get_jobs(keyword: str = None, location: str = None, limit: int = 100):
    """
    Get jobs from PostgreSQL database
    """
    jobs = get_jobs_from_db(keyword, location, limit)
    return {"jobs": jobs, "count": len(jobs)}


@app.get("/jobs/search")
def search_jobs(
    keyword: str = "data scientist",
    location: str = "us",
    min_salary: int = None,
    max_salary: int = None,
    limit: int = 10
):
    """
    Search jobs from database with filters
    """
    # Get jobs from database instead of Adzuna
    jobs = get_jobs_from_db(keyword, location, limit=100)
    
    # Filter by salary if provided
    filtered_jobs = []
    for job in jobs:
        salary = job.get("salary_min", 0)
        
        if min_salary and salary < min_salary:
            continue
        if max_salary and salary > max_salary:
            continue
            
        filtered_jobs.append(job)
        if len(filtered_jobs) >= limit:
            break
    
    return {"jobs": filtered_jobs, "count": len(filtered_jobs)}


@app.get("/skills/top")
def get_top_skills(limit: int = 20):
    """
    Get most in-demand skills from database
    """
    # Get jobs from database
    jobs = get_jobs_from_db(limit=200)
    
    # Common tech skills to look for
    skills_list = [
        "python", "sql", "r", "java", "javascript", "aws", "azure", "gcp",
        "docker", "kubernetes", "spark", "hadoop", "tableau", "powerbi",
        "excel", "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn",
        "git", "machine learning", "deep learning", "nlp", "computer vision"
    ]
    
    # Count skill occurrences
    skill_counts = {}
    for skill in skills_list:
        skill_counts[skill] = 0
    
    # Search for skills in job descriptions
    for job in jobs:
        description = str(job.get("description", "")).lower()
        title = str(job.get("title", "")).lower()
        
        for skill in skills_list:
            if skill in description or skill in title:
                skill_counts[skill] += 1
    
    # Sort by count and get top N
    sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
    top_skills = [{"skill": skill, "count": count} for skill, count in sorted_skills[:limit]]
    
    return {"skills": top_skills, "total_jobs_analyzed": len(jobs)}


@app.get("/salaries/stats")
def get_salary_stats(keyword: str = None, location: str = None):
    """
    Get salary statistics from database
    """
    jobs = get_jobs_from_db(keyword, location, limit=200)
    
    salaries = []
    for job in jobs:
        if job.get("salary_min"):
            salaries.append(job.get("salary_min"))
        if job.get("salary_max"):
            salaries.append(job.get("salary_max"))
    
    if not salaries:
        return {"message": "No salary data available"}
    
    salaries.sort()
    count = len(salaries)
    
    return {
        "keyword": keyword or "all",
        "location": location or "all",
        "min_salary": min(salaries),
        "max_salary": max(salaries),
        "average_salary": sum(salaries) // count,
        "median_salary": salaries[count // 2],
        "jobs_analyzed": len(jobs)
    }


@app.get("/companies/hiring")
def get_top_companies(keyword: str = None, location: str = None, limit: int = 10):
    """
    Get companies posting the most jobs from database
    """
    jobs = get_jobs_from_db(keyword, location, limit=200)
    
    company_counts = {}
    for job in jobs:
        company = job.get("company", "Unknown")
        if company and company != "Unknown":
            company_counts[company] = company_counts.get(company, 0) + 1
    
    sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)
    top_companies = [{"company": comp, "job_count": count} for comp, count in sorted_companies[:limit]]
    
    return {"companies": top_companies, "total_companies": len(company_counts)}


@app.get("/locations/best")
def get_best_locations(keyword: str = None, limit: int = 10):
    """
    Get locations with most job opportunities from database
    """
    jobs = get_jobs_from_db(keyword, limit=200)
    
    location_counts = {}
    for job in jobs:
        location = job.get("location", "Unknown")
        if location and location != "Unknown":
            location_counts[location] = location_counts.get(location, 0) + 1
    
    sorted_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
    top_locations = [{"location": loc, "job_count": count} for loc, count in sorted_locations[:limit]]
    
    return {"locations": top_locations, "total_locations": len(location_counts)}
    

@app.get("/jobs/{job_id}")
def get_job_by_id(job_id: str):
    """
    Get details for a specific job by ID from database
    """
    jobs = get_jobs_from_db(limit=1000)
    
    for job in jobs:
        if str(job.get("id")) == str(job_id):
            return {"job": job}
    
    return {"error": "Job not found", "job_id": job_id}


@app.get("/remote")
def get_remote_jobs(keyword: str = None, limit: int = 10):
    """
    Get remote job opportunities from database
    """
    jobs = get_jobs_from_db(keyword, limit=200)
    
    remote_jobs = []
    for job in jobs:
        title = str(job.get("title", "")).lower()
        description = str(job.get("description", "")).lower()
        location = str(job.get("location", "")).lower()
        
        if "remote" in title or "remote" in description or "remote" in location:
            remote_jobs.append(job)
            if len(remote_jobs) >= limit:
                break
    
    return {"jobs": remote_jobs, "count": len(remote_jobs)}


@app.post("/refresh")
def refresh_jobs(keyword: str = "data", location: str = "us", results: int = 50):
    """
    Fetch, clean, transform, and save jobs to database
    """
    # Step 1: Fetch from Adzuna
    raw_jobs = fetch_jobs_from_adzuna(keyword, location, results)
    
    # DEBUG: Print raw job sample
    if raw_jobs:
        print("=" * 80)
        print("RAW JOB SAMPLE (from Adzuna):")
        print(raw_jobs[0])
        print("=" * 80)
    
    # Step 2: ETL - Clean and transform
    cleaned_jobs = process_jobs(raw_jobs)
    
    # DEBUG: Print cleaned job sample
    if cleaned_jobs:
        print("=" * 80)
        print("CLEANED JOB SAMPLE (after ETL):")
        print(cleaned_jobs[0])
        print("=" * 80)
    
    # Step 3: Save to PostgreSQL
    saved_count = save_jobs_to_db(cleaned_jobs)
    
    return {
        "message": "Jobs refreshed with ETL processing",
        "fetched": len(raw_jobs),
        "cleaned": len(cleaned_jobs),
        "saved": saved_count
    }