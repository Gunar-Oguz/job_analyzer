from fastapi import FastAPI, HTTPException
from fetch_jobs import fetch_jobs_from_adzuna
from database import save_jobs_to_db, get_jobs_from_db, create_tables
from etl import process_jobs
from logger import logger
from ml_models.predict import predict_salary, get_model_stats
from ml_models.predict_category import predict_job_category

app = FastAPI(title="Job Analyzer API")

@app.on_event("startup")
async def startup_event():
    """Log when API starts"""
    logger.info("Job Analyzer API starting up...")
    create_tables()
    logger.info("Database tables verified")

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Job Analyzer API is running!"}

@app.get("/health")
def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}

@app.get("/jobs")
def get_jobs(keyword: str = None, location: str = None, limit: int = 100):
    """Get jobs from PostgreSQL database"""
    try:
        logger.info(f"Fetching jobs - keyword: {keyword}, location: {location}, limit: {limit}")
        jobs = get_jobs_from_db(keyword, location, limit)
        logger.info(f"Successfully retrieved {len(jobs)} jobs")
        return {"jobs": jobs, "count": len(jobs)}
    except Exception as e:
        logger.error(f"Error fetching jobs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving jobs from database")

@app.get("/jobs/search")
def search_jobs(
    keyword: str = "data scientist",
    location: str = "us",
    min_salary: int = None,
    max_salary: int = None,
    limit: int = 10
):
    """Search jobs from database with filters"""
    try:
        logger.info(f"Searching jobs - keyword: {keyword}, salary: {min_salary}-{max_salary}")
        jobs = get_jobs_from_db(keyword, location, limit=100)
        
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
        
        logger.info(f"Found {len(filtered_jobs)} jobs matching criteria")
        return {"jobs": filtered_jobs, "count": len(filtered_jobs)}
    except Exception as e:
        logger.error(f"Error searching jobs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error searching jobs")

@app.get("/skills/top")
def get_top_skills(limit: int = 20):
    """Get most in-demand skills from database"""
    try:
        logger.info(f"Fetching top {limit} skills")
        jobs = get_jobs_from_db(limit=200)
        
        skills_list = [
            "python", "sql", "r", "java", "javascript", "aws", "azure", "gcp",
            "docker", "kubernetes", "spark", "hadoop", "tableau", "powerbi",
            "excel", "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn",
            "git", "machine learning", "deep learning", "nlp", "computer vision"
        ]
        
        skill_counts = {}
        for skill in skills_list:
            skill_counts[skill] = 0
        
        for job in jobs:
            description = str(job.get("description", "")).lower()
            title = str(job.get("title", "")).lower()
            
            for skill in skills_list:
                if skill in description or skill in title:
                    skill_counts[skill] += 1
        
        sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        top_skills = [{"skill": skill, "count": count} for skill, count in sorted_skills[:limit]]
        
        logger.info(f"Successfully calculated top {limit} skills")
        return {"skills": top_skills, "total_jobs_analyzed": len(jobs)}
    except Exception as e:
        logger.error(f"Error calculating top skills: {str(e)}")
        raise HTTPException(status_code=500, detail="Error calculating skills")

@app.get("/salaries/stats")
def get_salary_stats(keyword: str = None, location: str = None):
    """Get salary statistics from database"""
    try:
        logger.info(f"Calculating salary stats - keyword: {keyword}, location: {location}")
        jobs = get_jobs_from_db(keyword, location, limit=200)
        
        salaries = []
        for job in jobs:
            if job.get("salary_min"):
                salaries.append(job.get("salary_min"))
            if job.get("salary_max"):
                salaries.append(job.get("salary_max"))
        
        if not salaries:
            logger.warning("No salary data available")
            return {"message": "No salary data available"}
        
        salaries.sort()
        count = len(salaries)
        
        stats = {
            "keyword": keyword or "all",
            "location": location or "all",
            "min_salary": min(salaries),
            "max_salary": max(salaries),
            "average_salary": sum(salaries) // count,
            "median_salary": salaries[count // 2],
            "jobs_analyzed": len(jobs)
        }
        
        logger.info(f"Salary stats calculated: avg ${stats['average_salary']}")
        return stats
    except Exception as e:
        logger.error(f"Error calculating salary stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error calculating salary statistics")

@app.get("/companies/hiring")
def get_top_companies(keyword: str = None, location: str = None, limit: int = 10):
    """Get companies posting the most jobs from database"""
    try:
        logger.info(f"Fetching top {limit} hiring companies")
        jobs = get_jobs_from_db(keyword, location, limit=200)
        
        company_counts = {}
        for job in jobs:
            company = job.get("company", "Unknown")
            if company and company != "Unknown":
                company_counts[company] = company_counts.get(company, 0) + 1
        
        sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)
        top_companies = [{"company": comp, "job_count": count} for comp, count in sorted_companies[:limit]]
        
        logger.info(f"Found {len(top_companies)} companies")
        return {"companies": top_companies, "total_companies": len(company_counts)}
    except Exception as e:
        logger.error(f"Error fetching top companies: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching companies")

@app.get("/locations/best")
def get_best_locations(keyword: str = None, limit: int = 10):
    """Get locations with most job opportunities from database"""
    try:
        logger.info(f"Fetching top {limit} locations")
        jobs = get_jobs_from_db(keyword, limit=200)
        
        location_counts = {}
        for job in jobs:
            location = job.get("location", "Unknown")
            if location and location != "Unknown":
                location_counts[location] = location_counts.get(location, 0) + 1
        
        sorted_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
        top_locations = [{"location": loc, "job_count": count} for loc, count in sorted_locations[:limit]]
        
        logger.info(f"Found {len(top_locations)} locations")
        return {"locations": top_locations, "total_locations": len(location_counts)}
    except Exception as e:
        logger.error(f"Error fetching locations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching locations")

@app.get("/jobs/{job_id}")
def get_job_by_id(job_id: str):
    """Get details for a specific job by ID from database"""
    try:
        logger.info(f"Fetching job with ID: {job_id}")
        jobs = get_jobs_from_db(limit=1000)
        
        for job in jobs:
            if str(job.get("id")) == str(job_id):
                logger.info(f"Job {job_id} found")
                return {"job": job}
        
        logger.warning(f"Job {job_id} not found")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching job details")

@app.get("/remote")
def get_remote_jobs(keyword: str = None, limit: int = 10):
    """Get remote job opportunities from database"""
    try:
        logger.info(f"Fetching remote jobs - keyword: {keyword}")
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
        
        logger.info(f"Found {len(remote_jobs)} remote jobs")
        return {"jobs": remote_jobs, "count": len(remote_jobs)}
    except Exception as e:
        logger.error(f"Error fetching remote jobs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching remote jobs")

@app.post("/refresh")
def refresh_jobs(keyword: str = "data", location: str = "us", results: int = 50):
    """Fetch, clean, transform, and save jobs to database"""
    try:
        logger.info(f"Starting job refresh - keyword: {keyword}, location: {location}, results: {results}")
        
        # Fetch from Adzuna
        raw_jobs = fetch_jobs_from_adzuna(keyword, location, results)
        logger.info(f"Fetched {len(raw_jobs)} raw jobs from Adzuna")
        
        # ETL - Clean and transform
        cleaned_jobs = process_jobs(raw_jobs)
        logger.info(f"Cleaned {len(cleaned_jobs)} jobs through ETL")
        
        # Save to PostgreSQL
        saved_count = save_jobs_to_db(cleaned_jobs)
        logger.info(f"Saved {saved_count} jobs to database")
        
        return {
            "message": "Jobs refreshed with ETL processing",
            "fetched": len(raw_jobs),
            "cleaned": len(cleaned_jobs),
            "saved": saved_count
        }
    except Exception as e:
        logger.error(f"Error refreshing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error refreshing jobs: {str(e)}")


@app.post("/ml/predict-salary")
def predict_job_salary(title: str, location: str, company: str = "Unknown"):
    """
    Predict salary for a job using ML model
    """
    try:
        logger.info(f"Predicting salary for: {title} at {company} in {location}")
        result = predict_salary(title, location, company)
        
        if "error" in result:
            logger.error(f"Prediction error: {result['error']}")
            raise HTTPException(status_code=400, detail=result['error'])
        
        logger.info(f"Predicted salary: ${result['predicted_salary']:,.2f}")
        return result
    except Exception as e:
        logger.error(f"Error in salary prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/ml/salary-model-stats")
def get_salary_model_info():
    """
    Get information about the trained ML model
    """
    try:
        logger.info("Fetching ML model statistics")
        stats = get_model_stats()
        
        if "error" in stats:
            logger.error(f"Model stats error: {stats['error']}")
            raise HTTPException(status_code=500, detail=stats['error'])
        
        return stats
    except Exception as e:
        logger.error(f"Error fetching model stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@app.post("/ml/classify-job")
def classify_job(title: str, description: str = ""):
    """
    Predict job category using ML model
    """
    try:
        logger.info(f"Classifying job: {title}")
        result = predict_job_category(title, description)
        
        if "error" in result:
            logger.error(f"Classification error: {result['error']}")
            raise HTTPException(status_code=400, detail=result['error'])
        
        logger.info(f"Predicted category: {result['predicted_category']}")
        return result
    except Exception as e:
        logger.error(f"Error in classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")