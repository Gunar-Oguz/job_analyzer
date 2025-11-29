import sys
sys.path.append('..')
from database import get_jobs_from_db
from datetime import datetime

def analyze_market(keyword: str = "data scientist"):
    """
    Analyze current job market for a given role
    
    Returns: job count, salary stats, top skills, top companies
    """
    try:
        # Step 1: Get jobs for this keyword
        jobs = get_jobs_from_db(keyword=keyword, limit=500)
        
        if not jobs:
            return {"error": f"No jobs found for '{keyword}'"}
        
        # Step 2: Calculate salary statistics
        salaries = []
        for job in jobs:
            if job.get('salary_min') and job.get('salary_min') > 0:
                salaries.append(job['salary_min'])
            if job.get('salary_max') and job.get('salary_max') > 0:
                salaries.append(job['salary_max'])
        
        salary_stats = {}
        if salaries:
            salary_stats = {
                "min": min(salaries),
                "max": max(salaries),
                "average": round(sum(salaries) / len(salaries), 2),
                "median": sorted(salaries)[len(salaries) // 2]
            }
        
        # Step 3: Count top skills
        skill_list = ["python", "sql", "aws", "machine learning", "docker", 
                      "tensorflow", "spark", "tableau", "r", "java"]
        skill_counts = {skill: 0 for skill in skill_list}
        
        for job in jobs:
            text = (job.get('title', '') + ' ' + job.get('description', '')).lower()
            for skill in skill_list:
                if skill in text:
                    skill_counts[skill] += 1
        
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Step 4: Count top companies
        company_counts = {}
        for job in jobs:
            company = job.get('company', 'Unknown')
            if company and company != 'Unknown':
                company_counts[company] = company_counts.get(company, 0) + 1
        
        top_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Step 5: Build response
        return {
            "keyword": keyword,
            "total_jobs": len(jobs),
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "salary_stats": salary_stats,
            "top_skills": [{"skill": s, "count": c} for s, c in top_skills],
            "top_companies": [{"company": c, "jobs": n} for c, n in top_companies],
            "market_summary": f"Found {len(jobs)} {keyword} jobs with average salary ${salary_stats.get('average', 'N/A'):,}"
        }
    
    except Exception as e:
        return {"error": str(e)}