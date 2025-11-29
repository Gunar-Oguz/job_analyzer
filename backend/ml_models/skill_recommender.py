import sys
sys.path.append('..')
from database import get_jobs_from_db

# Common skills to look for in job descriptions
SKILL_LIST = [
    "python", "sql", "r", "java", "scala", "javascript",
    "aws", "azure", "gcp", "docker", "kubernetes",
    "spark", "hadoop", "kafka", "airflow",
    "tableau", "powerbi", "excel", "looker",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "machine learning", "deep learning", "nlp", "computer vision",
    "statistics", "a/b testing", "git", "linux", "api"
]

def get_skill_recommendations(target_role: str, top_n: int = 10):
    """
    Recommend skills based on target job role
    
    Input: "Data Scientist"
    Output: ["python", "sql", "machine learning", ...]
    """
    try:
        # Step 1: Get jobs matching target role
        jobs = get_jobs_from_db(keyword=target_role, limit=200)
        
        if not jobs:
            return {"error": f"No jobs found for '{target_role}'"}
        
        # Step 2: Count skill occurrences
        skill_counts = {skill: 0 for skill in SKILL_LIST}
        
        for job in jobs:
            # Combine title + description
            text = (job.get('title', '') + ' ' + job.get('description', '')).lower()
            
            # Count each skill
            for skill in SKILL_LIST:
                if skill in text:
                    skill_counts[skill] += 1
        
        # Step 3: Sort by count (most common first)
        sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Step 4: Get top N skills with counts > 0
        top_skills = [
            {"skill": skill, "frequency": count, "percentage": round(count/len(jobs)*100, 1)}
            for skill, count in sorted_skills[:top_n]
            if count > 0
        ]
        
        return {
            "target_role": target_role,
            "jobs_analyzed": len(jobs),
            "recommended_skills": top_skills,
            "message": f"Top {len(top_skills)} skills for {target_role} roles"
        }
    
    except Exception as e:
        return {"error": str(e)}