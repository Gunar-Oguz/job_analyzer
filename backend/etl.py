import re
from typing import List, Dict
from collections import Counter

def clean_html(text: str) -> str:
    """
    Remove HTML tags and clean text
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text.strip()


def extract_skills(description: str, title: str) -> List[str]:
    """
    Extract technical skills from job description and title
    """
    # Comprehensive skills list
    all_skills = {
        'python', 'sql', 'r', 'java', 'javascript', 'scala', 'c++',
        'aws', 'azure', 'gcp', 'google cloud', 'cloud',
        'docker', 'kubernetes', 'jenkins', 'ci/cd', 'git', 'github',
        'spark', 'hadoop', 'kafka', 'airflow', 'databricks',
        'tableau', 'powerbi', 'power bi', 'looker', 'qlik',
        'excel', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
        'machine learning', 'deep learning', 'nlp', 'computer vision', 'ai',
        'fastapi', 'flask', 'django', 'streamlit',
        'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
        'api', 'rest', 'etl', 'data pipeline', 'data warehouse'
    }
    
    combined_text = f"{description} {title}".lower()
    
    found_skills = []
    for skill in all_skills:
        if skill in combined_text:
            found_skills.append(skill)
    
    return found_skills


def remove_duplicates(jobs: List[Dict]) -> List[Dict]:
    """
    Remove duplicate jobs based on ID
    """
    seen_ids = set()
    unique_jobs = []
    
    for job in jobs:
        job_id = job.get('id')
        if job_id and job_id not in seen_ids:
            seen_ids.add(job_id)
            unique_jobs.append(job)
    
    return unique_jobs


def standardize_salary(salary_min: int, salary_max: int) -> Dict:
    """
    Standardize and validate salary data
    """
    # Handle None values
    if not salary_min:
        salary_min = 0
    if not salary_max:
        salary_max = 0
    
    # Ensure min is less than max
    if salary_min > salary_max and salary_max > 0:
        salary_min, salary_max = salary_max, salary_min
    
    # Calculate average if both exist
    avg_salary = (salary_min + salary_max) // 2 if (salary_min and salary_max) else (salary_min or salary_max)
    
    return {
        'salary_min': salary_min,
        'salary_max': salary_max,
        'salary_avg': avg_salary
    }


def clean_location(location_data: Dict) -> str:
    """
    Extract and clean location string
    """
    if isinstance(location_data, dict):
        return location_data.get('display_name', 'Unknown')
    elif isinstance(location_data, str):
        return location_data
    return 'Unknown'


def clean_company(company_data: Dict) -> str:
    """
    Extract and clean company name
    """
    if isinstance(company_data, dict):
        return company_data.get('display_name', 'Unknown')
    elif isinstance(company_data, str):
        return company_data
    return 'Unknown'


def transform_job(job: Dict) -> Dict:
    """
    Apply all transformations to a single job
    """
    # Extract raw data
    raw_description = job.get('description', '')
    raw_title = job.get('title', '')
    raw_location = job.get('location', {})
    raw_company = job.get('company', {})
    salary_min = job.get('salary_min')
    salary_max = job.get('salary_max')
    
    # Clean and transform
    clean_desc = clean_html(raw_description)
    skills = extract_skills(clean_desc, raw_title)
    location = clean_location(raw_location)
    company = clean_company(raw_company)
    salary_data = standardize_salary(salary_min, salary_max)
    
    # Build cleaned job object
    cleaned_job = {
        'id': job.get('id'),
        'title': raw_title,
        'company': company,
        'location': location,
        'salary_min': salary_data['salary_min'],
        'salary_max': salary_data['salary_max'],
        'salary_avg': salary_data['salary_avg'],
        'description': clean_desc,
        'skills': skills,
        'skills_count': len(skills),
        'original_url': job.get('redirect_url', '')
    }
    
    return cleaned_job


def process_jobs(jobs: List[Dict]) -> List[Dict]:
    """
    Main ETL function: Clean and transform all jobs
    """
    # Remove duplicates
    unique_jobs = remove_duplicates(jobs)
    
    # Transform each job
    cleaned_jobs = []
    for job in unique_jobs:
        try:
            cleaned_job = transform_job(job)
            cleaned_jobs.append(cleaned_job)
        except Exception as e:
            print(f"Error transforming job {job.get('id')}: {e}")
            continue
    
    return cleaned_jobs


def get_skill_statistics(jobs: List[Dict]) -> Dict:
    """
    Calculate skill statistics across all jobs
    """
    all_skills = []
    for job in jobs:
        all_skills.extend(job.get('skills', []))
    
    skill_counts = Counter(all_skills)
    
    return {
        'total_skills': len(all_skills),
        'unique_skills': len(skill_counts),
        'top_skills': skill_counts.most_common(20)
    }

