import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "job_analyzer"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", "5432"))
}

def get_db_connection():
    """
    Create and return a connection to PostgreSQL database
    """
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None
    
def create_tables():
    """
    Create tables in PostgreSQL if they don't exist
    """
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        
        # Create jobs table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id VARCHAR(50) PRIMARY KEY,
                title VARCHAR(255),
                company VARCHAR(255),
                location VARCHAR(255),
                salary_min INTEGER,
                salary_max INTEGER,
                salary_avg INTEGER,
                description TEXT,
                skills TEXT,
                skills_count INTEGER,
                original_url TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        print("Tables created successfully!")
        
    except Exception as e:
        print(f"Error creating tables: {e}")
    finally:
        cur.close()
        conn.close()    


def save_jobs_to_db(jobs):
    """
    Save cleaned jobs to PostgreSQL database
    """
    conn = get_db_connection()
    if not conn:
        return 0
    
    try:
        cur = conn.cursor()
        saved_count = 0
        
        for job in jobs:
            try:
                # Convert skills list to comma-separated string
                skills = job.get('skills', [])
                if isinstance(skills, list):
                    skills_str = ','.join(str(s) for s in skills)
                else:
                    skills_str = str(skills)
                
                # Ensure all values are strings or integers (not dicts)
                job_id = str(job.get("id", ""))
                title = str(job.get("title", ""))
                company = str(job.get("company", "Unknown"))
                location = str(job.get("location", "Unknown"))
                description = str(job.get("description", ""))
                original_url = str(job.get("original_url", ""))
                
                cur.execute("""
                    INSERT INTO jobs (id, title, company, location, salary_min, salary_max, 
                                    salary_avg, description, skills, skills_count, original_url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (
                    job_id,
                    title,
                    company,
                    location,
                    job.get("salary_min"),
                    job.get("salary_max"),
                    job.get("salary_avg"),
                    description,
                    skills_str,
                    job.get("skills_count", 0),
                    original_url
                ))
                saved_count += 1
            except Exception as e:
                print(f"Error saving job {job.get('id')}: {e}")
                continue
        
        conn.commit()
        return saved_count
        
    except Exception as e:
        print(f"Error saving jobs: {e}")
        return 0
    finally:
        cur.close()
        conn.close()


def get_jobs_from_db(keyword=None, location=None, limit=100):
    """
    Retrieve jobs from PostgreSQL database with optional filters
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        
        # Build query with filters
        query = "SELECT * FROM jobs WHERE 1=1"
        params = []
        
        if keyword:
            query += " AND (title ILIKE %s OR description ILIKE %s)"
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        
        if location:
            query += " AND location ILIKE %s"
            params.append(f"%{location}%")
        
        query += f" LIMIT {limit}"
        
        cur.execute(query, params)
        jobs = cur.fetchall()
        
        return jobs
        
    except Exception as e:
        print(f"Error retrieving jobs: {e}")
        return []
    finally:
        cur.close()
        conn.close()