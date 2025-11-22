import requests
import os
from dotenv import load_dotenv

load_dotenv()
APP_ID = os.getenv("ADZUNA_APP_ID")
API_KEY = os.getenv("ADZUNA_API_KEY")

def fetch_jobs_from_adzuna(keyword="data scientist", location="us", results_per_page=10):
    """
    Fetch real job data from Adzuna API
    """
    url = f"https://api.adzuna.com/v1/api/jobs/{location}/search/1"
    params = {
        "app_id": APP_ID,
        "app_key": API_KEY,
        "what": keyword,
        "results_per_page": results_per_page
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data.get("results", [])
    else:
        return []
