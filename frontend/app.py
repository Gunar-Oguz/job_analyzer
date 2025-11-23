import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page Config
st.set_page_config(
    page_title="Job Market Analyzer",
    page_icon="💼",
    layout="wide"
)

# Title
st.title("💼 Job Market Analyzer")
st.markdown("---")

# Sidebar Filters
st.sidebar.header("🔍 Search Filters")

# Job title dropdown
job_titles = [
    "data scientist",
    "data analyst", 
    "machine learning engineer",
    "data engineer",
    "business analyst",
    "AI engineer",
    "research scientist"
]
keyword = st.sidebar.selectbox("Job Title", job_titles)

# Location dropdown
locations = [
    ("United States", "us"),
    ("United Kingdom", "uk"),
    ("Canada", "ca"),
    ("Germany", "de"),
    ("France", "fr"),
    ("Australia", "au")
]
location_display = st.sidebar.selectbox("Location", [loc[0] for loc in locations])
location = dict(locations)[location_display]

min_salary = st.sidebar.number_input("Minimum Salary ($)", min_value=0, value=0, step=10000)
max_salary = st.sidebar.number_input("Maximum Salary ($)", min_value=0, value=200000, step=10000)
limit = st.sidebar.slider("Number of Results", min_value=5, max_value=50, value=20)

search_button = st.sidebar.button("🔍 Search Jobs", type="primary")

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Job Listings", "📊 Skills Analysis", "💰 Salary Stats", "🏢 Top Companies"])

# Tab 1: Job Listings
with tab1:
    if search_button:
        try:
            # Fetch jobs from API
            with st.spinner("Searching jobs..."):
                response = requests.get(
                    f"{API_BASE_URL}/jobs/search",
                    params={
                        "keyword": keyword,
                        "location": location,
                        "min_salary": min_salary if min_salary > 0 else None,
                        "max_salary": max_salary if max_salary < 200000 else None,
                        "limit": limit
                    }
                )
            
            if response.status_code == 200:
                data = response.json()
                jobs = data.get("jobs", [])
                
                if jobs:
                    st.success(f"Found {len(jobs)} jobs matching your criteria")
                    
                    # Display each job
                    for job in jobs:
                        with st.expander(f"{job.get('title', 'N/A')} - {job.get('company', 'N/A')}"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write(f"**Location:** {job.get('location', 'N/A')}")
                                st.write(f"**Description:** {job.get('description', 'N/A')[:300]}...")
                                
                            with col2:
                                st.metric("Min Salary", f"${job.get('salary_min', 0):,}")
                                st.metric("Max Salary", f"${job.get('salary_max', 0):,}")
                                
                            if job.get('skills'):
                                st.write(f"**Skills:** {job.get('skills')}")
                else:
                    st.warning("No jobs found. Try different search criteria or refresh database.")
            else:
                st.error(f"API Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API!")
            st.info("Make sure FastAPI is running: `uvicorn main:app --reload`")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("👈 Click 'Search Jobs' button to see results")

# Tab 2: Skills Analysis
with tab2:
    if search_button:
        try:
            with st.spinner("Analyzing skills..."):
                response = requests.get(f"{API_BASE_URL}/skills/top", params={"limit": 15})
            
            if response.status_code == 200:
                data = response.json()
                skills = data.get("skills", [])
                
                if skills:
                    # Create DataFrame
                    df_skills = pd.DataFrame(skills)
                    
                    # Bar chart
                    fig = px.bar(
                        df_skills, 
                        x='skill', 
                        y='count',
                        title='Top In-Demand Skills',
                        labels={'skill': 'Skill', 'count': 'Number of Jobs'},
                        color='count',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show table
                    st.dataframe(df_skills, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("👈 Click 'Search Jobs' to see skills analysis")

# Tab 3: Salary Statistics
with tab3:
    if search_button:
        try:
            with st.spinner("Calculating salaries..."):
                response = requests.get(
                    f"{API_BASE_URL}/salaries/stats",
                    params={"keyword": keyword, "location": location}
                )
            
            if response.status_code == 200:
                data = response.json()
                
                if "message" not in data:
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Min Salary", f"${data.get('min_salary', 0):,}")
                    with col2:
                        st.metric("Average Salary", f"${data.get('average_salary', 0):,}")
                    with col3:
                        st.metric("Median Salary", f"${data.get('median_salary', 0):,}")
                    with col4:
                        st.metric("Max Salary", f"${data.get('max_salary', 0):,}")
                    
                    st.info(f"Based on {data.get('jobs_analyzed', 0)} jobs analyzed")
                else:
                    st.warning(data.get("message"))
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("👈 Click 'Search Jobs' to see salary stats")

# Tab 4: Top Companies
with tab4:
    if search_button:
        try:
            with st.spinner("Finding top companies..."):
                response = requests.get(
                    f"{API_BASE_URL}/companies/hiring",
                    params={"keyword": keyword, "location": location, "limit": 10}
                )
            
            if response.status_code == 200:
                data = response.json()
                companies = data.get("companies", [])
                
                if companies:
                    # Create DataFrame
                    df_companies = pd.DataFrame(companies)
                    
                    # Horizontal bar chart
                    fig = px.bar(
                        df_companies,
                        x='job_count',
                        y='company',
                        orientation='h',
                        title='Top 10 Hiring Companies',
                        labels={'job_count': 'Number of Jobs', 'company': 'Company'},
                        color='job_count',
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show table
                    st.dataframe(df_companies, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("👈 Click 'Search Jobs' to see top companies")

# Footer
st.markdown("---")
st.markdown("**Job Market Analyzer** | Built with Streamlit & FastAPI")