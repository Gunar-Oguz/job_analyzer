"""A SYNTHETIC job feed shaped like the Himalayas API. No real postings.

Companies, titles and links are invented (example.com / example.org), and
salaries follow a simple made-up rule so the ML tests have a signal to learn.
"""

from __future__ import annotations

import random

NOW = 1_800_000_000  # a fixed "now" (epoch seconds) so tests are deterministic

ROLES = [
    ("Data Science", "Data Scientist", ["Python", "SQL", "machine learning", "statistics", "pandas"], 120_000),
    ("Data Science", "Data Analyst", ["SQL", "Excel", "Tableau", "statistics"], 80_000),
    ("Developer", "Backend Engineer", ["Python", "PostgreSQL", "Docker", "AWS", "REST APIs"], 130_000),
    ("Developer", "Frontend Engineer", ["JavaScript", "TypeScript", "React.js", "CSS"], 115_000),
    ("Developer", "DevOps Engineer", ["Kubernetes", "Terraform", "AWS", "Linux", "CI/CD"], 135_000),
    ("Sales", "Account Executive", ["Salesforce", "CRM", "B2B sales"], 90_000),
    ("Marketing", "Growth Marketer", ["SEO", "Google Analytics", "Google Ads", "copywriting"], 85_000),
    ("Customer Service", "Customer Success Manager", ["Zendesk", "customer success", "CRM"], 70_000),
]
SENIORITY = [("Entry-level", 0.7), ("Mid-level", 1.0), ("Senior", 1.35), ("Director", 1.7)]
COUNTRIES = [[], ["United States"], ["United States", "Canada"], ["Germany", "France"], ["Philippines"]]
EMPLOYMENT = ["Full Time", "Full Time", "Full Time", "Contractor", "Part Time"]


def make_raw_jobs(n: int = 160, seed: int = 0) -> list[dict]:
    """Return ``n`` raw jobs in Himalayas format, the same for the same seed."""
    rng = random.Random(seed)
    jobs = []
    for i in range(n):
        category, role, skills, base = ROLES[i % len(ROLES)]
        level, factor = SENIORITY[rng.randrange(len(SENIORITY))]
        title = f"{level.replace('-level', '')} {role}" if level != "Mid-level" else role
        chosen = rng.sample(skills, k=min(len(skills), rng.randint(2, len(skills))))
        description = (
            f"<div><h3>About Example Co {i}</h3><p>We are hiring a <strong>{role}</strong>.</p>"
            f"<ul>{''.join(f'<li>Experience with {s}</li>' for s in chosen)}</ul>"
            "<p>Benefits &amp; perks for everyone.</p></div>"
        )
        job = {
            "title": title,
            "excerpt": f"Join Example Co {i % 40} as a {role}.",
            "companyName": f"Example Co {i % 40}",
            "companySlug": f"example-co-{i % 40}",
            "companyLogo": None,
            "employmentType": EMPLOYMENT[i % len(EMPLOYMENT)],
            "minSalary": None,
            "maxSalary": None,
            "salaryPeriod": "annual",
            "seniority": [level],
            "currency": None,
            "locationRestrictions": COUNTRIES[i % len(COUNTRIES)],
            "timezoneRestrictions": [],
            "categories": [role.replace(" ", "-")],
            "parentCategories": [category],
            "description": description,
            "pubDate": NOW - rng.randint(0, 20) * 86_400 - rng.randint(0, 86_000),
            "expiryDate": NOW + 30 * 86_400,
            "applicationLink": f"https://jobs.example.com/{i}?utm_source=api",
            "guid": f"https://jobs.example.org/{i}",
        }
        if i % 3 != 2:  # two thirds of jobs have a USD salary
            mid = base * factor * rng.uniform(0.9, 1.1)
            job.update(minSalary=round(mid * 0.9), maxSalary=round(mid * 1.1), currency="USD")
        jobs.append(job)
    return jobs


def fake_page(jobs: list[dict], start: int, size: int = 20) -> dict:
    """One page of the browse endpoint, with a cursor when more jobs follow."""
    chunk = jobs[start : start + size]
    next_cursor = str(start + size) if start + size < len(jobs) else None
    return {"jobs": chunk, "nextCursor": next_cursor, "limit": size, "totalCount": len(jobs)}
