"""Skill dictionary and word-boundary skill extraction.

Every skill has a canonical display name and a list of aliases. Aliases are
regular-expression fragments that are wrapped in token boundaries, so "r"
never matches inside "for", "ai" never matches inside "maintain", and
"C++" or "Node.js" still work.

Some aliases are also ordinary English words ("excel at", "react to", "the
rest of", "go further"). For those skills the short alias is only accepted
with its exact capitalisation (see ``CASE_SENSITIVE``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# "+", "#", "." and "-" count as part of a token, so "C++" is one token and
# "r" in "R&D" or "e.g." is not a skill.
_LEFT = r"(?<![\w+#.-])"
_RIGHT = r"(?![\w+#&-]|\.\w)"


def _word(*aliases: str) -> str:
    """Wrap alias fragments in token boundaries."""
    return _LEFT + "(?:" + "|".join(aliases) + ")" + _RIGHT


# Case-insensitive aliases.
SKILLS: dict[str, list[str]] = {
    # Programming languages
    "Python": ["python3?"],
    "SQL": ["sql", "t-sql", "pl/sql"],
    "R": [],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "ecmascript"],
    "TypeScript": ["typescript"],
    "Go": ["golang"],
    "Rust": ["rust"],
    "C++": [r"c\+\+", "cpp"],
    "C#": ["c#", r"\.net", "dotnet"],
    "Scala": ["scala"],
    "Kotlin": ["kotlin"],
    "Swift": ["swiftui"],
    "Ruby": ["ruby"],
    "PHP": ["php"],
    # Web and backend frameworks
    "React": [r"react\.js", "reactjs", "react native"],
    "Vue": ["vue", r"vue\.js", "vuejs"],
    "Angular": ["angular", "angularjs"],
    "Node.js": [r"node\.js", "nodejs"],
    "Next.js": [r"next\.js", "nextjs"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Spring": ["spring boot"],
    "Ruby on Rails": ["ruby on rails"],
    "GraphQL": ["graphql"],
    "REST APIs": ["restful", "rest apis?"],
    # Data engineering and analytics
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Spark": ["spark", "pyspark"],
    "Kafka": ["kafka"],
    "Airflow": ["airflow"],
    "dbt": ["dbt"],
    "Snowflake": ["snowflake"],
    "Databricks": ["databricks"],
    "BigQuery": ["bigquery", "big query"],
    "Redshift": ["redshift"],
    "ETL": ["etl", "elt"],
    "Hadoop": ["hadoop"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi", "powerbi"],
    "Looker": ["looker"],
    "Excel": ["microsoft excel", "ms excel", "excel spreadsheets?"],
    "Statistics": ["statistics", "statistical"],
    "A/B Testing": ["a/b testing", "a/b tests?", "ab testing"],
    # Databases
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "Elasticsearch": ["elasticsearch", "elastic search", "opensearch"],
    # Cloud and DevOps
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Terraform": ["terraform"],
    "CI/CD": ["ci/cd", "continuous integration", "continuous delivery"],
    "Linux": ["linux"],
    "Git": ["git", "github", "gitlab"],
    # Machine learning and AI
    "Machine Learning": ["machine learning"],
    "Deep Learning": ["deep learning", "neural networks?"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow", "keras"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "LLMs": ["llms?", "large language models?", "prompt engineering"],
    "Generative AI": ["generative ai", "genai", "gen ai"],
    "AI": ["artificial intelligence"],
    # Design, product and delivery
    "Figma": ["figma"],
    "UX Research": ["ux research", "user research"],
    "Agile": ["agile", "scrum", "kanban"],
    "Jira": ["jira"],
    "Product Management": ["product management", "product roadmaps?"],
    "Project Management": ["project management", "pmp"],
    # Go-to-market and business tools
    "Salesforce": ["salesforce"],
    "HubSpot": ["hubspot"],
    "CRM": ["crm"],
    "SEO": ["seo", "search engine optimi[sz]ation"],
    "Google Analytics": ["google analytics", "ga4"],
    "Google Ads": ["google ads", "adwords", "ppc"],
    "Copywriting": ["copywriting", "copywriter"],
    "B2B Sales": ["b2b sales", "enterprise sales"],
    "Customer Success": ["customer success"],
    "Zendesk": ["zendesk"],
    "Accounting": ["accounting", "gaap", "ifrs"],
    "QuickBooks": ["quickbooks"],
    "Cybersecurity": ["cybersecurity", "cyber security", "information security", "infosec"],
}

# Aliases that only count with this exact capitalisation.
CASE_SENSITIVE: dict[str, list[str]] = {
    "R": ["R"],
    "Swift": ["Swift"],
    "React": ["React"],
    "Node.js": ["Node"],
    "Spring": ["Spring"],
    "Ruby on Rails": ["Rails"],
    "REST APIs": ["REST"],
    "Excel": ["Excel"],
    "Machine Learning": ["ML"],
    "LLMs": ["RAG"],
    "AI": ["AI"],
}

# "Go" is a very common verb, so we only accept it in a list-like context
# ("Python, Go and Rust", "Go developer", "(Go)").
_GO = _LEFT + r"Go(?=\s*(?:[,/)]|and\b|or\b|developer|engineer|programming|lang))"


def _compile(name: str) -> re.Pattern[str]:
    parts = []
    if SKILLS.get(name):
        parts.append("(?i:" + _word(*SKILLS[name]) + ")")
    if name in CASE_SENSITIVE:
        parts.append(_word(*CASE_SENSITIVE[name]))
    if name == "Go":
        parts.append(_GO)
    return re.compile("|".join(parts))


COMPILED: dict[str, re.Pattern[str]] = {name: _compile(name) for name in SKILLS}
ALL_SKILLS: list[str] = list(SKILLS)


def extract_skills(*texts: str | None) -> list[str]:
    """Return the canonical skills mentioned in any of ``texts``, in dictionary order."""
    combined = "\n".join(t for t in texts if t)
    if not combined:
        return []
    return [name for name, regex in COMPILED.items() if regex.search(combined)]


def parse_user_skills(text: str) -> list[str]:
    """Turn free text typed by a user (a skill list or a short resume) into canonical skills.

    Users often type in lower case ("python, sql, r, go"), so comma- or
    line-separated items that exactly equal a skill name are accepted as well
    as anything the normal extractor finds.
    """
    if not text:
        return []
    found = set(extract_skills(text))
    by_lower = {name.lower(): name for name in ALL_SKILLS}
    for item in re.split(r"[,;\n|]+", text):
        name = by_lower.get(item.strip().lower())
        if name:
            found.add(name)
    return [name for name in ALL_SKILLS if name in found]


def normalise_skill_list(skills: Iterable[str]) -> list[str]:
    """Map a list of skill names onto canonical names, dropping unknown ones."""
    return parse_user_skills("\n".join(skills))
