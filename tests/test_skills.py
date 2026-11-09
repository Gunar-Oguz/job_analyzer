import pytest

from core.skills import ALL_SKILLS, extract_skills, parse_user_skills


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("We use Python, Go and Rust", ["Python", "Go", "Rust"]),
        ("Experience with R or Python", ["Python", "R"]),
        ("C++ and C#, .NET", ["C++", "C#"]),
        ("Node.js/React", ["React", "Node.js"]),
        ("PostgreSQL, k8s and CI/CD", ["PostgreSQL", "Kubernetes", "CI/CD"]),
        ("ML engineer who knows machine learning", ["Machine Learning"]),
        ("Build AI agents", ["AI"]),
    ],
)
def test_extracts_known_skills(text, expected):
    assert extract_skills(text) == [s for s in ALL_SKILLS if s in expected]


@pytest.mark.parametrize(
    "text",
    [
        "for the rest of the team",  # "r" inside "for", "rest" as a word
        "maintain and train the chain",  # "ai" inside words
        "our R&D department",  # R in R&D is not the language
        "Let's go further together",  # "go" as a verb
        "react to issues and excel at work",  # ordinary verbs
        "e.g. a remote role",
        "javascripting gopher spreadsheets",  # skill names inside longer words
    ],
)
def test_word_boundaries_prevent_false_matches(text):
    assert extract_skills(text) == []


def test_extract_handles_empty_and_multiple_texts():
    assert extract_skills(None, "") == []
    assert extract_skills("Python", "SQL") == ["Python", "SQL"]


def test_parse_user_skills_accepts_lowercase_list_items():
    assert parse_user_skills("python, sql, r, go\nexcel; ai") == ["Python", "SQL", "R", "Go", "Excel", "AI"]


def test_parse_user_skills_from_resume_text():
    resume = "Analyst with 3 years of SQL and Tableau, some Python (pandas) and A/B testing."
    assert parse_user_skills(resume) == ["Python", "SQL", "Pandas", "Tableau", "A/B Testing"]
