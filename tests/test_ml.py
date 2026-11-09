import pytest

from core import ml


def test_salary_model_learns_the_synthetic_signal(models):
    model = models.salary
    m = model.metrics
    assert m["rows"] == m["train_rows"] + m["test_rows"]
    assert m["mae_usd"] < m["baseline_mae_usd"]  # better than guessing the median
    assert m["r2"] > 0.5
    senior = model.predict("Senior Data Scientist", "Senior", "Data Science", "United States", ["Python", "SQL"])
    junior = model.predict("Entry Data Analyst", "Entry-level", "Data Science", "United States", ["Excel"])
    assert senior["low"] <= senior["predicted"] <= senior["high"]
    assert senior["predicted"] > junior["predicted"]


def test_training_is_deterministic(jobs, models):
    again = ml.train_salary_model(jobs)
    assert again.metrics == models.salary.metrics
    assert ml.train_category_model(jobs).metrics == models.category.metrics


def test_category_model(models):
    model = models.category
    assert model.metrics["accuracy"] > model.metrics["baseline_accuracy"]
    top = model.predict("Senior DevOps Engineer", "Kubernetes and Terraform on AWS")
    assert len(top) == 3
    assert top[0]["category"] == "Developer"
    assert top[0]["probability"] >= top[1]["probability"] >= top[2]["probability"]
    assert set(model.classes) >= {"Developer", "Sales", "Data Science"}


def test_not_enough_data_is_reported_not_raised(jobs):
    small = ml.train_all(jobs.head(10))
    assert small.salary is None and small.category is None
    assert "salary" in small.problems and "category" in small.problems
    with pytest.raises(ml.NotEnoughDataError):
        ml.train_salary_model(jobs.head(10))
