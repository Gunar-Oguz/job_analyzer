"""Two small scikit-learn models trained on the live data in a few seconds.

* Salary model: TF-IDF on the title + one-hot seniority / category /
  location / employment type + multi-hot skills -> Ridge regression on
  log(annual USD salary).
* Category model: TF-IDF on title and excerpt -> logistic regression that
  predicts the job's main Himalayas category.

Both are evaluated on a held-out test split and report a naive baseline,
so the numbers in the UI can be compared with "just guess the median" or
"just guess the most common category". The salary model also reports 5-fold
cross-validation, because a single split of a few hundred rows is noisy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from core.etl import UNCATEGORIZED

RANDOM_STATE = 42
MIN_SALARY_ROWS = 40
MIN_CATEGORY_ROWS = 40
MIN_JOBS_PER_CLASS = 8


class NotEnoughDataError(ValueError):
    """Raised when there are too few rows to train and evaluate a model honestly."""


def _identity(tokens: list[str]) -> list[str]:
    return tokens


def primary_location(locations: list[str]) -> str:
    """First allowed country, or "Worldwide" when the job has no restriction."""
    return locations[0] if locations else "Worldwide"


# ---------------------------------------------------------------- salary model


def salary_features(df: pd.DataFrame) -> pd.DataFrame:
    """The feature table used by the salary model."""
    return pd.DataFrame(
        {
            "title": df["title"].fillna("").astype(str),
            "seniority": df["seniority"].fillna("Unknown").astype(str),
            "category": df["category"].fillna(UNCATEGORIZED).astype(str),
            "location": df["locations"].apply(primary_location),
            "employment_type": df["employment_type"].fillna("Unknown").astype(str),
            "skills": df["skills"].apply(list),
        },
        index=df.index,
    )


ONE_HOT_COLUMNS = ["seniority", "category", "location", "employment_type"]


def _salary_pipeline() -> Pipeline:
    features = ColumnTransformer(
        [
            ("title", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "title"),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3), ONE_HOT_COLUMNS),
            (
                "skills",
                CountVectorizer(analyzer=_identity, binary=True, lowercase=False),
                "skills",
            ),
        ]
    )
    # alpha=5 was the best of 1 / 3 / 5 / 10 / 20 in 5-fold CV on a 1,500-job sample.
    return Pipeline([("features", features), ("model", Ridge(alpha=5.0))])


@dataclass
class SalaryModel:
    """A fitted salary pipeline plus its honest test-set metrics."""

    pipeline: Pipeline
    metrics: dict
    # 10th / 90th percentile of test residuals in log space -> prediction range.
    residual_low: float
    residual_high: float
    options: dict = field(default_factory=dict)

    def predict(
        self,
        title: str,
        seniority: str,
        category: str,
        location: str = "United States",
        skills: list[str] | None = None,
        employment_type: str = "Full Time",
    ) -> dict:
        """Predict an annual USD salary and a typical (10th-90th percentile) range around it."""
        row = pd.DataFrame(
            [
                {
                    "title": title,
                    "seniority": seniority,
                    "category": category,
                    "location": location,
                    "employment_type": employment_type,
                    "skills": list(skills or []),
                }
            ]
        )
        log_pred = float(self.pipeline.predict(row)[0])
        return {
            "predicted": round(float(np.exp(log_pred)), -2),
            "low": round(float(np.exp(log_pred + self.residual_low)), -2),
            "high": round(float(np.exp(log_pred + self.residual_high)), -2),
        }


def train_salary_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE) -> SalaryModel:
    """Train the salary model on jobs with a usable annual USD salary."""
    data = df[df["salary_usd"].notna()]
    if len(data) < MIN_SALARY_ROWS:
        raise NotEnoughDataError(f"Only {len(data)} jobs have a USD salary; need at least {MIN_SALARY_ROWS}.")
    X = salary_features(data)
    y = np.log(data["salary_usd"].to_numpy())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipeline = _salary_pipeline().fit(X_train, y_train)
    pred_test = pipeline.predict(X_test)
    actual, predicted = np.exp(y_test), np.exp(pred_test)
    baseline = np.full_like(actual, np.exp(np.median(y_train)))
    residuals = y_test - pred_test
    cv_folds = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_pred = cross_val_predict(_salary_pipeline(), X, y, cv=cv_folds)

    metrics = {
        "rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae_usd": round(float(mean_absolute_error(actual, predicted)), 0),
        "r2": round(float(r2_score(actual, predicted)), 3),
        "r2_log": round(float(r2_score(y_test, pred_test)), 3),
        "baseline_mae_usd": round(float(mean_absolute_error(actual, baseline)), 0),
        "cv_mae_usd": round(float(mean_absolute_error(np.exp(y), np.exp(cv_pred))), 0),
        "cv_r2": round(float(r2_score(np.exp(y), np.exp(cv_pred))), 3),
        "median_salary_usd": round(float(np.median(data["salary_usd"])), 0),
    }
    # Refit on all rows so predictions use every example; metrics stay from the held-out split.
    pipeline = _salary_pipeline().fit(X, y)
    options = {
        "seniority": sorted(X["seniority"].unique()),
        "category": sorted(X["category"].unique()),
        "location": X["location"].value_counts().index.tolist(),
        "employment_type": X["employment_type"].value_counts().index.tolist(),
    }
    return SalaryModel(
        pipeline=pipeline,
        metrics=metrics,
        residual_low=float(np.quantile(residuals, 0.1)),
        residual_high=float(np.quantile(residuals, 0.9)),
        options=options,
    )


# -------------------------------------------------------------- category model


def _category_pipeline(random_state: int) -> Pipeline:
    features = ColumnTransformer(
        [
            ("title", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True), "title"),
            ("text", TfidfVectorizer(min_df=2, max_features=20_000, sublinear_tf=True, stop_words="english"), "text"),
        ]
    )
    model = LogisticRegression(C=5.0, max_iter=2000, random_state=random_state)
    return Pipeline([("features", features), ("model", model)])


def category_features(titles: pd.Series, texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"title": titles.fillna("").astype(str), "text": texts.fillna("").astype(str)})


@dataclass
class CategoryModel:
    """A fitted category classifier plus its honest test-set metrics."""

    pipeline: Pipeline
    metrics: dict
    classes: list[str]

    def predict(self, title: str, description: str = "", top_k: int = 3) -> list[dict]:
        """Return the ``top_k`` most likely categories with probabilities."""
        row = category_features(pd.Series([title]), pd.Series([description]))
        probabilities = self.pipeline.predict_proba(row)[0]
        order = np.argsort(probabilities)[::-1][:top_k]
        labels = self.pipeline.classes_
        return [{"category": str(labels[i]), "probability": round(float(probabilities[i]), 3)} for i in order]


def train_category_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE) -> CategoryModel:
    """Train the classifier on categories that have at least ``MIN_JOBS_PER_CLASS`` jobs."""
    counts = df["category"].value_counts()
    keep = counts[(counts >= MIN_JOBS_PER_CLASS) & (counts.index != UNCATEGORIZED)].index
    data = df[df["category"].isin(keep)]
    if len(data) < MIN_CATEGORY_ROWS or len(keep) < 2:
        raise NotEnoughDataError(f"Only {len(data)} labelled jobs in {len(keep)} categories; not enough to train.")
    X = category_features(data["title"], data["excerpt"].fillna("") + " " + data["description"].fillna("").str[:2000])
    y = data["category"].to_numpy()
    all_labels = data["parent_categories"].tolist()
    idx_train, idx_test = train_test_split(
        np.arange(len(data)), test_size=test_size, random_state=random_state, stratify=y
    )
    pipeline = _category_pipeline(random_state).fit(X.iloc[idx_train], y[idx_train])

    proba = pipeline.predict_proba(X.iloc[idx_test])
    labels = pipeline.classes_
    top1 = labels[proba.argmax(axis=1)]
    top3 = [set(labels[np.argsort(row)[::-1][:3]]) for row in proba]
    y_test = y[idx_test]
    majority = pd.Series(y[idx_train]).mode()[0]
    any_label = [pred in all_labels[i] for pred, i in zip(top1, idx_test, strict=True)]

    metrics = {
        "rows": int(len(data)),
        "train_rows": int(len(idx_train)),
        "test_rows": int(len(idx_test)),
        "classes": int(len(keep)),
        "accuracy": round(float(accuracy_score(y_test, top1)), 3),
        "top3_accuracy": round(float(np.mean([t in s for t, s in zip(y_test, top3, strict=True)])), 3),
        "any_label_accuracy": round(float(np.mean(any_label)), 3),
        "macro_f1": round(float(f1_score(y_test, top1, average="macro")), 3),
        "baseline_accuracy": round(float(np.mean(y_test == majority)), 3),
        "dropped_rare_or_other": int(len(df) - len(data)),
    }
    pipeline = _category_pipeline(random_state).fit(X, y)
    return CategoryModel(pipeline=pipeline, metrics=metrics, classes=sorted(keep))


@dataclass
class Models:
    """Both models (either may be None) plus the reason a model is missing."""

    salary: SalaryModel | None
    category: CategoryModel | None
    problems: dict = field(default_factory=dict)


def train_all(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> Models:
    """Train both models; a model that lacks data is None and its reason is kept."""
    problems = {}
    salary = category = None
    try:
        salary = train_salary_model(df, random_state=random_state)
    except NotEnoughDataError as error:
        problems["salary"] = str(error)
    try:
        category = train_category_model(df, random_state=random_state)
    except NotEnoughDataError as error:
        problems["category"] = str(error)
    return Models(salary=salary, category=category, problems=problems)
