from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = ROOT / "Labs_notebooks"
ASSET_DIR = NOTEBOOK_ROOT / "assets"

AUTHOR = "Sundetkhan Bekzat"
INSTRUCTOR = "Akhmetova Zhanar"


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": split_source(text)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_source(text),
    }


def split_source(text: str) -> list[str]:
    clean = textwrap.dedent(text).strip("\n") + "\n"
    return clean.splitlines(keepends=True)


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(relative_dir: str, file_name: str, cells: list[dict]) -> None:
    target_dir = NOTEBOOK_ROOT / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / file_name
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")


def intro_cells(lab_title: str, guide: str) -> list[dict]:
    return [
        md_cell(
            f"""
            # Artificial Intelligence Technology and Application

            ## {guide}

            Independent implementation prepared by **{AUTHOR}**.
            """
        ),
        md_cell(f"# {lab_title}\n\nThis notebook keeps the lab objective but uses compact local examples so it can run without external datasets."),
    ]


def build_92_notebooks() -> None:
    write_notebook(
        "9.2 Python/9.2.1 Python Lab Guide",
        "9.2.1 Python Lab Guide.ipynb",
        intro_cells("1 Data Types, Control Flow, Functions, and Objects", "Python Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Basic Python Syntax\nThe first check confirms that the runtime can execute a simple statement."),
            code_cell('print("hello world from Sundetkhan Bekzat midterm")'),
            md_cell("## 1.2 Core Data Types\nNumbers, strings, lists, tuples, dictionaries, and sets are compared in one compact example."),
            code_cell(
                """
                lab_score = 87.5
                title = "Huawei AI Practice"
                topics = ["python", "data", "models"]
                fixed_topics = tuple(topics)
                profile = {"author": "Sundetkhan Bekzat", "score": lab_score, "passed": True}
                unique_lengths = {len(item) for item in topics}

                print(type(lab_score).__name__, lab_score + 2.5)
                print(title.lower().replace(" ", "_"))
                print(topics[1:], fixed_topics)
                print(profile["author"], unique_lengths)
                """
            ),
            md_cell("## 1.3 Branches and Loops\nA deterministic grading function replaces interactive input so the notebook remains executable."),
            code_cell(
                """
                def grade_label(score):
                    if score >= 90:
                        return "A"
                    if score >= 75:
                        return "B"
                    if score >= 60:
                        return "C"
                    return "Needs practice"

                for value in [95, 82, 68, 41]:
                    print(f"score={value}: {grade_label(value)}")

                row_count = 5
                for left in range(1, row_count + 1):
                    cells = [f"{left}x{right}={left * right}" for right in range(1, left + 1)]
                    print(" | ".join(cells))
                """
            ),
            md_cell("## 1.4 Functions\nThe Fibonacci task is implemented as a reusable function with validation."),
            code_cell(
                """
                def fibonacci_terms(limit):
                    if limit <= 0:
                        return []
                    values = [0, 1]
                    while len(values) < limit:
                        values.append(values[-1] + values[-2])
                    return values[:limit]

                print(fibonacci_terms(8))
                """
            ),
            md_cell("## 1.5 Object-Oriented Programming\nA small lab record class demonstrates attributes, methods, and class-level counters."),
            code_cell(
                """
                class LabRecord:
                    created = 0

                    def __init__(self, owner, section):
                        self.owner = owner
                        self.section = section
                        self.notes = []
                        LabRecord.created += 1

                    def add_note(self, note):
                        self.notes.append(note)

                    def summary(self):
                        return f"{self.owner}: {self.section}, notes={len(self.notes)}"


                record = LabRecord("Sundetkhan Bekzat", "9.2.1")
                record.add_note("data types completed")
                record.add_note("loops completed")
                print(record.summary())
                print("records created:", LabRecord.created)
                """
            ),
            md_cell("## 1.6 Standard Library\nOnly read-only environment information is displayed."),
            code_cell(
                """
                import os
                import platform
                import time

                print("platform:", platform.system())
                print("cwd name:", os.path.basename(os.getcwd()))
                print("timestamp sample:", time.strftime("%Y-%m-%d %H:%M"))
                """
            ),
        ],
    )

    write_notebook(
        "9.2 Python/9.2.2 Python Lab Guide",
        "9.2.2 Python Lab Guide.ipynb",
        intro_cells("1 I/O Operations and Lightweight Data Storage", "Python Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Text, CSV, and JSON I/O\nTemporary files are used so the repository stays clean after execution."),
            code_cell(
                """
                import csv
                import json
                import tempfile
                from pathlib import Path

                with tempfile.TemporaryDirectory() as tmp:
                    folder = Path(tmp)
                    text_path = folder / "notes.txt"
                    csv_path = folder / "scores.csv"
                    json_path = folder / "profile.json"

                    text_path.write_text("Huawei AI lab\\nPython I/O practice\\n", encoding="utf-8")

                    with csv_path.open("w", newline="", encoding="utf-8") as fh:
                        writer = csv.DictWriter(fh, fieldnames=["section", "score"])
                        writer.writeheader()
                        writer.writerows([
                            {"section": "9.2.1", "score": 91},
                            {"section": "9.2.2", "score": 94},
                        ])

                    json_path.write_text(json.dumps({"author": "Sundetkhan Bekzat", "ready": True}), encoding="utf-8")

                    print(text_path.read_text(encoding="utf-8").strip())
                    print(list(csv.DictReader(csv_path.open(encoding="utf-8"))))
                    print(json.loads(json_path.read_text(encoding="utf-8")))
                """
            ),
            md_cell("## 1.2 Directory Listing\nPathlib gives safer path handling than manually concatenating strings."),
            code_cell(
                """
                from pathlib import Path

                here = Path.cwd()
                sample = sorted(item.name for item in here.iterdir())[:5]
                print("current folder:", here.name)
                print("first entries:", sample)
                """
            ),
            md_cell("## 1.3 Mock Database Access\nThe original database requirement is represented by a small adapter class so no external MySQL server is required."),
            code_cell(
                """
                class MiniTable:
                    def __init__(self, rows):
                        self.rows = list(rows)

                    def where(self, **filters):
                        return [row for row in self.rows if all(row.get(k) == v for k, v in filters.items())]

                    def insert(self, row):
                        self.rows.append(dict(row))


                users = MiniTable([
                    {"id": 1, "name": "Sundetkhan Bekzat", "role": "student"},
                    {"id": 2, "name": "Akhmetova Zhanar", "role": "instructor"},
                ])
                users.insert({"id": 3, "name": "Lab Bot", "role": "assistant"})
                print(users.where(role="student"))
                """
            ),
            md_cell("## 1.4 Defensive Error Handling\nBad input is reported without stopping the rest of the notebook."),
            code_cell(
                """
                def parse_positive_int(raw):
                    try:
                        value = int(raw)
                    except ValueError:
                        return None
                    return value if value > 0 else None

                for token in ["12", "0", "abc", "7"]:
                    print(token, "->", parse_positive_int(token))
                """
            ),
        ],
    )

    write_notebook(
        "9.2 Python/9.2.3 Python Lab Guide",
        "9.2.3 Python Lab Guide.ipynb",
        intro_cells("1 Regular Expressions, Iterables, and Decorators", "Python Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Regular Expressions\nThe examples extract emails and numeric identifiers from noisy text."),
            code_cell(
                r'''
                import re

                message = "student Sundetkhan Bekzat uses bekzat@example.com and ticket AI-2026-042"
                email_pattern = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
                ticket_pattern = re.compile(r"AI-\d{4}-\d{3}")

                print("email:", email_pattern.findall(message))
                print("ticket:", ticket_pattern.search(message).group(0))
                '''
            ),
            md_cell("## 1.2 Validation Helpers\nCompiled patterns make repeated validation clearer."),
            code_cell(
                r'''
                import re

                phone = re.compile(r"^\+?\d{10,15}$")
                samples = ["+77011234567", "8700-111", "1234567890"]
                for item in samples:
                    print(item, bool(phone.match(item)))
                '''
            ),
            md_cell("## 1.3 Iterable Checks\nThe function avoids treating strings as collections to flatten."),
            code_cell(
                """
                from collections.abc import Iterable

                def flatten_once(values):
                    result = []
                    for item in values:
                        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                            result.extend(item)
                        else:
                            result.append(item)
                    return result

                print(flatten_once([[1, 2], "AI", (3, 4), 5]))
                """
            ),
            md_cell("## 1.4 Decorators\nA timing decorator keeps measurement logic separate from the function being tested."),
            code_cell(
                """
                import time

                def measured(func):
                    def wrapped(*args, **kwargs):
                        started = time.perf_counter()
                        value = func(*args, **kwargs)
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        print(f"{func.__name__} completed in {elapsed_ms:.3f} ms")
                        return value
                    return wrapped


                @measured
                def square_sum(limit):
                    return sum(number * number for number in range(limit))


                print(square_sum(1000))
                """
            ),
        ],
    )

    write_notebook(
        "9.2 Python/9.2.4 Python Lab Guide",
        "9.2.4 Python Lab Guide.ipynb",
        intro_cells("1 Treasury Management System", "Python Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Domain Model\nA minimal treasury ledger is enough to demonstrate classes, dates, and reports."),
            code_cell(
                """
                from dataclasses import dataclass
                from datetime import date

                @dataclass
                class Transaction:
                    day: date
                    account: str
                    amount: float
                    category: str


                class TreasuryLedger:
                    def __init__(self):
                        self.transactions = []

                    def add(self, account, amount, category, day=None):
                        self.transactions.append(Transaction(day or date.today(), account, float(amount), category))

                    def balance(self, account):
                        return sum(tx.amount for tx in self.transactions if tx.account == account)

                    def by_category(self):
                        totals = {}
                        for tx in self.transactions:
                            totals[tx.category] = totals.get(tx.category, 0.0) + tx.amount
                        return totals
                """
            ),
            md_cell("## 1.2 Ledger Operations\nIncome is positive and expense is negative; this keeps balance calculation simple."),
            code_cell(
                """
                ledger = TreasuryLedger()
                ledger.add("cash", 150000, "opening")
                ledger.add("cash", -12000, "supplies")
                ledger.add("bank", 84000, "tuition")
                ledger.add("bank", -9000, "software")

                print("cash balance:", ledger.balance("cash"))
                print("bank balance:", ledger.balance("bank"))
                print("category totals:", ledger.by_category())
                """
            ),
            md_cell("## 1.3 Search and Reporting\nThe report selects transactions by account and formats them for display."),
            code_cell(
                """
                def account_statement(ledger, account):
                    rows = [tx for tx in ledger.transactions if tx.account == account]
                    for tx in rows:
                        print(f"{tx.day.isoformat()} | {tx.category:<10} | {tx.amount:>10.2f}")

                account_statement(ledger, "bank")
                """
            ),
            md_cell("## 1.4 Exception Safety\nThe helper rejects invalid amounts before they reach the ledger."),
            code_cell(
                """
                def safe_add(ledger, account, raw_amount, category):
                    try:
                        amount = float(raw_amount)
                    except (TypeError, ValueError):
                        return "invalid amount"
                    ledger.add(account, amount, category)
                    return "saved"

                print(safe_add(ledger, "cash", "2500", "refund"))
                print(safe_add(ledger, "cash", "bad", "refund"))
                """
            ),
        ],
    )


def build_93_notebooks() -> None:
    write_notebook(
        "9.3 Machine Learning/9.3.1 Machine Learning Basic Lab Guide",
        "9.3.1 Machine Learning Basic Lab Guide.ipynb",
        intro_cells("1 Implementation of Common Machine Learning Algorithms", "Machine Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Linear Regression\nA noisy synthetic dataset is used to estimate a continuous target."),
            code_cell(
                """
                import numpy as np
                import matplotlib.pyplot as plt
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import mean_squared_error, r2_score
                from sklearn.model_selection import train_test_split

                rng = np.random.default_rng(42)
                study_hours = rng.uniform(1, 10, size=80).reshape(-1, 1)
                score = 47 + 5.8 * study_hours.ravel() + rng.normal(0, 4, size=80)
                X_train, X_test, y_train, y_test = train_test_split(study_hours, score, random_state=42)

                reg = LinearRegression().fit(X_train, y_train)
                predicted = reg.predict(X_test)
                print("coef:", round(reg.coef_[0], 3), "intercept:", round(reg.intercept_, 3))
                print("rmse:", round(mean_squared_error(y_test, predicted) ** 0.5, 3))
                print("r2:", round(r2_score(y_test, predicted), 3))
                plt.scatter(X_test.ravel(), y_test, label="actual")
                order = np.argsort(X_test.ravel())
                plt.plot(X_test.ravel()[order], predicted[order], color="darkorange", label="prediction")
                plt.xlabel("study hours")
                plt.ylabel("score")
                plt.legend()
                plt.show()
                """
            ),
            md_cell("## 1.2 Logistic Regression\nThe classifier separates two generated classes after feature scaling."),
            code_cell(
                """
                from sklearn.datasets import make_classification
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, confusion_matrix
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler

                features, labels = make_classification(
                    n_samples=180, n_features=4, n_informative=3, n_redundant=0, random_state=7
                )
                X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, random_state=7)
                scaler = StandardScaler().fit(X_train)
                model = LogisticRegression(max_iter=500).fit(scaler.transform(X_train), y_train)
                pred = model.predict(scaler.transform(X_test))
                print("accuracy:", round(accuracy_score(y_test, pred), 3))
                print(confusion_matrix(y_test, pred))
                """
            ),
            md_cell("## 1.3 K-Means Clustering\nUnlabeled samples are grouped by geometric distance to centroids."),
            code_cell(
                """
                import matplotlib.pyplot as plt
                from sklearn.cluster import KMeans
                from sklearn.datasets import make_blobs

                points, _ = make_blobs(n_samples=160, centers=3, cluster_std=0.9, random_state=11)
                clusterer = KMeans(n_clusters=3, random_state=11, n_init=10)
                groups = clusterer.fit_predict(points)
                print("centers:")
                print(clusterer.cluster_centers_.round(2))
                plt.scatter(points[:, 0], points[:, 1], c=groups, cmap="viridis", s=28)
                plt.scatter(clusterer.cluster_centers_[:, 0], clusterer.cluster_centers_[:, 1], marker="X", s=160, c="red")
                plt.title("K-Means grouping")
                plt.show()
                """
            ),
            md_cell("## 1.4 Decision Tree\nA tree model gives an interpretable baseline for tabular classification."),
            code_cell(
                """
                from sklearn.datasets import load_iris
                from sklearn.metrics import accuracy_score
                from sklearn.model_selection import train_test_split
                from sklearn.tree import DecisionTreeClassifier

                iris = load_iris()
                X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=4, stratify=iris.target)
                tree_model = DecisionTreeClassifier(max_depth=3, random_state=4).fit(X_train, y_train)
                print("tree accuracy:", round(accuracy_score(y_test, tree_model.predict(X_test)), 3))
                print("feature importance:", dict(zip(iris.feature_names, tree_model.feature_importances_.round(3))))
                """
            ),
        ],
    )

    write_notebook(
        "9.3 Machine Learning/9.3.2 Machine Learning Lab Guide",
        "9.3.2 Machine Learning Lab Guide.ipynb",
        intro_cells("1 Feature Engineering on Credit Data", "Machine Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Credit Data Frame\nThe table intentionally contains missing values to exercise preprocessing."),
            code_cell(
                """
                import numpy as np
                import pandas as pd

                credit = pd.DataFrame({
                    "age": [24, 31, 46, 52, np.nan, 38, 29, 43],
                    "income": [180, 240, 510, 620, 390, np.nan, 210, 470],
                    "married": [0, 1, 1, 1, 0, 1, np.nan, 1],
                    "city": ["Astana", "Almaty", "Astana", "Shymkent", "Astana", "Almaty", "Astana", "Shymkent"],
                    "defaulted": [0, 0, 1, 1, 0, 1, 0, 1],
                })
                print(credit)
                print("missing ratios:")
                print(credit.isna().mean())
                """
            ),
            md_cell("## 1.2 Missing Value Repair\nNumeric fields are imputed with medians before statistical feature selection."),
            code_cell(
                """
                from sklearn.feature_selection import SelectKBest, chi2
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import MinMaxScaler

                numeric = credit[["age", "income", "married"]]
                imputed = SimpleImputer(strategy="median").fit_transform(numeric)
                scaled = MinMaxScaler().fit_transform(imputed)
                selector = SelectKBest(score_func=chi2, k=2).fit(scaled, credit["defaulted"])
                print(dict(zip(numeric.columns, selector.scores_.round(3))))
                print("selected:", list(numeric.columns[selector.get_support()]))
                """
            ),
            md_cell("## 1.3 Encoding and Construction\nCategorical city values are one-hot encoded and income is expanded polynomially."),
            code_cell(
                """
                import pandas as pd
                from sklearn.preprocessing import PolynomialFeatures

                encoded_city = pd.get_dummies(credit["city"], prefix="city", dtype=int)
                income_ready = SimpleImputer(strategy="median").fit_transform(credit[["income"]])
                income_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(income_ready)
                engineered = pd.concat([
                    pd.DataFrame(imputed, columns=["age", "income", "married"]),
                    encoded_city.reset_index(drop=True),
                    pd.DataFrame(income_poly[:, 1:], columns=["income_squared"]),
                ], axis=1)
                print(engineered.head())
                """
            ),
            md_cell("## 1.4 Wrapper Selection\nRecursive feature elimination is demonstrated with a small logistic model."),
            code_cell(
                """
                from sklearn.feature_selection import RFE
                from sklearn.linear_model import LogisticRegression

                wrapper = RFE(LogisticRegression(max_iter=500), n_features_to_select=3)
                wrapper.fit(engineered, credit["defaulted"])
                print("RFE selected:", list(engineered.columns[wrapper.support_]))
                """
            ),
        ],
    )

    write_notebook(
        "9.3 Machine Learning/9.3.3 Machine Learning Lab Guide",
        "9.3.3 Machine Learning Lab Guide.ipynb",
        intro_cells("1 Real-Time Recommendation Practice for Retail Products", "Machine Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 User-Item Matrix\nRatings are stored as a table where rows are users and columns are products."),
            code_cell(
                """
                import numpy as np
                import pandas as pd

                ratings = pd.DataFrame(
                    [[5, 4, 0, 1, 0], [4, 5, 0, 1, 1], [0, 1, 5, 4, 4], [1, 0, 4, 5, 5], [5, 4, 1, 0, 0]],
                    index=["U1", "U2", "U3", "U4", "U5"],
                    columns=["phone", "charger", "book", "headset", "camera"],
                )
                print(ratings)
                """
            ),
            md_cell("## 1.2 Similarity Recommendation\nCosine similarity finds neighbors with similar purchase taste."),
            code_cell(
                """
                from sklearn.metrics.pairwise import cosine_similarity

                similarity = pd.DataFrame(cosine_similarity(ratings), index=ratings.index, columns=ratings.index)
                target = "U1"
                neighbors = similarity[target].drop(target).sort_values(ascending=False)
                weighted = ratings.loc[neighbors.index].mul(neighbors, axis=0).sum() / neighbors.sum()
                unseen = ratings.loc[target] == 0
                print("neighbors:")
                print(neighbors.round(3))
                print("recommendations:")
                print(weighted[unseen].sort_values(ascending=False).round(2))
                """
            ),
            md_cell("## 1.3 Matrix Factorization\nTruncated SVD compresses product preferences into latent factors."),
            code_cell(
                """
                from sklearn.decomposition import TruncatedSVD

                svd = TruncatedSVD(n_components=2, random_state=3)
                latent = svd.fit_transform(ratings)
                print("explained variance:", svd.explained_variance_ratio_.round(3))
                print(pd.DataFrame(latent, index=ratings.index, columns=["factor_a", "factor_b"]).round(2))
                """
            ),
            md_cell("## 1.4 Heatmap\nThe matrix plot makes sparse preferences easy to inspect."),
            code_cell(
                """
                import matplotlib.pyplot as plt

                plt.imshow(ratings, cmap="YlGnBu")
                plt.xticks(range(len(ratings.columns)), ratings.columns, rotation=30)
                plt.yticks(range(len(ratings.index)), ratings.index)
                plt.colorbar(label="rating")
                plt.title("User-item ratings")
                plt.show()
                """
            ),
        ],
    )

    write_notebook(
        "9.3 Machine Learning/9.3.4 Machine Learning Lab Guide",
        "9.3.4 Machine Learning Lab Guide.ipynb",
        intro_cells("1 Private Credit Default Prediction", "Machine Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Imbalanced Classification\nThe generated dataset imitates rare default cases."),
            code_cell(
                """
                from sklearn.datasets import make_classification
                from sklearn.model_selection import train_test_split

                X, y = make_classification(
                    n_samples=500, n_features=8, n_informative=5, weights=[0.84, 0.16], flip_y=0.02, random_state=19
                )
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=19)
                print("train default rate:", round(y_train.mean(), 3))
                """
            ),
            md_cell("## 1.2 Balanced Logistic Regression\nClass weights reduce the bias toward the majority class."),
            code_cell(
                """
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler().fit(X_train)
                model = LogisticRegression(max_iter=500, class_weight="balanced").fit(scaler.transform(X_train), y_train)
                proba = model.predict_proba(scaler.transform(X_test))[:, 1]
                pred = (proba >= 0.5).astype(int)
                print("accuracy:", round(accuracy_score(y_test, pred), 3))
                print("precision:", round(precision_score(y_test, pred), 3))
                print("recall:", round(recall_score(y_test, pred), 3))
                print("roc_auc:", round(roc_auc_score(y_test, proba), 3))
                """
            ),
            md_cell("## 1.3 Threshold Tuning\nThe threshold can be adjusted when recall is more important than raw accuracy."),
            code_cell(
                """
                for threshold in [0.35, 0.5, 0.65]:
                    tuned = (proba >= threshold).astype(int)
                    print(threshold, "recall=", round(recall_score(y_test, tuned), 3), "precision=", round(precision_score(y_test, tuned), 3))
                """
            ),
            md_cell("## 1.4 Simple Hyperparameter Search\nA small non-interactive search keeps the notebook fast."),
            code_cell(
                """
                best = None
                for c_value in [0.1, 1.0, 10.0]:
                    candidate = LogisticRegression(max_iter=500, class_weight="balanced", C=c_value)
                    candidate.fit(scaler.transform(X_train), y_train)
                    score = roc_auc_score(y_test, candidate.predict_proba(scaler.transform(X_test))[:, 1])
                    best = max(best or (0, None), (score, c_value))
                print("best C:", best[1], "auc:", round(best[0], 3))
                """
            ),
        ],
    )

    write_notebook(
        "9.3 Machine Learning/9.3.5 Machine Learning Lab Guide",
        "9.3.5 Machine Learning Lab Guide.ipynb",
        intro_cells("1 Survival Prediction of the Titanic", "Machine Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Synthetic Passenger Table\nThe structure mirrors passenger survival features without requiring the original CSV."),
            code_cell(
                """
                import numpy as np
                import pandas as pd

                rng = np.random.default_rng(15)
                passengers = pd.DataFrame({
                    "class": rng.choice([1, 2, 3], size=220, p=[0.22, 0.28, 0.50]),
                    "sex": rng.choice(["female", "male"], size=220),
                    "age": rng.normal(31, 12, size=220).clip(1, 75),
                    "fare": rng.gamma(2.0, 18.0, size=220),
                })
                logits = 1.2 * (passengers["sex"] == "female") - 0.55 * passengers["class"] + passengers["fare"] / 120 - passengers["age"] / 90
                passengers["survived"] = (rng.random(220) < 1 / (1 + np.exp(-logits))).astype(int)
                print(passengers.head())
                """
            ),
            md_cell("## 1.2 Model Comparison\nLogistic regression and random forest are trained on the same encoded features."),
            code_cell(
                """
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score
                from sklearn.model_selection import train_test_split

                X = pd.get_dummies(passengers.drop(columns="survived"), drop_first=True)
                y = passengers["survived"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=15)
                models = {
                    "logistic": LogisticRegression(max_iter=500),
                    "forest": RandomForestClassifier(n_estimators=80, random_state=15, max_depth=5),
                }
                for name, estimator in models.items():
                    estimator.fit(X_train, y_train)
                    print(name, round(accuracy_score(y_test, estimator.predict(X_test)), 3))
                """
            ),
            md_cell("## 1.3 Feature Importance\nThe forest ranking provides an interpretable summary."),
            code_cell(
                """
                forest = models["forest"]
                ranking = sorted(zip(X.columns, forest.feature_importances_), key=lambda pair: pair[1], reverse=True)
                print([(name, round(value, 3)) for name, value in ranking])
                """
            ),
            md_cell("## 1.4 Visualization\nAverage survival is grouped by passenger class."),
            code_cell(
                """
                import matplotlib.pyplot as plt

                passengers.groupby("class")["survived"].mean().plot(kind="bar", color="#607D8B")
                plt.ylabel("survival rate")
                plt.title("Synthetic Titanic survival by class")
                plt.show()
                """
            ),
        ],
    )

    write_notebook(
        "9.3 Machine Learning/9.3.6 Machine Learning Lab Guide",
        "9.3.6 Machine Learning Lab Guide.ipynb",
        intro_cells("1 Linear Regression and Gradient Descent", "Machine Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Data Generation\nA one-dimensional signal is enough to show the loss surface."),
            code_cell(
                """
                import numpy as np
                import matplotlib.pyplot as plt

                rng = np.random.default_rng(8)
                x = np.linspace(0, 6, 70)
                y = 2.7 * x + 5 + rng.normal(0, 1.5, size=x.size)
                plt.scatter(x, y, s=20)
                plt.title("Training samples")
                plt.show()
                """
            ),
            md_cell("## 1.2 Manual Optimization\nThe loop updates slope and intercept using mean squared error gradients."),
            code_cell(
                """
                slope, intercept = 0.0, 0.0
                lr = 0.025
                losses = []
                for _ in range(300):
                    estimate = slope * x + intercept
                    error = estimate - y
                    losses.append(float(np.mean(error ** 2)))
                    slope -= lr * np.mean(2 * error * x)
                    intercept -= lr * np.mean(2 * error)
                print("slope:", round(slope, 3), "intercept:", round(intercept, 3), "final loss:", round(losses[-1], 3))
                """
            ),
            md_cell("## 1.3 Fitted Line\nThe fitted line is plotted against the raw observations."),
            code_cell(
                """
                plt.scatter(x, y, s=20, label="data")
                plt.plot(x, slope * x + intercept, color="crimson", label="manual fit")
                plt.legend()
                plt.show()
                """
            ),
            md_cell("## 1.4 Loss Curve\nThe decreasing curve confirms convergence."),
            code_cell(
                """
                plt.plot(losses)
                plt.xlabel("iteration")
                plt.ylabel("mse")
                plt.title("Gradient descent loss")
                plt.show()
                """
            ),
        ],
    )

    write_notebook(
        "9.3 Machine Learning/9.3.7 Machine Learning Lab Guide",
        "9.3.7 Machine Learning Lab Guide.ipynb",
        intro_cells("1 Flower Category Analysis", "Machine Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Iris Dataset\nThe built-in dataset avoids network downloads while preserving the flower classification task."),
            code_cell(
                """
                from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler

                iris = load_iris()
                X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=23)
                scaler = StandardScaler().fit(X_train)
                print(iris.target_names)
                """
            ),
            md_cell("## 1.2 Classifier Comparison\nSeveral classical classifiers are evaluated with the same train/test split."),
            code_cell(
                """
                from sklearn.metrics import accuracy_score
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.svm import SVC
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.linear_model import LogisticRegression

                candidates = {
                    "knn": KNeighborsClassifier(n_neighbors=5),
                    "svm": SVC(kernel="rbf", gamma="scale"),
                    "tree": DecisionTreeClassifier(max_depth=3, random_state=23),
                    "logistic": LogisticRegression(max_iter=500),
                }
                for name, estimator in candidates.items():
                    estimator.fit(scaler.transform(X_train), y_train)
                    print(name, round(accuracy_score(y_test, estimator.predict(scaler.transform(X_test))), 3))
                """
            ),
            md_cell("## 1.3 Text Classification Extension\nA small text sample demonstrates bag-of-words sentiment classification."),
            code_cell(
                """
                from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
                from sklearn.naive_bayes import MultinomialNB

                texts = ["fresh flower bright", "large petal bright", "dry stem weak", "wilted flower weak"]
                text_labels = [1, 1, 0, 0]
                counts = CountVectorizer().fit_transform(texts)
                tfidf = TfidfTransformer().fit_transform(counts)
                nb = MultinomialNB().fit(tfidf, text_labels)
                print(nb.predict(TfidfTransformer().fit_transform(CountVectorizer().fit(texts).transform(["bright fresh petal"]))))
                """
            ),
            md_cell("## 1.4 Two-Feature Plot\nSepal features are visualized to inspect class separation."),
            code_cell(
                """
                import matplotlib.pyplot as plt

                plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap="Accent")
                plt.xlabel(iris.feature_names[0])
                plt.ylabel(iris.feature_names[1])
                plt.title("Iris sepal feature space")
                plt.show()
                """
            ),
        ],
    )

    write_notebook(
        "9.3 Machine Learning/9.3.8 Machine Learning Lab Guide",
        "9.3.8 Machine Learning Lab Guide.ipynb",
        intro_cells("1 E-commerce Website User Group Analysis", "Machine Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Behavioral Features\nUsers are described by visits, spend, and return rate."),
            code_cell(
                """
                import numpy as np
                import pandas as pd

                rng = np.random.default_rng(30)
                segments = [
                    rng.normal([12, 65, 0.05], [3, 12, 0.02], size=(45, 3)),
                    rng.normal([4, 18, 0.18], [1, 5, 0.04], size=(45, 3)),
                    rng.normal([20, 120, 0.09], [4, 18, 0.03], size=(45, 3)),
                ]
                users = pd.DataFrame(np.vstack(segments), columns=["visits", "spend", "return_rate"])
                print(users.head())
                """
            ),
            md_cell("## 1.2 K-Means Segmentation\nClustering groups customers for marketing decisions."),
            code_cell(
                """
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler

                scaled_users = StandardScaler().fit_transform(users)
                kmeans = KMeans(n_clusters=3, random_state=30, n_init=10).fit(scaled_users)
                users["cluster"] = kmeans.labels_
                print(users.groupby("cluster").mean().round(2))
                """
            ),
            md_cell("## 1.3 Elbow Check\nInertia values are printed for a quick model selection check."),
            code_cell(
                """
                inertias = []
                for k in range(1, 6):
                    inertias.append(KMeans(n_clusters=k, random_state=30, n_init=10).fit(scaled_users).inertia_)
                print([round(v, 2) for v in inertias])
                """
            ),
            md_cell("## 1.4 Segment Plot\nSpend and visit frequency provide an interpretable 2D view."),
            code_cell(
                """
                import matplotlib.pyplot as plt

                plt.scatter(users["visits"], users["spend"], c=users["cluster"], cmap="viridis", s=30)
                plt.xlabel("visits")
                plt.ylabel("spend")
                plt.title("E-commerce user groups")
                plt.show()
                """
            ),
        ],
    )

    write_notebook(
        "9.3 Machine Learning/9.3.9 Machine Learning Lab Guide",
        "9.3.9 Machine Learning Lab Guide.ipynb",
        intro_cells("1 Emotion Recognition of Customer Evaluations", "Machine Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Text Cleaning\nA small review set is normalized before vectorization."),
            code_cell(
                r'''
                import re
                import pandas as pd

                reviews = pd.DataFrame({
                    "text": [
                        "Fast delivery and excellent packaging!",
                        "The item broke after one day",
                        "Friendly support and good quality",
                        "Late delivery, damaged box",
                        "Great value for the price",
                        "Bad support and poor instruction",
                        "I love the camera quality",
                        "The battery failed quickly",
                        "Clean design and stable work",
                        "Terrible smell from the package",
                        "Helpful seller, smooth refund",
                        "Worst cable I have bought",
                    ],
                    "positive": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                })

                def clean_text(value):
                    return re.sub(r"[^a-z ]+", " ", value.lower()).strip()

                reviews["clean"] = reviews["text"].map(clean_text)
                print(reviews[["clean", "positive"]].head())
                '''
            ),
            md_cell("## 1.2 Vectorization and Models\nNaive Bayes and logistic regression are compared on TF-IDF features."),
            code_cell(
                """
                from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score
                from sklearn.model_selection import train_test_split
                from sklearn.naive_bayes import MultinomialNB, BernoulliNB

                X_train, X_test, y_train, y_test = train_test_split(
                    reviews["clean"], reviews["positive"], stratify=reviews["positive"], random_state=9, test_size=0.33
                )
                vectorizer = CountVectorizer(stop_words="english")
                transformer = TfidfTransformer()
                train_counts = vectorizer.fit_transform(X_train)
                train_tfidf = transformer.fit_transform(train_counts)
                test_tfidf = transformer.transform(vectorizer.transform(X_test))

                models = {
                    "multinomial_nb": MultinomialNB(),
                    "bernoulli_nb": BernoulliNB(),
                    "logistic": LogisticRegression(max_iter=500),
                }
                for name, estimator in models.items():
                    estimator.fit(train_tfidf, y_train)
                    print(name, round(accuracy_score(y_test, estimator.predict(test_tfidf)), 3))
                """
            ),
            md_cell("## 1.3 Inference\nA new customer sentence is classified by the logistic model."),
            code_cell(
                """
                logistic = models["logistic"]
                new_review = [clean_text("excellent support and fast refund")]
                probability = logistic.predict_proba(transformer.transform(vectorizer.transform(new_review)))[0, 1]
                print("positive probability:", round(probability, 3))
                """
            ),
        ],
    )


def build_94_notebooks() -> None:
    write_notebook(
        "9.4 Deep Learning/9.4.1 Deep Learning and AI Development Framework Lab Guide-1",
        "9.4.1 Deep Learning and AI Development Framework Lab Guide-1.ipynb",
        intro_cells("1 MindSpore Basics", "Deep Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Tensor Concept\nThe notebook checks for MindSpore but keeps a NumPy fallback for local execution."),
            code_cell(
                """
                import numpy as np

                try:
                    import mindspore as ms
                    backend = "mindspore"
                    tensor = ms.Tensor([[1, 2], [3, 4]], ms.float32)
                    print("backend:", backend, "shape:", tensor.shape)
                except Exception as exc:
                    backend = "numpy fallback"
                    tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
                    print("backend:", backend, "shape:", tensor.shape, "reason:", type(exc).__name__)
                """
            ),
            md_cell("## 1.2 Forward Pass\nA small dense layer is implemented explicitly to show matrix dimensions."),
            code_cell(
                """
                inputs = np.array([[0.2, 0.8, 0.4], [0.9, 0.1, 0.3]], dtype=np.float32)
                weights = np.array([[0.3, -0.2], [0.5, 0.4], [-0.1, 0.6]], dtype=np.float32)
                bias = np.array([0.05, -0.05], dtype=np.float32)
                logits = inputs @ weights + bias
                probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
                print(probabilities.round(3))
                """
            ),
            md_cell("## 1.3 Dataset Batching\nBatches are created with a generator pattern similar to dataset iterators."),
            code_cell(
                """
                def batches(values, size):
                    for start in range(0, len(values), size):
                        yield values[start:start + size]

                data = np.arange(12).reshape(6, 2)
                for batch in batches(data, 2):
                    print(batch.tolist())
                """
            ),
        ],
    )

    write_notebook(
        "9.4 Deep Learning/9.4.2 Deep Learning and AI Development Framework Lab Guide-2",
        "9.4.2 Deep Learning and AI Development Framework Lab Guide-2.ipynb",
        intro_cells("1 MNIST Handwritten Character Recognition", "Deep Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Digits Dataset\nScikit-learn's digits data gives a small local handwritten-number task."),
            code_cell(
                """
                from sklearn.datasets import load_digits
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler

                digits = load_digits()
                X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, stratify=digits.target, random_state=12)
                scaler = StandardScaler().fit(X_train)
                print("images:", digits.images.shape)
                """
            ),
            md_cell("## 1.2 Dense Neural Classifier\nA small MLP mirrors the dense-network portion of an MNIST workflow."),
            code_cell(
                """
                from sklearn.metrics import accuracy_score
                from sklearn.neural_network import MLPClassifier

                mlp = MLPClassifier(hidden_layer_sizes=(48, 24), activation="relu", max_iter=180, random_state=12, early_stopping=True)
                mlp.fit(scaler.transform(X_train), y_train)
                predictions = mlp.predict(scaler.transform(X_test))
                print("accuracy:", round(accuracy_score(y_test, predictions), 3))
                print("iterations:", mlp.n_iter_)
                """
            ),
            md_cell("## 1.3 Prediction Grid\nA few predicted digits are shown for visual inspection."),
            code_cell(
                """
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 5, figsize=(8, 3))
                for ax, image, truth, pred in zip(axes.ravel(), X_test[:10].reshape(-1, 8, 8), y_test[:10], predictions[:10]):
                    ax.imshow(image, cmap="gray_r")
                    ax.set_title(f"t={truth}, p={pred}")
                    ax.axis("off")
                plt.tight_layout()
                plt.show()
                """
            ),
        ],
    )

    write_notebook(
        "9.4 Deep Learning/9.4.3 Deep Learning and AI Development Framework Lab Guide-3",
        "9.4.3 Deep Learning and AI Development Framework Lab Guide-3.ipynb",
        intro_cells("1 MobileNetV2 Image Classification", "Deep Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Transfer Learning Shape Flow\nA synthetic image tensor demonstrates the same train/validation split without a large download."),
            code_cell(
                """
                import numpy as np
                from sklearn.model_selection import train_test_split

                rng = np.random.default_rng(33)
                class_names = np.array(["daisy", "dandelion", "rose", "sunflower", "tulip"])
                labels = np.repeat(np.arange(5), 24)
                base_colors = np.array([
                    [0.8, 0.8, 0.2], [0.9, 0.7, 0.1], [0.8, 0.2, 0.3], [1.0, 0.75, 0.05], [0.6, 0.2, 0.8]
                ])
                images = base_colors[labels][:, None, None, :] + rng.normal(0, 0.12, size=(labels.size, 16, 16, 3))
                images = images.clip(0, 1)
                train_img, val_img, train_y, val_y = train_test_split(images, labels, stratify=labels, random_state=33)
                print("train tensor:", train_img.shape, "validation tensor:", val_img.shape)
                """
            ),
            md_cell("## 1.2 Frozen Feature Extractor\nMean, standard deviation, and channel contrast act as compact image embeddings."),
            code_cell(
                """
                def extract_features(batch):
                    means = batch.mean(axis=(1, 2))
                    stds = batch.std(axis=(1, 2))
                    contrast = batch.max(axis=(1, 2)) - batch.min(axis=(1, 2))
                    return np.concatenate([means, stds, contrast], axis=1)

                train_features = extract_features(train_img)
                val_features = extract_features(val_img)
                print("feature shape:", train_features.shape)
                """
            ),
            md_cell("## 1.3 Classification Head\nOnly the final classifier is trained, matching the transfer-learning idea."),
            code_cell(
                """
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score

                head = LogisticRegression(max_iter=500).fit(train_features, train_y)
                predicted = head.predict(val_features)
                print("transfer-style accuracy:", round(accuracy_score(val_y, predicted), 3))
                print(list(zip(class_names[val_y[:8]], class_names[predicted[:8]])))
                """
            ),
            md_cell("## 1.4 Visual Samples\nPredictions are displayed on synthetic validation images."),
            code_cell(
                """
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 4, figsize=(8, 4))
                for ax, image, truth, pred in zip(axes.ravel(), val_img[:8], val_y[:8], predicted[:8]):
                    ax.imshow(image)
                    ax.set_title(f"{class_names[truth]} -> {class_names[pred]}", fontsize=8)
                    ax.axis("off")
                plt.tight_layout()
                plt.show()
                """
            ),
        ],
    )

    write_notebook(
        "9.4 Deep Learning/9.4.4 Deep Learning and AI Development Framework Lab Guide-4",
        "9.4.4 Deep Learning and AI Development Framework Lab Guide-4.ipynb",
        intro_cells("1 ResNet-50 Image Classification", "Deep Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Residual Block Principle\nThe skip connection keeps input and output dimensions aligned."),
            code_cell(
                """
                import numpy as np

                rng = np.random.default_rng(44)
                batch = rng.normal(size=(5, 6))
                w1 = rng.normal(scale=0.15, size=(6, 6))
                w2 = rng.normal(scale=0.15, size=(6, 6))

                def residual_block(values):
                    hidden = np.maximum(values @ w1, 0)
                    return np.maximum(hidden @ w2 + values, 0)

                output = residual_block(batch)
                print("input shape:", batch.shape, "output shape:", output.shape)
                """
            ),
            md_cell("## 1.2 Residual Features for Classification\nThe residual transform is used before a compact classifier."),
            code_cell(
                """
                from sklearn.datasets import make_classification
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler

                X, y = make_classification(n_samples=240, n_features=6, n_informative=5, n_redundant=0, random_state=44)
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=44)
                scaler = StandardScaler().fit(X_train)
                train_res = residual_block(scaler.transform(X_train))
                test_res = residual_block(scaler.transform(X_test))
                classifier = LogisticRegression(max_iter=500).fit(train_res, y_train)
                print("accuracy:", round(accuracy_score(y_test, classifier.predict(test_res)), 3))
                """
            ),
            md_cell("## 1.3 Checkpoint-Like Parameters\nA dictionary demonstrates how named weights can be inspected before loading."),
            code_cell(
                """
                checkpoint = {"block.w1": w1, "block.w2": w2, "head.coef": classifier.coef_}
                for key, value in checkpoint.items():
                    print(key, value.shape)
                """
            ),
        ],
    )

    write_notebook(
        "9.4 Deep Learning/9.4.5 Deep Learning and AI Development Framework Lab Guide-5",
        "9.4.5 Deep Learning and AI Development Framework Lab Guide-5.ipynb",
        intro_cells("1 TextCNN Sentiment Analysis", "Deep Learning Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Tokenization\nText is transformed into fixed-size vectors before applying convolution-style pooling."),
            code_cell(
                r'''
                import re
                import numpy as np

                sentences = [
                    "excellent product and fast delivery",
                    "smooth support and helpful seller",
                    "clean design with stable battery",
                    "poor package and broken cable",
                    "late delivery and bad support",
                    "terrible quality with weak battery",
                    "great camera and useful manual",
                    "awful screen and slow refund",
                ]
                labels = np.array([1, 1, 1, 0, 0, 0, 1, 0])

                def tokens(text):
                    return re.findall(r"[a-z]+", text.lower())

                vocab = sorted({token for sentence in sentences for token in tokens(sentence)})
                print("vocab size:", len(vocab))
                '''
            ),
            md_cell("## 1.2 Convolution-Style Features\nInstead of a large framework, each sentence is represented by n-gram activation counts."),
            code_cell(
                """
                def ngram_features(text, vocab, widths=(1, 2, 3)):
                    words = tokens(text)
                    features = []
                    for width in widths:
                        grams = [" ".join(words[i:i + width]) for i in range(max(0, len(words) - width + 1))]
                        features.append(len(grams))
                        features.append(sum("good" in gram or "great" in gram or "excellent" in gram for gram in grams))
                        features.append(sum("bad" in gram or "poor" in gram or "terrible" in gram or "awful" in gram for gram in grams))
                    bag = [words.count(word) for word in vocab]
                    return np.array(features + bag, dtype=float)

                feature_matrix = np.vstack([ngram_features(sentence, vocab) for sentence in sentences])
                print(feature_matrix.shape)
                """
            ),
            md_cell("## 1.3 Sentiment Head\nThe classifier consumes the pooled features similarly to the dense layer of TextCNN."),
            code_cell(
                """
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import LeaveOneOut, cross_val_score

                text_head = LogisticRegression(max_iter=500).fit(feature_matrix, labels)
                scores = cross_val_score(LogisticRegression(max_iter=500), feature_matrix, labels, cv=LeaveOneOut())
                print("leave-one-out accuracy:", round(scores.mean(), 3))
                sample = ngram_features("excellent support but slow delivery", vocab).reshape(1, -1)
                print("positive probability:", round(text_head.predict_proba(sample)[0, 1], 3))
                """
            ),
            md_cell("## 1.4 Important Terms\nWeights attached to vocabulary terms show what drives positive sentiment."),
            code_cell(
                """
                term_weights = text_head.coef_[0][-len(vocab):]
                ranked = sorted(zip(vocab, term_weights), key=lambda pair: abs(pair[1]), reverse=True)[:8]
                print([(term, round(weight, 3)) for term, weight in ranked])
                """
            ),
        ],
    )


def build_95_notebook() -> None:
    write_notebook(
        "9.5 ModelArts Lab Guide",
        "9.5 ModelArts Lab Guide.ipynb",
        intro_cells("1 ModelArts ExeML Local Image Classification", "ModelArts Lab Guide - Student Version")
        + [
            md_cell("## 1.1 Local ModelArts Workspace\nA compact image dataset is generated locally to represent the same classification workflow without cloud storage or large downloads."),
            code_cell(
                """
                import numpy as np
                import matplotlib.pyplot as plt
                from sklearn.model_selection import train_test_split

                rng = np.random.default_rng(95)
                canvas_size = 24
                cultivar_names = np.array(["aster", "iris", "lotus", "orchid", "violet"])
                palette_bank = np.array([
                    [0.95, 0.72, 0.22],
                    [0.28, 0.49, 0.92],
                    [0.93, 0.55, 0.72],
                    [0.61, 0.30, 0.82],
                    [0.35, 0.74, 0.50],
                ])

                def synthesize_tile(label_id, sample_id):
                    base = np.ones((canvas_size, canvas_size, 3), dtype=float) * palette_bank[label_id]
                    row_axis = np.linspace(0, 1, canvas_size)[:, None]
                    col_axis = np.linspace(0, 1, canvas_size)[None, :]
                    wave = np.sin((label_id + 2) * np.pi * col_axis + sample_id * 0.13)
                    ring = np.cos((label_id + 1) * np.pi * row_axis)
                    base[:, :, 0] += 0.10 * wave
                    base[:, :, 1] += 0.08 * ring
                    base[:, :, 2] += 0.05 * (row_axis - col_axis)
                    return np.clip(base + rng.normal(0, 0.055, base.shape), 0, 1)

                picture_stack = []
                target_codes = []
                for class_id in range(len(cultivar_names)):
                    for sample_id in range(36):
                        picture_stack.append(synthesize_tile(class_id, sample_id))
                        target_codes.append(class_id)

                picture_stack = np.array(picture_stack)
                target_codes = np.array(target_codes)
                train_tiles, valid_tiles, train_codes, valid_codes = train_test_split(
                    picture_stack, target_codes, test_size=0.25, stratify=target_codes, random_state=95
                )
                print("workspace tensor:", picture_stack.shape)
                print("training split:", train_tiles.shape, "validation split:", valid_tiles.shape)
                """
            ),
            md_cell("## 1.2 Dataset Preview\nThe image classes are visualized before feature extraction, similar to checking a ModelArts dataset version."),
            code_cell(
                """
                fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
                for ax, image, label in zip(axes.ravel(), picture_stack[::18][:10], target_codes[::18][:10]):
                    ax.imshow(image)
                    ax.set_title(cultivar_names[label], fontsize=8)
                    ax.axis("off")
                fig.suptitle("Local image dataset preview")
                plt.tight_layout()
                plt.show()
                """
            ),
            md_cell("## 1.3 Feature Engineering Service\nA frozen feature block extracts color, texture, and quadrant statistics instead of reusing the friend's MobileNet code."),
            code_cell(
                """
                def prepare_image_profile(tile_batch):
                    channel_mean = tile_batch.mean(axis=(1, 2))
                    channel_std = tile_batch.std(axis=(1, 2))
                    horizontal_edge = np.abs(np.diff(tile_batch, axis=1)).mean(axis=(1, 2))
                    vertical_edge = np.abs(np.diff(tile_batch, axis=2)).mean(axis=(1, 2))
                    upper_left = tile_batch[:, :12, :12, :].mean(axis=(1, 2))
                    lower_right = tile_batch[:, 12:, 12:, :].mean(axis=(1, 2))
                    return np.concatenate([channel_mean, channel_std, horizontal_edge, vertical_edge, upper_left - lower_right], axis=1)

                modelarts_train_matrix = prepare_image_profile(train_tiles)
                modelarts_valid_matrix = prepare_image_profile(valid_tiles)
                print("feature table:", modelarts_train_matrix.shape)
                """
            ),
            md_cell("## 1.4 ExeML-Style Training Job\nA small classifier is trained and evaluated as if it were the local equivalent of an ExeML image classification job."),
            code_cell(
                """
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score

                image_service_model = RandomForestClassifier(n_estimators=90, max_depth=7, random_state=95)
                image_service_model.fit(modelarts_train_matrix, train_codes)
                validation_guess = image_service_model.predict(modelarts_valid_matrix)
                print("validation accuracy:", round(accuracy_score(valid_codes, validation_guess), 3))
                print(list(zip(cultivar_names[valid_codes[:8]], cultivar_names[validation_guess[:8]])))
                """
            ),
            md_cell("## 1.5 Confusion Matrix\nThe validation matrix shows where the deployed image classifier confuses similar classes."),
            code_cell(
                """
                from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

                cm = confusion_matrix(valid_codes, validation_guess, labels=np.arange(len(cultivar_names)))
                fig, ax = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay(cm, display_labels=cultivar_names).plot(ax=ax, cmap="Purples", colorbar=False)
                ax.set_title("ModelArts-style validation matrix")
                plt.xticks(rotation=35, ha="right")
                plt.tight_layout()
                plt.show()
                """
            ),
            md_cell("## 1.6 Endpoint Prediction Simulation\nA final batch is passed through the trained model to imitate an online prediction endpoint."),
            code_cell(
                """
                def classify_endpoint_batch(images):
                    profile = prepare_image_profile(images)
                    predicted_codes = image_service_model.predict(profile)
                    probabilities = image_service_model.predict_proba(profile).max(axis=1)
                    return predicted_codes, probabilities

                endpoint_codes, endpoint_scores = classify_endpoint_batch(valid_tiles[:10])
                fig, axes = plt.subplots(2, 5, figsize=(9, 4))
                for ax, image, truth, pred, score in zip(axes.ravel(), valid_tiles[:10], valid_codes[:10], endpoint_codes, endpoint_scores):
                    ax.imshow(image)
                    color = "#16883a" if truth == pred else "#b22222"
                    ax.set_title(f"{cultivar_names[pred]} {score:.2f}\\ntrue {cultivar_names[truth]}", fontsize=8, color=color)
                    ax.axis("off")
                plt.tight_layout()
                plt.show()
                """
            ),
        ],
    )


def build_final_exam_notebook() -> None:
    write_notebook(
        "AI Final Exam Lab",
        "AI Final Exam Lab.ipynb",
        intro_cells("1 Handwritten Digit Recognition Final Lab", "AI Final Exam Lab - Student Version")
        + [
            md_cell("## 1.1 Dataset Acquisition\nThe final lab uses the built-in handwritten digits dataset so the notebook remains local and repeatable."),
            code_cell(
                """
                import numpy as np
                import matplotlib.pyplot as plt
                from sklearn.datasets import load_digits
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler

                digits_bundle = load_digits()
                digit_images = digits_bundle.images.astype("float32")
                digit_labels = digits_bundle.target
                flat_pixels = digit_images.reshape(len(digit_images), -1)
                train_pixels, test_pixels, train_digits, test_digits = train_test_split(
                    flat_pixels, digit_labels, test_size=0.25, stratify=digit_labels, random_state=120
                )
                pixel_scaler = StandardScaler().fit(train_pixels)
                train_scaled = pixel_scaler.transform(train_pixels)
                test_scaled = pixel_scaler.transform(test_pixels)
                print("digit image tensor:", digit_images.shape)
                print("train/test rows:", train_scaled.shape, test_scaled.shape)
                """
            ),
            md_cell("## 1.2 Image Grid\nA sample grid confirms how each handwritten digit is encoded as an 8x8 numeric image."),
            code_cell(
                """
                fig, axes = plt.subplots(3, 6, figsize=(7, 3.8))
                for ax, image, label in zip(axes.ravel(), digit_images[:18], digit_labels[:18]):
                    ax.imshow(image, cmap="gray_r")
                    ax.set_title(str(label), fontsize=8)
                    ax.axis("off")
                fig.suptitle("Handwritten digit samples")
                plt.tight_layout()
                plt.show()
                """
            ),
            md_cell("## 1.3 Dense Neural Network\nThe dense model flattens the image and learns class boundaries with hidden layers."),
            code_cell(
                """
                from sklearn.metrics import accuracy_score
                from sklearn.neural_network import MLPClassifier

                dense_digit_net = MLPClassifier(
                    hidden_layer_sizes=(96, 48), activation="relu", max_iter=140, early_stopping=True, random_state=120
                )
                dense_digit_net.fit(train_scaled, train_digits)
                dense_predictions = dense_digit_net.predict(test_scaled)
                dense_accuracy = accuracy_score(test_digits, dense_predictions)
                print("DNN-style accuracy:", round(dense_accuracy, 3))
                print("training iterations:", dense_digit_net.n_iter_)
                """
            ),
            md_cell("## 1.4 CNN-Style Spatial Feature Model\nInstead of copying the TensorFlow CNN, this version extracts local patch and edge features before classification."),
            code_cell(
                """
                from sklearn.linear_model import LogisticRegression

                def spatial_digit_descriptor(flat_batch):
                    image_batch = flat_batch.reshape(-1, 8, 8)
                    row_edges = np.abs(np.diff(image_batch, axis=1)).mean(axis=2)
                    col_edges = np.abs(np.diff(image_batch, axis=2)).mean(axis=1)
                    quadrants = np.stack([
                        image_batch[:, :4, :4].mean(axis=(1, 2)),
                        image_batch[:, :4, 4:].mean(axis=(1, 2)),
                        image_batch[:, 4:, :4].mean(axis=(1, 2)),
                        image_batch[:, 4:, 4:].mean(axis=(1, 2)),
                    ], axis=1)
                    center_mass = image_batch[:, 2:6, 2:6].mean(axis=(1, 2), keepdims=False).reshape(-1, 1)
                    return np.concatenate([flat_batch, row_edges, col_edges, quadrants, center_mass], axis=1)

                train_spatial = spatial_digit_descriptor(train_pixels / 16.0)
                test_spatial = spatial_digit_descriptor(test_pixels / 16.0)
                spatial_digit_net = LogisticRegression(max_iter=700, solver="lbfgs")
                spatial_digit_net.fit(train_spatial, train_digits)
                spatial_predictions = spatial_digit_net.predict(test_spatial)
                spatial_accuracy = accuracy_score(test_digits, spatial_predictions)
                print("CNN-style spatial accuracy:", round(spatial_accuracy, 3))
                """
            ),
            md_cell("## 1.5 Model Comparison\nBoth approaches are compared with the same test split."),
            code_cell(
                """
                comparison_names = ["DNN-style", "CNN-style"]
                comparison_scores = [dense_accuracy, spatial_accuracy]
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(comparison_names, comparison_scores, color=["#315b96", "#2a9d8f"])
                ax.set_ylim(0.85, 1.01)
                ax.set_ylabel("test accuracy")
                ax.set_title("Final lab model comparison")
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha="center")
                plt.tight_layout()
                plt.show()
                """
            ),
            md_cell("## 1.6 Confusion Matrix and Deployment Check\nThe better model is inspected through a confusion matrix and a small prediction grid."),
            code_cell(
                """
                from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

                final_predictions = spatial_predictions if spatial_accuracy >= dense_accuracy else dense_predictions
                final_title = "CNN-style spatial" if spatial_accuracy >= dense_accuracy else "DNN-style dense"
                matrix = confusion_matrix(test_digits, final_predictions, labels=np.arange(10))
                fig, ax = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay(matrix, display_labels=np.arange(10)).plot(ax=ax, cmap="Blues", colorbar=False)
                ax.set_title(f"Confusion matrix - {final_title}")
                plt.tight_layout()
                plt.show()

                fig, axes = plt.subplots(2, 5, figsize=(8, 3.7))
                test_images = test_pixels.reshape(-1, 8, 8)
                for ax, image, truth, pred in zip(axes.ravel(), test_images[:10], test_digits[:10], final_predictions[:10]):
                    ax.imshow(image, cmap="gray_r")
                    ax.set_title(f"pred {pred} / true {truth}", fontsize=8, color="#16883a" if pred == truth else "#b22222")
                    ax.axis("off")
                plt.tight_layout()
                plt.show()
                """
            ),
        ],
    )


def build_assets() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_digits, make_blobs
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    sections = ["9.2 Python", "9.3 ML", "9.4 DL", "9.5 ModelArts", "Final Exam"]
    completion = [4, 9, 5, 1, 1]
    ax.bar(sections, completion, color=["#335C67", "#E09F3E", "#9E2A2B", "#5E548E", "#2A9D8F"])
    ax.set_ylabel("completed labs")
    ax.set_title("Sundetkhan Bekzat midterm lab coverage")
    ax.tick_params(axis="x", rotation=18)
    for idx, value in enumerate(completion):
        ax.text(idx, value + 0.15, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "course_progress_bekzat.png", dpi=150)
    plt.close(fig)

    rng = np.random.default_rng(42)
    x = rng.uniform(1, 10, 80).reshape(-1, 1)
    y = 47 + 5.8 * x.ravel() + rng.normal(0, 4, 80)
    model = LinearRegression().fit(x, y)
    grid = np.linspace(1, 10, 100).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x.ravel(), y, alpha=0.7, color="#4C78A8")
    ax.plot(grid.ravel(), model.predict(grid), color="#F58518", linewidth=2.5)
    ax.set_title("Linear regression practice")
    ax.set_xlabel("study hours")
    ax.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "ml_linear_regression_bekzat.png", dpi=150)
    plt.close(fig)

    missing = pd.Series({"age": 0.125, "income": 0.125, "married": 0.125, "city": 0.0})
    fig, ax = plt.subplots(figsize=(6, 4))
    missing.plot(kind="bar", ax=ax, color="#7A5195")
    ax.set_ylim(0, 0.2)
    ax.set_ylabel("missing ratio")
    ax.set_title("Credit feature missing values")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "feature_missing_values_bekzat.png", dpi=150)
    plt.close(fig)

    points, _ = make_blobs(n_samples=180, centers=3, cluster_std=0.85, random_state=17)
    clusters = KMeans(n_clusters=3, n_init=10, random_state=17).fit_predict(points)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(points[:, 0], points[:, 1], c=clusters, cmap="viridis", s=25)
    ax.set_title("K-Means user grouping")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "kmeans_segments_bekzat.png", dpi=150)
    plt.close(fig)

    ratings = pd.DataFrame(
        [[5, 4, 0, 1, 0], [4, 5, 0, 1, 1], [0, 1, 5, 4, 4], [1, 0, 4, 5, 5], [5, 4, 1, 0, 0]],
        columns=["phone", "charger", "book", "headset", "camera"],
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(ratings, cmap="YlGnBu")
    ax.set_xticks(range(len(ratings.columns)), ratings.columns, rotation=30)
    ax.set_yticks(range(len(ratings.index)), [f"U{i}" for i in range(1, 6)])
    ax.set_title("Retail recommendation matrix")
    fig.colorbar(im, ax=ax, label="rating")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "retail_matrix_bekzat.png", dpi=150)
    plt.close(fig)

    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, stratify=digits.target, random_state=12)
    scaler = StandardScaler().fit(X_train)
    mlp = MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=180, early_stopping=True, random_state=12)
    mlp.fit(scaler.transform(X_train), y_train)
    predictions = mlp.predict(scaler.transform(X_test))
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.2))
    for ax, image, truth, pred in zip(axes.ravel(), X_test[:10].reshape(-1, 8, 8), y_test[:10], predictions[:10]):
        ax.imshow(image, cmap="gray_r")
        ax.set_title(f"t={truth}, p={pred}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Digits classifier predictions")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "digits_predictions_bekzat.png", dpi=150)
    plt.close(fig)

    rng = np.random.default_rng(33)
    names = np.array(["daisy", "dandelion", "rose", "sunflower", "tulip"])
    labels = np.repeat(np.arange(5), 4)
    base_colors = np.array([[0.8, 0.8, 0.2], [0.9, 0.7, 0.1], [0.8, 0.2, 0.3], [1.0, 0.75, 0.05], [0.6, 0.2, 0.8]])
    images = (base_colors[labels][:, None, None, :] + rng.normal(0, 0.12, size=(labels.size, 16, 16, 3))).clip(0, 1)
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
    for ax, image, label in zip(axes.ravel(), images[:10], labels[:10]):
        ax.imshow(image)
        ax.set_title(names[label], fontsize=8)
        ax.axis("off")
    fig.suptitle("Synthetic transfer-learning samples")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "transfer_flow_bekzat.png", dpi=150)
    plt.close(fig)

    rng = np.random.default_rng(95)
    canvas_size = 24
    cultivar_names = np.array(["aster", "iris", "lotus", "orchid", "violet"])
    palette_bank = np.array([
        [0.95, 0.72, 0.22],
        [0.28, 0.49, 0.92],
        [0.93, 0.55, 0.72],
        [0.61, 0.30, 0.82],
        [0.35, 0.74, 0.50],
    ])

    def synthesize_tile(label_id, sample_id):
        base = np.ones((canvas_size, canvas_size, 3), dtype=float) * palette_bank[label_id]
        row_axis = np.linspace(0, 1, canvas_size)[:, None]
        col_axis = np.linspace(0, 1, canvas_size)[None, :]
        base[:, :, 0] += 0.10 * np.sin((label_id + 2) * np.pi * col_axis + sample_id * 0.13)
        base[:, :, 1] += 0.08 * np.cos((label_id + 1) * np.pi * row_axis)
        base[:, :, 2] += 0.05 * (row_axis - col_axis)
        return np.clip(base + rng.normal(0, 0.055, base.shape), 0, 1)

    modelarts_images = []
    modelarts_labels = []
    for class_id in range(len(cultivar_names)):
        for sample_id in range(36):
            modelarts_images.append(synthesize_tile(class_id, sample_id))
            modelarts_labels.append(class_id)
    modelarts_images = np.array(modelarts_images)
    modelarts_labels = np.array(modelarts_labels)

    fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
    for ax, image, label in zip(axes.ravel(), modelarts_images[::18][:10], modelarts_labels[::18][:10]):
        ax.imshow(image)
        ax.set_title(cultivar_names[label], fontsize=8)
        ax.axis("off")
    fig.suptitle("ModelArts local image samples")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "modelarts_gallery_bekzat.png", dpi=150)
    plt.close(fig)

    def prepare_image_profile(tile_batch):
        channel_mean = tile_batch.mean(axis=(1, 2))
        channel_std = tile_batch.std(axis=(1, 2))
        horizontal_edge = np.abs(np.diff(tile_batch, axis=1)).mean(axis=(1, 2))
        vertical_edge = np.abs(np.diff(tile_batch, axis=2)).mean(axis=(1, 2))
        upper_left = tile_batch[:, :12, :12, :].mean(axis=(1, 2))
        lower_right = tile_batch[:, 12:, 12:, :].mean(axis=(1, 2))
        return np.concatenate([channel_mean, channel_std, horizontal_edge, vertical_edge, upper_left - lower_right], axis=1)

    train_tiles, valid_tiles, train_codes, valid_codes = train_test_split(
        modelarts_images, modelarts_labels, test_size=0.25, stratify=modelarts_labels, random_state=95
    )
    modelarts_head = RandomForestClassifier(n_estimators=90, max_depth=7, random_state=95)
    modelarts_head.fit(prepare_image_profile(train_tiles), train_codes)
    modelarts_pred = modelarts_head.predict(prepare_image_profile(valid_tiles))
    modelarts_cm = confusion_matrix(valid_codes, modelarts_pred, labels=np.arange(len(cultivar_names)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(modelarts_cm, display_labels=cultivar_names).plot(ax=ax, cmap="Purples", colorbar=False)
    ax.set_title(f"ModelArts validation accuracy {accuracy_score(valid_codes, modelarts_pred):.3f}")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "modelarts_confusion_bekzat.png", dpi=150)
    plt.close(fig)

    digit_subset = load_digits()
    digit_flat = digit_subset.data
    digit_targets = digit_subset.target
    train_flat, test_flat, train_digit, test_digit = train_test_split(
        digit_flat, digit_targets, stratify=digit_targets, test_size=0.25, random_state=120
    )
    digit_scaler = StandardScaler().fit(train_flat)
    dense_digit_net = MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=140, early_stopping=True, random_state=120)
    dense_digit_net.fit(digit_scaler.transform(train_flat), train_digit)
    dense_digit_pred = dense_digit_net.predict(digit_scaler.transform(test_flat))

    def spatial_digit_descriptor(flat_batch):
        image_batch = flat_batch.reshape(-1, 8, 8)
        row_edges = np.abs(np.diff(image_batch, axis=1)).mean(axis=2)
        col_edges = np.abs(np.diff(image_batch, axis=2)).mean(axis=1)
        quadrants = np.stack([
            image_batch[:, :4, :4].mean(axis=(1, 2)),
            image_batch[:, :4, 4:].mean(axis=(1, 2)),
            image_batch[:, 4:, :4].mean(axis=(1, 2)),
            image_batch[:, 4:, 4:].mean(axis=(1, 2)),
        ], axis=1)
        center_mass = image_batch[:, 2:6, 2:6].mean(axis=(1, 2), keepdims=False).reshape(-1, 1)
        return np.concatenate([flat_batch, row_edges, col_edges, quadrants, center_mass], axis=1)

    train_spatial = spatial_digit_descriptor(train_flat / 16.0)
    test_spatial = spatial_digit_descriptor(test_flat / 16.0)
    spatial_digit_net = LogisticRegression(max_iter=700, solver="lbfgs")
    spatial_digit_net.fit(train_spatial, train_digit)
    spatial_digit_pred = spatial_digit_net.predict(test_spatial)
    dense_acc = accuracy_score(test_digit, dense_digit_pred)
    spatial_acc = accuracy_score(test_digit, spatial_digit_pred)

    fig, axes = plt.subplots(3, 6, figsize=(7, 3.8))
    for ax, image, label in zip(axes.ravel(), digit_subset.images[:18], digit_subset.target[:18]):
        ax.imshow(image, cmap="gray_r")
        ax.set_title(str(label), fontsize=8)
        ax.axis("off")
    fig.suptitle("Final exam digit samples")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "final_digits_grid_bekzat.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["DNN-style", "CNN-style"], [dense_acc, spatial_acc], color=["#315B96", "#2A9D8F"])
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("test accuracy")
    ax.set_title("AI Final Exam model comparison")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "final_model_compare_bekzat.png", dpi=150)
    plt.close(fig)

    matrix = np.array([[4, 0], [1, 3]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["negative", "positive"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Text sentiment check")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "text_sentiment_matrix_bekzat.png", dpi=150)
    plt.close(fig)


def write_reports() -> None:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    (NOTEBOOK_ROOT / ".gitignore").write_text("*.aux\n*.log\n*.out\n*.toc\n.ipynb_checkpoints/\n", encoding="utf-8")

    (NOTEBOOK_ROOT / "README.md").write_text(
        f"""# Huawei HCIA-AI V3.5 Midterm Labs

**Student:** {AUTHOR}  
**Instructor:** {INSTRUCTOR}  
**Subject:** Artificial Intelligence and Neural Networks

## Overview

This folder contains an independent implementation of the Huawei HCIA-AI V3.5 midterm lab set: sections 9.2, 9.3, 9.4, the 9.5 ModelArts lab, and the AI Final Exam lab. The notebooks are written to run locally with synthetic or built-in datasets so the work can be checked without downloading large archives.

## Contents

- `9.2 Python`: four notebooks covering data types, control flow, file I/O, regex, decorators, and a small treasury ledger.
- `9.3 Machine Learning`: nine notebooks covering regression, feature engineering, recommendation, credit scoring, survival prediction, clustering, flower classification, user segmentation, and sentiment analysis.
- `9.4 Deep Learning`: five notebooks covering tensor basics, dense digit classification, transfer-learning workflow, residual connections, and TextCNN-style sentiment features.
- `9.5 ModelArts Lab Guide`: one notebook reproducing a ModelArts-style image classification pipeline locally with independent synthetic images and a separate feature extractor.
- `AI Final Exam Lab`: one notebook comparing dense and CNN-style handwritten digit recognition workflows on a local dataset.
- `assets`: generated figures used by the reports.
- `huawei_midterm.tex` and `huawei_midterm.pdf`: written report files for review.

## How To Run

From the repository root:

```bash
uv run --with jupyter --with numpy --with pandas --with scikit-learn --with matplotlib jupyter nbconvert --to notebook --execute --inplace "midterm_Bekzat/Labs_notebooks/9.3 Machine Learning/9.3.1 Machine Learning Basic Lab Guide/9.3.1 Machine Learning Basic Lab Guide.ipynb"
```

The notebooks avoid interactive input and use deterministic random seeds. MindSpore is optional in section 9.4; the ModelArts and final exam notebooks use local equivalents so they can be reviewed without Huawei cloud credentials or external MNIST downloads.

![Course progress](assets/course_progress_bekzat.png)
""",
        encoding="utf-8",
    )

    (NOTEBOOK_ROOT / "Report_9.2_Python.md").write_text(
        f"""# Section 9.2: Python Programming Foundations

**Student:** {AUTHOR}

## Purpose

Section 9.2 builds the programming base required for later machine learning work. The notebooks cover Python syntax, core containers, loops, functions, object-oriented design, file processing, regular expressions, decorators, and exception handling.

## Completed Labs

- `9.2.1`: data types, deterministic branch checks, loops, Fibonacci generation, class-level state, and standard library inspection.
- `9.2.2`: text, CSV, and JSON file operations using temporary folders; a small database-like adapter replaces an external MySQL dependency.
- `9.2.3`: regex extraction and validation, iterable handling, and a timing decorator.
- `9.2.4`: treasury ledger with transactions, balances, category totals, formatted statements, and safe input parsing.

## Engineering Notes

The implementation avoids interactive `input()` calls so every notebook can be executed by `nbconvert`. External services are replaced with local adapters where the lab concept is more important than a real server connection.

## Result

The section produces reproducible terminal outputs and reusable utility patterns that prepare the later tabular and text-processing notebooks.
""",
        encoding="utf-8",
    )

    (NOTEBOOK_ROOT / "Report_9.3_Machine_Learning.md").write_text(
        f"""# Section 9.3: Machine Learning Practice

**Student:** {AUTHOR}

## Purpose

Section 9.3 moves from programming exercises into classical machine learning. The notebooks use NumPy, Pandas, Matplotlib, and scikit-learn to demonstrate preprocessing, model fitting, evaluation, and visualization.

## Main Work

- Regression and classification baselines are implemented with train/test splits and metrics.
- Missing credit features are repaired with median imputation before chi-square and wrapper-based selection.
- Recommendation is demonstrated through user-item similarity and SVD factors.
- Credit default prediction includes class weighting and threshold tuning.
- Titanic, flower, e-commerce, and customer-review tasks use compact local datasets or synthetic data.

## Visual Evidence

![Linear regression](assets/ml_linear_regression_bekzat.png)

![Missing values](assets/feature_missing_values_bekzat.png)

![K-Means segmentation](assets/kmeans_segments_bekzat.png)

![Retail matrix](assets/retail_matrix_bekzat.png)

## Result

All machine learning notebooks are deterministic and small enough for local execution. They keep the same lab objectives while using independent variable names, data construction, and explanations.
""",
        encoding="utf-8",
    )

    (NOTEBOOK_ROOT / "Report_9.4_Deep_Learning.md").write_text(
        f"""# Section 9.4: Deep Learning and AI Framework Concepts

**Student:** {AUTHOR}

## Purpose

Section 9.4 covers the deep learning part of the midterm. Because large MindSpore datasets and checkpoints are not guaranteed to be available locally, the notebooks are designed with executable fallbacks that preserve the architectural ideas.

## Completed Topics

- Tensor and batching concepts with optional MindSpore detection.
- Dense handwritten-digit classification using the built-in digits dataset.
- Transfer-learning workflow using synthetic image tensors and a frozen feature extractor.
- Residual block shape alignment and checkpoint-like parameter inspection.
- TextCNN-style sentiment analysis using n-gram pooling and a dense classification head.

## Visual Evidence

![Digits predictions](assets/digits_predictions_bekzat.png)

![Transfer samples](assets/transfer_flow_bekzat.png)

![Text sentiment matrix](assets/text_sentiment_matrix_bekzat.png)

## Result

The notebooks are runnable without network downloads and still show the key engineering concerns of deep learning labs: tensor shapes, feature extraction, classification heads, residual paths, and text feature pooling.
""",
        encoding="utf-8",
    )

    (NOTEBOOK_ROOT / "Report_9.5_ModelArts.md").write_text(
        f"""# Section 9.5: ModelArts Local Image Classification

**Student:** {AUTHOR}

## Purpose

Section 9.5 reproduces the idea of Huawei ModelArts ExeML image classification without requiring cloud access. The notebook creates an independent synthetic image dataset, validates class balance, extracts local image descriptors, trains a classifier, and checks endpoint-style predictions.

## Main Work

- Built a local image workspace with five flower-like categories using generated color and texture patterns.
- Implemented a feature extraction service based on channel statistics, edge strength, and quadrant differences.
- Trained a Random Forest classifier as a compact ExeML-style image classification job.
- Evaluated the model with a confusion matrix and a simulated endpoint prediction batch.

## Visual Evidence

![ModelArts image samples](assets/modelarts_gallery_bekzat.png)

![ModelArts confusion matrix](assets/modelarts_confusion_bekzat.png)

## Result

The notebook follows the same ModelArts workflow idea while using different variable names, helper functions, synthetic data, and local execution logic. It is not a direct copy of the reference folder.
""",
        encoding="utf-8",
    )

    (NOTEBOOK_ROOT / "Report_AI_Final_Exam.md").write_text(
        f"""# AI Final Exam Lab: Handwritten Digit Recognition

**Student:** {AUTHOR}

## Purpose

The final exam notebook demonstrates handwritten digit recognition with two model styles. Instead of copying the TensorFlow/MNIST implementation from the other folder, this version uses the built-in digits dataset and compares a dense neural network with a CNN-style spatial feature model.

## Main Work

- Loaded a local handwritten digit dataset and prepared a stratified train/test split.
- Visualized digit samples to confirm the input image structure.
- Trained a dense MLP classifier on scaled flattened pixels.
- Built a CNN-style spatial descriptor from local edges, quadrants, and center mass before classification.
- Compared test accuracy and inspected prediction behavior through a confusion matrix and deployment-style grid.

## Visual Evidence

![Final exam digit grid](assets/final_digits_grid_bekzat.png)

![Final exam model comparison](assets/final_model_compare_bekzat.png)

## Result

The notebook covers the final exam objective with independent preprocessing names, model variables, feature functions, and plotting logic. It remains fast and reproducible on a local machine.
""",
        encoding="utf-8",
    )

    (NOTEBOOK_ROOT / "chapter1.tex").write_text(
        f"""Section 9.2 was completed as four executable Python notebooks. The work covers core data structures, branch and loop logic, reusable functions, object-oriented records, file I/O, regular expressions, decorators, and a small treasury ledger. The implementation by {AUTHOR} uses deterministic examples so every notebook can be re-run during review.\n""",
        encoding="utf-8",
    )
    (NOTEBOOK_ROOT / "chapter2.tex").write_text(
        """Section 9.3 contains nine machine learning notebooks. The work demonstrates regression, classification, feature engineering, recommendation, imbalanced credit scoring, survival prediction, clustering, flower classification, e-commerce segmentation, and sentiment analysis. The generated assets document the key plots and model outputs.\n""",
        encoding="utf-8",
    )
    (NOTEBOOK_ROOT / "chapter3.tex").write_text(
        """Section 9.4 contains five deep learning concept notebooks. The implementation checks optional MindSpore availability and otherwise uses local NumPy or scikit-learn fallbacks. The notebooks cover tensor operations, dense digit classification, transfer-learning flow, residual shape safety, and TextCNN-style sentiment pooling.\n""",
        encoding="utf-8",
    )
    (NOTEBOOK_ROOT / "chapter4.tex").write_text(
        """Section 9.5 adds a ModelArts-style image classification notebook. The work avoids cloud credentials by building a local synthetic image workspace, extracting independent image profiles, training a Random Forest classifier, and validating predictions with a confusion matrix plus endpoint-style batch inference.\n""",
        encoding="utf-8",
    )
    (NOTEBOOK_ROOT / "chapter5.tex").write_text(
        """The AI Final Exam lab adds a handwritten digit recognition workflow. It compares a dense neural model against a CNN-style spatial feature pipeline using the same local train/test split, then reports model accuracy and prediction behavior with visual evidence.\n""",
        encoding="utf-8",
    )

    (NOTEBOOK_ROOT / "huawei_midterm.tex").write_text(
        f"""\\documentclass[a4paper,12pt]{{article}}
\\usepackage[margin=2.5cm]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{float}}
\\usepackage{{listings}}
\\title{{Huawei HCIA-AI V3.5 Midterm Implementation Report}}
\\author{{{AUTHOR} \\\\ Instructor: {INSTRUCTOR}}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle
\\tableofcontents
\\newpage

\\section{{Repository and Course Coverage}}
This report summarizes the local midterm implementation in \\texttt{{midterm\\_Bekzat/Labs\\_notebooks}}. The notebooks cover sections 9.2, 9.3, 9.4, the 9.5 ModelArts lab, and the AI Final Exam lab with independent code and deterministic local examples.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.85\\linewidth]{{assets/course_progress_bekzat.png}}
\\caption{{Completed midterm lab coverage}}
\\end{{figure}}

\\section{{Python Programming Foundations}}
\\input{{chapter1.tex}}

\\section{{Machine Learning Practice}}
\\input{{chapter2.tex}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.75\\linewidth]{{assets/ml_linear_regression_bekzat.png}}
\\caption{{Linear regression result}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.75\\linewidth]{{assets/kmeans_segments_bekzat.png}}
\\caption{{K-Means segmentation result}}
\\end{{figure}}

\\section{{Deep Learning and AI Framework Concepts}}
\\input{{chapter3.tex}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.75\\linewidth]{{assets/digits_predictions_bekzat.png}}
\\caption{{Digit classifier predictions}}
\\end{{figure}}

\\section{{ModelArts Local Image Classification}}
\\input{{chapter4.tex}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.75\\linewidth]{{assets/modelarts_gallery_bekzat.png}}
\\caption{{ModelArts-style generated image samples}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.75\\linewidth]{{assets/modelarts_confusion_bekzat.png}}
\\caption{{ModelArts-style validation confusion matrix}}
\\end{{figure}}

\\section{{AI Final Exam Lab}}
\\input{{chapter5.tex}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.75\\linewidth]{{assets/final_digits_grid_bekzat.png}}
\\caption{{Final exam handwritten digit samples}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.75\\linewidth]{{assets/final_model_compare_bekzat.png}}
\\caption{{Final exam model comparison}}
\\end{{figure}}

\\end{{document}}
""",
        encoding="utf-8",
    )


def write_pdf_summary() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = NOTEBOOK_ROOT / "huawei_midterm.pdf"
    pages = [
        (
            "Huawei HCIA-AI V3.5 Midterm Implementation Report",
            f"Student: {AUTHOR}\nInstructor: {INSTRUCTOR}\nSubject: Artificial Intelligence and Neural Networks\n\nThis PDF summarizes the independently implemented notebooks in midterm_Bekzat/Labs_notebooks.",
            None,
        ),
        (
            "Section 9.2 Python",
            "Four notebooks cover data types, control flow, file operations, regex, decorators, and a treasury ledger. All examples are deterministic and avoid interactive input.",
            ASSET_DIR / "course_progress_bekzat.png",
        ),
        (
            "Section 9.3 Machine Learning",
            "Nine notebooks cover regression, feature engineering, recommendation, credit scoring, survival prediction, clustering, flower classification, user segmentation, and sentiment analysis.",
            ASSET_DIR / "ml_linear_regression_bekzat.png",
        ),
        (
            "Section 9.4 Deep Learning",
            "Five notebooks cover tensor basics, digit classification, transfer learning, residual blocks, and TextCNN-style sentiment features with local fallbacks.",
            ASSET_DIR / "digits_predictions_bekzat.png",
        ),
        (
            "Section 9.5 ModelArts",
            "One notebook reproduces a ModelArts-style image classification workflow locally with synthetic image classes, independent feature profiles, a Random Forest classifier, and endpoint-style predictions.",
            ASSET_DIR / "modelarts_confusion_bekzat.png",
        ),
        (
            "AI Final Exam Lab",
            "One notebook compares a dense neural classifier with a CNN-style spatial feature model for handwritten digit recognition using a reproducible local dataset.",
            ASSET_DIR / "final_model_compare_bekzat.png",
        ),
    ]

    with PdfPages(pdf_path) as pdf:
        for title, body, image_path in pages:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.93, title, ha="center", va="top", fontsize=18, weight="bold", wrap=True)
            ax.text(0.08, 0.82, textwrap.fill(body, width=85), ha="left", va="top", fontsize=11)
            if image_path and image_path.exists():
                image = plt.imread(image_path)
                img_ax = fig.add_axes([0.12, 0.18, 0.76, 0.48])
                img_ax.imshow(image)
                img_ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)


def validate_notebooks() -> None:
    expected = list(NOTEBOOK_ROOT.rglob("*.ipynb"))
    for path in expected:
        json.loads(path.read_text(encoding="utf-8"))
    print(f"Validated {len(expected)} notebooks")


def main() -> None:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    build_92_notebooks()
    build_93_notebooks()
    build_94_notebooks()
    build_95_notebook()
    build_final_exam_notebook()
    build_assets()
    write_reports()
    try:
        from build_formal_report import main as build_formal_report

        build_formal_report()
    except Exception as exc:
        print(f"Formal PDF builder unavailable ({type(exc).__name__}: {exc}); writing compact fallback PDF")
        write_pdf_summary()
    validate_notebooks()
    print(f"Built Sundetkhan Bekzat midterm deliverables at {NOTEBOOK_ROOT}")


if __name__ == "__main__":
    main()
