"""Titanic survival classification with Logistic Regression.

Run from the repository root:
    python src/titanic_classification.py
Expects the Kaggle training file at data/train.csv.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/train.csv")
IMAGE_DIR = Path("images")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    return df


def plot_eda(df: pd.DataFrame) -> None:
    IMAGE_DIR.mkdir(exist_ok=True)
    for col, name in [("Pclass", "survival_by_class"), ("Sex", "survival_by_sex")]:
        plt.figure()
        sns.barplot(x=col, y="Survived", data=df)
        plt.title(f"Survival Rate by {'ticket class' if col == 'Pclass' else 'Sex'}")
        plt.savefig(IMAGE_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close()


def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Fill missing values
    df[["Age"]] = SimpleImputer(strategy="median").fit_transform(df[["Age"]])
    df[["Embarked"]] = SimpleImputer(strategy="most_frequent").fit_transform(df[["Embarked"]])

    # Drop identifier columns and Cabin (77% missing)
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    # Features and target
    x = df.drop(columns=["Survived"])
    y = df["Survived"]

    # One-hot encode categorical columns
    cat_cols = ["Sex", "Embarked"]
    enc = OneHotEncoder()
    encoded = pd.DataFrame(
        enc.fit_transform(x[cat_cols]).toarray(),
        columns=enc.get_feature_names_out(cat_cols),
        index=x.index,
    )
    x = pd.concat([x.drop(columns=cat_cols), encoded], axis=1)
    return x, y


def main() -> None:
    df = load_data()
    plot_eda(df)
    x, y = preprocess(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    print(x_train.shape, x_test.shape)

    # Scale numeric columns: fit on training data only, then apply to test data
    num_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
    sc = StandardScaler()
    x_train[num_cols] = sc.fit_transform(x_train[num_cols])
    x_test[num_cols] = sc.transform(x_test[num_cols])

    # Train
    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_train, y_train)

    # Evaluate
    predict = lr.predict(x_test)
    print(f"Training accuracy: {lr.score(x_train, y_train) * 100:.1f}%")
    print(f"Test accuracy:     {accuracy_score(y_test, predict) * 100:.1f}%")
    print(f"Precision:         {precision_score(y_test, predict) * 100:.1f}%")
    print(f"Recall:            {recall_score(y_test, predict) * 100:.1f}%")
    print(f"F1-score:          {f1_score(y_test, predict) * 100:.1f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, predict)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    IMAGE_DIR.mkdir(exist_ok=True)
    plt.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
