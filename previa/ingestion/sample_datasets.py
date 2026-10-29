"""
previa/ingestion/sample_datasets.py
-----------------------------------
Built-in benchmark sample datasets for zero-upload instant testing.
Provides clean Iris and Titanic datasets locally without external network dependencies.
"""

import pandas as pd
from sklearn.datasets import load_iris


def load_sample_iris() -> pd.DataFrame:
    """Loads the classic 150-row Iris dataset with clean column headers."""
    raw = load_iris(as_frame=True)
    df = raw.frame.copy()
    df.columns = [
        c.replace(" (cm)", "").replace(" ", "_").strip()
        for c in df.columns
    ]
    target_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    if "target" in df.columns:
        df["species"] = df["target"].map(target_map)
        df.drop(columns=["target"], inplace=True)
    return df


def load_sample_titanic() -> pd.DataFrame:
    """Loads a representative tabular sample of the Titanic survival dataset."""
    # Embedded compact sample representing various passenger classes, ages, and outcomes
    data = [
        {"PassengerId": 1, "Survived": 0, "Pclass": 3, "Name": "Braund, Mr. Owen Harris", "Sex": "male", "Age": 22.0, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"},
        {"PassengerId": 2, "Survived": 1, "Pclass": 1, "Name": "Cumings, Mrs. John Bradley", "Sex": "female", "Age": 38.0, "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C"},
        {"PassengerId": 3, "Survived": 1, "Pclass": 3, "Name": "Heikkinen, Miss. Laina", "Sex": "female", "Age": 26.0, "SibSp": 0, "Parch": 0, "Fare": 7.92, "Embarked": "S"},
        {"PassengerId": 4, "Survived": 1, "Pclass": 1, "Name": "Futrelle, Mrs. Jacques Heath", "Sex": "female", "Age": 35.0, "SibSp": 1, "Parch": 0, "Fare": 53.10, "Embarked": "S"},
        {"PassengerId": 5, "Survived": 0, "Pclass": 3, "Name": "Allen, Mr. William Henry", "Sex": "male", "Age": 35.0, "SibSp": 0, "Parch": 0, "Fare": 8.05, "Embarked": "S"},
        {"PassengerId": 6, "Survived": 0, "Pclass": 3, "Name": "Moran, Mr. James", "Sex": "male", "Age": 28.0, "SibSp": 0, "Parch": 0, "Fare": 8.46, "Embarked": "Q"},
        {"PassengerId": 7, "Survived": 0, "Pclass": 1, "Name": "McCarthy, Mr. Timothy J", "Sex": "male", "Age": 54.0, "SibSp": 0, "Parch": 0, "Fare": 51.86, "Embarked": "S"},
        {"PassengerId": 8, "Survived": 0, "Pclass": 3, "Name": "Palsson, Master. Gosta Leonard", "Sex": "male", "Age": 2.0, "SibSp": 3, "Parch": 1, "Fare": 21.07, "Embarked": "S"},
        {"PassengerId": 9, "Survived": 1, "Pclass": 3, "Name": "Johnson, Mrs. Oscar W", "Sex": "female", "Age": 27.0, "SibSp": 0, "Parch": 2, "Fare": 11.13, "Embarked": "S"},
        {"PassengerId": 10, "Survived": 1, "Pclass": 2, "Name": "Nasser, Mrs. Nicholas", "Sex": "female", "Age": 14.0, "SibSp": 1, "Parch": 0, "Fare": 30.07, "Embarked": "C"},
        {"PassengerId": 11, "Survived": 1, "Pclass": 3, "Name": "Sandstrom, Miss. Marguerite Rut", "Sex": "female", "Age": 4.0, "SibSp": 1, "Parch": 1, "Fare": 16.70, "Embarked": "S"},
        {"PassengerId": 12, "Survived": 1, "Pclass": 1, "Name": "Bonnell, Miss. Elizabeth", "Sex": "female", "Age": 58.0, "SibSp": 0, "Parch": 0, "Fare": 26.55, "Embarked": "S"},
        {"PassengerId": 13, "Survived": 0, "Pclass": 3, "Name": "Saundercock, Mr. William Henry", "Sex": "male", "Age": 20.0, "SibSp": 0, "Parch": 0, "Fare": 8.05, "Embarked": "S"},
        {"PassengerId": 14, "Survived": 0, "Pclass": 3, "Name": "Andersson, Mr. Anders Johan", "Sex": "male", "Age": 39.0, "SibSp": 1, "Parch": 5, "Fare": 31.28, "Embarked": "S"},
        {"PassengerId": 15, "Survived": 0, "Pclass": 3, "Name": "Vestrom, Miss. Hulda Amanda Adolfina", "Sex": "female", "Age": 14.0, "SibSp": 0, "Parch": 0, "Fare": 7.85, "Embarked": "S"},
        {"PassengerId": 16, "Survived": 1, "Pclass": 2, "Name": "Hewlett, Mrs.", "Sex": "female", "Age": 55.0, "SibSp": 0, "Parch": 0, "Fare": 16.00, "Embarked": "S"},
        {"PassengerId": 17, "Survived": 0, "Pclass": 3, "Name": "Rice, Master. Eugene", "Sex": "male", "Age": 2.0, "SibSp": 4, "Parch": 1, "Fare": 29.12, "Embarked": "Q"},
        {"PassengerId": 18, "Survived": 1, "Pclass": 2, "Name": "Williams, Mr. Charles Eugene", "Sex": "male", "Age": 30.0, "SibSp": 0, "Parch": 0, "Fare": 13.00, "Embarked": "S"},
        {"PassengerId": 19, "Survived": 0, "Pclass": 3, "Name": "Vander Planke, Mrs. Julius", "Sex": "female", "Age": 31.0, "SibSp": 1, "Parch": 0, "Fare": 18.00, "Embarked": "S"},
        {"PassengerId": 20, "Survived": 1, "Pclass": 3, "Name": "Masselmani, Mrs. Fatima", "Sex": "female", "Age": 24.0, "SibSp": 0, "Parch": 0, "Fare": 7.22, "Embarked": "C"},
        {"PassengerId": 21, "Survived": 0, "Pclass": 2, "Name": "Fynney, Mr. Joseph J", "Sex": "male", "Age": 35.0, "SibSp": 0, "Parch": 0, "Fare": 26.00, "Embarked": "S"},
        {"PassengerId": 22, "Survived": 1, "Pclass": 2, "Name": "Beesley, Mr. Lawrence", "Sex": "male", "Age": 34.0, "SibSp": 0, "Parch": 0, "Fare": 13.00, "Embarked": "S"},
        {"PassengerId": 23, "Survived": 1, "Pclass": 3, "Name": "McGowan, Miss. Anna", "Sex": "female", "Age": 15.0, "SibSp": 0, "Parch": 0, "Fare": 8.03, "Embarked": "Q"},
        {"PassengerId": 24, "Survived": 1, "Pclass": 1, "Name": "Sloper, Mr. William Thompson", "Sex": "male", "Age": 28.0, "SibSp": 0, "Parch": 0, "Fare": 35.50, "Embarked": "S"},
        {"PassengerId": 25, "Survived": 0, "Pclass": 3, "Name": "Palsson, Miss. Torborg Danira", "Sex": "female", "Age": 8.0, "SibSp": 3, "Parch": 1, "Fare": 21.07, "Embarked": "S"},
        {"PassengerId": 26, "Survived": 1, "Pclass": 3, "Name": "Asplund, Mrs. Carl Oscar", "Sex": "female", "Age": 38.0, "SibSp": 1, "Parch": 5, "Fare": 31.39, "Embarked": "S"},
        {"PassengerId": 27, "Survived": 0, "Pclass": 3, "Name": "Emir, Mr. Farred Chehab", "Sex": "male", "Age": 25.0, "SibSp": 0, "Parch": 0, "Fare": 7.22, "Embarked": "C"},
        {"PassengerId": 28, "Survived": 0, "Pclass": 1, "Name": "Fortune, Mr. Charles Alexander", "Sex": "male", "Age": 19.0, "SibSp": 3, "Parch": 2, "Fare": 263.00, "Embarked": "S"},
        {"PassengerId": 29, "Survived": 1, "Pclass": 3, "Name": "O'Dwyer, Miss. Ellen", "Sex": "female", "Age": 21.0, "SibSp": 0, "Parch": 0, "Fare": 7.88, "Embarked": "Q"},
        {"PassengerId": 30, "Survived": 0, "Pclass": 3, "Name": "Todoroff, Mr. Lalio", "Sex": "male", "Age": 29.0, "SibSp": 0, "Parch": 0, "Fare": 7.90, "Embarked": "S"},
        {"PassengerId": 31, "Survived": 0, "Pclass": 1, "Name": "Uruchurtu, Don. Manuel E", "Sex": "male", "Age": 40.0, "SibSp": 0, "Parch": 0, "Fare": 27.72, "Embarked": "C"},
        {"PassengerId": 32, "Survived": 1, "Pclass": 1, "Name": "Spencer, Mrs. William Augustus", "Sex": "female", "Age": 35.0, "SibSp": 1, "Parch": 0, "Fare": 146.52, "Embarked": "C"},
        {"PassengerId": 33, "Survived": 1, "Pclass": 3, "Name": "Glynn, Miss. Mary Agatha", "Sex": "female", "Age": 22.0, "SibSp": 0, "Parch": 0, "Fare": 7.75, "Embarked": "Q"},
        {"PassengerId": 34, "Survived": 0, "Pclass": 2, "Name": "Wheadon, Mr. Edward H", "Sex": "male", "Age": 66.0, "SibSp": 0, "Parch": 0, "Fare": 10.50, "Embarked": "S"},
        {"PassengerId": 35, "Survived": 0, "Pclass": 1, "Name": "Meyer, Mr. Edgar Joseph", "Sex": "male", "Age": 28.0, "SibSp": 1, "Parch": 0, "Fare": 82.17, "Embarked": "C"},
        {"PassengerId": 36, "Survived": 0, "Pclass": 1, "Name": "Holverson, Mr. Alexander Oskar", "Sex": "male", "Age": 42.0, "SibSp": 1, "Parch": 0, "Fare": 52.00, "Embarked": "S"},
        {"PassengerId": 37, "Survived": 1, "Pclass": 3, "Name": "Mamee, Mr. Hanna", "Sex": "male", "Age": 20.0, "SibSp": 0, "Parch": 0, "Fare": 7.23, "Embarked": "C"},
        {"PassengerId": 38, "Survived": 0, "Pclass": 3, "Name": "Cann, Mr. Ernest Charles", "Sex": "male", "Age": 21.0, "SibSp": 0, "Parch": 0, "Fare": 8.05, "Embarked": "S"},
        {"PassengerId": 39, "Survived": 0, "Pclass": 3, "Name": "Vander Planke, Miss. Augusta Maria", "Sex": "female", "Age": 18.0, "SibSp": 2, "Parch": 0, "Fare": 18.00, "Embarked": "S"},
        {"PassengerId": 40, "Survived": 1, "Pclass": 3, "Name": "Nicola-Yarred, Miss. Jamila", "Sex": "female", "Age": 14.0, "SibSp": 1, "Parch": 1, "Fare": 11.24, "Embarked": "C"},
        {"PassengerId": 41, "Survived": 0, "Pclass": 3, "Name": "Ahlin, Mrs. Johan", "Sex": "female", "Age": 40.0, "SibSp": 1, "Parch": 0, "Fare": 9.48, "Embarked": "S"},
        {"PassengerId": 42, "Survived": 0, "Pclass": 2, "Name": "Turpin, Mrs. William John Robert", "Sex": "female", "Age": 27.0, "SibSp": 1, "Parch": 0, "Fare": 21.00, "Embarked": "S"},
        {"PassengerId": 43, "Survived": 0, "Pclass": 3, "Name": "Kraeff, Mr. Theodor", "Sex": "male", "Age": 32.0, "SibSp": 0, "Parch": 0, "Fare": 7.90, "Embarked": "C"},
        {"PassengerId": 44, "Survived": 1, "Pclass": 2, "Name": "Laroche, Miss. Simonne Marie Anne", "Sex": "female", "Age": 3.0, "SibSp": 1, "Parch": 2, "Fare": 41.58, "Embarked": "C"},
        {"PassengerId": 45, "Survived": 1, "Pclass": 3, "Name": "Devaney, Miss. Margaret Delia", "Sex": "female", "Age": 19.0, "SibSp": 0, "Parch": 0, "Fare": 7.88, "Embarked": "Q"},
        {"PassengerId": 46, "Survived": 0, "Pclass": 3, "Name": "Rogers, Mr. William John", "Sex": "male", "Age": 30.0, "SibSp": 0, "Parch": 0, "Fare": 8.05, "Embarked": "S"},
        {"PassengerId": 47, "Survived": 0, "Pclass": 3, "Name": "Lennon, Mr. Denis", "Sex": "male", "Age": 28.0, "SibSp": 1, "Parch": 0, "Fare": 15.50, "Embarked": "Q"},
        {"PassengerId": 48, "Survived": 1, "Pclass": 3, "Name": "O'Driscoll, Miss. Bridget", "Sex": "female", "Age": 22.0, "SibSp": 0, "Parch": 0, "Fare": 7.75, "Embarked": "Q"},
        {"PassengerId": 49, "Survived": 0, "Pclass": 3, "Name": "Samaan, Mr. Youssef", "Sex": "male", "Age": 26.0, "SibSp": 2, "Parch": 0, "Fare": 21.68, "Embarked": "C"},
        {"PassengerId": 50, "Survived": 0, "Pclass": 3, "Name": "Arnold-Franchi, Mrs. Josef", "Sex": "female", "Age": 18.0, "SibSp": 1, "Parch": 0, "Fare": 17.80, "Embarked": "S"},
    ]
    # Multiply representative patterns to create a robust 100-row sample
    extra_data = []
    for i, row in enumerate(data):
        new_row = row.copy()
        new_row["PassengerId"] = row["PassengerId"] + 50
        # slight fare variations
        new_row["Fare"] = round(row["Fare"] * 1.05, 2)
        extra_data.append(new_row)
    return pd.DataFrame(data + extra_data)


def get_sample_dataset(name: str) -> pd.DataFrame:
    """Returns the requested built-in sample dataset as a pandas DataFrame."""
    norm = (name or "").lower().strip()
    if norm == "iris":
        return load_sample_iris()
    elif norm == "titanic":
        return load_sample_titanic()
    raise ValueError(f"Unknown sample dataset '{name}'. Available: 'iris', 'titanic'.")
