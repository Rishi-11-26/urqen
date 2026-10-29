"""
previa/export/code_generator.py
-------------------------------
Generates standalone, reproducible scikit-learn Python scripts
reproducing the winning ML pipeline from a URQEN experiment.
"""

from typing import Dict, Any


def generate_pipeline_python_code(experiment_record: Dict[str, Any]) -> str:
    """
    Produces a clean, copy-pasteable Python script to retrain and predict
    with the winning model using scikit-learn.
    """
    best_model = experiment_record.get("best_model") or "RandomForestClassifier"
    target = experiment_record.get("target_column") or "target"
    task_type = experiment_record.get("task_type") or "classification"
    ds_name = experiment_record.get("dataset_name") or "dataset.csv"
    config = experiment_record.get("configuration") or {}
    split_ratio = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)

    is_classif = "classification" in task_type.lower()

    # Import mapping for algorithms
    algo_imports = {
        "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
        "RandomForestRegressor": "from sklearn.ensemble import RandomForestRegressor",
        "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
        "GradientBoostingRegressor": "from sklearn.ensemble import GradientBoostingRegressor",
        "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
        "LinearRegression": "from sklearn.linear_model import LinearRegression",
        "Ridge": "from sklearn.linear_model import Ridge",
        "Lasso": "from sklearn.linear_model import Lasso",
        "ElasticNet": "from sklearn.linear_model import ElasticNet",
        "DecisionTreeClassifier": "from sklearn.tree import DecisionTreeClassifier",
        "DecisionTreeRegressor": "from sklearn.tree import DecisionTreeRegressor",
        "KNeighborsClassifier": "from sklearn.neighbors import KNeighborsClassifier",
        "KNeighborsRegressor": "from sklearn.neighbors import KNeighborsRegressor",
        "SVC": "from sklearn.svm import SVC",
        "SVR": "from sklearn.svm import SVR",
        "GaussianNB": "from sklearn.naive_bayes import GaussianNB",
        "AdaBoostClassifier": "from sklearn.ensemble import AdaBoostClassifier",
        "ExtraTreesClassifier": "from sklearn.ensemble import ExtraTreesClassifier",
        "ExtraTreesRegressor": "from sklearn.ensemble import ExtraTreesRegressor",
        "XGBClassifier": "from xgboost import XGBClassifier",
        "XGBRegressor": "from xgboost import XGBRegressor",
    }

    # Normalize best model to known class name
    model_class = best_model
    if is_classif:
        if "Forest" in best_model:
            model_class = "RandomForestClassifier"
        elif "Gradient" in best_model:
            model_class = "GradientBoostingClassifier"
        elif "Logistic" in best_model:
            model_class = "LogisticRegression"
        elif "XGB" in best_model:
            model_class = "XGBClassifier"
        elif "Tree" in best_model:
            model_class = "DecisionTreeClassifier"
        elif "KNN" in best_model or "Neighbor" in best_model:
            model_class = "KNeighborsClassifier"
        elif "SVM" in best_model or "SVC" in best_model:
            model_class = "SVC"
        elif "Naive" in best_model or "Bayes" in best_model:
            model_class = "GaussianNB"
        elif "Ada" in best_model:
            model_class = "AdaBoostClassifier"
        elif "Extra" in best_model:
            model_class = "ExtraTreesClassifier"
    else:
        if "Forest" in best_model:
            model_class = "RandomForestRegressor"
        elif "Gradient" in best_model:
            model_class = "GradientBoostingRegressor"
        elif "Linear" in best_model:
            model_class = "LinearRegression"
        elif "Ridge" in best_model:
            model_class = "Ridge"
        elif "Lasso" in best_model:
            model_class = "Lasso"
        elif "XGB" in best_model:
            model_class = "XGBRegressor"
        elif "Tree" in best_model:
            model_class = "DecisionTreeRegressor"
        elif "KNN" in best_model:
            model_class = "KNeighborsRegressor"
        elif "SVR" in best_model:
            model_class = "SVR"
        elif "Extra" in best_model:
            model_class = "ExtraTreesRegressor"

    import_stmt = algo_imports.get(model_class, "from sklearn.ensemble import RandomForestClassifier")
    eval_imports = (
        "from sklearn.metrics import classification_report, accuracy_score, f1_score"
        if is_classif
        else "from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score"
    )

    stratify_arg = ", stratify=y" if is_classif else ""

    code = f'''# ==============================================================================
# URQEN Auto-Generated ML Pipeline Script
# Experiment ID : {experiment_record.get("experiment_id", "exp_1")}
# Winning Model : {best_model}
# Prediction Target: {target}
# Task Type      : {task_type}
# Generated by   : URQEN Platform (https://urqen.onrender.com)
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
{import_stmt}
{eval_imports}

# 1. Load your dataset
DATASET_PATH = "{ds_name}"
print(f"[*] Loading dataset from {{DATASET_PATH}}...")
df = pd.read_csv(DATASET_PATH)

# 2. Separate Features (X) and Target (y)
TARGET_COLUMN = "{target}"
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{{TARGET_COLUMN}}' not found in dataset columns: {{list(df.columns)}}")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Identify numerical and categorical features
numeric_features = X.select_dtypes(include=["int64", "float64", "number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print(f"[*] Features: {{len(numeric_features)}} numerical, {{len(categorical_features)}} categorical")

# 3. Build Leakage-Safe Preprocessing Transformers
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

# 4. Instantiate the Winning Model ({best_model})
model = {model_class}(random_state={random_state}) if hasattr({model_class}, "random_state") else {model_class}()

# Full Pipeline: Preprocessing + Estimator
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# 5. Train/Test Holdout Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size={split_ratio},
    random_state={random_state}{stratify_arg}
)

# 6. Fit Pipeline Strictly on Training Partition (Zero Leakage)
print(f"[*] Training {{model.__class__.__name__}} on {{len(X_train)}} samples...")
pipeline.fit(X_train, y_train)

# 7. Evaluate on Holdout Set
print(f"[*] Evaluating on {{len(X_test)}} holdout samples...")
y_pred = pipeline.predict(X_test)

'''
    if is_classif:
        code += '''print("\n" + "="*50)
print("EVALUATION METRICS:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
print("="*50)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
'''
    else:
        code += '''print("\n" + "="*50)
print("EVALUATION METRICS:")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²  : {r2_score(y_test, y_pred):.4f}")
print("="*50)
'''

    code += '''
# 8. Inference Example on Unseen Samples
# sample_unseen_data = X_test.head(5)
# predictions = pipeline.predict(sample_unseen_data)
# print("Predictions on sample:\n", predictions)
'''
    return code
