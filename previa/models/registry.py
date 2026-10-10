"""
previa/models/registry.py
-------------------------
Centralized Model Registry for Previa.

Provides a unified catalog of baseline algorithms for tabular
regression and classification tasks.

Supported Models
----------------
Classification:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost (optional, enabled if xgboost package is installed)

Regression:
  - Linear Regression
  - Ridge
  - Decision Tree
  - Random Forest
  - XGBoost (optional, enabled if xgboost package is installed)

Design Principles:
- Fixed random state (42) for deterministic reproducibility.
- Explicit metadata describing algorithmic family and scale sensitivity.
- Graceful handling of optional dependencies (XGBoost).
- Clean factory functions so downstream training code doesn't hardcode model constructors.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


RANDOM_STATE = 42

# Check optional XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def is_xgboost_available() -> bool:
    """Return True if XGBoost is installed and importable."""
    return XGBOOST_AVAILABLE


def get_available_models(task_type: str, include_optional: bool = False) -> list[dict]:
    """
    Return metadata list of available models for the given task.

    Parameters
    ----------
    task_type : str
        "binary_classification", "multiclass_classification", or "regression".
    include_optional : bool
        Whether to include optional algorithms (e.g. XGBoost) in addition
        to core baselines. Defaults to False to ensure consistent baseline suites.

    Returns
    -------
    list of dicts
    """
    is_classification = task_type in ("binary_classification", "multiclass_classification")

    if is_classification:
        models = [
            {
                "name": "Logistic Regression",
                "family": "linear",
                "scale_sensitive": True,
                "description": "Linear classifier using cross-entropy loss with L2 regularization. Fast, interpretable, scale-sensitive.",
            },
            {
                "name": "Decision Tree",
                "family": "tree",
                "scale_sensitive": False,
                "description": "Non-parametric tree partitioner. Captures nonlinear interactions; scale-invariant; prone to overfitting on small data.",
            },
            {
                "name": "Random Forest",
                "family": "ensemble",
                "scale_sensitive": False,
                "description": "Bagged ensemble of de-correlated decision trees. Robust to noise and variance; scale-invariant.",
            },
        ]
        if include_optional and XGBOOST_AVAILABLE:
            models.append({
                "name": "XGBoost",
                "family": "boosting",
                "scale_sensitive": False,
                "description": "Gradient boosted decision tree ensemble with regularized loss. High predictive performance on structured tables.",
            })
        return models

    else:  # regression
        models = [
            {
                "name": "Linear Regression",
                "family": "linear",
                "scale_sensitive": True,
                "description": "Ordinary least squares regressor. Transparent coefficients; sensitive to multicollinearity and feature scaling.",
            },
            {
                "name": "Ridge",
                "family": "linear",
                "scale_sensitive": True,
                "description": "L2-regularized linear model. Stabilizes coefficient estimation in the presence of collinearity.",
            },
            {
                "name": "Decision Tree",
                "family": "tree",
                "scale_sensitive": False,
                "description": "Piecewise-constant regression tree. Fits complex surfaces without feature scaling.",
            },
            {
                "name": "Random Forest",
                "family": "ensemble",
                "scale_sensitive": False,
                "description": "Averaged ensemble of randomized regression trees. High stability and predictive strength.",
            },
        ]
        if include_optional and XGBOOST_AVAILABLE:
            models.append({
                "name": "XGBoost",
                "family": "boosting",
                "scale_sensitive": False,
                "description": "Gradient boosted trees optimizing squared error. State-of-the-art tabular baseline.",
            })
        return models


def create_model_instance(name: str, task_type: str, random_state: int = RANDOM_STATE):
    """
    Instantiate an unfitted model object by name and task type.

    Parameters
    ----------
    name : str
        Model name (e.g. 'Logistic Regression', 'Ridge', 'XGBoost').
    task_type : str
        "binary_classification", "multiclass_classification", or "regression".
    random_state : int
        Seed for stochastic algorithms.

    Returns
    -------
    estimator
        Unfitted scikit-learn compatible estimator.

    Raises
    ------
    ValueError
        If model name is not recognized or optional dependency is missing.
    """
    is_classification = task_type in ("binary_classification", "multiclass_classification")

    if is_classification:
        if name == "Logistic Regression":
            return LogisticRegression(max_iter=1000, random_state=random_state)
        elif name == "Decision Tree":
            return DecisionTreeClassifier(random_state=random_state)
        elif name == "Random Forest":
            return RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is requested but not installed in the current environment.")
            return xgb.XGBClassifier(
                n_estimators=100,
                random_state=random_state,
                eval_metric="logloss" if task_type == "binary_classification" else "mlogloss",
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown classification model name: '{name}'")

    else:  # regression
        if name == "Linear Regression":
            return LinearRegression()
        elif name == "Ridge":
            return Ridge(random_state=random_state)
        elif name == "Decision Tree":
            return DecisionTreeRegressor(random_state=random_state)
        elif name == "Random Forest":
            return RandomForestRegressor(n_estimators=100, random_state=random_state)
        elif name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is requested but not installed in the current environment.")
            return xgb.XGBRegressor(
                n_estimators=100,
                random_state=random_state,
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown regression model name: '{name}'")
