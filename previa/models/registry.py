"""
previa/models/registry.py
-------------------------
Centralized Model Registry for Previa.

Provides a unified catalog of algorithms for tabular regression
and classification tasks:
- Linear SVM (LVSM)
- Naive Bayes
- RBF Network / RBF Kernel (RBFN)
- Random Forest
- XGBoost (optional)
- Logistic Regression / Decision Tree / Linear Regression / Ridge

Design Principles:
- Fixed random state (42) for deterministic reproducibility.
- Explicit metadata describing algorithmic family and scale sensitivity.
- Graceful handling of optional dependencies (XGBoost).
- Clean factory functions so downstream training code doesn't hardcode model constructors.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, BayesianRidge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB


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
        Whether to include expanded algorithms (e.g. LVSM, Naive Bayes, RBFN, XGBoost)
        in addition to standard baseline suites.

    Returns
    -------
    list of dicts
    """
    is_classification = task_type in ("binary_classification", "multiclass_classification")

    if is_classification:
        core_models = [
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
        if not include_optional:
            return core_models

        expanded = list(core_models)
        expanded.extend([
            {
                "name": "Linear SVM (LVSM)",
                "family": "svm",
                "scale_sensitive": True,
                "description": "Support Vector Machine with linear hyperplane margin. Effective in high-dimensional structured spaces.",
            },
            {
                "name": "Naive Bayes",
                "family": "probabilistic",
                "scale_sensitive": False,
                "description": "Gaussian Naive Bayes classifier based on Bayes' theorem with feature independence assumption. Extremely fast and lightweight.",
            },
            {
                "name": "RBF Network (RBFN)",
                "family": "kernel",
                "scale_sensitive": True,
                "description": "Radial Basis Function (RBF) kernel SVM. Projects features into infinite-dimensional Hilbert space for non-linear decision boundaries.",
            },
        ])
        if XGBOOST_AVAILABLE:
            expanded.append({
                "name": "XGBoost",
                "family": "boosting",
                "scale_sensitive": False,
                "description": "Gradient boosted decision tree ensemble with regularized loss. High predictive performance on structured tables.",
            })
        return expanded

    else:  # regression
        core_models = [
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
        if not include_optional:
            return core_models

        expanded = list(core_models)
        expanded.extend([
            {
                "name": "Linear SVM (LVSM)",
                "family": "svm",
                "scale_sensitive": True,
                "description": "Support Vector Regressor with linear kernel. Optimizes epsilon-insensitive loss.",
            },
            {
                "name": "Naive Bayes",
                "family": "probabilistic",
                "scale_sensitive": True,
                "description": "Bayesian Ridge regression. Formulates regression with spherical Gaussian priors over parameters.",
            },
            {
                "name": "RBF Network (RBFN)",
                "family": "kernel",
                "scale_sensitive": True,
                "description": "Radial Basis Function (RBF) Support Vector Regressor. Fits nonlinear continuous manifolds.",
            },
        ])
        if XGBOOST_AVAILABLE:
            expanded.append({
                "name": "XGBoost",
                "family": "boosting",
                "scale_sensitive": False,
                "description": "Gradient boosted trees optimizing squared error. State-of-the-art tabular baseline.",
            })
        return expanded


def create_model_instance(name: str, task_type: str, random_state: int = RANDOM_STATE, **hyperparams):
    """
    Instantiate an unfitted model object by name and task type with optional hyperparameters.

    Parameters
    ----------
    name : str
        Model name (e.g. 'Linear SVM (LVSM)', 'Naive Bayes', 'RBF Network (RBFN)', 'Random Forest', 'XGBoost').
    task_type : str
        "binary_classification", "multiclass_classification", or "regression".
    random_state : int
        Seed for stochastic algorithms.
    **hyperparams : dict
        Optional keyword arguments to configure model hyperparameters
        (e.g. n_estimators, max_depth, learning_rate, min_child_weight, C).

    Returns
    -------
    estimator
        Unfitted scikit-learn compatible estimator.
    """
    is_classification = task_type in ("binary_classification", "multiclass_classification")
    name_clean = name.strip()

    if is_classification:
        if name_clean == "Logistic Regression":
            c_val = float(hyperparams.get("C", 1.0))
            max_iter = int(hyperparams.get("max_iter", 1000))
            return LogisticRegression(C=c_val, max_iter=max_iter, random_state=random_state)

        elif name_clean == "Decision Tree":
            max_depth = hyperparams.get("max_depth", None)
            if max_depth is not None:
                max_depth = int(max_depth)
            min_split = int(hyperparams.get("min_samples_split", 2))
            return DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_split, random_state=random_state)

        elif name_clean == "Random Forest":
            n_est = int(hyperparams.get("n_estimators", 100))
            max_depth = hyperparams.get("max_depth", None)
            if max_depth is not None:
                max_depth = int(max_depth)
            return RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=random_state)

        elif name_clean in ("Linear SVM (LVSM)", "LVSM", "LSVM", "Linear SVM"):
            c_val = float(hyperparams.get("C", 1.0))
            return SVC(C=c_val, kernel="linear", probability=True, random_state=random_state)

        elif name_clean in ("Naive Bayes", "GaussianNB"):
            var_sm = float(hyperparams.get("var_smoothing", 1e-9))
            return GaussianNB(var_smoothing=var_sm)

        elif name_clean in ("RBF Network (RBFN)", "RBFN", "RBF SVM", "RBF Network"):
            c_val = float(hyperparams.get("C", 1.0))
            gamma = hyperparams.get("gamma", "scale")
            return SVC(C=c_val, kernel="rbf", gamma=gamma, probability=True, random_state=random_state)

        elif name_clean == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is requested but not installed in the current environment.")
            n_est = int(hyperparams.get("n_estimators", 100))
            max_d = int(hyperparams.get("max_depth", 6))
            lr = float(hyperparams.get("learning_rate", hyperparams.get("eta", 0.1)))
            mcw = float(hyperparams.get("min_child_weight", 1.0))
            sub = float(hyperparams.get("subsample", 1.0))
            return xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                learning_rate=lr,
                min_child_weight=mcw,
                subsample=sub,
                random_state=random_state,
                eval_metric="logloss" if task_type == "binary_classification" else "mlogloss",
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown classification model name: '{name}'")

    else:  # regression
        if name_clean == "Linear Regression":
            return LinearRegression()

        elif name_clean == "Ridge":
            alpha = float(hyperparams.get("alpha", 1.0))
            return Ridge(alpha=alpha, random_state=random_state)

        elif name_clean == "Decision Tree":
            max_depth = hyperparams.get("max_depth", None)
            if max_depth is not None:
                max_depth = int(max_depth)
            min_split = int(hyperparams.get("min_samples_split", 2))
            return DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_split, random_state=random_state)

        elif name_clean == "Random Forest":
            n_est = int(hyperparams.get("n_estimators", 100))
            max_depth = hyperparams.get("max_depth", None)
            if max_depth is not None:
                max_depth = int(max_depth)
            return RandomForestRegressor(n_estimators=n_est, max_depth=max_depth, random_state=random_state)

        elif name_clean in ("Linear SVM (LVSM)", "LVSM", "LSVM", "Linear SVM"):
            c_val = float(hyperparams.get("C", 1.0))
            return SVR(C=c_val, kernel="linear")

        elif name_clean in ("Naive Bayes", "Bayesian Ridge", "GaussianNB"):
            return BayesianRidge()

        elif name_clean in ("RBF Network (RBFN)", "RBFN", "RBF SVM", "RBF Network"):
            c_val = float(hyperparams.get("C", 1.0))
            gamma = hyperparams.get("gamma", "scale")
            return SVR(C=c_val, kernel="rbf", gamma=gamma)

        elif name_clean == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is requested but not installed in the current environment.")
            n_est = int(hyperparams.get("n_estimators", 100))
            max_d = int(hyperparams.get("max_depth", 6))
            lr = float(hyperparams.get("learning_rate", hyperparams.get("eta", 0.1)))
            mcw = float(hyperparams.get("min_child_weight", 1.0))
            sub = float(hyperparams.get("subsample", 1.0))
            return xgb.XGBRegressor(
                n_estimators=n_est,
                max_depth=max_d,
                learning_rate=lr,
                min_child_weight=mcw,
                subsample=sub,
                random_state=random_state,
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown regression model name: '{name}'")
