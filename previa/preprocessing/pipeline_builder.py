"""
previa/preprocessing/pipeline_builder.py
------------------------------------------
Builds a leakage-safe sklearn preprocessing pipeline from the recommender output.

Why this matters
----------------
A common mistake in ML is fitting preprocessing (e.g., computing the mean
for imputation, computing the scaling parameters) on ALL the data, then
splitting into train/test. This is data leakage: the model indirectly
"sees" information from the test set.

The correct approach is:
  1. Split the data into train and test FIRST.
  2. Fit the preprocessing pipeline ONLY on training data.
  3. Apply (transform) the same fitted pipeline to both train and test data.

sklearn's Pipeline + ColumnTransformer make this automatic.

How this module works
---------------------
1. Read the recommendations list from the recommender.
2. Allow user_overrides (accept/reject/modify actions per feature).
3. Sort features into groups:
   - numeric_features   : SimpleImputer(median) + optional StandardScaler
   - categorical_features : SimpleImputer(most_frequent) + OneHotEncoder(ignore)
   - dropped_features   : constant/identifier/flag-for-review/target — excluded
4. Build a ColumnTransformer that applies the right pipeline to each group.
5. Return the ColumnTransformer (not yet fitted — fitting happens at training time).
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder


# These recommendation strings mean the feature should be excluded from the pipeline
# NOTE: "flag_for_review" is NOT here — flagged features still get encoded (OHE),
# so the model receives the signal. The UI warns the user, but training continues.
DROP_RECOMMENDATIONS = {"drop", "keep_as_is"}


def build_pipeline(
    recommendations: list[dict],
    user_overrides: dict[str, str] | None = None,
    scale_numeric: bool = True,
    scaler_type: str = "standard",
) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    """
    Build a ColumnTransformer preprocessing pipeline from recommendations.

    Parameters
    ----------
    recommendations : list[dict]
        Output of previa.preprocessing.recommender.generate_recommendations().
    user_overrides : dict[str, str] or None
        Optional mapping of feature_name -> action string to override
        automatic recommendations (e.g., {"age": "drop"}, {"education": "one_hot_encode"}).
    scale_numeric : bool
        Whether to include a scaler in the numeric pipeline. Set to False
        when training tree-based models that are scale-invariant.
    scaler_type : str
        Type of scaler to apply when scale_numeric is True:
        "standard" (StandardScaler, default), "minmax" (MinMaxScaler),
        "robust" (RobustScaler), or "none".

    Returns
    -------
    tuple of:
        pipeline             : sklearn ColumnTransformer (not yet fitted)
        numeric_features     : list of column names going through the numeric pipeline
        categorical_features : list of column names going through the categorical pipeline
        dropped_features     : list of column names excluded from the pipeline
    """
    numeric_features = []
    categorical_features = []
    dropped_features = []
    overrides = user_overrides or {}

    for rec in recommendations:
        feature = rec["feature"]
        # Allow user override if provided
        recommendation = overrides.get(feature, rec["recommendation"])
        detected_type = rec["detected_type"]

        # --- Decide which group this feature belongs to ---
        if _should_drop(recommendation):
            dropped_features.append(feature)

        elif detected_type in ("numeric", "boolean"):
            numeric_features.append(feature)

        elif detected_type == "categorical":
            categorical_features.append(feature)

        else:
            # Unknown type — skip it to be safe
            dropped_features.append(feature)

    # --- Build individual sub-pipelines ---

    # Numeric pipeline
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric and scaler_type and scaler_type.lower() != "none":
        st_clean = scaler_type.lower().strip()
        if st_clean == "minmax":
            numeric_steps.append(("scaler", MinMaxScaler()))
        elif st_clean == "robust":
            numeric_steps.append(("scaler", RobustScaler()))
        else:
            numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # --- Assemble with ColumnTransformer ---
    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError(
            "No usable feature columns remain after applying recommendations. "
            "All features were either dropped, flagged, or the target. "
            "Please review your dataset and recommendations."
        )

    pipeline = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return pipeline, numeric_features, categorical_features, dropped_features


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _should_drop(recommendation: str) -> bool:
    """
    Return True if ANY of the recommendation's actions is in DROP_RECOMMENDATIONS.

    The recommendation field may contain multiple comma-separated actions
    (e.g., "median_imputation, standard_scale"), so we check each token.
    """
    tokens = {t.strip() for t in recommendation.split(",")}
    return bool(tokens & DROP_RECOMMENDATIONS)
