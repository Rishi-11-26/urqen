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
2. Sort features into three groups:
   - numeric_features   : will get SimpleImputer(median) + StandardScaler
   - categorical_features : will get SimpleImputer(most_frequent) + OneHotEncoder
   - dropped_features   : constant/identifier/flag-for-review — excluded
3. Build a ColumnTransformer that applies the right pipeline to each group.
4. Return the ColumnTransformer (not yet fitted — fitting happens at training time).

Important: this module does NOT split data or fit anything.
It only constructs the pipeline object.
Fitting is done by the benchmarker (on training data only).
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# These recommendation strings mean the feature should be excluded from the pipeline
DROP_RECOMMENDATIONS = {"drop", "flag_for_review", "keep_as_is"}


def build_pipeline(recommendations: list[dict]) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    """
    Build a ColumnTransformer preprocessing pipeline from recommendations.

    Parameters
    ----------
    recommendations : list[dict]
        Output of previa.preprocessing.recommender.generate_recommendations().

    Returns
    -------
    tuple of:
        pipeline         : sklearn ColumnTransformer (not yet fitted)
        numeric_features : list of column names going through the numeric pipeline
        categorical_features : list of column names going through the categorical pipeline
        dropped_features : list of column names excluded from the pipeline
    """
    numeric_features = []
    categorical_features = []
    dropped_features = []

    for rec in recommendations:
        feature = rec["feature"]
        recommendation = rec["recommendation"]
        detected_type = rec["detected_type"]

        # --- Decide which group this feature belongs to ---
        if _should_drop(recommendation):
            dropped_features.append(feature)

        elif detected_type in ("numeric", "boolean"):
            # Boolean (0/1) is treated as numeric — it can be imputed and scaled
            numeric_features.append(feature)

        elif detected_type == "categorical":
            categorical_features.append(feature)

        else:
            # Unknown type — skip it to be safe
            dropped_features.append(feature)

    # --- Build individual sub-pipelines ---

    # Numeric pipeline:
    #   Step 1: SimpleImputer(strategy="median") — fills NaN with the median
    #           (fitted on train data only → no leakage)
    #   Step 2: StandardScaler() — centers to mean=0, scales to std=1
    #           (parameters fitted on train data only → no leakage)
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline:
    #   Step 1: SimpleImputer(strategy="most_frequent") — fills NaN with mode
    #   Step 2: OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    #           - handle_unknown="ignore" → unseen categories in test set become
    #             all-zero rows (safe, does not crash)
    #           - sparse_output=False → return a dense numpy array
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # --- Assemble with ColumnTransformer ---
    # remainder="drop" means any column not explicitly listed is excluded.
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
        remainder="drop",  # exclude columns not in any transformer
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
