import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# LGD MODEL DEVELOPMENT
# Uses the original portfolio stored in st.session_state.
# Does not modify the source portfolio.
# ============================================================


# ------------------------------------------------------------
# PAGE STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #071426, #0B1D36);
        color: #F1F5F9;
    }

    h1, h2, h3 {
        color: #60A5FA !important;
    }

    div[data-testid="stMetric"] {
        background: #102B4C;
        border: 1px solid #2563EB;
        padding: 14px;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# PAGE HEADER
# ------------------------------------------------------------

st.title("LGD Model Development")

st.caption(
    "Practical development of a Loss Given Default model "
    "using the original CreditRisk Lab synthetic portfolio."
)

st.info(
    "This module uses the same original portfolio generated in Data & Models. "
    "The source dataset is copied for modelling and is never overwritten."
)


# ============================================================
# 1. LOAD ORIGINAL PORTFOLIO
# ============================================================

st.subheader("1. Original portfolio")

if "credit_data" not in st.session_state:

    st.warning(
        "No original portfolio is available in this session. "
        "Go to Data & Models, generate the synthetic portfolio, "
        "and return here."
    )

    st.stop()


source_data = st.session_state["credit_data"].copy()


if source_data.empty:

    st.error("The original portfolio is empty.")

    st.stop()


m1, m2, m3 = st.columns(3)

m1.metric(
    "Portfolio records",
    f"{len(source_data):,}",
)

m2.metric(
    "Available variables",
    f"{source_data.shape[1]:,}",
)

if "default" in source_data.columns:

    default_values = pd.to_numeric(
        source_data["default"],
        errors="coerce",
    )

    default_rate = default_values.mean()

    m3.metric(
        "Portfolio default rate",
        f"{default_rate:.2%}",
    )

else:

    m3.metric(
        "Portfolio default rate",
        "Unavailable",
    )


# ------------------------------------------------------------
# Inspect LGD-related fields
# ------------------------------------------------------------

expected_lgd_fields = [
    "default",
    "ead",
    "collateral_type",
    "collateral_value",
    "collateral_ratio",
    "recovery_rate",
    "recovery_amount",
    "collection_cost",
    "lgd",
    "adjusted_lgd",
]

available_lgd_fields = [
    c for c in expected_lgd_fields
    if c in source_data.columns
]

missing_lgd_fields = [
    c for c in expected_lgd_fields
    if c not in source_data.columns
]


with st.expander(
    "Inspect LGD-related variables",
    expanded=True,
):

    if available_lgd_fields:

        st.dataframe(
            source_data[available_lgd_fields].describe(
                include="all"
            ).T,
            use_container_width=True,
        )

    if missing_lgd_fields:

        st.warning(
            "The following expected LGD fields are not present: "
            + ", ".join(missing_lgd_fields)
        )

    st.caption("Original portfolio preview")

    st.dataframe(
        source_data.head(20),
        use_container_width=True,
    )


# ============================================================
# 2. LGD MODELLING CONCEPT
# ============================================================

st.subheader("2. Define the LGD modelling approach")

st.markdown(
    """
    Loss Given Default measures the proportion of an exposure that is not
    recovered after a borrower defaults.

    A simplified representation is:

    \\[
    LGD = \\frac{EAD - Net\\ Recoveries}{EAD}
    \\]

    Equivalently:

    \\[
    LGD = 1 - \\frac{Net\\ Recoveries}{EAD}
    \\]

    Where:

    - **EAD:** Exposure at Default.
    - **Net Recoveries:** Recoveries after applicable collection costs.
    - **LGD:** Proportion of the exposure lost after recoveries.

    LGD is generally modelled using defaulted facilities because the
    recovery and loss outcome is observed following default.
    """
)

st.markdown(
    """
    The model developed in this module is a supervised regression model.

    It estimates LGD using borrower, facility and collateral characteristics.
    The predicted LGD can subsequently be combined with PD and EAD to
    calculate Expected Credit Loss.
    """
)

st.warning(
    "The synthetic portfolio is intended for methodological demonstration. "
    "The model's performance depends on how the synthetic LGD and recovery "
    "variables were generated. It should not be interpreted as evidence of "
    "real-world LGD predictive performance."
)


# ============================================================
# 3. PREPARE LGD TARGET
# ============================================================

st.subheader("3. Define the LGD target")

work = source_data.copy()


# ------------------------------------------------------------
# Convert relevant variables to numeric
# ------------------------------------------------------------

numeric_fields = [
    "ead",
    "recovery_rate",
    "recovery_amount",
    "collection_cost",
    "lgd",
    "adjusted_lgd",
    "collateral_value",
    "collateral_ratio",
    "loan_amount",
]

for col in numeric_fields:

    if col in work.columns:

        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )


# ------------------------------------------------------------
# Select target
# ------------------------------------------------------------

target_options = []

if "adjusted_lgd" in work.columns:

    target_options.append("adjusted_lgd")

if "lgd" in work.columns:

    target_options.append("lgd")

if (
    "ead" in work.columns
    and "recovery_amount" in work.columns
):

    target_options.append(
        "Derived LGD from EAD and recoveries"
    )


if not target_options:

    st.error(
        "No suitable LGD target is available. "
        "The original portfolio must contain an LGD measure, "
        "or sufficient EAD and recovery information to derive one."
    )

    st.stop()


target_choice = st.selectbox(
    "Select the LGD target",
    options=target_options,
    index=0,
    help=(
        "Adjusted LGD is preferred when available. "
        "You can also select the original LGD or derive LGD "
        "from EAD and recovery information."
    ),
)


# ------------------------------------------------------------
# Construct selected target
# ------------------------------------------------------------

if target_choice == "adjusted_lgd":

    work["_lgd_target"] = work["adjusted_lgd"]

    target_description = (
        "Existing adjusted_lgd field"
    )

elif target_choice == "lgd":

    work["_lgd_target"] = work["lgd"]

    target_description = (
        "Existing lgd field"
    )

else:

    if "collection_cost" in work.columns:

        net_recovery = (
            work["recovery_amount"]
            - work["collection_cost"].fillna(0)
        )

    else:

        net_recovery = work["recovery_amount"]

    work["_lgd_target"] = (
        1
        -
        net_recovery
        / work["ead"].replace(0, np.nan)
    )

    target_description = (
        "Derived as 1 - (recovery_amount - collection_cost) / EAD, "
        "where collection_cost is available"
    )


# ------------------------------------------------------------
# Target validation
# ------------------------------------------------------------

work["_lgd_target"] = pd.to_numeric(
    work["_lgd_target"],
    errors="coerce",
)

work["_lgd_target"] = work["_lgd_target"].replace(
    [np.inf, -np.inf],
    np.nan,
)


st.write(
    f"**Selected target:** `{target_choice}`"
)

st.caption(
    f"Target definition: {target_description}"
)


# ------------------------------------------------------------
# Optional LGD bounds
# ------------------------------------------------------------

constrain_lgd = st.checkbox(
    "Constrain LGD to the range 0 to 1",
    value=True,
    help=(
        "Standard LGD is generally expressed between 0 and 100%. "
        "However, realised economic losses can exceed EAD in some "
        "circumstances, depending on costs and the loss definition."
    ),
)


if constrain_lgd:

    work["_lgd_target"] = work["_lgd_target"].clip(
        lower=0,
        upper=1,
    )


# ============================================================
# 4. DEVELOPMENT SAMPLE
# ============================================================

st.subheader("4. Development sample")

if "default" not in work.columns:

    st.error(
        "The original dataset does not contain a default indicator. "
        "The LGD development sample cannot be restricted to defaulted accounts."
    )

    st.stop()


default_values = pd.to_numeric(
    work["default"],
    errors="coerce",
)


default_count = int(
    (default_values == 1).sum()
)


use_defaulted_only = st.checkbox(
    f"Use defaulted accounts only ({default_count:,} records)",
    value=True,
    help=(
        "LGD is a loss measure conditional on default. "
        "Restricting the sample to defaulted accounts is therefore "
        "the preferred starting point for this demonstration."
    ),
)


if use_defaulted_only:

    model_data = work.loc[
        default_values == 1
    ].copy()

else:

    model_data = work.copy()


# ------------------------------------------------------------
# Remove unusable target records
# ------------------------------------------------------------

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan,
)

model_data = model_data.dropna(
    subset=["_lgd_target"]
)


if len(model_data) < 30:

    st.error(
        f"Only {len(model_data):,} observations have usable LGD targets. "
        "At least 30 observations are required for this educational "
        "train/test workflow."
    )

    st.stop()


# ------------------------------------------------------------
# Sample statistics
# ------------------------------------------------------------

s1, s2, s3, s4 = st.columns(4)

s1.metric(
    "Development observations",
    f"{len(model_data):,}",
)

s2.metric(
    "Mean LGD",
    f"{model_data['_lgd_target'].mean():.2%}",
)

s3.metric(
    "Median LGD",
    f"{model_data['_lgd_target'].median():.2%}",
)

s4.metric(
    "LGD standard deviation",
    f"{model_data['_lgd_target'].std():.2%}",
)


# ------------------------------------------------------------
# LGD distribution
# ------------------------------------------------------------

st.markdown("#### LGD distribution")

fig, ax = plt.subplots(figsize=(9, 4))

ax.hist(
    model_data["_lgd_target"].dropna(),
    bins=30,
    edgecolor="white",
    alpha=0.85,
)

ax.set_title("LGD distribution in the development sample")
ax.set_xlabel("LGD")
ax.set_ylabel("Number of accounts")
ax.grid(alpha=0.2)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 5. PREDICTOR SELECTION
# ============================================================

st.subheader("5. Predictor selection")


# ------------------------------------------------------------
# Exclude target and post-default outcome variables
# ------------------------------------------------------------

excluded_predictors = {
    "_lgd_target",
    "lgd",
    "adjusted_lgd",
    "recovery_rate",
    "recovery_amount",
    "collection_cost",
    "default",
    "loan_status",
    "pd_probability",
}


# EAD can be considered as an exposure characteristic.
# The recovery outcomes and realised loss measures are excluded.


identifier_tokens = (
    "id",
    "account_number",
    "customer_number",
    "reference",
    "ref_no",
)


identifier_candidates = [
    c for c in model_data.columns
    if any(
        token in c.lower()
        for token in identifier_tokens
    )
]


candidate_predictors = []


for col in model_data.columns:

    if col in excluded_predictors:
        continue

    if col.startswith("_"):
        continue

    if col in identifier_candidates:
        continue

    if model_data[col].isna().all():
        continue

    if model_data[col].nunique(dropna=True) <= 1:
        continue

    if pd.api.types.is_datetime64_any_dtype(
        model_data[col]
    ):
        continue

    candidate_predictors.append(col)


if not candidate_predictors:

    st.error(
        "No eligible LGD predictors were identified."
    )

    st.stop()


# ------------------------------------------------------------
# Default predictor selection
# ------------------------------------------------------------

selected_predictors = st.multiselect(
    "Select variables for the LGD model",
    options=candidate_predictors,
    default=candidate_predictors,
    help=(
        "Select borrower, facility and collateral characteristics. "
        "Recovery outcomes and realised LGD fields are excluded "
        "to prevent target leakage."
    ),
)


if not selected_predictors:

    st.warning(
        "Select at least one predictor to continue."
    )

    st.stop()


# ------------------------------------------------------------
# Predictor profile
# ------------------------------------------------------------

with st.expander(
    "Inspect predictor characteristics"
):

    predictor_profile = pd.DataFrame(
        {
            "Predictor": selected_predictors,

            "Data type": [
                str(model_data[c].dtype)
                for c in selected_predictors
            ],

            "Unique values": [
                model_data[c].nunique(dropna=True)
                for c in selected_predictors
            ],

            "Missing (%)": [
                model_data[c].isna().mean() * 100
                for c in selected_predictors
            ],
        }
    )

    st.dataframe(
        predictor_profile,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# 6. TRAIN / TEST CONFIGURATION
# ============================================================

st.subheader("6. Train/test configuration")


col1, col2, col3 = st.columns(3)


with col1:

    test_size = st.slider(
        "Test sample proportion",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
    )


with col2:

    random_state = st.number_input(
        "Random state",
        min_value=1,
        max_value=9999,
        value=42,
        step=1,
    )


with col3:

    model_choice = st.selectbox(
        "Regression algorithm",
        [
            "Random Forest",
            "Extra Trees",
            "Histogram Gradient Boosting",
        ],
    )


# ============================================================
# 7. MODEL DEVELOPMENT
# ============================================================

st.subheader("7. Fit the LGD model")


if st.button(
    "Train LGD model",
    type="primary",
    use_container_width=True,
):

    st.session_state.pop(
        "lgd_model_pipeline",
        None,
    )

    st.session_state.pop(
        "lgd_model_results",
        None,
    )

    st.session_state.pop(
        "lgd_model_metadata",
        None,
    )


    # --------------------------------------------------------
    # Define X and y
    # --------------------------------------------------------

    X = model_data[
        selected_predictors
    ].copy()

    y = model_data[
        "_lgd_target"
    ].astype(float).copy()


    valid_target = (
        y.notna()
        & np.isfinite(y)
    )

    X = X.loc[
        valid_target
    ].copy()

    y = y.loc[
        valid_target
    ].copy()


    if len(X) < 30:

        st.error(
            "Insufficient observations after target validation."
        )

        st.stop()


    # --------------------------------------------------------
    # Identify feature types
    # --------------------------------------------------------

    numeric_features = [
        c for c in selected_predictors
        if pd.api.types.is_numeric_dtype(X[c])
    ]

    categorical_features = [
        c for c in selected_predictors
        if c not in numeric_features
    ]


    # --------------------------------------------------------
    # Numeric preprocessing
    # --------------------------------------------------------

    transformers = []


    if numeric_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )


    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),

                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )


    # --------------------------------------------------------
    # Combined preprocessing
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


    # --------------------------------------------------------
    # Select regression algorithm
    # --------------------------------------------------------

    if model_choice == "Random Forest":

        regressor = RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=int(random_state),
            n_jobs=-1,
        )


    elif model_choice == "Extra Trees":

        regressor = ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=int(random_state),
            n_jobs=-1,
        )


    else:

        regressor = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=int(random_state),
        )


    # --------------------------------------------------------
    # Build modelling pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),

            (
                "regressor",
                regressor,
            ),
        ]
    )


    # --------------------------------------------------------
    # Split development sample
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
    )


    try:

        # Fit model

        pipeline.fit(
            X_train,
            y_train,
        )


        # Test predictions

        predictions = pipeline.predict(
            X_test
        )


        # Constrain predicted LGD if selected

        if constrain_lgd:

            predictions = np.clip(
                predictions,
                0,
                1,
            )


        # ----------------------------------------------------
        # Performance metrics
        # ----------------------------------------------------

        metrics = {
            "MAE": float(
                mean_absolute_error(
                    y_test,
                    predictions,
                )
            ),

            "RMSE": float(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions,
                    )
                )
            ),

            "R2": float(
                r2_score(
                    y_test,
                    predictions,
                )
            ),
        }


        # ----------------------------------------------------
        # Test prediction table
        # ----------------------------------------------------

        result_table = pd.DataFrame(
            {
                "Actual_LGD": y_test,
                "Predicted_LGD": predictions,
            },
            index=X_test.index,
        )


        result_table["LGD_Error"] = (
            result_table["Actual_LGD"]
            - result_table["Predicted_LGD"]
        )


        result_table["Absolute_Error"] = (
            result_table["LGD_Error"].abs()
        )


        # ----------------------------------------------------
        # Store model
        # ----------------------------------------------------

        st.session_state[
            "lgd_model_pipeline"
        ] = pipeline


        st.session_state[
            "lgd_model_results"
        ] = result_table


        st.session_state[
            "lgd_model_metadata"
        ] = {
            "predictors": selected_predictors,

            "numeric_features": numeric_features,

            "categorical_features": categorical_features,

            "target": target_choice,

            "target_description": target_description,

            "model_choice": model_choice,

            "constrain_lgd": constrain_lgd,

            "test_size": float(test_size),

            "random_state": int(random_state),

            "train_rows": len(X_train),

            "test_rows": len(X_test),

            "metrics": metrics,
        }


        st.success(
            "LGD model successfully trained."
        )


    except Exception as exc:

        st.error(
            f"LGD model development failed: {exc}"
        )

        st.exception(exc)


# ============================================================
# 8. MODEL PERFORMANCE AND VALIDATION
# ============================================================

if (
    "lgd_model_results" in st.session_state
    and "lgd_model_metadata" in st.session_state
):

    results = st.session_state[
        "lgd_model_results"
    ]

    metadata = st.session_state[
        "lgd_model_metadata"
    ]


    st.subheader("8. Model performance")


    # --------------------------------------------------------
    # Performance metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    col1.metric(
        "LGD MAE",
        f"{metadata['metrics']['MAE']:.4f}",
    )


    col2.metric(
        "LGD RMSE",
        f"{metadata['metrics']['RMSE']:.4f}",
    )


    col3.metric(
        "R-squared",
        f"{metadata['metrics']['R2']:.4f}",
    )


    st.markdown(
        "#### Performance metric definitions"
    )


    metric_definitions = pd.DataFrame(
        {
            "Metric": [
                "Mean Absolute Error (MAE)",
                "Root Mean Squared Error (RMSE)",
                "R-squared",
            ],

            "Interpretation": [
                "Average absolute difference between actual and predicted LGD.",

                "Penalises larger prediction errors more heavily.",

                "Proportion of target variation explained by the model.",
            ],
        }
    )


    st.dataframe(
        metric_definitions,
        hide_index=True,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # Actual vs predicted LGD
    # --------------------------------------------------------

    st.markdown(
        "#### Actual versus predicted LGD"
    )


    fig, ax = plt.subplots(
        figsize=(8, 4)
    )


    ax.scatter(
        results["Actual_LGD"],
        results["Predicted_LGD"],
        alpha=0.55,
    )


    low = min(
        results["Actual_LGD"].min(),
        results["Predicted_LGD"].min(),
    )


    high = max(
        results["Actual_LGD"].max(),
        results["Predicted_LGD"].max(),
    )


    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
    )


    ax.set_title(
        "Actual vs predicted LGD"
    )

    ax.set_xlabel(
        "Actual LGD"
    )

    ax.set_ylabel(
        "Predicted LGD"
    )

    ax.grid(alpha=0.2)


    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------------
    # Prediction error distribution
    # --------------------------------------------------------

    st.markdown(
        "#### LGD prediction error distribution"
    )


    fig, ax = plt.subplots(
        figsize=(8, 4)
    )


    ax.hist(
        results["LGD_Error"],
        bins=30,
        edgecolor="white",
        alpha=0.85,
    )


    ax.axvline(
        0,
        linestyle="--",
    )


    ax.set_title(
        "Distribution of LGD prediction errors"
    )

    ax.set_xlabel(
        "Actual LGD - Predicted LGD"
    )

    ax.set_ylabel(
        "Number of accounts"
    )

    ax.grid(alpha=0.2)


    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------------
    # Actual vs predicted LGD by decile
    # --------------------------------------------------------

    st.markdown(
        "#### LGD calibration by predicted decile"
    )


    calibration = results[
        [
            "Actual_LGD",
            "Predicted_LGD",
        ]
    ].copy()


    number_of_bins = min(
        10,
        calibration["Predicted_LGD"].nunique(),
        len(calibration),
    )


    if number_of_bins >= 2:

        calibration["Predicted_Decile"] = (
            pd.qcut(
                calibration["Predicted_LGD"].rank(
                    method="first"
                ),
                q=number_of_bins,
                labels=False,
                duplicates="drop",
            ) + 1
        )


        calibration_table = (
            calibration
            .groupby(
                "Predicted_Decile",
                as_index=False,
            )
            .agg(
                Accounts=(
                    "Actual_LGD",
                    "size",
                ),

                Mean_Predicted_LGD=(
                    "Predicted_LGD",
                    "mean",
                ),

                Mean_Actual_LGD=(
                    "Actual_LGD",
                    "mean",
                ),
            )
        )


        st.dataframe(
            calibration_table,
            hide_index=True,
            use_container_width=True,
        )


        fig, ax = plt.subplots(
            figsize=(9, 4)
        )


        ax.plot(
            calibration_table["Predicted_Decile"],
            calibration_table["Mean_Predicted_LGD"],
            marker="o",
            label="Mean predicted LGD",
        )


        ax.plot(
            calibration_table["Predicted_Decile"],
            calibration_table["Mean_Actual_LGD"],
            marker="o",
            label="Mean actual LGD",
        )


        ax.set_xlabel(
            "Predicted LGD decile"
        )

        ax.set_ylabel(
            "LGD"
        )

        ax.set_title(
            "Actual versus predicted LGD by decile"
        )

        ax.legend()

        ax.grid(alpha=0.2)


        st.pyplot(fig)

        plt.close(fig)


    # ========================================================
    # 9. PREDICTOR IMPORTANCE
    # ========================================================

    st.subheader("9. Predictor importance")


    fitted_pipeline = st.session_state[
        "lgd_model_pipeline"
    ]


    fitted_regressor = fitted_pipeline.named_steps[
        "regressor"
    ]


    fitted_preprocessor = fitted_pipeline.named_steps[
        "preprocessor"
    ]


    if hasattr(
        fitted_regressor,
        "feature_importances_",
    ):

        try:

            transformed_names = (
                fitted_preprocessor.get_feature_names_out()
            )


            importance = pd.DataFrame(
                {
                    "Transformed Predictor": transformed_names,

                    "Importance": (
                        fitted_regressor.feature_importances_
                    ),
                }
            ).sort_values(
                "Importance",
                ascending=False,
            )


            st.dataframe(
                importance.head(30),
                hide_index=True,
                use_container_width=True,
            )


            top_importance = (
                importance
                .head(20)
                .sort_values(
                    "Importance"
                )
            )


            fig, ax = plt.subplots(
                figsize=(
                    9,
                    max(
                        4,
                        min(
                            9,
                            len(top_importance) * 0.35,
                        ),
                    ),
                )
            )


            ax.barh(
                top_importance[
                    "Transformed Predictor"
                ],
                top_importance[
                    "Importance"
                ],
            )


            ax.set_title(
                "Top LGD predictor importance"
            )

            ax.set_xlabel(
                "Importance"
            )

            ax.grid(
                axis="x",
                alpha=0.2,
            )


            st.pyplot(fig)

            plt.close(fig)


        except Exception as exc:

            st.warning(
                f"Could not calculate feature importance: {exc}"
            )


    else:

        st.info(
            "The selected algorithm does not expose native feature importance."
        )


    # ========================================================
    # 10. SCORE THE ORIGINAL PORTFOLIO
    # ========================================================

    st.subheader(
        "10. Score the original portfolio"
    )


    scoring_data = source_data.copy()


    scoring_predictors = metadata[
        "predictors"
    ]


    missing_predictors = [
        c for c in scoring_predictors
        if c not in scoring_data.columns
    ]


    if missing_predictors:

        st.error(
            "The original portfolio is missing the following model predictors: "
            + ", ".join(missing_predictors)
        )


    else:

        score_X = scoring_data[
            scoring_predictors
        ].copy()


        # Predict LGD for all records with available predictors.
        # The output is a model estimate, not an observed LGD
        # for non-defaulted accounts.

        try:

            predicted_lgd = fitted_pipeline.predict(
                score_X
            )


            if metadata["constrain_lgd"]:

                predicted_lgd = np.clip(
                    predicted_lgd,
                    0,
                    1,
                )


            scored = scoring_data.copy()


            scored["predicted_lgd"] = predicted_lgd


            st.metric(
                "Average modelled LGD",
                f"{scored['predicted_lgd'].mean():.2%}",
            )


            st.dataframe(
                scored.head(100),
                use_container_width=True,
            )


            st.download_button(
                "Download original portfolio with predicted LGD",

                data=scored.to_csv(
                    index=False
                ).encode("utf-8"),

                file_name="original_portfolio_modelled_lgd.csv",

                mime="text/csv",
            )


        except Exception as exc:

            st.error(
                f"Portfolio scoring failed: {exc}"
            )

            st.exception(exc)


    # ========================================================
    # 11. MODEL SPECIFICATION
    # ========================================================

    st.subheader("11. Model specification")


    specification = {
        "Model": "Loss Given Default Regression",

        "Algorithm": metadata["model_choice"],

        "Target variable": metadata["target"],

        "Target definition": metadata["target_description"],

        "Predictors": metadata["predictors"],

        "Numeric predictors": metadata["numeric_features"],

        "Categorical predictors": metadata["categorical_features"],

        "Training observations": metadata["train_rows"],

        "Test observations": metadata["test_rows"],

        "Test sample proportion": metadata["test_size"],

        "Random state": metadata["random_state"],

        "LGD constrained to 0-1": metadata["constrain_lgd"],
    }


    st.json(
        specification
    )


    # ========================================================
    # 12. MODEL VALIDATION AND GOVERNANCE
    # ========================================================

    st.subheader("12. Model validation and governance")


    st.markdown(
        """
        A production LGD model requires additional validation beyond
        a random train/test split.

        **Important validation considerations:**

        - **Representativeness:** The development sample should represent
          the portfolio's defaulted facilities and relevant economic conditions.

        - **Recovery definition:** Recoveries should be measured consistently,
          including the treatment of collection costs and recovery timing.

        - **Discounting:** Recovery cash flows should be discounted where
          required by the applicable modelling framework.

        - **Collateral:** Collateral type, value, enforceability and
          realisation costs should be considered.

        - **Segmentation:** Different products, collateral classes and
          recovery processes may require separate LGD estimates.

        - **Out-of-time validation:** The model should be tested on a later
          observation period to evaluate stability.

        - **Backtesting:** Predicted LGD should be compared with realised
          loss outcomes as they become available.

        - **Conservatism:** Model uncertainty and data limitations should
          be addressed through appropriate adjustments.

        - **Monitoring:** LGD performance, recovery patterns and model
          stability should be monitored regularly.
        """
    )


    st.warning(
        "This module uses synthetic data and a random train/test split. "
        "It does not implement a complete workout LGD framework, "
        "discounted recovery cash flows, downturn LGD estimation, "
        "economic-cycle adjustments or regulatory model overlays. "
        "Those are separate components of a production LGD framework."
    )


# ============================================================
# PAGE FOOTER
# ============================================================

st.divider()

st.caption(
    "CreditRisk Lab · LGD Model Development · "
    "Synthetic demonstration data · Not production-ready"
)