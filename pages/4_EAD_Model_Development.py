import io
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
# EAD MODEL DEVELOPMENT
# Uses the original portfolio stored in st.session_state.
# Does not modify the source portfolio.
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #071426, #0B1D36);
        color: #F1F5F9;
    }
    h1, h2, h3 { color: #60A5FA !important; }
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

st.title("EAD Model Development")
st.caption(
    "Practical development of an Exposure at Default model using the original "
    "CreditRisk Lab synthetic portfolio."
)

st.info(
    "This page uses the same original portfolio generated in Data & Models. "
    "The source dataset is copied for modelling and is never overwritten."
)


# ------------------------------------------------------------
# 1. LOAD THE ORIGINAL PORTFOLIO
# ------------------------------------------------------------

if "credit_data" not in st.session_state:
    st.warning(
        "No original portfolio is available in this session. "
        "Go to Data & Models, generate the synthetic portfolio, and return here."
    )
    st.stop()

source_data = st.session_state["credit_data"].copy()

if source_data.empty:
    st.error("The original portfolio is empty.")
    st.stop()

st.subheader("1. Original portfolio")

m1, m2, m3 = st.columns(3)

m1.metric("Portfolio records", f"{len(source_data):,}")
m2.metric("Available variables", f"{source_data.shape[1]:,}")

m3.metric(
    "Default rate",
    f"{source_data['default'].mean():.2%}"
    if "default" in source_data.columns
    and pd.api.types.is_numeric_dtype(source_data["default"])
    else "Not available",
)

with st.expander("Inspect original EAD-related variables", expanded=True):

    expected_ead_fields = [
        "credit_limit",
        "current_balance",
        "current_utilization",
        "undrawn_amount",
        "ccf",
        "ead",
    ]

    available_ead_fields = [
        c for c in expected_ead_fields if c in source_data.columns
    ]

    missing_ead_fields = [
        c for c in expected_ead_fields if c not in source_data.columns
    ]

    if available_ead_fields:
        st.dataframe(
            source_data[available_ead_fields].describe(include="all").T,
            use_container_width=True,
        )

    if missing_ead_fields:
        st.warning(
            "The following expected EAD fields are not present in this portfolio: "
            + ", ".join(missing_ead_fields)
        )

    st.caption("Preview of the original source data")
    st.dataframe(source_data.head(20), use_container_width=True)


# ------------------------------------------------------------
# 2. EAD / CCF MODELLING CONCEPT
# ------------------------------------------------------------

st.subheader("2. Define the modelling approach")

st.markdown(
    """
    For revolving facilities, EAD is commonly estimated using a credit
    conversion factor (CCF):

    \\[
    EAD = Drawn\\ Exposure + CCF \\times Undrawn\\ Commitment
    \\]

    The CCF estimates the proportion of the currently undrawn commitment
    that is expected to be drawn before default.

    This page develops a **CCF regression model** and then uses the predicted
    CCF to calculate EAD. It does not treat the already-calculated EAD as an
    independent predictor.
    """
)

st.warning(
    "The synthetic portfolio is for methodology demonstration. If its EAD and "
    "CCF values were generated from a predefined formula, the model will learn "
    "that synthetic relationship. This is not evidence of real-world predictive "
    "performance."
)

required_exposure_fields = [
    "current_balance",
    "undrawn_amount",
]

missing_exposure_fields = [
    c for c in required_exposure_fields if c not in source_data.columns
]

if missing_exposure_fields:
    st.error(
        "The original portfolio is missing the exposure fields required to "
        "calculate EAD: " + ", ".join(missing_exposure_fields)
    )
    st.stop()

work = source_data.copy()

# Numeric conversion for core exposure fields.
for col in [
    "current_balance",
    "undrawn_amount",
    "credit_limit",
    "current_utilization",
    "ccf",
    "ead",
]:
    if col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")

# Derive undrawn amount from limit less drawn balance only if needed.
if (
    work["undrawn_amount"].isna().all()
    and "credit_limit" in work.columns
):
    work["undrawn_amount"] = (
        work["credit_limit"] - work["current_balance"]
    ).clip(lower=0)

# The observed CCF is the preferred modelling target.
if "ccf" in work.columns:

    work["_ead_target_ccf"] = pd.to_numeric(
        work["ccf"],
        errors="coerce",
    )

    target_source_label = "Existing ccf field"

elif "ead" in work.columns:

    denom = work["undrawn_amount"].replace(0, np.nan)

    work["_ead_target_ccf"] = (
        (work["ead"] - work["current_balance"]) / denom
    )

    target_source_label = (
        "Derived from (ead - current_balance) / undrawn_amount"
    )

else:

    st.error(
        "Neither a ccf nor an ead target is available. The original dataset "
        "needs an observed CCF or EAD-at-default measure to train a supervised model."
    )

    st.stop()


# Clip only for training stability and make the intervention transparent.
# CCF can exceed 100% in some real portfolios; therefore clipping is configurable.

clip_ccf = st.checkbox(
    "Constrain the modelling target to the range 0 to 1",
    value=True,
    help=(
        "Enable this for the standard 0%-100% CCF demonstration. "
        "In real portfolios, CCF can exceed 100% depending on product definitions "
        "and observed exposure movements."
    ),
)

if clip_ccf:
    work["_ead_target_ccf"] = work["_ead_target_ccf"].clip(0, 1)

st.caption(f"CCF target source: {target_source_label}")


# ------------------------------------------------------------
# 3. DEFINE THE DEVELOPMENT SAMPLE
# ------------------------------------------------------------

st.subheader("3. Development sample")

use_defaulted_only = False

if "default" in work.columns:

    default_values = pd.to_numeric(
        work["default"],
        errors="coerce",
    )

    default_count = int((default_values == 1).sum())

    use_defaulted_only = st.checkbox(
        f"Use defaulted accounts only for CCF model development ({default_count:,} records)",
        value=default_count >= 30,
        help=(
            "In conventional CCF estimation, the realised exposure movement is "
            "observed around default. This option restricts model development to "
            "records labelled as defaulted."
        ),
    )

if use_defaulted_only:

    numeric_default = pd.to_numeric(
        work["default"],
        errors="coerce",
    )

    model_data = work.loc[numeric_default == 1].copy()

else:

    model_data = work.copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan,
)

model_data = model_data.dropna(
    subset=[
        "_ead_target_ccf",
        "current_balance",
        "undrawn_amount",
    ]
)

model_data = model_data.loc[
    model_data["undrawn_amount"] > 0
].copy()

if len(model_data) < 30:

    st.error(
        f"Only {len(model_data):,} usable observations remain. "
        "At least 30 are required for this educational train/test workflow. "
        "Check the CCF/EAD fields or disable the default-only filter if appropriate."
    )

    st.stop()

sample_a, sample_b, sample_c = st.columns(3)

sample_a.metric(
    "Usable observations",
    f"{len(model_data):,}",
)

sample_b.metric(
    "Mean CCF",
    f"{model_data['_ead_target_ccf'].mean():.2%}",
)

sample_c.metric(
    "Median CCF",
    f"{model_data['_ead_target_ccf'].median():.2%}",
)

fig, ax = plt.subplots(figsize=(9, 3.5))

ax.hist(
    model_data["_ead_target_ccf"].dropna(),
    bins=30,
    edgecolor="white",
    alpha=0.85,
)

ax.set_title("Observed CCF distribution")
ax.set_xlabel("CCF")
ax.set_ylabel("Number of accounts")
ax.grid(alpha=0.2)

st.pyplot(fig)
plt.close(fig)



# ------------------------------------------------------------
# 4. SELECT PREDICTORS
# ------------------------------------------------------------

st.subheader("4. Predictor selection")

# Do not use direct target fields or post-default outcome fields as predictors.
excluded_predictors = {
    "_ead_target_ccf",
    "ccf",
    "ead",
    "default",
    "loan_status",
    "recovery_rate",
    "recovery_amount",
    "collection_cost",
    "lgd",
    "adjusted_lgd",
    "pd_probability",
}

# Exposure components are used in the final EAD formula,
# not as CCF model predictors.
# Current utilisation and facility characteristics remain eligible.

excluded_predictors.update({
    "current_balance",
    "undrawn_amount",
})

identifier_tokens = (
    "id",
    "account_number",
    "customer_number",
    "reference",
    "ref_no",
)

identifier_candidates = [
    c for c in model_data.columns
    if any(token in c.lower() for token in identifier_tokens)
]

candidate_predictors = []

for col in model_data.columns:

    if col in excluded_predictors:
        continue

    if col.startswith("_"):
        continue

    if col in identifier_candidates:
        continue

    if model_data[col].nunique(dropna=True) <= 1:
        continue

    if model_data[col].isna().all():
        continue

    if pd.api.types.is_datetime64_any_dtype(model_data[col]):
        continue

    candidate_predictors.append(col)


# Include derived utilisation if it can be calculated
# and the original field is absent.

if "current_utilization" not in candidate_predictors:

    if "credit_limit" in model_data.columns:

        model_data["_derived_current_utilization"] = (
            model_data["current_balance"]
            / model_data["credit_limit"].replace(0, np.nan)
        )

        if (
            model_data["_derived_current_utilization"]
            .nunique(dropna=True) > 1
        ):
            candidate_predictors.append(
                "_derived_current_utilization"
            )


if not candidate_predictors:

    st.error(
        "No eligible CCF predictors were identified in the original portfolio."
    )

    st.stop()

default_predictors = candidate_predictors.copy()

selected_predictors = st.multiselect(
    "Select variables to explain CCF",
    options=candidate_predictors,
    default=default_predictors,
    help=(
        "The model excludes CCF, EAD, default outcome, current drawn balance "
        "and undrawn amount from the CCF predictor set to avoid target leakage "
        "and preserve the EAD calculation structure."
    ),
)

if not selected_predictors:

    st.warning("Select at least one predictor to continue.")
    st.stop()


with st.expander("Predictor data types"):

    type_table = pd.DataFrame({
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
    })

    st.dataframe(
        type_table,
        hide_index=True,
        use_container_width=True,
    )


# ------------------------------------------------------------
# 5. TRAIN / TEST CONFIGURATION
# ------------------------------------------------------------

st.subheader("5. Train/test configuration")

c1, c2, c3 = st.columns(3)

with c1:

    test_size = st.slider(
        "Test sample proportion",
        0.10,
        0.40,
        0.20,
        0.05,
    )

with c2:

    random_state = st.number_input(
        "Random state",
        1,
        9999,
        42,
        1,
    )

with c3:

    model_choice = st.selectbox(
        "Regression algorithm",
        [
            "Random Forest",
            "Extra Trees",
            "Histogram Gradient Boosting",
        ],
    )


if st.button(
    "Prepare EAD modelling data",
    type="primary",
    use_container_width=True,
):

    st.session_state.pop("ead_model_pipeline", None)
    st.session_state.pop("ead_model_results", None)
    st.session_state.pop("ead_model_metadata", None)

    X = model_data[selected_predictors].copy()

    y = model_data["_ead_target_ccf"].astype(float).copy()

    # Remove rows with missing target, but let the
    # preprocessing pipeline handle missing predictor values.

    valid_target = y.notna() & np.isfinite(y)

    X = X.loc[valid_target].copy()
    y = y.loc[valid_target].copy()

    if len(X) < 30:

        st.error(
            "Insufficient observations after target validation."
        )

        st.stop()


    # Identify numeric and categorical predictors.

    numeric_features = [
        c for c in selected_predictors
        if pd.api.types.is_numeric_dtype(X[c])
    ]

    categorical_features = [
        c for c in selected_predictors
        if c not in numeric_features
    ]


    # Numeric preprocessing.

    transformers = []

    if numeric_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
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


    # Categorical preprocessing.

    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
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


    # Combine preprocessing pipelines.

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


    # Select regression algorithm.

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


    # Complete modelling pipeline.

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


    # Split development sample.

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
    )


    try:

        # Fit the CCF model.

        pipeline.fit(
            X_train,
            y_train,
        )


        # Predict CCF on the test sample.

        predictions = pipeline.predict(X_test)

        if clip_ccf:
            predictions = np.clip(
                predictions,
                0,
                1,
            )


        # Reconstruct predicted EAD.

        predicted_ead = (
            model_data.loc[
                X_test.index,
                "current_balance",
            ].to_numpy()
            +
            predictions
            *
            model_data.loc[
                X_test.index,
                "undrawn_amount",
            ].to_numpy()
        )


        # Reconstruct actual EAD using observed CCF.

        actual_ead = (
            model_data.loc[
                X_test.index,
                "current_balance",
            ].to_numpy()
            +
            y_test.to_numpy()
            *
            model_data.loc[
                X_test.index,
                "undrawn_amount",
            ].to_numpy()
        )


        # Calculate performance metrics.

        metrics = {
            "MAE_CCF": float(
                mean_absolute_error(
                    y_test,
                    predictions,
                )
            ),

            "RMSE_CCF": float(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions,
                    )
                )
            ),

            "R2_CCF": float(
                r2_score(
                    y_test,
                    predictions,
                )
            ),

            "MAE_EAD": float(
                mean_absolute_error(
                    actual_ead,
                    predicted_ead,
                )
            ),

            "RMSE_EAD": float(
                np.sqrt(
                    mean_squared_error(
                        actual_ead,
                        predicted_ead,
                    )
                )
            ),

            "R2_EAD": float(
                r2_score(
                    actual_ead,
                    predicted_ead,
                )
            ),
        }


        # Create test-set prediction table.

        result_table = pd.DataFrame(
            {
                "Actual_CCF": y_test,
                "Predicted_CCF": predictions,

                "Current_Balance": model_data.loc[
                    X_test.index,
                    "current_balance",
                ],

                "Undrawn_Amount": model_data.loc[
                    X_test.index,
                    "undrawn_amount",
                ],

                "Actual_EAD_Reconstructed": actual_ead,
                "Predicted_EAD": predicted_ead,
            },
            index=X_test.index,
        )


        result_table["CCF_Error"] = (
            result_table["Actual_CCF"]
            - result_table["Predicted_CCF"]
        )

        result_table["EAD_Error"] = (
            result_table["Actual_EAD_Reconstructed"]
            - result_table["Predicted_EAD"]
        )


        # Store fitted model and results in the Streamlit session.

        st.session_state["ead_model_pipeline"] = pipeline

        st.session_state["ead_model_results"] = result_table

        st.session_state["ead_model_metadata"] = {
            "predictors": selected_predictors,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target_source": target_source_label,
            "clip_ccf": clip_ccf,
            "model_choice": model_choice,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }

        st.success(
            "CCF model fitted and EAD test predictions calculated."
        )


    except Exception as exc:

        st.error(
            f"EAD model development failed: {exc}"
        )

        st.exception(exc)


# ------------------------------------------------------------
# 6. MODEL PERFORMANCE
# ------------------------------------------------------------

if (
    "ead_model_results" in st.session_state
    and "ead_model_metadata" in st.session_state
):

    results = st.session_state["ead_model_results"]
    metadata = st.session_state["ead_model_metadata"]

    st.subheader("6. Model performance")

    a, b, c, d = st.columns(4)

    a.metric(
        "CCF MAE",
        f"{results['CCF_Error'].abs().mean():.4f}",
    )

    b.metric(
        "EAD MAE",
        f"{results['EAD_Error'].abs().mean():,.2f}",
    )

    c.metric(
        "Test observations",
        f"{len(results):,}",
    )

    d.metric(
        "Predictors",
        f"{len(metadata['predictors']):,}",
    )


    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    st.markdown("#### Performance metrics")

    metric_table = pd.DataFrame(
        [
            [
                "CCF MAE",
                mean_absolute_error(
                    results["Actual_CCF"],
                    results["Predicted_CCF"],
                ),
            ],

            [
                "CCF RMSE",
                np.sqrt(
                    mean_squared_error(
                        results["Actual_CCF"],
                        results["Predicted_CCF"],
                    )
                ),
            ],

            [
                "CCF R-squared",
                r2_score(
                    results["Actual_CCF"],
                    results["Predicted_CCF"],
                ),
            ],

            [
                "EAD MAE",
                mean_absolute_error(
                    results["Actual_EAD_Reconstructed"],
                    results["Predicted_EAD"],
                ),
            ],

            [
                "EAD RMSE",
                np.sqrt(
                    mean_squared_error(
                        results["Actual_EAD_Reconstructed"],
                        results["Predicted_EAD"],
                    )
                ),
            ],

            [
                "EAD R-squared",
                r2_score(
                    results["Actual_EAD_Reconstructed"],
                    results["Predicted_EAD"],
                ),
            ],
        ],
        columns=[
            "Metric",
            "Value",
        ],
    )

    st.dataframe(
        metric_table,
        hide_index=True,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # ACTUAL VS PREDICTED CCF AND EAD
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.scatter(
            results["Actual_CCF"],
            results["Predicted_CCF"],
            alpha=0.55,
        )

        low = min(
            results["Actual_CCF"].min(),
            results["Predicted_CCF"].min(),
        )

        high = max(
            results["Actual_CCF"].max(),
            results["Predicted_CCF"].max(),
        )

        ax.plot(
            [low, high],
            [low, high],
            linestyle="--",
        )

        ax.set_title("Actual vs predicted CCF")
        ax.set_xlabel("Actual CCF")
        ax.set_ylabel("Predicted CCF")
        ax.grid(alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)


    with right:

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.scatter(
            results["Actual_EAD_Reconstructed"],
            results["Predicted_EAD"],
            alpha=0.55,
        )

        low = min(
            results["Actual_EAD_Reconstructed"].min(),
            results["Predicted_EAD"].min(),
        )

        high = max(
            results["Actual_EAD_Reconstructed"].max(),
            results["Predicted_EAD"].max(),
        )

        ax.plot(
            [low, high],
            [low, high],
            linestyle="--",
        )

        ax.set_title("Actual vs predicted EAD")
        ax.set_xlabel("Actual EAD")
        ax.set_ylabel("Predicted EAD")
        ax.grid(alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)


    # --------------------------------------------------------
    # CCF CALIBRATION BY PREDICTED DECILE
    # --------------------------------------------------------

    st.markdown(
        "#### Predicted versus actual CCF by decile"
    )

    calibration = results[
        [
            "Actual_CCF",
            "Predicted_CCF",
        ]
    ].copy()

    calibration["Predicted_Decile"] = pd.qcut(
        calibration["Predicted_CCF"].rank(
            method="first"
        ),
        q=min(
            10,
            calibration["Predicted_CCF"].nunique(),
            len(calibration),
        ),
        labels=False,
        duplicates="drop",
    ) + 1

    calibration_table = (
        calibration
        .groupby(
            "Predicted_Decile",
            as_index=False,
        )
        .agg(
            Accounts=("Actual_CCF", "size"),
            Mean_Predicted_CCF=("Predicted_CCF", "mean"),
            Mean_Actual_CCF=("Actual_CCF", "mean"),
        )
    )

    st.dataframe(
        calibration_table,
        hide_index=True,
        use_container_width=True,
    )


    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        calibration_table["Predicted_Decile"],
        calibration_table["Mean_Predicted_CCF"],
        marker="o",
        label="Mean predicted CCF",
    )

    ax.plot(
        calibration_table["Predicted_Decile"],
        calibration_table["Mean_Actual_CCF"],
        marker="o",
        label="Mean actual CCF",
    )

    ax.set_xlabel("Predicted CCF decile")
    ax.set_ylabel("CCF")
    ax.set_title("CCF calibration by predicted-risk decile")
    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)


    # --------------------------------------------------------
    # TEST SET PREDICTIONS
    # --------------------------------------------------------

    st.markdown("#### Test-set predictions")

    st.dataframe(
        results.head(100),
        use_container_width=True,
    )

    st.download_button(
        "Download test-set EAD predictions",
        data=results.to_csv(index=True).encode("utf-8"),
        file_name="ead_test_predictions.csv",
        mime="text/csv",
    )


    # --------------------------------------------------------
    # 7. FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("7. Predictor importance")

    fitted_pipeline = st.session_state[
        "ead_model_pipeline"
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
                    "Importance": fitted_regressor.feature_importances_,
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


            fig, ax = plt.subplots(
                figsize=(
                    9,
                    max(
                        4,
                        min(
                            9,
                            len(importance.head(20)) * 0.35,
                        ),
                    ),
                )
            )

            top_imp = (
                importance
                .head(20)
                .sort_values("Importance")
            )

            ax.barh(
                top_imp["Transformed Predictor"],
                top_imp["Importance"],
            )

            ax.set_title("Top predictor importance")
            ax.set_xlabel("Importance")
            ax.grid(axis="x", alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)

        except Exception as exc:

            st.warning(
                f"Could not display predictor importance: {exc}"
            )


    # --------------------------------------------------------
    # 8. SCORE THE ORIGINAL PORTFOLIO
    # --------------------------------------------------------

    st.subheader(
        "8. Estimate EAD for the original portfolio"
    )

    scoring_predictors = metadata["predictors"]

    scoring_data = source_data.copy()

    missing_predictors = [
        c for c in scoring_predictors
        if c not in scoring_data.columns
    ]

    if missing_predictors:

        st.error(
            "The original source data is missing model predictors: "
            + ", ".join(missing_predictors)
        )

    else:

        usable = scoring_data[
            [
                "current_balance",
                "undrawn_amount",
            ] + scoring_predictors
        ].copy()

        usable["current_balance"] = pd.to_numeric(
            usable["current_balance"],
            errors="coerce",
        )

        usable["undrawn_amount"] = pd.to_numeric(
            usable["undrawn_amount"],
            errors="coerce",
        )

        valid_exposure = (
            usable["current_balance"].notna()
            & usable["undrawn_amount"].notna()
            & (usable["undrawn_amount"] >= 0)
        )

        scored = scoring_data.loc[
            valid_exposure
        ].copy()

        score_X = usable.loc[
            valid_exposure,
            scoring_predictors,
        ]

        predicted_ccf_all = fitted_pipeline.predict(
            score_X
        )

        if metadata["clip_ccf"]:

            predicted_ccf_all = np.clip(
                predicted_ccf_all,
                0,
                1,
            )

        scored["predicted_ccf"] = predicted_ccf_all

        scored["predicted_ead"] = (
            usable.loc[
                valid_exposure,
                "current_balance",
            ].to_numpy()
            +
            predicted_ccf_all
            *
            usable.loc[
                valid_exposure,
                "undrawn_amount",
            ].to_numpy()
        )

        st.metric(
            "Total modelled portfolio EAD",
            f"{scored['predicted_ead'].sum():,.2f}",
        )

        st.dataframe(
            scored.head(100),
            use_container_width=True,
        )

        st.download_button(
            "Download original portfolio with modelled EAD",
            data=scored.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="original_portfolio_modelled_ead.csv",
            mime="text/csv",
        )


    # --------------------------------------------------------
    # 9. MODEL SPECIFICATION / LIMITATIONS
    # --------------------------------------------------------

    st.subheader("9. Model specification")

    st.write(
        {
            "Algorithm": metadata["model_choice"],
            "CCF target source": metadata["target_source"],
            "Predictors": metadata["predictors"],
            "Numeric predictors": metadata["numeric_features"],
            "Categorical predictors": metadata["categorical_features"],
            "Training observations": metadata["train_rows"],
            "Test observations": metadata["test_rows"],
            "Test proportion": metadata["test_size"],
            "Random state": metadata["random_state"],
            "CCF constrained to 0-1": metadata["clip_ccf"],
        }
    )

    st.warning(
        "A random train/test split is an initial demonstration only. "
        "A production EAD model should use a clearly defined reference date, "
        "default observation window, exposure-at-default measurement, "
        "representative defaulted facilities, product segmentation, "
        "out-of-time validation and monitoring. The synthetic target may also "
        "be formula-generated, so strong test metrics can simply reflect the "
        "data-generation process."
    )


st.divider()

st.caption(
    "CreditRisk Lab · EAD model development · "
    "Synthetic demonstration data · Not production-ready"
)