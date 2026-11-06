import io
import pickle
import warnings
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss
)
from scipy.stats import ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor


st.markdown("""
<style>

/* =========================================================
   PD MODEL DEVELOPMENT PAGE
   ========================================================= */

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
    padding: 16px;
    border-radius: 12px;
}

div[data-testid="stTabs"] button {
    color: #93C5FD;
}

/* =========================================================
   SELECTED MULTISELECT TAGS - GREY WITH BLACK TEXT
   ========================================================= */

/* Outer selected-variable banner */
div[data-testid="stMultiSelect"] span[data-tag] {
    background: #E8EEF5 !important;
    background-color: #E8EEF5 !important;
    border: 1px solid #93B4D5 !important;
    border-radius: 7px !important;
    box-shadow: none !important;
}

/* Inner variable text */
div[data-testid="stMultiSelect"] span[data-tag] > span {
    background: transparent !important;
    background-color: transparent !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* Remove button */
div[data-testid="stMultiSelect"] span[data-tag] > button {
    background: transparent !important;
    background-color: transparent !important;
    color: #111827 !important;
    border: none !important;
}

/* Remove icon */
div[data-testid="stMultiSelect"] span[data-tag] > button svg {
    color: #111827 !important;
}

/* Hover */
div[data-testid="stMultiSelect"] span[data-tag]:hover {
    background: #DCE8F5 !important;
    background-color: #DCE8F5 !important;
}

/* =========================================================
   CONFIRM PREDICTOR SELECTION BUTTON - GREEN
   ========================================================= */

div[data-testid="stForm"] button[kind="primary"] {
    background: #15803D !important;
    background-color: #15803D !important;
    background-image: none !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #15803D !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    font-weight: 700 !important;
}

div[data-testid="stForm"] button[kind="primary"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

div[data-testid="stForm"] button[kind="primary"]:hover {
    background: #166534 !important;
    background-color: #166534 !important;
    border-color: #166534 !important;
}

</style>
""", unsafe_allow_html=True)



st.title("PD Model Development")
st.caption(
    "Practical development of a Probability of Default model "
    "using optimal binning, WOE and logistic regression."
)
st.info(
    "The original portfolio in Data & Models is preserved. "
    "All preparation and modelling are performed on separate copies."
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def invalidate_pd_model():
    """Clear model outputs when the development sample is rebuilt."""
    keys = [
        "pd_binning_objects", "pd_train_woe", "pd_test_woe",
        "pd_iv_results", "pd_final_features", "pd_logit_model",
        "pd_train_predictions", "pd_test_predictions",
        "pd_model_fitted", "pd_vif_results",
        "pd_correlation_matrix", "pd_oot_results",
        "pd_oot_model", "pd_oot_binning_objects",
        "pd_oot_predictions"
    ]
    for key in keys:
        st.session_state.pop(key, None)


def calculate_vif(data):
    vif_data = data.replace([np.inf, -np.inf], np.nan).dropna()
    vif_data = vif_data.loc[:, vif_data.nunique(dropna=True) > 1]

    if vif_data.shape[1] < 2 or len(vif_data) <= vif_data.shape[1] + 1:
        return pd.DataFrame(columns=["Predictor", "VIF", "Assessment"])

    vif_input = sm.add_constant(vif_data, has_constant="add")
    results = []

    for i, variable in enumerate(vif_input.columns):
        if variable == "const":
            continue
        try:
            vif = variance_inflation_factor(vif_input.to_numpy(dtype=float), i)
        except Exception:
            vif = np.nan
        results.append({"Predictor": variable, "VIF": vif})

    result = pd.DataFrame(results)
    if not result.empty:
        result["Assessment"] = result["VIF"].apply(
            lambda x: "Acceptable" if pd.notna(x) and x < 5 else
            "Review" if pd.notna(x) and x <= 10 else
            "High multicollinearity" if pd.notna(x) else
            "Unable to calculate"
        )
    return result


def make_correlation_plot(data, title):
    raw_correlation = data.corr(numeric_only=True)
    if raw_correlation.empty:
        st.warning("No numeric variables are available.")
        return None, raw_correlation

    plot_correlation = raw_correlation.fillna(0)
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#071426")
    ax.set_facecolor("#071426")

    image = ax.matshow(
        plot_correlation.to_numpy(),
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(plot_correlation.columns)))
    ax.set_yticks(range(len(plot_correlation.columns)))
    ax.set_xticklabels(
        plot_correlation.columns,
        rotation=45,
        ha="left",
        color="white",
        fontsize=9
    )
    ax.set_yticklabels(
        plot_correlation.columns,
        color="white",
        fontsize=9
    )
    ax.xaxis.tick_bottom()
    ax.set_title(title, color="white", fontsize=15, pad=20)

    for i in range(len(plot_correlation.columns)):
        for j in range(len(plot_correlation.columns)):
            value = plot_correlation.iloc[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                color="black", fontsize=7
            )

    colourbar = fig.colorbar(image, ax=ax)
    colourbar.set_label("Correlation", color="white")
    colourbar.ax.tick_params(colors="white")
    ax.tick_params(colors="white")
    fig.tight_layout()
    return fig, raw_correlation


def identify_correlated_pairs(correlation, threshold=0.70):
    pairs = []
    columns = correlation.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = correlation.iloc[i, j]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({
                    "Predictor 1": columns[i],
                    "Predictor 2": columns[j],
                    "Correlation": value,
                    "Absolute Correlation": abs(value)
                })

    if not pairs:
        return pd.DataFrame(
            columns=[
                "Predictor 1", "Predictor 2",
                "Correlation", "Absolute Correlation"
            ]
        )
    return pd.DataFrame(pairs).sort_values(
        "Absolute Correlation", ascending=False
    )


def prepare_feature_series(series, dtype):
    """Prepare one raw predictor consistently for OptBinning."""
    if dtype == "numerical":
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return (
        series.astype("string")
        .fillna("Missing")
        .astype(str)
        .to_numpy()
    )


def fit_binning_objects(X_train, y_train, variables, max_bins):
    """Fit OptBinning only on the supplied development sample."""
    from optbinning import OptimalBinning

    objects = {}
    train_woe = pd.DataFrame(index=X_train.index)
    iv_results = []
    errors = []

    for variable in variables:
        try:
            x_train = X_train[variable]
            dtype = (
                "numerical"
                if pd.api.types.is_numeric_dtype(x_train)
                else "categorical"
            )
            x_fit = prepare_feature_series(x_train, dtype)

            if pd.Series(x_fit).nunique() < 2:
                errors.append(f"{variable}: insufficient variation.")
                continue

            optb = OptimalBinning(
                name=variable,
                dtype=dtype,
                max_n_bins=max_bins,
                min_bin_size=0.05
            )
            optb.fit(x_fit, y_train.to_numpy())
            train_woe[variable + "_woe"] = optb.transform(
                x_fit, metric="woe"
            )
            objects[variable] = optb
            optb.binning_table.build()
            iv_results.append({
                "Variable": variable,
                "Information Value": optb.binning_table.iv
            })
        except Exception as error:
            errors.append(f"{variable}: {error}")

    return objects, train_woe, pd.DataFrame(iv_results), errors


def transform_with_binning(X, binning_objects):
    transformed = pd.DataFrame(index=X.index)
    for variable, optb in binning_objects.items():
        dtype = getattr(optb, "dtype", None)
        if dtype not in ("numerical", "categorical"):
            dtype = (
                "numerical"
                if pd.api.types.is_numeric_dtype(X[variable])
                else "categorical"
            )
        values = prepare_feature_series(X[variable], dtype)
        transformed[variable + "_woe"] = optb.transform(
            values, metric="woe"
        )
    return transformed


def calculate_psi(reference, comparison, bins=10):
    """
    Calculate PSI using reference-derived quantile buckets for numeric
    variables and pooled categories for categorical variables.
    """
    ref = pd.Series(reference).replace([np.inf, -np.inf], np.nan).dropna()
    comp = pd.Series(comparison).replace([np.inf, -np.inf], np.nan).dropna()

    if len(ref) == 0 or len(comp) == 0:
        return np.nan

    numeric = (
        pd.api.types.is_numeric_dtype(ref)
        and pd.api.types.is_numeric_dtype(comp)
    )

    if numeric:
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(ref.quantile(quantiles).to_numpy(dtype=float))
        if len(edges) < 2:
            return 0.0
        edges[0] = -np.inf
        edges[-1] = np.inf
        ref_bucket = pd.cut(ref, bins=edges, include_lowest=True)
        comp_bucket = pd.cut(comp, bins=edges, include_lowest=True)
        categories = ref_bucket.cat.categories
        ref_counts = ref_bucket.value_counts(sort=False).reindex(categories, fill_value=0)
        comp_counts = comp_bucket.value_counts(sort=False).reindex(categories, fill_value=0)
    else:
        ref_text = ref.astype(str)
        comp_text = comp.astype(str)
        categories = sorted(set(ref_text.unique()).union(comp_text.unique()))
        ref_counts = ref_text.value_counts().reindex(categories, fill_value=0)
        comp_counts = comp_text.value_counts().reindex(categories, fill_value=0)

    ref_pct = np.maximum(ref_counts.to_numpy(dtype=float) / len(ref), 1e-6)
    comp_pct = np.maximum(comp_counts.to_numpy(dtype=float) / len(comp), 1e-6)
    return float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))


def psi_interpretation(value):
    if pd.isna(value):
        return "Not calculable"
    if value < 0.10:
        return "Little or no population change"
    if value < 0.25:
        return "Moderate population change: investigate"
    return "Material population change: investigate"


def calculate_calibration_table(y_true, predictions, groups=10):
    calibration = pd.DataFrame({
        "Predicted_PD": np.asarray(predictions, dtype=float),
        "Default": np.asarray(y_true, dtype=int)
    })
    n_groups = min(groups, calibration["Predicted_PD"].nunique(), len(calibration))
    if n_groups < 2:
        return pd.DataFrame()

    calibration["PD_Band"] = pd.qcut(
        calibration["Predicted_PD"].rank(method="first"),
        q=n_groups,
        labels=False
    ) + 1

    return (
        calibration.groupby("PD_Band", observed=False)
        .agg(
            Observations=("Default", "size"),
            Defaults=("Default", "sum"),
            Average_Predicted_PD=("Predicted_PD", "mean"),
            Observed_Default_Rate=("Default", "mean")
        )
        .reset_index()
    )


def calculate_binary_metrics(y_true, predictions):
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("The evaluation sample must contain both default and non-default cases.")

    auc = roc_auc_score(y_true, predictions)
    fpr, tpr, thresholds = roc_curve(y_true, predictions)
    ks = float(np.max(tpr - fpr))
    return {
        "AUC": float(auc),
        "Gini": float(2 * auc - 1),
        "KS": ks,
        "Brier score": float(brier_score_loss(y_true, predictions))
    }


def assign_illustrative_risk_grade(pd_values):
    """Illustrative bands only; not a regulatory or bank rating scale."""
    return pd.cut(
        pd.Series(pd_values),
        bins=[-np.inf, 0.01, 0.03, 0.05, 0.10, np.inf],
        labels=["A - Very low", "B - Low", "C - Moderate", "D - High", "E - Very high"],
        right=False
    ).astype(str)


# =========================================================
# DATA DISCOVERY, TARGET MAPPING AND GENERIC CONFIGURATION
# =========================================================

if "credit_data" not in st.session_state:
    st.warning(
        "No portfolio is currently available. Go to Data & Models "
        "and upload or generate a dataset first."
    )
    st.stop()

# Never modify the central source dataset. All mappings are applied to a copy.
source_data = st.session_state["credit_data"].copy()
if source_data.empty or len(source_data.columns) < 2:
    st.error("The uploaded dataset must contain observations and at least two columns.")
    st.stop()


def suggest_canonical_concept(column_name):
    """Suggest a business concept from names without forcing a source schema."""
    name = str(column_name).strip().lower().replace("-", "_").replace(" ", "_")
    concepts = {
        "Default outcome": ["default", "bad_flag", "is_default", "default_flag", "bad", "event", "target", "ever_default", "loan_status"],
        "Delinquency history": ["delinq", "delq", "delinquen", "arrears", "past_due", "dpd", "late_payment"],
        "Credit score": ["credit_score", "bureau_score", "risk_score", "score", "behaviour_score", "behavior_score"],
        "Income": ["income", "salary", "earnings", "gross_pay", "turnover"],
        "Loan amount / exposure": ["loan_amount", "principal", "credit_advanced", "facility_amount", "exposure", "balance", "outstanding_balance"],
        "Debt burden / affordability": ["dti", "debt_to_income", "affordability", "instalment_to_income", "payment_to_income"],
        "Credit utilisation": ["utilisation", "utilization", "util", "credit_limit_used"],
        "Interest rate": ["interest_rate", "rate", "apr", "pricing"],
        "Loan term": ["loan_term", "term_months", "tenor", "maturity"],
        "Employment": ["employment", "employer", "occupation", "job", "industry"],
        "Age": ["age", "birth", "dob", "date_of_birth"],
        "Credit history length": ["credit_history", "history_months", "oldest_account", "account_vintage"],
        "Enquiries": ["inq", "enquir", "application_count", "credit_checks"],
        "Public records": ["pub_rec", "judgement", "judgment", "insolvency", "write_off", "writeoff"],
        "Customer / loan identifier": ["loan_id", "customer_id", "client_id", "account_id", "application_id", "reference", "record_id", "id"],
        "Observation / origination date": ["issue_d", "origination_date", "application_date", "booking_date", "observation_date", "snapshot_date", "as_of_date"]
    }
    for concept, terms in concepts.items():
        if any(term in name for term in terms):
            return concept
    return "Unmapped / other"


def infer_variable_type(series):
    """Profile a column for display and sensible modelling defaults."""
    non_missing = series.dropna()
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date"
    if len(non_missing) == 0:
        return "Empty"
    if pd.api.types.is_bool_dtype(series):
        return "Binary"
    numeric = pd.to_numeric(non_missing, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        unique = numeric.nunique()
        if unique <= 2:
            return "Binary"
        if unique <= 12:
            return "Numeric / discrete"
        return "Numeric / continuous"
    if non_missing.nunique() <= max(20, int(len(non_missing) * 0.05)):
        return "Categorical"
    return "Text / high-cardinality"


st.subheader("1. Dataset discovery and variable mapping")
st.write(
    "The application inspects the uploaded dataset instead of expecting a "
    "predefined list of column names. Suggested concepts are advisory: "
    "confirm the target and predictors before modelling."
)

profile_rows = []
for column in source_data.columns:
    series = source_data[column]
    profile_rows.append({
        "Source column": column,
        "Suggested concept": suggest_canonical_concept(column),
        "Detected type": infer_variable_type(series),
        "Missing %": round(series.isna().mean() * 100, 2),
        "Unique values": int(series.nunique(dropna=True)),
        "Example values": ", ".join(series.dropna().astype(str).head(3).tolist())
    })
profile = pd.DataFrame(profile_rows)
with st.expander("View dataset profile and automatic variable suggestions", expanded=True):
    st.dataframe(profile, use_container_width=True, hide_index=True)

# Find binary columns as possible targets, but require the user to choose one.
binary_candidates = []
for column in source_data.columns:
    values = pd.to_numeric(source_data[column], errors="coerce").dropna().unique()
    if len(values) == 2 and set(values.tolist()).issubset({0, 1}):
        binary_candidates.append(column)

suggested_targets = [
    column for column in source_data.columns
    if suggest_canonical_concept(column) == "Default outcome"
]
target_options = source_data.columns.tolist()
preferred_target = next((c for c in suggested_targets if c in target_options), None)
if preferred_target is None and binary_candidates:
    preferred_target = binary_candidates[0]

st.markdown("#### Confirm the default outcome")
target_index = target_options.index(preferred_target) if preferred_target in target_options else 0
target_column = st.selectbox(
    "Which column represents default / bad outcome?",
    options=target_options,
    index=target_index,
    key="pd_target_column"
)

# Normalize the selected outcome internally. The uploaded source remains unchanged.
# Both numeric 0/1 targets and two-class labels (e.g. Good/Bad) are supported.
raw_target = source_data[target_column]
observed_target_values = raw_target.dropna().unique().tolist()
if len(observed_target_values) != 2:
    st.error(
        f"The selected target '{target_column}' has {len(observed_target_values)} "
        "observed classes. A binary PD model requires exactly two outcome classes."
    )
    st.stop()

numeric_target = pd.to_numeric(raw_target, errors="coerce")
if set(numeric_target.dropna().unique().tolist()) == {0, 1} and numeric_target.notna().sum() == raw_target.notna().sum():
    # Conventional encoding: 1 is default, 0 is non-default.
    source_data["default"] = numeric_target
    st.caption(f"'{target_column}' is interpreted as 1 = default and 0 = non-default.")
else:
    target_labels = sorted(observed_target_values, key=lambda value: str(value))
    default_label = st.selectbox(
        "Which value represents DEFAULT / BAD?",
        options=target_labels,
        format_func=lambda value: str(value),
        key="pd_default_event_label"
    )
    source_data["default"] = np.where(raw_target.isna(), np.nan, (raw_target.astype(str) == str(default_label)).astype(int))
    st.caption(
        f"Target mapping: {default_label!s} = 1 (default); "
        f"{next(value for value in target_labels if str(value) != str(default_label))!s} = 0 (non-default)."
    )

target_values = source_data["default"].dropna()
if set(target_values.unique()) != {0, 1}:
    st.error("The selected target could not be mapped to binary values 0 and 1.")
    st.stop()

# Automatically identify likely IDs and dates, but allow the user to override.
likely_ids = [c for c in source_data.columns if suggest_canonical_concept(c) == "Customer / loan identifier"]
likely_dates = [c for c in source_data.columns if suggest_canonical_concept(c) == "Observation / origination date"]
# Exclude known synthetic data-generation fields that encode the target
# mechanism rather than borrower information. In particular, pd_probability
# is the latent probability used to generate the synthetic default outcome and
# must never be used as a PD model predictor.
known_target_leakage_columns = {
    "pd_probability",
    "true_pd",
    "default_probability_true",
    "latent_default_probability",
}
leakage_columns_present = {
    column for column in source_data.columns
    if str(column).strip().lower() in known_target_leakage_columns
}

likely_exclusions = set(
    likely_ids + likely_dates + [target_column, "default"]
) | leakage_columns_present
# Remove constant, empty, target-equivalent and known leakage columns from
# predictor choices. The original source dataset remains unchanged.
eligible_features = []
for column in source_data.columns:
    if column in likely_exclusions:
        continue
    if source_data[column].nunique(dropna=True) < 2:
        continue
    if source_data[column].isna().all():
        continue
    eligible_features.append(column)

st.markdown("#### Confirm predictor variables")
st.caption(
    "All eligible columns from this dataset are available, including variables "
    "that have never appeared in the example portfolio. Identifier and date "
    "columns are excluded by default but can be included if appropriate."
)
include_excluded = st.checkbox(
    "Show identifier and date columns as selectable predictors",
    value=False,
    key="pd_include_ids_dates"
)
if include_excluded:
    feature_options = [c for c in source_data.columns if c not in [target_column, "default"] and source_data[c].nunique(dropna=True) > 1]
else:
    feature_options = eligible_features

if not feature_options:
    st.error("No eligible predictor columns were found after excluding the selected target and constant columns.")
    st.stop()

# Session-state-safe defaults for each newly uploaded dataset.
dataset_signature = (st.session_state.get("data_name", "uploaded_dataset"), tuple(source_data.columns))
if st.session_state.get("pd_dataset_signature") != dataset_signature:
    st.session_state["pd_dataset_signature"] = dataset_signature
    st.session_state["pd_candidate_features"] = feature_options.copy()
    invalidate_pd_model()

current_selected = [c for c in st.session_state.get("pd_candidate_features", feature_options) if c in feature_options]
selected_features = st.multiselect(
    "Select candidate predictors",
    options=feature_options,
    default=current_selected,
    key="pd_candidate_features"
)
st.session_state["pd_target_source_column"] = target_column

# Offer a compact semantic mapping review. The source names are retained as model features.
mapping_review = profile[["Source column", "Suggested concept", "Detected type"]].copy()
st.caption(
    "Mappings are descriptive metadata only. The model uses the original source "
    "column names and does not treat related measures as interchangeable. "
    "For example, 2-year and 3-year delinquency windows remain distinct variables."
)
if leakage_columns_present:
    st.warning(
        "Excluded known target-leakage / data-generation columns from predictor "
        "selection: " + ", ".join(sorted(leakage_columns_present))
        + ". These fields must not be used to predict the outcome they helped generate."
    )

col1, col2, col3 = st.columns(3)
with col1:
    test_size = st.slider(
        "Random test dataset proportion",
        min_value=0.10, max_value=0.40, value=0.20, step=0.05
    )
with col2:
    random_state = st.number_input(
        "Random state", min_value=1, max_value=9999, value=42
    )
with col3:
    max_bins = st.slider(
        "Maximum bins per variable", min_value=3, max_value=10, value=5
    )

if not selected_features:
    st.warning("Select at least one predictor.")
    st.stop()


# =========================================================
# 2. PD-SPECIFIC DATA PREPARATION
# =========================================================

st.divider()
st.subheader("2. PD-specific data preparation")
st.write(
    "A separate modelling copy is created. The source portfolio remains unchanged."
)

required_columns = selected_features + ["default"]
model_data = source_data[required_columns].copy()
initial_rows = len(model_data)

model_data["default"] = pd.to_numeric(model_data["default"], errors="coerce")
model_data = model_data.dropna(subset=["default"])
model_data = model_data[model_data["default"].isin([0, 1])].copy()
model_data["default"] = model_data["default"].astype(int)

for col in selected_features:
    if pd.api.types.is_numeric_dtype(model_data[col]):
        model_data[col] = pd.to_numeric(model_data[col], errors="coerce")
    else:
        model_data[col] = (
            model_data[col].astype("string").fillna("Missing").astype(str)
        )

dropped_rows = initial_rows - len(model_data)
default_rate = model_data["default"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Source observations", f"{initial_rows:,}")
m2.metric("Modelling observations", f"{len(model_data):,}")
m3.metric("Removed target rows", f"{dropped_rows:,}")
m4.metric("Default rate", f"{default_rate:.2%}")

st.dataframe(model_data.head(10), use_container_width=True)


# =========================================================
# 3. EXPLORATORY DATA ANALYSIS
# =========================================================

st.divider()
st.subheader("3. Exploratory data analysis")

eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
    "Default distribution", "Predictor distributions",
    "Missing values", "Correlation analysis"
])

with eda_tab1:
    distribution = model_data["default"].value_counts().reindex([0, 1], fill_value=0)
    distribution_df = pd.DataFrame({
        "Outcome": ["Non-default", "Default"],
        "Observations": distribution.values
    })
    st.bar_chart(distribution_df.set_index("Outcome"))
    st.dataframe(distribution_df, use_container_width=True)

numeric_features = [
    col for col in selected_features
    if pd.api.types.is_numeric_dtype(model_data[col])
]

with eda_tab2:
    if numeric_features:
        selected_eda_variable = st.selectbox(
            "Select numerical predictor", numeric_features
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        for outcome, label in [(0, "Non-default"), (1, "Default")]:
            values = model_data.loc[
                model_data["default"] == outcome, selected_eda_variable
            ].dropna()
            ax.hist(values, bins=30, alpha=0.5, density=True, label=label)
        ax.set_title(f"{selected_eda_variable} distribution")
        ax.set_xlabel(selected_eda_variable)
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numerical predictors are available.")

with eda_tab3:
    missing = pd.DataFrame({
        "Variable": model_data.columns,
        "Missing observations": model_data.isna().sum().values,
        "Missing percentage": model_data.isna().mean().values * 100
    })
    st.dataframe(missing, use_container_width=True)

with eda_tab4:
    st.write(
        "The heatmap examines correlations between original numerical "
        "predictors before WOE transformation."
    )
    if len(numeric_features) >= 2:
        fig, raw_correlation = make_correlation_plot(
            model_data[numeric_features],
            "Original Numerical Predictor Correlation"
        )
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        raw_pairs = identify_correlated_pairs(raw_correlation, threshold=0.70)
        if not raw_pairs.empty:
            st.write("Highly correlated original predictor pairs")
            st.dataframe(raw_pairs, use_container_width=True, hide_index=True)
        else:
            st.success("No original numerical predictor pairs exceeded 0.70.")
    else:
        st.info("At least two numerical predictors are required for correlation analysis.")


# =========================================================
# 4. RANDOM TRAINING AND TESTING SPLIT
# =========================================================

st.divider()
st.subheader("4. Random training and testing split")
st.caption(
    "This random split is useful for an initial development assessment. "
    "It does not replace out-of-time validation."
)

if st.button("Prepare training and test datasets", type="primary"):
    try:
        X = model_data[selected_features].copy()
        y = model_data["default"].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=y
        )
        invalidate_pd_model()
        st.session_state["pd_X_train"] = X_train
        st.session_state["pd_X_test"] = X_test
        st.session_state["pd_y_train"] = y_train
        st.session_state["pd_y_test"] = y_test
        st.session_state["pd_features"] = selected_features.copy()
        st.success("Training and test datasets prepared.")
    except Exception as error:
        st.error(f"Data splitting failed: {error}")

if "pd_X_train" in st.session_state:
    X_train = st.session_state["pd_X_train"]
    X_test = st.session_state["pd_X_test"]
    y_train = st.session_state["pd_y_train"]
    y_test = st.session_state["pd_y_test"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training observations", f"{len(X_train):,}")
    c2.metric("Test observations", f"{len(X_test):,}")
    c3.metric("Training default rate", f"{y_train.mean():.2%}")
    c4.metric("Test default rate", f"{y_test.mean():.2%}")
    st.write("Training data preview")
    st.dataframe(X_train.head(), use_container_width=True)


# =========================================================
# 5. OPTIMAL BINNING AND WOE TRANSFORMATION
# =========================================================

st.divider()
st.subheader("5. Optimal binning and WOE transformation")
st.write(
    "Optimal binning is fitted on the training dataset only. "
    "The fitted binning objects are applied to the test dataset."
)

st.info(
    "The objective is to transform continuous and categorical predictors "
    "into meaningful groups that have a useful relationship with the "
    "target variable. For a PD Model, the target variable will be the default flag that contains 0 and 1."
)

if "pd_X_train" not in st.session_state:
    st.info("Prepare the training and test datasets first.")
else:
    if st.button("Run optimal binning", type="primary"):
        X_train = st.session_state["pd_X_train"]
        X_test = st.session_state["pd_X_test"]
        y_train = st.session_state["pd_y_train"]

        try:
            objects, train_woe, iv_results, errors = fit_binning_objects(
                X_train, y_train, list(X_train.columns), max_bins
            )
            if not objects:
                st.error("Optimal binning failed for all selected predictors.")
                for message in errors:
                    st.write(message)
            else:
                test_woe = transform_with_binning(X_test, objects)
                st.session_state["pd_binning_objects"] = objects
                st.session_state["pd_train_woe"] = train_woe
                st.session_state["pd_test_woe"] = test_woe
                st.session_state["pd_iv_results"] = iv_results
                st.session_state.pop("pd_final_features", None)
                st.session_state["pd_model_fitted"] = False
                st.session_state.pop("pd_logit_model", None)
                st.session_state.pop("pd_train_predictions", None)
                st.session_state.pop("pd_test_predictions", None)
                st.session_state.pop("pd_oot_results", None)
                st.success("Optimal binning and WOE transformation completed.")
                if errors:
                    st.warning("Some predictors could not be binned.")
                    for message in errors:
                        st.write(message)
        except Exception as error:
            st.error(f"WOE transformation failed: {error}")

if "pd_iv_results" in st.session_state:
    iv_table = st.session_state["pd_iv_results"].sort_values(
        "Information Value", ascending=False
    )
    st.write("Information Value ranking")
    st.dataframe(iv_table, use_container_width=True, hide_index=True)
    if not iv_table.empty:
        st.bar_chart(iv_table.set_index("Variable")["Information Value"])

    if st.session_state.get("pd_binning_objects"):
        st.subheader("Inspect individual binning results")
        selected_variable = st.selectbox(
            "Select a predictor to inspect",
            list(st.session_state["pd_binning_objects"].keys())
        )
        binning_object = st.session_state["pd_binning_objects"][selected_variable]
        display_table = binning_object.binning_table.build().copy()
        for column in display_table.columns:
            if display_table[column].dtype == "object":
                display_table[column] = display_table[column].apply(str)
        st.dataframe(display_table, use_container_width=True, hide_index=True)

# =========================================================
# 6. PREDICTOR SELECTION AND DIAGNOSTICS
# =========================================================

st.divider()
st.subheader("6. Predictor selection and diagnostics")

if "pd_train_woe" not in st.session_state:
    st.info("Run optimal binning before selecting predictors.")

else:
    train_woe = st.session_state["pd_train_woe"].copy()
    iv_results = st.session_state["pd_iv_results"].copy()

    # -----------------------------------------------------
    # 6.1 INFORMATION VALUE ASSESSMENT
    # -----------------------------------------------------

    st.markdown("### 6.1 Information Value assessment")

    st.markdown("""
    | Information Value | General interpretation |
    |---|---|
    | Below 0.02 | Very weak predictive information |
    | 0.02 to 0.10 | Weak predictive information |
    | 0.10 to 0.30 | Moderate predictive information |
    | 0.30 to 0.50 | Strong predictive information |
    | Above 0.50 | Very strong; investigate possible leakage |
    """)

    st.caption(
        "These are general rules of thumb, not universal acceptance criteria. "
        "IV should not be the only basis for selecting predictors."
    )

    if not iv_results.empty:

        iv_table = iv_results.copy()

        iv_table["IV Interpretation"] = iv_table["Information Value"].apply(
            lambda x: "Very weak" if pd.isna(x) or x < 0.02 else
            "Weak" if x < 0.10 else
            "Moderate" if x < 0.30 else
            "Strong" if x <= 0.50 else
            "Investigate"
        )

        st.dataframe(
            iv_table.sort_values(
                "Information Value",
                ascending=False,
                na_position="last"
            ),
            use_container_width=True,
            hide_index=True
        )

    # -----------------------------------------------------
    # 6.2 WOE PREDICTOR CORRELATION ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 6.2 WOE predictor correlation analysis")

    if train_woe.shape[1] >= 2:

        fig, correlation_matrix = make_correlation_plot(
            train_woe,
            "WOE Predictor Correlation Matrix"
        )

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        correlation_pairs = identify_correlated_pairs(
            correlation_matrix,
            threshold=0.70
        )

        if not correlation_pairs.empty:

            st.warning(
                "Highly correlated WOE predictor pairs have been identified."
            )

            st.dataframe(
                correlation_pairs,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No WOE predictor pairs exceeded the absolute correlation "
                "threshold of 0.70."
            )

    else:

        correlation_matrix = train_woe.corr()

        st.info(
            "At least two WOE predictors are required for correlation analysis."
        )

    # -----------------------------------------------------
    # 6.3 MULTICOLLINEARITY ASSESSMENT
    # -----------------------------------------------------

    st.markdown("### 6.3 Multicollinearity assessment")

    st.caption(
        "VIF above 5 is flagged for review. VIF above 10 indicates more "
        "serious potential multicollinearity."
    )

    vif_table = calculate_vif(train_woe)

    if not vif_table.empty:

        st.dataframe(
            vif_table.sort_values("VIF", ascending=False),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "There are insufficient usable predictors to calculate VIF."
        )

    st.session_state["pd_vif_results"] = vif_table
    st.session_state["pd_correlation_matrix"] = correlation_matrix

    # -----------------------------------------------------
    # 6.4 FINAL PREDICTOR SELECTION
    # -----------------------------------------------------

    st.markdown("### 6.4 Final predictor selection")

    st.write(
        "Choose the final WOE predictors using IV, correlation, VIF, "
        "binning tables and business rationale."
    )

    # Assign each predictor to exactly one IV colour band.
    def classify_iv(value):

        if pd.isna(value):
            return "Missing IV", "#EF4444", "🔴"

        if value < 0.10:
            return "IV < 0.10", "#F97316", "🟠"

        if value < 0.20:
            return "0.10 ≤ IV < 0.20", "#FACC15", "🟡"

        if value <= 0.55:
            return "0.20 ≤ IV ≤ 0.55", "#22C55E", "🟢"

        return "IV > 0.55", "#EF4444", "🔴"

    # IV colour legend.
    st.markdown(
        """
        <div style="
            display:flex;
            flex-wrap:wrap;
            gap:18px;
            align-items:center;
            margin:10px 0 16px 0;
            font-weight:600;
            color:#FFFFFF;
        ">
            <span>🟠 IV &lt; 0.10</span>
            <span>🟡 0.10 ≤ IV &lt; 0.20</span>
            <span>🟢 0.20 ≤ IV ≤ 0.55</span>
            <span>🔴 IV &gt; 0.55 or Missing</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "The boundaries are mutually exclusive."

    )

    # Available WOE predictors come directly from the stored training data.
    available_woe_features = list(train_woe.columns)

    iv_lookup = (
        iv_results.set_index("Variable")["Information Value"].to_dict()
        if not iv_results.empty
        else {}
    )

    def get_feature_iv(woe_feature):

        raw_variable = (
            woe_feature[:-4]
            if woe_feature.endswith("_woe")
            else woe_feature
        )

        return iv_lookup.get(raw_variable, np.nan)

    def format_woe_feature(woe_feature):

        iv_value = get_feature_iv(woe_feature)

        _, _, colour_icon = classify_iv(iv_value)

        iv_text = (
            f"{iv_value:.4f}"
            if pd.notna(iv_value)
            else "N/A"
        )

        return f"{colour_icon} {woe_feature} (IV: {iv_text})"

    previous_selection = st.session_state.get(
        "pd_final_features",
        [
            col for col in available_woe_features
            if col != "pub_rec_woe"
        ]
    )

    previous_selection = [
        col for col in previous_selection
        if col in available_woe_features
    ]

    with st.form("pd_predictor_selection_form"):

        selected_final_features = st.multiselect(
            "Select final WOE predictors",
            options=available_woe_features,
            default=previous_selection,
            format_func=format_woe_feature,
            help=(
                "The coloured marker and IV value show the "
                "Information Value band for each predictor."
            )
        )

        submit_selection = st.form_submit_button(
            "Confirm predictor selection",
            type="primary"
        )

    if submit_selection:

        if not selected_final_features:

            st.error("Select at least one predictor.")

        else:

            st.session_state["pd_final_features"] = selected_final_features
            st.session_state["pd_model_fitted"] = False

            for key in [
                "pd_logit_model",
                "pd_train_predictions",
                "pd_test_predictions",
                "pd_oot_results"
            ]:

                st.session_state.pop(key, None)

            st.success("Final predictor selection confirmed.")

    # -----------------------------------------------------
    # FINAL PREDICTOR SELECTION SUMMARY
    # -----------------------------------------------------

    if "pd_final_features" in st.session_state:

        final_features = st.session_state["pd_final_features"]

        selection_table = iv_results.copy()

        selection_table["WOE Predictor"] = (
            selection_table["Variable"].astype(str) + "_woe"
        )

        selection_table["Decision"] = np.where(
            selection_table["WOE Predictor"].isin(final_features),
            "Retained",
            "Excluded"
        )

        selection_table["IV Band"] = (
            selection_table["Information Value"].apply(
                lambda value: classify_iv(value)[0]
            )
        )

        st.write("Final predictor selection summary")

        display_columns = [
            "Variable",
            "Information Value",
            "IV Band",
            "WOE Predictor",
            "Decision"
        ]

        display_table = selection_table[display_columns].sort_values(
            "Information Value",
            ascending=False,
            na_position="last"
        )

        def colour_iv_cells(row):

            _, background, _ = classify_iv(
                row["Information Value"]
            )

            foreground = (
                "#FFFFFF"
                if background == "#EF4444"
                else "#111827"
            )

            return [
                (
                    f"background-color: {background}; "
                    f"color: {foreground}; font-weight: 600;"
                )
                if column in [
                    "Variable",
                    "Information Value",
                    "IV Band"
                ]
                else ""
                for column in row.index
            ]

        st.dataframe(
            display_table.style.apply(
                colour_iv_cells,
                axis=1
            ).format(
                {"Information Value": "{:.4f}"}
            ),
            use_container_width=True,
            hide_index=True
        )

        st.success(
            f"{len(final_features)} predictors selected."
        )

        st.caption(
            "An IV above 0.55 is highlighted for investigation. It is not "
            "automatically excluded: review possible target leakage, "
            "overfitting and data quality before retaining the predictor."
        )# =========================================================
# 7. LOGISTIC REGRESSION
# =========================================================

st.divider()
st.subheader("7. Logistic regression PD model")
st.write(
    "The logistic regression estimates default probability using "
    "the selected WOE-transformed predictors."
)

if "pd_train_woe" not in st.session_state:
    st.info("Run optimal binning before estimating logistic regression.")
elif "pd_final_features" not in st.session_state:
    st.info("Confirm your final predictor selection in section 6.4.")
else:
    train_woe = st.session_state["pd_train_woe"]
    test_woe = st.session_state["pd_test_woe"]
    final_features = st.session_state["pd_final_features"]

    if st.button("Estimate logistic regression", type="primary"):
        try:
            Xtr = sm.add_constant(train_woe[final_features].copy(), has_constant="add")
            Xte = sm.add_constant(test_woe[final_features].copy(), has_constant="add")

            if Xtr.isna().any().any() or Xte.isna().any().any():
                st.error(
                    "Missing values were detected in WOE predictors. "
                    "Review data preparation and binning."
                )
            else:
                model = sm.Logit(
                    st.session_state["pd_y_train"], Xtr
                ).fit(disp=False, maxiter=300)

                train_pd = model.predict(Xtr).clip(0, 1)
                test_pd = model.predict(Xte).clip(0, 1)

                st.session_state["pd_logit_model"] = model
                st.session_state["pd_train_predictions"] = train_pd
                st.session_state["pd_test_predictions"] = test_pd
                st.session_state["pd_model_fitted"] = True
                st.success("Logistic regression estimated successfully.")
        except Exception as error:
            st.session_state["pd_model_fitted"] = False
            st.error(f"Model estimation failed: {error}")

if st.session_state.get("pd_model_fitted", False):
    model = st.session_state["pd_logit_model"]
    st.markdown("### Logistic regression model summary")
    st.text(model.summary().as_text())

    st.markdown("### Logistic regression coefficients")
    confidence_intervals = model.conf_int()
    coefficient_table = pd.DataFrame({
        "Coefficient": model.params,
        "Standard Error": model.bse,
        "P-value": model.pvalues,
        "Lower 95% CI": confidence_intervals[0],
        "Upper 95% CI": confidence_intervals[1]
    })
    st.dataframe(coefficient_table, use_container_width=True)


# =========================================================
# 8. MODEL PERFORMANCE AND DISCRIMINATION
# =========================================================

st.divider()
st.subheader("8. Model performance and discrimination")

if st.session_state.get("pd_model_fitted", False):
    y_test = st.session_state["pd_y_test"]
    predictions = np.asarray(st.session_state["pd_test_predictions"])

    try:
        metrics = calculate_binary_metrics(y_test, predictions)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC", f"{metrics['AUC']:.4f}")
        m2.metric("Gini", f"{metrics['Gini']:.4f}")
        m3.metric("KS statistic", f"{metrics['KS']:.4f}")
        m4.metric("Brier score", f"{metrics['Brier score']:.4f}")

        fpr, tpr, _ = roc_curve(y_test, predictions)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.write("Classification report at a 50% threshold")
        predicted_class = (predictions >= 0.5).astype(int)
        st.text(classification_report(
            y_test, predicted_class, zero_division=0
        ))
        st.caption(
            "The 50% threshold is illustrative. A lending decision threshold "
            "should reflect business objectives, risk appetite and error costs."
        )

        st.write("Confusion matrix")
        cm = confusion_matrix(y_test, predicted_class, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual non-default", "Actual default"],
            columns=["Predicted non-default", "Predicted default"]
        )
        st.dataframe(cm_df, use_container_width=True)
    except Exception as error:
        st.error(f"Performance calculation failed: {error}")
else:
    st.info("Estimate the logistic regression model to view performance metrics.")


# =========================================================
# 9. CALIBRATION ANALYSIS
# =========================================================

st.divider()
st.subheader("9. PD calibration and decile analysis")

if st.session_state.get("pd_model_fitted", False):
    y_test = st.session_state["pd_y_test"]
    predictions = st.session_state["pd_test_predictions"]
    decile_table = calculate_calibration_table(y_test, predictions)

    if not decile_table.empty:
        st.write("Predicted versus observed default rate by PD decile")
        st.dataframe(decile_table, use_container_width=True, hide_index=True)

        chart_data = decile_table.set_index("PD_Band")[
            ["Average_Predicted_PD", "Observed_Default_Rate"]
        ]
        st.line_chart(chart_data)

        overall_predicted = float(np.mean(predictions))
        overall_observed = float(np.mean(y_test))
        c1, c2, c3 = st.columns(3)
        c1.metric("Average predicted PD", f"{overall_predicted:.2%}")
        c2.metric("Observed default rate", f"{overall_observed:.2%}")
        c3.metric(
            "Difference",
            f"{(overall_predicted - overall_observed):+.2%}"
        )
        st.caption(
            "Calibration compares predicted probabilities with observed outcomes. "
            "Decile analysis is a diagnostic and does not by itself establish "
            "long-run or regulatory calibration."
        )
    else:
        st.info("There are insufficient distinct predicted PD values for decile analysis.")
else:
    st.info("Estimate the logistic regression model to activate calibration analysis.")


# =========================================================
# 10. OUT-OF-TIME VALIDATION
# =========================================================

st.divider()
st.subheader("10. Out-of-time (OOT) validation")

st.write(
    "OOT validation tests the model on a later observation period. "
    "The earlier period is used to fit the binning and logistic regression; "
    "the later period is kept separate for evaluation."
)

date_candidates = [
    column for column in source_data.columns
    if column not in [target_column, "default"]
    and (
        suggest_canonical_concept(column) == "Observation / origination date"
        or pd.api.types.is_datetime64_any_dtype(source_data[column])
    )
]
if not date_candidates:
    st.warning(
        "No likely observation/origination date was identified. OOT validation "
        "is unavailable until a reliable date variable is provided. Random "
        "train/test validation remains available."
    )
else:
    oot_date_column = st.selectbox(
        "Select the observation/origination date for OOT validation",
        options=date_candidates,
        key="pd_oot_date_column"
    )
    oot_data = source_data[selected_features + ["default", oot_date_column]].copy()
    oot_data["default"] = pd.to_numeric(oot_data["default"], errors="coerce")
    oot_data["issue_d_parsed"] = pd.to_datetime(
        oot_data[oot_date_column], errors="coerce"
    )
    oot_data = oot_data.dropna(subset=["default", "issue_d_parsed"])
    oot_data = oot_data[oot_data["default"].isin([0, 1])].copy()
    oot_data["default"] = oot_data["default"].astype(int)

    for col in selected_features:
        if pd.api.types.is_numeric_dtype(oot_data[col]):
            oot_data[col] = pd.to_numeric(oot_data[col], errors="coerce")
        else:
            oot_data[col] = (
                oot_data[col].astype("string").fillna("Missing").astype(str)
            )

    if oot_data["issue_d_parsed"].nunique() < 2:
        st.warning("At least two distinct dates are required for OOT validation.")
    else:
        st.write(
            f"Observations with valid dates and targets: {len(oot_data):,} "
            f"across {oot_data['issue_d_parsed'].nunique():,} distinct dates."
        )

        oot_fraction = st.slider(
            "Latest-period OOT validation proportion",
            min_value=0.10, max_value=0.40, value=0.20, step=0.05,
            key="pd_oot_fraction"
        )

        if st.button("Run out-of-time validation", type="primary"):
            try:
                oot_data = oot_data.sort_values("issue_d_parsed").copy()
                unique_dates = np.array(sorted(oot_data["issue_d_parsed"].unique()))
                cutoff_position = int(np.floor(
                    len(unique_dates) * (1 - oot_fraction)
                ))
                cutoff_position = min(max(cutoff_position, 1), len(unique_dates) - 1)
                cutoff_date = pd.Timestamp(unique_dates[cutoff_position])

                development = oot_data[
                    oot_data["issue_d_parsed"] < cutoff_date
                ].copy()
                validation = oot_data[
                    oot_data["issue_d_parsed"] >= cutoff_date
                ].copy()

                if (
                    len(development) < 30 or len(validation) < 10
                    or development["default"].nunique() < 2
                    or validation["default"].nunique() < 2
                ):
                    raise ValueError(
                        "The chronological development or validation sample is too "
                        "small, or does not contain both target classes. "
                        "Try a different OOT proportion or inspect the date coverage."
                    )

                oot_variables = [
                    feature for feature in st.session_state.get(
                        "pd_final_features", []
                    )
                ]
                raw_oot_features = [
                    feature[:-4] for feature in oot_variables
                    if feature.endswith("_woe")
                ]
                raw_oot_features = [
                    feature for feature in raw_oot_features
                    if feature in selected_features
                ]

                if not raw_oot_features:
                    raise ValueError(
                        "Confirm final WOE predictors in section 6.4 before running OOT."
                    )

                X_dev = development[raw_oot_features].copy()
                X_val = validation[raw_oot_features].copy()
                y_dev = development["default"]
                y_val = validation["default"]

                oot_objects, oot_train_woe, oot_iv, oot_errors = fit_binning_objects(
                    X_dev, y_dev, raw_oot_features, max_bins
                )
                if not oot_objects:
                    raise ValueError("No OOT binning objects could be fitted.")

                oot_val_woe = transform_with_binning(X_val, oot_objects)
                oot_woe_features = [
                    feature + "_woe"
                    for feature in raw_oot_features
                    if feature + "_woe" in oot_train_woe.columns
                    and feature + "_woe" in oot_val_woe.columns
                ]

                if not oot_woe_features:
                    raise ValueError("No common WOE predictors were available for OOT.")

                X_dev_model = sm.add_constant(
                    oot_train_woe[oot_woe_features], has_constant="add"
                )
                X_val_model = sm.add_constant(
                    oot_val_woe[oot_woe_features], has_constant="add"
                )

                if (
                    X_dev_model.isna().any().any()
                    or X_val_model.isna().any().any()
                ):
                    raise ValueError("Missing values were found in the OOT WOE data.")

                oot_model = sm.Logit(y_dev, X_dev_model).fit(
                    disp=False, maxiter=300
                )
                oot_predictions = np.asarray(
                    oot_model.predict(X_val_model).clip(0, 1)
                )
                oot_metrics = calculate_binary_metrics(y_val, oot_predictions)

                oot_result = {
                    "cutoff_date": cutoff_date,
                    "development": development,
                    "validation": validation,
                    "development_y": y_dev,
                    "validation_y": y_val,
                    "model": oot_model,
                    "binning_objects": oot_objects,
                    "development_woe": oot_train_woe,
                    "validation_woe": oot_val_woe,
                    "predictions": oot_predictions,
                    "metrics": oot_metrics,
                    "features": raw_oot_features,
                    "woe_features": oot_woe_features,
                    "binning_errors": oot_errors
                }
                st.session_state["pd_oot_results"] = oot_result
                st.success("Out-of-time validation completed.")
            except Exception as error:
                st.error(f"OOT validation failed: {error}")

        if "pd_oot_results" in st.session_state:
            result = st.session_state["pd_oot_results"]
            st.write(
                f"Development period: {result['development']['issue_d_parsed'].min().date()} "
                f"to {(result['cutoff_date'] - pd.Timedelta(days=1)).date()}"
            )
            st.write(
                f"OOT validation period: {result['cutoff_date'].date()} "
                f"to {result['validation']['issue_d_parsed'].max().date()}"
            )

            dev, val = st.columns(2)
            dev.metric("Development observations", f"{len(result['development']):,}")
            val.metric("OOT observations", f"{len(result['validation']):,}")

            dev, val = st.columns(2)
            dev.metric(
                "Development default rate",
                f"{result['development_y'].mean():.2%}"
            )
            val.metric(
                "OOT observed default rate",
                f"{result['validation_y'].mean():.2%}"
            )

            st.markdown("### OOT discrimination and accuracy")
            mm = result["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("OOT AUC", f"{mm['AUC']:.4f}")
            c2.metric("OOT Gini", f"{mm['Gini']:.4f}")
            c3.metric("OOT KS", f"{mm['KS']:.4f}")
            c4.metric("OOT Brier score", f"{mm['Brier score']:.4f}")

            fpr, tpr, _ = roc_curve(
                result["validation_y"], result["predictions"]
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(fpr, tpr, label=f"OOT AUC = {mm['AUC']:.4f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Out-of-time ROC curve")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            oot_calibration = calculate_calibration_table(
                result["validation_y"], result["predictions"]
            )
            st.markdown("### OOT calibration")
            if not oot_calibration.empty:
                st.dataframe(
                    oot_calibration, use_container_width=True, hide_index=True
                )
                st.line_chart(
                    oot_calibration.set_index("PD_Band")[
                        ["Average_Predicted_PD", "Observed_Default_Rate"]
                    ]
                )

            if result["binning_errors"]:
                st.warning("Some predictors could not be binned in the OOT development sample.")
                for message in result["binning_errors"]:
                    st.write(message)

            st.caption(
                "OOT results are meaningful only when issue_d represents a valid "
                "time ordering and the target definition is consistent across periods. "
                "The latest period must not influence binning or model fitting."
            )


# =========================================================
# 11. POPULATION STABILITY INDEX
# =========================================================

st.divider()
st.subheader("11. Population Stability Index (PSI)")

st.write(
    "PSI compares the distribution of a reference population with a later "
    "comparison population. Here, the random-split training sample is the "
    "reference and the random-split test sample is the comparison. "
    "Where OOT results exist, an additional OOT comparison is provided."
)

if "pd_X_train" in st.session_state and "pd_X_test" in st.session_state:
    reference_data = st.session_state["pd_X_train"]
    comparison_data = st.session_state["pd_X_test"]

    psi_rows = []
    for variable in reference_data.columns:
        value = calculate_psi(
            reference_data[variable], comparison_data[variable]
        )
        psi_rows.append({
            "Variable": variable,
            "PSI": value,
            "Interpretation": psi_interpretation(value)
        })

    psi_table = pd.DataFrame(psi_rows).sort_values("PSI", ascending=False)
    st.markdown("### Random test sample versus training sample")
    st.dataframe(psi_table, use_container_width=True, hide_index=True)

    if "pd_oot_results" in st.session_state:
        oot_result = st.session_state["pd_oot_results"]
        oot_psi_rows = []
        for variable in oot_result["features"]:
            value = calculate_psi(
                oot_result["development"][variable],
                oot_result["validation"][variable]
            )
            oot_psi_rows.append({
                "Variable": variable,
                "OOT PSI": value,
                "Interpretation": psi_interpretation(value)
            })
        oot_psi_table = pd.DataFrame(oot_psi_rows).sort_values(
            "OOT PSI", ascending=False
        )
        st.markdown("### OOT validation sample versus OOT development sample")
        st.dataframe(oot_psi_table, use_container_width=True, hide_index=True)

    st.caption(
        "Common screening conventions are PSI below 0.10 (limited change), "
        "0.10 to 0.25 (moderate change) and above 0.25 (material change). "
        "These are investigation triggers, not universal regulatory thresholds. "
        "PSI measures distribution change, not predictive performance."
    )
else:
    st.info("Prepare the random training and test datasets to calculate PSI.")


# =========================================================
# 12. GENERIC BORROWER-LEVEL PD SCORING
# =========================================================

st.divider()
st.subheader("12. Generic borrower-level PD scoring")

if st.session_state.get("pd_model_fitted", False):
    final_features = list(st.session_state["pd_final_features"])
    binning_objects = st.session_state["pd_binning_objects"]
    fitted_model = st.session_state["pd_logit_model"]

    # The fitted WOE feature names determine the exact raw input schema.
    required_raw_features = [
        feature[:-4] if feature.endswith("_woe") else feature
        for feature in final_features
    ]
    required_raw_features = list(dict.fromkeys(required_raw_features))
    st.markdown("### 12.1 Model input requirements")
    st.write(
        "Scoring is driven by the fitted model specification. Only the raw "
        "predictors selected in the final model are mandatory. Identifier "
        "columns and all other unused columns are optional and are preserved "
        "if they are present in the uploaded file."
    )
    st.write("**Required raw predictors:** " + ", ".join(required_raw_features))
    st.caption(
        "The scoring file does not need a default outcome, loan_id, age, or "
        "any other variable that is not a selected model predictor."
    )

    # Provide a minimal CSV template containing only model-required predictors.
    template = pd.DataFrame({
        variable: pd.Series(dtype="float64")
        if getattr(binning_objects[variable], "dtype", None) == "numerical"
        else pd.Series(dtype="object")
        for variable in required_raw_features
    })
    template_csv = template.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scoring template (required predictors only)",
        data=template_csv,
        file_name="pd_scoring_template.csv",
        mime="text/csv",
        help="The template contains only the raw variables required by this fitted model."
    )

    st.markdown("### 12.2 Generate synthetic new borrowers")
    st.write(
        "If you do not have a separate applicant dataset, generate synthetic "
        "records by resampling predictor profiles from the source portfolio. "
        "These records contain no default outcome and are intended to test the "
        "scoring workflow, not to represent an independent applicant population."
    )
    synthetic_count = st.number_input(
        "Number of synthetic borrower records",
        min_value=1, max_value=10000, value=100, step=25,
        key="pd_synthetic_scoring_count"
    )
    if st.button("Generate synthetic borrowers", key="pd_generate_synthetic_borrowers"):
        try:
            available_required = [
                variable for variable in required_raw_features
                if variable in source_data.columns
            ]
            if len(available_required) != len(required_raw_features):
                missing_source = sorted(set(required_raw_features) - set(available_required))
                st.error(
                    "Cannot generate synthetic records because the source "
                    "portfolio is missing: " + ", ".join(missing_source)
                )
            else:
                # Row-wise bootstrap preserves observed relationships between predictors.
                synthetic = source_data[required_raw_features].sample(
                    n=int(synthetic_count), replace=True,
                    random_state=2026
                ).reset_index(drop=True)
                synthetic.insert(
                    0, "synthetic_application_id",
                    [f"SYN{i:06d}" for i in range(1, len(synthetic) + 1)]
                )
                st.session_state["pd_synthetic_scoring_data"] = synthetic
                st.success(
                    f"Generated {len(synthetic):,} synthetic borrower records."
                )
        except Exception as error:
            st.error(f"Synthetic borrower generation failed: {error}")

    def score_borrower_dataframe(borrower_data):
        forbidden_scoring_fields = {
            "pd_probability",
            "true_pd",
            "default_probability_true",
            "latent_default_probability",
        }
        fitted_raw_features = {
            feature[:-4] if feature.endswith("_woe") else feature
            for feature in final_features
        }
        leakage_in_model = sorted(
            variable for variable in fitted_raw_features
            if str(variable).strip().lower() in forbidden_scoring_fields
        )
        if leakage_in_model:
            raise ValueError(
                "The currently fitted model contains a target-leakage field: "
                + ", ".join(leakage_in_model)
                + ". Return to predictor selection, exclude this field, and refit the model."
            )

        missing = [
            variable for variable in required_raw_features
            if variable not in borrower_data.columns
        ]
        if missing:
            raise ValueError(
                "Missing required model predictors: " + ", ".join(missing)
            )

        # Extra columns are retained in the output but never sent to the model.
        raw_model_input = borrower_data[required_raw_features].copy()
        woe_all = transform_with_binning(raw_model_input, binning_objects)
        missing_woe = [feature for feature in final_features if feature not in woe_all]
        if missing_woe:
            raise ValueError(
                "The fitted model's WOE transformations are unavailable for: "
                + ", ".join(missing_woe)
            )

        model_input = woe_all[final_features].copy()
        model_input = model_input.replace([np.inf, -np.inf], np.nan)
        if model_input.isna().any().any():
            bad_features = model_input.columns[model_input.isna().any()].tolist()
            raise ValueError(
                "WOE transformation produced missing values for: "
                + ", ".join(bad_features)
                + ". Review the source values and fitted binning definitions."
            )
        design_matrix = sm.add_constant(model_input, has_constant="add")
        probabilities = np.asarray(fitted_model.predict(design_matrix), dtype=float)
        probabilities = np.clip(probabilities, 0, 1)

        result = borrower_data.copy()
        result["Predicted_PD"] = probabilities
        result["Predicted_PD_Percent"] = probabilities * 100
        result["Illustrative_Risk_Grade"] = (
            assign_illustrative_risk_grade(probabilities).to_numpy()
        )
        return result

    synthetic_data = st.session_state.get("pd_synthetic_scoring_data")
    if synthetic_data is not None:
        st.markdown("#### Synthetic borrower preview")
        st.dataframe(synthetic_data.head(20), use_container_width=True)
        if st.button("Score generated synthetic borrowers", key="pd_score_synthetic"):
            try:
                synthetic_scores = score_borrower_dataframe(synthetic_data)
                st.session_state["pd_synthetic_scoring_results"] = synthetic_scores
                st.success(
                    f"Successfully scored {len(synthetic_scores):,} synthetic borrowers."
                )
            except Exception as error:
                st.error(f"Synthetic borrower scoring failed: {error}")

        synthetic_results = st.session_state.get("pd_synthetic_scoring_results")
        if synthetic_results is not None:
            st.dataframe(synthetic_results.head(50), use_container_width=True)
            st.download_button(
                "Download synthetic borrower PD scores",
                data=synthetic_results.to_csv(index=False).encode("utf-8"),
                file_name="synthetic_borrower_pd_scores.csv",
                mime="text/csv",
                key="pd_download_synthetic_scores"
            )

    st.markdown("### 12.3 Upload a new borrower dataset")
    uploaded_scoring_file = st.file_uploader(
        "Upload a CSV containing the required raw predictor columns",
        type=["csv"],
        key="pd_new_borrower_upload"
    )

    if uploaded_scoring_file is not None:
        try:
            new_borrowers = pd.read_csv(uploaded_scoring_file)
            missing_columns = [
                col for col in required_raw_features
                if col not in new_borrowers.columns
            ]
            if missing_columns:
                st.error(
                    "The uploaded file is missing required model predictors: "
                    + ", ".join(missing_columns)
                )
            elif new_borrowers.empty:
                st.warning("The uploaded CSV contains no borrower records.")
            else:
                st.write(
                    f"Uploaded {len(new_borrowers):,} records and "
                    f"{len(new_borrowers.columns):,} columns."
                )
                ignored_columns = [
                    col for col in new_borrowers.columns
                    if col not in required_raw_features
                ]
                if ignored_columns:
                    st.caption(
                        "Additional columns will be retained in the output "
                        "but ignored by the model: " + ", ".join(ignored_columns)
                    )
                if st.button("Score uploaded borrowers", key="pd_score_uploaded"):
                    scored_new = score_borrower_dataframe(new_borrowers)
                    st.session_state["pd_uploaded_scoring_results"] = scored_new
                    st.success(
                        f"Successfully scored {len(scored_new):,} borrower records."
                    )

                uploaded_results = st.session_state.get("pd_uploaded_scoring_results")
                if uploaded_results is not None:
                    st.dataframe(uploaded_results.head(50), use_container_width=True)
                    st.download_button(
                        "Download new borrower PD scores",
                        data=uploaded_results.to_csv(index=False).encode("utf-8"),
                        file_name="new_borrower_pd_scores.csv",
                        mime="text/csv",
                        key="pd_download_uploaded_scores"
                    )
        except Exception as error:
            st.error(f"New borrower scoring failed: {error}")

    st.markdown("### 12.4 Held-out test sample scoring")
    test_woe = st.session_state["pd_test_woe"]
    scored_data = test_woe.copy()
    scored_data["Predicted_PD"] = np.asarray(
        st.session_state["pd_test_predictions"]
    )
    scored_data["Predicted_PD_Percent"] = scored_data["Predicted_PD"] * 100
    scored_data["Illustrative_Risk_Grade"] = assign_illustrative_risk_grade(
        scored_data["Predicted_PD"]
    )
    scored_data["Actual_Default"] = st.session_state["pd_y_test"].to_numpy()
    st.dataframe(scored_data.head(25), use_container_width=True)
    st.download_button(
        "Download test-set PD scoring results",
        data=scored_data.to_csv(index=True).encode("utf-8"),
        file_name="pd_test_scoring_results.csv",
        mime="text/csv",
        key="pd_download_test_scores"
    )

    st.caption(
        "Risk grades are illustrative PD bands for demonstration only. "
        "They are not calibrated bank rating grades and must be approved "
        "before operational use. Missing numerical and categorical values "
        "are passed through the fitted OptBinning transformations; the "
        "application does not invent replacement values."
    )
else:
    st.info("Estimate logistic regression to activate generic borrower-level scoring.")


# =========================================================
# 13. MODEL EXPORT
# =========================================================

st.divider()
st.subheader("13. Export fitted PD model")

if st.session_state.get("pd_model_fitted", False):
    model_bundle = {
        "model_type": "WOE logistic regression PD model",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features": st.session_state["pd_final_features"],
        "required_raw_predictors": [
            feature[:-4] if feature.endswith("_woe") else feature
            for feature in st.session_state["pd_final_features"]
        ],
        "binning_objects": st.session_state["pd_binning_objects"],
        "logistic_regression_model": st.session_state["pd_logit_model"],
        "target": "default",
        "target_source_column": st.session_state.get("pd_target_source_column", "default"),
        "required_raw_predictors": [
            feature[:-4] if feature.endswith("_woe") else feature
            for feature in st.session_state["pd_final_features"]
        ],
        "source_variable_names": list(source_data.columns),
        "notes": (
            "Educational model bundle. Validate model performance, "
            "calibration, stability, data definitions and governance "
            "before production use."
        )
    }

    bundle_bytes = pickle.dumps(model_bundle)
    st.download_button(
        "Download fitted PD model bundle",
        data=bundle_bytes,
        file_name="creditrisk_lab_pd_model.pkl",
        mime="application/octet-stream"
    )
    st.warning(
        "Only load pickle files from trusted sources. The exported bundle "
        "contains Python model objects and is intended for use in the same "
        "compatible Python environment."
    )
else:
    st.info("Fit the logistic regression model before exporting it.")


# =========================================================
# 14. MODEL DEVELOPMENT SUMMARY
# =========================================================

st.divider()
st.subheader("14. PD model development summary")

summary_rows = [
    {
        "Component": "Random train/test split",
        "Status": "Completed" if "pd_X_train" in st.session_state else "Not run"
    },
    {
        "Component": "Optimal binning and WOE",
        "Status": "Completed" if "pd_binning_objects" in st.session_state else "Not run"
    },
    {
        "Component": "Final predictor selection",
        "Status": "Completed" if "pd_final_features" in st.session_state else "Not confirmed"
    },
    {
        "Component": "Logistic regression",
        "Status": "Completed" if st.session_state.get("pd_model_fitted", False) else "Not fitted"
    },
    {
        "Component": "Out-of-time validation",
        "Status": "Completed" if "pd_oot_results" in st.session_state else "Not run"
    }
]
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.caption(
    "CreditRisk Lab | Educational and analytical modelling environment. "
    "Model outputs require independent validation, appropriate calibration, "
    "governance approval and monitoring before production or regulatory use."
)
