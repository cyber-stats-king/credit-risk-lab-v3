"""
CreditRisk Lab
Integrated Credit Risk Tool

Integrates:
- Registered PD, EAD and LGD models
- Central portfolio and prediction management
- IFRS 9 staging
- Scenario-weighted 12-month and lifetime ECL
- Portfolio reporting and downloadable account-level results

Initial product scope: unsecured personal loans and revolving credit.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.portfolio_manager import (
    PortfolioManager,
    PortfolioError,
    create_portfolio_manager,
)

from engine.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    ModelNotFoundError,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Integrated Credit Risk Tool | CreditRisk Lab",
    page_icon="📊",
    layout="wide",
)

st.title("Integrated Credit Risk Tool")
st.caption(
    "Portfolio-level credit risk modelling, IFRS 9 staging and "
    "expected credit loss calculation."
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .integrated-header {
        background: linear-gradient(120deg, #10243A, #1D4263);
        padding: 25px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 22px;
    }
    .integrated-header h2 {
        color: white;
        margin-bottom: 7px;
    }
    .integrated-header p {
        color: #DCE8F2;
        margin-bottom: 0;
    }
    .section-heading {
        color: #173B5E;
        font-weight: 700;
        font-size: 1.15rem;
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="integrated-header">
        <h2>Integrated Credit Risk & IFRS 9 Engine</h2>
        <p>
            Consolidate model outputs, determine impairment stages,
            calculate expected credit losses and review portfolio risk.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE AND SERVICES
# ============================================================

@st.cache_resource
def get_registry():
    return ModelRegistry()


@st.cache_resource
def get_portfolio_manager():
    return PortfolioManager()


registry = get_registry()
portfolio_manager = get_portfolio_manager()


def get_portfolio():
    """
    Retrieve the central portfolio.

    Supports the PortfolioManager session-state interface and
    the original credit_data compatibility interface.
    """

    try:
        portfolio_manager.load_from_session_state()
        original = portfolio_manager.get_original_portfolio()

        if original is not None and not original.empty:
            return original.copy()

    except Exception:
        pass

    credit_data = st.session_state.get("credit_data")

    if isinstance(credit_data, pd.DataFrame) and not credit_data.empty:
        return credit_data.copy()

    return None


def get_predictions():
    try:
        return portfolio_manager.get_prediction_portfolio()
    except Exception:
        return None


portfolio = get_portfolio()

if portfolio is None:
    st.warning(
        "No portfolio is currently loaded. Go to **Data & Models** "
        "and load or generate a portfolio before using this page."
    )
    st.stop()


# ============================================================
# PORTFOLIO VALIDATION
# ============================================================

if "loan_id" not in portfolio.columns:
    st.error(
        "The portfolio must contain a unique `loan_id` column. "
        "The integrated tool uses loan_id to join model predictions "
        "and preserve account-level records."
    )
    st.stop()

if portfolio["loan_id"].isna().any():
    st.error("The portfolio contains missing loan_id values.")
    st.stop()

if portfolio["loan_id"].duplicated().any():
    st.error("The portfolio contains duplicate loan_id values.")
    st.stop()

portfolio = portfolio.copy()
portfolio["loan_id"] = portfolio["loan_id"].astype(str)


# ============================================================
# HELPERS
# ============================================================

def numeric_series(data, column, default=np.nan):
    if column is None or column not in data.columns:
        return pd.Series(default, index=data.index, dtype=float)

    return pd.to_numeric(
        data[column],
        errors="coerce",
    )


def probability_series(series):
    """
    Convert percentage-form probabilities to decimal form.

    Values above 1 are interpreted as percentages.
    """

    result = pd.to_numeric(series, errors="coerce").astype(float)

    percentage_mask = result > 1
    result.loc[percentage_mask] = (
        result.loc[percentage_mask] / 100
    )

    return result.clip(0, 1)


def safe_mean(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def format_currency(value):
    if pd.isna(value):
        return "N/A"

    return f"R {value:,.2f}"


def annual_marginal_pd(lifetime_pd, horizon):
    """
    Convert cumulative lifetime PD into annual marginal PDs
    under a constant conditional hazard assumption.

    This is an approximation. It is not a substitute for a
    calibrated period-specific PD term structure.
    """

    lifetime_pd = float(np.clip(lifetime_pd, 0, 1))
    horizon = max(int(horizon), 1)

    if lifetime_pd >= 1:
        conditional_pd = 1.0
    else:
        conditional_pd = 1 - (
            (1 - lifetime_pd) ** (1 / horizon)
        )

    survival = 1.0
    marginal = []

    for _ in range(horizon):
        period_pd = survival * conditional_pd
        marginal.append(period_pd)
        survival *= (1 - conditional_pd)

    return marginal


def calculate_account_ecl(
    ead,
    lgd,
    lifetime_pd,
    stage,
    horizon,
    eir,
    scenario_pd_multiplier,
    scenario_lgd_multiplier,
):
    """
    Calculate discounted ECL using annual marginal PDs.

    Stage 1:
        Defaults possible over the next 12 months.

    Stage 2:
        Defaults possible over the remaining lifetime.

    Stage 3:
        Credit-impaired exposure. The simplified calculation
        recognises the current expected loss using EAD x LGD.

    The Stage 3 calculation is an approximation and does not
    replace a contractual cash-shortfall/workout cash-flow model.
    """

    ead = max(float(ead), 0.0)
    lgd = float(np.clip(lgd, 0, 1))
    lifetime_pd = float(np.clip(lifetime_pd, 0, 1))
    horizon = max(int(horizon), 1)

    if stage == 3:
        return ead * lgd * scenario_lgd_multiplier

    marginal_pds = annual_marginal_pd(
        lifetime_pd,
        horizon,
    )

    if stage == 1:
        marginal_pds = marginal_pds[:1]

    total_ecl = 0.0

    for period, marginal_pd in enumerate(marginal_pds, start=1):

        adjusted_pd = min(
            marginal_pd * scenario_pd_multiplier,
            1.0,
        )

        adjusted_lgd = min(
            lgd * scenario_lgd_multiplier,
            1.0,
        )

        discount_factor = (
            1 + max(float(eir), 0.0)
        ) ** period

        period_ecl = (
            adjusted_pd
            * ead
            * adjusted_lgd
            / discount_factor
        )

        total_ecl += period_ecl

    return total_ecl


def model_prediction(model, model_type, data, metadata):
    """
    Attempt to score an active registered model.

    Supports common scikit-learn estimator interfaces.
    Models requiring specialised transformations must be scored
    through their original modelling-page scoring workflow.
    """

    features = metadata.get("features", [])

    if not features:
        raise ValueError(
            "The registered model does not specify its predictor features."
        )

    missing = [
        feature for feature in features
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model predictors: {missing}"
        )

    X = data[features].copy()

    if hasattr(model, "predict_proba"):
        prediction = model.predict_proba(X)

        if np.asarray(prediction).ndim == 2:
            prediction = np.asarray(prediction)[:, -1]

    elif hasattr(model, "predict"):
        prediction = model.predict(X)

    else:
        raise ValueError(
            "The registered artifact does not expose a supported "
            "predict or predict_proba method."
        )

    prediction = np.asarray(prediction).reshape(-1)

    if len(prediction) != len(data):
        raise ValueError(
            "The model prediction count does not match the portfolio."
        )

    return prediction


# ============================================================
# PORTFOLIO OVERVIEW
# ============================================================

st.markdown("### 1. Portfolio overview")

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Portfolio accounts", f"{len(portfolio):,}")

with metric2:
    balance_column = next(
        (
            column for column in [
                "current_balance",
                "ead",
                "loan_amount",
            ]
            if column in portfolio.columns
        ),
        None,
    )

    total_balance = (
        numeric_series(portfolio, balance_column)
        .fillna(0)
        .sum()
        if balance_column
        else 0
    )

    st.metric(
        "Portfolio exposure",
        format_currency(total_balance),
    )

with metric3:
    st.metric(
        "Unique accounts",
        f"{portfolio['loan_id'].nunique():,}",
    )

with metric4:
    product_column = next(
        (
            column for column in [
                "product_type",
                "loan_purpose",
            ]
            if column in portfolio.columns
        ),
        None,
    )

    st.metric(
        "Product categories",
        (
            f"{portfolio[product_column].nunique():,}"
            if product_column
            else "Not specified"
        ),
    )


with st.expander("View portfolio data", expanded=False):
    st.dataframe(
        portfolio.head(500),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

st.markdown("### 2. Registered model management")

registry_summary = pd.DataFrame(
    registry.registry_summary()
)

st.dataframe(
    registry_summary,
    use_container_width=True,
    hide_index=True,
)

model_columns = st.columns(3)

model_types = ["PD", "EAD", "LGD"]

for column, model_type in zip(model_columns, model_types):

    with column:

        st.markdown(f"**{model_type} model**")

        try:
            active_id = registry.get_active_model_id(model_type)

            if active_id:
                metadata = registry.get_model_metadata(
                    model_type,
                    active_id,
                )

                st.success(f"Active: {active_id}")
                st.caption(
                    f"Algorithm: {metadata.get('algorithm') or 'Not specified'}"
                )

            else:
                st.warning("No active model")

        except ModelRegistryError as error:
            st.warning(str(error))


with st.expander("View registered model versions"):
    try:
        registered_models = pd.DataFrame(
            registry.list_models()
        )

        if registered_models.empty:
            st.info("No registered models are available.")
        else:
            display_columns = [
                column for column in [
                    "model_id",
                    "model_type",
                    "algorithm",
                    "version",
                    "target",
                    "training_observations",
                    "is_active",
                    "registered_at",
                ]
                if column in registered_models.columns
            ]

            st.dataframe(
                registered_models[display_columns],
                use_container_width=True,
                hide_index=True,
            )

    except ModelRegistryError as error:
        st.error(str(error))


# ============================================================
# PREDICTION PORTFOLIO
# ============================================================

st.markdown("### 3. Model predictions")

st.write(
    "The integrated tool uses predictions already merged into the "
    "central prediction portfolio. Registered models can also be "
    "scored below when their artifacts expose a compatible prediction "
    "interface and all required predictors are available."
)

prediction_portfolio = get_predictions()

if prediction_portfolio is not None and not prediction_portfolio.empty:

    prediction_portfolio = prediction_portfolio.copy()

    if "loan_id" in prediction_portfolio.columns:

        prediction_portfolio["loan_id"] = (
            prediction_portfolio["loan_id"].astype(str)
        )

        prediction_portfolio = prediction_portfolio.drop_duplicates(
            subset=["loan_id"]
        )

        portfolio = portfolio.merge(
            prediction_portfolio,
            on="loan_id",
            how="left",
            suffixes=("", "_prediction"),
            validate="one_to_one",
        )

        st.success(
            "Central prediction portfolio loaded and matched using loan_id."
        )

else:
    st.info(
        "No central prediction portfolio is available yet. "
        "You can use existing prediction columns in the original portfolio "
        "or score compatible registered models."
    )


score_columns = st.columns(3)

for column, model_type in zip(score_columns, model_types):

    with column:

        st.markdown(f"**Score {model_type} model**")

        if st.button(
            f"Run {model_type} scoring",
            key=f"score_{model_type.lower()}",
            use_container_width=True,
        ):

            try:
                active_id = registry.get_active_model_id(model_type)

                if not active_id:
                    st.warning(
                        f"No active {model_type} model is registered."
                    )
                    continue

                model = registry.load_model(
                    model_type,
                    active_id,
                )

                metadata = registry.get_model_metadata(
                    model_type,
                    active_id,
                )

                predictions = model_prediction(
                    model,
                    model_type,
                    portfolio,
                    metadata,
                )

                if model_type == "PD":
                    prediction_column = "predicted_pd"

                elif model_type == "EAD":
                    prediction_column = "predicted_ead"

                else:
                    prediction_column = "predicted_lgd"

                if model_type == "PD":
                    predictions = probability_series(
                        pd.Series(predictions)
                    ).to_numpy()

                elif model_type == "LGD":
                    predictions = np.clip(
                        predictions,
                        0,
                        1,
                    )

                scoring_output = pd.DataFrame({
                    "loan_id": portfolio["loan_id"].values,
                    prediction_column: predictions,
                })

                if model_type == "EAD":
                    scoring_output["predicted_ead"] = predictions

                portfolio_manager.merge_predictions(
                    scoring_output,
                    model_type=model_type,
                    allow_partial=False,
                )

                portfolio_manager.sync_to_session_state()

                st.success(
                    f"{model_type} model scored and merged successfully."
                )

                st.rerun()

            except Exception as error:
                st.error(
                    f"{model_type} scoring could not be completed: {error}"
                )


# ============================================================
# MODEL OUTPUT SELECTION
# ============================================================

st.markdown("### 4. Configure account-level risk inputs")

st.caption(
    "Select the columns containing the PD, EAD and LGD estimates. "
    "The tool recognises the canonical prediction column names."
)

available_columns = list(portfolio.columns)


def choose_column(label, preferred, alternatives=None):

    alternatives = alternatives or []

    candidates = [
        preferred,
        *alternatives,
    ]

    default_index = 0

    options = ["Not available"] + available_columns

    for candidate in candidates:
        if candidate in available_columns:
            default_index = options.index(candidate)
            break

    selected = st.selectbox(
        label,
        options,
        index=default_index,
        key=f"integrated_{label}",
    )

    return None if selected == "Not available" else selected


input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    pd_column = choose_column(
        "12-month PD",
        "predicted_pd",
        ["pd_probability", "pd", "probability_of_default"],
    )

with input_col2:
    ead_column = choose_column(
        "EAD",
        "predicted_ead",
        ["ead", "exposure_at_default", "current_balance"],
    )

with input_col3:
    lgd_column = choose_column(
        "LGD",
        "predicted_lgd",
        ["adjusted_lgd", "lgd", "loss_given_default"],
    )


origination_pd_column = choose_column(
    "Origination lifetime PD",
    "origination_lifetime_pd",
    ["original_lifetime_pd", "lifetime_pd_at_origination"],
)

current_lifetime_pd_column = choose_column(
    "Current lifetime PD",
    "current_lifetime_pd",
    ["lifetime_pd", "predicted_lifetime_pd"],
)

dpd_column = choose_column(
    "Days past due",
    "days_past_due",
    ["dpd", "days_past_due_current"],
)

default_column = choose_column(
    "Default indicator",
    "default",
    ["is_default", "default_flag"],
)

eir_column = choose_column(
    "Effective interest rate",
    "effective_interest_rate",
    ["eir", "interest_rate"],
)

st.markdown("#### Calculation assumptions")

assumption_col1, assumption_col2, assumption_col3 = st.columns(3)

with assumption_col1:
    lifetime_horizon = st.number_input(
        "Lifetime horizon (years)",
        min_value=1,
        max_value=40,
        value=5,
        step=1,
    )

with assumption_col2:
    default_dpd_threshold = st.number_input(
        "Default presumption (DPD)",
        min_value=1,
        max_value=365,
        value=90,
    )

with assumption_col3:
    sicr_dpd_threshold = st.number_input(
        "SICR presumption (DPD)",
        min_value=1,
        max_value=89,
        value=30,
    )


st.caption(
    "The 30 DPD SICR and 90 DPD default thresholds are rebuttable "
    "presumptions. Any rebuttal must be supported by documented "
    "institutional policy and evidence."
)


# ============================================================
# STAGING AND SICR CONFIGURATION
# ============================================================

st.markdown("### 5. IFRS 9 staging")

st.write(
    "The staging assessment distinguishes credit deterioration "
    "since initial recognition from current default status."
)

sicr_col1, sicr_col2 = st.columns(2)

with sicr_col1:
    sicr_multiplier = st.number_input(
        "Lifetime PD deterioration threshold",
        min_value=1.0,
        max_value=20.0,
        value=2.0,
        step=0.1,
        help=(
            "Illustrative relative deterioration threshold. "
            "IFRS 9 does not prescribe a universal multiplier."
        ),
    )

with sicr_col2:
    sicr_absolute_increase = st.number_input(
        "Minimum absolute lifetime PD increase",
        min_value=0.0,
        max_value=1.0,
        value=0.02,
        step=0.01,
        format="%.3f",
        help="Optional additional SICR trigger.",
    )


# ============================================================
# SCENARIO CONFIGURATION
# ============================================================

st.markdown("### 6. Forward-looking economic scenarios")

st.write(
    "ECL is calculated under multiple economic scenarios and "
    "weighted using the configured scenario probabilities."
)

scenario_data = pd.DataFrame(
    [
        {
            "Scenario": "Baseline",
            "Weight": 0.60,
            "PD multiplier": 1.00,
            "LGD multiplier": 1.00,
        },
        {
            "Scenario": "Upside",
            "Weight": 0.20,
            "PD multiplier": 0.85,
            "LGD multiplier": 0.90,
        },
        {
            "Scenario": "Downside",
            "Weight": 0.20,
            "PD multiplier": 1.30,
            "LGD multiplier": 1.15,
        },
    ]
)

edited_scenarios = st.data_editor(
    scenario_data,
    use_container_width=True,
    hide_index=True,
    disabled=["Scenario"],
    column_config={
        "Weight": st.column_config.NumberColumn(
            "Weight",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
        ),
        "PD multiplier": st.column_config.NumberColumn(
            "PD multiplier",
            min_value=0.0,
            max_value=10.0,
            step=0.05,
        ),
        "LGD multiplier": st.column_config.NumberColumn(
            "LGD multiplier",
            min_value=0.0,
            max_value=10.0,
            step=0.05,
        ),
    },
)

scenario_weights = pd.to_numeric(
    edited_scenarios["Weight"],
    errors="coerce",
).fillna(0)

if (
    (scenario_weights < 0).any()
    or scenario_weights.sum() <= 0
):
    st.error("Scenario weights must be non-negative and sum to a positive value.")
    st.stop()

scenario_weights = scenario_weights / scenario_weights.sum()

edited_scenarios = edited_scenarios.copy()
edited_scenarios["Weight"] = scenario_weights


# ============================================================
# PREPARE RISK INPUTS
# ============================================================

if pd_column is None or ead_column is None or lgd_column is None:
    st.warning(
        "PD, EAD and LGD columns must all be available before "
        "the ECL calculation can run."
    )
    st.stop()


work = portfolio.copy()

work["calculation_pd"] = probability_series(
    numeric_series(work, pd_column)
)

work["calculation_ead"] = numeric_series(
    work,
    ead_column,
)

work["calculation_lgd"] = probability_series(
    numeric_series(work, lgd_column)
)

work["calculation_eir"] = (
    numeric_series(work, eir_column).fillna(0)
    if eir_column
    else pd.Series(0.0, index=work.index)
)

# Normalise EIR if the source stores it as a percentage.
work.loc[
    work["calculation_eir"] > 1,
    "calculation_eir",
] = (
    work.loc[
        work["calculation_eir"] > 1,
        "calculation_eir",
    ] / 100
)

work["calculation_eir"] = work[
    "calculation_eir"
].clip(lower=0)

if current_lifetime_pd_column:
    work["calculation_lifetime_pd"] = probability_series(
        numeric_series(work, current_lifetime_pd_column)
    )
else:
    # Explicit approximation from 12-month PD under a constant
    # conditional hazard assumption.
    work["calculation_lifetime_pd"] = 1 - (
        (1 - work["calculation_pd"])
        ** int(lifetime_horizon)
    )

if origination_pd_column:
    work["calculation_origination_lifetime_pd"] = probability_series(
        numeric_series(work, origination_pd_column)
    )
else:
    work["calculation_origination_lifetime_pd"] = np.nan

work["calculation_dpd"] = (
    numeric_series(work, dpd_column).fillna(0)
    if dpd_column
    else pd.Series(0.0, index=work.index)
)

if default_column:
    default_values = work[default_column]

    if default_values.dtype == object:
        default_values = (
            default_values.astype(str)
            .str.strip()
            .str.lower()
            .isin(["1", "true", "yes", "default", "y"])
        )

    work["calculation_default"] = (
        pd.Series(default_values, index=work.index)
        .fillna(False)
        .astype(bool)
    )
else:
    work["calculation_default"] = False


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

st.markdown("### 7. Input quality and readiness")

missing_pd = work["calculation_pd"].isna().sum()
missing_ead = work["calculation_ead"].isna().sum()
missing_lgd = work["calculation_lgd"].isna().sum()

invalid_ead = (
    work["calculation_ead"].notna()
    & (work["calculation_ead"] < 0)
).sum()

invalid_lgd = (
    work["calculation_lgd"].notna()
    & (
        (work["calculation_lgd"] < 0)
        | (work["calculation_lgd"] > 1)
    )
).sum()

quality_columns = st.columns(5)

quality_columns[0].metric("Missing PD", f"{missing_pd:,}")
quality_columns[1].metric("Missing EAD", f"{missing_ead:,}")
quality_columns[2].metric("Missing LGD", f"{missing_lgd:,}")
quality_columns[3].metric("Negative EAD", f"{invalid_ead:,}")
quality_columns[4].metric("Invalid LGD", f"{invalid_lgd:,}")

if missing_pd or missing_ead or missing_lgd or invalid_ead or invalid_lgd:
    st.warning(
        "Some accounts have incomplete or invalid risk inputs. "
        "Those accounts will be excluded from the ECL calculation "
        "and reported separately."
    )


# ============================================================
# STAGE ASSIGNMENT
# ============================================================

st.markdown("### 8. Determine impairment stage")

work["stage"] = 1

# Stage 3: default / credit-impaired.
work.loc[
    work["calculation_default"]
    | (
        work["calculation_dpd"]
        >= default_dpd_threshold
    ),
    "stage",
] = 3

# SICR based on lifetime PD deterioration from initial recognition.
has_origination_pd = (
    work["calculation_origination_lifetime_pd"].notna()
    & work["calculation_lifetime_pd"].notna()
)

pd_deterioration = (
    work["calculation_lifetime_pd"]
    >= (
        work["calculation_origination_lifetime_pd"]
        * sicr_multiplier
    )
)

pd_absolute_increase = (
    work["calculation_lifetime_pd"]
    - work["calculation_origination_lifetime_pd"]
) >= sicr_absolute_increase

sicr_trigger = (
    has_origination_pd
    & (pd_deterioration | pd_absolute_increase)
)

dpd_sicr_trigger = (
    work["calculation_dpd"] >= sicr_dpd_threshold
)

stage_2_trigger = sicr_trigger | dpd_sicr_trigger

work.loc[
    (work["stage"] != 3) & stage_2_trigger,
    "stage",
] = 2

# Keep the reason for each stage assignment.
work["stage_reason"] = "No identified SICR trigger"

work.loc[
    dpd_sicr_trigger,
    "stage_reason",
] = "30 DPD SICR presumption"

work.loc[
    sicr_trigger,
    "stage_reason",
] = "Lifetime PD deterioration"

work.loc[
    work["calculation_default"],
    "stage_reason",
] = "Default indicator"

work.loc[
    work["calculation_dpd"] >= default_dpd_threshold,
    "stage_reason",
] = "90 DPD default presumption"

# Origination lifetime PD is required for a full quantitative
# SICR comparison. Flag accounts where that comparison is unavailable.
work["sicr_assessment_status"] = np.where(
    has_origination_pd,
    "Origination comparison available",
    "Origination lifetime PD unavailable",
)

work.loc[
    dpd_sicr_trigger,
    "sicr_assessment_status",
] = "SICR presumption triggered by DPD"

work.loc[
    work["stage"] == 3,
    "sicr_assessment_status",
] = "Credit-impaired / default assessment"


# ============================================================
# ECL CALCULATION
# ============================================================

st.markdown("### 9. Calculate expected credit losses")

st.write(
    "The calculation applies 12-month ECL to Stage 1, lifetime ECL "
    "to Stage 2 and a simplified credit-impaired loss estimate to "
    "Stage 3."
)

valid_inputs = (
    work["calculation_pd"].notna()
    & work["calculation_ead"].notna()
    & work["calculation_lgd"].notna()
    & (work["calculation_ead"] >= 0)
)

work["ECL"] = np.nan

for index in work.index:

    if not valid_inputs.loc[index]:
        continue

    account_ecl = 0.0

    for _, scenario in edited_scenarios.iterrows():

        scenario_ecl = calculate_account_ecl(
            ead=work.at[index, "calculation_ead"],
            lgd=work.at[index, "calculation_lgd"],
            lifetime_pd=work.at[index, "calculation_lifetime_pd"],
            stage=int(work.at[index, "stage"]),
            horizon=int(lifetime_horizon),
            eir=work.at[index, "calculation_eir"],
            scenario_pd_multiplier=float(scenario["PD multiplier"]),
            scenario_lgd_multiplier=float(scenario["LGD multiplier"]),
        )

        account_ecl += (
            float(scenario["Weight"])
            * scenario_ecl
        )

    work.at[index, "ECL"] = account_ecl

work["ECL_12M"] = np.nan
work["ECL_Lifetime"] = np.nan

for index in work.index:

    if not valid_inputs.loc[index]:
        continue

    ead = work.at[index, "calculation_ead"]
    lgd = work.at[index, "calculation_lgd"]
    lifetime_pd = work.at[index, "calculation_lifetime_pd"]
    eir = work.at[index, "calculation_eir"]

    ecl_12m = 0.0
    ecl_lifetime = 0.0

    for _, scenario in edited_scenarios.iterrows():

        weight = float(scenario["Weight"])
        pd_multiplier = float(scenario["PD multiplier"])
        lgd_multiplier = float(scenario["LGD multiplier"])

        ecl_12m += weight * calculate_account_ecl(
            ead,
            lgd,
            lifetime_pd,
            1,
            lifetime_horizon,
            eir,
            pd_multiplier,
            lgd_multiplier,
        )

        ecl_lifetime += weight * calculate_account_ecl(
            ead,
            lgd,
            lifetime_pd,
            2,
            lifetime_horizon,
            eir,
            pd_multiplier,
            lgd_multiplier,
        )

    work.at[index, "ECL_12M"] = ecl_12m
    work.at[index, "ECL_Lifetime"] = ecl_lifetime


work["ECL_Percentage_of_EAD"] = np.where(
    work["calculation_ead"] > 0,
    work["ECL"] / work["calculation_ead"],
    np.nan,
)


# ============================================================
# PORTFOLIO ECL SUMMARY
# ============================================================

st.markdown("### 10. Portfolio ECL results")

valid_results = work.loc[work["ECL"].notna()].copy()

if valid_results.empty:
    st.error(
        "No accounts have complete PD, EAD and LGD inputs. "
        "Review the selected columns and input data."
    )
    st.stop()

total_ead = valid_results["calculation_ead"].sum()
total_ecl = valid_results["ECL"].sum()
coverage_ratio = (
    total_ecl / total_ead
    if total_ead > 0
    else np.nan
)

stage1_ecl = valid_results.loc[
    valid_results["stage"] == 1,
    "ECL",
].sum()

stage2_ecl = valid_results.loc[
    valid_results["stage"] == 2,
    "ECL",
].sum()

stage3_ecl = valid_results.loc[
    valid_results["stage"] == 3,
    "ECL",
].sum()

result_columns = st.columns(5)

result_columns[0].metric(
    "Calculated accounts",
    f"{len(valid_results):,}",
)

result_columns[1].metric(
    "Total EAD",
    format_currency(total_ead),
)

result_columns[2].metric(
    "Total ECL",
    format_currency(total_ecl),
)

result_columns[3].metric(
    "ECL coverage",
    f"{coverage_ratio:.2%}" if pd.notna(coverage_ratio) else "N/A",
)

result_columns[4].metric(
    "Excluded accounts",
    f"{len(work) - len(valid_results):,}",
)


# ============================================================
# STAGE SUMMARY
# ============================================================

st.markdown("### 11. Stage distribution")

stage_summary = (
    work.groupby("stage")
    .agg(
        Accounts=("loan_id", "count"),
        EAD=("calculation_ead", "sum"),
        ECL=("ECL", "sum"),
    )
    .reset_index()
)

stage_summary["Stage"] = (
    "Stage " + stage_summary["stage"].astype(str)
)

stage_summary["ECL Coverage"] = np.where(
    stage_summary["EAD"] > 0,
    stage_summary["ECL"] / stage_summary["EAD"],
    np.nan,
)

st.dataframe(
    stage_summary[
        [
            "Stage",
            "Accounts",
            "EAD",
            "ECL",
            "ECL Coverage",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "EAD": st.column_config.NumberColumn(
            format="R %.2f",
        ),
        "ECL": st.column_config.NumberColumn(
            format="R %.2f",
        ),
        "ECL Coverage": st.column_config.NumberColumn(
            format="%.2%%",
        ),
    },
)


# ============================================================
# STAGE CHARTS
# ============================================================

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Accounts by stage**")

    account_chart = (
        stage_summary.set_index("Stage")["Accounts"]
    )

    st.bar_chart(account_chart)

with chart_col2:
    st.markdown("**ECL by stage**")

    ecl_chart = (
        stage_summary.set_index("Stage")["ECL"]
    )

    st.bar_chart(ecl_chart)


# ============================================================
# SCENARIO CONTRIBUTION
# ============================================================

st.markdown("### 12. Scenario analysis")

scenario_results = []

for _, scenario in edited_scenarios.iterrows():

    scenario_total = 0.0

    for index in valid_results.index:

        scenario_total += calculate_account_ecl(
            ead=work.at[index, "calculation_ead"],
            lgd=work.at[index, "calculation_lgd"],
            lifetime_pd=work.at[index, "calculation_lifetime_pd"],
            stage=int(work.at[index, "stage"]),
            horizon=int(lifetime_horizon),
            eir=work.at[index, "calculation_eir"],
            scenario_pd_multiplier=float(scenario["PD multiplier"]),
            scenario_lgd_multiplier=float(scenario["LGD multiplier"]),
        )

    scenario_results.append({
        "Scenario": scenario["Scenario"],
        "Weight": scenario["Weight"],
        "Scenario ECL": scenario_total,
        "Weighted ECL": scenario_total * scenario["Weight"],
    })

scenario_summary = pd.DataFrame(scenario_results)

st.dataframe(
    scenario_summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Weight": st.column_config.NumberColumn(
            format="%.2%%",
        ),
        "Scenario ECL": st.column_config.NumberColumn(
            format="R %.2f",
        ),
        "Weighted ECL": st.column_config.NumberColumn(
            format="R %.2f",
        ),
    },
)


# ============================================================
# ACCOUNT-LEVEL RESULTS
# ============================================================

st.markdown("### 13. Account-level results")

account_result_columns = [
    "loan_id",
    "stage",
    "stage_reason",
    "sicr_assessment_status",
    "calculation_pd",
    "calculation_lifetime_pd",
    "calculation_origination_lifetime_pd",
    "calculation_dpd",
    "calculation_ead",
    "calculation_lgd",
    "calculation_eir",
    "ECL_12M",
    "ECL_Lifetime",
    "ECL",
    "ECL_Percentage_of_EAD",
]

account_result_columns = [
    column for column in account_result_columns
    if column in work.columns
]

account_results = work[account_result_columns].copy()

account_results = account_results.rename(
    columns={
        "stage": "IFRS9_Stage",
        "stage_reason": "Stage_Reason",
        "sicr_assessment_status": "SICR_Assessment_Status",
        "calculation_pd": "12M_PD",
        "calculation_lifetime_pd": "Current_Lifetime_PD",
        "calculation_origination_lifetime_pd": "Origination_Lifetime_PD",
        "calculation_dpd": "Days_Past_Due",
        "calculation_ead": "EAD",
        "calculation_lgd": "LGD",
        "calculation_eir": "EIR",
        "ECL_12M": "ECL_12M",
        "ECL_Lifetime": "ECL_Lifetime",
        "ECL_Percentage_of_EAD": "ECL_to_EAD",
    }
)

st.dataframe(
    account_results.head(1000),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download account-level ECL results",
    data=account_results.to_csv(index=False).encode("utf-8"),
    file_name="credit_risk_lab_account_level_ecl.csv",
    mime="text/csv",
    use_container_width=True,
)


# ============================================================
# STAGE REVIEW
# ============================================================

st.markdown("### 14. Staging review and exceptions")

stage_review = work.loc[
    (work["stage"] > 1)
    | (
        work["sicr_assessment_status"]
        == "Origination lifetime PD unavailable"
    ),
    [
        "loan_id",
        "stage",
        "stage_reason",
        "sicr_assessment_status",
        "calculation_dpd",
        "calculation_origination_lifetime_pd",
        "calculation_lifetime_pd",
    ],
].copy()

st.write(
    "Review accounts assigned to Stage 2 or Stage 3, as well as "
    "accounts where the origination lifetime PD comparison is unavailable."
)

st.dataframe(
    stage_review.head(1000),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download staging review",
    data=stage_review.to_csv(index=False).encode("utf-8"),
    file_name="credit_risk_lab_staging_review.csv",
    mime="text/csv",
)


# ============================================================
# PORTFOLIO SEGMENTATION
# ============================================================

st.markdown("### 15. Portfolio segmentation")

segment_column = next(
    (
        column for column in [
            "product_type",
            "loan_purpose",
            "loan_term",
            "risk_grade",
            "credit_grade",
        ]
        if column in work.columns
    ),
    None,
)

if segment_column:

    segment_summary = (
        work.groupby(segment_column, dropna=False)
        .agg(
            Accounts=("loan_id", "count"),
            EAD=("calculation_ead", "sum"),
            ECL=("ECL", "sum"),
            Average_PD=("calculation_pd", "mean"),
            Average_LGD=("calculation_lgd", "mean"),
        )
        .reset_index()
    )

    segment_summary["ECL_Coverage"] = np.where(
        segment_summary["EAD"] > 0,
        segment_summary["ECL"] / segment_summary["EAD"],
        np.nan,
    )

    st.dataframe(
        segment_summary,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "No recognised product or risk segmentation column is available."
    )


# ============================================================
# MODEL AND CALCULATION GOVERNANCE
# ============================================================

st.markdown("### 16. Calculation governance")

governance_records = []

for model_type in model_types:

    try:
        active_id = registry.get_active_model_id(model_type)

        if active_id:
            metadata = registry.get_model_metadata(
                model_type,
                active_id,
            )

            governance_records.append({
                "Model Type": model_type,
                "Active Model ID": active_id,
                "Algorithm": metadata.get("algorithm"),
                "Version": metadata.get("version"),
                "Training Date": metadata.get("training_date"),
                "Training Observations": metadata.get(
                    "training_observations"
                ),
            })

        else:
            governance_records.append({
                "Model Type": model_type,
                "Active Model ID": "Not registered",
                "Algorithm": None,
                "Version": None,
                "Training Date": None,
                "Training Observations": None,
            })

    except ModelRegistryError as error:
        governance_records.append({
            "Model Type": model_type,
            "Active Model ID": str(error),
            "Algorithm": None,
            "Version": None,
            "Training Date": None,
            "Training Observations": None,
        })

st.dataframe(
    pd.DataFrame(governance_records),
    use_container_width=True,
    hide_index=True,
)

st.markdown("#### Calculation configuration")

configuration = pd.DataFrame(
    [
        {"Parameter": "Portfolio records", "Value": len(portfolio)},
        {"Parameter": "Lifetime horizon (years)", "Value": lifetime_horizon},
        {"Parameter": "SICR DPD threshold", "Value": sicr_dpd_threshold},
        {"Parameter": "Default DPD threshold", "Value": default_dpd_threshold},
        {"Parameter": "Lifetime PD multiplier", "Value": sicr_multiplier},
        {"Parameter": "Absolute lifetime PD increase", "Value": sicr_absolute_increase},
        {"Parameter": "PD input column", "Value": pd_column},
        {"Parameter": "EAD input column", "Value": ead_column},
        {"Parameter": "LGD input column", "Value": lgd_column},
        {"Parameter": "Origination lifetime PD column", "Value": origination_pd_column},
        {"Parameter": "Current lifetime PD column", "Value": current_lifetime_pd_column},
    ]
)

st.dataframe(
    configuration,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown("### 17. Portfolio interpretation")

stage_counts = work["stage"].value_counts()

stage1_count = int(stage_counts.get(1, 0))
stage2_count = int(stage_counts.get(2, 0))
stage3_count = int(stage_counts.get(3, 0))

st.write(
    f"The portfolio contains **{len(work):,} accounts**, of which "
    f"**{stage1_count:,}** are assigned to Stage 1, "
    f"**{stage2_count:,}** to Stage 2 and "
    f"**{stage3_count:,}** to Stage 3."
)

st.write(
    f"The calculated ECL is **{format_currency(total_ecl)}**, "
    f"against calculated EAD of **{format_currency(total_ead)}**. "
    f"The resulting ECL coverage ratio is "
    f"**{coverage_ratio:.2%}**."
)

if not has_origination_pd.all():
    st.warning(
        "Origination lifetime PD is missing for some accounts. "
        "A complete quantitative SICR comparison cannot be performed "
        "for those accounts. The staging results should therefore be "
        "reviewed before being treated as final."
    )

if stage3_count > 0:
    st.info(
        "Stage 3 ECL uses the simplified EAD × LGD approach in this "
        "version. A full Stage 3 implementation should estimate "
        "discounted expected contractual cash shortfalls using "
        "expected recoveries and the original effective interest rate."
    )


# ============================================================
# LIMITATIONS AND PRODUCTION CONTROLS
# ============================================================

st.markdown("### 18. Methodology and implementation controls")

with st.expander("Calculation methodology and limitations", expanded=False):

    st.markdown(
        """
        **PD term structure**

        Where a lifetime PD column is unavailable, the tool derives an
        illustrative lifetime PD from the 12-month PD using a constant
        conditional hazard assumption. It then derives annual marginal
        PDs from that cumulative lifetime PD. This is a modelling
        approximation, not a calibrated period-specific PD term structure.

        **SICR**

        The quantitative SICR assessment compares current lifetime PD
        with lifetime PD at initial recognition. The configured
        deterioration multiplier and absolute increase are illustrative
        controls and must be calibrated and approved under institutional
        policy. The 30 DPD presumption is also applied.

        **Default**

        The 90 DPD default presumption is applied unless an approved
        rebuttal process is implemented. Default indicators supplied
        in the portfolio are also recognised.

        **ECL**

        Stage 1 uses 12-month ECL. Stage 2 uses lifetime ECL. The
        calculation discounts period-level expected losses using the
        supplied EIR. Scenario weights are normalised before aggregation.

        **Stage 3**

        Stage 3 currently uses a simplified EAD × LGD estimate. A full
        implementation requires expected contractual cash flows,
        recovery timing, collection costs and discounting at the
        original effective interest rate.

        **Production requirements**

        Before institutional use, the framework requires approved
        portfolio data, origination risk information, calibrated PD
        term structures, forward-looking macroeconomic forecasts,
        scenario governance, approved EIR treatment, default and cure
        policies, model validation, reconciliations, audit trails and
        accounting sign-off.
        """
    )


st.caption(
    "CreditRisk Lab | Integrated Credit Risk Tool | "
    "Results are modelling outputs and require independent review "
    "before accounting or regulatory use."
)