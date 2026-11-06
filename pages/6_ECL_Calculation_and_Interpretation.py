import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# ECL CALCULATION AND INTERPRETATION
# Uses the original CreditRisk Lab portfolio.
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

st.title("ECL Calculation & Interpretation")

st.caption(
    "Combining Probability of Default, Loss Given Default and "
    "Exposure at Default to estimate Expected Credit Loss."
)


# ============================================================
# 1. LOAD ORIGINAL PORTFOLIO
# ============================================================

st.subheader("1. Original credit portfolio")

if "credit_data" not in st.session_state:

    st.warning(
        "No original portfolio is available. "
        "Generate the portfolio under Data & Models first."
    )

    st.stop()


source_data = st.session_state["credit_data"].copy()

if source_data.empty:

    st.error("The original portfolio is empty.")

    st.stop()


c1, c2, c3 = st.columns(3)

c1.metric(
    "Portfolio accounts",
    f"{len(source_data):,}",
)

c2.metric(
    "Portfolio variables",
    f"{source_data.shape[1]:,}",
)

if "default" in source_data.columns:

    default_series = pd.to_numeric(
        source_data["default"],
        errors="coerce",
    )

    c3.metric(
        "Observed default rate",
        f"{default_series.mean():.2%}",
    )

else:

    c3.metric(
        "Observed default rate",
        "Unavailable",
    )


with st.expander("Inspect original portfolio", expanded=False):

    st.dataframe(
        source_data.head(30),
        use_container_width=True,
    )


# ============================================================
# 2. UNDERSTANDING EXPECTED CREDIT LOSS
# ============================================================

st.subheader("2. Understanding Expected Credit Loss")

st.markdown(
    """
    Expected Credit Loss (ECL) is the probability-weighted estimate of
    credit losses over the relevant assessment horizon.

    The three core components are:

    - **PD:** Probability that a borrower defaults.
    - **LGD:** Proportion of exposure lost if default occurs.
    - **EAD:** Exposure outstanding when default occurs.

    For a simplified single-period calculation:

    \\[
    ECL = PD \\times LGD \\times EAD
    \\]

    For example, suppose a facility has:

    - PD = 5%
    - LGD = 40%
    - EAD = R100,000

    Then:

    \\[
    ECL = 0.05 \\times 0.40 \\times 100,000
    \\]

    \\[
    ECL = R2,000
    \\]

    The expected credit loss is R2,000.

    This does not mean that the bank will necessarily lose R2,000.
    It is the probability-weighted expected loss across possible outcomes.
    """
)


with st.expander("Work through the basic ECL calculation"):

    example_pd = st.slider(
        "Probability of Default (%)",
        0.0,
        100.0,
        5.0,
        0.5,
        key="example_pd",
    )

    example_lgd = st.slider(
        "Loss Given Default (%)",
        0.0,
        100.0,
        40.0,
        0.5,
        key="example_lgd",
    )

    example_ead = st.number_input(
        "Exposure at Default (R)",
        min_value=0.0,
        value=100000.0,
        step=5000.0,
        key="example_ead",
    )

    example_ecl = (
        example_pd / 100
        * example_lgd / 100
        * example_ead
    )

    st.metric(
        "Illustrative Expected Credit Loss",
        f"R {example_ecl:,.2f}",
    )


# ============================================================
# 3. SINGLE-PERIOD VS MULTI-PERIOD ECL
# ============================================================

st.subheader("3. Single-period and multi-period ECL")

st.markdown(
    """
    The single-period formula is useful for understanding the relationship
    between PD, LGD and EAD.

    However, IFRS 9 lifetime ECL requires a multi-period calculation.

    \\[
    ECL = \\sum_{t=1}^{T}
    DF_t \\times MPD_t \\times LGD_t \\times EAD_t
    \\]

    Where:

    - \(MPD_t\) is the marginal probability of default in period \(t\).
    - \(LGD_t\) is the loss severity if default occurs in period \(t\).
    - \(EAD_t\) is the exposure expected at default in period \(t\).
    - \(DF_t\) is the discount factor.
    - \(T\) is the relevant assessment horizon.
    """
)

st.info(
    "**Important:** Marginal PD is not the same as cumulative PD. "
    "Marginal PD measures the probability of default in a particular period, "
    "conditional on survival to the beginning of that period. "
    "Using cumulative PD in every period would double-count default risk."
)


# ============================================================
# 4. MARGINAL PD AND CUMULATIVE PD
# ============================================================

st.subheader("4. Understanding the PD term structure")

st.markdown(
    """
    Assume a borrower has a constant conditional annual PD of 5%.

    The probability of surviving the first year is 95%.

    The marginal probability of default in year 2 is therefore:

    \\[
    MPD_2 = 95\\% \\times 5\\% = 4.75\\%
    \\]

    The marginal probability of default in year 3 is:

    \\[
    MPD_3 = 95\\% \\times 95\\% \\times 5\\%
    \\]

    The probability of default in each year is calculated using the
    probability of surviving all previous years.
    """
)


term_pd = st.slider(
    "Illustrative conditional annual PD (%)",
    0.1,
    50.0,
    5.0,
    0.5,
    key="term_pd",
)

term_years = st.slider(
    "Lifetime horizon (years)",
    1,
    10,
    5,
    1,
    key="term_years",
)

annual_pd = term_pd / 100

term_rows = []

survival = 1.0
cumulative_pd = 0.0

for year in range(1, term_years + 1):

    marginal_pd = survival * annual_pd

    survival = survival * (1 - annual_pd)

    cumulative_pd += marginal_pd

    term_rows.append(
        {
            "Year": year,
            "Survival Probability": survival + marginal_pd,
            "Marginal PD": marginal_pd,
            "Cumulative PD": cumulative_pd,
        }
    )


term_table = pd.DataFrame(term_rows)

st.dataframe(
    term_table.style.format(
        {
            "Survival Probability": "{:.2%}",
            "Marginal PD": "{:.2%}",
            "Cumulative PD": "{:.2%}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)


fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(
    term_table["Year"],
    term_table["Marginal PD"],
    marker="o",
    label="Marginal PD",
)

ax.plot(
    term_table["Year"],
    term_table["Cumulative PD"],
    marker="o",
    label="Cumulative PD",
)

ax.set_title("Marginal and cumulative PD term structure")
ax.set_xlabel("Year")
ax.set_ylabel("Probability")
ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)
plt.close(fig)


# ============================================================
# 5. IFRS 9 STAGING
# ============================================================

st.subheader("5. IFRS 9 staging")

st.markdown(
    """
    IFRS 9 classifies financial assets into three stages.

    **Stage 1: Performing assets**

    - No significant increase in credit risk since initial recognition.
    - Recognise 12-month ECL.
    - Interest revenue is generally calculated on the gross carrying amount.

    **Stage 2: Significant increase in credit risk**

    - Credit risk has increased significantly since initial recognition.
    - Recognise lifetime ECL.
    - Interest revenue is generally calculated on the gross carrying amount.

    **Stage 3: Credit-impaired assets**

    - The asset is credit-impaired.
    - Recognise lifetime ECL.
    - Interest revenue is generally calculated on the net carrying amount,
      after the loss allowance.
    """
)


st.markdown("#### How staging affects the ECL horizon")

stage_comparison = pd.DataFrame(
    {
        "Stage": [
            "Stage 1",
            "Stage 2",
            "Stage 3",
        ],

        "Credit risk position": [
            "No significant increase in credit risk",
            "Significant increase in credit risk",
            "Credit-impaired",
        ],

        "ECL horizon": [
            "12-month ECL",
            "Lifetime ECL",
            "Lifetime ECL",
        ],

        "Interest calculation": [
            "Gross carrying amount",
            "Gross carrying amount",
            "Net carrying amount",
        ],
    }
)


st.dataframe(
    stage_comparison,
    hide_index=True,
    use_container_width=True,
)


st.warning(
    "12-month ECL does not mean that only cash shortfalls occurring "
    "during the next 12 months are recognised. It considers lifetime "
    "cash shortfalls resulting from defaults that are possible during "
    "the next 12 months."
)


# ============================================================
# 6. SELECT PD, LGD AND EAD INPUTS
# ============================================================

st.subheader("6. Select the model inputs")

st.markdown(
    """
    The next step is to combine the PD, LGD and EAD estimates.

    Select the columns that represent the relevant model outputs in your
    original portfolio. If a fitted model output is not stored in the
    portfolio, you can use an explicit illustrative assumption.
    """
)


def numeric_columns(data):

    return [
        c for c in data.columns
        if pd.api.types.is_numeric_dtype(data[c])
    ]


numeric_cols = numeric_columns(source_data)


def choose_column(label, candidates, key):

    options = ["Use an illustrative assumption"] + numeric_cols

    default_index = 0

    for candidate in candidates:

        if candidate in options:

            default_index = options.index(candidate)

            break

    return st.selectbox(
        label,
        options=options,
        index=default_index,
        key=key,
    )


pd_col = choose_column(
    "PD input column",
    [
        "predicted_pd",
        "pd",
        "pd_12m",
        "pd_probability",
    ],
    "ecl_pd_col",
)

lgd_col = choose_column(
    "LGD input column",
    [
        "predicted_lgd",
        "adjusted_lgd",
        "lgd",
    ],
    "ecl_lgd_col",
)

ead_col = choose_column(
    "EAD input column",
    [
        "predicted_ead",
        "ead",
        "current_balance",
        "loan_amount",
    ],
    "ecl_ead_col",
)


# ------------------------------------------------------------
# Illustrative assumptions
# ------------------------------------------------------------

assumption_left, assumption_middle, assumption_right = st.columns(3)


with assumption_left:

    fallback_pd = st.number_input(
        "Illustrative annual PD (%)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.5,
    )


with assumption_middle:

    fallback_lgd = st.number_input(
        "Illustrative LGD (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.5,
    )


with assumption_right:

    fallback_ead = st.number_input(
        "Illustrative EAD (R)",
        min_value=0.0,
        value=100000.0,
        step=5000.0,
    )


# ============================================================
# 7. BUILD THE ECL WORKING DATASET
# ============================================================

st.subheader("7. Prepare the ECL calculation dataset")


ecl_data = source_data.copy()


# ------------------------------------------------------------
# PD
# ------------------------------------------------------------

if pd_col == "Use an illustrative assumption":

    ecl_data["_ecl_pd"] = fallback_pd / 100

    pd_source_description = (
        f"Illustrative annual PD assumption: {fallback_pd:.2f}%"
    )

else:

    ecl_data["_ecl_pd"] = pd.to_numeric(
        ecl_data[pd_col],
        errors="coerce",
    )

    # If the chosen field is a probability percentage,
    # convert values greater than 1 to proportions.

    if ecl_data["_ecl_pd"].dropna().median() > 1:

        ecl_data["_ecl_pd"] = (
            ecl_data["_ecl_pd"] / 100
        )

    pd_source_description = (
        f"Selected portfolio variable: {pd_col}"
    )


# ------------------------------------------------------------
# LGD
# ------------------------------------------------------------

if lgd_col == "Use an illustrative assumption":

    ecl_data["_ecl_lgd"] = fallback_lgd / 100

    lgd_source_description = (
        f"Illustrative LGD assumption: {fallback_lgd:.2f}%"
    )

else:

    ecl_data["_ecl_lgd"] = pd.to_numeric(
        ecl_data[lgd_col],
        errors="coerce",
    )

    if ecl_data["_ecl_lgd"].dropna().median() > 1:

        ecl_data["_ecl_lgd"] = (
            ecl_data["_ecl_lgd"] / 100
        )

    lgd_source_description = (
        f"Selected portfolio variable: {lgd_col}"
    )


# ------------------------------------------------------------
# EAD
# ------------------------------------------------------------

if ead_col == "Use an illustrative assumption":

    ecl_data["_ecl_ead"] = fallback_ead

    ead_source_description = (
        f"Illustrative EAD assumption: R{fallback_ead:,.2f}"
    )

else:

    ecl_data["_ecl_ead"] = pd.to_numeric(
        ecl_data[ead_col],
        errors="coerce",
    )

    ead_source_description = (
        f"Selected portfolio variable: {ead_col}"
    )


# ------------------------------------------------------------
# Validate inputs
# ------------------------------------------------------------

ecl_data["_ecl_pd"] = ecl_data["_ecl_pd"].clip(0, 1)

ecl_data["_ecl_lgd"] = ecl_data["_ecl_lgd"].clip(0, 1)

ecl_data["_ecl_ead"] = ecl_data["_ecl_ead"].clip(lower=0)


ecl_data = ecl_data.replace(
    [np.inf, -np.inf],
    np.nan,
)


valid_inputs = (
    ecl_data["_ecl_pd"].notna()
    & ecl_data["_ecl_lgd"].notna()
    & ecl_data["_ecl_ead"].notna()
)


st.write("**Selected input sources**")

st.write(
    {
        "PD": pd_source_description,
        "LGD": lgd_source_description,
        "EAD": ead_source_description,
    }
)


st.metric(
    "Accounts with valid ECL inputs",
    f"{int(valid_inputs.sum()):,}",
)


if valid_inputs.sum() == 0:

    st.error(
        "No records have valid PD, LGD and EAD inputs."
    )

    st.stop()


# ============================================================
# 8. STAGING ASSIGNMENT
# ============================================================

st.subheader("8. Assign IFRS 9 stages")


st.markdown(
    """
    Staging determines whether an account receives 12-month or lifetime ECL.

    In a production IFRS 9 framework, Stage 2 requires an assessment of
    significant increase in credit risk since initial recognition.

    A comparison of current lifetime PD against lifetime PD at initial
    recognition is one important approach. Days past due and other
    qualitative indicators may also be considered.
    """
)


# ------------------------------------------------------------
# Available staging variables
# ------------------------------------------------------------

stage_candidates = [
    c for c in source_data.columns
    if c.lower() in [
        "stage",
        "ifrs9_stage",
        "ifrs_stage",
    ]
]


dpd_candidates = [
    c for c in source_data.columns
    if c.lower() in [
        "dpd",
        "days_past_due",
        "days_past_due_current",
    ]
]


default_candidates = [
    c for c in source_data.columns
    if c.lower() == "default"
]


origination_pd_candidates = [
    c for c in source_data.columns
    if c.lower() in [
        "origination_lifetime_pd",
        "initial_lifetime_pd",
        "origination_pd_lifetime",
    ]
]


current_lifetime_pd_candidates = [
    c for c in source_data.columns
    if c.lower() in [
        "current_lifetime_pd",
        "lifetime_pd",
        "current_pd_lifetime",
    ]
]


stage_source = st.selectbox(
    "Staging method",
    [
        "Use existing stage column",
        "Use default and SICR indicators",
        "Manually assign a stage to the whole portfolio",
    ],
)


# ------------------------------------------------------------
# Existing stage column
# ------------------------------------------------------------

if stage_source == "Use existing stage column":

    if not stage_candidates:

        st.error(
            "No recognised stage column was found. "
            "Select another staging method."
        )

        st.stop()


    selected_stage_col = st.selectbox(
        "Existing stage variable",
        stage_candidates,
    )


    def normalize_stage(value):

        if pd.isna(value):
            return np.nan

        value_text = str(value).strip().lower()

        if value_text in [
            "1",
            "stage 1",
            "stage1",
            "s1",
        ]:
            return 1

        if value_text in [
            "2",
            "stage 2",
            "stage2",
            "s2",
        ]:
            return 2

        if value_text in [
            "3",
            "stage 3",
            "stage3",
            "s3",
        ]:
            return 3

        try:

            numeric_value = int(float(value))

            if numeric_value in [1, 2, 3]:
                return numeric_value

        except Exception:

            pass

        return np.nan


    ecl_data["_ecl_stage"] = (
        ecl_data[selected_stage_col]
        .apply(normalize_stage)
    )


# ------------------------------------------------------------
# Default and SICR indicators
# ------------------------------------------------------------

elif stage_source == "Use default and SICR indicators":

    if not default_candidates:

        st.error(
            "A default indicator is required for this staging method."
        )

        st.stop()


    selected_default_col = st.selectbox(
        "Default indicator",
        default_candidates,
    )


    default_indicator = pd.to_numeric(
        ecl_data[selected_default_col],
        errors="coerce",
    )


    # Default accounts are assigned to Stage 3.

    ecl_data["_ecl_stage"] = 1

    ecl_data.loc[
        default_indicator == 1,
        "_ecl_stage",
    ] = 3


    # Optional DPD assessment.

    if dpd_candidates:

        selected_dpd_col = st.selectbox(
            "Days past due variable",
            ["Do not use DPD"] + dpd_candidates,
        )

        if selected_dpd_col != "Do not use DPD":

            dpd_values = pd.to_numeric(
                ecl_data[selected_dpd_col],
                errors="coerce",
            )

            ecl_data.loc[
                (dpd_values >= 30)
                & (default_indicator != 1),
                "_ecl_stage",
            ] = 2


    # Optional quantitative SICR comparison.

    if (
        origination_pd_candidates
        and current_lifetime_pd_candidates
    ):

        use_sicr = st.checkbox(
            "Use lifetime PD comparison for SICR",
            value=True,
        )

        if use_sicr:

            origination_col = st.selectbox(
                "Origination lifetime PD",
                origination_pd_candidates,
            )

            current_lifetime_col = st.selectbox(
                "Current lifetime PD",
                current_lifetime_pd_candidates,
            )

            sicr_multiplier = st.number_input(
                "SICR relative PD threshold",
                min_value=1.0,
                max_value=10.0,
                value=2.0,
                step=0.25,
            )


            origination_values = pd.to_numeric(
                ecl_data[origination_col],
                errors="coerce",
            )

            current_values = pd.to_numeric(
                ecl_data[current_lifetime_col],
                errors="coerce",
            )


            sicr_condition = (
                (origination_values > 0)
                & (
                    current_values
                    >= origination_values * sicr_multiplier
                )
            )


            ecl_data.loc[
                sicr_condition
                & (default_indicator != 1),
                "_ecl_stage",
            ] = 2


            st.caption(
                "The selected relative PD threshold is an illustrative "
                "rule, not a universal IFRS 9 SICR threshold. "
                "A production policy requires calibrated quantitative "
                "and qualitative SICR criteria."
            )


# ------------------------------------------------------------
# Manual stage
# ------------------------------------------------------------

else:

    manual_stage = st.selectbox(
        "Assign stage",
        [
            1,
            2,
            3,
        ],
        format_func=lambda x: f"Stage {x}",
    )


    ecl_data["_ecl_stage"] = manual_stage


# ------------------------------------------------------------
# Validate stage assignment
# ------------------------------------------------------------

ecl_data.loc[
    ~ecl_data["_ecl_stage"].isin([1, 2, 3]),
    "_ecl_stage",
] = np.nan


valid_staging = ecl_data["_ecl_stage"].isin([1, 2, 3])


ecl_data = ecl_data.loc[
    valid_inputs & valid_staging
].copy()


if ecl_data.empty:

    st.error(
        "No accounts have valid inputs and a valid stage assignment."
    )

    st.stop()


stage_counts = (
    ecl_data["_ecl_stage"]
    .value_counts()
    .reindex([1, 2, 3], fill_value=0)
)


stage_summary = pd.DataFrame(
    {
        "Stage": [
            "Stage 1",
            "Stage 2",
            "Stage 3",
        ],

        "Accounts": [
            int(stage_counts.loc[1]),
            int(stage_counts.loc[2]),
            int(stage_counts.loc[3]),
        ],
    }
)


st.dataframe(
    stage_summary,
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# 9. ECL CALCULATION ASSUMPTIONS
# ============================================================

st.subheader("9. ECL calculation assumptions")


st.markdown(
    """
    The calculation below uses a simplified constant conditional annual PD
    assumption to construct the lifetime PD term structure.

    This is an educational approximation. In a production model, PD,
    LGD and EAD may vary by period, and the calculation should use
    appropriately calibrated forward-looking estimates.
    """
)


horizon = st.slider(
    "Lifetime ECL horizon (years)",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)


eir = st.number_input(
    "Effective interest rate (annual %)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=0.5,
)


eir_decimal = eir / 100


# ============================================================
# 10. CALCULATE PERIOD-BY-PERIOD ECL
# ============================================================

st.subheader("10. Calculate Expected Credit Loss")


st.markdown(
    """
    The calculation applies different default horizons by stage:

    - Stage 1: Default risk over the next 12 months.
    - Stage 2: Default risk over the remaining lifetime.
    - Stage 3: Credit-impaired exposure, using a simplified immediate
      loss estimate of EAD multiplied by LGD.

    For Stage 1 and Stage 2, the conditional annual PD is converted
    into marginal PD using survival probability.
    """
)


# ------------------------------------------------------------
# Annual marginal PD function
# ------------------------------------------------------------

def calculate_marginal_pds(conditional_pd, number_of_years):

    survival_probability = 1.0

    marginal_probabilities = []

    for year in range(1, number_of_years + 1):

        marginal_pd = (
            survival_probability
            * conditional_pd
        )

        marginal_probabilities.append(
            marginal_pd
        )

        survival_probability *= (
            1 - conditional_pd
        )

    return marginal_probabilities


# ------------------------------------------------------------
# Calculate ECL by account
# ------------------------------------------------------------

account_results = []

period_results = []


for idx, row in ecl_data.iterrows():

    pd_value = float(row["_ecl_pd"])

    lgd_value = float(row["_ecl_lgd"])

    ead_value = float(row["_ecl_ead"])

    stage = int(row["_ecl_stage"])


    account_ecl = 0.0


    if stage == 1:

        # Stage 1: 12-month ECL.
        # The first-year marginal PD is the annual conditional PD.

        marginal_pds = calculate_marginal_pds(
            pd_value,
            1,
        )

        applicable_years = [1]


    elif stage == 2:

        # Stage 2: Lifetime ECL.

        marginal_pds = calculate_marginal_pds(
            pd_value,
            horizon,
        )

        applicable_years = list(
            range(1, horizon + 1)
        )


    else:

        # Stage 3: Simplified immediate credit-impaired loss.
        # A full Stage 3 calculation should use expected cash shortfalls.

        stage3_loss = ead_value * lgd_value

        account_results.append(
            {
                "Account_Index": idx,
                "Stage": stage,
                "PD": pd_value,
                "LGD": lgd_value,
                "EAD": ead_value,
                "ECL": stage3_loss,
            }
        )

        period_results.append(
            {
                "Account_Index": idx,
                "Stage": stage,
                "Year": 0,
                "Marginal_PD": 1.0,
                "LGD": lgd_value,
                "EAD": ead_value,
                "Discount_Factor": 1.0,
                "ECL_Contribution": stage3_loss,
            }
        )

        continue


    for year, marginal_pd in zip(
        applicable_years,
        marginal_pds,
    ):

        discount_factor = (
            1 / ((1 + eir_decimal) ** year)
        )

        period_ecl = (
            marginal_pd
            * lgd_value
            * ead_value
            * discount_factor
        )

        account_ecl += period_ecl


        period_results.append(
            {
                "Account_Index": idx,
                "Stage": stage,
                "Year": year,
                "Marginal_PD": marginal_pd,
                "LGD": lgd_value,
                "EAD": ead_value,
                "Discount_Factor": discount_factor,
                "ECL_Contribution": period_ecl,
            }
        )


    account_results.append(
        {
            "Account_Index": idx,
            "Stage": stage,
            "PD": pd_value,
            "LGD": lgd_value,
            "EAD": ead_value,
            "ECL": account_ecl,
        }
    )


ecl_results = pd.DataFrame(
    account_results
).set_index(
    "Account_Index"
)


period_results_df = pd.DataFrame(
    period_results
)


# ------------------------------------------------------------
# Attach ECL to portfolio
# ------------------------------------------------------------

ecl_portfolio = ecl_data.copy()

ecl_portfolio["calculated_ecl"] = (
    ecl_results["ECL"]
)


ecl_portfolio["ecl_stage"] = (
    ecl_results["Stage"]
)


ecl_portfolio["ecl_pd"] = (
    ecl_results["PD"]
)


ecl_portfolio["ecl_lgd"] = (
    ecl_results["LGD"]
)


ecl_portfolio["ecl_ead"] = (
    ecl_results["EAD"]
)


# ============================================================
# 11. PORTFOLIO ECL SUMMARY
# ============================================================

st.subheader("11. Portfolio Expected Credit Loss")


total_ecl = ecl_portfolio[
    "calculated_ecl"
].sum()


total_ead = ecl_portfolio[
    "ecl_ead"
].sum()


weighted_ecl_rate = (
    total_ecl / total_ead
    if total_ead > 0
    else 0
)


k1, k2, k3 = st.columns(3)


k1.metric(
    "Total portfolio EAD",
    f"R {total_ead:,.2f}",
)


k2.metric(
    "Total portfolio ECL",
    f"R {total_ecl:,.2f}",
)


k3.metric(
    "ECL / EAD",
    f"{weighted_ecl_rate:.2%}",
)


# ------------------------------------------------------------
# Stage-level summary
# ------------------------------------------------------------

stage_ecl_summary = (
    ecl_portfolio
    .groupby(
        "ecl_stage",
        as_index=False,
    )
    .agg(
        Accounts=(
            "calculated_ecl",
            "size",
        ),

        Total_EAD=(
            "ecl_ead",
            "sum",
        ),

        Total_ECL=(
            "calculated_ecl",
            "sum",
        ),

        Average_PD=(
            "ecl_pd",
            "mean",
        ),

        Average_LGD=(
            "ecl_lgd",
            "mean",
        ),
    )
)


stage_ecl_summary["ECL_to_EAD"] = (
    stage_ecl_summary["Total_ECL"]
    / stage_ecl_summary["Total_EAD"].replace(
        0,
        np.nan,
    )
)


stage_ecl_summary["Stage"] = (
    "Stage "
    + stage_ecl_summary["ecl_stage"].astype(int).astype(str)
)


st.markdown("#### ECL by stage")


st.dataframe(
    stage_ecl_summary[
        [
            "Stage",
            "Accounts",
            "Total_EAD",
            "Total_ECL",
            "Average_PD",
            "Average_LGD",
            "ECL_to_EAD",
        ]
    ].style.format(
        {
            "Total_EAD": "R {:,.2f}",
            "Total_ECL": "R {:,.2f}",
            "Average_PD": "{:.2%}",
            "Average_LGD": "{:.2%}",
            "ECL_to_EAD": "{:.2%}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)


# ------------------------------------------------------------
# Stage ECL chart
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))


ax.bar(
    stage_ecl_summary["Stage"],
    stage_ecl_summary["Total_ECL"],
)


ax.set_title(
    "Expected Credit Loss by IFRS 9 stage"
)

ax.set_xlabel(
    "IFRS 9 stage"
)

ax.set_ylabel(
    "Expected Credit Loss (R)"
)

ax.grid(
    axis="y",
    alpha=0.2,
)


st.pyplot(fig)

plt.close(fig)


# ============================================================
# 12. ACCOUNT-LEVEL ECL RESULTS
# ============================================================

st.subheader("12. Account-level ECL results")


display_columns = [
    "ecl_stage",
    "ecl_pd",
    "ecl_lgd",
    "ecl_ead",
    "calculated_ecl",
]


st.dataframe(
    ecl_portfolio[display_columns].head(100),
    use_container_width=True,
)


st.download_button(
    "Download account-level ECL results",

    data=ecl_portfolio.to_csv(
        index=False
    ).encode("utf-8"),

    file_name="credit_risk_ecl_results.csv",

    mime="text/csv",
)


# ============================================================
# 13. PERIOD-BY-PERIOD ECL BREAKDOWN
# ============================================================

st.subheader("13. Period-by-period ECL breakdown")


selected_account = st.selectbox(
    "Select an account index to inspect",

    options=ecl_portfolio.index.tolist(),

    format_func=lambda x: str(x),
)


account_periods = period_results_df.loc[
    period_results_df["Account_Index"] == selected_account
].copy()


if not account_periods.empty:

    st.markdown(
        """
        This table shows how an account's ECL is accumulated
        across the relevant assessment periods.
        """
    )


    st.dataframe(
        account_periods[
            [
                "Stage",
                "Year",
                "Marginal_PD",
                "LGD",
                "EAD",
                "Discount_Factor",
                "ECL_Contribution",
            ]
        ].style.format(
            {
                "Marginal_PD": "{:.4%}",
                "LGD": "{:.2%}",
                "EAD": "R {:,.2f}",
                "Discount_Factor": "{:.4f}",
                "ECL_Contribution": "R {:,.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


    selected_account_ecl = ecl_portfolio.loc[
        selected_account,
        "calculated_ecl",
    ]


    st.metric(
        "Selected account ECL",
        f"R {selected_account_ecl:,.2f}",
    )


# ============================================================
# 14. INTERPRETATION OF ECL
# ============================================================

st.subheader("14. Interpreting the ECL results")


st.markdown(
    """
    ECL is a monetary estimate of expected credit losses. Its value
    is affected by the probability of default, the loss severity,
    the exposure at default and the discount factor.

    **How to interpret the portfolio results:**

    - A higher PD increases the likelihood of a credit loss.
    - A higher LGD increases the expected loss if default occurs.
    - A higher EAD increases the amount exposed to potential loss.
    - A longer lifetime horizon generally captures additional default risk.
    - Discounting reduces the present value of future expected cash shortfalls.
    - Moving an account from Stage 1 to Stage 2 can increase its allowance
      because the assessment changes from 12-month to lifetime ECL.
    """
)


# ------------------------------------------------------------
# Contribution analysis
# ------------------------------------------------------------

ecl_contributions = pd.DataFrame(
    {
        "Risk Component": [
            "PD",
            "LGD",
            "EAD",
        ],

        "Interpretation": [
            "Likelihood of default",
            "Severity of loss after default",
            "Exposure outstanding at default",
        ],
    }
)


st.dataframe(
    ecl_contributions,
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# 15. SENSITIVITY ANALYSIS
# ============================================================

st.subheader("15. ECL sensitivity analysis")


st.markdown(
    """
    Sensitivity analysis helps us understand how changes in the
    underlying risk parameters affect the expected credit loss.

    The following exercise applies percentage changes to the
    selected portfolio PD, LGD and EAD inputs.
    """
)


sens_pd = st.slider(
    "Change in PD (%)",
    min_value=-50,
    max_value=100,
    value=0,
    step=5,
)


sens_lgd = st.slider(
    "Change in LGD (%)",
    min_value=-50,
    max_value=100,
    value=0,
    step=5,
)


sens_ead = st.slider(
    "Change in EAD (%)",
    min_value=-50,
    max_value=100,
    value=0,
    step=5,
)


sensitivity_portfolio = ecl_portfolio.copy()


sensitivity_portfolio["_sensitivity_pd"] = (
    sensitivity_portfolio["ecl_pd"]
    * (1 + sens_pd / 100)
).clip(0, 1)


sensitivity_portfolio["_sensitivity_lgd"] = (
    sensitivity_portfolio["ecl_lgd"]
    * (1 + sens_lgd / 100)
).clip(0, 1)


sensitivity_portfolio["_sensitivity_ead"] = (
    sensitivity_portfolio["ecl_ead"]
    * (1 + sens_ead / 100)
).clip(lower=0)


sensitivity_ecl_total = 0.0


for idx, row in sensitivity_portfolio.iterrows():

    stage = int(row["ecl_stage"])

    pd_value = row["_sensitivity_pd"]

    lgd_value = row["_sensitivity_lgd"]

    ead_value = row["_sensitivity_ead"]


    if stage == 3:

        sensitivity_ecl = (
            ead_value * lgd_value
        )

    else:

        years = 1 if stage == 1 else horizon

        marginal_pds = calculate_marginal_pds(
            pd_value,
            years,
        )

        sensitivity_ecl = 0.0

        for year, marginal_pd in enumerate(
            marginal_pds,
            start=1,
        ):

            discount_factor = (
                1 / ((1 + eir_decimal) ** year)
            )

            sensitivity_ecl += (
                marginal_pd
                * lgd_value
                * ead_value
                * discount_factor
            )


    sensitivity_ecl_total += sensitivity_ecl


sensitivity_change = (
    sensitivity_ecl_total - total_ecl
)


sensitivity_change_pct = (
    sensitivity_change / total_ecl * 100
    if total_ecl != 0
    else 0
)


s1, s2, s3 = st.columns(3)


s1.metric(
    "Original ECL",
    f"R {total_ecl:,.2f}",
)


s2.metric(
    "Scenario ECL",
    f"R {sensitivity_ecl_total:,.2f}",
)


s3.metric(
    "Change in ECL",
    f"{sensitivity_change_pct:+.2f}%",
)


# ============================================================
# 16. MODEL ASSUMPTIONS AND LIMITATIONS
# ============================================================

st.subheader("16. Assumptions and limitations")


st.markdown(
    """
    This module is designed to demonstrate the mechanics of ECL calculation.

    It uses the following simplified assumptions:

    - A constant conditional annual PD is used to construct the PD term structure.
    - LGD and EAD are held constant across the projection horizon.
    - A single effective interest rate is used for discounting.
    - Stage 1 is calculated using the first year's marginal PD.
    - Stage 2 is calculated using marginal PDs across the selected lifetime horizon.
    - Stage 3 uses a simplified immediate EAD multiplied by LGD calculation.
    - The staging rules are illustrative and are not a complete IFRS 9 SICR policy.

    A production IFRS 9 ECL framework should additionally consider:

    - Multiple forward-looking macroeconomic scenarios.
    - Probability-weighted scenario outcomes.
    - Scenario-specific PD, LGD and EAD estimates.
    - Lifetime PD term structures.
    - Contractual cash flows and expected prepayments.
    - Effective interest rate discounting.
    - Expected future drawings for revolving facilities.
    - Appropriate Stage 3 cash-shortfall estimation.
    - Qualitative SICR indicators and rebuttable presumptions.
    - Model validation, governance and monitoring.
    """
)


st.warning(
    "This is an educational ECL engine, not a complete IFRS 9 production "
    "implementation. In particular, Stage 3 requires a more detailed "
    "expected cash-shortfall calculation, and the constant-PD approach "
    "is not a substitute for calibrated lifetime PD term structures."
)


# ============================================================
# 17. EXPORT ECL RESULTS
# ============================================================

st.subheader("17. Export ECL results")


export_columns = [
    "ecl_stage",
    "ecl_pd",
    "ecl_lgd",
    "ecl_ead",
    "calculated_ecl",
]


export_data = ecl_portfolio[
    export_columns
].copy()


st.download_button(
    "Download ECL results",

    data=export_data.to_csv(
        index=True
    ).encode("utf-8"),

    file_name="credit_risk_ecl_summary.csv",

    mime="text/csv",
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CreditRisk Lab | ECL Calculation & Interpretation | "
    "Educational IFRS 9 demonstration"
)