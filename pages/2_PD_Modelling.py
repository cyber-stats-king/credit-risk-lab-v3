
import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="PD Modelling | CreditRisk Lab",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PAGE STYLING
# ============================================================

st.markdown("""
<style>

.pd-hero {
    background: linear-gradient(115deg, #102B50, #145A9E, #087EAD);
    border: 1px solid #258BD1;
    border-radius: 20px;
    padding: 35px;
    margin-bottom: 25px;
}

.pd-hero h1 {
    color: white !important;
    font-size: 38px;
}

.pd-hero p {
    color: #DCEEFF;
    font-size: 16px;
}

.concept-card {
    background: linear-gradient(145deg, #102744, #15365B);
    border: 1px solid #285A88;
    border-radius: 14px;
    padding: 20px;
    min-height: 150px;
}

.formula-card {
    background: #0C203B;
    border-left: 4px solid #299BE8;
    border-radius: 8px;
    padding: 18px;
    margin: 12px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="pd-hero">
    <h1>Probability of Default (PD)</h1>
    <p>
        Understand, develop and evaluate Probability of Default
        models, and explore their application in IFRS 9
        impairment measurement.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
Probability of Default is one of the fundamental components
of credit risk measurement. This section introduces the
theoretical foundations of PD, statistical modelling,
model validation and the application of PD under IFRS 9.
""")


# ============================================================
# PAGE NAVIGATION
# ============================================================

st.subheader("PD Learning Centre")

section = st.selectbox(
    "Select a topic",
    [
        "1. Understanding Probability of Default",
        "2. Types and Horizons of PD",
        "3. How PD Models Work",
        "4. Logistic Regression",
        "5. PD Model Development Methodology",
        "6. Model Performance and Validation",
        "7. PD Calibration",
        "8. IFRS 9: SICR and Credit Risk Staging",
        "9. Practical SICR Calculator"
    ]
)

st.divider()


# ============================================================
# SECTION 1: UNDERSTANDING PD
# ============================================================

if section == "1. Understanding Probability of Default":

    st.header("1. Understanding Probability of Default")

    st.write("""
    Probability of Default (PD) represents the likelihood that
    a borrower or counterparty will fail to meet its contractual
    credit obligations over a specified period.

    It is expressed as a probability between zero and one,
    or as a percentage.
    """)

    st.markdown("""
    <div class="formula-card">
        <b>Example</b><br><br>
        A borrower has an estimated PD of 5%. This means that,
        over the specified prediction horizon, the model estimates
        a 5% probability that the borrower will default.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("The three fundamental credit risk parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="concept-card">
            <h2 style="color:#59C3FF;">PD</h2>
            <h4>Probability of Default</h4>
            <p>The likelihood that the borrower defaults.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="concept-card">
            <h2 style="color:#59C3FF;">LGD</h2>
            <h4>Loss Given Default</h4>
            <p>The proportion of exposure lost after recoveries.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="concept-card">
            <h2 style="color:#59C3FF;">EAD</h2>
            <h4>Exposure at Default</h4>
            <p>The expected exposure when default occurs.</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Expected loss relationship")

    st.latex(r"EL = PD \times LGD \times EAD")

    st.write("""
    Expected Loss (EL) estimates the average loss that a
    financial institution expects from a credit exposure.

    For example, consider a loan with:
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("PD", "5%")
    col2.metric("LGD", "40%")
    col3.metric("EAD", "R100,000")

    expected_loss = 0.05 * 0.40 * 100000

    st.success(
        f"Expected Loss = R{expected_loss:,.2f}"
    )

    st.write("""
    The expected loss is R2,000.

    This does not mean the institution will necessarily lose
    R2,000 on that particular loan. It is an expected value
    based on the probability and severity of default.
    """)

    st.subheader("Why do financial institutions estimate PD?")

    st.markdown("""
    - Assess the creditworthiness of borrowers.
    - Support credit approval and lending decisions.
    - Determine risk-based pricing.
    - Monitor deterioration in portfolio credit quality.
    - Estimate expected credit losses.
    - Calculate regulatory credit risk capital.
    - Support portfolio management and stress testing.
    """)


# ============================================================
# SECTION 2: TYPES AND HORIZONS
# ============================================================

elif section == "2. Types and Horizons of PD":

    st.header("2. Types and Horizons of PD")

    st.write("""
    A PD estimate is incomplete without specifying its
    prediction horizon. The same borrower can have different
    PD estimates depending on whether the horizon is 12 months
    or the remaining lifetime of the loan.
    """)

    st.subheader("12-month Probability of Default")

    st.latex(
        r"PD_{12m} = P(\text{Default within the next 12 months})"
    )

    st.write("""
    The 12-month PD estimates the probability of a borrower
    defaulting within the next twelve months from the assessment
    date.

    It is commonly used in performing credit risk portfolios
    and is relevant to Stage 1 IFRS 9 impairment measurement.
    """)

    st.subheader("Lifetime Probability of Default")

    st.latex(
        r"PD_{Lifetime} = P(\text{Default over the remaining expected life})"
    )

    st.write("""
    Lifetime PD estimates the probability of default over the
    remaining expected life of the financial instrument.

    It captures the cumulative risk of default across all
    remaining periods.
    """)

    st.subheader("Marginal and cumulative PD")

    st.write("""
    These two concepts are important when building a lifetime
    PD term structure.
    """)

    st.markdown("**Marginal PD**")

    st.latex(
        r"MPD_t = P(T=t \mid T \geq t)"
    )

    st.write("""
    Marginal PD represents the probability of default during
    a particular period, conditional on the borrower having
    survived without default until the beginning of that period.
    """)

    st.markdown("**Cumulative PD**")

    st.latex(
        r"CPD_t = 1-\prod_{j=1}^{t}(1-MPD_j)"
    )

    st.write("""
    Cumulative PD represents the probability that default has
    occurred at any point up to and including period t.
    """)

    st.subheader("Illustrative lifetime PD term structure")

    term_structure = pd.DataFrame({
        "Year": [1, 2, 3, 4, 5],
        "Marginal PD": [0.05, 0.04, 0.035, 0.03, 0.025]
    })

    term_structure["Cumulative PD"] = (
        1 - (1 - term_structure["Marginal PD"]).cumprod()
    )

    st.dataframe(
        term_structure.style.format({
            "Marginal PD": "{:.2%}",
            "Cumulative PD": "{:.2%}"
        }),
        use_container_width=True
    )

    st.line_chart(
        term_structure.set_index("Year")[
            ["Marginal PD", "Cumulative PD"]
        ]
    )

    st.info("""
    Important: Lifetime PD is not simply 12-month PD multiplied
    by the number of years. It requires consideration of
    survival probabilities, changing risk over time and
    the expected life of the instrument.
    """)


# ============================================================
# SECTION 3: HOW PD MODELS WORK
# ============================================================

elif section == "3. How PD Models Work":

    st.header("3. How PD Models Work")

    st.write("""
    A statistical PD model estimates the relationship between
    borrower characteristics and the probability of default.

    The model uses historical credit observations to identify
    patterns associated with default.
    """)

    st.subheader("Typical model inputs")

    st.markdown("""
    **Borrower characteristics**
    - Age
    - Annual income
    - Employment history
    - Employment type

    **Credit bureau characteristics**
    - Credit score
    - Number of historical delinquencies
    - Credit enquiries
    - Public records
    - Credit history length

    **Loan characteristics**
    - Loan amount
    - Interest rate
    - Loan term
    - Loan purpose
    """)

    st.subheader("Target variable")

    st.write("""
    A supervised PD model requires a target variable indicating
    whether an observation defaulted during the defined
    performance window.
    """)

    st.code("""
    default = 1  # Default
    default = 0  # Non-default
    """, language="python")

    st.write("""
    The model learns how the input characteristics relate
    to the observed default outcome.
    """)

    st.subheader("General modelling process")

    st.markdown("""
    1. Collect historical loan data.
    2. Define the default event and observation window.
    3. Prepare and assess the explanatory variables.
    4. Transform variables where necessary.
    5. Estimate model parameters.
    6. Generate predicted probabilities.
    7. Evaluate model discrimination and calibration.
    8. Validate and monitor the model.
    """)

    st.warning("""
    A critical modelling consideration is data leakage.
    Variables that are only observed after default or after
    the prediction date must not be used as predictors in a
    model intended to predict future default.
    """)


# ============================================================
# SECTION 4: LOGISTIC REGRESSION
# ============================================================

elif section == "4. Logistic Regression":

    st.header("4. Logistic Regression")

    st.write("""
    Logistic regression is a widely used statistical method
    for estimating PD because the dependent variable is binary:
    default or non-default.
    """)

    st.subheader("The logistic function")

    st.latex(
        r"PD_i = \frac{1}{1+e^{-z_i}}"
    )

    st.write("""
    The logistic function transforms a linear combination
    of explanatory variables into a probability between
    zero and one.
    """)

    st.subheader("Linear predictor")

    st.latex(
        r"z_i = \beta_0 + \beta_1X_{1i} + \cdots + \beta_kX_{ki}"
    )

    st.write("""
    Where:

    - β0 is the intercept.
    - β1 to βk are estimated regression coefficients.
    - X1 to Xk represent borrower or loan characteristics.
    """)

    st.subheader("Log-odds representation")

    st.latex(
        r"\log\left(\frac{PD_i}{1-PD_i}\right)=X_i\beta"
    )

    st.write("""
    Logistic regression estimates the log-odds of default.
    The logistic transformation converts these log-odds
    into predicted probabilities.
    """)

    st.subheader("Interactive logistic function")

    x = st.slider(
        "Linear predictor (z)",
        min_value=-6.0,
        max_value=6.0,
        value=-1.0,
        step=0.1
    )

    probability = 1 / (1 + np.exp(-x))

    st.metric(
        "Estimated probability",
        f"{probability:.2%}"
    )

    st.progress(float(probability))

    st.write("""
    As the linear predictor increases, the estimated
    probability of default increases. The output remains
    bounded between zero and one.
    """)

    st.info("""
    Logistic regression estimates statistical relationships.
    The sign and magnitude of coefficients should be interpreted
    in the context of the variable transformations, reference
    categories and model specification.
    """)


# ============================================================
# SECTION 5: PD DEVELOPMENT METHODOLOGY
# ============================================================

elif section == "5. PD Model Development Methodology":

    st.header("5. PD Model Development Methodology")

    st.write("""
    CreditRisk Lab will use an interpretable PD modelling
    methodology based on optimal binning, Information Value,
    Weight of Evidence transformation and logistic regression.
    """)

    st.subheader("Step 1: Data preparation")

    st.write("""
    The modelling dataset must contain an appropriate default
    target and explanatory variables available at the prediction
    date.

    Data preparation includes:
    """)

    st.markdown("""
    - Checking missing observations.
    - Identifying duplicate records.
    - Reviewing variable types.
    - Assessing outliers.
    - Checking target distribution.
    - Identifying possible data leakage.
    """)

    st.subheader("Step 2: Exploratory Data Analysis")

    st.write("""
    EDA helps us understand the portfolio before estimating
    a model. We examine distributions, relationships with
    default, outliers and possible data quality issues.
    """)

    st.subheader("Step 3: Train/test split")

    st.write("""
    The data is divided into training and testing samples.
    The training sample is used for model estimation, while
    the test sample evaluates performance on observations
    not used to fit the model.
    """)

    st.latex(
        r"Training=80\%, \qquad Testing=20\%"
    )

    st.write("""
    We will use stratification to preserve the approximate
    default proportion in both samples. Out-of-time validation
    can subsequently be added to assess temporal stability.
    """)

    st.subheader("Step 4: Optimal binning")

    st.write("""
    Optimal binning groups continuous or categorical
    characteristics into meaningful categories that
    distinguish between default and non-default observations.

    The OptBinning package can identify suitable splits
    subject to configurable constraints.
    """)

    st.subheader("Step 5: Information Value")

    st.latex(
        r"IV = \sum_i (Dist_{NonDefault,i}-Dist_{Default,i})\times WOE_i"
    )

    st.write("""
    Information Value measures the discriminatory information
    contained in a characteristic with respect to the target.

    It is commonly used as an initial variable screening
    measure, not as proof that a variable should be included
    in the final model.
    """)

    st.subheader("Step 6: Weight of Evidence")

    st.latex(
        r"WOE_i = \ln\left(\frac{Dist_{NonDefault,i}}{Dist_{Default,i}}\right)"
    )

    st.write("""
    WOE transforms each bin into a numerical representation
    based on the relative distribution of defaults and
    non-defaults.

    This can improve interpretability and help represent
    nonlinear relationships.
    """)

    st.subheader("Step 7: Logistic regression")

    st.write("""
    The WOE-transformed variables are used as inputs to
    logistic regression. Coefficients are estimated from
    the training sample.
    """)

    st.subheader("Step 8: Model evaluation")

    st.write("""
    We evaluate the model using AUC, KS, Gini and calibration.
    Each metric measures a different aspect of model quality.
    """)

    st.subheader("Step 9: Calibration")

    st.write("""
    Calibration examines whether predicted PDs are consistent
    with observed default frequencies over the relevant
    calibration horizon.
    """)

    st.success("""
    This is the methodology we will implement in the
    interactive PD modelling engine.
    """)


# ============================================================
# SECTION 6: MODEL PERFORMANCE
# ============================================================

elif section == "6. Model Performance and Validation":

    st.header("6. Model Performance and Validation")

    st.write("""
    A PD model should be assessed using multiple performance
    measures. Discrimination and calibration are different
    dimensions of model performance.
    """)

    st.subheader("Area Under the ROC Curve (AUC)")

    st.latex(
        r"AUC = P(S_{default}>S_{nondefault})"
    )

    st.write("""
    AUC measures the model's ability to rank defaulted
    observations above non-defaulted observations.
    """)

    st.subheader("Gini coefficient")

    st.latex(
        r"Gini=2\times AUC-1"
    )

    st.write("""
    The Gini coefficient is a transformation of AUC
    frequently used in credit scoring.
    """)

    st.subheader("Kolmogorov-Smirnov statistic")

    st.latex(
        r"KS=\max_x|F_{default}(x)-F_{nondefault}(x)|"
    )

    st.write("""
    KS measures the maximum separation between the
    cumulative score or PD distributions of defaulted
    and non-defaulted observations.

    Its implementation must consistently define the
    score direction and the populations being compared.
    """)

    st.subheader("Calibration")

    st.write("""
    Calibration assesses whether predicted probabilities
    correspond to actual observed default frequencies.
    """)

    st.warning("""
    A high AUC does not necessarily imply good calibration.
    A model can rank borrowers correctly while systematically
    overestimating or underestimating default probabilities.
    """)


# ============================================================
# SECTION 7: PD CALIBRATION
# ============================================================

elif section == "7. PD Calibration":

    st.header("7. PD Calibration")

    st.write("""
    Calibration is the process of ensuring that predicted
    default probabilities are aligned with observed default
    experience over a defined horizon.
    """)

    st.subheader("Illustrative calibration example")

    calibration = pd.DataFrame({
        "PD band": [
            "0-2%", "2-4%", "4-6%", "6-8%", "8-10%"
        ],
        "Average predicted PD": [
            0.01, 0.03, 0.05, 0.07, 0.09
        ],
        "Observed default rate": [
            0.012, 0.028, 0.054, 0.075, 0.085
        ]
    })

    st.dataframe(
        calibration.style.format({
            "Average predicted PD": "{:.2%}",
            "Observed default rate": "{:.2%}"
        }),
        use_container_width=True
    )

    st.line_chart(
        calibration.set_index("PD band")[
            ["Average predicted PD", "Observed default rate"]
        ]
    )

    st.write("""
    If the observed default rate is consistently higher
    than the predicted PD, the model may be underestimating
    risk. If it is consistently lower, the model may be
    overestimating risk.

    Calibration should consider the appropriate portfolio,
    prediction horizon, economic conditions and observation
    window.
    """)


# ============================================================
# SECTION 8: IFRS 9, SICR AND STAGING
# ============================================================

elif section == "8. IFRS 9: SICR and Credit Risk Staging":

    st.header("8. IFRS 9: SICR and Credit Risk Staging")

    st.write("""
    IFRS 9 uses a forward-looking expected credit loss
    framework. Financial assets are classified into three
    impairment stages based on their credit risk condition
    since initial recognition.
    """)

    st.subheader("Understanding Significant Increase in Credit Risk")

    st.write("""
    Significant Increase in Credit Risk (SICR) is the
    assessment of whether the risk of default over the
    expected life of a financial instrument has increased
    significantly since initial recognition.

    The assessment compares credit risk at the reporting
    date with credit risk at initial recognition.
    """)

    st.warning("""
    SICR is not simply a comparison of the original 12-month
    PD with the current 12-month PD. The assessment generally
    considers the change in the risk of default over the
    expected life of the instrument.
    """)

    st.subheader("Relative change in lifetime PD")

    st.latex(
        r"Relative\ Increase = "
        r"\frac{PD_{current,lifetime}-PD_{origination,lifetime}}"
        r"{PD_{origination,lifetime}}"
    )

    st.write("""
    Consider an account with an origination lifetime PD of 4%
    and a current lifetime PD of 8%.
    """)

    st.latex(
        r"\frac{8\%-4\%}{4\%}=100\%"
    )

    st.write("""
    The lifetime PD has doubled, representing a 100% relative
    increase in the estimated probability of default.
    """)

    st.subheader("Absolute change in lifetime PD")

    st.latex(
        r"Absolute\ Increase=PD_{current}-PD_{origination}"
    )

    st.write("""
    In the same example, the absolute increase is 4 percentage
    points.

    An institution may use relative and absolute deterioration
    measures together as part of its approved SICR methodology.
    """)

    st.subheader("Other SICR indicators")

    st.markdown("""
    - Deterioration in the borrower's credit score.
    - Adverse changes in financial circumstances.
    - Restructuring due to financial difficulty.
    - Placement on a credit watchlist.
    - Adverse industry or economic developments.
    - Significant increases in days past due.
    """)

    st.subheader("The 30-day and 90-day presumptions")

    st.write("""
    IFRS 9 includes a rebuttable presumption that SICR has
    occurred when contractual payments are more than 30 days
    past due.

    There is also a rebuttable presumption that default does
    not occur later than when a financial asset is 90 days
    past due, unless reasonable and supportable information
    demonstrates that a more lagging default criterion is
    appropriate.
    """)

    st.subheader("The three impairment stages")

    stage1, stage2, stage3 = st.columns(3)

    with stage1:
        st.markdown("""
        <div class="concept-card">
            <h3 style="color:#34D399;">Stage 1</h3>
            <h4>12-month ECL</h4>
            <p>No significant increase in credit risk since
            initial recognition.</p>
        </div>
        """, unsafe_allow_html=True)

    with stage2:
        st.markdown("""
        <div class="concept-card">
            <h3 style="color:#FBBF24;">Stage 2</h3>
            <h4>Lifetime ECL</h4>
            <p>Significant increase in credit risk, but the
            asset is not credit-impaired.</p>
        </div>
        """, unsafe_allow_html=True)

    with stage3:
        st.markdown("""
        <div class="concept-card">
            <h3 style="color:#F87171;">Stage 3</h3>
            <h4>Lifetime ECL</h4>
            <p>The financial asset is credit-impaired.</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Important IFRS 9 considerations")

    st.markdown("""
    **Stage 1:** 12-month ECL is recognised. Interest revenue
    is generally calculated on the gross carrying amount.

    **Stage 2:** Lifetime ECL is recognised. Interest revenue
    is generally calculated on the gross carrying amount.

    **Stage 3:** Lifetime ECL is recognised. Interest revenue
    is generally calculated on the net amortised cost
    (gross carrying amount less loss allowance).

    The low-credit-risk exemption may also be applied at the
    reporting date when its IFRS 9 conditions are satisfied.
    """)

    st.info("""
    Staging is a subsequent credit risk assessment. Building
    a PD model does not automatically assign an account to
    Stage 1, Stage 2 or Stage 3.
    """)


# ============================================================
# SECTION 9: INTERACTIVE SICR CALCULATOR
# ============================================================

elif section == "9. Practical SICR Calculator":

    st.header("9. Practical SICR Calculator")

    st.write("""
    Explore how changes in lifetime PD and days past due
    affect an illustrative IFRS 9 staging assessment.
    """)

    st.caption("""
    This is an educational demonstration using configurable
    thresholds. It is not a universal IFRS 9 staging policy.
    """)

    col1, col2 = st.columns(2)

    with col1:

        origination_pd = st.number_input(
            "Origination lifetime PD (%)",
            min_value=0.01,
            max_value=100.0,
            value=4.0,
            step=0.5
        )

        current_pd = st.number_input(
            "Current lifetime PD (%)",
            min_value=0.0,
            max_value=100.0,
            value=8.0,
            step=0.5
        )

        dpd = st.number_input(
            "Days past due",
            min_value=0,
            max_value=365,
            value=0
        )

    with col2:

        relative_threshold = st.number_input(
            "Relative SICR threshold (%)",
            min_value=1.0,
            max_value=1000.0,
            value=100.0,
            step=5.0
        )

        absolute_threshold = st.number_input(
            "Absolute SICR threshold (percentage points)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.5
        )

        credit_impaired = st.checkbox(
            "Asset is identified as credit-impaired"
        )

    if origination_pd > 0:

        relative_increase = (
            (current_pd - origination_pd)
            / origination_pd
        ) * 100

    else:

        relative_increase = np.nan

    absolute_increase = current_pd - origination_pd

    relative_trigger = (
        relative_increase >= relative_threshold
    )

    absolute_trigger = (
        absolute_increase >= absolute_threshold
    )

    dpd_trigger = dpd > 30

    sicr_trigger = (
        relative_trigger
        or absolute_trigger
        or dpd_trigger
    )

    if credit_impaired or dpd >= 90:

        stage = 3
        stage_name = "Stage 3"
        explanation = (
            "The asset is treated as credit-impaired "
            "or has reached the illustrative 90-day "
            "default presumption."
        )

    elif sicr_trigger:

        stage = 2
        stage_name = "Stage 2"
        explanation = (
            "At least one illustrative SICR trigger "
            "has been activated."
        )

    else:

        stage = 1
        stage_name = "Stage 1"
        explanation = (
            "No illustrative SICR trigger has been activated."
        )

    st.divider()

    st.subheader("Calculation results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Relative PD increase",
        f"{relative_increase:.2f}%"
    )

    col2.metric(
        "Absolute PD increase",
        f"{absolute_increase:.2f} pp"
    )

    col3.metric(
        "Days past due",
        f"{dpd}"
    )

    st.subheader("Trigger assessment")

    results = pd.DataFrame({
        "Assessment": [
            "Relative PD deterioration",
            "Absolute PD deterioration",
            "More than 30 days past due",
            "Credit-impaired status"
        ],
        "Result": [
            "Triggered" if relative_trigger else "Not triggered",
            "Triggered" if absolute_trigger else "Not triggered",
            "Triggered" if dpd_trigger else "Not triggered",
            "Triggered" if credit_impaired else "Not triggered"
        ]
    })

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    if stage == 1:

        st.success(f"{stage_name}: {explanation}")

    elif stage == 2:

        st.warning(f"{stage_name}: {explanation}")

    else:

        st.error(f"{stage_name}: {explanation}")

    if stage == 1:

        st.write("Applicable ECL horizon: 12-month ECL.")

    else:

        st.write("Applicable ECL horizon: Lifetime ECL.")

    st.info("""
    This simplified calculator uses OR logic for its quantitative
    SICR triggers. A production staging engine must also assess
    qualitative indicators, low-credit-risk exemption eligibility,
    reasonable and supportable information, rebuttals and the
    institution's approved default and SICR policies.

    A 90-day past-due presumption is rebuttable. The calculator
    does not implement a full credit-impaired assessment.
    """)
