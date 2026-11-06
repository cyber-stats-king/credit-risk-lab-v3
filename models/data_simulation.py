
import numpy as np
import pandas as pd


def generate_synthetic_data(n=10000, random_state=42):

    np.random.seed(random_state)

    # --------------------------------------------------
    # 1. Customer and loan characteristics
    # --------------------------------------------------

    loan_id = np.arange(1, n + 1)

    age = np.random.randint(21, 70, n)

    annual_income = np.random.lognormal(
        mean=np.log(60000),
        sigma=0.45,
        size=n
    )

    employment_years = np.random.randint(0, 35, n)

    credit_score = np.clip(
        np.random.normal(690, 55, n),
        500,
        850
    )

    dti = np.clip(
        np.random.normal(28, 12, n),
        2,
        70
    )

    credit_utilization = np.random.beta(2.5, 4, n)

    num_open_accounts = np.random.poisson(5, n) + 1

    delinq_2yrs = np.random.poisson(0.5, n)

    inq_last_6mths = np.random.poisson(1.2, n)

    pub_rec = np.random.binomial(1, 0.05, n)

    total_accounts = np.random.randint(3, 40, n)

    # --------------------------------------------------
    # 2. Loan characteristics
    # --------------------------------------------------

    loan_amount = np.random.lognormal(
        mean=np.log(15000),
        sigma=0.55,
        size=n
    )

    loan_term = np.random.choice(
        [36, 60],
        size=n,
        p=[0.65, 0.35]
    )

    interest_rate = np.clip(
        np.random.normal(13, 4, n),
        5,
        30
    )

    loan_purpose = np.random.choice(
        [
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "major_purchase",
            "medical",
            "other"
        ],
        size=n
    )

    home_ownership = np.random.choice(
        ["RENT", "MORTGAGE", "OWN"],
        size=n
    )

    verification_status = np.random.choice(
        ["Verified", "Source Verified", "Not Verified"],
        size=n
    )

    employment_type = np.random.choice(
        ["Salaried", "Self-employed", "Contract", "Other"],
        size=n
    )

    # --------------------------------------------------
    # 3. Credit history dates
    # --------------------------------------------------

    issue_months = pd.date_range(
        start="2018-01-01",
        end="2022-12-01",
        freq="MS"
    )

    issue_d = np.random.choice(issue_months, size=n)

    history_years = np.random.randint(2, 30, n)

    earliest_cr_line = pd.to_datetime([
        issue_date - pd.DateOffset(years=int(years))
        for issue_date, years in zip(issue_d, history_years)
    ])

    credit_history_months = (
        (pd.Series(issue_d) - pd.Series(earliest_cr_line))
        .dt.days / 30.4375
    ).to_numpy()

    # --------------------------------------------------
    # 4. Probability of Default generation
    # --------------------------------------------------

    risk_score = (
        -0.025 * (credit_score - 700)
        + 0.045 * (dti - 25)
        + 1.8 * (credit_utilization - 0.45)
        + 0.55 * delinq_2yrs
        + 0.18 * inq_last_6mths
        + 0.90 * pub_rec
        + 0.025 * (interest_rate - 12)
        - 0.000004 * (annual_income - 60000)
        - 0.015 * employment_years
        - 0.0015 * credit_history_months
    )

    base_probability = 1 / (
        1 + np.exp(-(risk_score - 1.2))
    )

    pd_probability = np.clip(
        base_probability,
        0.01,
        0.75
    )

    default = np.random.binomial(
        1,
        pd_probability
    )

    loan_status = np.where(
        default == 1,
        "Default",
        "Fully Paid"
    )

    # --------------------------------------------------
    # 5. Exposure at Default generation
    # --------------------------------------------------

    credit_limit = loan_amount * np.random.uniform(
        1.1, 1.8, n
    )

    current_balance = np.minimum(
        loan_amount * np.random.uniform(0.15, 0.95, n),
        credit_limit
    )

    current_utilization = (
        current_balance / credit_limit
    )

    undrawn_amount = (
        credit_limit - current_balance
    )

    ccf = np.clip(
        0.30
        + 0.40 * current_utilization
        + np.random.normal(0, 0.08, n),
        0.05,
        0.95
    )

    ead = (
        current_balance + ccf * undrawn_amount
    )

    # --------------------------------------------------
    # 6. Collateral and recovery generation
    # --------------------------------------------------

    collateral_type = np.random.choice(
        ["None", "Vehicle", "Property", "Other"],
        size=n,
        p=[0.4, 0.3, 0.2, 0.1]
    )

    collateral_value = (
        loan_amount * np.random.uniform(0.5, 1.5, n)
    )

    collateral_value = np.where(
        collateral_type == "None",
        0,
        collateral_value
    )

    collateral_ratio = (
        collateral_value / np.maximum(ead, 1)
    )

    recovery_rate = np.clip(
        0.15
        + 0.35 * np.clip(collateral_ratio, 0, 1)
        + 0.05 * (credit_score - 650) / 200
        + np.random.normal(0, 0.08, n),
        0,
        0.95
    )

    lgd = 1 - recovery_rate

    # LGD is observed only for defaulted loans
    lgd = np.where(
        default == 1,
        lgd,
        np.nan
    )

    recovery_amount = np.where(
        default == 1,
        ead * recovery_rate,
        0
    )

    collection_cost = np.where(
        default == 1,
        ead * np.random.uniform(0.02, 0.08, n),
        0
    )

    adjusted_lgd = np.where(
        default == 1,
        np.clip(
            1 - (recovery_amount - collection_cost)
            / np.maximum(ead, 1),
            0,
            1
        ),
        np.nan
    )

    # --------------------------------------------------
    # 7. Assemble portfolio
    # --------------------------------------------------

    df = pd.DataFrame({

        "loan_id": loan_id,

        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "employment_type": employment_type,

        "credit_score": credit_score,
        "credit_history_months": credit_history_months,
        "earliest_cr_line": earliest_cr_line,

        "dti": dti,
        "credit_utilization": credit_utilization,
        "num_open_accounts": num_open_accounts,
        "total_accounts": total_accounts,
        "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": inq_last_6mths,
        "pub_rec": pub_rec,

        "home_ownership": home_ownership,
        "verification_status": verification_status,

        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "interest_rate": interest_rate,
        "loan_purpose": loan_purpose,
        "issue_d": issue_d,

        "default": default,
        "loan_status": loan_status,
        "pd_probability": pd_probability,

        "credit_limit": credit_limit,
        "current_balance": current_balance,
        "current_utilization": current_utilization,
        "undrawn_amount": undrawn_amount,
        "ccf": ccf,
        "ead": ead,

        "collateral_type": collateral_type,
        "collateral_value": collateral_value,
        "collateral_ratio": collateral_ratio,

        "recovery_rate": recovery_rate,
        "recovery_amount": recovery_amount,
        "collection_cost": collection_cost,
        "lgd": lgd,
        "adjusted_lgd": adjusted_lgd

    })

    return df
