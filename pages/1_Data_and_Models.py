import json
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CREDITRISK LAB | DATA & MODELS
# Governance-focused ingestion, canonical mapping and profiling
# ============================================================

st.set_page_config(
    page_title="Data & Models | CreditRisk Lab",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
PORTFOLIO_DIR = BASE_DIR / "data" / "portfolios"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ALIAS_FILE = BASE_DIR / "data" / "variable_aliases.json"
CUSTOM_DICTIONARY_FILE = BASE_DIR / "data" / "custom_variable_dictionary.json"

PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
ALIAS_FILE.parent.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
:root {
  --navy:#081321; --panel:#102238; --panel2:#142b43;
  --line:#29445e; --ink:#eaf2fb; --muted:#a9bfd2;
  --teal:#42e3d0; --gold:#e4bd78; --red:#ff7777;
}
[data-testid="stAppViewContainer"] {
  background:radial-gradient(ellipse at 10% 0%,#17344b 0%,transparent 35%),
  linear-gradient(135deg,#07111f,#0b192a 55%,#10243a);
  color:var(--ink);
}
[data-testid="stSidebar"] {
  background:#081321;
  border-right:1px solid var(--line);
}
.block-container {
  max-width:1580px;
  padding-top:1.6rem;
  padding-bottom:3rem;
}
h1,h2,h3 {
  color:#f3f7fc!important;
  letter-spacing:-.025em;
}
[data-testid="stMetric"] {
  background:linear-gradient(145deg,#183047,#0c1c2d);
  border:1px solid #34536c;
  padding:15px;
  border-radius:15px;
}
[data-testid="stMetricLabel"] {
  color:#a9bfd2!important;
}
[data-testid="stMetricValue"] {
  color:#f5fbff!important;
}
div[data-testid="stDataFrame"] {
  border:1px solid #29445e;
  border-radius:10px;
}
.cr-hero {
  padding:30px 34px;
  border-radius:22px;
  margin-bottom:22px;
  background:linear-gradient(115deg,#10243a,#122e45 60%,#10283c);
  border:1px solid #3a627b;
  box-shadow:0 16px 38px #0004;
}
.cr-kicker {
  color:#42e3d0;
  font-size:12px;
  font-weight:800;
  letter-spacing:2px;
}
.cr-title {
  font-size:34px;
  font-weight:800;
  color:#f5fbff;
  margin-top:6px;
}
.cr-sub {
  color:#c5d4e2;
  margin-top:7px;
  font-size:15px;
}
.status-good {
  color:#42e3d0;
  font-weight:700;
}
.status-warn {
  color:#e4bd78;
  font-weight:700;
}
.status-bad {
  color:#ff7777;
  font-weight:700;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="cr-hero">
  <div class="cr-title" style="text-align: center;">IMPORTANT NOTE !</div>

  <div class="cr-sub">
    Different datasets often describe the same business concept using
    different column names. This workspace maps source-specific variables
    to a common canonical dictionary so that datasets can be prepared
    consistently for downstream credit risk modelling.
  </div>

  <div class="cr-sub" style="margin-top:12px;">
    <strong style="color:#42e3d0;">Examples:</strong>
    <strong style="color:#e4bd78;">loan_id</strong>,
    <strong style="color:#e4bd78;">account_number</strong> and
    <strong style="color:#e4bd78;">customer_id</strong>
    map to <strong style="color:#42e3d0;">LOANID</strong>;
    <strong style="color:#e4bd78;">dob</strong>,
    <strong style="color:#e4bd78;">birth_date</strong> and
    <strong style="color:#e4bd78;">date_of_birth</strong>
    map to <strong style="color:#42e3d0;">DATE_OF_BIRTH</strong>;
    while <strong style="color:#e4bd78;">salary</strong>,
    <strong style="color:#e4bd78;">income</strong> and
    <strong style="color:#e4bd78;">monthly_income</strong>
    map to <strong style="color:#42e3d0;">MONTHLY_INCOME</strong>.
  </div>

  <div class="cr-sub" style="margin-top:12px;">
    The objective is to make the modelling environment dataset-agnostic:
    standardise unfamiliar data quickly, reduce repetitive preparation,
    and enable rapid exploratory credit risk model development across
    different portfolio datasets.
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 1. CANONICAL VARIABLE DICTIONARY
# ============================================================

VARIABLE_DICTIONARY = {

    "LOANID": {
        "label": "Loan / account identifier",
        "type": "Identifier",
        "description": "Unique facility, loan, account or observation identifier.",
        "domains": ["PD", "LGD", "EAD", "IFRS9"],
        "aliases": [
            "loan_id", "loanid", "account_id", "accountid",
            "account_number", "accountnumber", "customer_id",
            "customerid", "client_id", "clientid", "borrower_id",
            "application_id", "facility_id", "reference",
            "record_id", "id"
        ]
    },

    "AGE": {
        "label": "Age",
        "type": "Numeric",
        "description": "Borrower age at observation/origination.",
        "domains": ["PD"],
        "aliases": ["age", "borrower_age", "customer_age"]
    },

    "DATE_OF_BIRTH": {
        "label": "Date of birth",
        "type": "Date",
        "description": "Borrower date of birth.",
        "domains": ["PD"],
        "aliases": ["dob", "date_of_birth", "birth_date", "birthdate"]
    },

    "ANNUAL_INCOME": {
        "label": "Annual income",
        "type": "Numeric",
        "description": "Borrower annual income.",
        "domains": ["PD"],
        "aliases": [
            "annual_income", "yearly_income",
            "gross_annual_income", "salary_annual",
            "income_annual"
        ]
    },

    "MONTHLY_INCOME": {
        "label": "Monthly income",
        "type": "Numeric",
        "description": "Borrower monthly income.",
        "domains": ["PD"],
        "aliases": [
            "monthly_income", "income", "salary",
            "net_monthly_income", "gross_monthly_income"
        ]
    },

    "EMPLOYMENT_YEARS": {
        "label": "Employment tenure",
        "type": "Numeric",
        "description": "Years in current employment.",
        "domains": ["PD"],
        "aliases": [
            "employment_years", "years_employed",
            "employment_length", "emp_length", "job_tenure"
        ]
    },

    "EMPLOYMENT_TYPE": {
        "label": "Employment type",
        "type": "Categorical",
        "description": "Employment category or status.",
        "domains": ["PD"],
        "aliases": [
            "employment_type", "employment_status",
            "occupation_type"
        ]
    },

    "EMPLOYER": {
        "label": "Employer",
        "type": "Categorical",
        "description": "Employer name or category.",
        "domains": ["PD"],
        "aliases": ["employer", "employer_name", "company_name"]
    },

    "HOME_OWNERSHIP": {
        "label": "Home ownership",
        "type": "Categorical",
        "description": "Borrower housing/ownership status.",
        "domains": ["PD"],
        "aliases": [
            "home_ownership", "homeowner_status", "housing_status"
        ]
    },

    "CREDIT_SCORE": {
        "label": "Credit score",
        "type": "Numeric",
        "description": "Bureau or internal credit score.",
        "domains": ["PD"],
        "aliases": [
            "credit_score", "score", "bureau_score", "risk_score",
            "behaviour_score", "behavior_score", "bureau_score_value"
        ]
    },

    "CREDIT_HISTORY_MONTHS": {
        "label": "Credit history length",
        "type": "Numeric",
        "description": "Length of observed credit history in months.",
        "domains": ["PD"],
        "aliases": [
            "credit_history_months", "history_months",
            "credit_vintage_months", "account_vintage_months"
        ]
    },

    "EARLIEST_CREDIT_LINE": {
        "label": "Earliest credit line",
        "type": "Date",
        "description": "Date of earliest recorded credit line.",
        "domains": ["PD"],
        "aliases": [
            "earliest_cr_line", "earliest_credit_line",
            "oldest_credit_line", "first_credit_date"
        ]
    },

    "DEBT_TO_INCOME": {
        "label": "Debt-to-income ratio",
        "type": "Numeric",
        "description": "Debt obligations relative to income.",
        "domains": ["PD"],
        "aliases": [
            "dti", "debt_to_income", "debt_income_ratio",
            "debt_to_income_ratio"
        ]
    },

    "CREDIT_UTILIZATION": {
        "label": "Credit utilization",
        "type": "Numeric",
        "description": "Utilisation of available revolving credit.",
        "domains": ["PD"],
        "aliases": [
            "credit_utilization", "credit_utilisation",
            "utilization", "utilisation", "util", "util_rate"
        ]
    },

    "NUM_OPEN_ACCOUNTS": {
        "label": "Number of open accounts",
        "type": "Numeric",
        "description": "Count of currently open credit accounts.",
        "domains": ["PD"],
        "aliases": [
            "num_open_accounts", "open_accounts",
            "number_open_accounts", "open_account_count"
        ]
    },

    "TOTAL_ACCOUNTS": {
        "label": "Total accounts",
        "type": "Numeric",
        "description": "Total number of credit accounts.",
        "domains": ["PD"],
        "aliases": [
            "total_accounts", "account_count",
            "number_of_accounts"
        ]
    },

    "DELINQUENCIES_2YRS": {
        "label": "Delinquencies in 2 years",
        "type": "Numeric",
        "description": "Recorded delinquency count over two years.",
        "domains": ["PD"],
        "aliases": [
            "delinq_2yrs", "delinquencies_2yrs",
            "delinquency_2y", "delinq_2y", "past_due_count_2y"
        ]
    },

    "INQUIRIES_6MTHS": {
        "label": "Credit inquiries in 6 months",
        "type": "Numeric",
        "description": "Credit bureau inquiries over six months.",
        "domains": ["PD"],
        "aliases": [
            "inq_last_6mths", "inquiries_6mths",
            "credit_inquiries_6m", "enquiries_6m",
            "inquiries_last_6_months"
        ]
    },

    "PUBLIC_RECORDS": {
        "label": "Public records",
        "type": "Numeric",
        "description": "Adverse public-record count.",
        "domains": ["PD"],
        "aliases": [
            "pub_rec", "public_records", "judgements",
            "judgments", "insolvencies"
        ]
    },

    "VERIFICATION_STATUS": {
        "label": "Verification status",
        "type": "Categorical",
        "description": "Income or application verification category.",
        "domains": ["PD"],
        "aliases": [
            "verification_status", "income_verification",
            "verified_status"
        ]
    },

    "LOAN_AMOUNT": {
        "label": "Loan amount",
        "type": "Numeric",
        "description": "Original principal or approved facility amount.",
        "domains": ["PD", "LGD", "EAD", "IFRS9"],
        "aliases": [
            "loan_amount", "principal", "credit_advanced",
            "facility_amount", "original_balance", "original_exposure"
        ]
    },

    "LOAN_TERM": {
        "label": "Loan term",
        "type": "Numeric",
        "description": "Contractual term or tenor.",
        "domains": ["PD", "IFRS9"],
        "aliases": [
            "loan_term", "term_months", "tenor",
            "term", "duration_months"
        ]
    },

    "INTEREST_RATE": {
        "label": "Interest rate",
        "type": "Numeric",
        "description": "Contractual interest rate.",
        "domains": ["PD", "IFRS9"],
        "aliases": [
            "interest_rate", "rate", "apr",
            "pricing_rate", "coupon_rate"
        ]
    },

    "LOAN_PURPOSE": {
        "label": "Loan purpose",
        "type": "Categorical",
        "description": "Purpose or product use of the facility.",
        "domains": ["PD"],
        "aliases": [
            "loan_purpose", "purpose",
            "purpose_category", "product_purpose"
        ]
    },

    "ISSUE_DATE": {
        "label": "Origination date",
        "type": "Date",
        "description": "Loan issue or origination date.",
        "domains": ["PD", "IFRS9"],
        "aliases": [
            "issue_d", "issue_date", "origination_date",
            "booking_date", "application_date"
        ]
    },

    "MATURITY_DATE": {
        "label": "Maturity date",
        "type": "Date",
        "description": "Contractual maturity date.",
        "domains": ["IFRS9"],
        "aliases": [
            "maturity_date", "contractual_maturity_date",
            "final_due_date"
        ]
    },

    "DEFAULT": {
        "label": "Default indicator",
        "type": "Binary",
        "description": "Observed default outcome, normally 1 = default and 0 = non-default.",
        "domains": ["PD", "LGD"],
        "aliases": [
            "default", "default_flag", "is_default", "bad_flag",
            "ever_default", "default_indicator", "defaulted"
        ]
    },

    "LOAN_STATUS": {
        "label": "Loan status",
        "type": "Categorical",
        "description": "Observed account or loan status.",
        "domains": ["PD", "LGD"],
        "aliases": ["loan_status", "account_status", "status"]
    },

    "PD_PROBABILITY": {
        "label": "Probability of default",
        "type": "Numeric",
        "description": "Estimated or observed PD value. Confirm horizon and model source.",
        "domains": ["PD", "IFRS9"],
        "aliases": [
            "pd_probability", "pd", "probability_of_default",
            "default_probability"
        ]
    },

    "DEFAULT_DATE": {
        "label": "Default date",
        "type": "Date",
        "description": "Date on which default occurred.",
        "domains": ["LGD", "IFRS9"],
        "aliases": [
            "default_date", "date_of_default", "default_dt"
        ]
    },

    "DAYS_PAST_DUE": {
        "label": "Days past due",
        "type": "Numeric",
        "description": "Days in arrears at observation date.",
        "domains": ["PD", "LGD", "IFRS9"],
        "aliases": [
            "days_past_due", "dpd",
            "days_in_arrears", "arrears_days"
        ]
    },

    "CREDIT_LIMIT": {
        "label": "Credit limit",
        "type": "Numeric",
        "description": "Approved revolving credit limit.",
        "domains": ["EAD"],
        "aliases": [
            "credit_limit", "approved_limit",
            "facility_limit", "limit_amount"
        ]
    },

    "CURRENT_BALANCE": {
        "label": "Current drawn balance",
        "type": "Numeric",
        "description": "Outstanding drawn exposure at observation/default.",
        "domains": ["EAD", "LGD", "IFRS9"],
        "aliases": [
            "current_balance", "balance", "outstanding_balance",
            "drawn_balance", "balance_at_default"
        ]
    },

    "CURRENT_UTILIZATION": {
        "label": "Current utilization",
        "type": "Numeric",
        "description": "Current drawn balance as a proportion of credit limit.",
        "domains": ["EAD"],
        "aliases": [
            "current_utilization", "current_utilisation",
            "utilization_at_observation"
        ]
    },

    "UNDRAWN_AMOUNT": {
        "label": "Undrawn amount",
        "type": "Numeric",
        "description": "Available but undrawn commitment.",
        "domains": ["EAD"],
        "aliases": [
            "undrawn_amount", "available_credit",
            "undrawn_commitment", "unused_limit"
        ]
    },

    "CCF": {
        "label": "Credit conversion factor",
        "type": "Numeric",
        "description": "Proportion of undrawn commitment expected to be drawn before default.",
        "domains": ["EAD"],
        "aliases": [
            "ccf", "credit_conversion_factor", "conversion_factor"
        ]
    },

    "EAD": {
        "label": "Exposure at default",
        "type": "Numeric",
        "description": "Exposure outstanding at the point of default.",
        "domains": ["EAD", "LGD", "IFRS9"],
        "aliases": [
            "ead", "exposure_at_default", "ex_at_def",
            "exposure_default", "exposure_at_def", "ead_at_default"
        ]
    },

    "COLLATERAL_TYPE": {
        "label": "Collateral type",
        "type": "Categorical",
        "description": "Type of security or collateral.",
        "domains": ["LGD"],
        "aliases": [
            "collateral_type", "security_type", "asset_type"
        ]
    },

    "COLLATERAL_VALUE": {
        "label": "Collateral value",
        "type": "Numeric",
        "description": "Value of collateral available to support recovery.",
        "domains": ["LGD"],
        "aliases": [
            "collateral_value", "security_value",
            "market_value_collateral"
        ]
    },

    "COLLATERAL_RATIO": {
        "label": "Collateral ratio",
        "type": "Numeric",
        "description": "Collateral value relative to exposure or balance.",
        "domains": ["LGD"],
        "aliases": [
            "collateral_ratio", "loan_to_value",
            "ltv", "security_coverage"
        ]
    },

    "RECOVERY_RATE": {
        "label": "Recovery rate",
        "type": "Numeric",
        "description": "Recovered amount as a proportion of exposure.",
        "domains": ["LGD"],
        "aliases": [
            "recovery_rate", "recovery_pct", "recovery_percentage"
        ]
    },

    "RECOVERY_AMOUNT": {
        "label": "Recovery amount",
        "type": "Numeric",
        "description": "Cash recoveries associated with defaulted exposure.",
        "domains": ["LGD"],
        "aliases": [
            "recovery_amount", "recoveries",
            "total_recovery", "cash_recovered"
        ]
    },

    "COLLECTION_COST": {
        "label": "Collection cost",
        "type": "Numeric",
        "description": "Costs incurred in the recovery process.",
        "domains": ["LGD"],
        "aliases": [
            "collection_cost", "recovery_cost",
            "workout_cost", "collection_expense"
        ]
    },

    "LGD": {
        "label": "Loss given default",
        "type": "Numeric",
        "description": "Observed loss severity following default.",
        "domains": ["LGD"],
        "aliases": [
            "lgd", "loss_given_default", "loss_severity"
        ]
    },

    "ADJUSTED_LGD": {
        "label": "Adjusted loss given default",
        "type": "Numeric",
        "description": "Adjusted LGD measure, for example after costs or policy adjustments.",
        "domains": ["LGD"],
        "aliases": [
            "adjusted_lgd", "downturn_lgd",
            "economic_lgd", "lgd_adjusted"
        ]
    },
}


# ============================================================
# 2. MODELLING DOMAIN REQUIREMENTS
# ============================================================

DOMAIN_REQUIRED = {
    "PD": ["DEFAULT"],
    "LGD": ["DEFAULT", "EAD"],
    "EAD": ["CURRENT_BALANCE", "UNDRAWN_AMOUNT"],
    "IFRS9": ["PD_PROBABILITY", "LGD", "EAD"],
}


# ============================================================
# 3. GENERAL UTILITY FUNCTIONS
# ============================================================

def normalise_name(value):
    value = str(value).strip()

    value = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1_\2",
        value
    )

    value = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        value
    )

    return re.sub(
        r"_+",
        "_",
        value
    ).strip("_").upper()


def alias_key(value):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        normalise_name(value)
    )


def infer_variable_type(series):

    non_missing = series.dropna()

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date"

    if non_missing.empty:
        return "Empty"

    if pd.api.types.is_bool_dtype(series):
        return "Binary"

    numeric = pd.to_numeric(
        non_missing,
        errors="coerce"
    )

    if numeric.notna().mean() >= 0.95:

        if numeric.nunique() <= 2:
            return "Binary"

        if numeric.nunique() <= 12:
            return "Numeric / discrete"

        return "Numeric / continuous"

    if non_missing.nunique() <= max(
        20,
        int(len(non_missing) * 0.05)
    ):
        return "Categorical"

    return "Text / high-cardinality"


def csv_bytes(df):
    return df.to_csv(
        index=False
    ).encode("utf-8-sig")


def invalidate_downstream():

    prefixes = (
        "pd_",
        "ead_",
        "lgd_",
        "ecl_",
        "integrated_"
    )

    for key in list(st.session_state.keys()):

        if key.startswith(prefixes):
            st.session_state.pop(key, None)


# ============================================================
# 4. LOAD PERSISTED ALIASES AND CUSTOM DICTIONARY
# ============================================================

def load_custom_aliases():

    if not ALIAS_FILE.exists():
        return {}

    try:

        payload = json.loads(
            ALIAS_FILE.read_text(encoding="utf-8")
        )

        if isinstance(payload, dict) and "custom_aliases" in payload:
            payload = payload["custom_aliases"]

        return payload if isinstance(payload, dict) else {}

    except Exception:

        return {}


def load_custom_dictionary():

    if not CUSTOM_DICTIONARY_FILE.exists():
        return {}

    try:

        payload = json.loads(
            CUSTOM_DICTIONARY_FILE.read_text(encoding="utf-8")
        )

        return payload if isinstance(payload, dict) else {}

    except Exception:

        return {}


if "custom_variable_aliases" not in st.session_state:
    st.session_state["custom_variable_aliases"] = load_custom_aliases()

if "custom_variable_dictionary" not in st.session_state:
    st.session_state["custom_variable_dictionary"] = load_custom_dictionary()

if "variable_decisions" not in st.session_state:
    st.session_state["variable_decisions"] = {}

if "mapping_audit_log" not in st.session_state:
    st.session_state["mapping_audit_log"] = []


custom_aliases = st.session_state["custom_variable_aliases"].copy()


def get_active_dictionary():

    active = dict(VARIABLE_DICTIONARY)

    custom = st.session_state.get(
        "custom_variable_dictionary",
        {}
    )

    for canonical, metadata in custom.items():

        if canonical not in active:
            active[canonical] = metadata

    return active


def build_alias_lookup(custom_aliases):

    lookup = {}

    dictionary = get_active_dictionary()

    for canonical, meta in dictionary.items():

        names = [canonical] + meta.get("aliases", [])

        for name in names:

            key = alias_key(name)

            if key:
                lookup.setdefault(key, canonical)

    for source_alias, canonical in custom_aliases.items():

        if canonical in dictionary:
            lookup[alias_key(source_alias)] = canonical

    return lookup


# ============================================================
# 5. GOVERNED DATASET CONSTRUCTION
# ============================================================

def build_governed_datasets(
    original,
    custom_aliases,
    decisions
):
    """
    Create three distinct outputs:

    Original:
        Preserved exactly as ingested.

    Standardised:
        All columns retained, with recognised names standardised.

    Modelling:
        Only approved and mapped variables included.
    """

    lookup = build_alias_lookup(custom_aliases)
    dictionary = get_active_dictionary()

    standardised_columns = {}
    modelling_columns = {}
    register_rows = []

    used_names = {}

    for source_col in original.columns:

        source_name = str(source_col)
        key = alias_key(source_name)

        decision = decisions.get(source_name, {})

        action = decision.get("action", "Pending")
        selected_target = decision.get("target")

        detected_target = lookup.get(key)

        canonical = None

        if action == "Discard":

            canonical = None

        elif action in ("Map", "Add to dictionary"):

            canonical = selected_target

        elif detected_target:

            canonical = detected_target
            action = "Automatically mapped"

        else:

            action = "Pending review"

        if canonical and canonical in dictionary:

            base_name = canonical

        else:

            base_name = normalise_name(source_name)

        count = used_names.get(base_name, 0) + 1
        used_names[base_name] = count

        final_name = (
            base_name
            if count == 1
            else f"{base_name}_{count}"
        )

        series = original[source_col].copy()

        standardised_columns[final_name] = series

        if canonical and action != "Discard":

            modelling_columns[final_name] = series

        if action == "Discard":

            mapping_status = "Discarded"

        elif action == "Pending review":

            mapping_status = "Pending review"

        elif count > 1:

            mapping_status = "Mapping conflict"

        elif canonical:

            mapping_status = "Mapped"

        else:

            mapping_status = "Unmapped"

        register_rows.append({

            "Source column": source_name,

            "Standardised column": final_name,

            "Canonical variable": canonical or "",

            "Mapping status": mapping_status,

            "Governance decision": action,

            "Detected type": infer_variable_type(series),

            "Missing %": round(
                series.isna().mean() * 100,
                2
            ),

            "Unique values": int(
                series.nunique(dropna=True)
            ),

            "Example values": " | ".join(
                series.dropna()
                .astype(str)
                .head(3)
                .tolist()
            ),

        })

    standardised = pd.DataFrame(
        standardised_columns,
        index=original.index
    )

    modelling = pd.DataFrame(
        modelling_columns,
        index=original.index
    )

    register = pd.DataFrame(register_rows)

    return standardised, modelling, register


# ============================================================
# 6. ACTIVATE DATASET
# ============================================================

def activate_dataset(
    original,
    filename,
    source_label,
    custom_aliases
):

    decisions = st.session_state.get(
        "variable_decisions",
        {}
    )

    standardised, modelling, register = (
        build_governed_datasets(
            original,
            custom_aliases,
            decisions
        )
    )

    st.session_state["credit_data_original"] = (
        original.copy()
    )

    st.session_state["credit_data_standardized"] = (
        standardised.copy()
    )

    st.session_state["credit_data"] = (
        modelling.copy()
    )

    st.session_state["variable_recognition"] = (
        register.copy()
    )

    st.session_state["data_source"] = source_label
    st.session_state["data_name"] = filename

    st.session_state["variable_dictionary"] = (
        get_active_dictionary()
    )

    st.session_state["custom_variable_aliases"] = (
        custom_aliases.copy()
    )

    st.session_state["data_ingested_at"] = (
        datetime.now(timezone.utc).isoformat()
    )

    try:

        fingerprint_data = pd.util.hash_pandas_object(
            original,
            index=True
        ).values.tobytes()

        st.session_state["data_fingerprint"] = (
            hashlib.sha256(fingerprint_data).hexdigest()
        )

    except Exception:

        st.session_state["data_fingerprint"] = (
            "Fingerprint unavailable for this dataset"
        )

    invalidate_downstream()


# ============================================================
# 7. PORTFOLIO SOURCE
# ============================================================

st.subheader("1. Portfolio source")

source_choice = st.radio(
    "Choose how to populate the portfolio",
    [
        "Generate synthetic portfolio",
        "Upload CSV portfolio"
    ],
    horizontal=True,
    key="portfolio_source_choice"
)


# ------------------------------------------------------------
# SYNTHETIC DATA GENERATION
# ------------------------------------------------------------

if source_choice == "Generate synthetic portfolio":

    st.caption(
        "Synthetic data is intended for development, demonstration "
        "and testing. It is not production borrower data."
    )

    c1, c2 = st.columns(2)

    n_obs = c1.number_input(
        "Number of observations",
        min_value=100,
        max_value=1_000_000,
        value=10_000,
        step=100
    )

    seed = c2.number_input(
        "Random seed",
        min_value=1,
        max_value=999_999,
        value=42
    )

    if st.button(
        "Generate synthetic portfolio",
        type="primary"
    ):

        try:

            from models.data_simulation import (
                generate_synthetic_data
            )

            generated = generate_synthetic_data(
                int(n_obs),
                random_state=int(seed)
            )

            st.session_state["variable_decisions"] = {}

            activate_dataset(
                generated,
                f"synthetic_portfolio_{int(n_obs)}.csv",
                "Synthetic generator",
                custom_aliases
            )

            st.success(
                "Synthetic portfolio generated and activated."
            )

            st.rerun()

        except Exception as exc:

            st.error(
                f"Synthetic data generation failed: {exc}"
            )


# ------------------------------------------------------------
# CSV UPLOAD
# ------------------------------------------------------------

else:

    st.caption(
        "CSV ingestion. The configured Streamlit upload limit "
        "does not guarantee that the hosting platform has sufficient "
        "memory for very large files."
    )

    uploaded = st.file_uploader(
        "Upload portfolio CSV",
        type=["csv"],
        key="creditrisk_csv_upload"
    )

    if uploaded is not None:

        st.write(
            f"**File:** {uploaded.name}  ·  "
            f"**Size:** {uploaded.size / (1024**2):,.2f} MB"
        )

        delimiter = st.selectbox(
            "CSV delimiter",
            [",", ";", "\\t", "|"],
            index=0
        )

        encoding = st.selectbox(
            "Text encoding",
            ["utf-8", "utf-8-sig", "latin-1"],
            index=0
        )

        if st.button(
            "Ingest and standardise CSV",
            type="primary"
        ):

            try:

                uploaded.seek(0)

                raw = pd.read_csv(
                    uploaded,
                    sep=delimiter,
                    encoding=encoding,
                    low_memory=False
                )

                if raw.empty or raw.shape[1] < 2:

                    st.error(
                        "The file must contain observations "
                        "and at least two columns."
                    )

                else:

                    st.session_state["variable_decisions"] = {}

                    activate_dataset(
                        raw,
                        uploaded.name,
                        "CSV upload",
                        custom_aliases
                    )

                    st.success(
                        "CSV ingested. Original source preserved "
                        "and standardised dataset activated."
                    )

                    st.rerun()

            except Exception as exc:

                st.error(
                    f"CSV ingestion failed: {exc}"
                )


# ============================================================
# 8. ACTIVE PORTFOLIO
# ============================================================

if "credit_data_original" not in st.session_state:

    st.info(
        "Generate or upload a portfolio to begin the data governance review."
    )

    st.stop()


original_data = st.session_state[
    "credit_data_original"
].copy()

standard_data = st.session_state[
    "credit_data_standardized"
].copy()

modelling_data = st.session_state[
    "credit_data"
].copy()

register = st.session_state[
    "variable_recognition"
].copy()


mapped_count = int(
    register["Mapping status"].eq("Mapped").sum()
)

unmapped_count = int(
    register["Mapping status"].eq("Unmapped").sum()
)

pending_count = int(
    register["Mapping status"].eq("Pending review").sum()
)

discarded_count = int(
    register["Mapping status"].eq("Discarded").sum()
)

conflict_count = int(
    register["Mapping status"].eq("Mapping conflict").sum()
)

coverage = (
    mapped_count / len(register) * 100
    if len(register)
    else 0
)

default_col = (
    "DEFAULT"
    if "DEFAULT" in modelling_data.columns
    else None
)

default_rate = (
    pd.to_numeric(
        modelling_data[default_col],
        errors="coerce"
    ).mean()
    if default_col
    else np.nan
)


# ============================================================
# 9. GOVERNANCE OVERVIEW
# ============================================================

st.divider()

st.subheader("2. Portfolio governance overview")

st.caption(
    "The original portfolio is retained separately. "
    "Only approved mapped variables are supplied to downstream "
    "modelling pages through st.session_state['credit_data']."
)

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Observations",
    f"{len(original_data):,}"
)

k2.metric(
    "Source variables",
    f"{original_data.shape[1]:,}"
)

k3.metric(
    "Mapped variables",
    f"{mapped_count:,}"
)

k4.metric(
    "Pending review",
    f"{pending_count + unmapped_count:,}"
)

k5.metric(
    "Discarded variables",
    f"{discarded_count:,}"
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Mapping conflicts",
    f"{conflict_count:,}"
)

m2.metric(
    "Default rate",
    f"{default_rate:.2%}"
    if pd.notna(default_rate)
    else "Not available"
)

m3.metric(
    "Modelling variables",
    f"{modelling_data.shape[1]:,}"
)

metadata = pd.DataFrame([{

    "Portfolio name": st.session_state.get(
        "data_name",
        "Unknown"
    ),

    "Source": st.session_state.get(
        "data_source",
        "Unknown"
    ),

    "Ingested (UTC)": st.session_state.get(
        "data_ingested_at",
        "Not recorded"
    ),

    "Rows": len(original_data),

    "Original columns": original_data.shape[1],

    "Standardised columns": standard_data.shape[1],

    "Approved modelling columns": modelling_data.shape[1],

    "SHA-256 fingerprint": st.session_state.get(
        "data_fingerprint",
        "Not available"
    ),

}])

with st.expander(
    "Data provenance and ingestion metadata",
    expanded=False
):

    st.dataframe(
        metadata,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 10. VARIABLE MAPPING REGISTER
# ============================================================

st.divider()

st.subheader("3. Variable mapping register")

st.write(
    "Every source column is accounted for. Variables are either "
    "automatically mapped, manually mapped, awaiting review, or discarded."
)

if not register.empty:

    filter_status = st.multiselect(
        "Filter mapping status",
        [
            "Mapped",
            "Unmapped",
            "Pending review",
            "Discarded",
            "Mapping conflict"
        ],
        default=[
            "Mapped",
            "Unmapped",
            "Pending review",
            "Discarded",
            "Mapping conflict"
        ],
        key="mapping_status_filter"
    )

    display_register = register[
        register["Mapping status"].isin(filter_status)
    ]

    st.dataframe(
        display_register,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download mapping register (CSV)",
        csv_bytes(register),
        "creditrisk_mapping_register.csv",
        "text/csv"
    )


# ============================================================
# 11. VARIABLE GOVERNANCE DECISIONS
# ============================================================

st.divider()

st.subheader("4. Variable governance decisions")

st.write(
    "Review variables that are not recognised by the canonical "
    "dictionary. You can map them, add a new canonical variable, "
    "or discard them from the modelling dataset."
)

review_sources = (
    register.loc[
        register["Mapping status"].isin([
            "Unmapped",
            "Pending review"
        ]),
        "Source column"
    ].tolist()
)

if review_sources:

    st.warning(
        f"{len(review_sources)} variable(s) require a governance decision."
    )

    selected_source = st.selectbox(
        "Select a variable to review",
        review_sources,
        key="governance_source"
    )

    source_series = original_data[selected_source]

    profile1, profile2, profile3 = st.columns(3)

    profile1.metric(
        "Detected type",
        infer_variable_type(source_series)
    )

    profile2.metric(
        "Missing values",
        f"{source_series.isna().mean():.1%}"
    )

    profile3.metric(
        "Unique values",
        f"{source_series.nunique(dropna=True):,}"
    )

    st.markdown("**Example values**")

    examples = (
        source_series.dropna()
        .astype(str)
        .head(10)
        .tolist()
    )

    st.code(
        ", ".join(examples)
        if examples
        else "No non-missing values"
    )

    action = st.radio(
        "What would you like to do with this variable?",
        [
            "Map to an existing canonical variable",
            "Add a new canonical variable to the dictionary",
            "Discard this variable"
        ],
        key="governance_action"
    )

    dictionary = get_active_dictionary()

    # --------------------------------------------------------
    # MAP TO EXISTING CANONICAL VARIABLE
    # --------------------------------------------------------

    if action == "Map to an existing canonical variable":

        target = st.selectbox(
            "Select the canonical variable",
            list(dictionary.keys()),
            format_func=lambda x: (
                f"{x} · {dictionary[x]['label']}"
            ),
            key="governance_existing_target"
        )

        confirm = st.checkbox(
            "I confirm that the source variable represents this canonical variable.",
            key="governance_map_confirm"
        )

        if st.button(
            "Confirm mapping",
            type="primary",
            disabled=not confirm
        ):

            aliases = st.session_state[
                "custom_variable_aliases"
            ].copy()

            aliases[selected_source] = target

            decisions = st.session_state[
                "variable_decisions"
            ].copy()

            decisions[selected_source] = {
                "action": "Map",
                "target": target,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat()
            }

            st.session_state["custom_variable_aliases"] = aliases
            st.session_state["variable_decisions"] = decisions

            st.session_state["mapping_audit_log"].append({

                "Timestamp UTC": datetime.now(
                    timezone.utc
                ).isoformat(),

                "Source column": selected_source,

                "Canonical target": target,

                "Action": "Mapped to existing variable",

                "Status": "User-confirmed"

            })

            activate_dataset(
                original_data,
                st.session_state["data_name"],
                st.session_state["data_source"],
                aliases
            )

            st.success(
                f"{selected_source} mapped to {target}."
            )

            st.rerun()

    # --------------------------------------------------------
    # ADD A NEW CANONICAL VARIABLE
    # --------------------------------------------------------

    elif action == "Add a new canonical variable to the dictionary":

        st.info(
            "Use this option when the source variable represents "
            "a genuinely new business concept that does not belong "
            "to an existing canonical variable."
        )

        new_name = st.text_input(
            "New canonical variable name",
            value=normalise_name(selected_source),
            key="new_canonical_name"
        )

        new_label = st.text_input(
            "Business label",
            value=selected_source.replace("_", " ").title(),
            key="new_canonical_label"
        )

        new_type = st.selectbox(
            "Variable type",
            [
                "Numeric",
                "Binary",
                "Categorical",
                "Date",
                "Identifier",
                "Text"
            ],
            key="new_canonical_type"
        )

        new_description = st.text_area(
            "Business definition",
            key="new_canonical_description"
        )

        new_domains = st.multiselect(
            "Relevant modelling domains",
            ["PD", "LGD", "EAD", "IFRS9"],
            key="new_canonical_domains"
        )

        confirm_new = st.checkbox(
            "I confirm that this is a new canonical business variable.",
            key="new_canonical_confirm"
        )

        if st.button(
            "Add variable to dictionary",
            type="primary",
            disabled=(
                not confirm_new
                or not new_name.strip()
                or not new_label.strip()
                or not new_description.strip()
            )
        ):

            canonical = normalise_name(new_name)

            if not canonical:

                st.error(
                    "Enter a valid canonical variable name."
                )

            elif canonical in dictionary:

                st.error(
                    f"{canonical} already exists in the dictionary."
                )

            else:

                custom_dictionary = st.session_state[
                    "custom_variable_dictionary"
                ].copy()

                custom_dictionary[canonical] = {

                    "label": new_label.strip(),

                    "type": new_type,

                    "description": new_description.strip(),

                    "domains": new_domains,

                    "aliases": [selected_source]

                }

                aliases = st.session_state[
                    "custom_variable_aliases"
                ].copy()

                aliases[selected_source] = canonical

                decisions = st.session_state[
                    "variable_decisions"
                ].copy()

                decisions[selected_source] = {
                    "action": "Add to dictionary",
                    "target": canonical,
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat()
                }

                st.session_state[
                    "custom_variable_dictionary"
                ] = custom_dictionary

                st.session_state[
                    "custom_variable_aliases"
                ] = aliases

                st.session_state[
                    "variable_decisions"
                ] = decisions

                st.session_state["mapping_audit_log"].append({

                    "Timestamp UTC": datetime.now(
                        timezone.utc
                    ).isoformat(),

                    "Source column": selected_source,

                    "Canonical target": canonical,

                    "Action": "New canonical variable created",

                    "Status": "User-confirmed"

                })

                activate_dataset(
                    original_data,
                    st.session_state["data_name"],
                    st.session_state["data_source"],
                    aliases
                )

                st.success(
                    f"{canonical} added to the custom dictionary."
                )

                st.rerun()

    # --------------------------------------------------------
    # DISCARD VARIABLE
    # --------------------------------------------------------

    else:

        st.warning(
            "Discarding excludes this variable from the modelling "
            "dataset. It remains available in the original source data."
        )

        confirm_discard = st.checkbox(
            "I confirm that this variable is not required for the intended modelling purpose.",
            key="governance_discard_confirm"
        )

        if st.button(
            "Discard variable from modelling dataset",
            type="primary",
            disabled=not confirm_discard
        ):

            decisions = st.session_state[
                "variable_decisions"
            ].copy()

            decisions[selected_source] = {
                "action": "Discard",
                "target": None,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat()
            }

            st.session_state["variable_decisions"] = decisions

            st.session_state["mapping_audit_log"].append({

                "Timestamp UTC": datetime.now(
                    timezone.utc
                ).isoformat(),

                "Source column": selected_source,

                "Canonical target": "",

                "Action": "Discarded from modelling dataset",

                "Status": "User-confirmed"

            })

            activate_dataset(
                original_data,
                st.session_state["data_name"],
                st.session_state["data_source"],
                st.session_state["custom_variable_aliases"]
            )

            st.success(
                f"{selected_source} excluded from the modelling dataset."
            )

            st.rerun()

else:

    st.success(
        "All source variables have received a governance decision."
    )


# ============================================================
# 12. CUSTOM ALIAS MANAGEMENT AND AUDIT HISTORY
# ============================================================

with st.expander(
    "Manage custom aliases and mapping audit history",
    expanded=False
):

    current_aliases = st.session_state[
        "custom_variable_aliases"
    ]

    alias_df = pd.DataFrame([

        {
            "Source alias": key,
            "Canonical variable": value
        }

        for key, value in current_aliases.items()

    ])

    if not alias_df.empty:

        st.dataframe(
            alias_df,
            use_container_width=True,
            hide_index=True
        )

        alias_to_remove = st.selectbox(
            "Select custom alias to remove",
            list(current_aliases.keys()),
            key="remove_alias_select"
        )

        if st.button(
            "Remove selected custom alias"
        ):

            revised = current_aliases.copy()

            revised.pop(alias_to_remove, None)

            decisions = st.session_state[
                "variable_decisions"
            ].copy()

            decisions.pop(alias_to_remove, None)

            st.session_state["variable_decisions"] = decisions
            st.session_state["custom_variable_aliases"] = revised

            st.session_state["mapping_audit_log"].append({

                "Timestamp UTC": datetime.now(
                    timezone.utc
                ).isoformat(),

                "Source column": alias_to_remove,

                "Canonical target": current_aliases[alias_to_remove],

                "Action": "Custom alias removed",

                "Status": "User action"

            })

            activate_dataset(
                original_data,
                st.session_state["data_name"],
                st.session_state["data_source"],
                revised
            )

            st.rerun()

    else:

        st.info(
            "No custom aliases have been added."
        )

    audit = pd.DataFrame(
        st.session_state.get(
            "mapping_audit_log",
            []
        )
    )

    if not audit.empty:

        st.markdown("**Session audit history**")

        st.dataframe(
            audit,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download mapping audit history",
            csv_bytes(audit),
            "creditrisk_mapping_audit.csv",
            "text/csv"
        )

    st.caption(
        "The audit log shown here is session-level. "
        "It is not a formal approval workflow or immutable audit log."
    )


# ============================================================
# 13. CANONICAL VARIABLE DICTIONARY
# ============================================================

active_dictionary = get_active_dictionary()

dictionary_rows = []

for canonical, meta in active_dictionary.items():

    dictionary_rows.append({

        "Canonical variable": canonical,

        "Business label": meta.get("label", ""),

        "Data type": meta.get("type", ""),

        "Description": meta.get("description", ""),

        "Modelling domains": ", ".join(
            meta.get("domains", [])
        ),

        "Built-in aliases": ", ".join(
            meta.get("aliases", [])
        ),

        "Custom aliases": ", ".join([
            alias
            for alias, target in st.session_state[
                "custom_variable_aliases"
            ].items()
            if target == canonical
        ]),

    })


dictionary_df = pd.DataFrame(dictionary_rows)

with st.expander(
    "Canonical variable dictionary",
    expanded=False
):

    search_text = st.text_input(
        "Search dictionary",
        key="dictionary_search"
    )

    filtered_dict = dictionary_df

    if search_text:

        mask = dictionary_df.astype(str).apply(
            lambda col: col.str.contains(
                search_text,
                case=False,
                na=False
            )
        ).any(axis=1)

        filtered_dict = dictionary_df[mask]

    st.dataframe(
        filtered_dict,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download canonical dictionary (CSV)",
        csv_bytes(dictionary_df),
        "creditrisk_variable_dictionary.csv",
        "text/csv"
    )

    st.download_button(
        "Download custom dictionary (JSON)",
        json.dumps(
            st.session_state[
                "custom_variable_dictionary"
            ],
            indent=2
        ).encode("utf-8"),
        "creditrisk_custom_variable_dictionary.json",
        "application/json"
    )


# ============================================================
# 14. CUSTOM ALIAS AND DICTIONARY PERSISTENCE
# ============================================================

st.divider()

st.subheader("5. Dictionary and alias management")

alias_payload = {
    "version": 1,
    "exported_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "custom_aliases": st.session_state[
        "custom_variable_aliases"
    ]
}

custom_dictionary_payload = st.session_state[
    "custom_variable_dictionary"
]

c1, c2 = st.columns(2)

with c1:

    st.download_button(
        "Download custom alias register (JSON)",
        json.dumps(
            alias_payload,
            indent=2
        ).encode("utf-8"),
        "creditrisk_custom_aliases.json",
        "application/json"
    )

with c2:

    st.download_button(
        "Download custom dictionary (JSON)",
        json.dumps(
            custom_dictionary_payload,
            indent=2
        ).encode("utf-8"),
        "creditrisk_custom_dictionary.json",
        "application/json"
    )


if st.button(
    "Save dictionary and aliases to application files",
    type="primary"
):

    try:

        ALIAS_FILE.write_text(
            json.dumps(
                st.session_state[
                    "custom_variable_aliases"
                ],
                indent=2
            ),
            encoding="utf-8"
        )

        CUSTOM_DICTIONARY_FILE.write_text(
            json.dumps(
                custom_dictionary_payload,
                indent=2
            ),
            encoding="utf-8"
        )

        st.success(
            "Custom aliases and dictionary saved to the application data folder."
        )

    except Exception as exc:

        st.error(
            f"Could not save dictionary files: {exc}"
        )

st.caption(
    "On hosted deployments, local filesystem changes may not survive "
    "redeployment. Download and commit approved JSON changes to your "
    "application repository for durable persistence."
)


# ============================================================
# 15. DATA QUALITY AND MODELLING READINESS
# ============================================================

st.divider()

st.subheader("6. Data quality and modelling readiness")

st.caption(
    "Ingestion acceptance and modelling readiness are separate decisions. "
    "Only approved mapped variables are assessed for modelling readiness."
)

domain = st.selectbox(
    "Select modelling domain for readiness assessment",
    ["PD", "LGD", "EAD", "IFRS9"],
    key="readiness_domain"
)

required = DOMAIN_REQUIRED[domain]

available = set(
    modelling_data.columns
)

missing_required = [
    variable
    for variable in required
    if variable not in available
]

available_required = [
    variable
    for variable in required
    if variable in available
]


quality_rows = []

for col in modelling_data.columns:

    series = modelling_data[col]

    quality_rows.append({

        "Variable": col,

        "Data type": infer_variable_type(series),

        "Missing observations": int(
            series.isna().sum()
        ),

        "Missing %": round(
            series.isna().mean() * 100,
            2
        ),

        "Unique values": int(
            series.nunique(dropna=True)
        ),

        "Constant": (
            series.nunique(dropna=True) <= 1
        ),

    })


quality_df = pd.DataFrame(quality_rows)


q1, q2, q3, q4 = st.columns(4)

q1.metric(
    "Required fields available",
    f"{len(available_required)}/{len(required)}"
)

q2.metric(
    "Required fields missing",
    len(missing_required)
)

q3.metric(
    "Columns with missing data",
    int(
        (quality_df["Missing observations"] > 0).sum()
    ) if not quality_df.empty else 0
)

q4.metric(
    "Discarded variables",
    discarded_count
)


if missing_required:

    st.error(
        "Modelling readiness: NOT READY. "
        "Missing required canonical variables: "
        + ", ".join(missing_required)
    )

else:

    st.success(
        f"Required-field check passed for {domain}. "
        "This is a structural check, not model validation "
        "or production approval."
    )


if conflict_count:

    st.warning(
        f"{conflict_count} mapping/name collision(s) require review."
    )


if pending_count or unmapped_count:

    st.warning(
        f"{pending_count + unmapped_count} variable(s) remain unresolved. "
        "They are excluded from the modelling dataset."
    )


if not quality_df.empty:

    with st.expander(
        "Detailed data quality profile",
        expanded=False
    ):

        st.dataframe(
            quality_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download data quality report (CSV)",
            csv_bytes(quality_df),
            "creditrisk_data_quality_report.csv",
            "text/csv"
        )


with st.expander(
    "Required variables for selected modelling domain",
    expanded=False
):

    st.write(
        f"**Domain:** {domain}"
    )

    st.write(
        "**Required canonical variables:** "
        + ", ".join(required)
    )

    st.write(
        "**Available:** "
        + (
            ", ".join(available_required)
            if available_required
            else "None"
        )
    )

    st.write(
        "**Missing:** "
        + (
            ", ".join(missing_required)
            if missing_required
            else "None"
        )
    )


# ============================================================
# 16. PORTFOLIO EXPLORATION
# ============================================================

st.divider()

st.subheader("7. Portfolio exploration")


# ------------------------------------------------------------
# DEFAULT DISTRIBUTION
# ------------------------------------------------------------

if "DEFAULT" in modelling_data.columns:

    default_values = pd.to_numeric(
        modelling_data["DEFAULT"],
        errors="coerce"
    )

    distribution = (
        default_values
        .value_counts(dropna=False)
        .rename_axis("Default outcome")
        .reset_index(name="Observations")
    )

    c1, c2 = st.columns([1, 2])

    with c1:

        st.markdown("**Default distribution**")

        st.dataframe(
            distribution,
            use_container_width=True,
            hide_index=True
        )

    with c2:

        fig = px.bar(
            distribution,
            x="Default outcome",
            y="Observations",
            color="Default outcome",
            title="Default outcome distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

else:

    st.info(
        "DEFAULT is not available in the approved modelling dataset. "
        "Default distribution is therefore not displayed."
    )


# ------------------------------------------------------------
# MODELLING DATA PREVIEW
# ------------------------------------------------------------

with st.expander(
    "Approved modelling dataset",
    expanded=True
):

    st.caption(
        "This dataset contains only approved mapped variables."
    )

    st.dataframe(
        modelling_data.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Export approved modelling dataset (CSV)",
        csv_bytes(modelling_data),
        "creditrisk_modelling_dataset.csv",
        "text/csv"
    )


# ------------------------------------------------------------
# STANDARDISED DATA PREVIEW
# ------------------------------------------------------------

with st.expander(
    "Complete standardised dataset",
    expanded=False
):

    st.caption(
        "Includes mapped, unmapped, pending and discarded variables. "
        "This is not the modelling hand-off dataset."
    )

    st.dataframe(
        standard_data.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Export complete standardised dataset (CSV)",
        csv_bytes(standard_data),
        "creditrisk_standardised_portfolio.csv",
        "text/csv"
    )


# ------------------------------------------------------------
# VARIABLE INFORMATION
# ------------------------------------------------------------

with st.expander(
    "Variable information",
    expanded=False
):

    info = pd.DataFrame({

        "Variable": modelling_data.columns,

        "Data type": [
            str(modelling_data[c].dtype)
            for c in modelling_data.columns
        ],

        "Inferred type": [
            infer_variable_type(modelling_data[c])
            for c in modelling_data.columns
        ],

        "Non-missing": [
            int(modelling_data[c].notna().sum())
            for c in modelling_data.columns
        ],

        "Unique": [
            int(modelling_data[c].nunique(dropna=True))
            for c in modelling_data.columns
        ],

    })

    st.dataframe(
        info,
        use_container_width=True,
        hide_index=True
    )


# ------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

with st.expander(
    "Descriptive statistics",
    expanded=False
):

    if modelling_data.shape[1] > 0:

        st.dataframe(
            modelling_data.describe(
                include="all"
            ).T,
            use_container_width=True
        )

    else:

        st.info(
            "There are currently no approved modelling variables."
        )


# ------------------------------------------------------------
# MISSING-VALUE ANALYSIS
# ------------------------------------------------------------

with st.expander(
    "Missing-value analysis",
    expanded=False
):

    if not quality_df.empty:

        missing_table = quality_df[
            [
                "Variable",
                "Missing observations",
                "Missing %"
            ]
        ].sort_values(
            "Missing %",
            ascending=False
        )

        st.dataframe(
            missing_table,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            missing_table.head(30),
            x="Variable",
            y="Missing %",
            title="Missing data by approved modelling variable"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No approved modelling variables are available for this analysis."
        )


# ------------------------------------------------------------
# ORIGINAL SOURCE DATA
# ------------------------------------------------------------

with st.expander(
    "Original source data",
    expanded=False
):

    st.caption(
        "Read-only copy of the data as ingested, before canonical renaming "
        "or governance decisions."
    )

    st.dataframe(
        original_data.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Export original source data (CSV)",
        csv_bytes(original_data),
        "creditrisk_original_source.csv",
        "text/csv"
    )


# ============================================================
# 17. SESSION CONTRACT FOR DOWNSTREAM PAGES
# ============================================================

st.session_state["credit_data_original"] = (
    original_data.copy()
)

st.session_state["credit_data_standardized"] = (
    standard_data.copy()
)

st.session_state["credit_data"] = (
    modelling_data.copy()
)

st.session_state["variable_recognition"] = (
    register.copy()
)

st.session_state["variable_dictionary"] = (
    get_active_dictionary()
)

st.session_state["custom_variable_aliases"] = (
    st.session_state["custom_variable_aliases"].copy()
)


# ============================================================
# 18. MODELLING HAND-OFF
# ============================================================

st.divider()

st.subheader("8. Modelling hand-off")

h1, h2, h3 = st.columns(3)

h1.metric(
    "Approved modelling variables",
    modelling_data.shape[1]
)

h2.metric(
    "Pending review",
    pending_count + unmapped_count
)

h3.metric(
    "Discarded variables",
    discarded_count
)

st.write(
    f"**Modelling dataset:** {modelling_data.shape[0]:,} observations "
    f"and {modelling_data.shape[1]:,} approved variables."
)

st.write(
    f"**Standardised dataset:** {standard_data.shape[0]:,} observations "
    f"and {standard_data.shape[1]:,} source variables."
)

if pending_count or unmapped_count:

    st.warning(
        "Some variables remain unresolved. They are excluded from "
        "the modelling dataset until a governance decision is made."
    )

else:

    st.success(
        "All source variables have received a governance decision."
    )


st.code(
    'source_data = st.session_state["credit_data"].copy()\n'
    'original_data = st.session_state["credit_data_original"].copy()\n'
    'standardised_data = st.session_state["credit_data_standardized"].copy()\n'
    'mapping_register = st.session_state["variable_recognition"].copy()',
    language="python"
)


st.caption(
    "Governance note: discarded variables remain in the original "
    "source dataset. Pending and discarded variables are excluded "
    "from the modelling hand-off. A mapping decision does not certify "
    "data accuracy, model suitability, regulatory compliance or "
    "production approval."
)