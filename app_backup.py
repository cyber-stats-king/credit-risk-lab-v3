
import streamlit as st

st.set_page_config(
    page_title="CreditRisk Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Shared application theme
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #071426 0%, #0B1F3A 55%, #102B50 100%);
    color: #EAF2FF;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #08172B 0%, #102747 100%);
    border-right: 1px solid #1D4B7A;
}

section[data-testid="stSidebar"] * {
    color: #DCEBFF;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
h1, h2, h3 {
    color: #F1F7FF !important;
}

/* Paragraphs */
p, label {
    color: #C5D7EF;
}

/* Hero banner */
.hero {
    background: linear-gradient(115deg, #102B50, #145A9E, #087EAD);
    border: 1px solid #258BD1;
    border-radius: 20px;
    padding: 42px;
    margin-bottom: 30px;
    box-shadow: 0 10px 35px rgba(0, 100, 200, 0.16);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 44px;
    font-weight: 750;
    margin-bottom: 12px;
}

.hero p {
    color: #DCEEFF;
    font-size: 17px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #102744;
    border: 1px solid #205080;
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetricValue"] {
    color: #59C3FF;
}

/* General cards */
.section-card {
    background: linear-gradient(145deg, #102744, #15365B);
    border: 1px solid #285A88;
    border-radius: 16px;
    padding: 23px;
    min-height: 175px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #1266B3, #168CCB);
    color: white;
    border: 1px solid #39A8E8;
    border-radius: 9px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #168CCB, #26A8E8);
    border-color: #70CEFF;
    color: white;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid #285080;
    border-radius: 10px;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #BBD6F2;
}

/* Dividers */
hr {
    border-color: #24476D;
}

</style>
""", unsafe_allow_html=True)


# Home page content
def home():

    st.markdown("""
    <div class="hero">
        <h1>CreditRisk Lab</h1>
        <p>
            Advanced Credit Risk Modelling, Analytics and Regulatory
            Risk Measurement
        </p>
        <p>
            An interactive environment for developing, evaluating
            and applying credit risk models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.title("Welcome to CreditRisk Lab")

    st.write("""
    CreditRisk Lab is a practical credit risk modelling environment
    bringing together statistical modelling, portfolio analytics,
    model validation and regulatory frameworks.
    """)

    st.subheader("Core modelling components")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>📊 Probability of Default</h3>
            <p>Develop predictive PD models using optimal binning,
            Weight of Evidence and logistic regression.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>💰 Exposure at Default</h3>
            <p>Estimate exposure at default and model credit
            conversion factors.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="section-card">
            <h3>🛡️ Loss Given Default</h3>
            <p>Estimate loss severity after recoveries,
            collateral and collection costs.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Beyond individual models")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class="section-card">
            <h3>🔗 Integrated Risk Engine</h3>
            <p>Combine PD, LGD and EAD into a unified credit
            risk measurement framework.</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class="section-card">
            <h3>📘 IFRS 9 & Basel</h3>
            <p>Explore expected credit loss, staging,
            regulatory capital and risk measurement.</p>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class="section-card">
            <h3>⚡ Stress Testing</h3>
            <p>Assess portfolio resilience under adverse
            economic and credit scenarios.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.info(
        "Start by opening Data & Models from the navigation menu."
    )


# Explicit navigation: Home and the individual modelling pages
pages = [
    st.Page(home, title="Home", icon="🏠", default=True),

    st.Page(
        "pages/1_Data_and_Models.py",
        title="Data & Models",
        icon="🗄️"
    ),

    st.Page(
        "pages/2_PD_Modelling.py",
        title="PD Modelling",
        icon="📊"
    ),

st.Page(
    "pages/3_PD_Model_Development.py",
    title="PD Model Development",
    icon="🧮"
),

st.Page(
    "pages/4_EAD_Model_Development.py",
    title="EAD Model Development",
    icon="🧮",
),

st.Page(
    "pages/5_LGD_Model_Development.py",
    title="LGD Model Development",
    icon="🧮",
),

st.Page(
    "pages/6_ECL_Calculation_and_Interpretation.py",
    title="ECL Calculation & Interpretation",
    icon="📊",
),

st.Page(
    "pages/7_Integrated_Credit_Risk_Tool.py",
    title="Integrated Credit Risk Tool",
    icon="📊",
),
]

navigation = st.navigation(pages)
navigation.run()