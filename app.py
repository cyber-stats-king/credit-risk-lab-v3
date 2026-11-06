import streamlit as st

st.set_page_config(
    page_title="CreditRisk Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# RISKLAB DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --ink:#eaf2fb;
    --muted:#9eb2c8;
    --navy:#081321;
    --panel:#101f32;
    --panel2:#142941;
    --line:#29445e;
    --teal:#42e3d0;
    --gold:#e4bd78;
    --violet:#988cff;
}

html, body, [class*="css"] {
    font-family:'DM Sans',sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 12% 0%,#17344b 0%,transparent 34%),
        linear-gradient(135deg,#07111f 0%,#0b192a 52%,#10243a 100%);
    color:var(--ink);
}

[data-testid="stHeader"] {
    background:rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#07111e,#0d1d30 70%,#10263a);
    border-right:1px solid #29445e;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color:#b7c8d9;
}

.block-container {
    padding-top:1.5rem;
    padding-bottom:3rem;
    max-width:1580px;
}

h1,h2,h3,h4 {
    color:#f3f7fc !important;
    font-family:'Manrope',sans-serif !important;
    letter-spacing:-.025em;
}

p, label, .stMarkdown {
    color:#d5e1ed;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background:linear-gradient(145deg,rgba(24,48,71,.96),rgba(12,28,45,.98));
    border:1px solid #34536c;
    padding:17px 18px;
    border-radius:17px;
    box-shadow:0 10px 28px rgba(0,0,0,.16);
}

[data-testid="stMetricLabel"] {
    color:#a9bfd2 !important;
    font-size:.82rem;
}

[data-testid="stMetricValue"] {
    color:#f5fbff !important;
    font-family:'Manrope',sans-serif;
}

/* HERO BANNERS */

.hero {
    position:relative;
    overflow:hidden;
    background:
        radial-gradient(circle at 88% 12%,rgba(66,227,208,.16),transparent 28%),
        radial-gradient(circle at 8% 100%,rgba(152,140,255,.13),transparent 32%),
        linear-gradient(115deg,#10243a,#122e45 58%,#10283c);
    border:1px solid #3a627b;
    border-radius:25px;
    padding:34px 38px;
    margin:0 0 25px 0;
    box-shadow:0 18px 48px rgba(0,0,0,.22),
               inset 0 1px 0 rgba(255,255,255,.06);
}

.hero:after {
    content:'';
    position:absolute;
    right:-70px;
    top:-125px;
    width:330px;
    height:330px;
    border:1px solid rgba(66,227,208,.15);
    border-radius:50%;
    box-shadow:0 0 0 28px rgba(66,227,208,.025),
               0 0 0 58px rgba(66,227,208,.02);
    pointer-events:none;
}

.hero.centered {
    text-align:center;
    padding:48px 42px 46px;
}

.hero.centered .hero-sub {
    margin:12px auto 0;
}

.hero-kicker {
    color:var(--gold);
    font-size:11px;
    letter-spacing:3.2px;
    font-weight:700;
    text-transform:uppercase;
}

.hero-kicker .kicker-gold {
    color:var(--gold);
}

.hero-kicker .kicker-teal {
    color:var(--teal);
}

.hero-title {
    font-family:'Manrope',sans-serif;
    font-size:clamp(30px,4vw,48px);
    line-height:1.12;
    font-weight:800;
    margin:12px 0 8px;
    color:#f5fbff;
}

.hero-sub {
    color:#bfd0df;
    font-size:16px;
    line-height:1.65;
    max-width:850px;
}

.hero-rule {
    width:76px;
    height:3px;
    border-radius:5px;
    background:linear-gradient(90deg,var(--teal),var(--gold));
    margin:18px 0 0;
}

.centered .hero-rule {
    margin:20px auto 0;
}

/* SECTION PANELS */

.panel {
    background:linear-gradient(145deg,rgba(20,41,62,.94),rgba(11,27,44,.97));
    border:1px solid #2b4963;
    border-radius:19px;
    padding:20px 22px;
    box-shadow:0 12px 30px rgba(0,0,0,.15),
               inset 0 1px 0 rgba(255,255,255,.035);
}

.section-heading {
    display:flex;
    align-items:center;
    gap:10px;
    color:#f1f7fc;
    font-family:'Manrope',sans-serif;
    font-size:1.05rem;
    font-weight:750;
    margin:8px 0 14px;
}

.section-heading:before {
    content:'';
    display:inline-block;
    width:4px;
    height:21px;
    border-radius:4px;
    background:linear-gradient(180deg,var(--teal),var(--gold));
}

.small-muted {
    color:var(--muted);
    font-size:13px;
}

/* TABLES */

div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    border:1px solid #36536b;
    border-radius:13px;
    overflow:hidden;
    background:#0e1d2d;
    box-shadow:0 8px 22px rgba(0,0,0,.14);
}

div[data-testid="stDataFrame"] [role="columnheader"],
div[data-testid="stTable"] thead tr th {
    background:#1a354d !important;
    color:#eaf5fb !important;
    font-weight:700 !important;
    border-bottom:1px solid #42617a !important;
}

div[data-testid="stDataFrame"] [role="gridcell"],
div[data-testid="stTable"] tbody td {
    color:#dce8f2 !important;
    border-bottom:1px solid rgba(74,105,130,.24) !important;
}

/* TABS */

[data-testid="stTabs"] button {
    color:#a9bfd2;
    font-weight:650;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color:var(--teal) !important;
    border-bottom-color:var(--teal) !important;
}

/* BUTTONS */

.stButton>button,
.stDownloadButton>button {
    border:1px solid #3b7c87;
    border-radius:11px;
    background:linear-gradient(110deg,#167e83,#176a7d);
    color:#f5ffff;
    font-weight:700;
    padding:.58rem 1rem;
    box-shadow:0 6px 18px rgba(0,0,0,.16);
}

.stButton>button:hover,
.stDownloadButton>button:hover {
    border-color:#6ff4e5;
    color:white;
    transform:translateY(-1px);
}

/* EXPANDERS AND UPLOADERS */

[data-testid="stExpander"] {
    border:1px solid #2d4b65;
    border-radius:14px;
    background:rgba(15,33,51,.76);
}

[data-testid="stFileUploader"] {
    border:1px dashed #45647d;
    border-radius:14px;
    padding:10px;
    background:rgba(16,35,54,.55);
}

/* HIDE STREAMLIT AUTOMATIC NAVIGATION */

[data-testid="stSidebarNav"] {
    display:none !important;
}

/* ============================================================
   WORKSPACE NAVIGATION - INDIVIDUAL MODERN CARDS
   ============================================================ */

/* Navigation container */
section[data-testid="stSidebar"]
div[data-testid="stRadio"] > div {
    display:flex;
    flex-direction:column;
    gap:8px;
}

/* Individual workspace cards */
section[data-testid="stSidebar"]
div[data-testid="stRadio"] label {
    display:flex;
    align-items:center;
    background:linear-gradient(
        115deg,
        rgba(24,45,66,.82),
        rgba(14,31,49,.88)
    );
    border:1px solid rgba(105,139,164,.20);
    border-radius:12px;
    padding:11px 13px;
    margin:0;
    min-height:43px;
    transition:all .22s ease;
    cursor:pointer;
    box-shadow:0 4px 12px rgba(0,0,0,.10);
}

/* Hide default radio circles */
section[data-testid="stSidebar"]
div[data-testid="stRadio"] label > div:first-child {
    display:none;
}

/* Workspace text */
section[data-testid="stSidebar"]
div[data-testid="stRadio"] label p {
    color:#c2d2e0;
    font-family:'DM Sans',sans-serif;
    font-size:13px;
    font-weight:600;
    margin:0;
    line-height:1.4;
    transition:all .2s ease;
}

/* Hover effect */
section[data-testid="stSidebar"]
div[data-testid="stRadio"] label:hover {
    background:linear-gradient(
        110deg,
        rgba(27,77,88,.65),
        rgba(18,48,66,.95)
    );
    border-color:rgba(66,227,208,.55);
    transform:translateX(3px);
    box-shadow:0 5px 16px rgba(0,0,0,.18);
}

section[data-testid="stSidebar"]
div[data-testid="stRadio"] label:hover p {
    color:#ffffff;
}

/* Selected workspace card */
section[data-testid="stSidebar"]
div[data-testid="stRadio"] label:has(input:checked) {
    background:
        linear-gradient(
            110deg,
            rgba(31,119,119,.34),
            rgba(20,49,68,.96)
        );
    border:1px solid rgba(66,227,208,.72);
    border-left:3px solid #42e3d0;
    box-shadow:
        0 5px 18px rgba(0,0,0,.20),
        inset 0 1px 0 rgba(255,255,255,.06);
}

/* Selected workspace text */
section[data-testid="stSidebar"]
div[data-testid="stRadio"] label:has(input:checked) p {
    color:#65f1df;
    font-weight:750;
}

/* Keyboard focus */
section[data-testid="stSidebar"]
div[data-testid="stRadio"] label:focus-within {
    outline:2px solid rgba(66,227,208,.35);
    outline-offset:2px;
}

/* SIDEBAR WORKSPACE LABEL */
section[data-testid="stSidebar"]
div[data-testid="stRadio"] > label {
    display:none;
}

hr {
    border-color:#29445e !important;
}

/* ============================================================
   GLOBAL MULTISELECT SELECTED-TAG DESIGN
   Applies to candidate predictors and final WOE predictors
   ============================================================ */

/* White background for every selected multiselect tag */
[data-testid="stMultiSelect"] [data-baseweb="tag"],
[data-testid="stMultiSelect"] [data-baseweb="tag"] > div,
[data-testid="stMultiSelect"] [data-baseweb="tag"] > span {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #2563EB !important;
    opacity: 1 !important;
}

/* Force predictor names to use professional blue text */
[data-testid="stMultiSelect"] [data-baseweb="tag"] *,
[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
    color: #2563EB !important;
}

/* Blue border around each selected variable */
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    border: 1px solid #93C5FD !important;
    border-radius: 7px !important;
    box-shadow: none !important;
}

/* Blue remove icon */
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
    fill: #2563EB !important;
    color: #2563EB !important;
    opacity: 1 !important;
}

/* Subtle hover effect */
[data-testid="stMultiSelect"] [data-baseweb="tag"]:hover,
[data-testid="stMultiSelect"] [data-baseweb="tag"]:hover > div {
    background: #EFF6FF !important;
    background-color: #EFF6FF !important;
}

with st.form("pd_predictor_selection_form"):
    selected_final_features = st.multiselect(
        "Select final WOE predictors",
        options=available_woe_features,
        default=previous_selection,
        format_func=format_woe_feature,
        help="The coloured marker and IV value show the Information Value band for each predictor."
    )

    submit_selection = st.form_submit_button(
        "Confirm predictor selection",
        type="primary"
    )
</style>
""", unsafe_allow_html=True)


# ============================================================
# HOME PAGE
# ============================================================

def home():

    st.html("""
    <div class="hero centered">

        <div class="hero-kicker">
            INNOVATE · ANALYSE · STRESS TEST · OPTIMISE
        </div>

        <div class="hero-title">
            CreditRisk Lab
        </div>

        <div class="hero-sub"
             style="font-size:19px;color:#e0ebf4;font-weight:600;">
            Advanced Credit Risk Modelling, Analytics and Regulatory
            Risk Measurement
        </div>

        <div class="hero-sub">
            An interactive environment for developing, evaluating
            and applying credit risk models.
        </div>

        <div class="hero-rule"></div>

    </div>
    """)

    st.html("""
    <div class="section-heading">
        Credit risk modelling workspaces
    </div>
    """)

    cards = [
        (
            "🗄️",
            "Data & Models",
            "Simulate, upload, inspect and prepare portfolio data and modelling inputs."
        ),
        (
            "📊",
            "PD Modelling",
            "Explore probability of default concepts, horizons, drivers and model outputs."
        ),
        (
            "🧮",
            "PD Model Development",
            "Generate development data, perform WoE binning and IV analysis, and develop PD models."
        ),
        (
            "📐",
            "EAD Model Development",
            "Develop exposure at default and credit conversion factor models."
        ),
        (
            "🛡️",
            "LGD Model Development",
            "Model recoveries, collateral, workout costs and loss severity."
        ),
        (
            "📘",
            "ECL Calculation & Interpretation",
            "Calculate and interpret expected credit losses and IFRS 9 staging."
        ),
        (
            "🔗",
            "Integrated Credit Risk Tool",
            "Bring PD, LGD and EAD together in an integrated credit risk workflow."
        )
    ]

    for start in range(0, len(cards), 3):

        cols = st.columns(3)

        for col, (icon, title, description) in zip(
            cols,
            cards[start:start + 3]
        ):

            with col:

                st.html(f"""
                <div class="panel"
                     style="min-height:165px;margin-bottom:16px;">

                    <div style="font-size:25px;margin-bottom:10px;">
                        {icon}
                    </div>

                    <div style="
                        font-family:Manrope,sans-serif;
                        font-size:17px;
                        font-weight:750;
                        color:#f3f7fc;">
                        {title}
                    </div>

                    <div style="
                        color:#b9cad9;
                        font-size:13px;
                        line-height:1.6;
                        margin-top:8px;">
                        {description}
                    </div>

                </div>
                """)

    st.info(
        "Choose a workspace from the RISKLAB navigation menu to begin."
    )


# ============================================================
# PAGE REGISTRATION
# ============================================================

page_definitions = [

    ("Home", "🏠", home),

    ("Data & Models", "🗄️", "pages/1_Data_and_Models.py"),

    ("PD Modelling", "📊", "pages/2_PD_Modelling.py"),

    ("PD Model Development", "🧮", "pages/3_PD_Model_Development.py"),

    ("EAD Model Development", "📐", "pages/4_EAD_Model_Development.py"),

    ("LGD Model Development", "🛡️", "pages/5_LGD_Model_Development.py"),

    ("ECL Calculation & Interpretation", "📘",
     "pages/6_ECL_Calculation_and_Interpretation.py"),

    ("Integrated Credit Risk Tool", "🔗",
     "pages/7_Integrated_Credit_Risk_Tool.py"),

]


# ============================================================
# REGISTER STREAMLIT PAGES
# ============================================================

streamlit_pages = []

for title, icon, page in page_definitions:

    streamlit_pages.append(
        st.Page(
            page,
            title=title,
            icon=icon,
            default=(title == "Home")
        )
    )


# ============================================================
# STREAMLIT NAVIGATION
# ============================================================

current_page = st.navigation(
    streamlit_pages,
    position="hidden"
)

current_title = current_page.title


# ============================================================
# CUSTOM RISKLAB NAVIGATION
# ============================================================

page_lookup = {
    title: registered_page
    for (title, icon, original_page), registered_page
    in zip(page_definitions, streamlit_pages)
}

page_titles = [
    title for title, icon, page in page_definitions
]

with st.sidebar:

    st.html("""
    <div style="
        font-family:Manrope,sans-serif;
        font-size:25px;
        font-weight:800;
        letter-spacing:-1px;
        color:#f3f7fc;
        margin-bottom:2px;">
        ◈ RISKLAB
    </div>

    <div style="
        font-size:10px;
        letter-spacing:2.5px;
        font-weight:700;
        color:#42e3d0;">
        CREDIT RISK INTELLIGENCE
    </div>
    """)

    st.divider()

    st.html("""
    <div style="
        font-size:10px;
        letter-spacing:2px;
        font-weight:700;
        color:#9eb2c8;
        margin-bottom:8px;">
        WORKSPACE
    </div>
    """)

    selected_title = st.radio(
        "WORKSPACE",
        page_titles,
        index=(
            page_titles.index(current_title)
            if current_title in page_titles
            else 0
        ),
        label_visibility="collapsed",
        key="risklab_workspace"
    )

    # Navigate to the selected registered Streamlit page.
    if selected_title != current_title:

        st.switch_page(page_lookup[selected_title])

    st.divider()

    st.html("""
    <div style="
        font-size:10px;
        letter-spacing:2px;
        font-weight:700;
        color:#9eb2c8;">
        ENVIRONMENT
    </div>

    <div style="
        font-size:13px;
        color:#d5e1ed;
        margin-top:8px;">
        CreditRisk Lab
    </div>

    <div style="
        font-size:11px;
        color:#9eb2c8;
        line-height:1.8;
        margin-top:5px;">
        PD · LGD · EAD · IFRS 9
    </div>

    <div style="
        font-size:10px;
        color:#42e3d0;
        letter-spacing:1.5px;
        margin-top:15px;">
        RISKLAB MODELLING WORKSPACE
    </div>
    """)


# ============================================================
# WORKSPACE HEADERS
# ============================================================

page_headers = {

    "Data & Models": (
        """
        <span class="kicker-gold">DATA MANAGEMENT</span>
        <span class="kicker-teal"> · DATA GOVERNANCE</span>
        <span class="kicker-gold"> · PORTFOLIO INGESTION</span>
        <span class="kicker-teal"> · VARIABLE CONTROL</span>
        """,
        "Data & Model Inventory",
        "Inspect portfolio inputs, simulation settings, datasets and model components."
    ),

    "PD Modelling": (
        "RISK QUANTIFICATION · PD",
        "Probability of Default",
        "Explore borrower default risk across the 12-month and lifetime horizons."
    ),

    "PD Model Development": (
        "MODEL DEVELOPMENT · PD",
        "PD Model Development",
        "Prepare development data, perform WoE analysis, estimate and evaluate predictive PD models."
    ),

    "EAD Model Development": (
        "EXPOSURE ANALYTICS · EAD",
        "EAD Model Development",
        "Develop exposure at default and credit conversion factor models."
    ),

    "LGD Model Development": (
        "RECOVERY ANALYTICS · LGD",
        "LGD Model Development",
        "Develop recovery and loss-severity models using collateral and workout information."
    ),

    "ECL Calculation & Interpretation": (
        "IFRS 9 · IMPAIRMENT",
        "ECL Calculation & Interpretation",
        "Calculate expected credit losses and interpret staging, SICR and measurement outputs."
    ),

    "Integrated Credit Risk Tool": (
        "END-TO-END RISK",
        "Integrated Credit Risk Tool",
        "Integrate PD, LGD and EAD outputs into a unified credit risk workflow."
    )

}


# ============================================================
# DISPLAY PAGE HERO
# ============================================================

if current_title != "Home":

    kicker, title, description = page_headers[current_title]

    st.html(f"""
    <div class="hero">

        <div class="hero-kicker">
            {kicker}
        </div>

        <div class="hero-title">
            {title}
        </div>

        <div class="hero-sub">
            {description}
        </div>

        <div class="hero-rule"></div>

    </div>
    """)


# ============================================================
# RUN SELECTED PAGE
# ============================================================

current_page.run()