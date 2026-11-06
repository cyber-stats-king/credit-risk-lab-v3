# CreditRisk Lab

A new Streamlit playground for demonstrative credit risk modelling.

## Run on Windows
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Included
- Synthetic portfolio generator
- PD, LGD and EAD demonstrators
- Integrated illustrative ECL / RWA engine
- Scenario controls and stress testing
- IFRS 9 vs Basel comparison
- T0 reconciliation inputs
- Learning and governance pages

**Important:** all formulas and data are illustrative. This is not a validated regulatory model. Basel calculations are proxies and must not be used for regulatory reporting.
