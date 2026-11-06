import pandas as pd

from engine.portfolio_manager import create_portfolio_manager


# Create a small test portfolio.
portfolio = pd.DataFrame({
    "loan_id": [101, 102, 103],
    "loan_amount": [10000, 15000, 20000],
    "current_balance": [8000, 12000, 16000]
})

# Initialise the manager.
manager = create_portfolio_manager(portfolio)

# Create PD predictions.
pd_predictions = pd.DataFrame({
    "loan_id": [101, 102, 103],
    "predicted_pd": [0.02, 0.05, 0.08]
})

# Merge predictions.
manager.merge_predictions(
    pd_predictions,
    model_type="PD"
)

# Create EAD predictions.
ead_predictions = pd.DataFrame({
    "loan_id": [101, 102, 103],
    "predicted_ccf": [0.35, 0.45, 0.55],
    "predicted_ead": [8500, 13000, 17500]
})

manager.merge_predictions(
    ead_predictions,
    model_type="EAD"
)

# Display the shared portfolio.
print("\nIntegrated portfolio:")
print(manager.get_prediction_portfolio())

# Display prediction status.
print("\nPrediction status:")
print(manager.prediction_status())

# Confirm original portfolio was not modified.
print("\nOriginal portfolio:")
print(manager.get_original_portfolio())

print("\nPortfolio manager test completed successfully.")