from sklearn.linear_model import LogisticRegression
import pandas as pd

from engine.model_registry import ModelRegistry


# ------------------------------------------------
# 1. Create a small example dataset
# ------------------------------------------------

X = pd.DataFrame({
    "income": [25000, 35000, 45000, 55000, 65000, 75000],
    "credit_score": [520, 580, 610, 680, 710, 760],
})

y = [1, 1, 1, 0, 0, 0]


# ------------------------------------------------
# 2. Fit an example PD model
# ------------------------------------------------

model = LogisticRegression()

model.fit(X, y)


# ------------------------------------------------
# 3. Register the fitted model
# ------------------------------------------------

registry = ModelRegistry()

registered = registry.register_model(
    model_type="PD",
    model=model,
    model_name="logistic_regression",
    algorithm="Logistic Regression",
    target="default",
    features=["income", "credit_score"],
    metrics={
        "accuracy": 0.85,
        "roc_auc": 0.90,
    },
    training_observations=len(X),
)

print("\nRegistered model:")
print(registered)


# ------------------------------------------------
# 4. Retrieve the active model
# ------------------------------------------------

loaded_model = registry.load_model("PD")


# ------------------------------------------------
# 5. Generate predictions
# ------------------------------------------------

probabilities = loaded_model.predict_proba(X)[:, 1]

print("\nPredicted default probabilities:")
print(probabilities)


# ------------------------------------------------
# 6. Display registry summary
# ------------------------------------------------

print("\nRegistry summary:")

for item in registry.registry_summary():
    print(item)


print("\nModel registry test completed successfully.")