import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 1. Generate data
np.random.seed(42)
n = 5000

df = pd.DataFrame({
    "age": np.random.randint(21, 65, n),
    "income": np.random.randint(20000, 150000, n),
    "loan_amount": np.random.randint(5000, 50000, n),
    "credit_score": np.random.randint(300, 850, n),
    "years_employed": np.random.randint(0, 20, n),
    "missed_payments": np.random.randint(0, 10, n)
})

df["default"] = (
    (df["credit_score"] < 600).astype(int) +
    (df["missed_payments"] > 3).astype(int) +
    (df["loan_amount"] > 30000).astype(int)
)
df["default"] = (df["default"] >= 2).astype(int)

# 2. Features and target
X = df.drop("default", axis=1)
y = df["default"]

# 3. Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)

# 6. Save fresh pkl files
joblib.dump(xgb_model, "credit_risk_xgb.pkl")
joblib.dump(scaler, "credit_risk_scaler.pkl")

print("Done. Fresh pkl files saved.")
print("credit_risk_xgb.pkl")
print("credit_risk_scaler.pkl")
