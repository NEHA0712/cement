import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

# Load dataset
df = pd.read_csv("yeh-concret-data.csv")

# Split features & target
X = df.drop("csMPa", axis=1)
y = df["csMPa"]

# Save feature names so Streamlit matches them
FEATURE_NAMES = X.columns.tolist()
pickle.dump(FEATURE_NAMES, open("features.pkl", "wb"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("🎉 Training complete! Saved: model.pkl & features.pkl")
