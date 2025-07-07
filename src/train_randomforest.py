import joblib
from sklearn.ensemble import RandomForestClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_cleaning import load_and_clean_data

# Load train/test splits
X_train, X_test, y_train, y_test = load_and_clean_data("data/cs-training.csv")

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, "models/random_forest_model.pkl")
print("✅ Random Forest model saved.")
