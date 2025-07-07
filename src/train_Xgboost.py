import joblib
from xgboost import XGBClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_cleaning import load_and_clean_data

# Load the data
X_train, X_test, y_train, y_test = load_and_clean_data("data/cs-training.csv")

# Compute scale_pos_weight manually (for imbalance handling)
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Train XGBoost model
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Save the model
joblib.dump(xgb_model, "models/xgboost_model.pkl")
print("✅ XGBoost model saved.")
