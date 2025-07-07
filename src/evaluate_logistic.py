import joblib
from sklearn.metrics import classification_report, roc_auc_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_cleaning import load_and_clean_data

model = joblib.load("models/logistic_model.pkl")
X_train, X_test, y_train, y_test = load_and_clean_data("data/cs-training.csv")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("AUC Score:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))
