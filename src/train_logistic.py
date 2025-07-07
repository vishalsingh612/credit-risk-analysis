from sklearn.linear_model import LogisticRegression
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_cleaning import load_and_clean_data

X_train, X_test, y_train, y_test = load_and_clean_data("data/cs-training.csv")

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

joblib.dump(model, "models/logistic_model.pkl")
