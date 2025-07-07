import pandas as pd
from sklearn.model_selection import train_test_split
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=["Unnamed: 0"])
    df = df.rename(columns={"SeriousDlqin2yrs": "default"})
    df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)
    df["NumberOfDependents"].fillna(0, inplace=True)
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1)

    X = df.drop("default", axis=1)
    y = df["default"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



