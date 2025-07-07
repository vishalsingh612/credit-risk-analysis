import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv("data/cs-training.csv")
df = df.drop(columns=["Unnamed: 0"])
df = df.rename(columns={"SeriousDlqin2yrs": "default"})
df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)
df["NumberOfDependents"].fillna(0, inplace=True)
df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1)

# Create reports directory if it doesn't exist
os.makedirs("reports/plots", exist_ok=True)

# 1. Class Balance Plot
plt.figure(figsize=(6, 4))
sns.countplot(x='default', data=df)
plt.title("Class Distribution (Default vs No Default)")
plt.savefig("reports/plots/class_distribution.png")
plt.close()

# 2. Feature Distributions by Class
features = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans'
]

for feature in features:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df[df['default'] == 0], x=feature, label="No Default", fill=True)
    sns.kdeplot(data=df[df['default'] == 1], x=feature, label="Default", fill=True)
    plt.title(f"Distribution of {feature} by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/plots/distribution_{feature}.png")
    plt.close()

# 3. Boxplots
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='default', y=feature, data=df)
    plt.title(f"{feature} vs Default Status")
    plt.tight_layout()
    plt.savefig(f"reports/plots/boxplot_{feature}.png")
    plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("reports/plots/correlation_heatmap.png")
plt.close()

print("Exploratory plots saved to reports/plots/")
