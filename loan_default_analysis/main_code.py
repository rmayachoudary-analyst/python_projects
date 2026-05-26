import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("../data/loan_data.csv")

print("==== FIRST 5 ROWS ====")
print(df.head())

# ==========================
# Basic Statistics
# ==========================
print("\n==== SUMMARY ====")
print(df.describe())

# ==========================
# KPI 1: Default Rate
# ==========================
default_rate = (
    df["Default"].value_counts(normalize=True) * 100
)

print("\n==== DEFAULT RATE (%) ====")
print(default_rate)

# ==========================
# KPI 2: Avg Income by Default
# ==========================
income_analysis = df.groupby("Default")["Income"].mean()

print("\n==== AVG INCOME ====")
print(income_analysis)

# ==========================
# KPI 3: Avg Credit Score
# ==========================
credit_analysis = df.groupby("Default")["Credit_Score"].mean()

print("\n==== CREDIT SCORE ANALYSIS ====")
print(credit_analysis)

# ==========================
# Feature Engineering
# EMI Ratio
# ==========================
df["EMI_Ratio"] = df["EMI"] / df["Income"]

print("\n==== EMI RATIO ====")
print(df[["Customer_ID", "EMI_Ratio"]])

# ==========================
# Risk Segmentation
# ==========================
def risk_segment(score):
    if score >= 750:
        return "Low Risk"
    elif score >= 650:
        return "Medium Risk"
    else:
        return "High Risk"

df["Risk_Category"] = df["Credit_Score"].apply(risk_segment)

print("\n==== RISK CATEGORY ====")
print(df[["Customer_ID", "Credit_Score", "Risk_Category"]])

# ==========================
# Visualization 1
# Default Count
# ==========================
df["Default"].value_counts().plot(
    kind="bar",
    title="Loan Default Distribution"
)

plt.xlabel("Default Status")
plt.ylabel("Count")
plt.show()

# ==========================
# Visualization 2
# Credit Score vs Loan Amount
# ==========================
plt.scatter(df["Credit_Score"], df["Loan_Amount"])

plt.xlabel("Credit Score")
plt.ylabel("Loan Amount")
plt.title("Credit Score vs Loan Amount")

plt.show()

# ==========================
# Save Processed Data
# ==========================
df.to_csv("../data/processed_loan_data.csv", index=False)

print("\nProcessed dataset saved!")
