import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the bank marketing dataset
df = pd.read_csv("bank-full.csv", sep=";")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Count customers by subscription status
print("\nSubscription Status:")
print(df["y"].value_counts())

# Calculate subscription percentages
print("\nSubscription Percentage:")
print(df["y"].value_counts(normalize=True) * 100)

# Plot subscription status
sns.countplot(data=df, x="y")

plt.title("Bank Customer Subscription Status")
plt.xlabel("Subscribed (No / Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("subscription_status.png")
plt.show()


# Subscription rate by job
job_subscription = (
    df.groupby("job")["y"]
    .apply(lambda x: (x == "yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\nSubscription Rate by Job (%):")
print(job_subscription.round(2))

# Plot job-wise subscription rate
job_subscription.plot(kind="bar", figsize=(10, 5))

plt.title("Subscription Rate by Job")
plt.xlabel("Job")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig("job_subscription.png")
plt.show()


# Compare average age by subscription status
age_analysis = df.groupby("y")["age"].mean()

print("\nAverage Age by Subscription Status:")
print(age_analysis.round(2))

# Plot average age
age_analysis.plot(kind="bar")

plt.title("Average Age: Subscribed vs Not Subscribed")
plt.xlabel("Subscription (No / Yes)")
plt.ylabel("Average Age")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("average_age.png")
plt.show()