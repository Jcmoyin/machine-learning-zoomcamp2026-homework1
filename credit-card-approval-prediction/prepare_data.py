
import pandas as pd

# 1. Load both datasets
application = pd.read_csv("data/application_record.csv.zip")

credit = pd.read_csv(
    "data/credit_record.csv.zip",
    dtype={"STATUS": str}
)

print("Application dataset:", application.shape)
print("Credit dataset:", credit.shape)

# 2. Remove repeated applicant IDs
application = application.drop_duplicates(
    subset="ID",
    keep="first"
)

# 3. Create credit risk target
# 1 = At least one payment 30+ days overdue
# 0 = No payment 30+ days overdue

bad_statuses = ["1", "2", "3", "4", "5"]

credit_target = (
    credit.groupby("ID")["STATUS"]
    .apply(lambda x: x.isin(bad_statuses).any())
    .astype(int)
    .reset_index(name="credit_risk")
)

# 4. Merge both datasets using applicant ID
df = application.merge(
    credit_target,
    on="ID",
    how="inner"
)

# 5. Remove ID because it is not a predictive feature
df = df.drop(columns=["ID"])

# 6. Save the processed dataset
df.to_csv("data/credit_approval_dataset.csv", index=False)

# 7. Display results
print("\nMerged dataset shape:", df.shape)

print("\nTarget distribution:")
print(df["credit_risk"].value_counts())

print("\nTarget percentages:")
print(df["credit_risk"].value_counts(normalize=True) * 100)

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

print("\nDataset successfully prepared!")
