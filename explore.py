import pandas as pd

df = pd.read_csv("data/ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv")

# --- Overview ---
total = len(df)
failures = df["Machine failure"].sum()

print("=== Overview ===")
print(f"Total rows:   {total}")
print(f"Failures:     {failures}")
print(f"Failure rate: {failures / total * 100:.2f}%")

# --- Failures by product type (L, M, H) ---
by_type = df.groupby("Type")["Machine failure"].agg(["count", "sum", "mean"])
by_type.columns = ["Rows", "Failures", "Failure rate %"]
by_type["Failure rate %"] = (by_type["Failure rate %"] * 100).round(2)

print()
print("=== Failures by product type ===")
print(by_type)

# --- Failures by failure type ---
names = {
    "TWF": "Tool wear",
    "HDF": "Heat dissipation",
    "PWF": "Power",
    "OSF": "Overstrain",
    "RNF": "Random",
}
counts = df[list(names)].sum()

by_failure = pd.DataFrame({
    "Meaning": names,
    "Count": counts,
    "% of all rows": (counts / total * 100).round(2),
})

print()
print("=== Failures by failure type ===")
print(by_failure)
print(f"Sum of the five types: {counts.sum()}")
