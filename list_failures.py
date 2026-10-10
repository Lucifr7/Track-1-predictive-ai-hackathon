import pandas as pd

df = pd.read_csv("data/ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv")

type_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# How many failure types are ticked in each row (0, 1, 2 or 3)
df["Types ticked"] = df[type_cols].sum(axis=1)


# Put every row into a group
def label(row):
    failed = row["Machine failure"] == 1
    ticked = row["Types ticked"]
    if failed and ticked == 1:
        return "1. Normal: one cause"
    if failed and ticked >= 2:
        return "2. Odd: several causes"
    if failed and ticked == 0:
        return "3. Odd: failed, no cause given"
    if not failed and ticked >= 1:
        return "4. Odd: cause ticked, but not a failure"
    return "No failure"


df["Group"] = df.apply(label, axis=1)

# Keep only the interesting rows, normal ones first, odd ones at the bottom
rows = df[df["Group"] != "No failure"].sort_values(["Group", "UDI"])

print("=== How many rows in each group ===")
print(rows["Group"].value_counts().sort_index())

show_cols = ["UDI", "Type", "Machine failure"] + type_cols + ["Group"]

print()
print("=== The odd rows ===")
odd = rows[rows["Group"] != "1. Normal: one cause"]
print(odd[show_cols].to_string(index=False))

# Save the full list (normal rows first, odd rows at the bottom)
rows.to_csv("data/failure_rows.csv", index=False)
print()
print(f"Saved all {len(rows)} rows to data/failure_rows.csv")
