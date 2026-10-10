import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv("data/ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv")

# Models only understand numbers, so turn L / M / H into 0 / 1 / 2
df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

df["Temp difference"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
df["Strain"] = df["Torque [Nm]"] * df["Tool wear [min]"]


features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp difference",
    "Power",
    "Strain"
]

# Split the whole table, so the failure-type columns are split too
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["Machine failure"]
)

X_train, X_test = train[features], test[features]                      # the clues
y_train, y_test = train["Machine failure"], test["Machine failure"]    # the answer

# Turn a model's guesses into readable numbers
def score(actual, predicted):
    matrix = confusion_matrix(actual, predicted)
    correct_fine, false_alarms = matrix[0]
    missed, caught = matrix[1]
    alarms = caught + false_alarms

    return {
        "Real cases": caught + missed,
        "Caught": caught,
        "Missed": missed,
        "False alarms": false_alarms,
        "Recall %": round(caught / (caught + missed) * 100),
        "Precision %": round(caught / alarms * 100) if alarms > 0 else 0,
    }


print("=== The split ===")
print(f"Training rows: {len(X_train)} ({y_train.sum()} failures)")
print(f"Test rows:     {len(X_test)} ({y_test.sum()} failures)")

# Main model: "will this machine fail?"
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

results = {"Any failure": score(y_test, predictions)}

# One small tree per failure type: "will this machine have THIS problem?"
failure_types = {
    "TWF": "Tool wear",
    "HDF": "Overheating",
    "PWF": "Power",
    "OSF": "Overstrain",
}
type_models = {}

for code, meaning in failure_types.items():
    type_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    type_model.fit(X_train, train[code])
    type_models[code] = type_model

    results[meaning] = score(test[code], type_model.predict(X_test))

print()
print("=== Results on the test set ===")
print(f"Accuracy of the main model: {accuracy_score(y_test, predictions) * 100:.1f}%")
print(pd.DataFrame(results).T)
