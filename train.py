"""
Track 1: Predictive AI Hackathon
Project: The Morning Work-Order List for CNC Maintenance Technicians
Theme: "Predict early, decide better."

This end-to-end pipeline implements the complete 4-pillar solution:
  1. DETECT: Predicts machine failure probability using machine learning.
  2. CATEGORIZE: Diagnoses specific failure types (TWF, HDF, PWF, OSF), supporting multi-label causes.
  3. PRIORITIZE: Calculates financial risk in Malaysian Ringgit (RM) and ranks machines by urgency.
  4. EXPLAIN: Produces technician-ready, plain-language explanations and actionable repair steps.

It also computes the exact financial ROI (cost savings in RM) based on model Recall and Precision.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class TeeLogger:
    """
    Duplicates stdout output to both console and a log file (result.txt).
    """
    def __init__(self, filename="result.txt"):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        if not self.logfile.closed:
            self.logfile.close()


# ==============================================================================
# 1. DOMAIN CONFIGURATION & COST MODEL (ASSUMPTIONS FROM TASK.TXT)
# ==============================================================================
# All costs in Malaysian Ringgit (RM)
FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF"]

COST_MODEL = {
    "TWF": {
        "name": "Tool Wear Failure",
        "cost_fail": 650.0,   # Breakdown: snapped tool, scrap workpiece, downtime
        "cost_fix": 100.0,    # Planned: 15 min labor + new cutting insert
        "explanation": "Tool nearing end of life: Cumulative cutting time exceeded safe limit (>200 min).",
        "action": "Inspect cutting insert for edge wear or chipping; replace before shift starts.",
    },
    "HDF": {
        "name": "Heat Dissipation Failure",
        "cost_fail": 850.0,   # Breakdown: thermal shutdown, cooldown, recalibration
        "cost_fix": 70.0,     # Planned: top up coolant fluid, clean radiator fan
        "explanation": "Insufficient cooling: Machine running slowly and process heat is not dissipating (temp difference < 8.6 K).",
        "action": "Check coolant fluid levels/nozzles; clear metal chips blocking radiator fans.",
    },
    "PWF": {
        "name": "Power Failure",
        "cost_fail": 700.0,   # Breakdown: spindle stall, inverter trip, electrical reset
        "cost_fix": 50.0,     # Planned: belt tension inspection, electrical terminal check
        "explanation": "Spindle power out of safe operating limits (<3.5 kW or >9.0 kW), risking a motor stall or electrical trip.",
        "action": "Check spindle drive belts for slip, verify supply voltage, and check bearings.",
    },
    "OSF": {
        "name": "Overstrain Failure",
        "cost_fail": 1400.0,  # Breakdown: tool binds & breaks, workpiece scrap, spindle misalignment
        "cost_fix": 100.0,    # Planned: replace worn insert and verify feed-rate
        "explanation": "High mechanical strain: Heavy cutting torque on a worn tool is causing severe spindle overload.",
        "action": "Replace cutting insert immediately and verify programmed feed rates before heavy cuts.",
    },
}

# Cost of 1 false alarm (Razif spends 20-30 min checking a healthy machine)
COST_FALSE_ALARM = 15.0

# Pre-calculate net risk value for each failure type: (C_fail - C_fix)
for f_type in FAILURE_TYPES:
    COST_MODEL[f_type]["net_risk"] = COST_MODEL[f_type]["cost_fail"] - COST_MODEL[f_type]["cost_fix"]


# ==============================================================================
# 2. DATA INGESTION & PHYSICAL FEATURE ENGINEERING
# ==============================================================================
def load_and_preprocess_data(filepath="data/ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv"):
    """
    Loads dataset and extracts physics-based features matching the underlying
    failure rules described in the AI4I dataset documentation.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    print(f"[*] Loaded dataset: {len(df):,} rows from '{filepath}'")

    # Physical feature engineering:
    # 1. Temperature difference (Process - Air) in Kelvin: Key indicator for HDF
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # 2. Spindle Mechanical Power [Watts] = Torque [Nm] * Rotational Speed [rad/s]
    #    omega = rpm * 2 * pi / 60
    omega = df["Rotational speed [rpm]"] * (2 * np.pi / 60.0)
    df["Power_W"] = df["Torque [Nm]"] * omega

    # 3. Overstrain Factor = Tool wear [min] * Torque [Nm]: Key indicator for OSF
    df["Overstrain_factor"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    # One-hot encode product quality type (L, M, H)
    df["Type_H"] = (df["Type"] == "H").astype(int)
    df["Type_M"] = (df["Type"] == "M").astype(int)
    df["Type_L"] = (df["Type"] == "L").astype(int)

    return df


# ==============================================================================
# 3. MODEL TRAINING: DETECT & CATEGORIZE
# ==============================================================================
def train_predictive_models(df):
    """
    Trains:
      1. Binary Detector Model: Predicts overall machine failure probability.
      2. Multi-label Categorizers: Predicts individual probabilities for TWF, HDF, PWF, OSF.
    """
    feature_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temp_diff",
        "Power_W",
        "Overstrain_factor",
        "Type_H",
        "Type_M",
        "Type_L",
    ]

    X = df[feature_cols]
    y_failure = df["Machine failure"]
    y_categories = df[FAILURE_TYPES]

    # Split: 80% train, 20% test (stratified by overall failure)
    X_train, X_test, y_train, y_test, cat_train, cat_test, meta_train, meta_test = train_test_split(
        X,
        y_failure,
        y_categories,
        df[["UDI", "Product ID", "Type", "Tool wear [min]"]],
        test_size=0.20,
        random_state=42,
        stratify=y_failure,
    )

    print(f"[*] Train set: {len(X_train)} samples ({y_train.sum()} failures)")
    print(f"[*] Test set:  {len(X_test)} samples ({y_test.sum()} failures)")

    # 1. Train Failure Detector (handles class imbalance with balanced weights)
    print("\n--- Training Step 1: Overall Failure Detector (RandomForest) ---")
    detector = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    detector.fit(X_train, y_train)

    # 2. Train Categorizer Models (one per failure mode for multi-label support)
    print("--- Training Step 2: Multi-Label Failure Categorizers (TWF, HDF, PWF, OSF) ---")
    categorizers = {}
    for f_type in FAILURE_TYPES:
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, cat_train[f_type])
        categorizers[f_type] = clf

    return {
        "detector": detector,
        "categorizers": categorizers,
        "feature_cols": feature_cols,
        "test_data": (X_test, y_test, cat_test, meta_test),
    }


# ==============================================================================
# 4. EVALUATION & BENCHMARKING
# ==============================================================================
def evaluate_models(models_dict):
    """
    Evaluates detector and categorizers on the test set.
    Reports Recall, Precision, F1-Score, and ROC-AUC.
    """
    detector = models_dict["detector"]
    categorizers = models_dict["categorizers"]
    X_test, y_test, cat_test, _ = models_dict["test_data"]

    print("\n================================================================================")
    print("MODEL PERFORMANCE EVALUATION (TEST SET)")
    print("================================================================================")

    # Detector evaluation
    y_prob = detector.predict_proba(X_test)[:, 1]
    # Using 0.40 probability threshold to balance high recall with practical precision
    threshold = 0.40
    y_pred = (y_prob >= threshold).astype(int)

    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Overall Failure Detector (Decision Threshold = {threshold:.2f}):")
    print(f"  - Recall:    {rec * 100:.2f}%  (Catches {cm[1, 1]} of {cm[1, 0] + cm[1, 1]} true failures)")
    print(f"  - Precision: {prec * 100:.2f}% (True alarms: {cm[1, 1]}, False alarms: {cm[0, 1]})")
    print(f"  - F1-Score:  {f1:.4f}")
    print(f"  - ROC-AUC:   {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  [TN={cm[0, 0]:<4}  FP={cm[0, 1]:<4}]")
    print(f"  [FN={cm[1, 0]:<4}  TP={cm[1, 1]:<4}]")

    print("\nIndividual Failure Type Diagnoses:")
    cat_metrics = {}
    for f_type in FAILURE_TYPES:
        clf = categorizers[f_type]
        c_prob = clf.predict_proba(X_test)[:, 1]
        c_pred = (c_prob >= 0.35).astype(int)
        c_true = cat_test[f_type]

        c_rec = recall_score(c_true, c_pred, zero_division=0)
        c_prec = precision_score(c_true, c_pred, zero_division=0)
        cat_metrics[f_type] = {"recall": c_rec, "precision": c_prec}
        print(f"  [{f_type} - {COST_MODEL[f_type]['name']}]: Recall = {c_rec * 100:.1f}%, Precision = {c_prec * 100:.1f}%")

    return {
        "detector_recall": rec,
        "detector_precision": prec,
        "confusion_matrix": cm,
        "threshold": threshold,
    }


# ==============================================================================
# 5. PRIORITIZE & EXPLAIN: THE MORNING WORK-ORDER GENERATOR
# ==============================================================================
def generate_morning_work_order(models_dict, top_n=10):
    """
    Ranks machines by expected financial risk:
      Expected Risk (RM) = SUM( P(Failure_type) * Net Risk Value )

    Generates plain-language explanation and actionable recommendation for each.
    """
    detector = models_dict["detector"]
    categorizers = models_dict["categorizers"]
    X_test, y_test, cat_test, meta_test = models_dict["test_data"]

    # Compute probabilities
    p_fail = detector.predict_proba(X_test)[:, 1]

    results = meta_test.copy()
    results["Failure_Prob"] = p_fail
    results["Actual_Failure"] = y_test.values

    # Predict category probabilities & calculate Expected Risk
    total_expected_risk = np.zeros(len(results))
    type_diagnoses = []
    explanations = []
    actions = []

    cat_probs = {}
    for f_type in FAILURE_TYPES:
        prob = categorizers[f_type].predict_proba(X_test)[:, 1]
        cat_probs[f_type] = prob
        # Contribution to financial risk
        total_expected_risk += prob * COST_MODEL[f_type]["net_risk"]

    results["Expected_Risk_RM"] = total_expected_risk

    # Build human-readable explanations and actions for each machine
    for i in range(len(results)):
        detected_types = []
        machine_explanations = []
        machine_actions = []

        for f_type in FAILURE_TYPES:
            # Threshold of 0.30 to catch early warning signs
            if cat_probs[f_type][i] >= 0.30:
                detected_types.append(f"{f_type} ({cat_probs[f_type][i] * 100:.0f}%)")
                machine_explanations.append(COST_MODEL[f_type]["explanation"])
                machine_actions.append(COST_MODEL[f_type]["action"])

        if not detected_types:
            if p_fail[i] >= 0.40:
                detected_types.append("Unspecified anomaly")
                machine_explanations.append("Sensor telemetry deviates from normal baseline; possible mechanical vibration or thermal instability.")
                machine_actions.append("Perform full manual visual & mechanical pre-shift inspection.")
            else:
                detected_types.append("Normal (Low Risk)")
                machine_explanations.append("All operating parameters within safe tolerances.")
                machine_actions.append("No immediate action required. Run normal production.")

        type_diagnoses.append(", ".join(detected_types))
        explanations.append(" | ".join(machine_explanations))
        actions.append(" | ".join(machine_actions))

    results["Diagnosed_Issues"] = type_diagnoses
    results["Explanation"] = explanations
    results["Recommended_Action"] = actions

    # Sort descending by financial risk: highest priority machines first
    ranked_work_order = results.sort_values(by="Expected_Risk_RM", ascending=False).reset_index(drop=True)
    ranked_work_order.index = ranked_work_order.index + 1
    ranked_work_order.index.name = "Priority_Rank"

    # Export to CSV for technician / demo use
    output_csv = "morning_work_order.csv"
    ranked_work_order.to_csv(output_csv)
    print(f"\n[*] Successfully generated morning work-order list saved to '{output_csv}'")

    print("\n================================================================================")
    print("RAZIF'S MORNING WORK-ORDER LIST (TOP URGENT MACHINES TO INSPECT)")
    print("================================================================================")
    cols_display = [
        "Product ID",
        "Type",
        "Failure_Prob",
        "Expected_Risk_RM",
        "Diagnosed_Issues",
        "Recommended_Action",
    ]

    top_display = ranked_work_order.head(top_n)[cols_display].copy()
    top_display["Failure_Prob"] = (top_display["Failure_Prob"] * 100).map("{:.1f}%".format)
    top_display["Expected_Risk_RM"] = top_display["Expected_Risk_RM"].map("RM {:,.2f}".format)

    print(top_display.to_string())

    print("\n--- Detailed Technician Briefing (Top 3 Machines) ---")
    for rank in range(1, min(4, len(ranked_work_order) + 1)):
        row = ranked_work_order.loc[rank]
        print(f"\n[Rank #{rank}] Machine: {row['Product ID']} (Type {row['Type']})")
        print(f"  - Failure Risk: {row['Failure_Prob'] * 100:.1f}% | Potential Loss: RM {row['Expected_Risk_RM']:,.2f}")
        print(f"  - Diagnosis:    {row['Diagnosed_Issues']}")
        print(f"  - Why:          {row['Explanation']}")
        print(f"  - Action:       {row['Recommended_Action']}")

    return ranked_work_order


# ==============================================================================
# 6. BUSINESS IMPACT & FINANCIAL ROI CALCULATION (TASK 4 FORMULA)
# ==============================================================================
def calculate_financial_roi(eval_results, monthly_breakdowns=15):
    """
    Calculates net financial savings per month based on the formula:
      Net Savings = [N_fail * Recall * (C_fail - C_fix)] - [False_Alarms * C_false]
    """
    rec = eval_results["detector_recall"]
    prec = eval_results["detector_precision"]

    # Weighted costs from Task 2 & Task 4
    c_fail_avg = 936.0  # RM
    c_fix_avg = 77.0    # RM
    net_saving_per_catch = c_fail_avg - c_fix_avg  # ~RM 859

    # Monthly scenario for 50-machine plant
    caught_failures = monthly_breakdowns * rec
    missed_failures = monthly_breakdowns * (1.0 - rec)

    # False alarms: based on model precision
    false_alarms = caught_failures * ((1.0 - prec) / prec) if prec > 0 else 0

    gross_savings = caught_failures * net_saving_per_catch
    false_alarm_cost = false_alarms * COST_FALSE_ALARM
    net_monthly_savings = gross_savings - false_alarm_cost
    annual_savings = net_monthly_savings * 12

    print("\n================================================================================")
    print("BUSINESS IMPACT & FINANCIAL PROOF (FOR PRESENTATION SLIDES)")
    print("================================================================================")
    print(f"Operational Scenario: 50 CNC Milling Machines in Factory")
    print(f"Baseline Unscheduled Breakdowns: {monthly_breakdowns} incidents/month (without AI)")
    print(f"\nModel Performance on Test Set:")
    print(f"  - Recall:    {rec * 100:.2f}%")
    print(f"  - Precision: {prec * 100:.2f}%")
    print(f"\nMonthly Operational Outcome:")
    print(f"  - Breakdowns Prevented:       {caught_failures:.1f} / {monthly_breakdowns} machines")
    print(f"  - Missed Breakdowns:          {missed_failures:.1f} machines")
    print(f"  - False Alarm Inspections:    {false_alarms:.1f} inspections/month")
    print(f"\nFinancial Proof (Malaysian Ringgit):")
    print(f"  - Gross Repair & Downtime Saved: RM {gross_savings:,.2f} / month")
    print(f"  - Penalty for False Alarms:      -RM {false_alarm_cost:,.2f} / month")
    print(f"  -------------------------------------------------------------")
    print(f"  >>> NET MONTHLY SAVINGS:         RM {net_monthly_savings:,.2f} / month")
    print(f"  >>> NET ANNUAL SAVINGS:          RM {annual_savings:,.2f} / year")
    print("================================================================================")


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================
if __name__ == "__main__":
    result_file = "result.txt"
    logger = TeeLogger(result_file)
    sys.stdout = logger

    print("=" * 80)
    print("TRACK 1 PREDICTIVE AI: MORNING WORK-ORDER PIPELINE")
    print("Theme: 'Predict early, decide better.'")
    print("=" * 80)

    # 1. Load data & engineer physics features
    dataset_path = "data/ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv"
    data = load_and_preprocess_data(dataset_path)

    # 2. Train models
    models = train_predictive_models(data)

    # 3. Evaluate performance
    eval_metrics = evaluate_models(models)

    # 4. Generate morning work-order list (Detect, Categorize, Prioritize, Explain)
    work_orders = generate_morning_work_order(models, top_n=10)

    # 5. Compute Business Impact & Dollar (RM) Proof
    calculate_financial_roi(eval_metrics, monthly_breakdowns=15)

    print(f"\n[*] All pipeline results have been successfully saved to '{result_file}'")
    logger.close()
    sys.stdout = logger.terminal


