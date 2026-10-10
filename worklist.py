from train import test, features, type_models, failure_types

# Which morning to show. The 2,000 test machines are cut into
# 40 mornings of 50 machines each (Razif looks after 50 machines).
MORNING = 2
MACHINES_PER_MORNING = 50
CHECKS_PER_MORNING = 3      # he only has time for about 3 inspections
MIN_SCORE = 100             # below RM 100 at risk, it is not worth a work order

SENSORS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Costs in RM, from Task.txt. These are our assumptions.
costs = {
    "Tool wear":   {"if_fails": 650,  "planned_fix": 100},
    "Overheating": {"if_fails": 850,  "planned_fix": 70},
    "Power":       {"if_fails": 700,  "planned_fix": 50},
    "Overstrain":  {"if_fails": 1400, "planned_fix": 100},
}

actions = {
    "Tool wear":   "Inspect the cutting insert for wear and replace it before the shift.",
    "Overheating": "Check coolant level and flow, and clear the cooling fans.",
    "Power":       "Check drive belt tension, supply voltage and spindle bearings.",
    "Overstrain":  "Replace the insert now and check the feed-rate settings.",
}


# The reason for each problem, written with the machine's own readings
def explain(problem, m):
    if problem == "Tool wear":
        return f"The tool has been cutting for {m['Tool wear [min]']:.0f} minutes. Tools tend to fail after about 200."
    if problem == "Overheating":
        return (f"It is running slowly ({m['Rotational speed [rpm]']:.0f} rpm) and the process is only "
                f"{m['Temp difference']:.1f} K warmer than the air, so heat is not escaping. "
                f"Danger zone: under 1380 rpm and under 8.6 K.")
    if problem == "Power":
        kilowatts = m["Power"] * 2 * 3.1416 / 60 / 1000
        return f"It is drawing {kilowatts:.1f} kW. The safe range is 3.5 to 9.0 kW."
    if problem == "Overstrain":
        return (f"High torque ({m['Torque [Nm]']:.1f} Nm) on a tool already used for "
                f"{m['Tool wear [min]']:.0f} minutes is overloading the machine.")


# Take a table of machine readings and work out the money at risk for each machine:
# money at risk = chance of failure x (cost if it fails - cost of a planned fix)
def score_machines(readings):
    m = readings.copy()

    # A machine with a missing reading cannot be judged. We flag it instead of guessing.
    m["Sensor fault"] = m[SENSORS].isna().any(axis=1)
    working = ~m["Sensor fault"]

    # The three combined readings are worked out again from the raw readings
    m["Temp difference"] = m["Process temperature [K]"] - m["Air temperature [K]"]
    m["Power"] = m["Torque [Nm]"] * m["Rotational speed [rpm]"]
    m["Strain"] = m["Torque [Nm]"] * m["Tool wear [min]"]

    m["Score"] = 0.0
    for code, problem in failure_types.items():
        saving = costs[problem]["if_fails"] - costs[problem]["planned_fix"]
        m["Chance of " + problem] = 0.0
        if working.any():
            chances = type_models[code].predict_proba(m.loc[working, features])[:, 1]
            m.loc[working, "Chance of " + problem] = chances
        m["Score"] += m["Chance of " + problem] * saving

    m["Morning"] = [i // MACHINES_PER_MORNING + 1 for i in range(len(m))]
    return m


# The problems worth telling Razif about for one machine
def problems_found(m):
    return [p for p in costs if m["Chance of " + p] >= 0.2]


# Split one morning into work orders, low priority machines and sensor faults
def morning_list(machines, morning):
    today = machines[machines["Morning"] == morning].sort_values("Score", ascending=False)
    faults = today[today["Sensor fault"]]
    today = today[~today["Sensor fault"]]
    orders = today[today["Score"] >= MIN_SCORE]
    watch = today[(today["Score"] > 0) & (today["Score"] < MIN_SCORE)]
    return today, orders, watch, faults


machines = score_machines(test)


def print_work_order(rank, m):
    print(f"#{rank}  Machine {m['UDI']}    Money at risk: RM {m['Score']:,.0f}")
    for problem in problems_found(m):
        print(f"    Problem: {problem} ({m['Chance of ' + problem] * 100:.0f}% chance)")
        print(f"    Why:     {explain(problem, m)}")
        print(f"    Do:      {actions[problem]}")
    print()


if __name__ == "__main__":
    today, orders, watch, faults = morning_list(machines, MORNING)

    print("=" * 70)
    print(f"MORNING WORK-ORDER LIST (morning {MORNING})")
    print(f"{len(today)} machines scanned, {len(orders)} need attention.")
    print(f"You have time for about {CHECKS_PER_MORNING} checks. Start at the top.")
    print("=" * 70)
    print()

    for rank, (label, m) in enumerate(orders.iterrows(), start=1):
        if rank == CHECKS_PER_MORNING + 1:
            print("--- If you have time left ---")
            print()
        print_work_order(rank, m)

    if len(watch) > 0:
        ids = ", ".join(str(udi) for udi in watch["UDI"])
        print("Low priority: small risk, no check needed today.")
        print("These tools are past 200 minutes, so replace them at the next planned stop.")
        print(f"    Machine(s): {ids}")
