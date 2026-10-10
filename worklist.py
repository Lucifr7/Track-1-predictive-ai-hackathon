from train import test, risk

# Which morning to show. The 2,000 test machines are cut into
# 40 mornings of 50 machines each (Razif looks after 50 machines).
MORNING = 2
MACHINES_PER_MORNING = 50
CHECKS_PER_MORNING = 3      # he only has time for about 3 inspections
MIN_SCORE = 100             # below RM 100 at risk, it is not worth a work order

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
        return f"The tool has been cutting for {m['Tool wear [min]']} minutes. Tools tend to fail after about 200."
    if problem == "Overheating":
        return (f"It is running slowly ({m['Rotational speed [rpm]']} rpm) and the process is only "
                f"{m['Temp difference']:.1f} K warmer than the air, so heat is not escaping. "
                f"Danger zone: under 1380 rpm and under 8.6 K.")
    if problem == "Power":
        kilowatts = m["Power"] * 2 * 3.1416 / 60 / 1000
        return f"It is drawing {kilowatts:.1f} kW. The safe range is 3.5 to 9.0 kW."
    if problem == "Overstrain":
        return (f"High torque ({m['Torque [Nm]']} Nm) on a tool already used for "
                f"{m['Tool wear [min]']} minutes is overloading the machine.")


# Money at risk = chance of failure x (cost if it fails - cost of a planned fix)
money = risk.copy()
for problem in costs:
    saving = costs[problem]["if_fails"] - costs[problem]["planned_fix"]
    money[problem] = risk[problem] * saving

machines = test.copy()
machines["Score"] = money.sum(axis=1)
machines["Morning"] = [i // MACHINES_PER_MORNING + 1 for i in range(len(machines))]


def print_work_order(rank, m):
    print(f"#{rank}  Machine {m['UDI']}    Money at risk: RM {m['Score']:,.0f}")
    for problem in costs:
        chance = risk.loc[m.name, problem]
        if chance >= 0.2:
            print(f"    Problem: {problem} ({chance * 100:.0f}% chance)")
            print(f"    Why:     {explain(problem, m)}")
            print(f"    Do:      {actions[problem]}")
    print()


if __name__ == "__main__":
    today = machines[machines["Morning"] == MORNING].sort_values("Score", ascending=False)
    orders = today[today["Score"] >= MIN_SCORE]
    watch = today[(today["Score"] > 0) & (today["Score"] < MIN_SCORE)]

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
