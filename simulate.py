import pandas as pd

from worklist import machines, costs, CHECKS_PER_MORNING, MACHINES_PER_MORNING, MIN_SCORE

# Does the list help? Replay all 40 mornings and compare with what really happened.
# Each morning Razif checks the top machines on the list (3 at most).
# We assume a check on a machine that was going to fail prevents the breakdown.

FALSE_ALARM_COST = 15       # RM, a check that finds nothing
AVERAGE_SAVING = 859        # RM, used when a failure has no cause recorded

codes = {"TWF": "Tool wear", "HDF": "Overheating", "PWF": "Power", "OSF": "Overstrain"}


# Money saved by fixing this machine before it breaks down
def saving(m):
    total = 0
    for code, problem in codes.items():
        if m[code] == 1:
            total += costs[problem]["if_fails"] - costs[problem]["planned_fix"]
    return total if total > 0 else AVERAGE_SAVING


# Replay every morning four ways and compare
def compare_methods(machines):
    rows = []            # our list, morning by morning
    tool_rule = []       # rule of thumb: check the 3 most worn tools
    rotation = []        # fixed schedule: machines 1-3 today, 4-6 tomorrow, ...

    for morning, today in machines.groupby("Morning"):
        orders = today[today["Score"] >= MIN_SCORE].sort_values("Score", ascending=False)
        checked = orders.head(CHECKS_PER_MORNING)

        failed = today[today["Machine failure"] == 1]
        caught = checked[checked["Machine failure"] == 1]
        wasted = checked[checked["Machine failure"] == 0]

        rows.append({
            "Morning": morning,
            "Real failures": len(failed),
            "Checks done": len(checked),
            "Failures prevented": len(caught),
            "Wasted checks": len(wasted),
            "RM saved": caught.apply(saving, axis=1).sum() - len(wasted) * FALSE_ALARM_COST,
            "RM possible": failed.apply(saving, axis=1).sum(),
        })

        # The same morning, choosing the 3 machines in two other ways
        most_worn = today.sort_values("Tool wear [min]", ascending=False).head(CHECKS_PER_MORNING)
        tool_rule.append(most_worn["Machine failure"].sum())

        start = (morning - 1) * CHECKS_PER_MORNING
        positions = [(start + i) % MACHINES_PER_MORNING for i in range(CHECKS_PER_MORNING)]
        rotation.append(today.iloc[positions]["Machine failure"].sum())

    table = pd.DataFrame(rows).set_index("Morning")
    total = table.sum()

    # Random: 3 machines picked out of 50 with no reason,
    # so each failure has a 3-in-50 chance of being found
    all_checks = len(table) * CHECKS_PER_MORNING
    random_prevented = total["Real failures"] * CHECKS_PER_MORNING / MACHINES_PER_MORNING

    comparison = pd.DataFrame({
        "Checks done": [total["Checks done"], all_checks, all_checks, all_checks],
        "Failures prevented": [total["Failures prevented"], sum(tool_rule), sum(rotation), random_prevented],
    }, index=["Our list", "Most worn tools first", "Fixed rotation", "Random"])
    comparison["Wasted checks"] = comparison["Checks done"] - comparison["Failures prevented"]

    return table, comparison.round(1)


if __name__ == "__main__":
    table, comparison = compare_methods(machines)

    print("=== Our list, first 10 mornings ===")
    print(table.head(10).to_string())

    print()
    print(f"=== All {len(table)} mornings: {len(machines)} machines, {table['Real failures'].sum():.0f} real failures ===")
    print(f"Each method may check up to {CHECKS_PER_MORNING} machines a morning.")
    print()
    print(comparison.to_string())
