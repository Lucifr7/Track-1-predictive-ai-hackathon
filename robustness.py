import numpy as np
import pandas as pd

from train import train, test
from worklist import score_machines, SENSORS, CHECKS_PER_MORNING, MIN_SCORE

# Real sensors are never perfect. This script damages the test readings on purpose
# and replays the 40 mornings, to see how much worse the list gets.

REPEATS = 20    # each test is repeated with different random damage, then averaged

# How much each sensor normally varies between machines (its standard deviation)
spread = train[SENSORS].std()


# Replay all mornings: Razif checks the top machines on the list each morning
def replay(machines):
    prevented = 0
    wasted = 0
    for morning, today in machines.groupby("Morning"):
        orders = today[today["Score"] >= MIN_SCORE].sort_values("Score", ascending=False)
        checked = orders.head(CHECKS_PER_MORNING)
        prevented += (checked["Machine failure"] == 1).sum()
        wasted += (checked["Machine failure"] == 0).sum()
    return prevented, wasted


# Add random error to every sensor reading.
# level 0.10 means the error is about 10% of the sensor's normal variation.
def add_noise(readings, level, seed):
    rng = np.random.default_rng(seed)
    noisy = readings.copy()
    for sensor in SENSORS:
        noisy[sensor] = noisy[sensor] + rng.normal(0, spread[sensor] * level, len(noisy))
    return noisy


# Blank out one reading on a share of the machines, as if a sensor stopped reporting
def break_sensors(readings, share, seed):
    rng = np.random.default_rng(seed)
    broken = readings.copy()
    broken[SENSORS] = broken[SENSORS].astype(float)
    unlucky = rng.choice(broken.index, size=int(len(broken) * share), replace=False)
    for label in unlucky:
        broken.loc[label, rng.choice(SENSORS)] = np.nan
    return broken


def average_replay(damage, amount):
    results = [replay(score_machines(damage(test, amount, seed))) for seed in range(REPEATS)]
    prevented = np.mean([r[0] for r in results])
    wasted = np.mean([r[1] for r in results])
    return round(prevented, 1), round(wasted, 1)


def noise_table():
    rows = {"No noise": replay(score_machines(test))}
    for level in [0.02, 0.05, 0.10, 0.20]:
        example = f"about {spread['Air temperature [K]'] * level:.2f} K, {spread['Rotational speed [rpm]'] * level:.0f} rpm"
        rows[f"{level * 100:.0f}% noise ({example})"] = average_replay(add_noise, level)
    return pd.DataFrame(rows, index=["Failures prevented", "Wasted checks"]).T


def missing_table():
    rows = {"All sensors working": replay(score_machines(test))}
    for share in [0.05, 0.10, 0.20]:
        rows[f"{share * 100:.0f}% of machines missing a reading"] = average_replay(break_sensors, share)
    return pd.DataFrame(rows, index=["Failures prevented", "Wasted checks"]).T


if __name__ == "__main__":
    print(f"There are 68 real failures in the 40 mornings. Each damaged test is averaged over {REPEATS} repeats.")

    print()
    print("=== Noisy sensors ===")
    print(noise_table().to_string())

    print()
    print("=== Missing readings ===")
    print("A machine with a missing reading is flagged as a sensor fault and is not ranked.")
    print(missing_table().to_string())
