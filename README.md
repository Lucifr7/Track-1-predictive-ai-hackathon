# Morning Work Orders

Team NR, Dormathon 2026, Track 1: Predictive Model Track ("Predict early, decide better").

A maintenance technician looks after 50 machines and has time to check about 3 of them before the shift starts. This tool reads each machine's sensor readings and gives a short ranked list: which machines to check first, what is likely wrong, why, and what to do.

## What it does

| Requirement | How |
|---|---|
| Detect | Decision-tree models estimate the chance of failure from temperature, speed, torque and tool wear. |
| Categorize | One model per problem: tool wear, overheating, power and overstrain. A machine can have more than one. |
| Prioritize | Money at risk = chance of failure x (cost if it fails - cost of a planned fix). The list is sorted by this. |
| Explain | Each work order states the reason using the machine's own readings, and the action to take. |

## Results

The models were trained on 8,000 machine runs and tested on 2,000 runs they had never seen (68 real failures).

- The main model catches 56 of the 68 failures with 3 false alarms.
- The test runs are replayed as 40 mornings of 50 machines, with at most 3 checks a morning:

| Method | Checks done | Failures prevented | Wasted checks |
|---|---|---|---|
| Our list | 58 | 53 | 5 |
| Most worn tools first | 120 | 17 | 103 |
| Fixed rotation | 120 | 2 | 118 |
| Random | 120 | 4.1 | 115.9 |

- With random error added to every sensor reading (10% of each sensor's normal variation), the list still prevents about 44 failures. Machines with a missing reading are flagged as sensor faults and are not ranked.

The full outputs are in the `results` folder.

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

Or run the steps one at a time:

| Command | What it shows |
|---|---|
| `python explore.py` | A first look at the data |
| `python train.py` | Model results on the test set |
| `python worklist.py` | The work-order list for one morning |
| `python simulate.py` | The 40-morning comparison |
| `python robustness.py` | The noisy-sensor and missing-reading tests |

## Limitations

- The data is the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) from the UCI Machine Learning Repository. It is synthetic, so real sensor data would be noisier and the scores lower.
- Each row is a single machine run with its readings and outcome. The models recognise machines running in a danger zone now. They do not forecast failures days ahead.
- All costs in Malaysian Ringgit are our assumptions. They are listed in `Task.txt`.
- Treating the test runs as 40 mornings of 50 machines is our way of showing how the list would be used.
- Tool wear failures happen at a random point after about 200 minutes of use, so the model catches few of them. Worn tools are shown as low-priority reminders.
