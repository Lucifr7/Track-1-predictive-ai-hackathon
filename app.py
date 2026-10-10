import pandas as pd
import streamlit as st

from train import test, failure_types
from worklist import score_machines, morning_list, problems_found, explain, actions, SENSORS
from simulate import compare_methods, forest_machines
from robustness import add_noise, noise_table, missing_table

st.set_page_config(page_title="Morning Work Orders", page_icon="🔧")

# --- Controls -----------------------------------------------------------------
st.sidebar.header("This morning")
morning = st.sidebar.slider("Morning", 1, 40, 2)
checks = st.sidebar.slider("Checks Razif has time for", 1, 5, 3)

st.sidebar.header("Stress test")
noise = st.sidebar.select_slider(
    "Sensor noise",
    options=[0, 2, 5, 10, 20],
    format_func=lambda level: "None" if level == 0 else f"{level}%",
)
broken = st.sidebar.checkbox("Two sensors stop reporting")
st.sidebar.caption(
    "Noise adds random error to every reading. "
    "A broken sensor leaves a reading blank."
)

# --- Score the machines -------------------------------------------------------
readings = test
if noise > 0:
    readings = add_noise(readings, noise / 100, seed=morning)

machines = score_machines(readings)

if broken:
    # Blank one reading on two of this morning's machines, then score again
    todays_labels = machines[machines["Morning"] == morning].index
    readings = readings.copy()
    readings[SENSORS] = readings[SENSORS].astype(float)
    readings.loc[todays_labels[4], "Torque [Nm]"] = None
    readings.loc[todays_labels[17], "Process temperature [K]"] = None
    machines = score_machines(readings)

today, orders, watch, faults = morning_list(machines, morning)

# --- Page ---------------------------------------------------------------------
st.title("Morning work orders")
st.caption("For Razif, maintenance technician. 50 machines, time for a few checks before the shift.")

tab_list, tab_proof, tab_limits = st.tabs(["Today's list", "Does it help?", "How it works"])

with tab_list:
    col1, col2, col3 = st.columns(3)
    col1.metric("Machines scanned", len(today) + len(faults))
    col2.metric("Need attention", len(orders))
    col3.metric("Money at risk", f"RM {orders['Score'].sum():,.0f}")

    if len(orders) == 0:
        st.success("No machine needs a check this morning.")

    for rank, (label, m) in enumerate(orders.iterrows(), start=1):
        if rank == checks + 1:
            st.subheader("If you have time left")

        with st.container(border=True):
            left, right = st.columns([3, 1])
            left.subheader(f"#{rank}  Machine {m['UDI']}")
            right.metric("At risk", f"RM {m['Score']:,.0f}")

            for problem in problems_found(m):
                st.markdown(f"**{problem}** ({m['Chance of ' + problem] * 100:.0f}% chance)")
                st.markdown(f"**Why:** {explain(problem, m)}")
                st.markdown(f"**Do:** {actions[problem]}")

    if len(watch) > 0:
        ids = ", ".join(str(udi) for udi in watch["UDI"])
        st.info(
            f"**Low priority, no check needed today:** machine(s) {ids}. "
            "These tools are past 200 minutes, so replace them at the next planned stop."
        )

    for label, m in faults.iterrows():
        missing = [s.split(" [")[0].lower() for s in SENSORS if pd.isna(m[s])]
        st.warning(
            f"**Sensor fault on machine {m['UDI']}:** no {', '.join(missing)} reading. "
            "This machine is not ranked. Check the sensor."
        )

    with st.expander("What really happened this morning (answer key, for the demo)"):
        st.write(
            "The model never saw these answers. They come from the dataset "
            "and show whether each machine really failed."
        )
        shown = pd.concat([orders, today[(today["Machine failure"] == 1) & (today["Score"] < 100)], faults])
        shown = shown[~shown.index.duplicated()]

        answer = pd.DataFrame({
            "Machine": shown["UDI"],
            "On the list": ["Yes" if label in orders.index else "No" for label in shown.index],
            "Really failed": ["Yes" if failed == 1 else "No" for failed in shown["Machine failure"]],
            "Real problem": [
                ", ".join(name for code, name in failure_types.items() if row[code] == 1) or "-"
                for label, row in shown.iterrows()
            ],
        })
        st.dataframe(answer, hide_index=True, width="stretch")

with tab_proof:
    st.subheader("40 mornings, replayed five ways")
    st.write(
        "Every method gets the same 50 machines each morning and at most 3 checks. "
        "Across the 40 mornings, 68 machines really failed."
    )

    @st.cache_data
    def comparison_table():
        table, comparison = compare_methods(score_machines(test), forest_machines())
        return comparison

    st.dataframe(comparison_table(), width="stretch")
    st.write(
        "Our list prevents about three times as many failures as checking the most "
        "worn tools first, with half the checks. Making the list with random forests, a more "
        "complex model, prevents 2 more failures but wastes about ten times as many checks, "
        "because it flags many worn tools that do not fail."
    )

    st.subheader("What if the sensors are noisy?")
    st.write(
        "We add random error to every reading and replay the 40 mornings. "
        "The percentage is the size of the error compared with how much each sensor normally varies."
    )
    st.dataframe(st.cache_data(noise_table)(), width="stretch")

    st.subheader("What if a reading is missing?")
    st.write("A machine with a missing reading is flagged as a sensor fault and is not ranked.")
    st.dataframe(st.cache_data(missing_table)(), width="stretch")

with tab_limits:
    st.subheader("How the list is made")
    st.markdown(
        """
1. **Detect.** Simple decision-tree models read each machine's temperature, speed, torque and tool wear.
2. **Categorize.** One model per problem: tool wear, overheating, power and overstrain.
3. **Prioritize.** Money at risk = chance of failure x (cost if it fails - cost of a planned fix).
4. **Explain.** Each work order gives the reason, using the machine's own readings, and what to do.
"""
    )

    st.subheader("What to keep in mind")
    st.markdown(
        """
- The data is the AI4I 2020 Predictive Maintenance dataset. It is **synthetic**, so real sensors would be noisier.
- The models were trained on 8,000 machine runs and tested on 2,000 runs they had never seen. This app only shows those 2,000.
- The model recognises machines running in a danger zone **now**. It does not forecast days ahead.
- All RM figures are **our assumptions**, not measured costs.
- Splitting the test runs into 40 mornings of 50 machines is our way of showing how the list would be used.
"""
    )
