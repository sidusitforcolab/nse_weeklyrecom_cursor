import streamlit as st


def calculate_time_to_reach_target_sip_step_up(initial_amount, final_amount, weekly_contribution, step_up_percentage, step_up_interval, weekly_return_percentage):
    weeks = 0
    amount = float(initial_amount or 0)
    current_contribution = float(weekly_contribution or 0)
    # defensively handle non-positive targets
    if final_amount <= amount:
        return years_months_weeks(0)
    # convert percentages
    weekly_return_pct = float(weekly_return_percentage or 0)
    step_up_pct = float(step_up_percentage or 0)
    step_up_int = int(step_up_interval or 1)
    while amount < final_amount:
        amount += current_contribution
        amount += amount * (weekly_return_pct / 100)
        weeks += 1
        if step_up_int > 0 and weeks % step_up_int == 0:
            current_contribution *= (1 + step_up_pct / 100)
        # safety to avoid infinite loops
        if weeks > 100 * 52:
            break
    return years_months_weeks(weeks)


def years_months_weeks(weeks):
    years = weeks // 52
    remaining_weeks = weeks % 52
    months = remaining_weeks // 4
    weeks = remaining_weeks % 4

    result = []
    if years > 0:
        result.append(f"{years} years")
    if months > 0:
        result.append(f"{months} months")
    if weeks > 0:
        result.append(f"{weeks} weeks")

    return ", ".join(result)


st.title("Time to reach financial independence")
st.markdown("Use the inputs in the sidebar to configure the scenario. Select SIP to enable weekly contributions and optional step-up.")

with st.sidebar:
    initial_amount = st.number_input('Initial Amount (Rs)', value=10000.0, min_value=0.0, step=100.0)
    final_amount = st.number_input('Final Amount (Rs)', value=100000000.0, min_value=0.0, step=1000.0)
    weekly_return_percentage = st.number_input('Weekly Expected Return (%)', value=2.0, step=0.1)
    use_sip = st.checkbox('SIP')
    weekly_contribution = 0.0
    step_up = False
    step_up_percentage = 0.0
    step_up_interval = 1
    if use_sip:
        weekly_contribution = st.number_input('Weekly Contribution (Rs)', value=1000.0, min_value=0.0, step=100.0)
        step_up = st.checkbox('Step-up')
        if step_up:
            step_up_percentage = st.number_input('Step-up Percentage (%)', value=5.0, step=0.1)
            step_up_interval = st.number_input('Step-up Interval (weeks)', value=52, min_value=1, step=1)

if st.button('Calculate'):
    if use_sip and step_up:
        time_required = calculate_time_to_reach_target_sip_step_up(initial_amount, final_amount, weekly_contribution, step_up_percentage, step_up_interval, weekly_return_percentage)
    elif use_sip:
        time_required = calculate_time_to_reach_target_sip_step_up(initial_amount, final_amount, weekly_contribution, 0, 1, weekly_return_percentage)
    else:
        time_required = calculate_time_to_reach_target_sip_step_up(initial_amount, final_amount, 0, 0, 1, weekly_return_percentage)
    st.success(f'Time Required: {time_required}')
    # show a simple projection table
    st.write('---')
    st.write('Projection snapshot (first 10 weeks)')
    snapshot = []
    weeks = 0
    amount = float(initial_amount)
    current_contribution = float(weekly_contribution)
    while weeks < 10 and amount < final_amount:
        amount += current_contribution
        amount += amount * (weekly_return_percentage / 100)
        weeks += 1
        snapshot.append({'week': weeks, 'amount': round(amount,2)})
    st.table(snapshot)
