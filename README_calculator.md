# Financial Independence Calculator

A Streamlit app to calculate the time required to reach financial independence based on initial investment, target amount, and expected returns.

## Features

- Calculate time to reach financial independence
- Support for Systematic Investment Plan (SIP)
- Optional step-up investments
- Weekly return projections
- 10-week projection table

## Setup

1. Ensure you have Python 3.7+ installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run streamlit_app.py
```

## Using the Calculator

1. Enter your initial investment amount
2. Set your target (final) amount
3. Specify weekly expected return percentage
4. Optionally enable SIP (Systematic Investment Plan)
   - Set weekly contribution amount
   - Enable step-up to increase contributions periodically
   - Configure step-up percentage and interval

The calculator will show:
- Time required to reach your goal
- A 10-week projection of your investment growth