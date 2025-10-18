# NSE Stock Performance & Prediction Dashboard

A comprehensive Streamlit application that analyzes NSE stock performance and provides predictions for upcoming weeks.

## Features

### 📊 Top Performers Analysis
- Shows top 5 performing stocks from the previous week
- Displays Monday opening and Friday closing prices
- Color-coded returns (green for positive, red for negative)
- Interactive charts and detailed performance metrics

### 🔮 Stock Predictions
- AI-powered predictions for next week's performance
- Based on technical analysis and fundamental indicators
- Confidence scoring system
- Covers 50+ major NSE stocks

### 📈 Model Performance Tracking
- Weekly performance metrics (Accuracy, Precision, Recall, F1-Score)
- Historical performance visualization
- Sector-wise performance analysis
- Aggregate performance over customizable time periods

### 🏭 Sector Analysis
- Performance analysis by industry sectors
- Banking, IT, Pharma, Auto, FMCG, Energy, Metals
- Sector-wise top and worst performers
- Comparative analysis across sectors

## Technical Features

- **Real-time Data**: Fetches live data from Yahoo Finance
- **Technical Indicators**: RSI, MACD, Moving Averages
- **Responsive Design**: Mobile-friendly interface
- **Interactive Charts**: Plotly-powered visualizations
- **Performance Tracking**: Model accuracy monitoring

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nse-stock-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Deploy the app with the following settings:
   - Main file path: `app.py`
   - Python version: 3.9+

## Usage

1. **Top Performers**: View the best performing stocks from the previous week
2. **Predictions**: Get AI-powered predictions for next week's performance
3. **Model Performance**: Monitor prediction accuracy and model performance
4. **Sector Analysis**: Analyze performance by industry sectors

## Data Sources

- **Stock Prices**: Yahoo Finance API
- **NSE Symbols**: Curated list of 50+ major NSE stocks
- **Technical Indicators**: Calculated in real-time

## Disclaimer

This application is for educational and informational purposes only. Stock market investments carry risks, and past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.

## Contributing

Feel free to contribute to this project by:
- Adding new technical indicators
- Improving prediction algorithms
- Adding more NSE stocks
- Enhancing the user interface

## License

This project is licensed under the MIT License.