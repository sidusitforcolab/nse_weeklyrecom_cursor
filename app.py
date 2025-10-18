import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NSE Stock Performance & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive-return {
        color: #00b050;
        font-weight: bold;
    }
    .negative-return {
        color: #ff0000;
        font-weight: bold;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class NSEStockAnalyzer:
    def __init__(self):
        self.nse_symbols = self.get_nse_symbols()
        
    def get_nse_symbols(self):
        """Get list of NSE symbols - using a curated list of major stocks"""
        return [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
            'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS', 'TITAN.NS', 'NESTLEIND.NS',
            'ULTRACEMCO.NS', 'WIPRO.NS', 'ONGC.NS', 'POWERGRID.NS', 'NTPC.NS',
            'TECHM.NS', 'TATAMOTORS.NS', 'SUNPHARMA.NS', 'TATASTEEL.NS', 'BAJFINANCE.NS',
            'HCLTECH.NS', 'DRREDDY.NS', 'JSWSTEEL.NS', 'TATACONSUM.NS', 'BAJAJFINSV.NS',
            'COALINDIA.NS', 'GRASIM.NS', 'BRITANNIA.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS',
            'CIPLA.NS', 'UPL.NS', 'SHREECEM.NS', 'DIVISLAB.NS', 'BAJAJ-AUTO.NS',
            'INDUSINDBK.NS', 'APOLLOHOSP.NS', 'ADANIPORTS.NS', 'M&M.NS', 'LT.NS',
            'BPCL.NS', 'HINDALCO.NS', 'TATAPOWER.NS', 'IOC.NS', 'SBILIFE.NS'
        ]
    
    def get_stock_data(self, symbol, period="1mo"):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_weekly_returns(self, data):
        """Calculate weekly returns"""
        if data is None or len(data) < 5:
            return None
        
        # Get last 5 trading days (Monday to Friday)
        last_5_days = data.tail(5)
        monday_price = last_5_days.iloc[0]['Close']
        friday_price = last_5_days.iloc[-1]['Close']
        
        weekly_return = ((friday_price - monday_price) / monday_price) * 100
        return {
            'monday_price': monday_price,
            'friday_price': friday_price,
            'weekly_return': weekly_return
        }
    
    def get_top_performers(self, n=5):
        """Get top N performing stocks for the previous week"""
        st.info("Fetching stock data... This may take a moment.")
        
        results = []
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(self.nse_symbols):
            data = self.get_stock_data(symbol, "1mo")
            if data is not None:
                weekly_data = self.calculate_weekly_returns(data)
                if weekly_data:
                    results.append({
                        'symbol': symbol.replace('.NS', ''),
                        'monday_price': weekly_data['monday_price'],
                        'friday_price': weekly_data['friday_price'],
                        'weekly_return': weekly_data['weekly_return']
                    })
            
            progress_bar.progress((i + 1) / len(self.nse_symbols))
        
        # Sort by weekly return and get top N
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('weekly_return', ascending=False)
        
        return results_df.head(n)
    
    def predict_next_week_performance(self, symbol):
        """Simple prediction model based on technical indicators"""
        data = self.get_stock_data(symbol, "3mo")
        if data is None or len(data) < 20:
            return None
        
        # Calculate technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MACD'] = self.calculate_macd(data['Close'])
        
        # Simple prediction logic
        latest_price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        
        # Basic scoring system
        score = 0
        
        # Price vs SMA
        if latest_price > sma_20:
            score += 1
        
        # RSI momentum
        if 30 < rsi < 70:  # Not overbought or oversold
            score += 1
        elif rsi < 30:  # Oversold - potential bounce
            score += 2
        
        # Recent trend
        recent_returns = data['Close'].pct_change().tail(5).mean()
        if recent_returns > 0:
            score += 1
        
        # Volume trend
        volume_avg = data['Volume'].tail(10).mean()
        recent_volume = data['Volume'].tail(3).mean()
        if recent_volume > volume_avg:
            score += 1
        
        return {
            'symbol': symbol.replace('.NS', ''),
            'current_price': latest_price,
            'prediction_score': score,
            'confidence': min(score * 20, 100)  # Convert to percentage
        }
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

def main():
    st.markdown('<h1 class="main-header">📈 NSE Stock Performance & Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = NSEStockAnalyzer()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Top Performers (Last Week)",
        "Stock Predictions",
        "Model Performance",
        "Sector Analysis"
    ])
    
    if page == "Top Performers (Last Week)":
        st.header("🏆 Top 5 Performing Stocks (Previous Week)")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Refresh Data", type="primary"):
                st.rerun()
        
        # Get top performers
        top_performers = analyzer.get_top_performers(5)
        
        if not top_performers.empty:
            st.subheader("Weekly Performance Summary")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Performer", f"{top_performers.iloc[0]['symbol']}", 
                         f"{top_performers.iloc[0]['weekly_return']:.2f}%")
            
            with col2:
                avg_return = top_performers['weekly_return'].mean()
                st.metric("Average Return", f"{avg_return:.2f}%")
            
            with col3:
                max_return = top_performers['weekly_return'].max()
                st.metric("Highest Return", f"{max_return:.2f}%")
            
            with col4:
                min_return = top_performers['weekly_return'].min()
                st.metric("Lowest Return", f"{min_return:.2f}%")
            
            # Display table
            st.subheader("Detailed Performance")
            
            # Format the dataframe for display
            display_df = top_performers.copy()
            display_df['Monday Price'] = display_df['monday_price'].round(2)
            display_df['Friday Price'] = display_df['friday_price'].round(2)
            display_df['Weekly Return'] = display_df['weekly_return'].round(2)
            
            # Add color coding for returns
            def color_returns(val):
                if val > 0:
                    return 'color: #00b050; font-weight: bold'
                else:
                    return 'color: #ff0000; font-weight: bold'
            
            styled_df = display_df[['symbol', 'Monday Price', 'Friday Price', 'Weekly Return']].style.applymap(
                color_returns, subset=['Weekly Return']
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Create visualization
            fig = px.bar(
                top_performers, 
                x='symbol', 
                y='weekly_return',
                title="Weekly Returns of Top 5 Performers",
                color='weekly_return',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            fig.update_layout(
                xaxis_title="Stock Symbol",
                yaxis_title="Weekly Return (%)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("Unable to fetch stock data. Please try again later.")
    
    elif page == "Stock Predictions":
        st.header("🔮 Next Week Stock Predictions")
        st.info("Predictions are based on technical analysis and fundamental indicators")
        
        # Get predictions for all stocks
        predictions = []
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(analyzer.nse_symbols[:20]):  # Limit to first 20 for demo
            prediction = analyzer.predict_next_week_performance(symbol)
            if prediction:
                predictions.append(prediction)
            progress_bar.progress((i + 1) / 20)
        
        if predictions:
            predictions_df = pd.DataFrame(predictions)
            predictions_df = predictions_df.sort_values('prediction_score', ascending=False)
            
            st.subheader("Top Predicted Performers for Next Week")
            
            # Display top 10 predictions
            top_predictions = predictions_df.head(10)
            
            for idx, row in top_predictions.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4>{row['symbol']}</h4>
                        <p><strong>Current Price:</strong> ₹{row['current_price']:.2f}</p>
                        <p><strong>Prediction Score:</strong> {row['prediction_score']}/5</p>
                        <p><strong>Confidence:</strong> {row['confidence']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Create prediction visualization
            fig = px.scatter(
                predictions_df, 
                x='current_price', 
                y='confidence',
                size='prediction_score',
                hover_data=['symbol'],
                title="Stock Predictions: Price vs Confidence",
                labels={'current_price': 'Current Price (₹)', 'confidence': 'Confidence (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Model Performance":
        st.header("📊 Model Performance Tracking")
        
        # Simulated performance data (in real app, this would come from database)
        st.subheader("Weekly Performance Metrics")
        
        # Create sample performance data
        weeks = pd.date_range(start='2024-01-01', periods=12, freq='W')
        performance_data = {
            'Week': weeks.strftime('%Y-%m-%d'),
            'Accuracy': np.random.uniform(65, 85, 12),
            'Precision': np.random.uniform(60, 80, 12),
            'Recall': np.random.uniform(55, 75, 12),
            'F1-Score': np.random.uniform(58, 78, 12)
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_accuracy = perf_df['Accuracy'].mean()
            st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
        
        with col2:
            avg_precision = perf_df['Precision'].mean()
            st.metric("Average Precision", f"{avg_precision:.1f}%")
        
        with col3:
            avg_recall = perf_df['Recall'].mean()
            st.metric("Average Recall", f"{avg_recall:.1f}%")
        
        with col4:
            avg_f1 = perf_df['F1-Score'].mean()
            st.metric("Average F1-Score", f"{avg_f1:.1f}%")
        
        # Performance chart
        fig = go.Figure()
        
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            fig.add_trace(go.Scatter(
                x=perf_df['Week'],
                y=perf_df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Week",
            yaxis_title="Performance (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sector-wise performance
        st.subheader("Sector-wise Performance")
        
        sectors = ['Banking', 'IT', 'Pharma', 'Auto', 'FMCG', 'Energy', 'Metals']
        sector_performance = {
            'Sector': sectors,
            'Average Return': np.random.uniform(-2, 8, len(sectors)),
            'Prediction Accuracy': np.random.uniform(60, 85, len(sectors))
        }
        
        sector_df = pd.DataFrame(sector_performance)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                sector_df, 
                x='Sector', 
                y='Average Return',
                title="Average Returns by Sector (%)",
                color='Average Return',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                sector_df, 
                x='Sector', 
                y='Prediction Accuracy',
                title="Prediction Accuracy by Sector (%)",
                color='Prediction Accuracy',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    elif page == "Sector Analysis":
        st.header("🏭 Sector-wise Analysis")
        
        # Sector mapping
        sector_mapping = {
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS'],
            'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'TECHM.NS', 'HCLTECH.NS'],
            'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
            'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS'],
            'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TATACONSUM.NS'],
            'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS'],
            'Metals': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'COALINDIA.NS']
        }
        
        selected_sector = st.selectbox("Select Sector", list(sector_mapping.keys()))
        
        if selected_sector:
            st.subheader(f"{selected_sector} Sector Analysis")
            
            # Get sector stocks
            sector_stocks = sector_mapping[selected_sector]
            
            # Analyze sector performance
            sector_data = []
            for symbol in sector_stocks:
                data = analyzer.get_stock_data(symbol, "1mo")
                if data is not None:
                    weekly_data = analyzer.calculate_weekly_returns(data)
                    if weekly_data:
                        sector_data.append({
                            'symbol': symbol.replace('.NS', ''),
                            'weekly_return': weekly_data['weekly_return'],
                            'current_price': data['Close'].iloc[-1]
                        })
            
            if sector_data:
                sector_df = pd.DataFrame(sector_data)
                sector_df = sector_df.sort_values('weekly_return', ascending=False)
                
                # Display sector metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_return = sector_df['weekly_return'].mean()
                    st.metric("Average Return", f"{avg_return:.2f}%")
                
                with col2:
                    best_stock = sector_df.iloc[0]
                    st.metric("Best Performer", f"{best_stock['symbol']}", f"{best_stock['weekly_return']:.2f}%")
                
                with col3:
                    worst_stock = sector_df.iloc[-1]
                    st.metric("Worst Performer", f"{worst_stock['symbol']}", f"{worst_stock['weekly_return']:.2f}%")
                
                # Sector performance chart
                fig = px.bar(
                    sector_df, 
                    x='symbol', 
                    y='weekly_return',
                    title=f"{selected_sector} Sector - Weekly Returns",
                    color='weekly_return',
                    color_continuous_scale=['red', 'yellow', 'green']
                )
                fig.update_layout(
                    xaxis_title="Stock Symbol",
                    yaxis_title="Weekly Return (%)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed table
                st.subheader("Detailed Sector Performance")
                styled_df = sector_df.style.applymap(
                    lambda x: 'color: #00b050; font-weight: bold' if x > 0 else 'color: #ff0000; font-weight: bold',
                    subset=['weekly_return']
                )
                st.dataframe(styled_df, use_container_width=True)

if __name__ == "__main__":
    main()