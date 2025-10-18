import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from nsepy import get_history
from nsepy.live import get_quote
import ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class NSEDataFetcher:
    """Enhanced NSE data fetcher with API integration and ML capabilities"""
    
    def __init__(self):
        self.nse_symbols_map = self.get_nse_symbol_mapping()
        self.scaler = StandardScaler()
        self.model = None
        
    def get_nse_symbol_mapping(self):
        """Map Yahoo Finance symbols to NSE symbols"""
        return {
            'RELIANCE.NS': 'RELIANCE',
            'TCS.NS': 'TCS',
            'HDFCBANK.NS': 'HDFCBANK',
            'INFY.NS': 'INFY',
            'HINDUNILVR.NS': 'HINDUNILVR',
            'ICICIBANK.NS': 'ICICIBANK',
            'KOTAKBANK.NS': 'KOTAKBANK',
            'BHARTIARTL.NS': 'BHARTIARTL',
            'ITC.NS': 'ITC',
            'SBIN.NS': 'SBIN',
            'ASIANPAINT.NS': 'ASIANPAINT',
            'MARUTI.NS': 'MARUTI',
            'AXISBANK.NS': 'AXISBANK',
            'TITAN.NS': 'TITAN',
            'NESTLEIND.NS': 'NESTLEIND',
            'ULTRACEMCO.NS': 'ULTRACEMCO',
            'WIPRO.NS': 'WIPRO',
            'ONGC.NS': 'ONGC',
            'POWERGRID.NS': 'POWERGRID',
            'NTPC.NS': 'NTPC',
            'TECHM.NS': 'TECHM',
            'TATAMOTORS.NS': 'TATAMOTORS',
            'SUNPHARMA.NS': 'SUNPHARMA',
            'TATASTEEL.NS': 'TATASTEEL',
            'BAJFINANCE.NS': 'BAJFINANCE',
            'HCLTECH.NS': 'HCLTECH',
            'DRREDDY.NS': 'DRREDDY',
            'JSWSTEEL.NS': 'JSWSTEEL',
            'TATACONSUM.NS': 'TATACONSUM',
            'BAJAJFINSV.NS': 'BAJAJFINSV',
            'COALINDIA.NS': 'COALINDIA',
            'GRASIM.NS': 'GRASIM',
            'BRITANNIA.NS': 'BRITANNIA',
            'EICHERMOT.NS': 'EICHERMOT',
            'HEROMOTOCO.NS': 'HEROMOTOCO',
            'CIPLA.NS': 'CIPLA',
            'UPL.NS': 'UPL',
            'SHREECEM.NS': 'SHREECEM',
            'DIVISLAB.NS': 'DIVISLAB',
            'BAJAJ-AUTO.NS': 'BAJAJ_AUTO',
            'INDUSINDBK.NS': 'INDUSINDBK',
            'APOLLOHOSP.NS': 'APOLLOHOSP',
            'ADANIPORTS.NS': 'ADANIPORTS',
            'M&M.NS': 'M_M',
            'LT.NS': 'LT',
            'BPCL.NS': 'BPCL',
            'HINDALCO.NS': 'HINDALCO',
            'TATAPOWER.NS': 'TATAPOWER',
            'IOC.NS': 'IOC',
            'SBILIFE.NS': 'SBILIFE'
        }
    
    def fetch_nse_historical_data(self, symbol, start_date, end_date):
        """Fetch historical data from NSE using nsepy"""
        try:
            nse_symbol = self.nse_symbols_map.get(symbol, symbol.replace('.NS', ''))
            data = get_history(symbol=nse_symbol, start=start_date, end=end_date)
            
            if data.empty:
                # Fallback to yfinance if NSE data is not available
                return self.fetch_yfinance_data(symbol, start_date, end_date)
            
            # Standardize column names
            data = data.rename(columns={
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            return data
            
        except Exception as e:
            print(f"Error fetching NSE data for {symbol}: {str(e)}")
            # Fallback to yfinance
            return self.fetch_yfinance_data(symbol, start_date, end_date)
    
    def fetch_yfinance_data(self, symbol, start_date, end_date):
        """Fallback method using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            return data
        except Exception as e:
            print(f"Error fetching yfinance data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_2022_data(self, symbols=None):
        """Fetch complete 2022 data for all or specified symbols"""
        if symbols is None:
            symbols = list(self.nse_symbols_map.keys())
        
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31)
        
        all_data = {}
        
        for symbol in symbols:
            print(f"Fetching 2022 data for {symbol}...")
            data = self.fetch_nse_historical_data(symbol, start_date, end_date)
            
            if not data.empty:
                all_data[symbol] = data
                
        return all_data
    
    def create_technical_features(self, data):
        """Create comprehensive technical indicators"""
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
            df[f'Price_SMA_{window}_Ratio'] = df['Close'] / df[f'SMA_{window}']
        
        # Volatility features
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # Volume features
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']
        
        # Technical indicators using ta library
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['MACD_Histogram'] = ta.trend.MACD(df['Close']).macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Williams %R
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        
        # Average True Range
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Commodity Channel Index
        df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        
        # Money Flow Index
        df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        # Future target (next day's return)
        df['Target'] = df['Returns'].shift(-1)
        
        return df
    
    def prepare_training_data(self, data_dict):
        """Prepare training data from multiple stocks"""
        all_features = []
        
        for symbol, data in data_dict.items():
            if len(data) < 100:  # Skip stocks with insufficient data
                continue
                
            df_features = self.create_technical_features(data)
            df_features['Symbol'] = symbol
            all_features.append(df_features)
        
        # Combine all data
        combined_data = pd.concat(all_features, ignore_index=True)
        
        # Remove rows with NaN values
        combined_data = combined_data.dropna()
        
        # Select feature columns (exclude target and non-numeric columns)
        feature_columns = [col for col in combined_data.columns 
                          if col not in ['Target', 'Symbol', 'Date'] and 
                          combined_data[col].dtype in ['float64', 'int64']]
        
        X = combined_data[feature_columns]
        y = combined_data['Target']
        
        return X, y, feature_columns, combined_data
    
    def train_prediction_model(self, X, y):
        """Train machine learning model for predictions"""
        # Split data chronologically (important for time series)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train models
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        
        # Ensemble prediction (average)
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        # Calculate metrics
        mse = mean_squared_error(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        
        print(f"Model Performance:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.4f}")
        
        # Store the best performing model (let's use gradient boosting)
        self.model = gb_model
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': gb_model,
            'scaler': self.scaler,
            'metrics': {'mse': mse, 'mae': mae, 'r2': r2},
            'feature_importance': feature_importance,
            'predictions': ensemble_pred,
            'actual': y_test.values
        }
    
    def predict_stock_movement(self, symbol, current_data):
        """Predict next day movement for a specific stock"""
        if self.model is None:
            return None
            
        try:
            # Create features for current data
            df_features = self.create_technical_features(current_data)
            
            # Get the latest row (most recent data)
            latest_data = df_features.iloc[-1:].copy()
            
            # Select same features used in training
            feature_columns = [col for col in latest_data.columns 
                              if col not in ['Target', 'Symbol', 'Date'] and 
                              latest_data[col].dtype in ['float64', 'int64']]
            
            X_latest = latest_data[feature_columns]
            
            # Handle any missing features
            for col in self.scaler.feature_names_in_:
                if col not in X_latest.columns:
                    X_latest[col] = 0
            
            # Ensure same order as training
            X_latest = X_latest.reindex(columns=self.scaler.feature_names_in_, fill_value=0)
            
            # Scale and predict
            X_scaled = self.scaler.transform(X_latest)
            prediction = self.model.predict(X_scaled)[0]
            
            # Convert to percentage and direction
            prediction_pct = prediction * 100
            direction = "UP" if prediction > 0.001 else "DOWN" if prediction < -0.001 else "SIDEWAYS"
            confidence = min(abs(prediction) * 1000, 100)  # Scale confidence
            
            return {
                'symbol': symbol.replace('.NS', ''),
                'predicted_return': prediction_pct,
                'direction': direction,
                'confidence': confidence,
                'current_price': current_data['Close'].iloc[-1]
            }
            
        except Exception as e:
            print(f"Error predicting for {symbol}: {str(e)}")
            return None
    
    def save_model(self, model_data, filename='nse_prediction_model.joblib'):
        """Save trained model and scaler"""
        joblib.dump(model_data, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='nse_prediction_model.joblib'):
        """Load trained model and scaler"""
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            print(f"Model loaded from {filename}")
            return model_data
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def get_live_quote(self, symbol):
        """Get live quote from NSE"""
        try:
            nse_symbol = self.nse_symbols_map.get(symbol, symbol.replace('.NS', ''))
            quote = get_quote(nse_symbol)
            return quote
        except Exception as e:
            print(f"Error fetching live quote for {symbol}: {str(e)}")
            return None

# Example usage and model training
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = NSEDataFetcher()
    
    # Fetch 2022 data for top 10 stocks (for demo)
    top_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS'
    ]
    
    print("Fetching 2022 historical data...")
    data_2022 = fetcher.fetch_2022_data(top_stocks)
    
    if data_2022:
        print(f"Successfully fetched data for {len(data_2022)} stocks")
        
        # Prepare training data
        print("Preparing training data...")
        X, y, feature_columns, combined_data = fetcher.prepare_training_data(data_2022)
        
        print(f"Training data shape: {X.shape}")
        print(f"Number of features: {len(feature_columns)}")
        
        # Train model
        print("Training prediction model...")
        model_results = fetcher.train_prediction_model(X, y)
        
        # Save model
        fetcher.save_model({
            'model': model_results['model'],
            'scaler': fetcher.scaler,
            'feature_columns': feature_columns,
            'metrics': model_results['metrics'],
            'feature_importance': model_results['feature_importance']
        })
        
        print("Model training completed successfully!")
        print("\nTop 10 Most Important Features:")
        print(model_results['feature_importance'].head(10))
    
    else:
        print("No data fetched. Please check your internet connection and try again.")