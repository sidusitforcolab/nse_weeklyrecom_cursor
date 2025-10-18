"""
NSE Stock Prediction Model Training Script

This script handles the training of machine learning models for NSE stock prediction.
It includes data fetching, feature engineering, model training, and model persistence.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from nsepy import get_history
import ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
import os
import json
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

class NSEModelTrainer:
    """
    Comprehensive NSE stock prediction model trainer
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the model trainer
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config = self.load_config(config_file)
        self.nse_symbols_map = self.get_nse_symbol_mapping()
        self.scaler = StandardScaler()
        self.models = {}
        self.training_history = []
        
    def load_config(self, config_file: Optional[str] = None) -> Dict:
        """Load configuration settings"""
        default_config = {
            "model_params": {
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42,
                    "n_jobs": -1
                },
                "gradient_boosting": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42
                }
            },
            "feature_params": {
                "ma_windows": [5, 10, 20, 50],
                "volatility_windows": [10, 20],
                "lag_periods": [1, 2, 3, 5],
                "min_periods_required": 50
            },
            "training_params": {
                "test_size": 0.2,
                "validation_splits": 5,
                "ensemble_weights": [0.4, 0.6]  # RF, GB
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def get_nse_symbol_mapping(self) -> Dict[str, str]:
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
    
    def fetch_nse_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical data from NSE using nsepy with fallback to yfinance
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with historical stock data
        """
        try:
            nse_symbol = self.nse_symbols_map.get(symbol, symbol.replace('.NS', ''))
            print(f"Fetching NSE data for {nse_symbol}...")
            
            data = get_history(symbol=nse_symbol, start=start_date, end=end_date)
            
            if data.empty:
                print(f"NSE data empty for {symbol}, falling back to yfinance...")
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
            print(f"Falling back to yfinance for {symbol}...")
            return self.fetch_yfinance_data(symbol, start_date, end_date)
    
    def fetch_yfinance_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fallback method using yfinance
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with historical stock data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            return data
        except Exception as e:
            print(f"Error fetching yfinance data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_training_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Fetch training data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping symbols to their data
        """
        all_data = {}
        
        print(f"Fetching data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}...")
        
        for i, symbol in enumerate(symbols):
            print(f"Progress: {i+1}/{len(symbols)} - {symbol}")
            data = self.fetch_nse_historical_data(symbol, start_date, end_date)
            
            if not data.empty and len(data) >= self.config["feature_params"]["min_periods_required"]:
                all_data[symbol] = data
            else:
                print(f"Insufficient data for {symbol} (got {len(data)} periods)")
                
        print(f"Successfully fetched data for {len(all_data)} symbols")
        return all_data
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical indicators
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with technical features
        """
        df = data.copy()
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Moving averages
        ma_windows = self.config["feature_params"]["ma_windows"]
        available_windows = [w for w in ma_windows if w <= len(df)]
        
        for window in available_windows:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, min_periods=1).mean()
            df[f'Price_SMA_{window}_Ratio'] = df['Close'] / df[f'SMA_{window}']
        
        # Volatility features
        vol_windows = self.config["feature_params"]["volatility_windows"]
        for window in vol_windows:
            if len(df) >= window:
                df[f'Volatility_{window}'] = df['Returns'].rolling(window=window, min_periods=1).std()
        
        # Volume features
        if len(df) >= 10:
            df['Volume_SMA_10'] = df['Volume'].rolling(window=10, min_periods=1).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']
        
        # Technical indicators with error handling
        self._add_technical_indicators(df)
        
        # Lag features
        lag_periods = self.config["feature_params"]["lag_periods"]
        available_lags = [lag for lag in lag_periods if lag < len(df)]
        
        for lag in available_lags:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        # Target variable (next day's return)
        df['Target'] = df['Returns'].shift(-1)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> None:
        """Add technical indicators with error handling"""
        
        # RSI
        try:
            if len(df) >= 14:
                df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=min(14, len(df)-1)).rsi()
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            df['RSI'] = np.nan
        
        # MACD
        try:
            if len(df) >= 26:
                macd = ta.trend.MACD(df['Close'])
                df['MACD'] = macd.macd()
                df['MACD_Signal'] = macd.macd_signal()
                df['MACD_Histogram'] = macd.macd_diff()
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            df['MACD'] = np.nan
            df['MACD_Signal'] = np.nan
            df['MACD_Histogram'] = np.nan
        
        # Bollinger Bands
        try:
            if len(df) >= 20:
                bb = ta.volatility.BollingerBands(df['Close'], window=min(20, len(df)-1))
                df['BB_Upper'] = bb.bollinger_hband()
                df['BB_Lower'] = bb.bollinger_lband()
                df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
                df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            df['BB_Upper'] = np.nan
            df['BB_Lower'] = np.nan
            df['BB_Width'] = np.nan
            df['BB_Position'] = np.nan
        
        # Additional indicators
        self._add_momentum_indicators(df)
        self._add_volatility_indicators(df)
        self._add_volume_indicators(df)
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> None:
        """Add momentum-based indicators"""
        try:
            if len(df) >= 14:
                # Stochastic Oscillator
                stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 
                                                       window=min(14, len(df)-1))
                df['Stoch_K'] = stoch.stoch()
                df['Stoch_D'] = stoch.stoch_signal()
                
                # Williams %R
                df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'], 
                                                                 lbp=min(14, len(df)-1)).williams_r()
        except Exception as e:
            print(f"Error calculating momentum indicators: {e}")
            df['Stoch_K'] = np.nan
            df['Stoch_D'] = np.nan
            df['Williams_R'] = np.nan
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> None:
        """Add volatility-based indicators"""
        try:
            if len(df) >= 14:
                # Average True Range
                df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 
                                                          window=min(14, len(df)-1)).average_true_range()
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            df['ATR'] = np.nan
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> None:
        """Add volume-based indicators"""
        try:
            if len(df) >= 14:
                # Money Flow Index
                df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], 
                                                  window=min(14, len(df)-1)).money_flow_index()
        except Exception as e:
            print(f"Error calculating MFI: {e}")
            df['MFI'] = np.nan
        
        try:
            if len(df) >= 20:
                # Commodity Channel Index
                df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], 
                                                 window=min(20, len(df)-1)).cci()
        except Exception as e:
            print(f"Error calculating CCI: {e}")
            df['CCI'] = np.nan
    
    def prepare_training_data(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare training data from multiple stocks
        
        Args:
            data_dict: Dictionary mapping symbols to their data
            
        Returns:
            Tuple of (features, target, feature_columns)
        """
        all_features = []
        
        print("Creating technical features for all stocks...")
        
        for symbol, data in data_dict.items():
            if len(data) < self.config["feature_params"]["min_periods_required"]:
                print(f"Skipping {symbol}: insufficient data ({len(data)} periods)")
                continue
                
            print(f"Processing features for {symbol}...")
            df_features = self.create_technical_features(data)
            df_features['Symbol'] = symbol
            all_features.append(df_features)
        
        if not all_features:
            raise ValueError("No valid data available for training")
        
        # Combine all data
        print("Combining data from all stocks...")
        combined_data = pd.concat(all_features, ignore_index=True)
        
        # Remove rows with NaN values
        initial_rows = len(combined_data)
        combined_data = combined_data.dropna()
        print(f"Removed {initial_rows - len(combined_data)} rows with missing values")
        
        # Select feature columns
        feature_columns = [col for col in combined_data.columns 
                          if col not in ['Target', 'Symbol', 'Date'] and 
                          combined_data[col].dtype in ['float64', 'int64']]
        
        X = combined_data[feature_columns]
        y = combined_data['Target']
        
        print(f"Final dataset: {len(X)} samples, {len(feature_columns)} features")
        return X, y, feature_columns
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train multiple models and create ensemble
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with model results
        """
        print("Starting model training...")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=self.config["training_params"]["validation_splits"])
        
        # Split data chronologically
        test_size = self.config["training_params"]["test_size"]
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_params = self.config["model_params"]["random_forest"]
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        # Train Gradient Boosting
        print("Training Gradient Boosting...")
        gb_params = self.config["model_params"]["gradient_boosting"]
        gb_model = GradientBoostingRegressor(**gb_params)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        
        # Create ensemble prediction
        ensemble_weights = self.config["training_params"]["ensemble_weights"]
        ensemble_pred = (rf_pred * ensemble_weights[0] + gb_pred * ensemble_weights[1])
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, rf_pred, gb_pred, ensemble_pred)
        
        # Feature importance (using GB model as primary)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store models
        self.models = {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'scaler': self.scaler
        }
        
        # Training history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X_train),
            'n_features': len(X.columns),
            'metrics': metrics,
            'config': self.config
        }
        self.training_history.append(training_record)
        
        return {
            'models': self.models,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': {
                'rf': rf_pred,
                'gb': gb_pred,
                'ensemble': ensemble_pred
            },
            'actual': y_test.values,
            'feature_columns': X.columns.tolist(),
            'training_history': training_record
        }
    
    def _calculate_metrics(self, y_true: pd.Series, rf_pred: np.ndarray, 
                          gb_pred: np.ndarray, ensemble_pred: np.ndarray) -> Dict:
        """Calculate comprehensive metrics for all models"""
        
        def calc_direction_accuracy(actual, predicted):
            actual_direction = np.sign(actual)
            predicted_direction = np.sign(predicted)
            return np.mean(actual_direction == predicted_direction)
        
        metrics = {}
        
        for name, pred in [('rf', rf_pred), ('gb', gb_pred), ('ensemble', ensemble_pred)]:
            metrics[name] = {
                'mse': mean_squared_error(y_true, pred),
                'mae': mean_absolute_error(y_true, pred),
                'r2': r2_score(y_true, pred),
                'direction_accuracy': calc_direction_accuracy(y_true, pred)
            }
        
        # Print results
        print("\n" + "="*50)
        print("MODEL TRAINING RESULTS")
        print("="*50)
        
        for name, metric in metrics.items():
            print(f"\n{name.upper()} Model:")
            print(f"  MSE: {metric['mse']:.6f}")
            print(f"  MAE: {metric['mae']:.6f}")
            print(f"  R²: {metric['r2']:.4f}")
            print(f"  Direction Accuracy: {metric['direction_accuracy']:.4f}")
        
        return metrics
    
    def save_model(self, results: Dict, filename: str = 'nse_prediction_model.joblib') -> None:
        """
        Save trained model and metadata
        
        Args:
            results: Training results dictionary
            filename: Output filename
        """
        model_data = {
            'models': results['models'],
            'feature_columns': results['feature_columns'],
            'metrics': results['metrics'],
            'feature_importance': results['feature_importance'],
            'training_history': self.training_history,
            'config': self.config,
            'created_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filename)
        print(f"\nModel saved as {filename}")
        
        # Save training summary
        summary_file = filename.replace('.joblib', '_summary.json')
        summary = {
            'metrics': results['metrics'],
            'feature_count': len(results['feature_columns']),
            'training_samples': results['training_history']['n_samples'],
            'created_at': model_data['created_at']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Training summary saved as {summary_file}")
    
    def train_full_pipeline(self, symbols: List[str] = None, 
                           start_date: datetime = None, 
                           end_date: datetime = None) -> Dict:
        """
        Complete training pipeline
        
        Args:
            symbols: List of symbols to train on
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            Training results dictionary
        """
        # Default parameters
        if symbols is None:
            symbols = list(self.nse_symbols_map.keys())[:15]  # Top 15 for faster training
        
        if start_date is None:
            start_date = datetime(2022, 1, 1)
        
        if end_date is None:
            end_date = datetime(2022, 12, 31)
        
        print("="*60)
        print("NSE STOCK PREDICTION MODEL TRAINING")
        print("="*60)
        print(f"Training period: {start_date.date()} to {end_date.date()}")
        print(f"Number of symbols: {len(symbols)}")
        print(f"Symbols: {', '.join([s.replace('.NS', '') for s in symbols[:10]])}{'...' if len(symbols) > 10 else ''}")
        print("="*60)
        
        # Step 1: Fetch data
        data_dict = self.fetch_training_data(symbols, start_date, end_date)
        
        if not data_dict:
            raise ValueError("No data available for training")
        
        # Step 2: Prepare features
        X, y, feature_columns = self.prepare_training_data(data_dict)
        
        # Step 3: Train models
        results = self.train_models(X, y)
        
        # Step 4: Save model
        self.save_model(results)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return results

def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NSE Stock Prediction Model')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--symbols', type=str, nargs='+', help='Stock symbols to train on')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='nse_prediction_model.joblib', 
                       help='Output model filename')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = NSEModelTrainer(args.config)
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    
    # Train model
    try:
        results = trainer.train_full_pipeline(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Save with custom filename
        if args.output != 'nse_prediction_model.joblib':
            trainer.save_model(results, args.output)
            
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()