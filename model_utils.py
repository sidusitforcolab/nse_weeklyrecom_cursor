"""
NSE Stock Prediction Model Utilities

This module provides utility functions for model operations, prediction,
and integration with the main application.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import joblib
import os
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ModelPredictor:
    """
    Utility class for making predictions with trained models
    """
    
    def __init__(self, model_path: str = 'nse_prediction_model.joblib'):
        """
        Initialize the model predictor
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model_data = None
        self.is_loaded = False
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained model
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                self.model_data = joblib.load(self.model_path)
                self.is_loaded = True
                return True
            else:
                print(f"Model file not found: {self.model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model metadata
        """
        if not self.is_loaded:
            return {}
        
        return {
            'created_at': self.model_data.get('created_at'),
            'feature_count': len(self.model_data.get('feature_columns', [])),
            'metrics': self.model_data.get('metrics', {}),
            'model_types': list(self.model_data.get('models', {}).keys())
        }
    
    def predict_stock_movement(self, symbol: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Predict stock movement for a given symbol
        
        Args:
            symbol: Stock symbol
            data: Historical OHLCV data
            
        Returns:
            Prediction dictionary or None if prediction fails
        """
        if not self.is_loaded:
            return None
        
        try:
            # Create features using the same process as training
            from nse_data_fetcher import NSEDataFetcher
            fetcher = NSEDataFetcher()
            df_features = fetcher.create_technical_features(data)
            
            # Get the latest row (most recent data)
            latest_data = df_features.iloc[-1:].copy()
            
            # Select same features used in training
            feature_columns = self.model_data.get('feature_columns', [])
            
            # Prepare features
            X_latest = pd.DataFrame(index=[0])
            for col in feature_columns:
                if col in latest_data.columns:
                    X_latest[col] = latest_data[col].iloc[0]
                else:
                    X_latest[col] = 0  # Fill missing features with 0
            
            # Scale features
            scaler = self.model_data['models']['scaler']
            X_scaled = scaler.transform(X_latest)
            
            # Make prediction using the best model (gradient boosting)
            model = self.model_data['models']['gradient_boosting']
            prediction = model.predict(X_scaled)[0]
            
            # Convert to percentage and direction
            prediction_pct = prediction * 100
            direction = self._get_direction(prediction)
            confidence = self._calculate_confidence(prediction, X_latest)
            
            return {
                'symbol': symbol.replace('.NS', ''),
                'predicted_return': prediction_pct,
                'direction': direction,
                'confidence': confidence,
                'current_price': data['Close'].iloc[-1],
                'model_type': 'ML Model (Gradient Boosting)'
            }
            
        except Exception as e:
            print(f"Error predicting for {symbol}: {str(e)}")
            return None
    
    def _get_direction(self, prediction: float) -> str:
        """
        Convert prediction to direction
        
        Args:
            prediction: Raw prediction value
            
        Returns:
            Direction string (UP/DOWN/SIDEWAYS)
        """
        if prediction > 0.001:
            return "UP"
        elif prediction < -0.001:
            return "DOWN"
        else:
            return "SIDEWAYS"
    
    def _calculate_confidence(self, prediction: float, features: pd.DataFrame) -> float:
        """
        Calculate prediction confidence
        
        Args:
            prediction: Raw prediction value
            features: Feature vector used for prediction
            
        Returns:
            Confidence score (0-100)
        """
        # Simple confidence calculation based on prediction magnitude
        # In practice, this could be enhanced with uncertainty quantification
        base_confidence = min(abs(prediction) * 1000, 80)
        
        # Adjust based on feature quality (e.g., if many features are missing)
        feature_quality = 1.0 - (features.isna().sum().sum() / len(features.columns))
        
        confidence = base_confidence * feature_quality
        return min(max(confidence, 20), 95)  # Clamp between 20-95%
    
    def batch_predict(self, symbols_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple stocks
        
        Args:
            symbols_data: Dictionary mapping symbols to their data
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for symbol, data in symbols_data.items():
            prediction = self.predict_stock_movement(symbol, data)
            if prediction:
                predictions.append(prediction)
        
        return predictions

class DataFetcher:
    """
    Utility class for fetching stock data
    """
    
    @staticmethod
    def get_stock_data(symbol: str, period: str = "3mo") -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data if not data.empty else None
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    @staticmethod
    def get_multiple_stocks_data(symbols: List[str], period: str = "3mo") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Data period
            
        Returns:
            Dictionary mapping symbols to their data
        """
        data_dict = {}
        
        for symbol in symbols:
            data = DataFetcher.get_stock_data(symbol, period)
            if data is not None:
                data_dict[symbol] = data
        
        return data_dict
    
    @staticmethod
    def calculate_weekly_returns(data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Calculate weekly returns from stock data
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary with weekly return information
        """
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

class ModelManager:
    """
    Utility class for managing model lifecycle
    """
    
    def __init__(self, model_dir: str = '.'):
        """
        Initialize model manager
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = model_dir
    
    def list_available_models(self) -> List[str]:
        """
        List all available model files
        
        Returns:
            List of model file paths
        """
        model_files = []
        
        for file in os.listdir(self.model_dir):
            if file.endswith('.joblib') and 'model' in file.lower():
                model_files.append(os.path.join(self.model_dir, file))
        
        return model_files
    
    def get_latest_model(self) -> Optional[str]:
        """
        Get the path to the latest model file
        
        Returns:
            Path to latest model or None if no models found
        """
        models = self.list_available_models()
        
        if not models:
            return None
        
        # Sort by modification time
        models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return models[0]
    
    def get_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """
        Get metadata for a specific model
        
        Args:
            model_path: Path to model file
            
        Returns:
            Model metadata dictionary
        """
        try:
            model_data = joblib.load(model_path)
            return {
                'path': model_path,
                'created_at': model_data.get('created_at'),
                'feature_count': len(model_data.get('feature_columns', [])),
                'metrics': model_data.get('metrics', {}),
                'file_size': os.path.getsize(model_path),
                'modified_at': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
            }
        except Exception as e:
            return {'path': model_path, 'error': str(e)}
    
    def compare_models(self, model_paths: List[str]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            model_paths: List of model file paths
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_path in model_paths:
            metadata = self.get_model_metadata(model_path)
            
            if 'error' not in metadata:
                metrics = metadata.get('metrics', {})
                ensemble_metrics = metrics.get('ensemble', {})
                
                comparison_data.append({
                    'Model': os.path.basename(model_path),
                    'Created': metadata.get('created_at', 'Unknown'),
                    'Features': metadata.get('feature_count', 0),
                    'R² Score': ensemble_metrics.get('r2', 0),
                    'Direction Accuracy': ensemble_metrics.get('direction_accuracy', 0),
                    'File Size (MB)': metadata.get('file_size', 0) / (1024 * 1024)
                })
        
        return pd.DataFrame(comparison_data)

class PerformanceTracker:
    """
    Utility class for tracking model performance over time
    """
    
    def __init__(self, tracking_file: str = 'model_performance_log.json'):
        """
        Initialize performance tracker
        
        Args:
            tracking_file: File to store performance logs
        """
        self.tracking_file = tracking_file
        self.performance_log = self.load_performance_log()
    
    def load_performance_log(self) -> List[Dict[str, Any]]:
        """
        Load performance log from file
        
        Returns:
            List of performance records
        """
        try:
            if os.path.exists(self.tracking_file):
                import json
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            print(f"Error loading performance log: {e}")
            return []
    
    def save_performance_log(self) -> None:
        """Save performance log to file"""
        try:
            import json
            with open(self.tracking_file, 'w') as f:
                json.dump(self.performance_log, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving performance log: {e}")
    
    def log_prediction_batch(self, predictions: List[Dict[str, Any]], 
                           actual_results: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Log a batch of predictions
        
        Args:
            predictions: List of prediction dictionaries
            actual_results: List of actual results (optional)
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction_count': len(predictions),
            'predictions': predictions,
            'actual_results': actual_results
        }
        
        self.performance_log.append(log_entry)
        self.save_performance_log()
    
    def calculate_recent_accuracy(self, days: int = 7) -> Optional[float]:
        """
        Calculate accuracy for recent predictions
        
        Args:
            days: Number of days to look back
            
        Returns:
            Accuracy percentage or None if no data
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_logs = [
            log for log in self.performance_log
            if datetime.fromisoformat(log['timestamp']) >= cutoff_date
            and log.get('actual_results') is not None
        ]
        
        if not recent_logs:
            return None
        
        correct_predictions = 0
        total_predictions = 0
        
        for log in recent_logs:
            predictions = log['predictions']
            actual_results = log['actual_results']
            
            for pred, actual in zip(predictions, actual_results):
                if pred['direction'] == actual.get('direction'):
                    correct_predictions += 1
                total_predictions += 1
        
        return (correct_predictions / total_predictions) * 100 if total_predictions > 0 else None

# Utility functions
def get_nse_symbols() -> List[str]:
    """
    Get list of supported NSE symbols
    
    Returns:
        List of NSE stock symbols
    """
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

def get_sector_mapping() -> Dict[str, List[str]]:
    """
    Get sector-wise stock mapping
    
    Returns:
        Dictionary mapping sectors to stock lists
    """
    return {
        'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'INDUSINDBK.NS'],
        'IT Services': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'TECHM.NS', 'HCLTECH.NS'],
        'Pharmaceuticals': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS'],
        'Automotive': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS'],
        'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TATACONSUM.NS'],
        'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'TATAPOWER.NS'],
        'Metals & Mining': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'COALINDIA.NS'],
        'Infrastructure': ['LT.NS', 'POWERGRID.NS', 'NTPC.NS', 'ADANIPORTS.NS'],
        'Financial Services': ['BAJFINANCE.NS', 'BAJAJFINSV.NS', 'SBILIFE.NS'],
        'Consumer Goods': ['TITAN.NS', 'ASIANPAINT.NS', 'ULTRACEMCO.NS', 'SHREECEM.NS'],
        'Telecom': ['BHARTIARTL.NS'],
        'Chemicals': ['UPL.NS', 'GRASIM.NS']
    }

def format_currency(amount: float, currency: str = '₹') -> str:
    """
    Format currency amount
    
    Args:
        amount: Amount to format
        currency: Currency symbol
        
    Returns:
        Formatted currency string
    """
    return f"{currency}{amount:,.2f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage value
    
    Args:
        value: Percentage value
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"

def calculate_returns(start_price: float, end_price: float) -> float:
    """
    Calculate percentage returns
    
    Args:
        start_price: Starting price
        end_price: Ending price
        
    Returns:
        Percentage return
    """
    return ((end_price - start_price) / start_price) * 100

# Configuration management
def load_config(config_file: str = 'model_config.json') -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        'model_path': 'nse_prediction_model.joblib',
        'data_refresh_interval': 300,  # seconds
        'prediction_confidence_threshold': 60,
        'max_symbols_per_batch': 20,
        'cache_duration': 3600  # seconds
    }
    
    try:
        if os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
    except Exception as e:
        print(f"Error loading config: {e}")
    
    return default_config

def save_config(config: Dict[str, Any], config_file: str = 'model_config.json') -> None:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_file: Path to configuration file
    """
    try:
        import json
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")