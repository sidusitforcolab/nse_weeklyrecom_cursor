# NSE Stock Prediction - Separated Architecture

## 🏗️ Architecture Overview

The NSE Stock Prediction system has been refactored into a modular, separated architecture for better maintainability, scalability, and development workflow.

## 📁 File Structure

```
├── app.py                          # Main Streamlit application
├── model_trainer.py               # Standalone model training script
├── model_evaluator.py             # Standalone model evaluation script
├── model_utils.py                 # Utility functions and classes
├── nse_data_fetcher.py            # Legacy data fetcher (kept for compatibility)
├── model_config.json             # Configuration file
├── demo_separated_scripts.py      # Demo script showing usage
├── requirements.txt               # Dependencies
└── README.md                      # Main documentation
```

## 🔧 Core Components

### 1. **Model Training (`model_trainer.py`)**
Standalone script for training machine learning models.

**Features:**
- ✅ Complete training pipeline from data fetching to model saving
- ✅ Configurable parameters via JSON config file
- ✅ Support for multiple model types (Random Forest, Gradient Boosting)
- ✅ Comprehensive feature engineering (50+ technical indicators)
- ✅ Time series validation and ensemble methods
- ✅ Command-line interface for automation

**Usage:**
```bash
# Basic training with default parameters
python model_trainer.py

# Custom training with specific symbols and dates
python model_trainer.py --symbols RELIANCE.NS TCS.NS HDFCBANK.NS \
                       --start-date 2022-01-01 \
                       --end-date 2022-12-31 \
                       --output custom_model.joblib

# Training with custom configuration
python model_trainer.py --config custom_config.json
```

**Key Classes:**
- `NSEModelTrainer`: Main training orchestrator
- Handles data fetching, feature engineering, model training, and persistence

### 2. **Model Evaluation (`model_evaluator.py`)**
Standalone script for comprehensive model evaluation and analysis.

**Features:**
- ✅ Comprehensive performance metrics (R², MAE, MSE, Direction Accuracy)
- ✅ Feature importance analysis and visualization
- ✅ Model comparison and ranking
- ✅ HTML and JSON report generation
- ✅ Interactive plots and charts
- ✅ Stability and trend analysis

**Usage:**
```bash
# Basic evaluation
python model_evaluator.py

# Evaluate specific model
python model_evaluator.py --model custom_model.joblib \
                         --output-dir evaluation_results

# Skip report or plot generation
python model_evaluator.py --no-report --no-plots
```

**Key Classes:**
- `NSEModelEvaluator`: Main evaluation orchestrator
- Generates comprehensive reports and visualizations

### 3. **Model Utilities (`model_utils.py`)**
Shared utility functions and classes used across the system.

**Key Classes:**

#### `ModelPredictor`
- Loads trained models and makes predictions
- Handles feature preparation and scaling
- Provides confidence scoring

#### `DataFetcher`
- Fetches stock data from Yahoo Finance
- Calculates weekly returns and other metrics
- Handles multiple stock data fetching

#### `ModelManager`
- Lists and compares available models
- Provides model metadata and statistics
- Handles model lifecycle management

#### `PerformanceTracker`
- Tracks prediction accuracy over time
- Logs prediction batches and results
- Calculates recent performance metrics

**Utility Functions:**
- `get_nse_symbols()`: Returns list of supported NSE symbols
- `get_sector_mapping()`: Returns sector-wise stock categorization
- `format_currency()`, `format_percentage()`: Formatting helpers
- `load_config()`, `save_config()`: Configuration management

### 4. **Main Application (`app.py`)**
Enhanced Streamlit application using the separated components.

**Key Changes:**
- ✅ Uses `ModelPredictor` for ML predictions
- ✅ Integrates with training and evaluation scripts via subprocess
- ✅ Enhanced model management interface
- ✅ Real-time model status and metrics
- ✅ Improved error handling and user feedback

## 🚀 Usage Examples

### Training a New Model

```python
from model_trainer import NSEModelTrainer
from datetime import datetime

# Initialize trainer
trainer = NSEModelTrainer('model_config.json')

# Train with specific symbols
symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
results = trainer.train_full_pipeline(
    symbols=symbols,
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31)
)

print(f"Model R² Score: {results['metrics']['ensemble']['r2']:.4f}")
```

### Evaluating a Model

```python
from model_evaluator import NSEModelEvaluator

# Initialize evaluator
evaluator = NSEModelEvaluator('nse_prediction_model.joblib')

# Run comprehensive evaluation
results = evaluator.run_comprehensive_evaluation(
    create_report=True,
    create_plots=True
)

print("Evaluation completed!")
print(f"Report saved: {results.get('report_file')}")
```

### Making Predictions

```python
from model_utils import ModelPredictor, DataFetcher

# Initialize predictor
predictor = ModelPredictor('nse_prediction_model.joblib')

# Fetch data and make prediction
data = DataFetcher.get_stock_data('RELIANCE.NS', '3mo')
prediction = predictor.predict_stock_movement('RELIANCE.NS', data)

print(f"Prediction: {prediction['direction']}")
print(f"Confidence: {prediction['confidence']:.1f}%")
```

### Model Management

```python
from model_utils import ModelManager

# Initialize manager
manager = ModelManager()

# List available models
models = manager.list_available_models()
print(f"Available models: {len(models)}")

# Compare models
comparison = manager.compare_models(models)
print(comparison)
```

## ⚙️ Configuration

### Model Configuration (`model_config.json`)

```json
{
  "model_params": {
    "random_forest": {
      "n_estimators": 100,
      "max_depth": 10,
      "random_state": 42
    },
    "gradient_boosting": {
      "n_estimators": 100,
      "max_depth": 6,
      "learning_rate": 0.1
    }
  },
  "feature_params": {
    "ma_windows": [5, 10, 20, 50],
    "volatility_windows": [10, 20],
    "lag_periods": [1, 2, 3, 5]
  },
  "training_params": {
    "test_size": 0.2,
    "validation_splits": 5,
    "ensemble_weights": [0.4, 0.6]
  }
}
```

## 🔄 Workflow Integration

### Development Workflow

1. **Model Development**
   ```bash
   # Train new model
   python model_trainer.py --start-date 2022-01-01 --end-date 2022-12-31
   
   # Evaluate model
   python model_evaluator.py
   
   # Test predictions
   python demo_separated_scripts.py
   ```

2. **Production Deployment**
   ```bash
   # Run Streamlit app
   streamlit run app.py
   ```

3. **Model Updates**
   ```bash
   # Retrain with new data
   python model_trainer.py --start-date 2023-01-01 --end-date 2023-12-31
   
   # Compare with existing models
   python model_evaluator.py --model new_model.joblib
   ```

### Automation Scripts

The separated architecture enables easy automation:

```bash
#!/bin/bash
# Daily model retraining script
python model_trainer.py --start-date $(date -d "1 year ago" +%Y-%m-%d) \
                       --end-date $(date +%Y-%m-%d) \
                       --output daily_model_$(date +%Y%m%d).joblib

# Evaluate new model
python model_evaluator.py --model daily_model_$(date +%Y%m%d).joblib
```

## 📊 Benefits of Separation

### 1. **Modularity**
- Each component has a single responsibility
- Easy to test and debug individual components
- Clear interfaces between modules

### 2. **Scalability**
- Training can run on separate infrastructure
- Evaluation can be scheduled independently
- Easy to add new model types or evaluation metrics

### 3. **Maintainability**
- Changes to training logic don't affect the main app
- Easier to version and track model experiments
- Clear separation of concerns

### 4. **Flexibility**
- Can use different training configurations
- Easy to compare multiple models
- Support for batch processing and automation

### 5. **Performance**
- Training doesn't block the main application
- Evaluation can run asynchronously
- Better resource utilization

## 🧪 Testing

### Running the Demo

```bash
# Run complete demo showing all separated components
python demo_separated_scripts.py
```

The demo will:
1. Train a model with sample data
2. Evaluate the trained model
3. Make predictions using the model
4. Demonstrate model management utilities

### Unit Testing

Each module can be tested independently:

```python
# Test model training
python -m pytest test_model_trainer.py

# Test model evaluation
python -m pytest test_model_evaluator.py

# Test utilities
python -m pytest test_model_utils.py
```

## 🔧 Troubleshooting

### Common Issues

1. **Training Timeout**
   - Reduce the number of symbols or date range
   - Increase timeout in subprocess calls
   - Check system resources

2. **Model Loading Errors**
   - Ensure model file exists and is not corrupted
   - Check file permissions
   - Verify model compatibility

3. **Data Fetching Issues**
   - Check internet connection
   - Verify symbol names are correct
   - Handle API rate limits

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run training with debug info
trainer = NSEModelTrainer()
trainer.train_full_pipeline()
```

## 🚀 Future Enhancements

### Planned Features

1. **Distributed Training**
   - Support for multiple GPUs/machines
   - Parallel data processing
   - Cloud-based training

2. **Advanced Evaluation**
   - Backtesting framework
   - Walk-forward analysis
   - Monte Carlo simulations

3. **Model Registry**
   - Centralized model storage
   - Version control for models
   - A/B testing framework

4. **Real-time Pipeline**
   - Streaming data processing
   - Online learning capabilities
   - Real-time model updates

### Extension Points

The architecture is designed for easy extension:

- Add new model types in `model_trainer.py`
- Add new evaluation metrics in `model_evaluator.py`
- Add new utilities in `model_utils.py`
- Integrate with MLOps tools (MLflow, Kubeflow, etc.)

## 📞 Support

For issues or questions about the separated architecture:

1. Check the demo script: `python demo_separated_scripts.py`
2. Review the configuration file: `model_config.json`
3. Check logs and error messages
4. Refer to individual script help: `python model_trainer.py --help`

---

**Last Updated**: October 2024  
**Architecture Version**: 2.0  
**Compatibility**: Python 3.8+