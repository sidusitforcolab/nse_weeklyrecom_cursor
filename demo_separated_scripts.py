#!/usr/bin/env python3
"""
Demo script showing how to use the separated modeling and evaluation scripts

This script demonstrates:
1. Training a new model using model_trainer.py
2. Evaluating the model using model_evaluator.py
3. Making predictions using model_utils.py
"""

import subprocess
import sys
import os
from datetime import datetime
from model_utils import ModelPredictor, DataFetcher, ModelManager

def run_training_demo():
    """Demonstrate model training"""
    print("🚀 DEMO: Model Training")
    print("=" * 50)
    
    # Train a model with a subset of stocks for faster demo
    training_symbols = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS'
    ]
    
    print(f"Training model with symbols: {[s.replace('.NS', '') for s in training_symbols]}")
    
    try:
        result = subprocess.run([
            sys.executable, 'model_trainer.py',
            '--symbols'] + training_symbols + [
            '--start-date', '2022-01-01',
            '--end-date', '2022-12-31',
            '--output', 'demo_model.joblib'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            print("\nTraining output:")
            print(result.stdout[-500:])  # Last 500 characters
        else:
            print("❌ Model training failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training timed out")
        return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False
    
    return True

def run_evaluation_demo():
    """Demonstrate model evaluation"""
    print("\n🔍 DEMO: Model Evaluation")
    print("=" * 50)
    
    if not os.path.exists('demo_model.joblib'):
        print("❌ Demo model not found. Run training first.")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, 'model_evaluator.py',
            '--model', 'demo_model.joblib',
            '--output-dir', 'demo_evaluation'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Model evaluation completed successfully!")
            print("\nEvaluation output:")
            print(result.stdout[-500:])  # Last 500 characters
            
            # List generated files
            if os.path.exists('demo_evaluation'):
                files = os.listdir('demo_evaluation')
                print(f"\n📁 Generated files: {files}")
        else:
            print("❌ Model evaluation failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Evaluation timed out")
        return False
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return False
    
    return True

def run_prediction_demo():
    """Demonstrate making predictions"""
    print("\n🔮 DEMO: Making Predictions")
    print("=" * 50)
    
    # Initialize predictor with demo model
    predictor = ModelPredictor('demo_model.joblib')
    
    if not predictor.is_loaded:
        print("❌ Could not load demo model")
        return False
    
    # Display model info
    model_info = predictor.get_model_info()
    print(f"Model created: {model_info.get('created_at', 'Unknown')}")
    print(f"Features: {model_info.get('feature_count', 0)}")
    
    # Make predictions for a few stocks
    test_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
    
    print(f"\nMaking predictions for: {[s.replace('.NS', '') for s in test_symbols]}")
    
    for symbol in test_symbols:
        print(f"\n📊 {symbol.replace('.NS', '')}:")
        
        # Fetch recent data
        data = DataFetcher.get_stock_data(symbol, "3mo")
        
        if data is not None:
            # Make prediction
            prediction = predictor.predict_stock_movement(symbol, data)
            
            if prediction:
                print(f"  Current Price: ₹{prediction['current_price']:.2f}")
                print(f"  Predicted Return: {prediction['predicted_return']:.2f}%")
                print(f"  Direction: {prediction['direction']}")
                print(f"  Confidence: {prediction['confidence']:.1f}%")
            else:
                print("  ❌ Prediction failed")
        else:
            print("  ❌ Could not fetch data")
    
    return True

def run_model_management_demo():
    """Demonstrate model management utilities"""
    print("\n📁 DEMO: Model Management")
    print("=" * 50)
    
    manager = ModelManager()
    
    # List available models
    models = manager.list_available_models()
    print(f"Available models: {len(models)}")
    
    for model_path in models:
        metadata = manager.get_model_metadata(model_path)
        print(f"\n📄 {os.path.basename(model_path)}:")
        print(f"  Created: {metadata.get('created_at', 'Unknown')}")
        print(f"  Features: {metadata.get('feature_count', 0)}")
        print(f"  Size: {metadata.get('file_size', 0) / 1024:.1f} KB")
    
    # Compare models if multiple exist
    if len(models) > 1:
        print("\n⚖️ Model Comparison:")
        comparison_df = manager.compare_models(models)
        print(comparison_df.to_string(index=False))
    
    return True

def cleanup_demo_files():
    """Clean up demo files"""
    print("\n🧹 Cleaning up demo files...")
    
    files_to_remove = ['demo_model.joblib', 'demo_model_summary.json']
    dirs_to_remove = ['demo_evaluation']
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"  Removed: {file}")
    
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            import shutil
            shutil.rmtree(dir_name)
            print(f"  Removed directory: {dir_name}")

def main():
    """Run the complete demo"""
    print("🎯 NSE MODEL SEPARATION DEMO")
    print("=" * 60)
    print("This demo shows the separated modeling and evaluation scripts")
    print("=" * 60)
    
    try:
        # Step 1: Training
        if not run_training_demo():
            print("❌ Demo failed at training step")
            return
        
        # Step 2: Evaluation
        if not run_evaluation_demo():
            print("❌ Demo failed at evaluation step")
            return
        
        # Step 3: Predictions
        if not run_prediction_demo():
            print("❌ Demo failed at prediction step")
            return
        
        # Step 4: Model Management
        if not run_model_management_demo():
            print("❌ Demo failed at model management step")
            return
        
        print("\n✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("All separated scripts are working correctly!")
        
        # Ask if user wants to clean up
        response = input("\nClean up demo files? (y/n): ").lower().strip()
        if response == 'y':
            cleanup_demo_files()
        
    except KeyboardInterrupt:
        print("\n\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    main()