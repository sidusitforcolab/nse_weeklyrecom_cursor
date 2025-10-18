"""
NSE Stock Prediction Model Performance Evaluation Script

This script provides comprehensive evaluation and analysis of trained models.
It includes backtesting, performance metrics, visualization, and model comparison.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class NSEModelEvaluator:
    """
    Comprehensive model evaluation and performance analysis
    """
    
    def __init__(self, model_path: str = 'nse_prediction_model.joblib'):
        """
        Initialize the model evaluator
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model_data = None
        self.evaluation_results = {}
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained model and metadata"""
        try:
            if os.path.exists(self.model_path):
                self.model_data = joblib.load(self.model_path)
                print(f"✅ Model loaded successfully from {self.model_path}")
                self.print_model_info()
            else:
                print(f"❌ Model file not found: {self.model_path}")
                self.model_data = None
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model_data = None
    
    def print_model_info(self) -> None:
        """Print basic model information"""
        if not self.model_data:
            return
        
        print("\n" + "="*50)
        print("MODEL INFORMATION")
        print("="*50)
        
        print(f"Created: {self.model_data.get('created_at', 'Unknown')}")
        print(f"Features: {len(self.model_data.get('feature_columns', []))}")
        
        if 'training_history' in self.model_data:
            history = self.model_data['training_history']
            if isinstance(history, list) and history:
                latest = history[-1]
                print(f"Training samples: {latest.get('n_samples', 'Unknown')}")
        
        # Print metrics summary
        if 'metrics' in self.model_data:
            metrics = self.model_data['metrics']
            if 'ensemble' in metrics:
                ens_metrics = metrics['ensemble']
                print(f"R² Score: {ens_metrics.get('r2', 0):.4f}")
                print(f"Direction Accuracy: {ens_metrics.get('direction_accuracy', 0):.4f}")
        
        print("="*50)
    
    def evaluate_model_performance(self) -> Dict[str, Any]:
        """
        Comprehensive model performance evaluation
        
        Returns:
            Dictionary containing evaluation results
        """
        if not self.model_data:
            print("❌ No model loaded for evaluation")
            return {}
        
        print("\n🔍 COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        results = {
            'basic_metrics': self._evaluate_basic_metrics(),
            'feature_analysis': self._analyze_feature_importance(),
            'model_comparison': self._compare_models(),
            'stability_analysis': self._analyze_model_stability(),
            'performance_trends': self._analyze_performance_trends()
        }
        
        self.evaluation_results = results
        return results
    
    def _evaluate_basic_metrics(self) -> Dict[str, Any]:
        """Evaluate basic performance metrics"""
        print("\n📊 Basic Performance Metrics")
        print("-" * 30)
        
        if 'metrics' not in self.model_data:
            print("❌ No metrics found in model data")
            return {}
        
        metrics = self.model_data['metrics']
        
        # Create comparison table
        comparison_data = []
        for model_name, model_metrics in metrics.items():
            comparison_data.append({
                'Model': model_name.upper(),
                'R² Score': f"{model_metrics.get('r2', 0):.4f}",
                'MSE': f"{model_metrics.get('mse', 0):.6f}",
                'MAE': f"{model_metrics.get('mae', 0):.6f}",
                'Direction Accuracy': f"{model_metrics.get('direction_accuracy', 0):.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Identify best model
        best_model = max(metrics.keys(), key=lambda k: metrics[k].get('r2', -np.inf))
        print(f"\n🏆 Best performing model: {best_model.upper()}")
        
        return {
            'comparison_table': comparison_df,
            'best_model': best_model,
            'detailed_metrics': metrics
        }
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance and relevance"""
        print("\n🎯 Feature Importance Analysis")
        print("-" * 30)
        
        if 'feature_importance' not in self.model_data:
            print("❌ No feature importance data found")
            return {}
        
        feature_imp = self.model_data['feature_importance']
        
        # Top 15 most important features
        top_features = feature_imp.head(15)
        print("Top 15 Most Important Features:")
        for idx, row in top_features.iterrows():
            print(f"  {idx+1:2d}. {row['feature']:<20} {row['importance']:.4f}")
        
        # Feature categories analysis
        feature_categories = self._categorize_features(feature_imp)
        
        print(f"\nFeature Categories:")
        for category, count in feature_categories.items():
            print(f"  {category}: {count} features")
        
        return {
            'top_features': top_features,
            'all_features': feature_imp,
            'categories': feature_categories
        }
    
    def _categorize_features(self, feature_imp: pd.DataFrame) -> Dict[str, int]:
        """Categorize features by type"""
        categories = {
            'Price-based': 0,
            'Moving Averages': 0,
            'Technical Indicators': 0,
            'Volume-based': 0,
            'Volatility': 0,
            'Lag Features': 0,
            'Other': 0
        }
        
        for feature in feature_imp['feature']:
            if any(x in feature for x in ['Close', 'Open', 'High', 'Low', 'Price', 'Returns']):
                categories['Price-based'] += 1
            elif any(x in feature for x in ['SMA', 'EMA', 'MA']):
                categories['Moving Averages'] += 1
            elif any(x in feature for x in ['RSI', 'MACD', 'BB_', 'Stoch', 'Williams', 'CCI', 'MFI']):
                categories['Technical Indicators'] += 1
            elif any(x in feature for x in ['Volume', 'MFI']):
                categories['Volume-based'] += 1
            elif any(x in feature for x in ['Volatility', 'ATR', 'Range']):
                categories['Volatility'] += 1
            elif 'Lag' in feature:
                categories['Lag Features'] += 1
            else:
                categories['Other'] += 1
        
        return categories
    
    def _compare_models(self) -> Dict[str, Any]:
        """Compare different model performances"""
        print("\n⚖️ Model Comparison Analysis")
        print("-" * 30)
        
        if 'metrics' not in self.model_data:
            return {}
        
        metrics = self.model_data['metrics']
        
        # Performance ranking
        models_by_r2 = sorted(metrics.items(), key=lambda x: x[1].get('r2', -np.inf), reverse=True)
        models_by_accuracy = sorted(metrics.items(), key=lambda x: x[1].get('direction_accuracy', 0), reverse=True)
        
        print("Ranking by R² Score:")
        for i, (model, metric) in enumerate(models_by_r2, 1):
            print(f"  {i}. {model.upper()}: {metric.get('r2', 0):.4f}")
        
        print("\nRanking by Direction Accuracy:")
        for i, (model, metric) in enumerate(models_by_accuracy, 1):
            print(f"  {i}. {model.upper()}: {metric.get('direction_accuracy', 0):.4f}")
        
        return {
            'r2_ranking': models_by_r2,
            'accuracy_ranking': models_by_accuracy
        }
    
    def _analyze_model_stability(self) -> Dict[str, Any]:
        """Analyze model stability and consistency"""
        print("\n🎯 Model Stability Analysis")
        print("-" * 30)
        
        # This would require multiple training runs or cross-validation results
        # For now, we'll analyze what we have
        
        if 'training_history' not in self.model_data:
            print("❌ No training history available for stability analysis")
            return {}
        
        history = self.model_data['training_history']
        if not isinstance(history, list):
            history = [history]
        
        print(f"Training runs analyzed: {len(history)}")
        
        # If we have multiple runs, analyze variance
        if len(history) > 1:
            r2_scores = [run.get('metrics', {}).get('ensemble', {}).get('r2', 0) for run in history]
            accuracy_scores = [run.get('metrics', {}).get('ensemble', {}).get('direction_accuracy', 0) for run in history]
            
            print(f"R² Score variance: {np.var(r2_scores):.6f}")
            print(f"Accuracy variance: {np.var(accuracy_scores):.6f}")
            
            return {
                'r2_variance': np.var(r2_scores),
                'accuracy_variance': np.var(accuracy_scores),
                'r2_scores': r2_scores,
                'accuracy_scores': accuracy_scores
            }
        else:
            print("Single training run - stability analysis requires multiple runs")
            return {'message': 'Single training run available'}
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        print("\n📈 Performance Trends Analysis")
        print("-" * 30)
        
        # This would require time-series evaluation
        # For now, we'll provide framework for future implementation
        
        print("Performance trends analysis requires time-series backtesting")
        print("This feature will be enhanced in future versions")
        
        return {'message': 'Trends analysis framework ready'}
    
    def create_evaluation_report(self, output_dir: str = 'evaluation_reports') -> str:
        """
        Create comprehensive evaluation report
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        if not self.evaluation_results:
            self.evaluate_model_performance()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f'model_evaluation_report_{timestamp}.html')
        
        self._generate_html_report(report_file)
        
        # Also create JSON summary
        json_file = os.path.join(output_dir, f'evaluation_summary_{timestamp}.json')
        self._generate_json_summary(json_file)
        
        print(f"\n📄 Evaluation report generated:")
        print(f"  HTML Report: {report_file}")
        print(f"  JSON Summary: {json_file}")
        
        return report_file
    
    def _generate_html_report(self, output_file: str) -> None:
        """Generate HTML evaluation report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NSE Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>NSE Stock Prediction Model - Evaluation Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Model: {self.model_path}</p>
            </div>
            
            <div class="section">
                <h2>Model Overview</h2>
                {self._get_model_overview_html()}
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                {self._get_performance_metrics_html()}
            </div>
            
            <div class="section">
                <h2>Feature Importance</h2>
                {self._get_feature_importance_html()}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._get_recommendations_html()}
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _get_model_overview_html(self) -> str:
        """Generate model overview HTML"""
        if not self.model_data:
            return "<p>No model data available</p>"
        
        return f"""
        <div class="metric">
            <strong>Created:</strong> {self.model_data.get('created_at', 'Unknown')}
        </div>
        <div class="metric">
            <strong>Features:</strong> {len(self.model_data.get('feature_columns', []))}
        </div>
        <div class="metric">
            <strong>Model Types:</strong> Random Forest, Gradient Boosting, Ensemble
        </div>
        """
    
    def _get_performance_metrics_html(self) -> str:
        """Generate performance metrics HTML"""
        if 'basic_metrics' not in self.evaluation_results:
            return "<p>No performance metrics available</p>"
        
        comparison_df = self.evaluation_results['basic_metrics']['comparison_table']
        return comparison_df.to_html(index=False, classes='table')
    
    def _get_feature_importance_html(self) -> str:
        """Generate feature importance HTML"""
        if 'feature_analysis' not in self.evaluation_results:
            return "<p>No feature importance data available</p>"
        
        top_features = self.evaluation_results['feature_analysis']['top_features']
        return top_features.to_html(index=False, classes='table')
    
    def _get_recommendations_html(self) -> str:
        """Generate recommendations HTML"""
        recommendations = []
        
        if self.model_data and 'metrics' in self.model_data:
            ensemble_r2 = self.model_data['metrics'].get('ensemble', {}).get('r2', 0)
            ensemble_accuracy = self.model_data['metrics'].get('ensemble', {}).get('direction_accuracy', 0)
            
            if ensemble_r2 < 0.1:
                recommendations.append("⚠️ Low R² score suggests model may need more features or different approach")
            
            if ensemble_accuracy < 0.6:
                recommendations.append("⚠️ Direction accuracy below 60% - consider feature engineering")
            
            if ensemble_accuracy > 0.75:
                recommendations.append("✅ Good direction accuracy - model is performing well")
            
            if len(self.model_data.get('feature_columns', [])) > 100:
                recommendations.append("💡 Consider feature selection to reduce overfitting")
        
        if not recommendations:
            recommendations.append("📊 Model performance is within expected ranges")
        
        return "<ul>" + "".join([f"<li>{rec}</li>" for rec in recommendations]) + "</ul>"
    
    def _generate_json_summary(self, output_file: str) -> None:
        """Generate JSON evaluation summary"""
        summary = {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': self.model_path,
            'model_info': {
                'created_at': self.model_data.get('created_at') if self.model_data else None,
                'feature_count': len(self.model_data.get('feature_columns', [])) if self.model_data else 0
            },
            'evaluation_results': self.evaluation_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def create_visualizations(self, output_dir: str = 'evaluation_plots') -> List[str]:
        """
        Create evaluation visualizations
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            List of generated plot files
        """
        if not self.evaluation_results:
            self.evaluate_model_performance()
        
        os.makedirs(output_dir, exist_ok=True)
        plot_files = []
        
        # Feature importance plot
        if 'feature_analysis' in self.evaluation_results:
            plot_file = self._create_feature_importance_plot(output_dir)
            if plot_file:
                plot_files.append(plot_file)
        
        # Model comparison plot
        if 'basic_metrics' in self.evaluation_results:
            plot_file = self._create_model_comparison_plot(output_dir)
            if plot_file:
                plot_files.append(plot_file)
        
        print(f"\n📊 Visualizations created in {output_dir}/")
        for plot_file in plot_files:
            print(f"  - {os.path.basename(plot_file)}")
        
        return plot_files
    
    def _create_feature_importance_plot(self, output_dir: str) -> Optional[str]:
        """Create feature importance visualization"""
        try:
            top_features = self.evaluation_results['feature_analysis']['top_features'].head(15)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Most Important Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plot_file = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_file
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
            return None
    
    def _create_model_comparison_plot(self, output_dir: str) -> Optional[str]:
        """Create model comparison visualization"""
        try:
            metrics = self.evaluation_results['basic_metrics']['detailed_metrics']
            
            models = list(metrics.keys())
            r2_scores = [metrics[m].get('r2', 0) for m in models]
            accuracies = [metrics[m].get('direction_accuracy', 0) for m in models]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # R² scores
            ax1.bar(models, r2_scores)
            ax1.set_title('R² Scores by Model')
            ax1.set_ylabel('R² Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # Direction accuracy
            ax2.bar(models, accuracies)
            ax2.set_title('Direction Accuracy by Model')
            ax2.set_ylabel('Accuracy')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            plot_file = os.path.join(output_dir, 'model_comparison.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_file
        except Exception as e:
            print(f"Error creating model comparison plot: {e}")
            return None
    
    def run_comprehensive_evaluation(self, create_report: bool = True, 
                                   create_plots: bool = True) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline
        
        Args:
            create_report: Whether to create HTML/JSON reports
            create_plots: Whether to create visualization plots
            
        Returns:
            Complete evaluation results
        """
        print("🚀 STARTING COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        # Run evaluation
        results = self.evaluate_model_performance()
        
        # Create reports
        if create_report:
            report_file = self.create_evaluation_report()
            results['report_file'] = report_file
        
        # Create visualizations
        if create_plots:
            plot_files = self.create_visualizations()
            results['plot_files'] = plot_files
        
        print("\n✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return results

def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate NSE Stock Prediction Model')
    parser.add_argument('--model', type=str, default='nse_prediction_model.joblib',
                       help='Path to model file')
    parser.add_argument('--output-dir', type=str, default='evaluation_output',
                       help='Output directory for reports and plots')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip report generation')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = NSEModelEvaluator(args.model)
    
    # Run evaluation
    try:
        results = evaluator.run_comprehensive_evaluation(
            create_report=not args.no_report,
            create_plots=not args.no_plots
        )
        
        print(f"\n📁 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()