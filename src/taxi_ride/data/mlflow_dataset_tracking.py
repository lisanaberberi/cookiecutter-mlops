"""
MLflow Dataset Tracking Example - NYC Taxi Dataset

This module demonstrates comprehensive dataset tracking capabilities in MLflow,
using NYC taxi parquet files from your cookiecutter ML course project:
- Dataset creation and versioning from parquet files
- Dataset lineage and source tracking
- Integration with model training and evaluation
- Production monitoring scenarios
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.data
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from typing import Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "nyc-taxi-duration-mlmodels-update"
# mlflow.set_experiment(EXPERIMENT_NAME)


class MLflowDatasetTrackingExample:
    """
    Example class demonstrating MLflow dataset tracking with NYC Taxi dataset.
    
    Expected data structure:
    - data/raw/*.parquet          # Raw taxi trip data
    - data/processed/*.parquet    # Processed/cleaned data
    """
    
    def __init__(
        self, 
        mlflow_tracking_uri: str = "http://localhost:5000",
        data_dir: Optional[str] = None
    ):
        """
        Initialize MLflow tracking example.
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
            data_dir: Path to data directory (default: ./data)
        """
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        mlflow.set_experiment("nyc-taxi-duration-mlmodels-update") 
        self.experiment_name = EXPERIMENT_NAME
        
        if data_dir is None:
            data_dir = Path("data")
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.raw_data_dir = data_dir / "raw"
        self.processed_data_dir = data_dir / "processed"
        
        # Create directories if they don't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Data directories initialized:")
        print(f"  Raw data: {self.raw_data_dir}")
        print(f"  Processed data: {self.processed_data_dir}")
    
    def load_taxi_data(self, parquet_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load NYC taxi data from parquet file.
        
        Args:
            parquet_file: Optional specific parquet file name
            
        Returns:
            Loaded DataFrame
        """
        if parquet_file is None:
            # Find first parquet file in raw data directory
            parquet_files = list(self.raw_data_dir.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(
                    f"No parquet files found in {self.raw_data_dir}. "
                    "Please add NYC taxi parquet files to data/raw/"
                )
            parquet_file = parquet_files[0]
        else:
            parquet_file = self.raw_data_dir / parquet_file
        
        print(f"✓ Loading data from: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
        
        return df
    
    def inspect_taxi_data(self, df: pd.DataFrame) -> None:
        """
        Inspect and display NYC taxi dataset information.
        
        Args:
            df: Input DataFrame
        """
        print(f"\nDataset Information:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Dtypes:\n{df.dtypes}")
        print(f"  Missing values:\n{df.isnull().sum()}")
        print(f"  Sample rows:\n{df.head(3)}")
    
    def example_1_basic_dataset_tracking(self):
        """
        Example 1: Basic Dataset Tracking with NYC Taxi Data
        
        Demonstrates:
        - Loading parquet data from data/raw/
        - Creating a dataset from NYC taxi data
        - Logging dataset to MLflow run
        - Accessing dataset metadata
        """
        print("\n" + "="*70)
        print("EXAMPLE 1: Basic Dataset Tracking - NYC Taxi Data")
        print("="*70)
        
        # Load NYC taxi data
        try:
            raw_data = self.load_taxi_data()
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print("Creating sample taxi data for demonstration...")
            raw_data = pd.DataFrame({
                'trip_distance': np.random.uniform(0.1, 50, 5000),
                'trip_duration': np.random.uniform(60, 3600, 5000),
                'fare_amount': np.random.uniform(2.5, 100, 5000),
                'passenger_count': np.random.randint(1, 7, 5000),
                'pickup_hour': np.random.randint(0, 24, 5000),
            })
        
        # Inspect data
        self.inspect_taxi_data(raw_data)
        
        # Create MLflow Dataset
        dataset = mlflow.data.from_pandas(
            raw_data,
            source="local_data",
            name="nyc-taxi-raw",
            targets="fare_amount"
        )
        
        print(f"\n✓ Created MLflow Dataset")
        
        # Log dataset to MLflow
        with mlflow.start_run(run_name="nyc_taxi_basic_tracking"):
            mlflow.log_input(dataset, context="training")
            
            # Log dataset metadata
            mlflow.log_params({
                "dataset_name": dataset.name,
                "dataset_digest": dataset.digest,
                "data_source": "NYC Taxi Parquet",
                "rows": len(raw_data),
                "columns": len(raw_data.columns),
            })
            
            # Print dataset information
            print(f"\nDataset Information:")
            print(f"  Name: {dataset.name}")
            print(f"  Digest: {dataset.digest}")
            print(f"  Rows: {len(raw_data)}")
            print(f"  Columns: {len(raw_data.columns)}")
            print(f"  Source: {dataset.source}")
            print(f"  Target: fare_amount")
            
            # Access profile and schema if available
            if dataset.profile:
                print(f"  Profile: {dataset.profile}")
            # if dataset.schema:
            #     print(f"  Schema columns: {len(dataset.schema.mlflow_colspec)}")
            
            mlflow.end_run()
        
        return dataset, raw_data
    
    def example_2_dataset_with_splits(self):
        """
        Example 2: Dataset Tracking with Train/Test Splits - NYC Taxi
        
        Demonstrates:
        - Creating datasets for training and evaluation from taxi data
        - Tracking data splits separately
        - Logging split information with taxi-specific metrics
        """
        print("\n" + "="*70)
        print("EXAMPLE 2: Dataset Tracking with Splits - NYC Taxi")
        print("="*70)
        
        # Load data
        try:
            data = self.load_taxi_data()
        except FileNotFoundError:
            print("⚠️  Using sample taxi data")
            data = pd.DataFrame({
                'trip_distance': np.random.uniform(0.1, 50, 1000),
                'trip_duration': np.random.uniform(60, 3600, 1000),
                'fare_amount': np.random.uniform(2.5, 100, 1000),
                'passenger_count': np.random.randint(1, 7, 1000),
                'pickup_hour': np.random.randint(0, 24, 1000),
                'day_of_week': np.random.randint(0, 7, 1000),
            })
        
        print(f"✓ Loaded taxi dataset: {data.shape}")
        
        # Split data
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != 'fare_amount']
        
        X = data[feature_cols] if feature_cols else data.drop('fare_amount', axis=1)
        y = data['fare_amount']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Create training dataset
        train_data = X_train.reset_index(drop=True).copy()
        train_data['fare_amount'] = y_train.reset_index(drop=True).values
        train_dataset = mlflow.data.from_pandas(
            train_data,
            source="nyc_taxi_training_split",
            name="nyc-taxi-train",
            targets="fare_amount"
        )
        
        # Create test dataset
        test_data = X_test.reset_index(drop=True).copy()
        test_data['fare_amount'] = y_test.reset_index(drop=True).values
        test_dataset = mlflow.data.from_pandas(
            test_data,
            source="nyc_taxi_test_split",
            name="nyc-taxi-test",
            targets="fare_amount"
        )
        
        print(f"✓ Created train dataset: {train_data.shape}")
        print(f"✓ Created test dataset: {test_data.shape}")
        
        with mlflow.start_run(run_name="nyc_taxi_splits"):
            # Log training dataset
            mlflow.log_input(train_dataset, context="training")
            
            # Log evaluation dataset
            mlflow.log_input(test_dataset, context="evaluation")
            
            # Log split information
            mlflow.log_params({
                "train_size": len(train_data),
                "test_size": len(test_data),
                "total_size": len(data),
                "test_ratio": 0.3,
                "random_state": 42,
                "target_variable": "fare_amount",
            })
            
            # Log taxi-specific metrics
            mlflow.log_metrics({
                "avg_fare_train": float(train_data['fare_amount'].mean()),
                "avg_fare_test": float(test_data['fare_amount'].mean()),
                "avg_distance_train": float(train_data[feature_cols[0]].mean()) if feature_cols else 0,
            })
            
            print(f"\nSplit Information:")
            print(f"  Total samples: {len(data)}")
            print(f"  Training samples: {len(train_data)}")
            print(f"  Test samples: {len(test_data)}")
            print(f"  Test ratio: 30%")
            print(f"  Average fare (train): ${train_data['fare_amount'].mean():.2f}")
            print(f"  Average fare (test): ${test_data['fare_amount'].mean():.2f}")
            
            mlflow.end_run()
        
        return train_dataset, test_dataset, train_data, test_data
    
    def example_3_dataset_versioning(self):
        """
        Example 3: Dataset Versioning - NYC Taxi
        
        Demonstrates:
        - Creating multiple versions of NYC taxi datasets
        - Tracking version history with data preprocessing steps
        - Comparing dataset versions
        """
        print("\n" + "="*70)
        print("EXAMPLE 3: Dataset Versioning - NYC Taxi Data Processing")
        print("="*70)
        
        # Load data
        try:
            base_data = self.load_taxi_data()
        except FileNotFoundError:
            print("⚠️  Using sample taxi data")
            base_data = pd.DataFrame({
                'trip_distance': np.random.uniform(0.1, 50, 1000),
                'trip_duration': np.random.uniform(60, 3600, 1000),
                'fare_amount': np.random.uniform(2.5, 100, 1000),
                'passenger_count': np.random.randint(1, 7, 1000),
                'pickup_hour': np.random.randint(0, 24, 1000),
            })
        
        print(f"✓ Loaded NYC taxi dataset: {base_data.shape}")
        
        # Create multiple versions with different preprocessing
        versions = {}
        
        with mlflow.start_run(run_name="nyc_taxi_versioning"):
            for version in ["1.0", "2.0", "3.0"]:
                # Simulate different preprocessing steps
                if version == "1.0":
                    # Raw data
                    data = base_data.copy()
                    description = "Raw NYC taxi data (no preprocessing)"
                    
                elif version == "2.0":
                    # Remove duplicates and extreme outliers
                    data = base_data.drop_duplicates().reset_index(drop=True)
                    # Remove fares > $200
                    if 'fare_amount' in data.columns:
                        data = data[data['fare_amount'] <= 200].reset_index(drop=True)
                    description = "Deduplicated and outliers removed (fare <= $200)"
                    
                elif version == "3.0":
                    # Previous step + remove missing values + distance filtering
                    data = base_data.drop_duplicates().reset_index(drop=True)
                    if 'fare_amount' in data.columns:
                        data = data[data['fare_amount'] <= 200]
                    data = data.dropna().reset_index(drop=True)
                    # Keep realistic trip distances (0.1 to 50 miles)
                    if 'trip_distance' in data.columns:
                        data = data[
                            (data['trip_distance'] >= 0.1) & 
                            (data['trip_distance'] <= 50)
                        ].reset_index(drop=True)
                    description = "Production-ready: cleaned, deduped, filtered for distance & fare"
                
                # Create versioned dataset
                dataset = mlflow.data.from_pandas(
                    data,
                    source=f"nyc_taxi_pipeline_v{version}",
                    name=f"nyc-taxi-v{version}",
                    targets="fare_amount"
                )
                
                versions[version] = dataset
                
                # Log nested run for each version
                with mlflow.start_run(run_name=f"nyc_taxi_version_{version}", nested=True):
                    mlflow.log_input(dataset, context="versioning")
                    
                    mlflow.log_params({
                        "version": version,
                        "description": description,
                        "row_count": len(data),
                        "columns": len(data.columns),
                        "missing_values": data.isnull().sum().sum(),
                        "duplicate_rows": base_data.duplicated().sum() - data.duplicated().sum(),
                    })
                    
                    # Log taxi-specific quality metrics
                    mlflow.log_metrics({
                        "data_completeness": (1 - data.isnull().sum().sum() / data.size) * 100 if data.size > 0 else 100,
                        "avg_fare": float(data['fare_amount'].mean()) if 'fare_amount' in data.columns else 0,
                        "fare_std": float(data['fare_amount'].std()) if 'fare_amount' in data.columns else 0,
                    })
                    
                    print(f"\nVersion {version}: {description}")
                    print(f"  Rows: {len(data)}")
                    print(f"  Missing values: {data.isnull().sum().sum()}")
                    print(f"  Data completeness: {(1 - data.isnull().sum().sum() / data.size) * 100:.2f}%")
                    if 'fare_amount' in data.columns:
                        print(f"  Avg fare: ${data['fare_amount'].mean():.2f}")
            
            mlflow.end_run()
        
        return versions
    
    def example_4_training_with_dataset_tracking(self):
        """
        Example 4: Model Training with Dataset Tracking - NYC Taxi
        
        Demonstrates:
        - Training fare prediction models with tracked datasets
        - Logging dataset and model together
        - Creating complete lineage for taxi fare prediction
        """
        print("\n" + "="*70)
        print("EXAMPLE 4: Model Training with Dataset Tracking - Fare Prediction")
        print("="*70)
        
        # Load and prepare data
        try:
            data = self.load_taxi_data()
        except FileNotFoundError:
            print("⚠️  Using sample taxi data")
            data = pd.DataFrame({
                'trip_distance': np.random.uniform(0.1, 50, 500),
                'trip_duration': np.random.uniform(60, 3600, 500),
                'fare_amount': np.random.uniform(2.5, 100, 500),
                'passenger_count': np.random.randint(1, 7, 500),
                'pickup_hour': np.random.randint(0, 24, 500),
            })
        
        # Prepare features
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != 'fare_amount']
        
        X = data[feature_cols] if feature_cols else data.drop('fare_amount', axis=1)
        y = data['fare_amount']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create dataset
        train_data = X_train.copy()
        train_data['fare_amount'] = y_train.values
        
        dataset = mlflow.data.from_pandas(
            train_data,
            source="nyc_taxi_training",
            name="nyc-taxi-fare-training",
            targets="fare_amount"
        )
        
        print(f"✓ Prepared training dataset: {train_data.shape}")
        
        with mlflow.start_run(run_name="nyc_taxi_fare_prediction"):
            # Log dataset
            mlflow.log_input(dataset, context="training")
            
            # Train model - Fare Amount Prediction
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Log parameters
            mlflow.log_params({
                "model_type": "RandomForestRegressor",
                "n_estimators": 100,
                "max_depth": 15,
                "dataset_name": dataset.name,
                "dataset_digest": dataset.digest,
                "target": "fare_amount",
                "features": ",".join(feature_cols[:3]) + "...",
            })
            
            # Log metrics
            mlflow.log_metrics({
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "test_samples": len(X_test),
                "train_samples": len(X_train),
            })
            
            # Log model
            mlflow.sklearn.log_model(model, "fare_prediction_model", input_example=X_test.head())
            
            print(f"\nModel Training Results:")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAE: ${mae:.2f}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  Test samples: {len(X_test)}")
            print(f"  Dataset digest: {dataset.digest}")
            
            mlflow.end_run()
        
        return model, X_test, y_test
    
    def example_5_evaluation_with_dataset(self):
        """
        Example 5: Model Evaluation with MLflow Evaluate - NYC Taxi
        
        Demonstrates:
        - Using datasets with model evaluation
        - Tracking evaluation datasets separately
        - Integration with evaluation workflows
        """
        print("\n" + "="*70)
        print("EXAMPLE 5: Model Evaluation with Datasets - NYC Taxi")
        print("="*70)
        
        # Prepare data
        try:
            data = self.load_taxi_data()
        except FileNotFoundError:
            print("⚠️  Using sample taxi data")
            data = pd.DataFrame({
                'trip_distance': np.random.uniform(0.1, 50, 500),
                'trip_duration': np.random.uniform(60, 3600, 500),
                'fare_amount': np.random.uniform(2.5, 100, 500),
                'passenger_count': np.random.randint(1, 7, 500),
                'pickup_hour': np.random.randint(0, 24, 500),
            })
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != 'fare_amount']
        
        X = data[feature_cols] if feature_cols else data.drop('fare_amount', axis=1)
        y = data['fare_amount']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        print(f"✓ Trained model on {len(X_train)} samples")
        
        with mlflow.start_run(run_name="nyc_taxi_model_evaluation"):
            # Log training dataset
            train_data = X_train.copy()
            train_data['fare_amount'] = y_train.values
            
            train_dataset = mlflow.data.from_pandas(
                train_data,
                source="nyc_taxi_training",
                name="nyc-taxi-train",
                targets="fare_amount"
            )
            mlflow.log_input(train_dataset, context="training")
            
            # Create and log evaluation dataset
            eval_data = X_test.copy()
            eval_data['fare_amount'] = y_test.values
            
            eval_dataset = mlflow.data.from_pandas(
                eval_data,
                source="nyc_taxi_evaluation",
                name="nyc-taxi-eval",
                targets="fare_amount"
            )
            mlflow.log_input(eval_dataset, context="evaluation")
            
            # Log the model first
            mlflow.sklearn.log_model(model, "model", input_example=X_test.head())
            
            # Get model URI
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            
            # Make predictions and evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            print(f"\nEvaluation Results:")
            print(f"  Dataset: {eval_dataset.name}")
            print(f"  Sample count: {len(eval_data)}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAE: ${mae:.2f}")
            print(f"  R² Score: {r2:.4f}")
            
            # Log evaluation metrics
            mlflow.log_metrics({
                "eval_rmse": rmse,
                "eval_mae": mae,
                "eval_r2": r2,
                "eval_test_samples": len(X_test),
            })
            
            # Log per-sample predictions for analysis
            eval_results = pd.DataFrame({
                'actual_fare': y_test.values,
                'predicted_fare': y_pred,
                'error': y_test.values - y_pred,
                'error_pct': ((y_test.values - y_pred) / y_test.values * 100).round(2)
            })
            
            mlflow.log_dict(eval_results.describe().to_dict(), "evaluation_summary.json")
            
            mlflow.end_run()
    
    def example_6_production_monitoring(self):
        """
        Example 6: Production Batch Monitoring - NYC Taxi
        
        Demonstrates:
        - Tracking production batch datasets (daily predictions)
        - Monitoring prediction distributions
        - Tracking model performance over time
        """
        print("\n" + "="*70)
        print("EXAMPLE 6: Production Batch Monitoring - Taxi Fare Prediction")
        print("="*70)
        
        # Simulate production batch data (daily predictions)
        np.random.seed(42)
        n_samples = 200
        
        X_batch = pd.DataFrame({
            'trip_distance': np.random.uniform(0.1, 50, n_samples),
            'trip_duration': np.random.uniform(60, 3600, n_samples),
            'passenger_count': np.random.randint(1, 7, n_samples),
            'pickup_hour': np.random.randint(0, 24, n_samples),
        })
        
        # Simulate predictions and ground truth (with some noise)
        y_true = np.random.uniform(5, 80, n_samples)
        y_pred = y_true + np.random.normal(0, 5, n_samples)  # Predictions with noise
        
        batch_data = X_batch.copy()
        batch_data['predicted_fare'] = y_pred
        batch_data['actual_fare'] = y_true
        batch_data['prediction_error'] = y_pred - y_true
        
        print(f"✓ Created production batch data: {batch_data.shape}")
        
        # Create dataset for monitoring
        batch_dataset = mlflow.data.from_pandas(
            batch_data,
            source=f"production_batch_{datetime.now().date()}",
            name=f"batch_predictions_{datetime.now().date()}",
            targets="actual_fare"
        )
        
        with mlflow.start_run(run_name=f"batch_monitor_{datetime.now().date()}"):
            mlflow.log_input(batch_dataset, context="production_batch")
            
            # Log batch metadata
            mlflow.log_params({
                "batch_date": str(datetime.now().date()),
                "batch_size": len(batch_data),
                "has_ground_truth": True,
                "has_predictions": True,
                "use_case": "NYC Taxi Fare Prediction",
            })
            
            # Monitor prediction distribution
            pred_mae = mean_absolute_error(y_true, y_pred)
            pred_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            pred_r2 = r2_score(y_true, y_pred)
            
            mlflow.log_metrics({
                "batch_mae": pred_mae,
                "batch_rmse": pred_rmse,
                "batch_r2": pred_r2,
                "prediction_mean": float(batch_data['predicted_fare'].mean()),
                "prediction_std": float(batch_data['predicted_fare'].std()),
                "prediction_min": float(batch_data['predicted_fare'].min()),
                "prediction_max": float(batch_data['predicted_fare'].max()),
                "actual_mean": float(batch_data['actual_fare'].mean()),
                "batch_samples": len(batch_data),
            })
            
            # Log prediction and error distribution
            pred_dist = {
                'predicted_fares': batch_data['predicted_fare'].describe().to_dict(),
                'actual_fares': batch_data['actual_fare'].describe().to_dict(),
                'errors': batch_data['prediction_error'].describe().to_dict(),
            }
            mlflow.log_dict(pred_dist, "prediction_distribution.json")
            
            print(f"\nProduction Batch Monitoring:")
            print(f"  Batch date: {datetime.now().date()}")
            print(f"  Batch size: {len(batch_data)}")
            print(f"  MAE: ${pred_mae:.2f}")
            print(f"  RMSE: ${pred_rmse:.2f}")
            print(f"  R² Score: {pred_r2:.4f}")
            print(f"  Avg predicted fare: ${batch_data['predicted_fare'].mean():.2f}")
            print(f"  Avg actual fare: ${batch_data['actual_fare'].mean():.2f}")
            print(f"  Prediction std: ${batch_data['prediction_error'].std():.2f}")
            
            mlflow.end_run()


def run_all_examples():
    """Run all MLflow dataset tracking examples with NYC taxi data."""
    
    print("\n" + "="*70)
    print("MLflow Dataset Tracking - NYC Taxi Examples")
    print("="*70)
    print("\nNote: This example uses NYC taxi parquet files from:")
    print("  data/raw/*.parquet")
    print("\nIf parquet files are not available, the examples will use")
    print("simulated taxi data for demonstration.")
    print("\nMake sure MLflow UI is running with:")
    print("  mlflow ui --host 127.0.0.1 --port 5000")
    
    # Initialize example runner
    example = MLflowDatasetTrackingExample(data_dir="data")
    
    # Run examples
    try:
        dataset, raw_data = example.example_1_basic_dataset_tracking()
        example.example_2_dataset_with_splits()
        example.example_3_dataset_versioning()
        example.example_4_training_with_dataset_tracking()
        example.example_5_evaluation_with_dataset()
        example.example_6_production_monitoring()
    except Exception as e:
        print(f"\n⚠️  Error during execution: {e}")
        print("\nMake sure you have:")
        print("  1. MLflow running: mlflow ui")
        print("  2. NYC taxi data in: data/raw/*.parquet")
        print("  3. Required packages: pip install mlflow pandas numpy scikit-learn")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nView results in MLflow UI at: http://localhost:5000")
    print("\nExamples demonstrated:")
    print("  ✓ Example 1: Basic dataset tracking from NYC taxi parquet files")
    print("  ✓ Example 2: Train/test split tracking with data lineage")
    print("  ✓ Example 3: Dataset versioning through preprocessing stages")
    print("  ✓ Example 4: Model training with dataset tracking (fare prediction)")
    print("  ✓ Example 5: Model evaluation with datasets")
    print("  ✓ Example 6: Production batch monitoring for daily predictions")
    print("\nNext steps:")
    print("  1. Check MLflow UI for all tracked experiments")
    print("  2. Compare different dataset versions")
    print("  3. Trace data lineage from raw parquet to predictions")
    print("  4. Monitor production batch metrics over time")


if __name__ == "__main__":
    run_all_examples()