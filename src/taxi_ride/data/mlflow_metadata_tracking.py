"""
MLflow Metadata Tracking for GREEN Taxi Dataset
Uses MetaDataset and Schema to track NYC TLC LPEP (GREEN Taxi) data structure
Based on NYC TLC LPEP Data Dictionary - March 18, 2025

Source: http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml
"""

import mlflow
import pandas as pd
import numpy as np
from mlflow.data.meta_dataset import MetaDataset
from mlflow.types import Schema, ColSpec, DataType



 
# GREEN Taxi Column Descriptions (from NYC TLC Data Dictionary - March 18, 2025)
GREEN_TAXI_DESCRIPTIONS = {
    "VendorID": "LPEP provider (1: Creative Mobile Technologies, 2: Curb Mobility, 6: Myle Technologies)",
    "lpep_pickup_datetime": "Date and time when the meter was engaged",
    "lpep_dropoff_datetime": "Date and time when the meter was disengaged",
    "store_and_fwd_flag": "Store and forward flag (Y/N) - trip held in vehicle memory before sending to vendor",
    "RatecodeID": "Final rate code (1: Standard, 2: JFK, 3: Newark, 4: Nassau/Westchester, 5: Negotiated, 6: Group, 99: Null/Unknown)",
    "PULocationID": "TLC Taxi Zone in which taximeter was engaged (1-263)",
    "DOLocationID": "TLC Taxi Zone in which taximeter was disengaged (1-263)",
    "passenger_count": "Number of passengers in the vehicle",
    "trip_distance": "Elapsed trip distance in miles reported by the taximeter",
    "fare_amount": "Time-and-distance fare calculated by the meter (in dollars)",
    "extra": "Miscellaneous extras and surcharges (in dollars)",
    "mta_tax": "Tax automatically triggered based on the metered rate (in dollars)",
    "tip_amount": "Tip amount - auto-populated for credit card tips, excludes cash tips (in dollars)",
    "tolls_amount": "Total amount of all tolls paid in trip (in dollars)",
    "improvement_surcharge": "Improvement surcharge assessed at flag drop, started being levied in 2015 (in dollars)",
    "total_amount": "Total amount charged to passengers, excludes cash tips (in dollars)",
    "payment_type": "Payment method (0: Flex Fare, 1: Credit, 2: Cash, 3: No charge, 4: Dispute, 5: Unknown, 6: Voided)",
    "trip_type": "Trip type (1: Street-hail, 2: Dispatch) - auto-assigned based on metered rate but can be altered by driver",
    "congestion_surcharge": "Total NYS congestion surcharge collected in trip (in dollars)",
    "cbd_congestion_fee": "Per-trip charge for MTA Congestion Relief Zone starting Jan 5, 2025 (in dollars)",
}
 
 
class MLflowGreenTaxiMetadataTracking:
    """Track GREEN Taxi (LPEP) dataset metadata using MLflow MetaDataset"""
    
    def __init__(self, mlflow_tracking_uri="http://localhost:5000", data_dir="data/raw"):
        """
        Initialize MLflow tracking for GREEN Taxi metadata.
        
        Args:
            mlflow_tracking_uri: MLflow server URI
            data_dir: Directory containing green taxi parquet files
        """
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Green_Taxi_LPEP_Metadata_Tracking")
        self.data_dir = data_dir
    
    def create_green_taxi_schema(self):
        """
        Create Schema for GREEN Taxi (LPEP) based on NYC TLC Data Dictionary
        March 18, 2025
        
        Returns:
            Schema: MLflow Schema with 20 columns
        """
        green_taxi_schema = Schema([
            ColSpec(type=DataType.integer, name="VendorID"),
            ColSpec(type=DataType.datetime, name="lpep_pickup_datetime"),
            ColSpec(type=DataType.datetime, name="lpep_dropoff_datetime"),
            ColSpec(type=DataType.string, name="store_and_fwd_flag"),
            ColSpec(type=DataType.double, name="RatecodeID"),
            ColSpec(type=DataType.integer, name="PULocationID"),
            ColSpec(type=DataType.integer, name="DOLocationID"),
            ColSpec(type=DataType.double, name="passenger_count"),
            ColSpec(type=DataType.double, name="trip_distance"),
            ColSpec(type=DataType.double, name="fare_amount"),
            ColSpec(type=DataType.double, name="extra"),
            ColSpec(type=DataType.double, name="mta_tax"),
            ColSpec(type=DataType.double, name="tip_amount"),
            ColSpec(type=DataType.double, name="tolls_amount"),
            ColSpec(type=DataType.double, name="improvement_surcharge"),
            ColSpec(type=DataType.double, name="total_amount"),
            ColSpec(type=DataType.double, name="payment_type"),
            ColSpec(type=DataType.double, name="trip_type"),
            ColSpec(type=DataType.double, name="congestion_surcharge"),
            ColSpec(type=DataType.double, name="cbd_congestion_fee"),
        ])
        
        return green_taxi_schema
    
    def create_green_taxi_metadataset(self, source_path="data/raw/green_tripdata_2023-10.parquet"):
        """
        Create a MetaDataset with GREEN Taxi (LPEP) schema metadata.
        
        This creates a metadata-only dataset that references the source data
        without storing the actual data in MLflow artifacts.
        
        Args:
            source_path: Path to the green taxi parquet file
        
        Returns:
            MetaDataset: Dataset with schema metadata from NYC TLC LPEP data dictionary
        """
        
        # Create schema
        schema = self.create_green_taxi_schema()
        
        # Create MetaDataset with schema
        meta_dataset = MetaDataset(
            source=source_path,
            name="nyc-green-taxi-lpep",
            schema=schema
        )
        
        return meta_dataset
    
    def log_metadataset_only(self, run_name="Green_Taxi_MetaDataset_Only"):
        """
        Log GREEN Taxi metadata-only dataset to MLflow run.
        
        No actual data is stored, just the schema and metadata reference.
        """
        
        meta_dataset = self.create_green_taxi_metadataset()
        
        with mlflow.start_run(run_name=run_name):
            # Log metadata-only dataset
            mlflow.log_input(meta_dataset, context="external_data")
            
            # Log tags with TLC information
            mlflow.set_tag("dataset_type", "NYC GREEN Taxi (LPEP)")
            mlflow.set_tag("data_dictionary_version", "March 18, 2025")
            mlflow.set_tag("source_url", "http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml")
            mlflow.set_tag("data_period", "October 2023")
            mlflow.set_tag("vendor_count", "3")
            mlflow.set_tag("taxi_zones", "1-263")
            mlflow.set_tag("trip_types", "2")
            mlflow.set_tag("metadata_only", "true")
            
            # Log metadata as parameters
            mlflow.log_params({
                "total_columns": 20,
                "vendors": "1: Creative Mobile Technologies, 2: Curb Mobility, 6: Myle Technologies",
                "key_columns": "VendorID, lpep_pickup_datetime, lpep_dropoff_datetime, passenger_count, trip_distance, fare_amount",
                "target_variable": "fare_amount",
                "monetary_columns": "fare_amount, extra, mta_tax, tip_amount, tolls_amount, improvement_surcharge, total_amount, congestion_surcharge, cbd_congestion_fee",
                "rate_codes": "1: Standard, 2: JFK, 3: Newark, 4: Nassau/Westchester, 5: Negotiated, 6: Group, 99: Null/Unknown",
                "payment_types": "0: Flex Fare, 1: Credit, 2: Cash, 3: No charge, 4: Dispute, 5: Unknown, 6: Voided",
                "trip_types": "1: Street-hail, 2: Dispatch",
            })
            
            print(f"\n✓ Logged GREEN Taxi (LPEP) MetaDataset")
            print(f"  Dataset: {meta_dataset.name}")
            print(f"  Schema columns: {len(list(meta_dataset.schema))}")
            print(f"  Source: {meta_dataset.source.uri}")
            print(f"  Data Dictionary: March 18, 2025")
            print(f"  Size: Metadata only (no data stored)")
            
            return meta_dataset
    
    def log_metadataset_with_data(self, data_df, run_name="Green_Taxi_With_Data_And_Metadata"):
        """
        Log GREEN Taxi dataset with both data and metadata.
        
        Args:
            data_df: Pandas DataFrame with green taxi data
            run_name: Name for the MLflow run
        """
        
        meta_dataset = self.create_green_taxi_metadataset()
        
        with mlflow.start_run(run_name=run_name):
            # Log both metadata and actual data
            mlflow.log_input(meta_dataset, context="training")
            
            # Log tags
            mlflow.set_tag("dataset_type", "NYC GREEN Taxi (LPEP)")
            mlflow.set_tag("data_dictionary_version", "March 18, 2025")
            mlflow.set_tag("metadata_only", "false")
            
            # Log parameters
            mlflow.log_params({
                "total_columns": len(data_df.columns),
                "total_rows": len(data_df),
                "target_variable": "fare_amount",
            })
            
            # Log metrics
            mlflow.log_metrics({
                "num_rows": len(data_df),
                "num_columns": len(data_df.columns),
                "missing_percentage": float((data_df.isnull().sum().sum() / data_df.size * 100)) if data_df.size > 0 else 0,
                "completeness": float((1 - data_df.isnull().sum().sum() / data_df.size) * 100) if data_df.size > 0 else 100,
            })
            
            print(f"\n✓ Logged GREEN Taxi Dataset with Data and Metadata")
            print(f"  Rows: {len(data_df):,}")
            print(f"  Columns: {len(data_df.columns)}")
            print(f"  Schema columns: {len(list(meta_dataset.schema))}")
            
            return meta_dataset
    
    def display_schema(self):
        schema = self.create_green_taxi_schema()
        
        print("GREEN Taxi (LPEP) Schema")
        
        # ✅ Iterate schema directly
        for i, col in enumerate(schema, 1):
            description = GREEN_TAXI_DESCRIPTIONS.get(col.name, "")
            print(f"\n{i}. {col.name}")
            print(f"   Type: {col.type}")
            print(f"   Description: {description}")
 

# Usage Examples
# ==============

if __name__ == "__main__":
    
    # Initialize tracker
    tracker = MLflowGreenTaxiMetadataTracking()
    
    # Example 1: Display schema
    print("\n" + "="*80)
    print("EXAMPLE 1: Display GREEN Taxi Schema")
    print("="*80)
    tracker.display_schema()
    
    # Example 2: Log metadata-only dataset
    print("\n" + "="*80)
    print("EXAMPLE 2: Log Metadata-Only Dataset (No Data Stored)")
    print("="*80)
    tracker.log_metadataset_only(run_name="Example_2_Metadata_Only")
    
    # Example 3: Log metadata with actual data
    print("\n" + "="*80)
    print("EXAMPLE 3: Log Metadata with Actual Data")
    print("="*80)
    
    # Create sample green taxi data
    sample_data = pd.DataFrame({
        'VendorID': np.random.choice([1, 2, 6], 1000),
        'lpep_pickup_datetime': pd.date_range('2023-10-01', periods=1000, freq='H'),
        'lpep_dropoff_datetime': pd.date_range('2023-10-01', periods=1000, freq='H') + pd.Timedelta(minutes=30),
        'store_and_fwd_flag': np.random.choice(['Y', 'N'], 1000),
        'RatecodeID': np.random.choice([1, 2, 3, 4, 5, 6, 99], 1000),
        'PULocationID': np.random.randint(1, 264, 1000),
        'DOLocationID': np.random.randint(1, 264, 1000),
        'passenger_count': np.random.randint(1, 7, 1000),
        'trip_distance': np.random.uniform(0.1, 50, 1000),
        'fare_amount': np.random.uniform(2.5, 100, 1000),
        'extra': np.random.uniform(0, 5, 1000),
        'mta_tax': 0.5,
        'tip_amount': np.random.uniform(0, 20, 1000),
        'tolls_amount': np.random.uniform(0, 15, 1000),
        'improvement_surcharge': np.random.uniform(0.3, 1, 1000),
        'total_amount': np.random.uniform(2.5, 150, 1000),
        'payment_type': np.random.choice([0, 1, 2, 3, 4, 5, 6], 1000),
        'trip_type': np.random.choice([1, 2], 1000),
        'congestion_surcharge': np.random.uniform(0, 5, 1000),
        'cbd_congestion_fee': np.random.uniform(0, 5, 1000),
    })
    
    tracker.log_metadataset_with_data(sample_data, run_name="Example_3_Metadata_With_Data")
    
    print("\n" + "="*80)
    print("✓ All examples completed!")
    print("="*80)
    print(f"\nView results in MLflow UI:")
    print(f"  http://localhost:5000")
    print(f"\nExperiment: Green_Taxi_LPEP_Metadata_Tracking")
    print(f"Runs:")
    print(f"  - Example_2_Metadata_Only")
    print(f"  - Example_3_Metadata_With_Data")