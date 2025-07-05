"""
Data Cleaning and Quality Assurance Module
Demonstrates data quality, tracking, and reporting infrastructure skills
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityEngine:
    """
    Comprehensive data quality engine for e-commerce analytics
    Demonstrates data quality, tracking, and reporting infrastructure
    """
    
    def __init__(self):
        self.quality_metrics = {}
        self.data_issues = []
        self.cleaning_log = []
    
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data with comprehensive validation
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Basic validation
            self._validate_basic_structure(df, file_path)
            
            # Data type validation
            df = self._validate_and_convert_dtypes(df)
            
            # Missing value analysis
            self._analyze_missing_values(df, file_path)
            
            # Duplicate detection
            self._detect_duplicates(df, file_path)
            
            # Outlier detection
            self._detect_outliers(df, file_path)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def _validate_basic_structure(self, df: pd.DataFrame, file_path: str):
        """Validate basic data structure"""
        issues = []
        
        # Check for empty dataframe
        if df.empty:
            issues.append("Empty dataframe")
        
        # Check for required columns based on file type
        if "Sale Report" in file_path:
            required_cols = ['SKU Code', 'Category', 'Stock']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")
        
        if issues:
            self.data_issues.extend([(file_path, issue) for issue in issues])
            logger.warning(f"Data structure issues in {file_path}: {issues}")
    
    def _validate_and_convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types"""
        df_clean = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['Stock', 'Weight', 'TP', 'MRP Old', 'Final MRP Old']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Convert date columns
        date_cols = ['Recived Amount']  # Note: typo in original data
        for col in date_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        return df_clean
    
    def _analyze_missing_values(self, df: pd.DataFrame, file_path: str):
        """Analyze missing values and create quality metrics"""
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        self.quality_metrics[f"{file_path}_missing_values"] = {
            'total_rows': len(df),
            'missing_counts': missing_data.to_dict(),
            'missing_percentages': missing_percentage.to_dict()
        }
        
        # Flag high missing value columns
        high_missing = missing_percentage[missing_percentage > 20]
        if not high_missing.empty:
            self.data_issues.append((file_path, f"High missing values: {high_missing.to_dict()}"))
    
    def _detect_duplicates(self, df: pd.DataFrame, file_path: str):
        """Detect and report duplicate records"""
        duplicates = df.duplicated().sum()
        duplicate_percentage = (duplicates / len(df)) * 100
        
        self.quality_metrics[f"{file_path}_duplicates"] = {
            'duplicate_count': duplicates,
            'duplicate_percentage': duplicate_percentage
        }
        
        if duplicate_percentage > 5:
            self.data_issues.append((file_path, f"High duplicate rate: {duplicate_percentage:.2f}%"))
    
    def _detect_outliers(self, df: pd.DataFrame, file_path: str):
        """Detect outliers in numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_percentage = (len(outliers) / len(df)) * 100
                
                if outlier_percentage > 10:
                    self.data_issues.append((file_path, f"High outliers in {col}: {outlier_percentage:.2f}%"))

class EcommerceDataCleaner:
    """
    Specialized data cleaner for e-commerce datasets
    """
    
    def __init__(self):
        self.quality_engine = DataQualityEngine()
    
    def clean_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean sales report data"""
        df_clean = df.copy()
        
        # Remove rows with invalid SKU codes
        df_clean = df_clean[df_clean['SKU Code'].notna()]
        df_clean = df_clean[df_clean['SKU Code'] != '#REF!']
        
        # Clean stock values
        df_clean['Stock'] = pd.to_numeric(df_clean['Stock'], errors='coerce')
        df_clean = df_clean[df_clean['Stock'] >= 0]  # Remove negative stock
        
        # Standardize category names
        df_clean['Category'] = df_clean['Category'].str.strip()
        
        # Clean size values
        df_clean['Size'] = df_clean['Size'].str.upper()
        
        return df_clean
    
    def clean_pricing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean pricing data from May-2022"""
        df_clean = df.copy()
        
        # Remove rows with invalid pricing
        price_cols = ['TP', 'MRP Old', 'Final MRP Old', 'Amazon MRP', 'Flipkart MRP']
        for col in price_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean = df_clean[df_clean[col] > 0]  # Remove zero/negative prices
        
        # Calculate price margins
        if 'TP' in df_clean.columns and 'Final MRP Old' in df_clean.columns:
            df_clean['Margin'] = df_clean['Final MRP Old'] - df_clean['TP']
            df_clean['Margin_Percentage'] = (df_clean['Margin'] / df_clean['Final MRP Old']) * 100
        
        return df_clean
    
    def clean_expense_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean expense data"""
        df_clean = df.copy()
        
        # Remove header rows
        df_clean = df_clean[df_clean['Recived Amount'] != 'Particular']
        
        # Clean amount columns
        df_clean['Recived Amount'] = pd.to_numeric(df_clean['Recived Amount'], errors='coerce')
        df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce')
        
        # Remove rows with invalid amounts
        df_clean = df_clean[df_clean['Amount'].notna()]
        
        return df_clean
    
    def generate_quality_report(self) -> Dict:
        """Generate comprehensive data quality report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'quality_metrics': self.quality_engine.quality_metrics,
            'data_issues': self.quality_engine.data_issues,
            'summary': {
                'total_files_processed': len(set([issue[0] for issue in self.quality_engine.data_issues])),
                'total_issues_found': len(self.quality_engine.data_issues),
                'quality_score': self._calculate_quality_score()
            }
        }
        
        return report
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score"""
        if not self.quality_engine.quality_metrics:
            return 0.0
        
        total_issues = len(self.quality_engine.data_issues)
        total_metrics = len(self.quality_engine.quality_metrics)
        
        # Simple scoring: fewer issues = higher score
        base_score = 100
        penalty_per_issue = 5
        penalty_per_metric = 2
        
        score = base_score - (total_issues * penalty_per_issue) - (total_metrics * penalty_per_metric)
        return max(0, min(100, score))

def main():
    """Main function to demonstrate data cleaning capabilities"""
    cleaner = EcommerceDataCleaner()
    
    # Load and clean each dataset
    datasets = {
        'Sale Report': 'Sale Report.csv',
        'May-2022': 'May-2022.csv',
        'Expense': 'Expense IIGF.csv'
    }
    
    cleaned_data = {}
    
    for name, file_path in datasets.items():
        try:
            # Load with validation
            df = cleaner.quality_engine.load_and_validate_data(file_path)
            
            # Apply specific cleaning
            if 'Sale Report' in name:
                df_clean = cleaner.clean_sales_data(df)
            elif 'May-2022' in name:
                df_clean = cleaner.clean_pricing_data(df)
            elif 'Expense' in name:
                df_clean = cleaner.clean_expense_data(df)
            else:
                df_clean = df
            
            cleaned_data[name] = df_clean
            logger.info(f"Successfully cleaned {name}: {len(df_clean)} rows")
            
        except Exception as e:
            logger.error(f"Error processing {name}: {str(e)}")
    
    # Generate quality report
    quality_report = cleaner.generate_quality_report()
    
    print("=== Data Quality Report ===")
    print(f"Quality Score: {quality_report['summary']['quality_score']:.1f}/100")
    print(f"Total Issues: {quality_report['summary']['total_issues_found']}")
    print(f"Files Processed: {quality_report['summary']['total_files_processed']}")
    
    return cleaned_data, quality_report

if __name__ == "__main__":
    cleaned_data, quality_report = main() 