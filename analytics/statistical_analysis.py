"""
Advanced Statistical Analysis Module
Demonstrates statistical analysis, regression, time series, forecasting, and hypothesis testing skills
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for e-commerce data
    Demonstrates advanced statistical and mathematical analysis techniques
    """
    
    def __init__(self):
        self.analysis_results = {}
        self.models = {}
    
    def correlation_analysis(self, df: pd.DataFrame, numeric_cols: list) -> dict:
        """
        Perform comprehensive correlation analysis
        """
        results = {}
        
        # Pearson correlation
        pearson_corr = df[numeric_cols].corr(method='pearson')
        results['pearson_correlation'] = pearson_corr
        
        # Spearman correlation
        spearman_corr = df[numeric_cols].corr(method='spearman')
        results['spearman_correlation'] = spearman_corr
        
        # Significant correlations
        significant_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                corr, p_value = pearsonr(df[col1].dropna(), df[col2].dropna())
                if p_value < 0.05:
                    significant_correlations.append({
                        'variables': (col1, col2),
                        'correlation': corr,
                        'p_value': p_value,
                        'strength': self._interpret_correlation(corr)
                    })
        
        results['significant_correlations'] = significant_correlations
        
        return results
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            return "Very Strong"
        elif abs_corr >= 0.6:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"
    
    def regression_analysis(self, df: pd.DataFrame, target: str, features: list) -> dict:
        """
        Perform linear regression analysis
        """
        # Prepare data
        X = df[features].dropna()
        y = df[target].dropna()
        
        # Align data
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) < 10:
            return {"error": "Insufficient data for regression analysis"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Model evaluation
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'coefficient': model.coef_,
            'abs_coefficient': abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        results = {
            'model': model,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mse_train': mse_train,
            'mse_test': mse_test,
            'feature_importance': feature_importance,
            'intercept': model.intercept_,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            }
        }
        
        self.models['regression'] = results
        return results
    
    def time_series_analysis(self, df: pd.DataFrame, date_col: str, value_col: str) -> dict:
        """
        Perform time series analysis and decomposition
        """
        # Prepare time series data
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.set_index(date_col)
        df_ts = df_ts.sort_index()
        
        # Resample to monthly data if needed
        if len(df_ts) > 12:
            monthly_data = df_ts[value_col].resample('M').mean()
        else:
            monthly_data = df_ts[value_col]
        
        # Remove missing values
        monthly_data = monthly_data.dropna()
        
        if len(monthly_data) < 4:
            return {"error": "Insufficient time series data"}
        
        results = {}
        
        # Basic statistics
        results['basic_stats'] = {
            'mean': monthly_data.mean(),
            'std': monthly_data.std(),
            'min': monthly_data.min(),
            'max': monthly_data.max(),
            'trend': self._calculate_trend(monthly_data)
        }
        
        # Seasonal decomposition
        try:
            decomposition = seasonal_decompose(monthly_data, period=min(12, len(monthly_data)//2))
            results['decomposition'] = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
        except Exception as e:
            results['decomposition_error'] = str(e)
        
        # Autocorrelation analysis
        try:
            acf = pd.Series(monthly_data).autocorr()
            results['autocorrelation'] = acf
        except Exception as e:
            results['autocorrelation_error'] = str(e)
        
        return results
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        if len(series) < 2:
            return "Insufficient data"
        
        # Simple linear trend
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        
        if p_value < 0.05:
            if slope > 0:
                return "Increasing"
            else:
                return "Decreasing"
        else:
            return "No significant trend"
    
    def forecasting_analysis(self, df: pd.DataFrame, date_col: str, value_col: str, periods: int = 12) -> dict:
        """
        Perform time series forecasting
        """
        # Prepare data
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.set_index(date_col)
        df_ts = df_ts.sort_index()
        
        # Resample to monthly data
        monthly_data = df_ts[value_col].resample('M').mean().dropna()
        
        if len(monthly_data) < 6:
            return {"error": "Insufficient data for forecasting"}
        
        results = {}
        
        # Simple moving average forecast
        ma_forecast = monthly_data.rolling(window=3).mean().iloc[-1]
        results['moving_average_forecast'] = ma_forecast
        
        # Exponential smoothing
        alpha = 0.3
        exp_smooth = monthly_data.ewm(alpha=alpha).mean()
        exp_forecast = exp_smooth.iloc[-1]
        results['exponential_smoothing_forecast'] = exp_forecast
        
        # ARIMA forecasting (if enough data)
        if len(monthly_data) >= 12:
            try:
                # Simple ARIMA model
                model = ARIMA(monthly_data, order=(1, 1, 1))
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=periods)
                results['arima_forecast'] = forecast.tolist()
                results['arima_aic'] = fitted_model.aic
            except Exception as e:
                results['arima_error'] = str(e)
        
        # Forecast accuracy metrics
        if len(monthly_data) >= 8:
            # Use last 4 points for validation
            train = monthly_data[:-4]
            test = monthly_data[-4:]
            
            # Simple forecast for validation
            ma_val = train.rolling(window=3).mean()
            mae = np.mean(np.abs(test - ma_val.iloc[-4:]))
            mape = np.mean(np.abs((test - ma_val.iloc[-4:]) / test)) * 100
            
            results['forecast_accuracy'] = {
                'mae': mae,
                'mape': mape
            }
        
        return results
    
    def hypothesis_testing(self, df: pd.DataFrame, test_type: str, **kwargs) -> dict:
        """
        Perform various hypothesis tests
        """
        results = {}
        
        if test_type == "t_test":
            # Independent t-test
            group1 = kwargs.get('group1', [])
            group2 = kwargs.get('group2', [])
            
            if len(group1) > 0 and len(group2) > 0:
                t_stat, p_value = stats.ttest_ind(group1, group2)
                results = {
                    'test_type': 'Independent t-test',
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': self._calculate_cohens_d(group1, group2)
                }
        
        elif test_type == "chi_square":
            # Chi-square test for independence
            contingency_table = kwargs.get('contingency_table', None)
            
            if contingency_table is not None:
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                results = {
                    'test_type': 'Chi-square test',
                    'chi2_statistic': chi2_stat,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'significant': p_value < 0.05
                }
        
        elif test_type == "anova":
            # One-way ANOVA
            groups = kwargs.get('groups', [])
            
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                results = {
                    'test_type': 'One-way ANOVA',
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return results
    
    def _calculate_cohens_d(self, group1: list, group2: list) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1) + (n2 - 1) * np.var(group2)) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def outlier_analysis(self, df: pd.DataFrame, columns: list) -> dict:
        """
        Perform comprehensive outlier analysis
        """
        results = {}
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                data = df[col].dropna()
                
                # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
                
                # Z-score method
                z_scores = np.abs(stats.zscore(data))
                outliers_zscore = data[z_scores > 3]
                
                results[col] = {
                    'iqr_outliers': len(outliers_iqr),
                    'zscore_outliers': len(outliers_zscore),
                    'outlier_percentage_iqr': (len(outliers_iqr) / len(data)) * 100,
                    'outlier_percentage_zscore': (len(outliers_zscore) / len(data)) * 100,
                    'iqr_bounds': (lower_bound, upper_bound),
                    'outlier_values_iqr': outliers_iqr.tolist(),
                    'outlier_values_zscore': outliers_zscore.tolist()
                }
        
        return results
    
    def distribution_analysis(self, df: pd.DataFrame, columns: list) -> dict:
        """
        Analyze data distributions
        """
        results = {}
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                data = df[col].dropna()
                
                # Basic statistics
                stats_dict = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'skewness': data.skew(),
                    'kurtosis': data.kurtosis(),
                    'min': data.min(),
                    'max': data.max(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75)
                }
                
                # Normality test
                shapiro_stat, shapiro_p = stats.shapiro(data)
                stats_dict['shapiro_test'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
                
                # Distribution fit
                distributions = ['norm', 'expon', 'gamma', 'lognorm']
                best_fit = None
                best_aic = np.inf
                
                for dist_name in distributions:
                    try:
                        dist = getattr(stats, dist_name)
                        params = dist.fit(data)
                        aic = stats.AIC(data, dist, params)
                        if aic < best_aic:
                            best_aic = aic
                            best_fit = dist_name
                    except:
                        continue
                
                stats_dict['best_fitting_distribution'] = best_fit
                stats_dict['aic'] = best_aic
                
                results[col] = stats_dict
        
        return results
    
    def generate_statistical_report(self) -> str:
        """
        Generate comprehensive statistical report
        """
        report = []
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary of analyses performed
        if self.analysis_results:
            report.append("ANALYSES PERFORMED:")
            for analysis_type, results in self.analysis_results.items():
                report.append(f"- {analysis_type}")
            report.append("")
        
        # Model performance summary
        if self.models:
            report.append("MODEL PERFORMANCE:")
            for model_name, model_results in self.models.items():
                if 'r2_test' in model_results:
                    report.append(f"- {model_name}: R² = {model_results['r2_test']:.3f}")
            report.append("")
        
        # Key findings
        report.append("KEY FINDINGS:")
        if self.analysis_results:
            for analysis_type, results in self.analysis_results.items():
                if analysis_type == 'correlation_analysis':
                    sig_corr = results.get('significant_correlations', [])
                    report.append(f"- Found {len(sig_corr)} significant correlations")
                elif analysis_type == 'outlier_analysis':
                    total_outliers = sum(len(results[col]['iqr_outliers']) for col in results)
                    report.append(f"- Identified {total_outliers} outliers across variables")
        
        return "\n".join(report)

def main():
    """Demonstrate statistical analysis capabilities"""
    print("=== Statistical Analysis Framework Demonstration ===")
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'price': np.random.normal(100, 20, n_samples),
        'stock': np.random.poisson(50, n_samples),
        'sales': np.random.exponential(10, n_samples),
        'margin': np.random.beta(2, 5, n_samples) * 100,
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Add some correlations
    sample_data['revenue'] = sample_data['price'] * sample_data['sales']
    sample_data['profit'] = sample_data['revenue'] * (sample_data['margin'] / 100)
    
    print("\n1. Correlation Analysis...")
    numeric_cols = ['price', 'stock', 'sales', 'margin', 'revenue', 'profit']
    corr_results = analyzer.correlation_analysis(sample_data, numeric_cols)
    
    print(f"Found {len(corr_results['significant_correlations'])} significant correlations")
    for corr in corr_results['significant_correlations'][:3]:
        print(f"- {corr['variables'][0]} vs {corr['variables'][1]}: {corr['correlation']:.3f} ({corr['strength']})")
    
    print("\n2. Regression Analysis...")
    reg_results = analyzer.regression_analysis(sample_data, 'revenue', ['price', 'sales', 'margin'])
    
    if 'error' not in reg_results:
        print(f"R² Score: {reg_results['r2_test']:.3f}")
        print(f"Top feature: {reg_results['feature_importance'].iloc[0]['feature']}")
    
    print("\n3. Outlier Analysis...")
    outlier_results = analyzer.outlier_analysis(sample_data, ['price', 'stock', 'sales'])
    
    for col, results in outlier_results.items():
        print(f"- {col}: {results['iqr_outliers']} outliers ({results['outlier_percentage_iqr']:.1f}%)")
    
    print("\n4. Distribution Analysis...")
    dist_results = analyzer.distribution_analysis(sample_data, ['price', 'stock', 'sales'])
    
    for col, results in dist_results.items():
        print(f"- {col}: Best fit = {results['best_fitting_distribution']}, Skewness = {results['skewness']:.3f}")
    
    print("\n5. Hypothesis Testing...")
    # T-test between categories
    category_a = sample_data[sample_data['category'] == 'A']['revenue']
    category_b = sample_data[sample_data['category'] == 'B']['revenue']
    
    t_test_results = analyzer.hypothesis_testing(
        sample_data, 
        "t_test", 
        group1=category_a, 
        group2=category_b
    )
    
    print(f"T-test p-value: {t_test_results['p_value']:.4f}")
    print(f"Significant difference: {t_test_results['significant']}")
    
    # Generate report
    print("\n6. Generating Statistical Report...")
    report = analyzer.generate_statistical_report()
    print(report[:500] + "...")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main() 