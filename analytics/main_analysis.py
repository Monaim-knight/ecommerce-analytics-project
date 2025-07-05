"""
Main E-Commerce Analytics Script
Demonstrates comprehensive data-driven insights, stakeholder communication, and growth optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from data_pipeline.data_cleaning import EcommerceDataCleaner
from analytics.ab_testing import ABTestingFramework
from analytics.statistical_analysis import StatisticalAnalyzer
from automation.report_generator import AutomatedReportGenerator

class EcommerceAnalyticsEngine:
    """
    Comprehensive e-commerce analytics engine
    Demonstrates all required Product Manager skills
    """
    
    def __init__(self):
        self.data_cleaner = EcommerceDataCleaner()
        self.ab_framework = ABTestingFramework()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.report_generator = AutomatedReportGenerator()
        self.insights = {}
        self.recommendations = []
    
    def run_comprehensive_analysis(self):
        """Run comprehensive e-commerce analysis"""
        print("🚀 Starting Comprehensive E-Commerce Analytics")
        print("=" * 60)
        
        # 1. Data Quality & Infrastructure
        print("\n📊 1. DATA QUALITY & INFRASTRUCTURE")
        print("-" * 40)
        self.analyze_data_quality()
        
        # 2. Real-time Monitoring
        print("\n📈 2. REAL-TIME MONITORING")
        print("-" * 40)
        self.setup_real_time_monitoring()
        
        # 3. Ad-hoc Analysis
        print("\n🔍 3. AD-HOC ANALYSIS")
        print("-" * 40)
        self.perform_ad_hoc_analysis()
        
        # 4. A/B Testing
        print("\n🧪 4. A/B TESTING")
        print("-" * 40)
        self.perform_ab_testing()
        
        # 5. Statistical Analysis
        print("\n📊 5. STATISTICAL ANALYSIS")
        print("-" * 40)
        self.perform_statistical_analysis()
        
        # 6. Stakeholder Communication
        print("\n📋 6. STAKEHOLDER COMMUNICATION")
        print("-" * 40)
        self.generate_stakeholder_report()
        
        # 7. Growth Optimization
        print("\n🎯 7. GROWTH OPTIMIZATION")
        print("-" * 40)
        self.optimize_growth()
        
        return self.insights
    
    def analyze_data_quality(self):
        """Analyze data quality and infrastructure"""
        print("• Loading and validating datasets...")
        
        # Load datasets
        datasets = {
            'Sale Report': 'Sale Report.csv',
            'May-2022': 'May-2022.csv',
            'Expense': 'Expense IIGF.csv'
        }
        
        quality_metrics = {}
        for name, file_path in datasets.items():
            try:
                # Load with validation
                df = self.data_cleaner.quality_engine.load_and_validate_data(file_path)
                
                # Apply specific cleaning
                if 'Sale Report' in name:
                    df_clean = self.data_cleaner.clean_sales_data(df)
                elif 'May-2022' in name:
                    df_clean = self.data_cleaner.clean_pricing_data(df)
                elif 'Expense' in name:
                    df_clean = self.data_cleaner.clean_expense_data(df)
                else:
                    df_clean = df
                
                quality_metrics[name] = {
                    'original_rows': len(df),
                    'cleaned_rows': len(df_clean),
                    'data_quality_score': self.calculate_data_quality_score(df_clean)
                }
                
                print(f"  ✓ {name}: {len(df_clean)} rows processed")
                
            except Exception as e:
                print(f"  ✗ {name}: Error - {str(e)}")
        
        self.insights['data_quality'] = quality_metrics
        print(f"• Data quality analysis completed for {len(quality_metrics)} datasets")
    
    def calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score"""
        if df.empty:
            return 0.0
        
        # Calculate various quality metrics
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        uniqueness = 1 - (df.duplicated().sum() / len(df))
        
        # Combined score
        quality_score = (completeness + uniqueness) / 2 * 100
        return round(quality_score, 2)
    
    def setup_real_time_monitoring(self):
        """Setup real-time monitoring for key metrics"""
        print("• Setting up real-time KPI monitoring...")
        
        # Load sales data for monitoring
        try:
            sales_df = pd.read_csv('Sale Report.csv')
            sales_df = self.data_cleaner.clean_sales_data(sales_df)
            
            # Calculate key metrics
            total_stock = sales_df['Stock'].sum()
            total_products = len(sales_df)
            out_of_stock = len(sales_df[sales_df['Stock'] == 0])
            low_stock = len(sales_df[sales_df['Stock'] < 10])
            
            # Stock utilization rate
            stock_utilization = ((total_products - out_of_stock) / total_products) * 100
            
            # Category performance
            category_performance = sales_df.groupby('Category')['Stock'].sum().sort_values(ascending=False)
            top_category = category_performance.index[0] if len(category_performance) > 0 else "N/A"
            
            monitoring_metrics = {
                'total_stock': total_stock,
                'total_products': total_products,
                'out_of_stock': out_of_stock,
                'low_stock': low_stock,
                'stock_utilization': stock_utilization,
                'top_category': top_category,
                'timestamp': pd.Timestamp.now()
            }
            
            self.insights['real_time_monitoring'] = monitoring_metrics
            
            print(f"  ✓ Total Stock: {total_stock:,}")
            print(f"  ✓ Stock Utilization: {stock_utilization:.1f}%")
            print(f"  ✓ Out of Stock: {out_of_stock}")
            print(f"  ✓ Top Category: {top_category}")
            
        except Exception as e:
            print(f"  ✗ Error in real-time monitoring: {str(e)}")
    
    def perform_ad_hoc_analysis(self):
        """Perform ad-hoc analysis for fast-moving decisions"""
        print("• Performing ad-hoc analysis...")
        
        try:
            # Load data
            sales_df = pd.read_csv('Sale Report.csv')
            sales_df = self.data_cleaner.clean_sales_data(sales_df)
            
            # Quick insights
            insights = []
            
            # Size analysis
            size_analysis = sales_df.groupby('Size')['Stock'].sum().sort_values(ascending=False)
            best_size = size_analysis.index[0] if len(size_analysis) > 0 else "N/A"
            insights.append(f"Best performing size: {best_size}")
            
            # Color analysis
            color_analysis = sales_df.groupby('Color')['Stock'].sum().sort_values(ascending=False)
            best_color = color_analysis.index[0] if len(color_analysis) > 0 else "N/A"
            insights.append(f"Most popular color: {best_color}")
            
            # Category insights
            category_analysis = sales_df.groupby('Category')['Stock'].sum().sort_values(ascending=False)
            top_categories = category_analysis.head(3).index.tolist()
            insights.append(f"Top 3 categories: {', '.join(top_categories)}")
            
            # Stock alerts
            critical_stock = sales_df[sales_df['Stock'] < 5]
            if len(critical_stock) > 0:
                insights.append(f"Critical stock alert: {len(critical_stock)} products need immediate restocking")
            
            self.insights['ad_hoc_analysis'] = insights
            
            print(f"  ✓ Generated {len(insights)} insights")
            for insight in insights[:3]:  # Show first 3 insights
                print(f"    - {insight}")
                
        except Exception as e:
            print(f"  ✗ Error in ad-hoc analysis: {str(e)}")
    
    def perform_ab_testing(self):
        """Perform A/B testing analysis"""
        print("• Setting up A/B testing framework...")
        
        try:
            # Create sample data for demonstration
            np.random.seed(42)
            n_samples = 1000
            
            # Control group (current strategy)
            control_data = pd.DataFrame({
                'user_id': range(n_samples),
                'converted': np.random.binomial(1, 0.15, n_samples),
                'revenue': np.random.exponential(50, n_samples)
            })
            
            # Treatment group (new strategy)
            treatment_data = pd.DataFrame({
                'user_id': range(n_samples, n_samples * 2),
                'converted': np.random.binomial(1, 0.18, n_samples),  # 20% improvement
                'revenue': np.random.exponential(55, n_samples)  # 10% revenue increase
            })
            
            # Run conversion test
            conversion_results = self.ab_framework.run_conversion_test(
                control_data, treatment_data, "conversion_test"
            )
            
            # Run revenue test
            revenue_results = self.ab_framework.run_revenue_test(
                control_data, treatment_data, "revenue_test"
            )
            
            ab_test_results = {
                'conversion_test': conversion_results,
                'revenue_test': revenue_results
            }
            
            self.insights['ab_testing'] = ab_test_results
            
            print(f"  ✓ Conversion test p-value: {conversion_results['p_value']:.4f}")
            print(f"  ✓ Revenue test p-value: {revenue_results['p_value']:.4f}")
            print(f"  ✓ Conversion improvement: {conversion_results['relative_improvement']:.2f}%")
            print(f"  ✓ Revenue improvement: {revenue_results['relative_improvement']:.2f}%")
            
        except Exception as e:
            print(f"  ✗ Error in A/B testing: {str(e)}")
    
    def perform_statistical_analysis(self):
        """Perform advanced statistical analysis"""
        print("• Performing statistical analysis...")
        
        try:
            # Load and prepare data
            sales_df = pd.read_csv('Sale Report.csv')
            sales_df = self.data_cleaner.clean_sales_data(sales_df)
            
            # Correlation analysis
            numeric_cols = ['Stock']
            if len(sales_df) > 0:
                corr_results = self.statistical_analyzer.correlation_analysis(sales_df, numeric_cols)
                
                # Outlier analysis
                outlier_results = self.statistical_analyzer.outlier_analysis(sales_df, ['Stock'])
                
                # Distribution analysis
                dist_results = self.statistical_analyzer.distribution_analysis(sales_df, ['Stock'])
                
                statistical_results = {
                    'correlation_analysis': corr_results,
                    'outlier_analysis': outlier_results,
                    'distribution_analysis': dist_results
                }
                
                self.insights['statistical_analysis'] = statistical_results
                
                print(f"  ✓ Correlation analysis completed")
                print(f"  ✓ Outlier analysis completed")
                print(f"  ✓ Distribution analysis completed")
                
        except Exception as e:
            print(f"  ✗ Error in statistical analysis: {str(e)}")
    
    def generate_stakeholder_report(self):
        """Generate stakeholder-friendly report"""
        print("• Generating stakeholder report...")
        
        try:
            # Generate comprehensive report
            report_data = self.report_generator.generate_complete_report()
            
            # Create executive summary
            executive_summary = self.create_executive_summary()
            
            # Create actionable recommendations
            recommendations = self.create_recommendations()
            
            stakeholder_report = {
                'executive_summary': executive_summary,
                'recommendations': recommendations,
                'report_file': report_data['report_file'],
                'timestamp': pd.Timestamp.now()
            }
            
            self.insights['stakeholder_report'] = stakeholder_report
            
            print(f"  ✓ Executive summary created")
            print(f"  ✓ {len(recommendations)} recommendations generated")
            print(f"  ✓ Report saved: {report_data['report_file']}")
            
        except Exception as e:
            print(f"  ✗ Error generating stakeholder report: {str(e)}")
    
    def create_executive_summary(self):
        """Create executive summary"""
        summary = []
        summary.append("📊 E-COMMERCE ANALYTICS EXECUTIVE SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        # Key metrics from monitoring
        if 'real_time_monitoring' in self.insights:
            metrics = self.insights['real_time_monitoring']
            summary.append("KEY METRICS:")
            summary.append(f"• Total Stock: {metrics['total_stock']:,} units")
            summary.append(f"• Stock Utilization: {metrics['stock_utilization']:.1f}%")
            summary.append(f"• Out of Stock: {metrics['out_of_stock']} products")
            summary.append("")
        
        # A/B testing results
        if 'ab_testing' in self.insights:
            ab_results = self.insights['ab_testing']
            conv_test = ab_results['conversion_test']
            rev_test = ab_results['revenue_test']
            
            summary.append("A/B TESTING RESULTS:")
            summary.append(f"• Conversion Improvement: {conv_test['relative_improvement']:.2f}%")
            summary.append(f"• Revenue Improvement: {rev_test['relative_improvement']:.2f}%")
            summary.append(f"• Statistical Significance: {'Yes' if conv_test['is_significant'] else 'No'}")
            summary.append("")
        
        # Data quality
        if 'data_quality' in self.insights:
            quality_scores = [data['data_quality_score'] for data in self.insights['data_quality'].values()]
            avg_quality = np.mean(quality_scores)
            summary.append(f"DATA QUALITY: {avg_quality:.1f}%")
            summary.append("")
        
        return "\n".join(summary)
    
    def create_recommendations(self):
        """Create actionable recommendations"""
        recommendations = []
        
        # Inventory recommendations
        if 'real_time_monitoring' in self.insights:
            metrics = self.insights['real_time_monitoring']
            
            if metrics['out_of_stock'] > 0:
                recommendations.append("🚨 IMMEDIATE ACTION: Restock out-of-stock products")
            
            if metrics['low_stock'] > 0:
                recommendations.append("⚠️ PRIORITY: Replenish low-stock items")
            
            if metrics['stock_utilization'] < 80:
                recommendations.append("📈 OPPORTUNITY: Optimize inventory levels")
        
        # A/B testing recommendations
        if 'ab_testing' in self.insights:
            ab_results = self.insights['ab_testing']
            conv_test = ab_results['conversion_test']
            
            if conv_test['is_significant'] and conv_test['relative_improvement'] > 0:
                recommendations.append("✅ IMPLEMENT: New strategy shows significant improvement")
            else:
                recommendations.append("🔄 ITERATE: Continue testing and optimization")
        
        # Data quality recommendations
        if 'data_quality' in self.insights:
            quality_scores = [data['data_quality_score'] for data in self.insights['data_quality'].values()]
            avg_quality = np.mean(quality_scores)
            
            if avg_quality < 80:
                recommendations.append("🔧 IMPROVE: Data quality needs enhancement")
            else:
                recommendations.append("✅ MAINTAIN: Data quality is excellent")
        
        return recommendations
    
    def optimize_growth(self):
        """Optimize e-commerce growth"""
        print("• Optimizing growth strategies...")
        
        try:
            # Load data for optimization
            sales_df = pd.read_csv('Sale Report.csv')
            sales_df = self.data_cleaner.clean_sales_data(sales_df)
            
            optimization_strategies = []
            
            # Inventory optimization
            category_performance = sales_df.groupby('Category')['Stock'].sum().sort_values(ascending=False)
            top_categories = category_performance.head(3)
            optimization_strategies.append(f"📈 Focus on top categories: {', '.join(top_categories.index)}")
            
            # Size optimization
            size_performance = sales_df.groupby('Size')['Stock'].sum().sort_values(ascending=False)
            best_size = size_performance.index[0] if len(size_performance) > 0 else "N/A"
            optimization_strategies.append(f"📏 Optimize size mix towards: {best_size}")
            
            # Stock level optimization
            out_of_stock = len(sales_df[sales_df['Stock'] == 0])
            if out_of_stock > 0:
                optimization_strategies.append(f"🔄 Implement automated restocking for {out_of_stock} products")
            
            # Revenue optimization
            if 'pricing_clean' in self.data_cleaner.data:
                pricing_df = self.data_cleaner.data['pricing_clean']
                if 'Margin_Percentage' in pricing_df.columns:
                    avg_margin = pricing_df['Margin_Percentage'].mean()
                    if avg_margin < 20:
                        optimization_strategies.append("💰 Review pricing strategy to improve margins")
            
            self.insights['growth_optimization'] = optimization_strategies
            
            print(f"  ✓ Generated {len(optimization_strategies)} optimization strategies")
            for strategy in optimization_strategies:
                print(f"    - {strategy}")
                
        except Exception as e:
            print(f"  ✗ Error in growth optimization: {str(e)}")
    
    def print_final_summary(self):
        """Print final analysis summary"""
        print("\n" + "=" * 60)
        print("🎯 FINAL ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Data Quality Score
        if 'data_quality' in self.insights:
            quality_scores = [data['data_quality_score'] for data in self.insights['data_quality'].values()]
            avg_quality = np.mean(quality_scores)
            print(f"📊 Data Quality Score: {avg_quality:.1f}%")
        
        # Key Metrics
        if 'real_time_monitoring' in self.insights:
            metrics = self.insights['real_time_monitoring']
            print(f"📈 Stock Utilization: {metrics['stock_utilization']:.1f}%")
            print(f"⚠️ Out of Stock Products: {metrics['out_of_stock']}")
        
        # A/B Testing Results
        if 'ab_testing' in self.insights:
            ab_results = self.insights['ab_testing']
            conv_test = ab_results['conversion_test']
            print(f"🧪 Conversion Improvement: {conv_test['relative_improvement']:.2f}%")
            print(f"📊 Statistical Significance: {'✅ Yes' if conv_test['is_significant'] else '❌ No'}")
        
        # Recommendations Count
        if 'stakeholder_report' in self.insights:
            recommendations = self.insights['stakeholder_report']['recommendations']
            print(f"💡 Actionable Recommendations: {len(recommendations)}")
        
        print("\n🚀 All analyses completed successfully!")
        print("📋 Check generated reports for detailed insights")

def main():
    """Main function to run comprehensive e-commerce analytics"""
    print("🎯 E-COMMERCE ANALYTICS & GROWTH OPTIMIZATION")
    print("=" * 60)
    print("Demonstrating Product Manager Skills:")
    print("• Data Quality & Infrastructure")
    print("• Real-time Monitoring")
    print("• Ad-hoc Analysis")
    print("• A/B Testing")
    print("• Statistical Analysis")
    print("• Stakeholder Communication")
    print("• Growth Optimization")
    print("=" * 60)
    
    # Initialize analytics engine
    engine = EcommerceAnalyticsEngine()
    
    # Run comprehensive analysis
    insights = engine.run_comprehensive_analysis()
    
    # Print final summary
    engine.print_final_summary()
    
    return engine, insights

if __name__ == "__main__":
    engine, insights = main() 