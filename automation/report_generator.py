"""
Automated Report Generation System
Demonstrates Python automation, stakeholder communication, and data-driven insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import yaml
import logging
from typing import Dict, List, Optional, Tuple
import schedule
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedReportGenerator:
    """
    Automated report generation system for e-commerce analytics
    Demonstrates Python automation and stakeholder communication
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.reports = {}
        self.insights = {}
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            # Default configuration
            return {
                'data_sources': {
                    'sales': 'Sale Report.csv',
                    'pricing': 'May-2022.csv',
                    'expenses': 'Expense IIGF.csv'
                },
                'report_settings': {
                    'frequency': 'daily',
                    'recipients': ['stakeholder@company.com'],
                    'include_charts': True,
                    'include_recommendations': True
                },
                'kpi_thresholds': {
                    'low_stock': 10,
                    'high_stock': 100,
                    'low_margin': 10,
                    'critical_margin': 5
                }
            }
    
    def load_and_process_data(self) -> Dict[str, pd.DataFrame]:
        """Load and process all data sources"""
        data = {}
        
        for source_name, file_path in self.config['data_sources'].items():
            try:
                df = pd.read_csv(file_path)
                data[source_name] = self.clean_data(df, source_name)
                logger.info(f"Loaded {source_name}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error loading {source_name}: {str(e)}")
                data[source_name] = pd.DataFrame()
        
        return data
    
    def clean_data(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Clean data based on source type"""
        df_clean = df.copy()
        
        if source_name == 'sales':
            # Clean sales data
            df_clean = df_clean[df_clean['SKU Code'].notna()]
            df_clean = df_clean[df_clean['SKU Code'] != '#REF!']
            df_clean['Stock'] = pd.to_numeric(df_clean['Stock'], errors='coerce')
            df_clean = df_clean[df_clean['Stock'] >= 0]
            
        elif source_name == 'pricing':
            # Clean pricing data
            price_cols = ['TP', 'MRP Old', 'Final MRP Old', 'Amazon MRP', 'Flipkart MRP']
            for col in price_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            if 'TP' in df_clean.columns and 'Final MRP Old' in df_clean.columns:
                df_clean['Margin'] = df_clean['Final MRP Old'] - df_clean['TP']
                df_clean['Margin_Percentage'] = (df_clean['Margin'] / df_clean['Final MRP Old']) * 100
                
        elif source_name == 'expenses':
            # Clean expense data
            df_clean = df_clean[df_clean['Recived Amount'] != 'Particular']
            df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce')
            df_clean = df_clean[df_clean['Amount'].notna()]
        
        return df_clean
    
    def generate_executive_summary(self, data: Dict[str, pd.DataFrame]) -> str:
        """Generate executive summary report"""
        summary = []
        summary.append("# Executive Summary Report")
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        # Key metrics
        if 'sales' in data and not data['sales'].empty:
            sales_df = data['sales']
            total_stock = sales_df['Stock'].sum()
            total_products = len(sales_df)
            out_of_stock = len(sales_df[sales_df['Stock'] == 0])
            
            summary.append("## Key Metrics")
            summary.append(f"- **Total Stock**: {total_stock:,.0f} units")
            summary.append(f"- **Total Products**: {total_products:,}")
            summary.append(f"- **Out of Stock**: {out_of_stock:,} products")
            summary.append(f"- **Stock Utilization**: {((total_products - out_of_stock) / total_products * 100):.1f}%")
            summary.append("")
        
        # Financial summary
        if 'pricing' in data and not data['pricing'].empty:
            pricing_df = data['pricing']
            if 'Margin_Percentage' in pricing_df.columns:
                avg_margin = pricing_df['Margin_Percentage'].mean()
                low_margin_products = len(pricing_df[pricing_df['Margin_Percentage'] < 10])
                
                summary.append("## Financial Summary")
                summary.append(f"- **Average Margin**: {avg_margin:.1f}%")
                summary.append(f"- **Low Margin Products**: {low_margin_products}")
                summary.append("")
        
        # Alerts and recommendations
        alerts = self.generate_alerts(data)
        if alerts:
            summary.append("## Critical Alerts")
            for alert in alerts[:5]:  # Top 5 alerts
                summary.append(f"- {alert}")
            summary.append("")
        
        return "\n".join(summary)
    
    def generate_alerts(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        """Generate critical alerts"""
        alerts = []
        
        if 'sales' in data and not data['sales'].empty:
            sales_df = data['sales']
            
            # Low stock alerts
            low_stock = sales_df[sales_df['Stock'] < self.config['kpi_thresholds']['low_stock']]
            if len(low_stock) > 0:
                alerts.append(f"🚨 {len(low_stock)} products have low stock (< {self.config['kpi_thresholds']['low_stock']})")
            
            # Out of stock alerts
            out_of_stock = sales_df[sales_df['Stock'] == 0]
            if len(out_of_stock) > 0:
                alerts.append(f"⚠️ {len(out_of_stock)} products are out of stock")
            
            # High stock alerts
            high_stock = sales_df[sales_df['Stock'] > self.config['kpi_thresholds']['high_stock']]
            if len(high_stock) > 0:
                alerts.append(f"📦 {len(high_stock)} products have excessive stock (> {self.config['kpi_thresholds']['high_stock']})")
        
        if 'pricing' in data and not data['pricing'].empty:
            pricing_df = data['pricing']
            if 'Margin_Percentage' in pricing_df.columns:
                low_margin = pricing_df[pricing_df['Margin_Percentage'] < self.config['kpi_thresholds']['low_margin']]
                if len(low_margin) > 0:
                    alerts.append(f"💰 {len(low_margin)} products have low margins (< {self.config['kpi_thresholds']['low_margin']}%)")
                
                critical_margin = pricing_df[pricing_df['Margin_Percentage'] < self.config['kpi_thresholds']['critical_margin']]
                if len(critical_margin) > 0:
                    alerts.append(f"🚨 {len(critical_margin)} products have critical margins (< {self.config['kpi_thresholds']['critical_margin']}%)")
        
        return alerts
    
    def generate_detailed_analysis(self, data: Dict[str, pd.DataFrame]) -> str:
        """Generate detailed analysis report"""
        analysis = []
        analysis.append("# Detailed Analysis Report")
        analysis.append("")
        
        # Category analysis
        if 'sales' in data and not data['sales'].empty:
            sales_df = data['sales']
            
            analysis.append("## Category Performance Analysis")
            category_analysis = sales_df.groupby('Category').agg({
                'Stock': ['sum', 'mean', 'count']
            }).round(2)
            
            category_analysis.columns = ['Total_Stock', 'Avg_Stock', 'Product_Count']
            category_analysis = category_analysis.sort_values('Total_Stock', ascending=False)
            
            analysis.append("### Top Categories by Stock")
            for category, row in category_analysis.head(5).iterrows():
                analysis.append(f"- **{category}**: {row['Total_Stock']:,.0f} units, {row['Product_Count']} products")
            analysis.append("")
        
        # Pricing analysis
        if 'pricing' in data and not data['pricing'].empty:
            pricing_df = data['pricing']
            
            analysis.append("## Pricing Analysis")
            
            if 'Margin_Percentage' in pricing_df.columns:
                margin_stats = pricing_df['Margin_Percentage'].describe()
                analysis.append(f"- **Average Margin**: {margin_stats['mean']:.1f}%")
                analysis.append(f"- **Margin Range**: {margin_stats['min']:.1f}% - {margin_stats['max']:.1f}%")
                analysis.append(f"- **Margin Std Dev**: {margin_stats['std']:.1f}%")
                analysis.append("")
            
            # Platform pricing comparison
            platform_cols = ['Amazon MRP', 'Flipkart MRP', 'Myntra MRP']
            available_platforms = [col for col in platform_cols if col in pricing_df.columns]
            
            if len(available_platforms) >= 2:
                analysis.append("### Cross-Platform Pricing")
                for platform in available_platforms:
                    avg_price = pricing_df[platform].mean()
                    analysis.append(f"- **{platform}**: ₹{avg_price:.2f}")
                analysis.append("")
        
        # Size analysis
        if 'sales' in data and not data['sales'].empty:
            sales_df = data['sales']
            
            analysis.append("## Size Analysis")
            size_analysis = sales_df.groupby('Size')['Stock'].sum().sort_values(ascending=False)
            
            analysis.append("### Stock Distribution by Size")
            for size, stock in size_analysis.head(5).items():
                analysis.append(f"- **{size}**: {stock:,.0f} units")
            analysis.append("")
        
        return "\n".join(analysis)
    
    def generate_recommendations(self, data: Dict[str, pd.DataFrame]) -> str:
        """Generate actionable recommendations"""
        recommendations = []
        recommendations.append("# Strategic Recommendations")
        recommendations.append("")
        
        if 'sales' in data and not data['sales'].empty:
            sales_df = data['sales']
            
            # Top performing categories
            category_performance = sales_df.groupby('Category')['Stock'].sum().sort_values(ascending=False)
            top_categories = category_performance.head(3)
            
            recommendations.append("## Inventory Optimization")
            recommendations.append(f"**Focus on top categories**: {', '.join(top_categories.index)}")
            recommendations.append("**Action**: Increase stock for these high-performing categories")
            recommendations.append("")
            
            # Restock recommendations
            out_of_stock = sales_df[sales_df['Stock'] == 0]
            if len(out_of_stock) > 0:
                recommendations.append(f"**Restock Priority**: {len(out_of_stock)} out-of-stock products")
                recommendations.append("**Action**: Immediate restocking required for revenue continuity")
                recommendations.append("")
            
            # Size optimization
            size_performance = sales_df.groupby('Size')['Stock'].sum()
            best_size = size_performance.idxmax()
            recommendations.append(f"**Size Strategy**: {best_size} size shows highest demand")
            recommendations.append("**Action**: Optimize inventory mix towards this size")
            recommendations.append("")
        
        if 'pricing' in data and not data['pricing'].empty:
            pricing_df = data['pricing']
            
            if 'Margin_Percentage' in pricing_df.columns:
                avg_margin = pricing_df['Margin_Percentage'].mean()
                
                if avg_margin < 20:
                    recommendations.append("## Pricing Strategy")
                    recommendations.append("**Issue**: Low average margins detected")
                    recommendations.append("**Action**: Review pricing strategy and consider price increases")
                    recommendations.append("")
                
                # Low margin products
                low_margin = pricing_df[pricing_df['Margin_Percentage'] < 10]
                if len(low_margin) > 0:
                    recommendations.append(f"**Margin Optimization**: {len(low_margin)} products have margins < 10%")
                    recommendations.append("**Action**: Review pricing for these products")
                    recommendations.append("")
        
        # Cross-platform optimization
        if 'pricing' in data and not data['pricing'].empty:
            pricing_df = data['pricing']
            platform_cols = ['Amazon MRP', 'Flipkart MRP', 'Myntra MRP']
            available_platforms = [col for col in platform_cols if col in pricing_df.columns]
            
            if len(available_platforms) >= 2:
                platform_analysis = {}
                for platform in available_platforms:
                    platform_analysis[platform] = pricing_df[platform].mean()
                
                recommendations.append("## Platform Strategy")
                best_platform = max(platform_analysis, key=platform_analysis.get)
                recommendations.append(f"**Best Platform**: {best_platform} shows highest average prices")
                recommendations.append("**Action**: Optimize inventory allocation to this platform")
                recommendations.append("")
        
        return "\n".join(recommendations)
    
    def create_visualizations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Create visualizations for the report"""
        charts = {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        if 'sales' in data and not data['sales'].empty:
            sales_df = data['sales']
            
            # Category stock distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            category_stock = sales_df.groupby('Category')['Stock'].sum().sort_values(ascending=False)
            category_stock.head(10).plot(kind='bar', ax=ax)
            ax.set_title('Stock Distribution by Category (Top 10)')
            ax.set_xlabel('Category')
            ax.set_ylabel('Total Stock')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            charts['category_stock'] = 'category_stock.png'
            plt.savefig('category_stock.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Stock status pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            stock_status = pd.cut(
                sales_df['Stock'],
                bins=[-np.inf, 0, 10, 50, 100, np.inf],
                labels=['Out of Stock', 'Critical Low', 'Low', 'Medium', 'High']
            ).value_counts()
            
            ax.pie(stock_status.values, labels=stock_status.index, autopct='%1.1f%%')
            ax.set_title('Stock Status Distribution')
            charts['stock_status'] = 'stock_status.png'
            plt.savefig('stock_status.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        if 'pricing' in data and not data['pricing'].empty:
            pricing_df = data['pricing']
            
            # Price distribution
            if 'Final MRP Old' in pricing_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                pricing_df['Final MRP Old'].hist(bins=30, ax=ax, alpha=0.7)
                ax.set_title('Price Distribution')
                ax.set_xlabel('Price (₹)')
                ax.set_ylabel('Frequency')
                charts['price_distribution'] = 'price_distribution.png'
                plt.savefig('price_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Margin analysis
            if 'Margin_Percentage' in pricing_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                pricing_df['Margin_Percentage'].hist(bins=30, ax=ax, alpha=0.7)
                ax.set_title('Margin Distribution')
                ax.set_xlabel('Margin Percentage (%)')
                ax.set_ylabel('Frequency')
                charts['margin_distribution'] = 'margin_distribution.png'
                plt.savefig('margin_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        return charts
    
    def generate_complete_report(self) -> Dict[str, str]:
        """Generate complete automated report"""
        logger.info("Starting automated report generation...")
        
        # Load and process data
        data = self.load_and_process_data()
        
        # Generate report sections
        executive_summary = self.generate_executive_summary(data)
        detailed_analysis = self.generate_detailed_analysis(data)
        recommendations = self.generate_recommendations(data)
        
        # Create visualizations
        charts = self.create_visualizations(data)
        
        # Combine into complete report
        complete_report = f"""
{executive_summary}

{detailed_analysis}

{recommendations}

## Generated Visualizations
"""
        
        for chart_name, chart_path in charts.items():
            complete_report += f"- {chart_name}: {chart_path}\n"
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"ecommerce_report_{timestamp}.md"
        
        with open(report_filename, 'w') as f:
            f.write(complete_report)
        
        logger.info(f"Report generated: {report_filename}")
        
        return {
            'report_file': report_filename,
            'charts': charts,
            'alerts': self.generate_alerts(data),
            'timestamp': timestamp
        }
    
    def schedule_reports(self):
        """Schedule automated report generation"""
        frequency = self.config['report_settings']['frequency']
        
        if frequency == 'daily':
            schedule.every().day.at("09:00").do(self.generate_complete_report)
        elif frequency == 'weekly':
            schedule.every().monday.at("09:00").do(self.generate_complete_report)
        elif frequency == 'monthly':
            schedule.every().month.at("09:00").do(self.generate_complete_report)
        
        logger.info(f"Scheduled reports to run {frequency}")
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def send_email_report(self, report_data: Dict[str, str]):
        """Send report via email"""
        try:
            # Email configuration (would be in config)
            sender_email = "analytics@company.com"
            sender_password = "password"
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ", ".join(self.config['report_settings']['recipients'])
            msg['Subject'] = f"E-Commerce Analytics Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Email body
            body = f"""
            Dear Stakeholders,
            
            Please find attached the latest e-commerce analytics report.
            
            Key Highlights:
            - {len(report_data['alerts'])} critical alerts identified
            - Report generated at {report_data['timestamp']}
            
            Best regards,
            Analytics Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach report file
            with open(report_data['report_file'], 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {report_data["report_file"]}'
            )
            msg.attach(part)
            
            # Send email (commented out for demo)
            # server = smtplib.SMTP('smtp.gmail.com', 587)
            # server.starttls()
            # server.login(sender_email, sender_password)
            # server.send_message(msg)
            # server.quit()
            
            logger.info("Email report sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")

def main():
    """Demonstrate automated report generation"""
    print("=== Automated Report Generation System ===")
    
    # Initialize report generator
    generator = AutomatedReportGenerator()
    
    # Generate report
    print("\n1. Generating automated report...")
    report_data = generator.generate_complete_report()
    
    print(f"Report generated: {report_data['report_file']}")
    print(f"Charts created: {len(report_data['charts'])}")
    print(f"Alerts identified: {len(report_data['alerts'])}")
    
    # Display sample alerts
    if report_data['alerts']:
        print("\nSample Alerts:")
        for alert in report_data['alerts'][:3]:
            print(f"- {alert}")
    
    # Schedule reports (optional)
    print("\n2. Scheduling automated reports...")
    print("Reports will be generated daily at 9:00 AM")
    # generator.schedule_reports()  # Uncomment to enable scheduling
    
    return generator

if __name__ == "__main__":
    generator = main() 