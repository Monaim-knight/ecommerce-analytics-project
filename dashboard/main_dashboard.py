"""
Interactive E-Commerce Analytics Dashboard
Demonstrates real-time monitoring, stakeholder communication, and data visualization skills
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: none;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Alert styling */
    .alert {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border: none;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 5px solid #ff6b6b;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 5px solid #4CAF50;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
</style>
""", unsafe_allow_html=True)

class EcommerceDashboard:
    """
    Comprehensive e-commerce analytics dashboard
    Demonstrates real-time monitoring and stakeholder communication
    """
    
    def __init__(self):
        self.data = {}
        self.load_data()
    
    def load_data(self):
        """Load and prepare data for dashboard"""
        try:
            # Load sales data
            self.data['sales'] = pd.read_csv('Sale Report.csv')
            self.data['pricing'] = pd.read_csv('May-2022.csv')
            self.data['expenses'] = pd.read_csv('Expense IIGF.csv')
            
            # Clean and prepare data
            self.prepare_data()
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    def prepare_data(self):
        """Clean and prepare data for analysis"""
        # Clean sales data
        sales_df = self.data['sales'].copy()
        sales_df = sales_df[sales_df['SKU Code'].notna()]
        sales_df = sales_df[sales_df['SKU Code'] != '#REF!']
        sales_df['Stock'] = pd.to_numeric(sales_df['Stock'], errors='coerce')
        sales_df = sales_df[sales_df['Stock'] >= 0]
        self.data['sales_clean'] = sales_df
        
        # Clean pricing data
        pricing_df = self.data['pricing'].copy()
        price_cols = ['TP', 'MRP Old', 'Final MRP Old', 'Amazon MRP', 'Flipkart MRP']
        for col in price_cols:
            if col in pricing_df.columns:
                pricing_df[col] = pd.to_numeric(pricing_df[col], errors='coerce')
        
        if 'TP' in pricing_df.columns and 'Final MRP Old' in pricing_df.columns:
            pricing_df['Margin'] = pricing_df['Final MRP Old'] - pricing_df['TP']
            pricing_df['Margin_Percentage'] = (pricing_df['Margin'] / pricing_df['Final MRP Old']) * 100
        
        self.data['pricing_clean'] = pricing_df
        
        # Clean expense data
        expense_df = self.data['expenses'].copy()
        expense_df = expense_df[expense_df['Recived Amount'] != 'Particular']
        # Check if 'Amount' column exists, if not use 'Recived Amount'
        amount_col = 'Amount' if 'Amount' in expense_df.columns else 'Recived Amount'
        expense_df[amount_col] = pd.to_numeric(expense_df[amount_col], errors='coerce')
        expense_df = expense_df[expense_df[amount_col].notna()]
        self.data['expenses_clean'] = expense_df
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown("""
        <div class="main-header">
            <h1>📊 E-Commerce Analytics Dashboard</h1>
            <p>Real-time monitoring and data-driven insights for e-commerce growth optimization</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Current timestamp and stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🕒 Last Updated</h3>
                <p style="font-size: 1.2rem; margin: 0;">{}</p>
            </div>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📁 Data Sources</h3>
                <p style="font-size: 1.2rem; margin: 0;">{}</p>
            </div>
            """.format(len(self.data)), unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📦 Total Products</h3>
                <p style="font-size: 1.2rem; margin: 0;">{:,}</p>
            </div>
            """.format(len(self.data.get('sales_clean', pd.DataFrame()))), unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Dashboard Status</h3>
                <p style="font-size: 1.2rem; margin: 0;">🟢 Active</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_kpi_cards(self):
        """Render key performance indicators"""
        st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)
        
        if 'sales_clean' in self.data and not self.data['sales_clean'].empty:
            sales_df = self.data['sales_clean']
            
            # Calculate KPIs
            total_stock = sales_df['Stock'].sum()
            total_products = len(sales_df)
            avg_stock = sales_df['Stock'].mean()
            out_of_stock = len(sales_df[sales_df['Stock'] == 0])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>📦 Total Stock</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0;">{:,}</p>
                    <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0;">Units Available</p>
                </div>
                """.format(total_stock), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>🏷️ Total Products</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0;">{:,}</p>
                    <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0;">Active SKUs</p>
                </div>
                """.format(total_products), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>📊 Average Stock</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0;">{:.1f}</p>
                    <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0;">Per Product</p>
                </div>
                """.format(avg_stock), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3>⚠️ Out of Stock</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0;">{:,}</p>
                    <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0;">Need Restocking</p>
                </div>
                """.format(out_of_stock), unsafe_allow_html=True)
    
    def render_inventory_analysis(self):
        """Render inventory analysis section"""
        st.markdown('<div class="section-header">📦 Inventory Analysis</div>', unsafe_allow_html=True)
        
        if 'sales_clean' in self.data and not self.data['sales_clean'].empty:
            sales_df = self.data['sales_clean']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Stock distribution by category
                category_stock = sales_df.groupby('Category')['Stock'].sum().sort_values(ascending=False)
                
                fig = px.bar(
                    x=category_stock.values,
                    y=category_stock.index,
                    orientation='h',
                    title="Stock Distribution by Category",
                    labels={'x': 'Total Stock', 'y': 'Category'},
                    color=category_stock.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#333')
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Stock status distribution
                stock_status = pd.cut(
                    sales_df['Stock'],
                    bins=[-np.inf, 0, 10, 50, 100, np.inf],
                    labels=['Out of Stock', 'Critical Low', 'Low', 'Medium', 'High']
                ).value_counts()
                
                fig = px.pie(
                    values=stock_status.values,
                    names=stock_status.index,
                    title="Stock Status Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    def render_pricing_analysis(self):
        """Render pricing analysis section"""
        st.markdown('<div class="section-header">💰 Pricing Analysis</div>', unsafe_allow_html=True)
        
        if 'pricing_clean' in self.data and not self.data['pricing_clean'].empty:
            pricing_df = self.data['pricing_clean']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Price distribution
                if 'Final MRP Old' in pricing_df.columns:
                    fig = px.histogram(
                        pricing_df,
                        x='Final MRP Old',
                        nbins=30,
                        title="Price Distribution",
                        labels={'Final MRP Old': 'Price (₹)'},
                        color_discrete_sequence=['#667eea']
                    )
                    fig.update_layout(
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#333')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Margin analysis
                if 'Margin_Percentage' in pricing_df.columns:
                    margin_data = pricing_df[pricing_df['Margin_Percentage'].notna()]
                    
                    fig = px.box(
                        margin_data,
                        y='Margin_Percentage',
                        title="Margin Distribution",
                        color='Category' if 'Category' in margin_data.columns else None,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#333')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    def render_category_analysis(self):
        """Render category performance analysis"""
        st.markdown('<div class="section-header">🏷️ Category Performance</div>', unsafe_allow_html=True)
        
        if 'sales_clean' in self.data and not self.data['sales_clean'].empty:
            sales_df = self.data['sales_clean']
            
            # Category analysis
            category_analysis = sales_df.groupby('Category').agg({
                'Stock': ['sum', 'mean', 'count'],
                'SKU Code': 'count'
            }).round(2)
            
            # Flatten column names
            category_analysis.columns = ['_'.join(col).strip() for col in category_analysis.columns.values]
            # Rename columns for clarity
            category_analysis = category_analysis.rename(columns={
                'Stock_sum': 'Total_Stock',
                'Stock_mean': 'Avg_Stock', 
                'Stock_count': 'Stock_Count',
                'SKU Code_count': 'Product_Count'
            })
            category_analysis = category_analysis.sort_values('Total_Stock', ascending=False)
            
            # Display category table
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Category Performance Summary**")
            st.dataframe(
                category_analysis,
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Category visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = px.bar(
                    x=category_analysis.index,
                    y=category_analysis['Total_Stock'],
                    title="Total Stock by Category",
                    labels={'x': 'Category', 'y': 'Total Stock'},
                    color=category_analysis['Total_Stock'],
                    color_continuous_scale='viridis'
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#333')
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = px.bar(
                    x=category_analysis.index,
                    y=category_analysis['Product_Count'],
                    title="Product Count by Category",
                    labels={'x': 'Category', 'y': 'Product Count'},
                    color=category_analysis['Product_Count'],
                    color_continuous_scale='plasma'
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#333')
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    def render_size_analysis(self):
        """Render size analysis"""
        st.markdown('<div class="section-header">📏 Size Analysis</div>', unsafe_allow_html=True)
        
        if 'sales_clean' in self.data and not self.data['sales_clean'].empty:
            sales_df = self.data['sales_clean']
            
            # Size analysis
            size_analysis = sales_df.groupby('Size').agg({
                'Stock': ['sum', 'mean', 'count']
            }).round(2)
            
            # Flatten column names
            size_analysis.columns = ['_'.join(col).strip() for col in size_analysis.columns.values]
            # Rename columns for clarity
            size_analysis = size_analysis.rename(columns={
                'Stock_sum': 'Total_Stock',
                'Stock_mean': 'Avg_Stock',
                'Stock_count': 'Product_Count'
            })
            size_analysis = size_analysis.sort_values('Total_Stock', ascending=False)
            
            # Display size table
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Size Performance Summary**")
            st.dataframe(
                size_analysis,
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Size visualization
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = px.bar(
                x=size_analysis.index,
                y=size_analysis['Total_Stock'],
                title="Stock Distribution by Size",
                labels={'x': 'Size', 'y': 'Total Stock'},
                color=size_analysis['Total_Stock'],
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_alerts(self):
        """Render alerts and warnings"""
        st.markdown('<div class="section-header">⚠️ Alerts & Warnings</div>', unsafe_allow_html=True)
        
        alerts = []
        
        if 'sales_clean' in self.data and not self.data['sales_clean'].empty:
            sales_df = self.data['sales_clean']
            
            # Out of stock products
            out_of_stock = sales_df[sales_df['Stock'] == 0]
            if len(out_of_stock) > 0:
                alerts.append(f"🚨 {len(out_of_stock)} products are out of stock")
            
            # Low stock products
            low_stock = sales_df[sales_df['Stock'] < 10]
            if len(low_stock) > 0:
                alerts.append(f"⚠️ {len(low_stock)} products have low stock (< 10 units)")
            
            # High stock products
            high_stock = sales_df[sales_df['Stock'] > 100]
            if len(high_stock) > 0:
                alerts.append(f"📦 {len(high_stock)} products have high stock (> 100 units)")
        
        if 'pricing_clean' in self.data and not self.data['pricing_clean'].empty:
            pricing_df = self.data['pricing_clean']
            
            # Low margin products
            if 'Margin_Percentage' in pricing_df.columns:
                low_margin = pricing_df[pricing_df['Margin_Percentage'] < 10]
                if len(low_margin) > 0:
                    alerts.append(f"💰 {len(low_margin)} products have low margins (< 10%)")
        
        if alerts:
            for alert in alerts:
                st.markdown(f'<div class="alert">{alert}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-success">✅ No critical alerts at this time</div>', unsafe_allow_html=True)
    
    def render_recommendations(self):
        """Render data-driven recommendations"""
        st.markdown('<div class="section-header">💡 Data-Driven Recommendations</div>', unsafe_allow_html=True)
        
        recommendations = []
        
        if 'sales_clean' in self.data and not self.data['sales_clean'].empty:
            sales_df = self.data['sales_clean']
            
            # Top performing categories
            top_categories = sales_df.groupby('Category')['Stock'].sum().nlargest(3)
            recommendations.append(f"🎯 **Focus on top categories**: {', '.join(top_categories.index)}")
            
            # Restock recommendations
            out_of_stock = sales_df[sales_df['Stock'] == 0]
            if len(out_of_stock) > 0:
                recommendations.append(f"🔄 **Restock needed**: {len(out_of_stock)} out-of-stock products")
            
            # Size optimization
            size_performance = sales_df.groupby('Size')['Stock'].sum()
            best_size = size_performance.idxmax()
            recommendations.append(f"📏 **Size optimization**: {best_size} size performs best")
        
        if 'pricing_clean' in self.data and not self.data['pricing_clean'].empty:
            pricing_df = self.data['pricing_clean']
            
            if 'Margin_Percentage' in pricing_df.columns:
                avg_margin = pricing_df['Margin_Percentage'].mean()
                recommendations.append(f"💰 **Average margin**: {avg_margin:.1f}% - consider pricing strategy")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f'<div class="recommendation-card">{i}. {rec}</div>', unsafe_allow_html=True)
    
    def render_export_options(self):
        """Render data export options"""
        st.markdown('<div class="section-header">📤 Export Options</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'sales_clean' in self.data:
                csv = self.data['sales_clean'].to_csv(index=False)
                st.download_button(
                    label="📊 Download Sales Data",
                    data=csv,
                    file_name="sales_analysis.csv",
                    mime="text/csv"
                )
        
        with col2:
            if 'pricing_clean' in self.data:
                csv = self.data['pricing_clean'].to_csv(index=False)
                st.download_button(
                    label="💰 Download Pricing Data",
                    data=csv,
                    file_name="pricing_analysis.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Generate summary report
            if 'sales_clean' in self.data:
                summary = self.generate_summary_report()
                st.download_button(
                    label="📋 Download Summary Report",
                    data=summary,
                    file_name="summary_report.txt",
                    mime="text/plain"
                )
    
    def generate_summary_report(self):
        """Generate summary report"""
        report = []
        report.append("E-COMMERCE ANALYTICS SUMMARY REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if 'sales_clean' in self.data:
            sales_df = self.data['sales_clean']
            report.append("SALES SUMMARY:")
            report.append(f"- Total Products: {len(sales_df):,}")
            report.append(f"- Total Stock: {sales_df['Stock'].sum():,}")
            report.append(f"- Average Stock: {sales_df['Stock'].mean():.1f}")
            report.append(f"- Out of Stock: {len(sales_df[sales_df['Stock'] == 0]):,}")
            report.append("")
        
        if 'pricing_clean' in self.data:
            pricing_df = self.data['pricing_clean']
            if 'Margin_Percentage' in pricing_df.columns:
                report.append("PRICING SUMMARY:")
                report.append(f"- Average Margin: {pricing_df['Margin_Percentage'].mean():.1f}%")
                report.append(f"- Products with Low Margin: {len(pricing_df[pricing_df['Margin_Percentage'] < 10])}")
                report.append("")
        
        return "\n".join(report)
    
    def run_dashboard(self):
        """Run the complete dashboard"""
        self.render_header()
        
        # Sidebar filters
        with st.sidebar:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3>🔍 Filters</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if 'sales_clean' in self.data:
                categories = ['All'] + list(self.data['sales_clean']['Category'].unique())
                selected_category = st.selectbox("Select Category", categories)
                
                sizes = ['All'] + list(self.data['sales_clean']['Size'].unique())
                selected_size = st.selectbox("Select Size", sizes)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <h3>📊 Quick Stats</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if 'sales_clean' in self.data:
                sales_df = self.data['sales_clean']
                st.metric("Total Products", len(sales_df))
                st.metric("Total Stock", f"{sales_df['Stock'].sum():,}")
                st.metric("Out of Stock", len(sales_df[sales_df['Stock'] == 0]))
        
        # Main dashboard content
        self.render_kpi_cards()
        self.render_alerts()
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📦 Inventory", "💰 Pricing", "🏷️ Categories", "📏 Sizes"])
        
        with tab1:
            self.render_inventory_analysis()
        
        with tab2:
            self.render_pricing_analysis()
        
        with tab3:
            self.render_category_analysis()
        
        with tab4:
            self.render_size_analysis()
        
        # Recommendations and export
        self.render_recommendations()
        self.render_export_options()

def main():
    """Main function to run the dashboard"""
    # Initialize and run dashboard
    dashboard = EcommerceDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 