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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
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
        expense_df['Amount'] = pd.to_numeric(expense_df['Amount'], errors='coerce')
        expense_df = expense_df[expense_df['Amount'].notna()]
        self.data['expenses_clean'] = expense_df
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">📊 E-Commerce Analytics Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### Real-time monitoring and data-driven insights for e-commerce growth optimization")
        
        # Current timestamp
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M"))
        with col2:
            st.metric("Data Sources", len(self.data))
        with col3:
            st.metric("Total Products", len(self.data.get('sales_clean', pd.DataFrame())))
    
    def render_kpi_cards(self):
        """Render key performance indicators"""
        st.markdown("## 📈 Key Performance Indicators")
        
        if 'sales_clean' in self.data and not self.data['sales_clean'].empty:
            sales_df = self.data['sales_clean']
            
            # Calculate KPIs
            total_stock = sales_df['Stock'].sum()
            total_products = len(sales_df)
            avg_stock = sales_df['Stock'].mean()
            out_of_stock = len(sales_df[sales_df['Stock'] == 0])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Stock",
                    f"{total_stock:,.0f}",
                    delta=f"{total_stock - (total_stock * 0.95):,.0f}"
                )
            
            with col2:
                st.metric(
                    "Total Products",
                    f"{total_products:,}",
                    delta=f"{total_products - (total_products * 0.98):,}"
                )
            
            with col3:
                st.metric(
                    "Average Stock",
                    f"{avg_stock:.1f}",
                    delta=f"{avg_stock - (avg_stock * 0.95):.1f}"
                )
            
            with col4:
                st.metric(
                    "Out of Stock",
                    f"{out_of_stock:,}",
                    delta=f"{out_of_stock - (out_of_stock * 1.05):,}",
                    delta_color="inverse"
                )
    
    def render_inventory_analysis(self):
        """Render inventory analysis section"""
        st.markdown("## 📦 Inventory Analysis")
        
        if 'sales_clean' in self.data and not self.data['sales_clean'].empty:
            sales_df = self.data['sales_clean']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stock distribution by category
                category_stock = sales_df.groupby('Category')['Stock'].sum().sort_values(ascending=False)
                
                fig = px.bar(
                    x=category_stock.values,
                    y=category_stock.index,
                    orientation='h',
                    title="Stock Distribution by Category",
                    labels={'x': 'Total Stock', 'y': 'Category'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Stock status distribution
                stock_status = pd.cut(
                    sales_df['Stock'],
                    bins=[-np.inf, 0, 10, 50, 100, np.inf],
                    labels=['Out of Stock', 'Critical Low', 'Low', 'Medium', 'High']
                ).value_counts()
                
                fig = px.pie(
                    values=stock_status.values,
                    names=stock_status.index,
                    title="Stock Status Distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_pricing_analysis(self):
        """Render pricing analysis section"""
        st.markdown("## 💰 Pricing Analysis")
        
        if 'pricing_clean' in self.data and not self.data['pricing_clean'].empty:
            pricing_df = self.data['pricing_clean']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Price distribution
                if 'Final MRP Old' in pricing_df.columns:
                    fig = px.histogram(
                        pricing_df,
                        x='Final MRP Old',
                        nbins=30,
                        title="Price Distribution",
                        labels={'Final MRP Old': 'Price (₹)'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Margin analysis
                if 'Margin_Percentage' in pricing_df.columns:
                    margin_data = pricing_df[pricing_df['Margin_Percentage'].notna()]
                    
                    fig = px.box(
                        margin_data,
                        y='Margin_Percentage',
                        title="Margin Distribution by Category",
                        color='Category' if 'Category' in margin_data.columns else None
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_category_analysis(self):
        """Render category performance analysis"""
        st.markdown("## 🏷️ Category Performance")
        
        if 'sales_clean' in self.data and not self.data['sales_clean'].empty:
            sales_df = self.data['sales_clean']
            
            # Category analysis
            category_analysis = sales_df.groupby('Category').agg({
                'Stock': ['sum', 'mean', 'count'],
                'SKU Code': 'count'
            }).round(2)
            
            category_analysis.columns = ['Total_Stock', 'Avg_Stock', 'Product_Count']
            category_analysis = category_analysis.sort_values('Total_Stock', ascending=False)
            
            # Display category table
            st.dataframe(
                category_analysis,
                use_container_width=True,
                caption="Category Performance Summary"
            )
            
            # Category visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=category_analysis.index,
                    y=category_analysis['Total_Stock'],
                    title="Total Stock by Category",
                    labels={'x': 'Category', 'y': 'Total Stock'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=category_analysis.index,
                    y=category_analysis['Product_Count'],
                    title="Product Count by Category",
                    labels={'x': 'Category', 'y': 'Product Count'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_size_analysis(self):
        """Render size analysis"""
        st.markdown("## 📏 Size Analysis")
        
        if 'sales_clean' in self.data and not self.data['sales_clean'].empty:
            sales_df = self.data['sales_clean']
            
            # Size analysis
            size_analysis = sales_df.groupby('Size').agg({
                'Stock': ['sum', 'mean', 'count']
            }).round(2)
            
            size_analysis.columns = ['Total_Stock', 'Avg_Stock', 'Product_Count']
            size_analysis = size_analysis.sort_values('Total_Stock', ascending=False)
            
            # Display size table
            st.dataframe(
                size_analysis,
                use_container_width=True,
                caption="Size Performance Summary"
            )
            
            # Size visualization
            fig = px.bar(
                x=size_analysis.index,
                y=size_analysis['Total_Stock'],
                title="Stock Distribution by Size",
                labels={'x': 'Size', 'y': 'Total Stock'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts(self):
        """Render alerts and warnings"""
        st.markdown("## ⚠️ Alerts & Warnings")
        
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
            st.success("✅ No critical alerts at this time")
    
    def render_recommendations(self):
        """Render data-driven recommendations"""
        st.markdown("## 💡 Data-Driven Recommendations")
        
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
            st.markdown(f"{i}. {rec}")
    
    def render_export_options(self):
        """Render data export options"""
        st.markdown("## 📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'sales_clean' in self.data:
                csv = self.data['sales_clean'].to_csv(index=False)
                st.download_button(
                    label="Download Sales Data",
                    data=csv,
                    file_name="sales_analysis.csv",
                    mime="text/csv"
                )
        
        with col2:
            if 'pricing_clean' in self.data:
                csv = self.data['pricing_clean'].to_csv(index=False)
                st.download_button(
                    label="Download Pricing Data",
                    data=csv,
                    file_name="pricing_analysis.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Generate summary report
            if 'sales_clean' in self.data:
                summary = self.generate_summary_report()
                st.download_button(
                    label="Download Summary Report",
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
            st.markdown("## 🔍 Filters")
            
            if 'sales_clean' in self.data:
                categories = ['All'] + list(self.data['sales_clean']['Category'].unique())
                selected_category = st.selectbox("Select Category", categories)
                
                sizes = ['All'] + list(self.data['sales_clean']['Size'].unique())
                selected_size = st.selectbox("Select Size", sizes)
            
            st.markdown("## 📊 Quick Stats")
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
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem; margin-bottom: 2rem;">
        <h3>🚀 E-Commerce Analytics Dashboard</h3>
        <p>Real-time monitoring and data-driven insights for growth optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize and run dashboard
    dashboard = EcommerceDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 