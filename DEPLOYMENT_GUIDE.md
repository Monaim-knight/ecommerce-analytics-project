# E-Commerce Analytics Dashboard - Deployment Guide

## Streamlit Cloud Deployment

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at https://share.streamlit.io/)

### Step 1: Prepare Your Repository
1. Ensure all files are committed to your GitHub repository
2. Make sure your repository is public (required for free Streamlit Cloud)

### Step 2: Deploy to Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `your-username/your-repo-name`
5. Set the main file path: `dashboard/main_dashboard.py`
6. Set Python version: `3.10`
7. Click "Deploy!"

### Step 3: Configure Environment
- The app will automatically install dependencies from `requirements.txt`
- Python 3.10.12 will be used (specified in `runtime.txt`)
- The app will be available at: `https://your-app-name.streamlit.app`

### Step 4: Share Your Dashboard
- Copy the URL from Streamlit Cloud
- Add it to your CV/resume as a live demo
- The dashboard will automatically update when you push changes to GitHub

### Troubleshooting
If you encounter deployment issues:

1. **Check requirements.txt**: Ensure all packages are compatible
2. **Verify file paths**: Make sure `dashboard/main_dashboard.py` exists
3. **Check data files**: Ensure CSV files are in the root directory
4. **Review logs**: Check the Streamlit Cloud logs for error messages

### Files Included in Deployment
- `dashboard/main_dashboard.py` - Main dashboard application
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version specification
- `.streamlit/config.toml` - Streamlit configuration
- CSV data files (Sale Report.csv, May-2022.csv, etc.)

### Features Demonstrated
- Real-time data visualization
- Interactive charts and graphs
- KPI monitoring
- Inventory analysis
- Pricing analysis
- Category performance tracking
- Size analysis
- Automated alerts and recommendations
- Data export capabilities

This dashboard showcases skills relevant to Product Manager roles:
- Data-driven decision making
- Real-time monitoring
- Stakeholder communication
- Growth optimization
- E-commerce analytics 