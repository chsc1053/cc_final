# Retail Data Analytics Dashboard

## Project Overview

This project is a web-based analytics dashboard for retail transaction data analysis. It provides insights into customer behavior, product performance, and sales trends using data from household demographics, product information, and transaction records.

The application features:

- Data visualization for customer analytics, product analytics, and sales analytics
- Market basket analysis to identify commonly purchased product combinations
- Customer churn prediction to identify at-risk customers
- Data management with upload capabilities for CSV files

## Project Structure

### Core Application Files

- `app.py` - Main Flask application with all routes, database models, and analysis logic
- `requirements.txt` - Python dependencies required to run the application

### Data Files

- `/data/`
  - `400_households.csv` - Household demographic data (loyalty, age range, income range, etc.)
  - `400_products.csv` - Product information (department, commodity, brand type, etc.)
  - `400_transactions.csv` - Transaction records (purchases, dates, spend amounts, etc.)

### Database

- `/instance/`
  - `retail.db` - SQLite database storing all application data
  - `retail copy.db` - Backup copy of the database

### Frontend Assets

- `/static/`
  - `/css/style.css` - Custom CSS styling for the application
  - `/js/main.js` - JavaScript functionality for interactive elements

### HTML Templates

- `/templates/`
  - `base.html` - Base template with common layout elements
  - `index.html` - Landing page
  - `login.html` - User authentication page
  - `signup.html` - New user registration page
  - `dashboard.html` - Main dashboard with key metrics and navigation
  - `sample_data.html` - Display sample of integrated data from all sources
  - `search_data.html` - Search and view data by household ID
  - `upload_data.html` - Interface for uploading new CSV data files
  - `customer_analytics.html` - Customer segmentation and behavior analysis
  - `product_analytics.html` - Product performance and category analysis
  - `sales_analytics.html` - Sales trends and revenue analysis
  - `basket_analysis.html` - Market basket analysis showing product associations
  - `churn_prediction.html` - Customer churn risk analysis and segmentation

## Analysis Features

1. **Customer Analytics** - Visualizations of customer segments by income, household size, age range, and loyalty status

2. **Product Analytics** - Analysis of top-performing products, sales by department, brand type distribution, and organic vs. non-organic product sales

3. **Sales Analytics** - Trends in daily, weekly, and monthly sales, revenue by department, and basket size analysis

4. **Basket Analysis** - Co-occurrence analysis to identify products frequently purchased together, showing support, confidence, and lift metrics

5. **Churn Prediction** - RFM (Recency, Frequency, Monetary) analysis to identify customers at risk of churning, with demographic insights of at-risk segments

## Getting Started

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the Flask application:

   ```
   python app.py
   ```

3. Access the application at http://localhost:5000

4. Login with your credentials or sign up for a new account

## Data Management

The application supports uploading new data files through the Upload Data interface. Files must follow the specified naming convention:

- 400_households.csv
- 400_products.csv
- 400_transactions.csv
