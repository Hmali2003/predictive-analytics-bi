# Predictive Analytics-Based Business Intelligence System for Sales Forecasting

**Developed by:** Harshal Mali  
**Course:** Master of Computer Applications (MCA)  
**Project Domain:** Predictive Analytics in Business Intelligence

## Project Overview

The Predictive Analytics-Based Business Intelligence System for Sales Forecasting is a Python-based web application that analyzes historical sales data, visualizes business performance, and predicts future revenue.

The system combines Business Intelligence (BI) and Predictive Analytics to help users understand sales trends and make data-driven business decisions.

Users can upload a sales dataset in CSV format and explore interactive dashboards, business metrics, and six-month revenue forecasts.

## Objectives

- Analyze historical sales data.
- Display key business performance indicators.
- Visualize monthly revenue trends and sales performance.
- Identify top-performing products and sales channels.
- Predict future revenue using time-series forecasting.
- Evaluate forecasting performance using MAE and MAPE-derived accuracy.

## Features

### 1. CSV Upload and Validation
- Upload sales datasets in CSV format.
- Validate the required columns.
- Display clear error messages for missing columns or invalid data.

### 2. Data Cleaning
- Convert date and revenue fields into appropriate formats.
- Remove records with invalid dates or revenue values.

### 3. Business Intelligence Dashboard
The dashboard displays:
- Total Revenue
- Total Orders
- Unique Customers
- Average Order Value (AOV)
- Month-over-Month (MoM) Revenue Change

### 4. Interactive Visualizations
- Monthly Revenue Trend
- Sales by Channel
- Top Products by Revenue

### 5. Predictive Revenue Forecasting
The system uses the Prophet time-series forecasting model to predict revenue for the next six months. It also displays predicted values and forecast ranges.

### 6. Model Evaluation
- Mean Absolute Error (MAE)
- MAPE-derived forecast accuracy

The evaluation compares predictions with held-out historical observations. Forecasts are estimates and may differ from actual future revenue.

### 7. Business Insights
The dashboard summarizes revenue trends, the leading sales channel, the top product, and predicted future revenue.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application and dashboard |
| Pandas | Data processing and cleaning |
| Plotly | Interactive charts |
| Prophet | Revenue forecasting |
| Scikit-learn | Machine learning utilities |
| SQLAlchemy | Database-related functionality |
| python-dotenv | Environment configuration |

## System Workflow

```text
Sales Dataset (CSV)
        ↓
Data Validation and Cleaning
        ↓
Sales Analysis
        ↓
Interactive BI Dashboard
        ↓
Prophet Forecasting Model
        ↓
Model Evaluation
        ↓
Business Insights and Future Revenue Forecast
```

## Required CSV Format

The uploaded CSV must contain these six columns:

| Column | Description |
|---|---|
| `date` | Date of the sale |
| `revenue` | Revenue generated |
| `order_id` | Unique order identifier |
| `customer_id` | Customer identifier |
| `sales_channel` | Channel through which the sale occurred |
| `product` | Product name |

Example:

```csv
date,revenue,order_id,customer_id,sales_channel,product
2025-01-01,50000,ORD001,CUST001,Online,Laptop
2025-02-01,65000,ORD002,CUST002,Store,Mobile
2025-03-01,70000,ORD003,CUST003,Online,Tablet
```

## Installation and Setup

### Prerequisites
- Python 3.11 or a compatible Python version
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Hmali2003/predictive-analytics-bi.git
cd predictive-analytics-bi
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## Model Evaluation

The system uses historical data to train the forecasting model and reserves the last three months for evaluation when sufficient monthly observations are available.

- **MAE:** Measures the average absolute difference between actual and predicted revenue.
- **MAPE:** Measures the average absolute prediction error as a percentage of actual revenue.

The displayed forecast accuracy is calculated as `100 − MAPE`, with negative values limited to zero. It is a derived indicator, not a guarantee of future prediction performance.

## Project Limitations

- The current input format requires all six specified CSV columns.
- Forecast quality depends on the quality, length, and patterns of the uploaded sales data.
- Future revenue predictions are estimates, not guaranteed results.
- The application currently focuses on revenue forecasting rather than forecasting every business metric.

## Future Enhancements

- Support Excel file uploads.
- Compare multiple forecasting models.
- Add more advanced business analytics.
- Introduce interactive what-if analysis.
- Generate downloadable business reports.

## Project Background and Attribution

This project was developed for MCA academic research in the domain of Predictive Analytics in Business Intelligence. It adapts and extends the open-source Market-IQ project with changes for CSV-based sales analysis, data validation, dashboard functionality, forecasting evaluation, and business insights.

Original project: https://github.com/AmirhosseinHonardoust/Market-IQ

The original project's applicable license and copyright notice are retained in the `LICENSE` file.

## Author

**Harshal Mali**  
Master of Computer Applications (MCA)  
Project: Predictive Analytics-Based Business Intelligence System for Sales Forecasting

GitHub: https://github.com/Hmali2003