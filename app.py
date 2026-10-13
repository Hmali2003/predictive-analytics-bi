import streamlit as st
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
from marketiq.forecast import prophet_forecast

load_dotenv()


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Predictive Analytics in Business Intelligence",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# SESSION STATE
# ==========================================

if "show_home" not in st.session_state:
    st.session_state["show_home"] = True

if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0


# ==========================================
# SIDEBAR - UPLOAD DATASET
# ==========================================

st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Sales CSV",
    type=["csv"],
    key=f"csv_uploader_{st.session_state['upload_key']}"
)


# ==========================================
# HOME PAGE
# ==========================================

if uploaded_file is None or st.session_state["show_home"]:

    st.title(
        "📊 Predictive Analytics in Business Intelligence"
    )

    st.subheader(
        "Sales Forecasting and Business Intelligence System"
    )

    st.write(
        "This system analyzes historical sales data and predicts "
        "future revenue using predictive analytics."
    )

    st.info(
        """
        ### 📌 How to Use This System

        1. Prepare your sales data in CSV format.
        2. Upload the CSV using the sidebar.
        3. The system will validate and process your data.
        4. If the data is valid, the dashboard will be displayed.
        5. If there is a data problem, instructions will be shown.

        ### 📋 Required CSV Columns

        - `date`
        - `revenue`
        - `order_id`
        - `customer_id`
        - `sales_channel`
        - `product`
        """
    )

    st.warning(
        "📤 Please upload a CSV file from the sidebar to view the dashboard."
    )

    # If user uploads a file from home page
    if uploaded_file is not None:
        st.session_state["show_home"] = False
        st.rerun()

    st.stop()


# ==========================================
# READ CSV
# ==========================================

try:

    uploaded_data = pd.read_csv(
        uploaded_file
    )

except Exception:

    st.error(
        "❌ Unable to read the uploaded CSV file."
    )

    st.info(
        "Please check your CSV file and upload it again."
    )

    st.stop()


# ==========================================
# REQUIRED COLUMN VALIDATION
# ==========================================

required_columns = [
    "date",
    "revenue",
    "order_id",
    "customer_id",
    "sales_channel",
    "product"
]


missing_columns = [
    column
    for column in required_columns
    if column not in uploaded_data.columns
]


if missing_columns:

    st.error(
        "❌ Dataset cannot be processed because required columns are missing."
    )

    st.write(
        "### Missing Columns"
    )

    for column in missing_columns:
        st.write(
            f"- `{column}`"
        )

    st.info(
        "Please check your CSV file and upload a dataset "
        "containing all required columns."
    )

    st.stop()


# ==========================================
# DATA CLEANING
# ==========================================

uploaded_data["date"] = pd.to_datetime(
    uploaded_data["date"],
    errors="coerce"
)

uploaded_data["revenue"] = pd.to_numeric(
    uploaded_data["revenue"],
    errors="coerce"
)


uploaded_data = uploaded_data.dropna(
    subset=[
        "date",
        "revenue"
    ]
)


# ==========================================
# CHECK VALID DATA
# ==========================================

if uploaded_data.empty:

    st.error(
        "❌ No valid sales data was found."
    )

    st.info(
        "Please check the `date` and `revenue` columns "
        "and upload the file again."
    )

    st.stop()


# ==========================================
# SUCCESS MESSAGE
# ==========================================

st.sidebar.success(
    "✅ CSV validated successfully!"
)

st.sidebar.write(
    "Records:",
    f"{len(uploaded_data):,}"
)


# ==========================================
# BACK TO HOME
# ==========================================

if st.button("⬅️ Back to Home"):

    st.session_state["show_home"] = True

    st.session_state["upload_key"] += 1

    st.rerun()


# ==========================================
# DASHBOARD TITLE
# ==========================================

st.title(
    "📊 Predictive Analytics in Business Intelligence"
)

st.caption(
    "Sales Analysis and Predictive Revenue Forecasting Dashboard"
)


# ==========================================
# DATE FILTER
# ==========================================

min_date = uploaded_data["date"].min().date()

max_date = uploaded_data["date"].max().date()


start = st.sidebar.date_input(
    "Start date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)


end = st.sidebar.date_input(
    "End date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)


# ==========================================
# APPLY DATE FILTER
# ==========================================

data = uploaded_data[
    (uploaded_data["date"].dt.date >= start)
    & (uploaded_data["date"].dt.date <= end)
].copy()


if data.empty:

    st.error(
        "❌ No sales data is available for the selected date range."
    )

    st.info(
        "Please select a different date range."
    )

    st.stop()


# ==========================================
# MONTHLY DATA
# ==========================================

monthly = (
    data
    .set_index("date")
    .resample("MS")
    .agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique")
    )
    .reset_index()
    .rename(
        columns={
            "date": "month"
        }
    )
)


# ==========================================
# AVERAGE ORDER VALUE
# ==========================================

monthly["aov"] = (
    monthly["revenue"]
    / monthly["orders"].replace(
        0,
        pd.NA
    )
)

monthly["aov"] = monthly["aov"].fillna(0)


# ==========================================
# OVERVIEW
# ==========================================

st.subheader("Overview")


total_revenue = data["revenue"].sum()

total_orders = data["order_id"].nunique()

total_customers = data["customer_id"].nunique()

aov = (
    total_revenue / total_orders
    if total_orders > 0
    else 0
)


# ==========================================
# MONTH-OVER-MONTH REVENUE
# ==========================================

monthly_sorted = monthly.sort_values(
    "month"
)


if len(monthly_sorted) >= 2:

    previous_revenue = (
        monthly_sorted.iloc[-2]["revenue"]
    )

    current_revenue = (
        monthly_sorted.iloc[-1]["revenue"]
    )

    if previous_revenue != 0:

        mom_change = (
            (current_revenue - previous_revenue)
            / previous_revenue
        ) * 100

    else:

        mom_change = 0

else:

    mom_change = 0


# ==========================================
# KPI CARDS
# ==========================================

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Total Revenue",
        f"₹{total_revenue:,.0f}"
    )


with col2:

    st.metric(
        "Total Orders",
        f"{total_orders:,}"
    )


with col3:

    st.metric(
        "Unique Customers",
        f"{total_customers:,}"
    )


with col4:

    st.metric(
        "Average Order Value",
        f"₹{aov:,.0f}"
    )


with col5:

    st.metric(
        "MoM Revenue",
        f"{mom_change:+.1f}%"
    )


# ==========================================
# MONTHLY REVENUE
# ==========================================

st.subheader(
    "Monthly Revenue Trend"
)


revenue_chart = px.line(
    monthly,
    x="month",
    y="revenue",
    markers=True,
    labels={
        "month": "Month",
        "revenue": "Revenue"
    }
)


revenue_chart.update_layout(
    yaxis_tickprefix="₹",
    hovermode="x unified"
)


st.plotly_chart(
    revenue_chart,
    use_container_width=True
)


# ==========================================
# SALES CHANNEL + TOP PRODUCTS
# ==========================================

col1, col2 = st.columns(2)


# ==========================================
# SALES CHANNEL
# ==========================================

with col1:

    st.subheader(
        "Sales by Channel"
    )


    channel_data = (
        data
        .groupby(
            "sales_channel",
            as_index=False
        )["revenue"]
        .sum()
        .sort_values(
            "revenue",
            ascending=False
        )
    )


    channel_chart = px.pie(
        channel_data,
        names="sales_channel",
        values="revenue",
        hole=0.5
    )


    channel_chart.update_layout(
        legend_title="Sales Channel"
    )


    st.plotly_chart(
        channel_chart,
        use_container_width=True
    )


# ==========================================
# TOP PRODUCTS
# ==========================================

with col2:

    st.subheader(
        "Top Products by Revenue"
    )


    product_data = (
        data
        .groupby(
            "product",
            as_index=False
        )["revenue"]
        .sum()
        .sort_values(
            "revenue",
            ascending=False
        )
        .head(8)
    )


    product_chart = px.bar(
        product_data,
        x="revenue",
        y="product",
        orientation="h",
        labels={
            "revenue": "Revenue",
            "product": "Product"
        }
    )


    product_chart.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        },
        xaxis_tickprefix="₹"
    )


    st.plotly_chart(
        product_chart,
        use_container_width=True
    )


# ==========================================
# FORECAST
# ==========================================

st.header(
    "🔮 Forecasts"
)


fc, mae, accuracy = prophet_forecast(
    monthly[
        [
            "month",
            "revenue"
        ]
    ]
)


if isinstance(mae, str):

    st.warning(
        f"Forecast unavailable: {mae}"
    )

else:

    col1, col2 = st.columns(2)


    with col1:

        st.success(
            f"Model Evaluation — MAE: ₹{mae:,.0f}"
        )


    with col2:

        st.success(
            f"Forecast Accuracy: {accuracy:.1f}%"
        )


    # ======================================
    # FORECAST TABLE
    # ======================================

    st.subheader(
        "Future Revenue Forecast"
    )


    forecast_table = fc.copy()


    forecast_table = forecast_table.rename(
        columns={
            "ds": "Forecast Date",
            "yhat": "Predicted Revenue",
            "yhat_lower": "Lower Estimate",
            "yhat_upper": "Upper Estimate"
        }
    )


    forecast_table["Forecast Date"] = (
        pd.to_datetime(
            forecast_table["Forecast Date"]
        ).dt.strftime(
            "%d %b %Y"
        )
    )


    forecast_table["Predicted Revenue"] = (
        forecast_table["Predicted Revenue"]
        .round(0)
        .astype(int)
    )


    forecast_table["Lower Estimate"] = (
        forecast_table["Lower Estimate"]
        .round(0)
        .astype(int)
    )


    forecast_table["Upper Estimate"] = (
        forecast_table["Upper Estimate"]
        .round(0)
        .astype(int)
    )


    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True
    )


    # ======================================
    # HISTORICAL + FUTURE FORECAST GRAPH
    # ======================================

    history_chart = monthly[
        [
            "month",
            "revenue"
        ]
    ].copy()


    history_chart = history_chart.rename(
        columns={
            "month": "Date",
            "revenue": "Revenue"
        }
    )


    future_chart = fc[
        [
            "ds",
            "yhat"
        ]
    ].copy()


    future_chart = future_chart.rename(
        columns={
            "ds": "Date",
            "yhat": "Revenue"
        }
    )


    history_chart["Type"] = (
        "Historical Revenue"
    )

    future_chart["Type"] = (
        "Predicted Revenue"
    )


    combined_chart = pd.concat(
        [
            history_chart,
            future_chart
        ],
        ignore_index=True
    )


    forecast_graph = px.line(
        combined_chart,
        x="Date",
        y="Revenue",
        color="Type",
        markers=True,
        labels={
            "Date": "Date",
            "Revenue": "Revenue",
            "Type": "Data Type"
        }
    )


    forecast_graph.update_layout(
        title="Historical Revenue vs Future Forecast",
        yaxis_tickprefix="₹",
        hovermode="x unified"
    )


    st.plotly_chart(
        forecast_graph,
        use_container_width=True
    )


# ==========================================
# BUSINESS INSIGHTS
# ==========================================

st.header(
    "💡 Business Insights"
)


st.write(
    "The dashboard provides business insights from "
    "historical sales data and predictive forecasting."
)


# ==========================================
# REVENUE TREND
# ==========================================

first_revenue = (
    monthly.iloc[0]["revenue"]
)

last_revenue = (
    monthly.iloc[-1]["revenue"]
)


if last_revenue > first_revenue:

    revenue_trend = "increasing"

elif last_revenue < first_revenue:

    revenue_trend = "decreasing"

else:

    revenue_trend = "stable"


st.info(
    f"📈 **Revenue Trend:** Historical revenue is generally "
    f"{revenue_trend} over the selected period."
)


# ==========================================
# TOP CHANNEL
# ==========================================

channel_summary = (
    data
    .groupby("sales_channel")["revenue"]
    .sum()
    .sort_values(
        ascending=False
    )
)


top_channel = (
    channel_summary.index[0]
)


st.info(
    f"🛒 **Top Sales Channel:** "
    f"{top_channel} generated the highest revenue."
)


# ==========================================
# TOP PRODUCT
# ==========================================

product_summary = (
    data
    .groupby("product")["revenue"]
    .sum()
    .sort_values(
        ascending=False
    )
)


top_product = (
    product_summary.index[0]
)


st.info(
    f"🏆 **Top Product:** "
    f"{top_product} generated the highest revenue."
)


# ==========================================
# FORECAST INSIGHT
# ==========================================

if fc is not None and not fc.empty:

    forecast_revenue = (
        fc["yhat"].sum()
    )


    st.info(
        f"🔮 **Forecast Insight:** The model predicts approximately "
        f"₹{forecast_revenue:,.0f} revenue over the next "
        f"{len(fc)} months."
    )