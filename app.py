import streamlit as st
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
from marketiq.forecast import prophet_forecast, linear_regression_forecast

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

    st.markdown("""
        <style>
        .welcome-banner {
            padding: 36px 30px;
            border-radius: 18px;
            background: linear-gradient(120deg, #102a43, #176b87);
            color: white;
            text-align: center;
            margin-bottom: 24px;
        }
        .welcome-banner h1 {
            color: white;
            font-size: 36px;
            margin-bottom: 12px;
        }
        .welcome-banner p {
            color: #e5f4ff;
            font-size: 17px;
        }
        .feature-card {
            background: #f3f7fb;
            border: 1px solid #dce6ef;
            border-radius: 12px;
            padding: 18px;
            min-height: 145px;
        }
        .feature-card h3 {
            color: #174a6e;
            font-size: 19px;
        }
        .feature-card p {
            color: #334155;
            font-size: 15px;
        }
        </style>

        <div class="welcome-banner">
            <h1>📊 Predictive Analytics in Business Intelligence</h1>
            <p>Sales Analysis • Revenue Forecasting • Data-Driven Insights</p>
            <p>Turn historical sales data into meaningful business decisions.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Welcome to Your Business Intelligence System")

    st.write(
        "Explore historical sales performance, understand customer and "
        "product trends, and compare two models that forecast future revenue."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>📈 Sales Analytics</h3>
                <p>Explore revenue trends, orders, customers and sales channels.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>🔮 Revenue Forecasting</h3>
                <p>Compare Prophet and Linear Regression predictions.</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="feature-card">
                <h3>💡 Business Insights</h3>
                <p>Identify top products, leading channels and revenue trends.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🚀 Get Started")

    st.write(
        "Upload your sales CSV using the sidebar to begin. "
        "Once your file is uploaded, open the dashboard below."
    )

    st.caption(
        "Required columns: date, revenue, order_id, customer_id, "
        "sales_channel, product"
    )

    if uploaded_file is not None:
        if st.button("📊 Open Dashboard", type="primary",
                     use_container_width=True):
            st.session_state["show_home"] = False
            st.rerun()
    else:
     st.info("👈 Upload your sales CSV using the sidebar to continue.")

    st.stop()

    st.markdown("---")

    st.markdown(
        """
        <div style="text-align:center; color:#64748b; padding:12px 0;">
            Developed by <strong>Harshal Mali</strong> | © 2026
        </div>
        """,
        unsafe_allow_html=True
    )



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
# FORECAST MODEL COMPARISON
# ==========================================

st.header("🔮 Sales Forecasting Model Comparison")

forecast_data = monthly[["month", "revenue"]].copy()

# Run both models
prophet_fc, prophet_mae, prophet_accuracy = prophet_forecast(
    forecast_data.copy(),
    periods=6
)
fc = prophet_fc
linear_fc, linear_mae, linear_accuracy = linear_regression_forecast(
    forecast_data.copy(),
    periods=6
)

# Check whether forecasts were generated successfully
prophet_ok = (
    prophet_fc is not None
    and not prophet_fc.empty
    and not isinstance(prophet_mae, str)
)

linear_ok = (
    linear_fc is not None
    and not linear_fc.empty
    and not isinstance(linear_mae, str)
)

if not prophet_ok and not linear_ok:
    st.warning("Both forecasting models are unavailable.")
else:
    # ======================================
    # MODEL EVALUATION
    # ======================================

    st.subheader("Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Prophet")

        if prophet_ok:
            st.metric("MAE", f"₹{prophet_mae:,.2f}")
            st.metric(
                "MAPE-based score",
                f"{prophet_accuracy:.1f}%"
            )
        else:
            st.warning(f"Prophet unavailable: {prophet_mae}")

    with col2:
        st.markdown("### Linear Regression")

        if linear_ok:
            st.metric("MAE", f"₹{linear_mae:,.2f}")
            st.metric(
                "MAPE-based score",
                f"{linear_accuracy:.1f}%"
            )
        else:
            st.warning(f"Linear Regression unavailable: {linear_mae}")

    # ======================================
    # FORECAST COMPARISON TABLE
    # ======================================

    st.subheader("Next 6 Months: Forecast Comparison")

    comparison = None

    if prophet_ok and linear_ok:
        comparison = prophet_fc[["ds", "yhat"]].merge(
            linear_fc[["ds", "yhat"]],
            on="ds",
            suffixes=("_Prophet", "_LinearRegression")
        )

        comparison = comparison.rename(
            columns={
                "ds": "Forecast Date",
                "yhat_Prophet": "Prophet Prediction",
                "yhat_LinearRegression": "Linear Regression Prediction"
            }
        )

        comparison["Forecast Date"] = pd.to_datetime(
            comparison["Forecast Date"]
        ).dt.strftime("%d %b %Y")

        comparison["Prophet Prediction"] = (
            comparison["Prophet Prediction"].round(2)
        )

        comparison["Linear Regression Prediction"] = (
            comparison["Linear Regression Prediction"].round(2)
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        # ======================================
        # FUTURE FORECAST GRAPH
        # ======================================

        prophet_plot = prophet_fc[["ds", "yhat"]].copy()
        prophet_plot["Model"] = "Prophet"

        linear_plot = linear_fc[["ds", "yhat"]].copy()
        linear_plot["Model"] = "Linear Regression"

        prophet_plot = prophet_plot.rename(
            columns={"ds": "Date", "yhat": "Predicted Revenue"}
        )

        linear_plot = linear_plot.rename(
            columns={"ds": "Date", "yhat": "Predicted Revenue"}
        )

        future_comparison = pd.concat(
            [prophet_plot, linear_plot],
            ignore_index=True
        )

        forecast_graph = px.line(
            future_comparison,
            x="Date",
            y="Predicted Revenue",
            color="Model",
            markers=True,
            title="Prophet vs Linear Regression: Future Revenue"
        )

        forecast_graph.update_layout(
            yaxis_tickprefix="₹",
            hovermode="x unified"
        )

        st.plotly_chart(
            forecast_graph,
            use_container_width=True
        )

        # ======================================
        # SIMPLE RESULT SUMMARY
        # ======================================

        st.subheader("Comparison Summary")

        if prophet_mae < linear_mae:
            st.info(
                "Prophet has the lower MAE on the test period."
            )
        elif linear_mae < prophet_mae:
            st.info(
                "Linear Regression has the lower MAE on the test period."
            )
        else:
            st.info("Both models have the same MAE.")

        st.caption(
            "Lower MAE means lower average absolute prediction error. "
            "The MAPE-based score is a derived score, not a guarantee "
            "of future forecasting accuracy."
        )

    elif prophet_ok:
        st.info("Only Prophet produced a forecast.")
    elif linear_ok:
        st.info("Only Linear Regression produced a forecast.")


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

    

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:12px 0;">
        Developed by <strong>Harshal Mali</strong> | © 2026
    </div>
    """,
    unsafe_allow_html=True
)
