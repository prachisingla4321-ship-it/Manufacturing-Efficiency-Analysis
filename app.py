import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Manufacturing Efficiency Analysis",
    page_icon="🏭",
    layout="wide"
)

@st.cache_data
def load_data():
    # Use the CSV already stored in the GitHub repository.
    return pd.read_csv("Manufacturing Efficiency.csv")

df = load_data()

st.title("🏭 AI-Based Manufacturing Efficiency Analysis")
st.caption("Internship project dashboard | Excel + Power BI analysis with a lightweight Streamlit web view")

# Sidebar filters
st.sidebar.header("Filters")

machines = sorted(df["Machine_ID"].dropna().unique().tolist())
selected_machines = st.sidebar.multiselect(
    "Machine ID",
    machines,
    default=[]
)

modes = sorted(df["Operation_Mode"].dropna().unique().tolist())
selected_modes = st.sidebar.multiselect(
    "Operation Mode",
    modes,
    default=[]
)

statuses = sorted(df["Efficiency_Status"].dropna().unique().tolist())
selected_statuses = st.sidebar.multiselect(
    "Efficiency Status",
    statuses,
    default=[]
)

filtered = df.copy()

if selected_machines:
    filtered = filtered[filtered["Machine_ID"].isin(selected_machines)]
if selected_modes:
    filtered = filtered[filtered["Operation_Mode"].isin(selected_modes)]
if selected_statuses:
    filtered = filtered[filtered["Efficiency_Status"].isin(selected_statuses)]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(filtered):,}")
c2.metric("Avg Production Speed", f"{filtered['Production_Speed_units_per_hr'].mean():.2f}")
c3.metric("Avg Error Rate", f"{filtered['Error_Rate_%'].mean():.2f}%")
c4.metric("Avg Network Latency", f"{filtered['Network_Latency_ms'].mean():.2f} ms")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Efficiency Status Distribution")
    status_counts = filtered["Efficiency_Status"].value_counts()
    st.bar_chart(status_counts)

with right:
    st.subheader("Efficiency by Operation Mode")
    mode_status = pd.crosstab(
        filtered["Operation_Mode"],
        filtered["Efficiency_Status"]
    )
    st.bar_chart(mode_status)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Average Production Speed by Efficiency")
    speed = (
        filtered.groupby("Efficiency_Status")["Production_Speed_units_per_hr"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(speed)

with right:
    st.subheader("Average Error Rate by Efficiency")
    error = (
        filtered.groupby("Efficiency_Status")["Error_Rate_%"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(error)

st.divider()

st.subheader("Operational Metrics by Efficiency")
summary = (
    filtered.groupby("Efficiency_Status")
    .agg(
        Avg_Temperature_C=("Temperature_C", "mean"),
        Avg_Vibration_Hz=("Vibration_Hz", "mean"),
        Avg_Power_kW=("Power_Consumption_kW", "mean"),
        Avg_Production_Speed=("Production_Speed_units_per_hr", "mean"),
        Avg_Error_Rate=("Error_Rate_%", "mean"),
        Avg_Network_Latency=("Network_Latency_ms", "mean"),
        Avg_Packet_Loss=("Packet_Loss_%", "mean"),
    )
    .round(2)
)

st.dataframe(summary, use_container_width=True)

st.subheader("Project Summary")
st.write(
    "This web view presents the same manufacturing-efficiency analysis used in the "
    "internship project. The original analysis was prepared using Excel and Power BI. "
    "Efficiency_Status is an existing field in the dataset; this web app does not "
    "claim to train a machine-learning model."
)

st.caption("Data source: Manufacturing Efficiency.csv | Internship analytics project")
