import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Path to your OneDrive Excel file
FILE_PATH = r"C:\Users\savya\OneDrive\MSRB_Yield.xlsx"
SHEET_NAME = "Sheet1"

st.set_page_config(page_title="Live MSRB Yield Dashboard", layout="wide")

st.title("📈 Live MSRB Yield Dashboard")
st.markdown("This chart updates automatically as new ticks are written to Excel.")

# Refresh every 5 seconds
st_autorefresh = st.experimental_rerun  # legacy
refresh_rate = 5000  # ms

# ✅ Better: use st_autorefresh
count = st.experimental_data_editor if hasattr(st, "autorefresh") else None
st_autorefresh = getattr(st, "autorefresh", None)

if st_autorefresh:
    st_autorefresh(interval=refresh_rate, key="msrb_refresh")

# Load and plot data
if os.path.exists(FILE_PATH):
    try:
        df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

        if not df.empty:
            fig = px.line(
                df,
                x="Timestamp",
                y="MSRB_Yield",
                title="Live MSRB Yields",
                markers=True
            )
            fig.update_layout(xaxis_title="Time", yaxis_title="Yield (%)")
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"✅ Last updated: {pd.Timestamp.now()} | Rows: {len(df)}")
        else:
            st.warning("Excel file is empty.")
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
else:
    st.warning(f"⚠️ File not found at {FILE_PATH}")
