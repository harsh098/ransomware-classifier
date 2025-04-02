import streamlit as st
import sqlite3
import pandas as pd
import time
import numpy as np
from datetime import datetime, timezone, timedelta

db_path = "../../db.sqlite3"
timezone_offset = timedelta(hours=5, minutes=30)

def convert_timestamp(ts):
    """Safely convert unix timestamp to formatted datetime string with error handling"""
    try:
        if pd.isna(ts) or ts is None:
            return "Invalid timestamp"
        # Convert to integer if it's not already
        ts = int(float(ts))
        dt = datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc) + timezone_offset
        return dt.strftime('%d-%m-%Y %H:%M:%S.%f')[:-3] + ' UTC+05:30'
    except (ValueError, TypeError, OverflowError) as e:
        # Return a placeholder for invalid timestamps
        return f"Invalid ({ts})"

def get_incidents(minutes):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM incident WHERE TS >= strftime('%s', 'now', '-{minutes} minutes');"
    df = pd.read_sql(query, conn)
    conn.close()
    
    if not df.empty:
        # Handle potential NaN or None values and conversion errors
        df['Original_TS'] = df['TS']  # Keep original for debugging
        df['TS'] = df['TS'].astype(object).apply(convert_timestamp)
    return df

def get_events(minutes):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM events WHERE TS >= strftime('%s', 'now', '-{minutes} minutes');"
    df = pd.read_sql(query, conn)
    conn.close()
    
    if not df.empty:
        # Handle potential NaN or None values and conversion errors
        df['Original_TS'] = df['TS']  # Keep original for debugging
        df['TS'] = df['TS'].astype(object).apply(convert_timestamp)
    return df

# Streamlit UI
st.title("Incident and Event Dashboard")

st.sidebar.header("Filter Options")
minutes = st.sidebar.number_input("View records from last X minutes:", min_value=1, value=10, key="time_filter")

# Add auto-refresh functionality
auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=True)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", min_value=5, max_value=60, value=30)
    st.sidebar.text(f"Dashboard will refresh every {refresh_interval} seconds")
    
    # Setup for auto-refresh using st.rerun()
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh_time > refresh_interval:
        st.session_state.last_refresh_time = current_time
        st.rerun()
    
    # Display last refresh time
    st.sidebar.write("Last refreshed: " + datetime.now().strftime("%H:%M:%S"))

# Use session_state to track changes to minutes value
if 'previous_minutes' not in st.session_state:
    st.session_state.previous_minutes = minutes
elif st.session_state.previous_minutes != minutes:
    st.session_state.previous_minutes = minutes
    st.rerun()  # Rerun immediately when minutes value changes

# Get current timestamp for display
current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
st.write(f"Data as of: {current_time} (UTC+05:30)")
st.write(f"Showing data from the last {minutes} minutes")

# Debug option
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# Incidents section with expander
with st.expander("Reported Incidents", expanded=True):
    incidents_df = get_incidents(minutes)
    if incidents_df.empty:
        st.info("No incidents found in the selected time period.")
    else:
        # Conditionally show original timestamps for debugging
        if show_debug and 'Original_TS' in incidents_df.columns:
            st.dataframe(incidents_df, use_container_width=True)
        else:
            display_df = incidents_df.drop(columns=['Original_TS']) if 'Original_TS' in incidents_df.columns else incidents_df
            st.dataframe(display_df, use_container_width=True)
        st.write(f"Total incidents: {len(incidents_df)}")

# Events section with expander
with st.expander("Historical Events", expanded=True):
    events_df = get_events(minutes)
    if events_df.empty:
        st.info("No events found in the selected time period.")
    else:
        # Conditionally show original timestamps for debugging
        if show_debug and 'Original_TS' in events_df.columns:
            st.dataframe(events_df, use_container_width=True)
        else:
            display_df = events_df.drop(columns=['Original_TS']) if 'Original_TS' in events_df.columns else events_df
            st.dataframe(display_df, use_container_width=True)
        st.write(f"Total events: {len(events_df)}")