#show_attendance.py
import streamlit as st
import pandas as pd
from datetime import datetime
import time

def show_attendance():
    """
    Displays the attendance data from a CSV file.
    """
    st.header("Show Attendance")

    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

    try:
        df = pd.read_csv("attendance/Attendance_" + date + ".csv")
        st.dataframe(df.style.highlight_max(axis=0))
    except FileNotFoundError:
        st.error("Attendance file not found for today.")
    except Exception as e:
        st.error(f"Error loading attendance data: {e}")