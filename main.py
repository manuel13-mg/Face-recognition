import streamlit as st
import add_faces
import test
import show_attendance

def main():
    st.title("Smart Attendance Register")

    menu = ["Add Face", "Take Attendance", "Show Attendance"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Add Face":
        add_faces.add_face()
    elif choice == "Take Attendance":
        test
        test.take_attendance()
    elif choice == "Show Attendance":
        show_attendance.show_attendance()

if __name__ == "__main__":
    main()