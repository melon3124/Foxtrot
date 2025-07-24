import streamlit as st

def authenticate():
    if "authenticated" not in st.session_state:
        password = st.text_input("Enter password", type="password")
        if password == "your_secret":
            st.session_state["authenticated"] = True
        else:
            st.stop()

