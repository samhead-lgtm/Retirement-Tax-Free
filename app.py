import streamlit as st

st.set_page_config(page_title="RTF Financial Planning", layout="wide")

st.title("RTF Financial Planning Suite")
st.write("Select your planning mode:")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Pre-Retirement")
    st.write("For working clients planning for retirement.")
    if st.button("Open Pre-Retirement Planner", type="primary"):
        st.switch_page("pages/1_Pre_Retirement_Planner.py")
with col2:
    st.subheader("Already Retired")
    st.write("For retired clients: tax analysis, income needs, Roth conversions.")
    if st.button("Open Retired Tax Analysis", type="primary"):
        st.switch_page("pages/2_Retired_Tax_Analysis.py")
