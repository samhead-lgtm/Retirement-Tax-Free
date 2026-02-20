import streamlit as st

st.set_page_config(page_title="RTF Financial Planning", layout="wide")

# Password gate — only enforced when APP_PASSWORD is set (Streamlit Cloud)
def check_password():
    try:
        app_pwd = st.secrets.get("APP_PASSWORD", "")
    except FileNotFoundError:
        app_pwd = ""
    if not app_pwd:
        st.session_state["authenticated"] = True
        return True
    if st.session_state.get("authenticated"):
        return True
    pwd = st.text_input("Enter password to access the app:", type="password")
    if pwd:
        if pwd == app_pwd:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if not check_password():
    st.stop()

st.switch_page("pages/3_Financial_Plan.py")
