import streamlit as st
from streamlit_pages import display_home, display_stock_analysis, display_model_overview

def sidebar():
    """
    Displays the sidebar for navigation.
    """
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to:", ["Home", "Stock Analysis", "Model Overview", "Settings"])

def main():
    """
    Main function to handle navigation between pages.
    """
    page = sidebar()
    if page == "Home":
        display_home()
    elif page == "Stock Analysis":
        display_stock_analysis()
    elif page == "Model Overview":
        display_model_overview()
        
if __name__ == "__main__":
    main()
