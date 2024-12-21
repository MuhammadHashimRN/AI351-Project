import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import logging
import numpy as np
import streamlit as st
import json
import os
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@st.cache_data
def download_data(stock_symbol, start_date, end_date):
    """
    Downloads stock data from Yahoo Finance and computes additional metrics.

    Parameters:
        stock_symbol (str): Stock ticker symbol.
        start_date (str): Start date for data.
        end_date (str): End date for data.

    Returns:
        pd.DataFrame: Data containing 'Close' prices, 'Volume', Moving Averages, and Daily Returns.
    """
    # Download stock data
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    # Handle MultiIndex columns (e.g., Adjusted Close in some cases)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]

    # Check for required columns
    close_col = f"Close {stock_symbol}" if f"Close {stock_symbol}" in data.columns else "Close"
    if close_col not in data.columns:
        raise KeyError(f"'Close' column not found for {stock_symbol}. Check if the symbol is correct.")

    # Prepare DataFrame with 'Close' and 'Volume'
    if 'Volume' in data.columns:
        data = data[[close_col, 'Volume']].rename(columns={close_col: 'Close'}).reset_index()
    else:
        # st.warning(f"'Volume' column not found for {stock_symbol}. Only 'Close' prices will be used.")
        data = data[[close_col]].rename(columns={close_col: 'Close'}).reset_index()

    # Add Moving Averages (e.g., 7-day, 30-day)
    data['MA_7'] = data['Close'].rolling(window=7).mean()  # 7-day moving average
    data['MA_30'] = data['Close'].rolling(window=30).mean()  # 30-day moving average

    # Compute Daily Returns
    data['Daily_Return'] = data['Close'].pct_change()  # Percentage change from previous day

    # Drop rows with NaN values (from moving averages or pct_change)
    data.dropna(inplace=True)

    return data



@st.cache_data
def preprocess_data(data):
    """
    Preprocesses the stock data by scaling the 'Close' prices.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing stock prices.
    
    Returns:
        np.array: Scaled data.
        MinMaxScaler: Fitted scaler.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    return scaled_data, scaler

def create_sequences(data, time_steps):
    """
    Creates sequences of data for time-series model input.
    
    Parameters:
        data (np.array): Scaled data array.
        time_steps (int): Number of time steps in each sequence.
    
    Returns:
        tuple: Arrays of sequences (X) and corresponding target values (y).
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)


def save_predictions_to_csv(predictions, actuals, errors):
    """
    Save predictions, actual values, and model errors to a CSV file.
    
    Parameters:
        predictions (dict): Dictionary where keys are model names and values are predicted values.
        actuals (np.array): Actual values from the test set.
        errors (dict): Dictionary where keys are model names and values are MSE errors.

    Returns:
        bytes: CSV file content as in-memory bytes.
    """
    results = []

    for model_name, preds in predictions.items():
        for i in range(len(preds)):
            results.append({
                "Model": model_name,
                "Actual": actuals[i][0],
                "Predicted": preds[i][0],
                "MSE": errors[model_name]
            })

    results_df = pd.DataFrame(results)

    # Save the results to a CSV in memory
    output = io.BytesIO()
    results_df.to_csv(output, index=False)
    output.seek(0)  # Reset the pointer to the start of the file
    return output

def save_feedback_to_file(feedback_data, file_path="feedback.json"):
    """
    Save user feedback to a JSON file.

    Parameters:
        feedback_data (dict): Dictionary containing feedback data.
        file_path (str): Path to the JSON file where feedback will be saved.
    """
    if os.path.exists(file_path):
        # Load existing feedback data
        with open(file_path, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    # Append new feedback to the existing data
    existing_data.append(feedback_data)

    # Save back to the file
    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=4)

    return file_path

def collect_user_feedback():
    """
    Collect user feedback on prediction accuracy and investment advice relevance.
    
    Uses a Streamlit form to ensure smooth interaction without page reruns.
    """
    st.write("### User Feedback")

    with st.form("feedback_form"):
        # Collect ratings
        accuracy_rating = st.slider(
            "Rate the accuracy of predictions (1: Poor, 5: Excellent):",
            1, 5, value=3
        )
        relevance_rating = st.slider(
            "Rate the relevance of investment advice (1: Poor, 5: Excellent):",
            1, 5, value=3
        )
        
        # Collect additional feedback
        additional_feedback = st.text_area("Any additional comments or suggestions?")

        # Submit button for the form
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            # Compile feedback into a dictionary
            feedback_data = {
                "Accuracy Rating": accuracy_rating,
                "Relevance Rating": relevance_rating,
                "Comments": additional_feedback,
            }

            # Save feedback to JSON file
            file_path = save_feedback_to_file(feedback_data)
            
            # Display success message
            st.success("Thank you for your feedback! Your feedback has been saved.")

            # Display submitted feedback
            st.write("### Submitted Feedback")
            st.json(feedback_data)
            st.write(f"Feedback saved to `{file_path}`.")

def validate_inputs(start_date, end_date):
    """
    Validate user inputs for date ranges and stock symbols.
    """
    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        return False
    return True

def fetch_symbols_from_alpha_vantage(api_key, keywords):
    """
    Fetch stock symbols dynamically using Alpha Vantage.
    
    Parameters:
        api_key (str): Your Alpha Vantage API key.
        keywords (str): Search keywords (e.g., company name or symbol).

    Returns:
        list: A list of matching stock symbols and company names.
    """
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={keywords}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "bestMatches" in data:
            symbols = [
                {
                    "symbol": match["1. symbol"],
                    "name": match["2. name"],
                    "region": match["4. region"],
                }
                for match in data["bestMatches"]
            ]
            return symbols
        else:
            return []
    else:
        st.error("Error fetching stock symbols. Please check your API key or try again later.")
        return []