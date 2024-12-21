import os
import random
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras import Model, Input
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Function to download stock data
@st.cache_data
def download_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]
    close_col = f"Close {stock_symbol}" if f"Close {stock_symbol}" in data.columns else "Close"
    if close_col in data.columns:
        if 'Volume' in data.columns:
            return data[[close_col, 'Volume']].rename(columns={close_col: 'Close'}).reset_index()
        else:
            st.warning(f"'Volume' column not found for {stock_symbol}. Only 'Close' prices will be used.")
            return data[[close_col]].rename(columns={close_col: 'Close'}).reset_index()
    else:
        raise KeyError(f"'Close' column not found for {stock_symbol}. Check if the symbol is correct.")


@st.cache_data
def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    return scaled_data, scaler


# Create sequences for LSTM input
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

# Build LSTM with attention
def build_lstm_attention(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(50, return_sequences=True)(inputs)
    attention = Dense(1, activation='tanh')(x)
    attention = tf.nn.softmax(attention, axis=1)
    context_vector = tf.reduce_sum(x * attention, axis=1)
    outputs = Dense(1)(context_vector)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Build GRU model
def build_gru_model(input_shape):
    model = Sequential([
        GRU(50, return_sequences=False, input_shape=input_shape),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Build BiLSTM model
def build_bilstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(50, return_sequences=False), input_shape=input_shape),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# SHAP explanation
def explain_predictions(model, X_sample):
    """
    Explain predictions using SHAP GradientExplainer or KernelExplainer based on compatibility.

    Parameters:
        model (tf.keras.Model): Trained model.
        X_sample (np.array): Sample data for SHAP analysis.
    """
    if X_sample.shape[0] == 0:
        raise ValueError("X_sample is empty. Ensure valid input data.")

    try:
        # GradientExplainer for models supporting TensorFlow gradients
        explainer = shap.GradientExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample[:10])
    except Exception as e:
        # Fallback to KernelExplainer in case of gradient computation issues
        st.warning("Falling back to KernelExplainer due to gradient computation issues.")
        def predict_function(data):
            return model.predict(data, verbose=0)
        background_size = min(100, X_sample.shape[0]) if X_sample.shape[0] > 0 else 1
        background = X_sample[np.random.choice(X_sample.shape[0], size=background_size, replace=False)]
        explainer = shap.KernelExplainer(predict_function, background)
        shap_values = explainer.shap_values(X_sample[:10], nsamples=100)

    return shap_values

# Investment Advice Based on SHAP
def investment_advice(shap_values, predictions, actuals):
    """
    Provide investment advice based on SHAP values and predictions.

    Parameters:
        shap_values (np.array): SHAP values for the test set.
        predictions (np.array): Model predictions (rescaled to original values).
        actuals (np.array): Actual stock prices (rescaled to original values).

    Returns:
        str: Investment advice ("Buy", "Hold", or "Sell").
    """
    # Aggregate SHAP values
    avg_shap = np.mean(np.abs(shap_values), axis=0)
    top_features = np.argsort(avg_shap)[-3:]  # Top 3 contributing features
    st.write(f"Top contributing features based on SHAP values: {top_features}")

    # Price increase/decrease analysis
    price_change = (predictions - actuals) / actuals * 100
    avg_price_change = np.mean(price_change)

    # Decision rules
    if avg_price_change > 5:
        return "Buy"
    elif avg_price_change < -5:
        return "Sell"
    else:
        return "Hold"


# Sidebar navigation
def sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to:", ["Home", "Stock Analysis", "Model Overview", "Settings"])

# Home page
def home():
    st.title("Stock Market Prediction Tool")
    st.write("Welcome to the Stock Market Prediction Tool. Navigate to different sections using the sidebar.")

# Stock analysis page
def stock_analysis():
    st.title("Stock Analysis")
    stock_symbol = st.selectbox("Select Stock Ticker", ["AAPL", "TSLA", "GOOGL", "MSFT"])
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

    if st.button("Fetch Data"):
        try:
            data = download_data(stock_symbol, start_date, end_date)
            st.write("## Data Overview")
            st.write(data.head())

            # Visualization
            st.line_chart(data[['Date', 'Close']].set_index('Date'))
            if 'Volume' in data.columns and st.checkbox("Show Volume Chart"):
                st.bar_chart(data[['Date', 'Volume']].set_index('Date'))

            # Preprocessing
            scaled_data, scaler = preprocess_data(data)
            time_steps = 60
            X, y = create_sequences(scaled_data, time_steps)

            # Train and compare models
            st.write("### Training and Comparing Models")
            input_shape = (X.shape[1], X.shape[2])
            models = {
                "LSTM with Attention": build_lstm_attention,
                "GRU": build_gru_model,
                "BiLSTM": build_bilstm_model
            }

            predictions_all, errors = {}, {}
            for name, model_fn in models.items():
                st.write(f"#### {name} Model")
                model = model_fn(input_shape)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

                # Predictions
                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)
                y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
                predictions_all[name] = predictions
                errors[name] = mean_squared_error(y_test_rescaled, predictions)

                # Plot predictions
                plt.figure(figsize=(10, 6))
                plt.plot(y_test_rescaled, label="Actual")
                plt.plot(predictions, label="Predicted")
                plt.title(f"{name} Predictions")
                plt.legend()
                st.pyplot(plt)

            # Select the best model
            best_model_name = min(errors, key=errors.get)
            st.write(f"### Best Model: {best_model_name}")

            # SHAP explanation and advice for the best model
            best_model_fn = models[best_model_name]
            best_model = best_model_fn(input_shape)
            best_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

            predictions = predictions_all[best_model_name]
            
            # SHAP Explanation
            shap_values = explain_predictions(best_model, X_test)

            # Aggregate SHAP values across time steps (axis=1 if time steps exist)
            if len(shap_values[0].shape) == 3:  # If SHAP values include time-step dimension
                shap_values_aggregated = np.mean(shap_values[0], axis=1)  # Aggregate across time steps
                st.write(f"SHAP values if time-step dimension shape: {shap_values_aggregated.shape}")
            else:
                shap_values_aggregated = shap_values[0]  # Use as-is if no time-step dimension
            
            # Ensure reshaping is valid
            X_test_flat = X_test[:60, :, 0]
            shap_values_aggregated = np.tile(shap_values_aggregated, (1, X_test_flat.shape[1]))
            
            # Check if shapes align
            st.write("### SHAP Explanation")
            st.write(f"X_test shape: {X_test.shape}")
            st.write(f"SHAP raw shape: {shap_values[0].shape}")
            st.write(f"SHAP aggregated shape: {shap_values_aggregated.shape}")
            st.write(f"SHAP values range: Min {np.min(shap_values_aggregated)}, Max {np.max(shap_values_aggregated)}")


            if shap_values_aggregated.shape[1] == X_test_flat.shape[1]:
                try:
                    # Generate the SHAP summary plot
                    feature_names = [f"Timestep {i}" for i in range(X_test_flat.shape[1])]
                    shap.summary_plot(shap_values_aggregated, X_test_flat, feature_names=feature_names, show=False)

                    st.pyplot(plt)

                    # Provide investment advice
                    advice = investment_advice(
                        shap_values_aggregated, predictions[:10].flatten(), y_test_rescaled[:10].flatten()
                    )
                    st.write(f"**Investment Advice**: {advice}")
                except Exception as e:
                    st.error(f"Unable to generate SHAP summary plot: {e}")
            else:
                st.error(
                    f"Shape mismatch: SHAP values {shap_values_aggregated.shape} and input data {X_test_flat.shape}."
                )
        except Exception as e:
            st.error(f"Error: {e}")

# Model overview page
def model_overview():
    st.title("Model Overview")
    st.write("This application uses an LSTM model with an attention mechanism, GRU, and BiLSTM.")
    st.write("The attention mechanism helps the model focus on important time steps during predictions.")

# Settings page
def settings():
    st.title("Settings")
    theme = st.selectbox("Select Theme", ["Light", "Dark"])
    st.write(f"Theme selected: {theme}")
    api_key = st.text_input("Enter API Key for Real-Time Data")
    if st.button("Save Settings"):
        st.success("Settings saved.")

# Main app
def main():
    page = sidebar()
    if page == "Home":
        home()
    elif page == "Stock Analysis":
        stock_analysis()
    elif page == "Model Overview":
        model_overview()
    elif page == "Settings":
        settings()

if __name__ == "__main__":
    main()