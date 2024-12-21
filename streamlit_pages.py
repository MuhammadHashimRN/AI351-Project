import streamlit as st
from utils import download_data, preprocess_data, create_sequences, save_predictions_to_csv
from utils import collect_user_feedback, validate_inputs, fetch_symbols_from_alpha_vantage
from models import build_lstm_attention, build_gru_model, build_bilstm_model, kfold_model_evaluation
from explainers import explain_predictions, display_additional_graphs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from plotly import graph_objects as go

API_KEY = "P0MQM42H5LQP5BHA"

# Configuration for model parameters
CONFIG = {
    "epochs": 10,
    "batch_size": 32,
    "time_steps": 60
}

def display_home():
    """
    Displays the home page for the Streamlit app.
    """
    st.title("Stock Market Prediction Tool")
    st.write("Welcome to the Stock Market Prediction Tool. Navigate using the sidebar.")

def display_stock_analysis():
    """
    Performs stock analysis including fetching dynamic stock symbols, 
    data fetching, preprocessing, model training, and analysis.
    """
    st.title("Stock Analysis")

    # Keyword input for dynamic symbol search
    st.write("### Search for Stock Symbols")
    keywords = st.text_input("Enter a keyword to search for stock symbols (e.g., 'Tesla'):", value="")

    stock_symbols = []

    if keywords:
        # Attempt to fetch stock symbols from Alpha Vantage
        try:
            stock_symbols = fetch_symbols_from_alpha_vantage(API_KEY, keywords)
            if not stock_symbols:
                st.warning("No matching symbols found with Alpha Vantage. Trying fallback method.")
        except Exception as e:
            st.warning(f"Alpha Vantage API error: {e}. Trying fallback method.")

        # Fallback to using yfinance if Alpha Vantage fails or returns no results
        if not stock_symbols:
            st.write("### Fallback to yfinance")
            try:
                # Use yfinance to find the closest match (manual suggestion or predefined list)
                stock_tickers = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]  # Replace with a broader or dynamic list
                stock_symbols = [{"symbol": ticker, "name": ticker, "region": "Fallback"} for ticker in stock_tickers]
                st.write("Using fallback ticker list:", [s["symbol"] for s in stock_symbols])
            except Exception as e:
                st.error(f"Fallback failed: {e}")
                return

    if stock_symbols:
        # Prepare dropdown options
        symbol_options = [f"{s['symbol']} - {s['name']} ({s['region']})" for s in stock_symbols]
        selected_symbol = st.selectbox("Select Stock Ticker", symbol_options)

        if selected_symbol:
            stock_symbol = selected_symbol.split(" - ")[0]  # Extract only the symbol
            start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
            end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

            # Validate inputs
            if not validate_inputs(start_date, end_date):
                return

            if st.button("Fetch Data"):
                try:
                    # Fetch and preprocess data
                    st.write("### Fetching Data")
                    data = download_data(stock_symbol, start_date, end_date)
                    if data.empty:
                        st.error("No data available for the selected stock and date range.")
                        return

                    st.write("## Data Overview")
                    st.write(data.head())

                    # Visualization
                    st.line_chart(data[['Date', 'Close']].set_index('Date'))
                    if 'Volume' in data.columns and st.checkbox("Show Volume Chart"):
                        st.bar_chart(data[['Date', 'Volume']].set_index('Date'))

                    # Additional Graphs Expander
                    with st.expander("Moving Averages, and Daily Returns", expanded=True):
                        display_additional_graphs(data)

                    # Preprocess data and create sequences
                    st.write("### Preprocessing Data")
                    scaled_data, scaler = preprocess_data(data)
                    time_steps = CONFIG["time_steps"]
                    X, y = create_sequences(scaled_data, time_steps)

                    # Progress bar for tracking
                    progress_bar = st.progress(0)

                    # Training and Comparing Models with K-Fold Cross-Validation
                    with st.expander("Training and Comparing Models", expanded=True):
                        st.write("### Model Training with K-Fold Cross-Validation")
                        input_shape = (X.shape[1], X.shape[2])
                        models = {
                            "LSTM with Attention": build_lstm_attention,
                            "GRU": build_gru_model,
                            "BiLSTM": build_bilstm_model
                        }

                        predictions_all, metrics = {}, {}
                        for idx, (name, model_fn) in enumerate(models.items()):
                            st.write(f"#### {name} Model")

                            # Evaluate model using K-Fold Cross-Validation
                            metrics[name], (predictions, y_test_rescaled) = kfold_model_evaluation(X, y, model_fn, scaler, k=5)
                            predictions_all[name] = predictions

                            # Plot predictions vs. actual values using Plotly
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(y=y_test_rescaled.flatten(), mode='lines', name='Actual'))
                            fig.add_trace(go.Scatter(y=predictions.flatten(), mode='lines', name='Predicted'))
                            fig.update_layout(
                                title=f"Actual vs Predicted Values for {name}",
                                xaxis_title="Time Steps",
                                yaxis_title="Price",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig)

                            # Update progress bar
                            progress_bar.progress(40 + int((idx + 1) * 20 / len(models)))

                        progress_bar.progress(100)

                        # Display comparison metrics
                        st.write("### Model Comparison Metrics")
                        metrics_df = pd.DataFrame(metrics).T
                        st.dataframe(metrics_df)

                        # Select the best model based on MSE
                        best_model_name = min(metrics, key=lambda x: metrics[x]["MSE"])
                        st.write(f"### Best Model: {best_model_name}")

                    # SHAP explanation and analysis
                    with st.expander("SHAP Explanation and Analysis", expanded=True):
                        st.write("### SHAP Explanation Details")
                        best_model_fn = models[best_model_name]
                        best_model = best_model_fn(input_shape)
                        best_model.fit(X, y, epochs=CONFIG["epochs"], batch_size=CONFIG["batch_size"], verbose=1)
                    
                        predictions = predictions_all[best_model_name]
                        shap_values, explanation_info = explain_predictions(best_model, X)
                        st.write(explanation_info)
                    
                        # Debugging output for SHAP values
                        st.write(f"Original SHAP values shape: {shap_values.shape}")
                        st.write(f"Feature matrix shape (X): {X.shape}")
                    
                        # Reshape SHAP values to align with features
                        if len(shap_values.shape) == 4:  # If 4D, remove the last two dimensions
                            shap_values = np.squeeze(shap_values, axis=(2, 3))
                        
                        st.write(f"Reshaped SHAP values shape: {shap_values.shape}")
                    
                        # Aggregate SHAP values for additional insights
                        aggregated_shap_values = np.mean(shap_values, axis=0)  # Aggregated across samples
                        st.write(f"Aggregated SHAP values shape: {aggregated_shap_values.shape}")
                    
                        # Extract feature names
                        if "Date" in data.columns:
                            feature_names = data["Date"].iloc[-X.shape[1]:].dt.strftime("%Y-%m-%d").tolist()
                        else:
                            feature_names = [f"Timestep {i}" for i in range(X.shape[1])]
                    
                        # Validate feature names and SHAP values alignment for summary plot
                        if shap_values.shape[1] == len(feature_names):
                            # Slice the feature matrix to match the SHAP samples
                            features_to_plot = X[:shap_values.shape[0], :, 0]  # Slice first 10 samples
                            st.write(f"Feature matrix shape for summary plot: {features_to_plot.shape}")
                    
                            # Generate SHAP summary plot
                            shap.summary_plot(shap_values, features=features_to_plot, feature_names=feature_names, show=False)
                            st.pyplot(plt)
                    
                            # Display aggregated SHAP values for reference
                            st.write("### Aggregated SHAP Values (Mean Absolute Contributions)")
                            shap_aggregate_df = pd.DataFrame(
                                {"Feature": feature_names, "Mean Absolute SHAP": np.mean(np.abs(aggregated_shap_values), axis=0)}
                            )
                            st.dataframe(shap_aggregate_df.sort_values(by="Mean Absolute SHAP", ascending=False))
                    
                        else:
                            st.error(
                                f"Shape mismatch between SHAP values ({shap_values.shape}) and feature names ({len(feature_names)})."
                            )
                    
                        # Analyze predictions for investment advice
                        avg_price_change = np.mean((predictions - y_test_rescaled) / y_test_rescaled * 100)
                        if avg_price_change > 5:
                            advice = "Buy"
                        elif avg_price_change < -5:
                            advice = "Sell"
                        else:
                            advice = "Hold"
                    
                        st.write(f"### Investment Advice: **{advice}**")
                        st.write(f"Predicted average price change: {avg_price_change:.2f}%")
                    


                    # Download Results
                    csv_data = save_predictions_to_csv(predictions_all, y_test_rescaled, metrics)
                    st.download_button(
                        label="Download Predictions and Metrics",
                        data=csv_data,
                        file_name="predictions_and_metrics.csv",
                        mime="text/csv",
                    )

                    # Feedback Section
                    with st.expander("Feedback", expanded=False):
                        st.write("We value your feedback! Please let us know how we're doing.")
                        collect_user_feedback()

                except Exception as e:
                    st.error(f"Error during analysis: {e}")


def display_model_overview():
    """
    Displays the Model Overview page.
    """
    st.title("Model Overview")
    st.write("### LSTM with Attention")
    st.write("""
    This model uses a Long Short-Term Memory (LSTM) layer with an attention mechanism to 
    focus on important time steps in the data for better predictions.
    """)
    st.write("### GRU")
    st.write("""
    The Gated Recurrent Unit (GRU) is a simplified version of LSTM that uses fewer parameters 
    while still achieving high accuracy for time-series data.
    """)
    st.write("### BiLSTM")
    st.write("""
    The Bidirectional LSTM processes data in both forward and backward directions, 
    improving the model's understanding of sequential dependencies.
    """)

def display_settings():
    """
    Displays the Settings page.
    """
    st.title("Settings")
    theme = st.selectbox("Select Theme", ["Light", "Dark"])
    st.write(f"Theme selected: {theme}")
    extended_training = st.checkbox("Enable Extended Training (More Epochs)")
    if st.button("Save Settings"):
        st.session_state['extended_training'] = extended_training
        st.success("Settings saved.")