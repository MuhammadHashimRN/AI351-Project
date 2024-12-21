import shap
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def explain_predictions(model, X_sample):
    """
    Explain predictions using SHAP GradientExplainer or KernelExplainer based on compatibility.

    Parameters:
        model (tf.keras.Model): Trained model.
        X_sample (np.array): Sample data for SHAP analysis.

    Returns:
        shap_values (np.array): Aggregated SHAP values for analysis.
        explanation_info (dict): Details about the SHAP explanation (e.g., aggregation method used).
    """
    explanation_info = {}
    if X_sample.shape[0] == 0:
        raise ValueError("X_sample is empty. Ensure valid input data.")

    try:
        # Check if GradientExplainer can be used
        explainer = shap.GradientExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample[:10])  # Compute SHAP values for a sample
        explanation_info['method'] = 'GradientExplainer'
    except Exception as e:
        st.warning("Falling back to KernelExplainer due to gradient computation issues.")
        explanation_info['fallback_reason'] = str(e)

        # Use KernelExplainer as a fallback
        def predict_function(data):
            return model.predict(data, verbose=0)

        # Select background for KernelExplainer
        background_size = min(100, X_sample.shape[0]) if X_sample.shape[0] > 0 else 1
        background = X_sample[np.random.choice(X_sample.shape[0], size=background_size, replace=False)]
        explainer = shap.KernelExplainer(predict_function, background)
        shap_values = explainer.shap_values(X_sample[:10], nsamples=100)
        explanation_info['method'] = 'KernelExplainer'

    # Aggregate SHAP values
    if isinstance(shap_values, list):  # For GradientExplainer, shap_values is a list
        shap_values = shap_values[0]  # Use the first output for regression tasks
    explanation_info['shap_values_shape'] = shap_values.shape

    # Validate SHAP value dimensions against input data
    if len(shap_values.shape) == 3:  # SHAP values include time steps
        shap_values_aggregated = np.mean(shap_values, axis=1)  # Aggregate across time steps
        explanation_info['aggregation'] = 'time_step_mean'
    else:
        shap_values_aggregated = shap_values  # No aggregation needed
        explanation_info['aggregation'] = 'none'

    return shap_values_aggregated, explanation_info

def display_additional_graphs(data):
    """
    Displays three graphs: Volume of Sales, Moving Averages, and Average Daily Return.

    Parameters:
        data (pd.DataFrame): Stock data containing columns like 'Volume', 'MA_7', 'MA_30', 'Daily_Return'.
    """

    # Moving Averages
    st.write("### Moving Averages (7-day and 30-day)")
    ma_fig = go.Figure()
    ma_fig.add_trace(go.Scatter(x=data["Date"], y=data["MA_7"], mode='lines', name='7-Day Moving Average', line=dict(color='blue')))
    ma_fig.add_trace(go.Scatter(x=data["Date"], y=data["MA_30"], mode='lines', name='30-Day Moving Average', line=dict(color='orange')))
    ma_fig.update_layout(
        title="Moving Averages Over Time",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        legend_title="Legend"
    )
    st.plotly_chart(ma_fig)
    
    # Average Daily Return
    st.write("### Average Daily Return")
    return_fig = go.Figure()
    return_fig.add_trace(go.Scatter(x=data["Date"], y=data["Daily_Return"], mode='lines', name='Daily Return', line=dict(color='green')))
    return_fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Zero Line", annotation_position="bottom left")
    return_fig.update_layout(
        title="Daily Return Over Time",
        xaxis_title="Date",
        yaxis_title="Return",
        template="plotly_white",
        legend_title="Legend"
    )
    st.plotly_chart(return_fig)