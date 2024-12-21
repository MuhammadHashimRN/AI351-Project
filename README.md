# Stock Market Prediction Using Advanced Deep Learning Models

## Overview
In this project we will also explore advanced deep learning techniques in predicting stock prices using the models: **LSTM with attention**, **GRU**, **BiLSTM**. **Yahoo Finance** data are leveraged, together with robust preprocessing and evaluation techniques, such as **K-fold cross validation and SHAP explainability analysis**, to achieve accuracy and interpretability. The results prove the strength of attention mechanisms and deep learning architectures in predicting stock markets.

## Features
- I integrate LSTM with Attention, GRU, and BiLSTM models.
- Advanced data preprocessing such as normalization and performing other feature engineering or creating sequence features.
- **Robust model evaluation through **K-Fold Cross-Validation**.
- **SHAP**: (SHapley Additive exPlanations) explanation.
- Reliance on comparison of models and visualization of prediction result.

## Workflow
### Steps:
1. **Data Collection**:
   - Get the raw historical stock data from **Yahoo Finance**.
   - Get features such as closing prices, trading volume, and moving averages

2. **Data Preprocessing**:
   - For data normalization, we can use **MinMaxScaler**.
   - Interpolate or remove missings.
   To put it simply: generate additional indicators such as RSI and volatility measures.
   - For segment data, break them down to sequences (60 timestep).

3. **Model Training**:
   - Train models using **TensorFlow**.
   - Employ **early stopping** to prevent overfitting.

4. **Model Evaluation**:
   - Use **K-Fold Cross-Validation** to ensure reliability.
   - Evaluate using metrics like **MSE**, **MAE**, and **R²**.

5. **Explainability**:
   - Analyze predictions using **SHAP** for feature importance.

6. **Visualization & Reporting**:
   - Generate plots for predicted vs. actual values.
   - Visualize SHAP-based feature importance.

## Installation
### Prerequisites:
- Python 3.8+
- TensorFlow
- NumPy
- pandas
- matplotlib
- seaborn
- SHAP
- scikit-learn

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-market-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd stock-market-prediction
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the data collection and preprocessing script:
   ```bash
   python preprocess_data.py
   ```
2. Train models and evaluate results:
   ```bash
   python train_models.py
   ```
3. Generate SHAP analysis and visualizations:
   ```bash
   python shap_analysis.py
   ```

## Results
- **LSTM with Attention** achieved the best performance with:
  - MSE: 0.012
  - R²: 0.89
- **Key Features Identified** by SHAP:
  - Recent time steps (last 15-20 days).
  - Daily returns and moving averages.

## Future Work
- Integrate external indicators like news sentiment and social media trends.
- Develop adaptive models capable of real-time learning.
- Optimize computational efficiency for large-scale applications.

## References
1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
2. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder.
3. Bahdanau, D., et al. (2014). Neural Machine Translation by Jointly Learning to Align and Translate.
4. Vaswani, A., et al. (2017). Attention Is All You Need.
5. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions.
6. Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
7. Goodfellow, I., et al. (2016). Deep Learning.
8. Zhang, Y., et al. (2018). Stock Price Prediction Using Attention Mechanisms.
