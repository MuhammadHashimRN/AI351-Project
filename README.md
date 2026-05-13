# XAI Stock Predictor — LSTM/GRU/BiLSTM with SHAP Explainability

> A deep-learning stock-price predictor that benchmarks **LSTM-with-Attention**, **GRU**, and **BiLSTM** on Yahoo Finance data, validates with K-fold CV, and explains every prediction with **SHAP** — wrapped in an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## Motivation

Most deep-learning stock predictors stop at "accuracy on a held-out test set." In production, that's not enough — you also need to know *why* the model made a given prediction (was it the recent volatility? RSI? volume spike?). This project goes one step further than a vanilla LSTM regressor by attaching **SHAP-based feature attribution** to every prediction, exposing the model's reasoning through an interactive Streamlit dashboard.

## What it does

1. **Pulls** historical OHLCV data for any user-supplied ticker from Yahoo Finance, plus symbol search via Alpha Vantage.
2. **Engineers features** — moving averages, RSI, volatility, volume-derived signals — and normalises with `MinMaxScaler`.
3. **Sequences** the data into 60-timestep windows.
4. **Trains three architectures** with early stopping and **K-fold cross-validation**:
   - LSTM with custom **attention** mechanism
   - GRU
   - Bidirectional LSTM
5. **Evaluates** each model with MSE, MAE, R², and compares side-by-side.
6. **Explains predictions** with SHAP (GradientExplainer with KernelExplainer fallback), surfaced as feature-importance bar charts and SHAP value heatmaps.
7. **Serves** all of the above through a Streamlit web UI with home / analysis / model-overview pages.

## Stack & module layout

```
AI351-Project/
├── main.py                         # Streamlit entry point + sidebar nav
├── streamlit_pages.py              # Home, Stock Analysis, Model Overview pages
├── models.py                       # LSTM-Attention / GRU / BiLSTM definitions + K-fold runner
├── explainers.py                   # SHAP integration (GradientExplainer + KernelExplainer)
├── utils.py                        # Data loading, preprocessing, sequence creation, IO
├── project.py                      # Standalone training script (non-Streamlit entry point)
├── AI-351 Project Proposal.docx    # Course proposal document
└── requirements.txt
```

### LSTM-with-Attention architecture

```
Input(60 timesteps, F features)
        │
        ▼
   LSTM(128, return_sequences=True)
        │
        ▼
   LSTM(64, return_sequences=True)
        │
        ▼
   ┌──────────────┐
   │  Attention   │  Dense(1, tanh) → softmax → weighted sum → context vector
   └──────┬───────┘
          ▼
      Dense(64) → Dropout
          │
          ▼
      Dense(1)  ── predicted price
```

## Getting started

### Prerequisites
- Python 3.10+
- An Alpha Vantage API key (free) — https://www.alphavantage.co/support/#api-key

### Install + run

```bash
git clone https://github.com/MuhammadHashimRN/AI351-Project.git
cd AI351-Project

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # then put your ALPHA_VANTAGE_API_KEY in .env

streamlit run main.py
```

The app opens in your browser. Use the sidebar to navigate:
- **Home** — overview + getting started
- **Stock Analysis** — pick a ticker, train a model, see predictions + SHAP explanations
- **Model Overview** — side-by-side comparison of the three architectures

## What's notable about this project (for engineering reviewers)

- **Three architectures benchmarked, not one** — direct apples-to-apples comparison with K-fold CV instead of single-split overfitting risk.
- **Custom attention layer** in pure Keras — not just stacking `tf.keras.layers.Attention`.
- **Real explainability** — SHAP integration that gracefully falls back from `GradientExplainer` to `KernelExplainer` when gradients are unstable.
- **End-to-end interactive UI** — not a notebook; a Streamlit app that anyone can clone and run.
- **No secrets in code** — Alpha Vantage API key loaded from `.env` (template at `.env.example`).

## Course context

Built as the final project for **AI-351 (Interpretable AI / XAI)** at GIK Institute. See `AI-351 Project Proposal.docx` for the original proposal and grading rubric.

## Author

**Muhammad Hashim** — BS Artificial Intelligence, GIK Institute (2026)
📧 muhammad808alvi@gmail.com · 🔗 [github.com/MuhammadHashimRN](https://github.com/MuhammadHashimRN)

## License

[MIT](LICENSE)
