import tensorflow as tf
from tensorflow.keras.models import Sequential, Model# type: ignore
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def build_lstm_attention(input_shape):
    """
    Builds an LSTM model with an attention mechanism.

    Parameters:
        input_shape (tuple): Shape of the input data.

    Returns:
        Model: Compiled LSTM model with attention.
    """
    # Input Layer
    inputs = Input(shape=input_shape)

    # LSTM Layers
    x = LSTM(128, return_sequences=True)(inputs)  # First LSTM layer
    x = LSTM(64, return_sequences=True)(x)        # Second LSTM layer (outputs sequences)

    # Attention Mechanism
    attention = Dense(1, activation='tanh')(x)    # Score function
    attention_weights = tf.nn.softmax(attention, axis=1)  # Apply softmax to calculate weights
    context_vector = tf.reduce_sum(x * attention_weights, axis=1)  # Compute context vector

    # Dense Layers
    dense_output = Dense(25, activation='relu')(context_vector)  # Fully connected layer
    outputs = Dense(1)(dense_output)  # Final output layer for regression

    # Build Model
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


def build_bilstm_model(input_shape):
    """
    Builds a BiLSTM model with reduced complexity and added regularization.

    Parameters:
        input_shape (tuple): Shape of the input data.

    Returns:
        Sequential: Compiled BiLSTM model.
    """
    model = Sequential([
        Bidirectional(LSTM(30, return_sequences=False), input_shape=input_shape),
        Dropout(0.4),  # Increased Dropout
        Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),  # L2 Regularization
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


def build_gru_model(input_shape):
    """
    Builds a GRU model.
    
    Parameters:
        input_shape (tuple): Shape of the input data.
    
    Returns:
        Sequential: Compiled GRU model.
    """
    model = Sequential([
        GRU(50, return_sequences=False, input_shape=input_shape),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

CONFIG = {
    "epochs": 30,
    "batch_size": 2,
    "time_steps": 60
}

def kfold_model_evaluation(X, y, model_fn, scaler, k=5):
    """
    Perform K-Fold Cross-Validation to evaluate model performance with Early Stopping.

    Parameters:
        X (np.array): Input features.
        y (np.array): Target values.
        model_fn (callable): Function to create a new model instance.
        scaler (object): Scaler used for preprocessing (e.g., MinMaxScaler).
        k (int): Number of folds for cross-validation.

    Returns:
        dict: Metrics (MSE, MAE, R²) aggregated across folds.
        list: Predictions for the final fold for visualization.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores, mae_scores, r2_scores = [], [], []

    # Early Stopping Callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True
    )

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Build and train the model
        model = model_fn((X.shape[1], X.shape[2]))
        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            callbacks=[early_stopping],  # Add Early Stopping
            verbose=0
        )

        # Predict and evaluate
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        mse_scores.append(mean_squared_error(y_test_rescaled, predictions))
        mae_scores.append(mean_absolute_error(y_test_rescaled, predictions))
        r2_scores.append(r2_score(y_test_rescaled, predictions))

        # Save the last fold predictions for visualization
        last_predictions = (predictions, y_test_rescaled)

    metrics = {
        "MSE": np.mean(mse_scores),
        "MAE": np.mean(mae_scores),
        "R²": np.mean(r2_scores),
    }
    return metrics, last_predictions