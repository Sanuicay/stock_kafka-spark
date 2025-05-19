import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Parameters
PARQUET_PATH = "./spark/data/data"
SEQUENCE_LENGTH = 20
EPOCHS = 30
BATCH_SIZE = 16
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(parquet_path, symbols=None):
    """Load data from parquet files with optional symbol filtering"""
    df = pd.read_parquet(parquet_path, filters=[('symbol', 'in', symbols)] if symbols else None)
    print(df.head())
    return df.sort_values(by=["symbol", "event_time"])

def add_features(df):
    """Add technical indicators and other features"""
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(5).std()
    df['moving_avg_5'] = df['close'].rolling(5).mean()
    df['moving_avg_10'] = df['close'].rolling(10).mean()
    df['price_diff'] = df['close'].diff()
    df.dropna(inplace=True)
    return df

def preprocess_data(df, symbol):
    """Preprocess data for a specific symbol"""
    df_symbol = df[df["symbol"] == symbol].copy()
    df_symbol.set_index("event_time", inplace=True)
    df_symbol = add_features(df_symbol)
    
    # Scale features
    feature_cols = ['close', 'returns', 'volatility', 'moving_avg_5', 'moving_avg_10', 'price_diff']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_symbol[feature_cols])
    
    # Build sequences
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(df_symbol)):
        X.append(scaled_features[i-SEQUENCE_LENGTH:i])
        y.append(scaled_features[i, 0])  # Predicting 'close' price
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler, df_symbol.index[SEQUENCE_LENGTH:]

def build_model_for_AAPL():
    """Custom LSTM model for AAPL with more complexity"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 6),
             activation='tanh', kernel_regularizer=l2(1e-3)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False, activation='relu', kernel_regularizer=l2(1e-3)),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model

def build_model_for_MSFT():
    """Original model that works well for MSFT"""
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 6),
             activation='tanh', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(16, return_sequences=False, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def build_model_for_GOOGL():
    """Custom LSTM model for GOOGL with different architecture"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 6),
             activation='tanh', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=True, activation='sigmoid', kernel_regularizer=l2(1e-4)),
        LSTM(16, return_sequences=False, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.00075),
        loss='mse',
        metrics=['mae']
    )
    return model

def get_model_builder(symbol):
    """Return the appropriate model builder for each symbol"""
    if symbol == 'AAPL':
        return build_model_for_AAPL
    elif symbol == 'MSFT':
        return build_model_for_MSFT
    elif symbol == 'GOOGL':
        return build_model_for_GOOGL
    else:
        return build_model_for_MSFT  # default

def evaluate_model(y_true, y_pred, scaler):
    """Evaluate model performance with multiple metrics"""
    y_true_rescaled = scaler.inverse_transform(np.concatenate([
        y_true.reshape(-1, 1),
        np.zeros((len(y_true), 5))
    ], axis=1))[:, 0]
    
    y_pred_rescaled = scaler.inverse_transform(np.concatenate([
        y_pred.reshape(-1, 1),
        np.zeros((len(y_pred), 5))
    ], axis=1))[:, 0]
    
    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_true_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true_rescaled - y_pred_rescaled) / y_true_rescaled)) * 100
    
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    return mae, mse, rmse, mape

def plot_predictions(y_true, y_pred, dates, title="Prediction vs True"):
    """Plot predictions vs actual values with dates"""
    plt.figure(figsize=(15, 6))
    plt.plot(dates, y_true, label="True", linewidth=2)
    plt.plot(dates, y_pred, label="Predicted", linestyle='--')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def time_series_cv(X, y, symbol, n_splits=3, epochs=10, batch_size=16):
    """Time-series cross validation with symbol-specific models"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    model_builder = get_model_builder(symbol)
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = KerasRegressor(
            build_fn=model_builder,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
        )
        
        y_pred = model.predict(X_test)
        mae, mse, rmse, mape = evaluate_model(y_test, y_pred, scaler)
        results.append({'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape})
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load data for specific symbols
    symbols_to_process = ['AAPL','GOOGL','MSFT']
    df = load_data(PARQUET_PATH, symbols=symbols_to_process)
    
    for symbol in df["symbol"].unique():
        print(f"\n{'='*50}")
        print(f"Processing symbol: {symbol}")
        print(f"{'='*50}")
        
        # Preprocess data
        X, y, scaler, dates = preprocess_data(df, symbol)
        
        # Time-based train-test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_test = dates[split_idx:]
        
        # Cross-validation with symbol-specific model
        print("\nRunning cross-validation...")
        cv_results = time_series_cv(
            X_train, y_train, symbol,
            n_splits=3,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        print("\nCross-validation results:")
        print(cv_results.mean())
        
        # Final model training with symbol-specific architecture
        print("\nTraining final model...")
        model_builder = get_model_builder(symbol)
        model = model_builder()
        
        checkpoint = ModelCheckpoint(
            f"{MODEL_DIR}/{symbol}_best_model.h5",
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                checkpoint
            ],
            verbose=1
        )
        
        # Evaluation
        print("\nEvaluating model...")
        y_pred = model.predict(X_test)
        mae, mse, rmse, mape = evaluate_model(y_test, y_pred, scaler)
        
        # Rescale for plotting
        y_test_rescaled = scaler.inverse_transform(np.concatenate([
            y_test.reshape(-1, 1),
            np.zeros((len(y_test), 5))
        ], axis=1))[:, 0]
        
        y_pred_rescaled = scaler.inverse_transform(np.concatenate([
            y_pred.reshape(-1, 1),
            np.zeros((len(y_pred), 5))
        ], axis=1))[:, 0]
        
        # Plot results
        plot_predictions(
            y_test_rescaled,
            y_pred_rescaled,
            dates_test,
            f"{symbol} Price Prediction (Test Set)"
        )
        
        # Save model and scaler
        model.save(f"{MODEL_DIR}/{symbol}_final_model.h5")
        np.save(f"{MODEL_DIR}/{symbol}_scaler.npy", scaler.scale_)
        print(f"\nSaved model and scaler for {symbol} in {MODEL_DIR}")