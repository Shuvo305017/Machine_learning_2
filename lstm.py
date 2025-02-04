import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('AirPassengers.csv')
data = data[['#Passengers']]  # Use the correct column name

# Step 2: Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(x), np.array(y)

sequence_length = 50
X, y = create_sequences(scaled_data, sequence_length)
X = np.expand_dims(X, axis=-1)  # Add feature dimension for LSTM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Function to build the LSTM model
def build_model(units, learning_rate, optimizer, activation, dropout_rate):
    model = Sequential([
        Input(shape=(sequence_length, 1)),  # Add an Input layer
        LSTM(units, activation=activation, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=optimizer(learning_rate=learning_rate), loss='mse')                                                     
    return model

# Step 4: Hyperparameter experiments (including optimizer and dropout)
experiments = {
    "lstm_units": [50, 100],  # Test with 50 and 100 units
    "learning_rate": [0.001, 0.01],  # Test with 0.001 and 0.01 learning rates
    "activation": ['tanh', 'relu'],  # Test with 'tanh' and 'relu' activation functions
    "optimizer": [Adam, SGD],  # Test with Adam and SGD optimizers
    "dropout_rate": [0.2, 0.5]  # Test with 0.2 and 0.5 dropout rates
}

results = []

# Run the experiments with fewer combinations (2 values for each hyperparameter)
for units in experiments["lstm_units"]:
    for lr in experiments["learning_rate"]:
        for activation in experiments["activation"]:
            for optimizer in experiments["optimizer"]:
                for dropout_rate in experiments["dropout_rate"]:
                    # Build and train the model
                    model = build_model(units, lr, optimizer, activation, dropout_rate)
                    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                        epochs=10, batch_size=16, verbose=0)
                    # Evaluate and log results
                    train_loss = history.history['loss'][-1]
                    val_loss = history.history['val_loss'][-1]
                    results.append({
                        "Units": units,
                        "Learning Rate": lr,
                        "Activation": activation,
                        "Optimizer": optimizer.__name__,
                        "Dropout Rate": dropout_rate,
                        "Train Loss": train_loss,
                        "Validation Loss": val_loss
                    }) 

# Step 5: Display results
results_df = pd.DataFrame(results)
print(results_df)

# Plot changes in Validation Loss
plt.figure(figsize=(12, 6))
for optimizer in experiments["optimizer"]:
    for dropout_rate in experiments["dropout_rate"]:
        subset = results_df[(results_df["Optimizer"] == optimizer.__name__) & 
                            (results_df["Dropout Rate"] == dropout_rate)]
        plt.plot(range(len(subset)), subset["Validation Loss"], label=f"Optimizer={optimizer.__name__}, Dropout={dropout_rate}")
plt.xlabel("Experiment Index")
plt.ylabel("Validation Loss")
plt.title("Validation Loss for Different Optimizers and Dropout Rates")
plt.legend()
plt.show()


