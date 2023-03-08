import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential

# TODO:
# How to deal with multiple zero values in time series data when using a RNN for example?
# How to handle both continuous and categorical data in a RNN?
# Define input data
data = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])

# Define target data
target = np.array([0.4, 0.5, 0.6, 0.7])

# Reshape the input data to be 3-dimensional
data = np.reshape(data, (data.shape[0], 1, data.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(1, 3)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit(data, target, epochs=1000, verbose=0)

# Make predictions on new data
new_data = np.array([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])
new_data = np.reshape(new_data, (new_data.shape[0], 1, new_data.shape[1]))
predictions = model.predict(new_data)

# Print the predictions
print(predictions)
