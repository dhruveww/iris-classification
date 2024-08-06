import pandas as pd
import numpy as np
import pickle

# Load the model and scaler
with open('C:\\Users\\SURESH PATEL\\Downloads\\iris classification\\models\\iris_model.pkl', 'rb') as f:
    scaler, model = pickle.load(f)

# Define new sample measurements (replace these with actual values)
new_samples = pd.DataFrame({
    'sepal_length': [5.1, 7.0],
    'sepal_width': [3.5, 3.2],
    'petal_length': [1.4, 4.7],
    'petal_width': [0.2, 1.4]
})

# Scale the new samples
new_samples_scaled = scaler.transform(new_samples)

# Predict the species
predictions = model.predict(new_samples_scaled)
print(predictions)
