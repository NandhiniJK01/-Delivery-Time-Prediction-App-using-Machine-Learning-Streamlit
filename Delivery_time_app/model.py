import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Create dummy training data
data = {
    'Distance': [2, 5, 10, 15, 20],
    'Weight': [1, 2, 3, 4, 5],
    'Delivery_Time': [10, 20, 30, 45, 60]
}

df = pd.DataFrame(data)

X = df[['Distance', 'Weight']]
y = df['Delivery_Time']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
