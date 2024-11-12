import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# Data Preparation
data = pd.read_csv('D:\\Colon-Cancer-Predicter\\data\\data4.csv')
X = data.drop(columns=['case_control', 'id'])
y = data['case_control']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Neural Network
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),  # Input layer with feature size
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Single output for binary classification
])

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the Model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Generate Risk Scores
risk_scores = model.predict(X_test_scaled)

# Save the Trained Model 
model.save('D:\\Colon-Cancer-Predicter\\Models\\trained_model.keras')
print("Model saved to: D:\\Colon-Cancer-Predicter\\Models\\trained_model.keras")

# Save the Model's Architecture (JSON Format)
with open('D:\\Colon-Cancer-Predicter\\Models\\model_architecture.json', 'w') as f:
    f.write(model.to_json())

# Save the Model's Weights 
model.save_weights('D:\\Colon-Cancer-Predicter\\Models\\model_weights.weights.h5')
print("Model weights saved to: D:\\Colon-Cancer-Predicter\\Models\\model_weights.weights.h5")

# Output Risk Scores
results = pd.DataFrame({'Risk Score': risk_scores.flatten(), 'Actual': y_test.values})
results.to_csv('D:\\Colon-Cancer-Predicter\\Models\\risk_scores.csv', index=False)
print("Risk scores saved to: D:\\Colon-Cancer-Predicter\\Models\\risk_scores.csv")