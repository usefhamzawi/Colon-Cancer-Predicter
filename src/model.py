import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import shap
import matplotlib.pyplot as plt

# Step 1: Data Preparation
# Load your dataset
data = pd.read_csv('D:\\Colon-Cancer-Predicter\\data\\data4.csv')

# Separate features and target
X = data.drop(columns=['case_control', 'id'])  # Drop 'id' and target columns
y = data['case_control']  # Target variable (case: 1, control: 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Build the Neural Network
# Define the model architecture
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Step 4: Evaluate the Model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Step 5: Generate Risk Scores
risk_scores = model.predict(X_test_scaled)

# Step 6: SHAP Interpretation
explainer = shap.DeepExplainer(model, X_train_scaled[:100])  # Use a small subset of training data for SHAP explainer
shap_values = explainer.shap_values(X_test_scaled[:100])  # Explain a subset of test data

# Step 7: Visualizations
# 1. Summary Plot
shap.summary_plot(shap_values[0], X_test.iloc[:100], feature_names=X.columns)
plt.savefig('D:\\Colon-Cancer-Predicter\\Pictures\\shap_summary_plot.png')  # Save the summary plot
plt.close()  # Close the plot to free up memory

# 2. Dependence Plots for all features
for feature_name in X.columns:
    shap.dependence_plot(feature_name, shap_values[0], X_test.iloc[:100], feature_names=X.columns)
    plt.savefig(f'D:\\Colon-Cancer-Predicter\\Pictures\\shap_dependence_plot_{feature_name}.png')  # Save the dependence plot
    plt.close()

# 3. Force Plot for an individual prediction (first instance)
shap.initjs()  # Initialize JavaScript visualization
force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.iloc[0], feature_names=X.columns)
shap.save_html('D:\\Colon-Cancer-Predicter\\Pictures\\shap_force_plot.html', force_plot)  # Save the force plot as HTML
plt.close()

# 4. Force Plot for multiple predictions
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[:10], feature_names=X.columns)  # For the first 10 instances
plt.savefig('D:\\Colon-Cancer-Predicter\\Pictures\\shap_force_plot_multiple.png')  # Save the multiple force plot
plt.close()

# Step 8: Output Risk Scores with SHAP Explanations
results = pd.DataFrame({'Risk Score': risk_scores.flatten(), 'Actual': y_test.values[:100]})
results['Explanation'] = shap_values[0].tolist()  # Adding SHAP explanations as a list of values

# Display the results
print(results.head())

# Step 8: Output Risk Scores with SHAP Explanations
results = pd.DataFrame({'Risk Score': risk_scores.flatten(), 'Actual': y_test.values[:100]})
results['Explanation'] = shap_values[0].tolist()  # Adding SHAP explanations as a list of values

# Display the results
print(results.head())