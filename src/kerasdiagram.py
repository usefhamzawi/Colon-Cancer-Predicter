import tensorflow as tf
from model import focal_loss
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

# Load the model
model_path = "D:\\Colon-Cancer-Predicter\\Models\\trained_model.keras"
try:
    model = load_model(model_path, custom_objects={'focal_loss': focal_loss})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Load the test dataset
data = pd.read_csv('D:\\Colon-Cancer-Predicter\\data\\normal_data_added_noise.csv')

# Assuming 'case_control' is the label column
X_test = data.drop(columns=['case_control', 'id'])  # Features
y_test = data["case_control"].values  # Labels (0 or 1)

# Convert y_test to numpy array if needed
y_test = np.array(y_test)

# Get predicted probabilities
y_proba = model.predict(X_test)

# Convert probabilities to binary predictions (0 or 1) using a threshold of 0.5
y_pred = (y_proba > 0.5).astype(int)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the raw confusion matrix
print("Confusion Matrix:")
print(cm)

# Print classification report (includes precision, recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Control (0)", "Case (1)"], yticklabels=["Control (0)", "Case (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
