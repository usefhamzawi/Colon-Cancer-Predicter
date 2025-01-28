import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras import regularizers
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

@tf.keras.utils.register_keras_serializable(package="Custom", name="focal_loss")
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal Loss function for binary classification to address class imbalance.
    
    Parameters:
    - y_true: Ground truth labels.
    - y_pred: Predicted probabilities.
    - gamma: Focusing parameter (higher values focus more on hard examples).
    - alpha: Class balancing factor for positive/negative class.
    
    Returns:
    - Scalar focal loss value.
    """
    # Ensure y_pred is clipped to avoid log(0) errors
    y_pred = tf.clip_by_value(y_pred, 1e-10, 1 - 1e-10)
    
    # Binary cross-entropy loss
    cross_entropy = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Probability of the true class
    pt = tf.exp(-cross_entropy)
    
    # Focal loss term
    focal_loss_value = alpha * tf.pow(1 - pt, gamma) * cross_entropy
    
    return tf.reduce_mean(focal_loss_value)


class ColoRecModel:
    def __init__(self, input_shape):
        """
        Initialize the Colon Recurrence Risk Prediction Model
        
        Parameters:
        - input_shape: Number of input features
        """
        self.input_shape = input_shape
        self.model = None

    def build_model(self, 
                    hidden_layers=[64, 32, 16],  
                    dropout_rates=[0.4, 0.4, 0.4],  
                    l2_lambda=0.02):
        """
        Build the neural network model with configurable architecture
        
        Parameters:
        - hidden_layers: List of neurons in each hidden layer
        - dropout_rates: Dropout rates for each hidden layer
        - l2_lambda: L2 regularization strength
        """

        self.model = Sequential()
        
        # Input layer
        self.model.add(Input(shape=(self.input_shape,)))
        
        # Hidden layers with BatchNormalization
        for neurons, dropout_rate in zip(hidden_layers, dropout_rates):
            # Dense layer
            self.model.add(Dense(neurons, 
                                activation=None,  # Use None here because BatchNormalization normalizes activations
                                kernel_regularizer=regularizers.l2(l2_lambda)))
            
            # BatchNormalization layer
            self.model.add(BatchNormalization())
            
            # ReLU activation after BatchNormalization
            self.model.add(Activation('relu'))
            
            # Dropout layer
            self.model.add(Dropout(dropout_rate))
        
        # Output layer
        self.model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model with Focal Loss
        self.model.compile(optimizer='adam', 
                           loss=focal_loss, 
                           metrics=['accuracy'])
        
        return self.model

    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Prepare and split the data, returning DataFrames with the same structure as the original X.
        
        Parameters:
        - X: Features (DataFrame)
        - y: Labels (Series or DataFrame)
        - test_size: Proportion of test data
        - random_state: Random seed for reproducibility
        
        Returns:
        - Scaled train and test datasets as DataFrames
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test


    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the model
        
        Parameters:
        - X_train: Scaled training features
        - y_train: Training labels
        - epochs: Number of training epochs
        - batch_size: Training batch size
        - validation_split: Proportion of training data for validation
        """
        history = self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split
        )
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        - X_test: Scaled test features
        - y_test: Test labels
        
        Returns:
        - Performance metrics
        """
        # Predict probabilities
        y_proba = self.model.predict(X_test).flatten()
        
        # Binary predictions
        y_pred = (y_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'AUC-ROC': roc_auc_score(y_test, y_proba),
            'F1 Score': f1_score(y_test, y_pred),
            'Confusion Matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def evaluate_calibration(self, y_true, y_proba, n_bins=10):
        """
        Evaluate prediction calibration using reliability diagram.
        Plots the reliability diagram and shows Expected Calibration Error (ECE)
        and Maximum Calibration Error (MCE) on the plot.
        """
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

        # Plot reliability diagram
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')  # Ideal line
        plt.plot(prob_pred, prob_true, 's-', label='Model')  # Model curve
        plt.xlabel('Mean predicted probability')
        plt.ylabel('True probability')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True)

        # Calculate calibration metrics: Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
        ece = np.mean(np.abs(prob_true - prob_pred))
        mce = np.max(np.abs(prob_true - prob_pred))

        plt.text(0.05, 0.95, 
                f'Expected Calibration Error: {ece:.3f}\n'
                f'Maximum Calibration Error: {mce:.3f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))

        # Display the plot
        plt.show()
    
    def save_model(self, base_path='D:\\Colon-Cancer-Predicter\\Models\\'):
        """
        Save the trained model, architecture, weights, and risk scores
        
        Parameters:
        - base_path: Base directory to save model files
        """
        # Create the directory if it doesn't exist
        import os
        os.makedirs(base_path, exist_ok=True)

        # Full paths for saving
        model_path = os.path.join(base_path, 'trained_model.keras')
        architecture_path = os.path.join(base_path, 'model_architecture.json')
        weights_path = os.path.join(base_path, 'model_weights.weights.h5')

        # Save the entire model
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")

        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(architecture_path, 'w') as f:
            f.write(model_json)
        print(f"Model architecture saved to: {architecture_path}")

        # Save model weights
        self.model.save_weights(weights_path)
        print(f"Model weights saved to: {weights_path}")

    def detailed_evaluation(self, X_test, y_test):
        """
        Perform detailed model evaluation and save risk scores
        
        Parameters:
        - X_test: Scaled test features
        - y_test: Test labels
        
        Returns:
        - Dictionary of evaluation metrics
        """
        # Predict probabilities and binary predictions
        y_proba = self.model.predict(X_test).flatten()
        y_pred = (y_proba > 0.5).astype(int)

        # Calculate metrics
        auc_roc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        # Print metrics
        print(f'Test AUC-ROC: {auc_roc:.4f}')
        print(f'Test F1 Score: {f1:.4f}')

        # Create and save risk scores DataFrame
        results = pd.DataFrame({
            'Risk Score': y_proba.flatten(), 
            'Actual': y_test.values
        })
        
        # Save risk scores
        risk_scores_path = 'D:\\Colon-Cancer-Predicter\\Models\\risk_scores.csv'
        results.to_csv(risk_scores_path, index=False)
        print(f"Risk scores saved to: {risk_scores_path}")

        # Return metrics for further analysis
        return {
            'auc_roc': auc_roc,
            'f1_score': f1,
            'risk_scores': results
        }

    def plot_shap_summary(self, X_train, X_test):
        """
        Plot SHAP beeswarm plots for feature importance using DeepExplainer.
        Generates two plots: one for the top 10 most impactful features and another for the remaining features.

        Parameters:
        - X_train: Training features (16 features)
        - X_test: Testing features (16 features)
        """

        X_train = pd.DataFrame(X_train, columns=X.columns)  # Convert X_train to DataFrame

        feature_names = X_train.columns

        # Convert X_train and X_test to NumPy arrays if they're DataFrames
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values  # Convert to NumPy array
            
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values  # Convert to NumPy array

        # Select a background dataset (X_train only, for model explanation)
        background = X_train[:1000]

        # Initialize DeepExplainer with the model and background dataset
        explainer = shap.DeepExplainer(self.model, background)

        # Compute SHAP values for the test data (X_test is the 16 features)
        shap_values = explainer.shap_values(X_test[:100])

        # Debugging: Print the shape of SHAP values and the data
        print("Shape of SHAP values:", [sv.shape for sv in shap_values])  # Check all the SHAP values

        # Since this is binary classification, take the SHAP values for the first class
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Get SHAP values for the first class (the predicted class)

        # Flatten the SHAP values to remove the extra dimension
        shap_values = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])

        # Debugging: Check the shape after reshaping
        print("Shape of SHAP values after reshaping:", shap_values.shape)
        print("Shape of X_test[:100]:", X_test[:100].shape)

        # Ensure that shap_values shape matches the input data shape
        assert shap_values.shape[0] == X_test[:100].shape[0], f"Shape mismatch: {shap_values.shape} vs {X_test[:100].shape}"

        # Convert SHAP values to Explanation object for better handling
        shap_explanation = shap.Explanation(
            values=shap_values,  # SHAP values for the predicted class
            data=X_test[:100],   # Corresponding input data (features only)
            feature_names=feature_names
        )

        # Sort the features by mean absolute SHAP value
        mean_abs_shap_values = shap_values.mean(axis=0)
        sorted_indices = mean_abs_shap_values.argsort()[::-1]  # Indices sorted by importance in descending order

        # Split into two groups: top 10 features and remaining features
        top_10_indices = sorted_indices[:10]
        remaining_indices = sorted_indices[10:]

        # Create Explanation objects for the two groups
        top_10_explanation = shap.Explanation(
            values=shap_values[:, top_10_indices],
            data=X_test[:100][:, top_10_indices],
            feature_names=[feature_names[i] for i in top_10_indices]
        )
        
        remaining_explanation = shap.Explanation(
            values=shap_values[:, remaining_indices],
            data=X_test[:100][:, remaining_indices],
            feature_names=[feature_names[i] for i in remaining_indices]
        )

        # Plot the SHAP beeswarm plot for the first 10 features
        print("Generating SHAP beeswarm plot for first 10 features...")
        shap.plots.beeswarm(top_10_explanation)

        # Plot the SHAP beeswarm plot for the remaining features
        print("Generating SHAP beeswarm plot for the remaining features...")
        shap.plots.beeswarm(remaining_explanation)

        # Print all the features on one plot
        print("Generating SHAP beeswarm plot for all the features...")
        shap.plots.beeswarm(shap_explanation)


    def plot_training_history(self, history):
        """
        Plot training history for loss and accuracy
        
        Parameters:
        - history: History object returned from model training
        """
        # Plot loss curve
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def print_model_summary(self, file_path="D:\\Colon-Cancer-Predicter\\Models\\model_summary.txt"):
        """
        Save a summary of the model's layers and weights to a text file.

        Parameters:
        - file_path: The file path where the summary will be saved (default is "model_summary.txt")
        """
        with open(file_path, "w", encoding="utf-8") as file:  # Explicitly specify UTF-8 encoding
            # Print the model summary (layers and parameters) to the text file
            self.model.summary(print_fn=lambda x: file.write(x + '\n'))

            # Print the weights for each layer to the text file
            for layer in self.model.layers:
                if hasattr(layer, 'get_weights'):
                    weights = layer.get_weights()
                    if weights:
                        file.write(f'Layer: {layer.name} | Weights shape: {weights[0].shape} | Biases shape: {weights[1].shape if len(weights) > 1 else None}\n')


    def mc_dropout_predict(self, X_test_scaled, n_samples=100):
        # Prepare an empty list to store predictions
        all_predictions = []
        
        # Ensure dropout is active by passing `training=True` during inference
        for _ in range(n_samples):
            # Forward pass through the model with dropout active
            predictions = self.model(X_test_scaled, training=True)  # Ensure dropout is applied
            all_predictions.append(predictions.numpy())  # Collect the predictions

        # Convert the list of predictions into a NumPy array
        all_predictions = np.array(all_predictions)

        # Calculate the mean prediction across the samples
        mean_predictions = np.mean(all_predictions, axis=0)

        # Calculate the uncertainty (standard deviation)
        uncertainty = np.std(all_predictions, axis=0)

        return mean_predictions, uncertainty


    def detailed_evaluation(self, X_test, y_test):
        """
        Perform detailed model evaluation and save risk scores
        
        Parameters:
        - X_test: Scaled test features
        - y_test: Test labels
        
        Returns:
        - Dictionary of evaluation metrics
        """
        # Predict probabilities and binary predictions
        y_proba = self.model.predict(X_test).flatten()
        y_pred = (y_proba > 0.5).astype(int)

        # Calculate metrics
        auc_roc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        # Print metrics
        print(f'Test AUC-ROC: {auc_roc:.4f}')
        print(f'Test F1 Score: {f1:.4f}')

        # Create and save risk scores DataFrame
        results = pd.DataFrame({
            'Risk Score': y_proba.flatten(), 
            'Actual': y_test.values
        })
        
        # Save risk scores
        risk_scores_path = 'D:\\Colon-Cancer-Predicter\\Models\\risk_scores.csv'
        results.to_csv(risk_scores_path, index=False)
        print(f"Risk scores saved to: {risk_scores_path}")

        # Return metrics for further analysis
        return {
            'auc_roc': auc_roc,
            'f1_score': f1,
            'risk_scores': results
        }

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('D:\\Colon-Cancer-Predicter\\data\\normal_data_added_noise.csv')
    X = data.drop(columns=['case_control', 'id'])  # Features
    y = data['case_control']  # Target variable

    # Initialize and build model
    model = ColoRecModel(input_shape=X.shape[1])
    model.build_model()

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data(X_scaled, y)

    # Print means and standard deviations of the data
    print("Means of the raw data:")
    print(X.mean(axis=0))

    print("\nStandard deviations of the raw data:")
    print(X.std(axis=0))

    # Train model
    history = model.train(X_train, y_train)

    # Make predictions
    mean_predictions, uncertainty = model.mc_dropout_predict(X_test, n_samples=100)

    # Save model
    model.save_model()

    # Plot calibration
    model.evaluate_calibration(y_test, mean_predictions)

    # Detailed evaluation
    model.detailed_evaluation(X_test, y_test)

    # Plot training history
    model.plot_training_history(history)

    # Plot feature importance
    model.plot_shap_summary(X_train, X_test)

    # Print model summary and weights
    model.print_model_summary()