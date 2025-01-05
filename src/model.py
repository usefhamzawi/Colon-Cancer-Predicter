import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ColoRecModel:
    def __init__(self, input_shape):
        """
        Initialize the Colon Recurrence Risk Prediction Model
        
        Parameters:
        - input_shape: Number of input features
        """
        self.input_shape = input_shape
        self.model = None
        self.scaler = StandardScaler()

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
        
        # Hidden layers
        for neurons, dropout_rate in zip(hidden_layers, dropout_rates):
            self.model.add(Dense(neurons, 
                                 activation='relu', 
                                 kernel_regularizer=regularizers.l2(l2_lambda)))
            self.model.add(Dropout(dropout_rate))
        
        # Output layer
        self.model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        self.model.compile(optimizer='adam', 
                           loss='binary_crossentropy', 
                           metrics=['accuracy'])
        
        return self.model

    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Prepare and split the data
        
        Parameters:
        - X: Features
        - y: Labels
        - test_size: Proportion of test data
        - random_state: Random seed for reproducibility
        
        Returns:
        - Scaled train and test datasets
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

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

    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Plot confusion matrix
        
        Parameters:
        - y_test: True labels
        - y_pred: Predicted labels
        """
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def calibrate_probabilities(self, X_test, y_test, method='sigmoid', cv=5):
        """
        Calibrate model probabilities using cross-validation
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import BaseEstimator, ClassifierMixin
        import numpy as np
        
        # Create proper sklearn-compatible wrapper
        class KerasWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self):
                self.classes_ = np.array([0, 1])
                
            def fit(self, X, y):
                return self
                
            def predict_proba(self, X):
                return np.hstack([1-X, X])
                
            def get_params(self, deep=True):
                return {}
                
            def set_params(self, **parameters):
                return self
        
        # Get uncalibrated probabilities
        y_proba = self.model.predict(X_test).flatten()
        y_proba = y_proba.reshape(-1, 1)
        
        # Create and fit calibrator
        calibrator = CalibratedClassifierCV(
            estimator=KerasWrapper(),
            method=method,
            cv=cv,
            n_jobs=-1,
            ensemble=True
        )
        
        calibrator.fit(y_proba, y_test)
        calibrated_proba = calibrator.predict_proba(y_proba)[:, 1]
        
        return calibrated_proba
    
    def evaluate_calibration(self, y_true, y_proba, n_bins=10):
        """
        Evaluate prediction calibration using reliability diagram
        """
        from sklearn.calibration import calibration_curve
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        
        # Plot reliability diagram
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        plt.plot(prob_pred, prob_true, 's-', label='Model')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('True probability')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True)
        
        # Calculate and return calibration metrics
        ece = np.mean(np.abs(prob_true - prob_pred))
        mce = np.max(np.abs(prob_true - prob_pred))
        
        return {
            'Expected Calibration Error': ece,
            'Maximum Calibration Error': mce
        }

    def temperature_scale(self, X_val, y_val, X_test):
        """
        Perform temperature scaling
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Prevent division by zero and numerical instability
        def logit_transform(probabilities):
            eps = 1e-12
            probabilities = np.clip(probabilities, eps, 1 - eps)
            return np.log(probabilities / (1 - probabilities))
        
        class TemperatureScaler(nn.Module):
            def __init__(self):
                super().__init__()
                self.temperature = nn.Parameter(torch.ones(1) * 1.5)
            
            def forward(self, logits):
                return logits / self.temperature
        
        # Get model predictions
        y_proba_val = self.model.predict(X_val).flatten()
        y_proba_test = self.model.predict(X_test).flatten()
        
        # Convert to logits more safely
        logits_val = logit_transform(y_proba_val)
        logits_test = logit_transform(y_proba_test)
        
        # Convert to torch tensors (with pandas Series conversion)
        logits_val_torch = torch.FloatTensor(logits_val)
        y_val_torch = torch.FloatTensor(y_val.values)  # Convert pandas Series to numpy array
        logits_test_torch = torch.FloatTensor(logits_test)
        
        # Initialize temperature scaler
        temperature_scaler = TemperatureScaler()
        
        # Use NLL loss instead of BCE for better numerical stability
        criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer with better parameters
        optimizer = optim.LBFGS([temperature_scaler.temperature], 
                            lr=0.01, 
                            max_iter=50,
                            line_search_fn='strong_wolfe')
        
        def eval():
            optimizer.zero_grad()
            scaled_logits = temperature_scaler(logits_val_torch)
            loss = criterion(scaled_logits, y_val_torch)
            loss.backward()
            return loss
        
        # Optimize temperature
        optimizer.step(eval)
        
        # Get the final temperature value
        final_temp = temperature_scaler.temperature.item()
        print(f"Optimal temperature: {final_temp:.3f}")
        
        # Scale test logits and convert back to probabilities
        with torch.no_grad():
            scaled_test_logits = temperature_scaler(logits_test_torch)
            scaled_test_proba = torch.sigmoid(scaled_test_logits).numpy()
        
        return scaled_test_proba
    
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

    def plot_calibration(self, y_test, y_pred_proba, n_bins=10):
        """
        Plot calibration curve for the model
        
        Parameters:
        - y_test: True labels
        - y_pred_proba: Predicted probabilities
        - n_bins: Number of bins for calibration curve (default=10)
        """
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        # Create the calibration curve plot
        plt.figure(figsize=(10, 10))
        
        # Plot the perfect calibration line
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')

        # Calculate and plot calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=n_bins)
        plt.plot(prob_pred, prob_true, 's-', label='Model')

        # Customize the plot
        plt.xlabel('Mean predicted probability')
        plt.ylabel('True probability')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        
        # Calculate and add calibration metrics
        metrics = self.evaluate_calibration(y_test, y_pred_proba)
        plt.text(0.05, 0.95, 
                f'Expected Calibration Error: {metrics["Expected Calibration Error"]:.3f}\n'
                f'Maximum Calibration Error: {metrics["Maximum Calibration Error"]:.3f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))

        plt.show()


if __name__ == "__main__":
    # Load data
    data = pd.read_csv('D:\\Colon-Cancer-Predicter\\data\\data4.csv')
    X = data.drop(columns=['case_control', 'id'])
    y = data['case_control']

    # Initialize and build model
    model = ColoRecModel(input_shape=X.shape[1])
    model.build_model()

    # Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test = model.prepare_data(X, y)

    # Train model
    history = model.train(X_train_scaled, y_train)

    # Get predictions and plot calibration
    y_pred_proba = model.model.predict(X_test_scaled).flatten()
    model.plot_calibration(y_test, y_pred_proba)

    # Save model
    model.save_model()
    model.detailed_evaluation(X_test_scaled, y_test)