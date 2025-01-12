import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def temperature_scale(self, X_val, y_val, X_test):
        """
        Perform temperature scaling
        """
        
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