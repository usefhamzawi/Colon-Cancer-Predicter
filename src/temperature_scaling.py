import torch
import numpy as np
import tensorflow as tf
from torch import nn, optim
from torch.nn import functional as F

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling for binary classification
    model (nn.Module):
        A binary classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Start with a temperature of 1.5

    def forward(self, input):
        # Ensure the input is reshaped to match the model's expected shape (batch_size, 17)
        input = self._reshape_input(input)
        logits = self.model(input)
        return self.temperature_scale(logits)  # Apply temperature scaling on logits

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits (used for calibration)
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def _reshape_input(self, input):
        """
        Ensure the input has the correct shape (batch_size, 17).
        If the input is only a single feature, replicate it to match the expected shape.
        """
        if input.shape[1] == 1:  # If the input has only 1 feature, expand it to 17 features
            input = input.repeat(1, 17)  # Replicating the feature to match the expected 17 features
        return input

    def set_temperature(self, valid_loader):
        """
        Tune the temperature of the model (using the validation set).
        We're going to set it to optimize NLL (Negative Log-Likelihood).
        valid_loader (DataLoader): validation set loader
        """
        self.cpu()  # Ensure model is on CPU
        nll_criterion = nn.BCEWithLogitsLoss().cpu()  # Use BCEWithLogitsLoss for binary classification
        ece_criterion = _ECELoss().cpu()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                # Convert numpy ndarray to PyTorch tensor if needed
                if isinstance(input, np.ndarray):
                    input = torch.tensor(input)

                input = input.cpu()  # Ensure the input is on the correct device
                input = self._reshape_input(input)  # Ensure the input is reshaped correctly

                # Ensure that input is a PyTorch tensor (in case it's TensorFlow)
                if isinstance(input, tf.Tensor):
                    input = torch.tensor(input.numpy())  # Convert TensorFlow tensor to PyTorch tensor

                # Get logits from the model (only input data is passed through the model)
                logits = self.model(input)

                # If the logits are a TensorFlow EagerTensor, convert it to a PyTorch tensor
                if isinstance(logits, tf.Tensor):
                    logits = torch.tensor(logits.numpy())  # Convert EagerTensor to PyTorch tensor

                logits_list.append(logits)

                # Ensure the labels are PyTorch tensors (in case they are numpy arrays or TensorFlow)
                if isinstance(label, np.ndarray):
                    label = torch.tensor(label)  # Convert numpy ndarray to PyTorch tensor
                elif isinstance(label, tf.Tensor):
                    label = torch.tensor(label.numpy())  # Convert TensorFlow tensor to PyTorch tensor

                labels_list.append(label)

        logits = torch.cat(logits_list).cpu()
        labels = torch.cat(labels_list).cpu()

        # Ensure labels are within the valid range for BCEWithLogitsLoss
        labels = labels.float()  # BCEWithLogitsLoss expects labels in float format (0 or 1)

        before_temperature_nll = nll_criterion(logits.squeeze(), labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(f'Before temperature - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}')

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits).squeeze(), labels)  # Apply temperature scaling
            loss.backward()
            return loss

        optimizer.step(eval)

        after_temperature_nll = nll_criterion(self.temperature_scale(logits).squeeze(), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print(f'Optimal temperature: {self.temperature.item():.3f}')
        print(f'After temperature - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}')

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece