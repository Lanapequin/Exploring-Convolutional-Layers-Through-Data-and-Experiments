"""SageMaker inference script for Fashion-MNIST CNN model"""

import json
import torch
import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    """Single convolutional block: Conv → ReLU → MaxPool"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class CNNClassifier(nn.Module):
    """Convolutional Neural Network for Fashion-MNIST"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvolutionalBlock(1, 32, kernel_size=3),
            ConvolutionalBlock(32, 64, kernel_size=3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def model_fn(model_dir):
    """Load the PyTorch model from the model_dir"""
    model = CNNClassifier()
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location="cpu"))
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    """Process incoming prediction request"""
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    data = json.loads(request_body)

    # Expect: {"inputs": [[28x28], [28x28], ...]}
    x = torch.tensor(data["inputs"], dtype=torch.float32)

    # If input is (batch, 28, 28), add channel dim -> (batch, 1, 28, 28)
    if x.ndim == 3:
        x = x.unsqueeze(1)

    # If input is single image (28, 28), make it (1, 1, 28, 28)
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)

    # Normalize if inputs look like 0..255
    if x.max() > 1.5:
        x = x / 255.0

    return x


def predict_fn(input_data, model):
    """Make predictions on input data"""
    with torch.no_grad():
        outputs = model(input_data)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    return predictions, probabilities


def output_fn(prediction_output, content_type):
    """Format prediction output"""
    predictions, probabilities = prediction_output

    return json.dumps({
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    })
