import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from utils.data_preparation import clean_loan_data
from utils.data_preparation import prepare_data_for_inference

def load_model_and_predict(data, model_path, preprocessing_path):
    """Load saved model and make predictions"""
    # Define model architecture
    class LoanDefaultModel(nn.Module):
        def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
            super(LoanDefaultModel, self).__init__()
            
            self.input_bn = nn.BatchNorm1d(input_dim)
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
                prev_dim = hidden_dim
            
            self.hidden_layers = nn.Sequential(*layers)
            self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        def forward(self, x):
            x = self.input_bn(x)
            x = self.hidden_layers(x)
            return torch.sigmoid(self.output_layer(x))

    # Load preprocessing parameters
    with open(preprocessing_path, 'rb') as f:
        preprocessing_params = pickle.load(f)
    
    # Clean data
    clean_data = clean_loan_data(data)
    # Prepare data
    processed_data = prepare_data_for_inference(clean_data, preprocessing_params)

    processed_data = processed_data.drop('bad_flag', axis=1)
    
    # Load model
    checkpoint = torch.load(model_path)
    model = LoanDefaultModel(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Convert to tensor
    X = torch.FloatTensor(processed_data.values)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X).numpy()
    
    return predictions