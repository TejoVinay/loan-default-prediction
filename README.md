# Loan Default Prediction

This project implements a binary classification neural network using PyTorch to predict loan defaults. The implementation includes comprehensive exploratory data analysis (EDA) and a PyTorch-based neural network model that helps predict whether a loan application will default based on various features.

## Project Overview

The model analyzes various loan application features such as:
- Annual income
- Debt-to-income ratio
- Credit utilization
- Employment length
- And other relevant credit attributes

## Project Structure

```
loan-default-prediction/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── data/ (maintain all csv files here)
│   │   └── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loan_default_model.pth # Best working nueral network model
│   │   └── processing_params.pkl  # Model processing parameters
│   └── utils/
│       ├── __init__.py
│       ├── data_preparation.py    # Data preparation utilities
│       └── load_model_and_predict.py # Model loading and prediction
├── notebooks/
│   ├── 01_eda_data_inspection.ipynb
│   ├── 02_eda_data_cleaning.ipynb
│   ├── 03_eda_data_viz_insights.ipynb
│   ├── 04_data_preparation_neural_network_build.ipynb
│   ├── 05_final_model_nn.ipynb
│   └── 06_predict.ipynb
│  
├── tests/
│   └── __init__.py
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Setup

1. Clone the repository:
    ```bash
    git clone [repository-url]
    cd loan-default-prediction
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation and EDA
1. Run the EDA notebook for cleaning training data where the data types are handled and missing values are filled through a detailed analysis of the features and their relations:
    ```bash
    jupyter notebook notebooks/02_eda_data_cleaning.ipynb
    ```
   Follow through notebooks 01-03 for complete EDA process.
   Notebook 01_eda_data_inspection.ipynb consists of initial data inspection not necessary for running the project
   Notebook 03_eda_data_viz_and_insights.ipynb has a detailed analysis through visualizations and statistical tests to make informed decisions on feature transformation, scaling, and engineering.

### Model Training
1. Prepare the data, train, and finalize the model:
    ```bash
    jupyter notebook notebooks/05_final_model_nn.ipynb
    ```
    Notebook 04_data_preparation_neural_network_build.ipynb consists of the data preparation approach and different models built and tested for improving model performance not necessary for running the project
    Notebook 05_final_model_nn.ipynb has the final data preparation strategy, final model trained on train-val and then tested on the test split. Preprocessing params and model params are saved.

### Making Predictions
1. For new predictions:
    ```bash
    jupyter notebook notebooks/06_predict.ipynb
    ```

## Alternate Usage (Simpler)
- You can alternately provide test csv to the main.py directly and run it which pulls all the required data preprocessing, model load, and predict functions from the utils package to execute the prediction process end-to-end and gives out a csv filled with the predicted values in the target variable.
    ```bash
    python src/main.py
    ```

## Model Architecture

The neural network implementation consists of:
- Input layer matching the feature dimensions
- Multiple hidden layers with configurable neurons
- Binary classification output layer
- Activation functions: ReLU for hidden layers, Sigmoid for output
- Custom Binary Cross-Entropy loss function with class weights for treating the high class imbalance in the training set
- Adam optimizer

## Data Description

The model uses various features including:
- Annual income
- Bankcard utilization
- Debt-to-income ratio
- Employment length
- Home ownership status
- Number of recent credit inquiries
- And more

## Assumptions

1. Data Quality:
   - Missing values are handled appropriately during preprocessing
   - Outliers are identified and treated in the data cleaning phase
   - Categorical variables are properly encoded

2. Model:
   - Binary classification problem (0: non-default, 1: default)
   - Features are normalized/standardized before model training
   - The model assumes independence between features

3. Implementation:
   - Training data and test data follow the same distribution
   - Sufficient memory is available for data processing
   - GPU acceleration is optional but recommended for larger datasets

## Contributing

Please read through the notebooks in order to understand the implementation details. Each notebook is well-documented with markdown cells explaining the process.
