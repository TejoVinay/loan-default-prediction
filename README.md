# Loan Default Prediction

This project implements a binary classification neural network using PyTorch to predict loan defaults. The implementation includes comprehensive exploratory data analysis (EDA) and a PyTorch-based neural network model.

## Project Structure
- `src/`: Source code for the project
  - `data/`: Data processing utilities
  - `models/`: Neural network implementation
  - `utils/`: Helper functions and visualization tools
- `notebooks/`: Jupyter notebooks for EDA and model training
- `tests/`: Unit tests

## Setup

1. Clone the repository:
    ```bash
    git clone [repository-url]
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
[To be added as we implement the project]

## Development Process
[To be added as we progress]


------

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
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_processor.py      # Data processing utilities
│   ├── models/
│   │   ├── __init__.py
│   │   └── loan_default_model.pth # Neural network implementation
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
├── data/
│   ├── cleaned_train.csv
│   ├── test_result.csv
│   ├── test_result_01.csv
│   ├── testing_loan_data.csv
│   └── training_loan_data.csv
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
1. Start with the EDA notebooks in order:
    ```bash
    jupyter notebook notebooks/01_eda_data_inspection.ipynb
    ```
   Follow through notebooks 01-03 for complete EDA process.

### Model Training
1. Prepare the data and build the neural network:
    ```bash
    jupyter notebook notebooks/04_data_preparation_neural_network_build.ipynb
    ```

2. Train and finalize the model:
    ```bash
    jupyter notebook notebooks/05_final_model_nn.ipynb
    ```

### Making Predictions
1. For new predictions:
    ```bash
    jupyter notebook notebooks/06_predict.ipynb
    ```

## Model Architecture

The neural network implementation consists of:
- Input layer matching the feature dimensions
- Multiple hidden layers with configurable neurons
- Binary classification output layer
- Activation functions: ReLU for hidden layers, Sigmoid for output
- Binary Cross-Entropy loss function
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

## License

[Specify your license here]