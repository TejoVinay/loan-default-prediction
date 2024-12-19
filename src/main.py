import os
import pandas as pd

from utils.load_model_and_predict import load_model_and_predict

if __name__ == "__main__":
#    import argparse
   
#    parser = argparse.ArgumentParser(description='Predict loan defaults from CSV file')
#    parser.add_argument('input_file', type=str, help='Path to input CSV file')
#    parser.add_argument('output_file', type=str, help='Path to save predictions CSV')
   
#    args = parser.parse_args()
   
#    if not success:
#        exit(1)
#   ---Avoiding the above argparser implementation to simplify main---
    data = pd.read_csv(os.path.join(os.getcwd(), "src", "data", "testing_loan_data.csv"))
    predictions = load_model_and_predict(
        data, 
        os.path.join(os.getcwd(), "src", "models", "loan_default_model.pth"),
        os.path.join(os.getcwd(), "src", "models", "processing_params.pkl")
    )
    result_data = data.drop('bad_flag', axis=1)
    result_data['bad_flag'] = (predictions.squeeze() >= 0.5).astype(int)
    print(result_data['bad_flag'].value_counts())
    result_data.to_csv(os.path.join(os.getcwd(), "src", "data", "test_result.csv"))
