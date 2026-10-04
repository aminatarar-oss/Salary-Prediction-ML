# Salary Prediction using Machine Learning

## Project Description
This project predicts salary based on years of experience using
a Linear Regression model and an interactive Streamlit application.

## Dataset Source
https://raw.githubusercontent.com/SagarChhabriya/data-science/refs/heads/main/datasets/TBD/salary_dataset.csv

## Model
Linear Regression

## Input
Years of Experience

## Target
Salary

## Methodology
1. Load and explore the dataset.
2. Check missing values and duplicate records.
3. Clean the data.
4. Split the data into training and testing sets (80:20).
5. Train a Linear Regression model.
6. Evaluate the model using MSE, RMSE, MAE, and R-squared.
7. Save the trained model using Joblib.
8. Build an interactive Streamlit application.

## Evaluation
See outputs/model_metrics.csv for the evaluation metrics
generated during model training.

## Run the Application
Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Limitations
Predictions are estimates based on the supplied dataset.
They are not guaranteed salaries or market salary benchmarks.
