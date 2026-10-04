import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# Load the saved model and dataset
model = joblib.load("model/salary_model.pkl")
df = pd.read_csv("data/salary_dataset.csv")

# Application heading
st.title("Salary Prediction Application")
st.write(
    "Enter your years of experience to estimate salary "
    "using a Linear Regression model."
)

# User input
max_experience = float(df["Experience Years"].max())

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=max_experience,
    value=min(1.0, max_experience),
    step=0.1
)

# Prediction button
if st.button("Predict Salary"):
    input_data = pd.DataFrame({
        "Experience Years": [experience]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Salary: {prediction:,.2f}")
    st.caption(
        "This is an estimate based on the training dataset, "
        "not a guaranteed salary."
    )

# Dataset information
st.subheader("Dataset Overview")
col1, col2 = st.columns(2)
col1.metric("Number of Records", len(df))
col2.metric("Number of Features", 1)

# Show dataset sample
with st.expander("View dataset sample"):
    st.dataframe(df.head(10), use_container_width=True)

# Visualize the dataset
st.subheader("Experience vs Salary")

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(df["Experience Years"], df["Salary"])
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.set_title("Experience and Salary")
fig.tight_layout()

st.pyplot(fig)

# Model evaluation metrics
metrics_path = "outputs/model_metrics.csv"

try:
    metrics = pd.read_csv(metrics_path)
    st.subheader("Model Performance")
    st.dataframe(metrics, use_container_width=True)
except FileNotFoundError:
    st.info("Model performance metrics are not available.")
