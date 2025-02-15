import streamlit as st
import pickle
import numpy as np
from typing import Tuple

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔄",
    layout="wide"
)


def load_model() -> object:
    """Load the trained model from pickle file."""
    try:
        with open("model.pkl", "rb") as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        st.error("Error: Model file not found. Please ensure 'model.pkl' exists in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


def create_feature_inputs() -> Tuple:
    """Create and return all feature inputs."""
    # Create two columns for a better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Demographics")
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            step=1,
            help="Customer's credit score (300-900)"
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            step=1,
            help="Customer's age in years"
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            help="Customer's gender"
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"],
            help="Customer's country of residence"
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            format="%.2f",
            help="Customer's estimated annual salary"
        )

    with col2:
        st.subheader("Banking Information")
        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=10,
            step=1,
            help="Number of years as a customer"
        )

        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            format="%.2f",
            help="Current account balance"
        )

        num_of_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            step=1,
            help="Number of bank products the customer uses"
        )

        has_cr_card = st.selectbox(
            "Has Credit Card?",
            ["Yes", "No"],
            help="Whether the customer has a credit card"
        )

        is_active_member = st.selectbox(
            "Is Active Member?",
            ["Yes", "No"],
            help="Whether the customer is actively using services"
        )

    return (credit_score, age, gender, geography, estimated_salary,
            tenure, balance, num_of_products, has_cr_card, is_active_member)


def encode_features(features: Tuple) -> np.ndarray:
    """Encode categorical features and return numpy array of all features."""
    (credit_score, age, gender, geography, estimated_salary,
     tenure, balance, num_of_products, has_cr_card, is_active_member) = features

    # Encode categorical variables
    gender_encoded = 1 if gender == "Male" else 0
    has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
    is_active_encoded = 1 if is_active_member == "Yes" else 0

    # Encode geography
    geo_france = 1 if geography == "France" else 0
    geo_germany = 1 if geography == "Germany" else 0

    return np.array([[
        credit_score, age, tenure, balance, num_of_products,
        has_cr_card_encoded, is_active_encoded, estimated_salary,
        gender_encoded, geo_france, geo_germany
    ]])


def predict_churn(model: object, features: np.ndarray) -> str:
    """Make prediction using the model."""
    try:
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]
        return "Churn" if prediction[0] == 1 else "Not Churn", probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None


def main():
    # Load model
    model = load_model()

    # Page header
    st.title("🔄 Customer Churn Prediction")
    st.markdown("""
    This application predicts whether a customer is likely to churn based on their demographics
    and banking behavior. Enter the customer details below to get a prediction.
    """)

    # Create form for better user experience
    with st.form("prediction_form"):
        # Get all feature inputs
        features = create_feature_inputs()

        # Add predict button to form
        submitted = st.form_submit_button("Predict Churn")

    # Make prediction when form is submitted
    if submitted:
        # Show spinner while processing
        with st.spinner("Making prediction..."):
            # Encode features
            encoded_features = encode_features(features)

            # Get prediction and probability
            result, probability = predict_churn(model, encoded_features)

            if result and probability is not None:
                # Create columns for results
                col1, col2 = st.columns(2)

                with col1:
                    # Display prediction with custom styling
                    st.markdown("### Prediction")
                    if result == "Churn":
                        st.error(f"⚠️ Customer is likely to {result}")
                    else:
                        st.success(f"✅ Customer is likely to {result}")

                with col2:
                    # Display probability
                    st.markdown("### Confidence")
                    probability_percentage = probability * 100 if result == "Churn" else (1 - probability) * 100
                    st.progress(probability_percentage / 100)
                    st.write(f"Confidence: {probability_percentage:.1f}%")


if __name__ == "__main__":
    main()