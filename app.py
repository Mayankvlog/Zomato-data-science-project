import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError # Import for custom_objects
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
# Ensure the model file name matches the one saved
try:
    # Provide the custom_objects argument to specify the loss function if needed
    model = load_model('zomato_cnn_recommendation_model.h5', custom_objects={'mse': MeanSquaredError()})
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

# Load the tokenizer
# Ensure the tokenizer file name matches the one saved
try:
    with open('tokenizer_recommendation.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    st.success("Tokenizer loaded successfully.")
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    tokenizer = None # Set tokenizer to None if loading fails

# Define the maximum sequence length used during training
max_sequence_length = 100 # Ensure this matches the value used in padding

st.title("Restaurant Rating Prediction (CNN-based)")

st.write("Enter a restaurant review to predict its rating based on our CNN model.")

# Get input from the user
review_input = st.text_area("Enter Review Here")

# Use the recommendation logic (predict_rating_from_review function)
# Define the function directly within the Streamlit app script or import if in a separate file
def predict_rating_from_review(review_text):
    """
    Predicts the rating of a restaurant based on a review text using the trained CNN model.

    Args:
        review_text (str): The text of the restaurant review.

    Returns:
        float: The predicted rating, or None if model/tokenizer not loaded or input is invalid.
    """
    if model is None or tokenizer is None:
        # Error message already shown during loading
        return None

    if not isinstance(review_text, str) or not review_text.strip():
        st.warning("Please enter a valid review text.")
        return None

    # Preprocess the input review
    sequence = tokenizer.texts_to_sequences([review_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post', truncating='post')

    # Ensure the input shape matches the model's expected input shape (batch_size, sequence_length, 1)
    model_input = padded_sequence.reshape(padded_sequence.shape[0], padded_sequence.shape[1], 1)

    # Make a prediction
    prediction = model.predict(model_input)

    # The model outputs a single numerical rating
    predicted_rating = prediction[0][0]

    return predicted_rating

if st.button("Predict Rating"):
    if review_input and model is not None and tokenizer is not None:
        predicted = predict_rating_from_review(review_input)
        if predicted is not None:
            st.subheader("Prediction:")
            st.success(f"Predicted Rating: {predicted:.2f}")
    elif not review_input:
        st.warning("Please enter a review before clicking Predict.")
