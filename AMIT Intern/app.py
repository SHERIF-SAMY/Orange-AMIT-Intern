import streamlit as st
import nltk  # Ensure nltk is imported
import pickle
import helper

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the model and vectorizer
model = pickle.load(open('Models/model.pkl', 'rb'))  # Use forward slashes for paths
vectorizer = pickle.load(open('Models/vectorizer.pkl', 'rb'))

# Streamlit app title
st.title('Sentiment Analysis App')

# Input text
text = st.text_input('Enter your review:')

# Preprocess and vectorize the input
if text:
    token = helper.preprocessing_step(text)
    vectorized_data = vectorizer.transform([token])

# Button for prediction
state = st.button('Predict')

# Prediction and output
if state:
    prediction = model.predict(vectorized_data)  # Predict sentiment
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    st.text(f'Sentiment: {sentiment}')
    # st.text(prediction)





