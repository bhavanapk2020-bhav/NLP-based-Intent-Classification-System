import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Load model & vectorizer also encode
label_encoder = joblib.load("label_encoder.pkl")
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# NLP setup
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("🛒 E-commerce Customer Support Chatbot")
st.write("Enter a customer query to predict intent")

user_input = st.text_input("Customer Query")

if st.button("Predict Intent"):
    if user_input.strip() == "":
        st.warning("Please enter a query")
    else:
        clean_text = preprocess_text(user_input)
        vector = vectorizer.transform([clean_text])
        predicted_label = model.predict(vector)[0]
        predicted_intent = label_encoder.inverse_transform([predicted_label])[0]
        st.success(f"Predicted Intent: **{predicted_intent}**")
