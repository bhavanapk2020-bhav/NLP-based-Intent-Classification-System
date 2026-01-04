import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------- Page Config ----------
st.set_page_config(
    page_title="Intent Studio",
    page_icon="✨",
    layout="wide"
)

# ---------- Load Models ----------
label_encoder = joblib.load("label_encoder.pkl")
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------- NLP ----------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ---------- Signature Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #FAFAFA;
}

.wrapper {
    max-width: 760px;
    margin: auto;
    padding-top: 48px;
}

.brand {
    font-size: 14px;
    font-weight: 600;
    color: #4F46E5;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.heading {
    font-size: 44px;
    font-weight: 700;
    color: #111827;
    letter-spacing: -1.4px;
    line-height: 1.1;
}

.subheading {
    font-size: 17px;
    color: #6B7280;
    margin-top: 14px;
    max-width: 620px;
}

.input-label {
    margin-top: 42px;
    font-size: 14px;
    font-weight: 600;
    color: #374151;
}

.intent-box {
    margin-top: 28px;
    padding: 26px;
    border-radius: 16px;
    background: linear-gradient(135deg, #EEF2FF, #FFFFFF);
    border: 1px solid #E0E7FF;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    color: #1E1B4B;
}
            
.block-container {
    padding-top: 0.2rem ;
}


.helper {
    font-size: 13px;
    color: #9CA3AF;
    margin-top: 6px;
}

.footer {
    margin-top: 64px;
    font-size: 12px;
    color: #9CA3AF;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------- Layout ----------
st.markdown("<div class='wrapper'>", unsafe_allow_html=True)

st.markdown("""
<div class='brand'>Intent Studio</div>
<div class='heading'>Designing clarity from customer language</div>
<div class='subheading'>
An NLP and ML powered system that understands e-commerce customer intent.
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='input-label'>Customer query</div>", unsafe_allow_html=True)

user_input = st.text_area(
    "",
    placeholder="What's the status of my product?",
    height=68
)

st.markdown("<div class='helper'>Use natural, conversational language.</div>", unsafe_allow_html=True)

if st.button("Predict intent →", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter a customer query.")
    else:
        clean_text = preprocess_text(user_input)
        vector = vectorizer.transform([clean_text])
        intent = label_encoder.inverse_transform(model.predict(vector))[0]

        st.markdown(
            f"<div class='intent-box'>Predicted intent: {intent}</div>",
            unsafe_allow_html=True
        )

st.markdown("""
<div class='footer'>
Crafted with NLP • ML • TF-IDF • Naive Bayes • Streamlit
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
