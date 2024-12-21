import streamlit as st

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Import libraries
import pickle
import pandas as pd
import requests
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import io  # For handling byte streams

# Load models from Hugging Face
@st.cache_resource
def load_model(url, model_type="bert"):
    try:
        if model_type == "bert":
            # Load BERT model using Hugging Face's transformers library
            model = BertForSequenceClassification.from_pretrained(url)
            tokenizer = BertTokenizer.from_pretrained(url)
            return model, tokenizer
        elif model_type == "svm":
            # Load SVM model using pickle
            response = requests.get(url)
            response.raise_for_status()
            model = pickle.load(io.BytesIO(response.content))
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# URL to the Hugging Face models
bert_model_url = "https://huggingface.co/Rendra7/Model_BERT/resolve/main/model_BERT.pkl"
svm_model_url = "https://huggingface.co/Rendra7/Model_BERT/resolve/main/svm_model.pkl"

# Load the models
bert_model, bert_tokenizer = load_model(bert_model_url, model_type="bert")
svm_model = load_model(svm_model_url, model_type="svm")

if bert_model is None or bert_tokenizer is None:
    st.error("Failed to load BERT model or tokenizer. Please check the model URL.")
    st.stop()

if svm_model is None:
    st.error("Failed to load SVM model. Please check the model URL.")
    st.stop()

history = []

# Function for sentiment classification using BERT
def classify_with_bert(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    return "Positive" if prediction == 1 else "Negative"

# Function for sentiment classification using SVM
def classify_with_svm(text):
    # Example: Using a dummy vectorizer for SVM (adjust as needed for actual vectorizer)
    vectorized_text = vectorizer.transform([text])  # Assuming `vectorizer` is already defined
    return svm_model.predict(vectorized_text)[0]

# Streamlit UI
st.title("Sentiment Analysis Application")

# Sidebar menu
menu = st.sidebar.radio("Menu", ["About This Application", "Sentiment Classification", "History"])

if menu == "About This Application":
    st.header("About This Application")
    st.write("""
    **Application Usage:**
    1. Input a sentence in the 'Sentiment Classification' menu.
    2. Click the 'Analyze' button to classify the sentiment using the BERT model.
    3. Optionally, compare the result with the SVM (BoW) model by clicking 'Compare'.
    4. Check the 'History' menu to view all previous classifications.

    **Developer:**
    RENDRA DWI PRASETYO
    """)

elif menu == "Sentiment Classification":
    st.header("Sentiment Classification")

    text_input = st.text_area("Enter a sentence for sentiment analysis:", "")

    if st.button("Analyze"):
        if text_input.strip():
            # BERT Classification
            bert_result = classify_with_bert(text_input)
            st.success(f"BERT Sentiment: {bert_result}")

            if st.button("Compare with SVM (BoW)"):
                svm_result = classify_with_svm(text_input)
                st.info(f"SVM (BoW) Sentiment: {svm_result}")

                # Update history
                history.append({
                    "Text": text_input,
                    "BERT Sentiment": bert_result,
                    "SVM (BoW) Sentiment": svm_result
                })
            else:
                # Update history without comparison
                history.append({
                    "Text": text_input,
                    "BERT Sentiment": bert_result,
                    "SVM (BoW) Sentiment": "None"
                })
        else:
            st.warning("Please enter a sentence before analysis.")

elif menu == "History":
    st.header("Classification History")

    if history:
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True)

        st.markdown("### Summary")
        st.write(f"Total analyses: {len(history)}")
        bert_count = df["BERT Sentiment"].value_counts()
        svm_count = df["SVM (BoW) Sentiment"].value_counts()

        st.subheader("BERT Model Sentiment Distribution")
        st.write(bert_count)

        st.subheader("SVM (BoW) Model Sentiment Distribution")
        st.write(svm_count)
    else:
        st.info("No classification history yet.")
