import streamlit as st
import pickle
import requests
import io
from torch import nn
from transformers import BertModel
import torch

# Define the BertClassifier class
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

# Function to load model from URL
def load_model(url):
    response = requests.get(url)
    response.raise_for_status()
    model = pickle.load(io.BytesIO(response.content))
    return model

# Load models
bert_model_url = "https://huggingface.co/Rendra7/Model_BERT/resolve/main/model_BERT.pkl"
svm_model_url = "https://huggingface.co/Rendra7/Model_BERT/resolve/main/svm_model.pkl"

@st.cache_resource
def load_bert_model():
    return load_model(bert_model_url)

@st.cache_resource
def load_svm_model():
    return load_model(svm_model_url)

bert_model = load_bert_model()
svm_model = load_svm_model()

# Helper function for BERT prediction
def bert_predict(model, text):
    tokenizer = BertModel.from_pretrained('bert-base-cased').tokenizer
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    return torch.argmax(output, dim=1).item()

# Helper function for SVM prediction
def svm_predict(model, text):
    return model.predict([text])[0]

# Initialize Streamlit App
st.title("Sentiment Analysis with BERT and SVM")
menu = ["About This Application", "Sentiment Analysis", "History"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "About This Application":
    st.header("About This Application")
    st.write("""
    **Developed by:** Rendra Dwi Prasetyo
    
    **Description:**  
    This application uses two models for sentiment analysis:
    1. A fine-tuned BERT model.
    2. A Support Vector Machine (SVM) model with Bag-of-Words (BoW) representation.
    
    **How to use:**  
    - Navigate to the "Sentiment Analysis" menu.
    - Input your sentence and analyze the sentiment using BERT.
    - Optionally, compare the result with SVM by enabling the "Compare" option.
    - The history of predictions will be displayed in the "History" menu.
    """)

elif choice == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    text = st.text_area("Enter a sentence for sentiment analysis:")
    compare = st.checkbox("Compare with SVM")
    if st.button("Analyze"):
        if text:
            bert_result = bert_predict(bert_model, text)
            st.write(f"**BERT Sentiment:** {'Positive' if bert_result == 1 else 'Negative'}")

            svm_result = None
            if compare:
                svm_result = svm_predict(svm_model, text)
                st.write(f"**SVM Sentiment:** {'Positive' if svm_result == 1 else 'Negative'}")
            
            # Save to history
            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].append({
                "Text": text,
                "BERT": "Positive" if bert_result == 1 else "Negative",
                "SVM": "Positive" if svm_result == 1 else "Negative" if compare else "None"
            })
        else:
            st.warning("Please enter a sentence!")

elif choice == "History":
    st.header("History")
    if "history" in st.session_state and st.session_state["history"]:
        history_data = st.session_state["history"]
        st.table(history_data)
    else:
        st.write("No history available.")
