import streamlit as st
import pickle
import pandas as pd
import requests
import io

# Load models from Hugging Face
@st.cache_resource
def load_model(url):
    response = requests.get(url)
    response.raise_for_status()
    model = pickle.load(io.BytesIO(response.content))
    return model

bert_model_url = "https://huggingface.co/Rendra7/Model_BERT/resolve/main/model_BERT.pkl"
svm_model_url = "https://huggingface.co/Rendra7/Model_BERT/resolve/main/svm_model.pkl"

bert_model = load_model(bert_model_url)
svm_model = load_model(svm_model_url)

history = []

# Function for sentiment classification
def classify_with_bert(text):
    # Dummy BERT classification for example
    return bert_model.predict([text])[0]

def classify_with_svm(text):
    # Dummy SVM classification for example
    return svm_model.predict([text])[0]

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
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
