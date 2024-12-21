from transformers import BertForSequenceClassification, BertTokenizer

@st.cache_resource
def load_model(url):
    try:
        # Example for Hugging Face's BERT model
        model = BertForSequenceClassification.from_pretrained(url)
        tokenizer = BertTokenizer.from_pretrained(url)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Example usage
bert_model, bert_tokenizer = load_model(bert_model_url)
