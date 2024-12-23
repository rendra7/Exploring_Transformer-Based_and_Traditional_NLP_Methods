
# Exploring Transformer-Based and Traditional NLP Methods for Text Classification

#### Project By : Rendra Dwi Prasetyo

---

## Documentation Contents 
- [Project Overview](#project-overview)
- [Resources](#Resources)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#Performance-Results)
- [Analysis](#Analysis)
- [Conclusion](#conclusion)

---

## Project Overview
### Key Features:
1. **Transformer-Based** approach with **BERT (Bidirectional Encoder Representations from Transformers)**.
   - **BERT**: A  bidirectional pretrained language model designed to understand context and semantic relationships in text by learning deep representations through tasks like Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
2. **Traditional NLP Methods** using **TF-IDF** and **Support Vector Machine (SVM)**.
   - **TF-IDF + SVM**: A traditional machine learning pipeline where TF-IDF converts text into numerical vectors by weighting terms based on their frequency and importance across documents, and SVM classifies these vectors by finding an optimal hyperplane that separates different sentiment classes.
3. Preprocessing steps like tokenization, stopword removal, and lemmatization.
4. Comparison of both models on tricky examples to assess their handling of subtle contextual differences.

### Objectives:
1. Compare the performance of **BERT** with **TF-IDF** + **SVM** on sentiment analysis tasks.
2. Highlight the trade-offs between accuracy and computational resources for each approach.
3. Highlight model behavior on nuanced examples

---

## Resources
This project leverages the following resources:

**Development Environment**:
1. **Google Colab**: The notebook was developed and executed using Google Colab, a cloud-based platform for Python programming.
2. **Python Version**: Python 3.10.12 was used for all scripts and implementations.

**Hardware:**
1. **GPU**: NVIDIA T4 Tensor Core GPU provided by Google Colab. This GPU accelerates training and inference tasks, especially for deep learning models like BERT.
Libraries and Frameworks:
2. **System RAM** : 51.0 GB (maximum capacity)
3. **VRAM** : 15.0 GB (maximum capacity)
4. **Disk** : 235.7 GB (maximum capacity)

**Library & Tools**
1. **Transformers (Hugging Face)**: For fine-tuning the BERT model.
2. **scikit-learn**: For implementing traditional NLP methods, including TF-IDF and SVM.
3. **PyTorch**: To build and train the BERT classifier.
4. **pandas** : To read CSV file.
5. **pickle** : To save the trained model.
6. **Matplotlib & Seaborn**: For data visualization.
7. **NLTK**: For text preprocessing such as tokenization, stopword removal, and lemmatization.

  
---
## Dataset
The dataset contains English-language text data labeled with binary sentiment.

- **Origin Size**: 1,600,000 entries
- **Total Features** : 5
  
Source: [Sentiment140 Dataset from Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## Methodology
This methodology also serves as the step-by-step process I followed in the [notebook.ipynb](https://github.com/rendra7/Exploring_Transformer-Based_and_Traditional_NLP_Methods/blob/main/Notebook%20File%20-%20BERT%20%26%20SVM%20%2B%20TFIDF_Rendra%20Dwi%20Prasetyo_Project.ipynb) file.
### I. **Import Dataset**
- Import using API from kaggle with dataset of : [Sentiment140 Dataset from Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
### II. **Exploration & Dataset Preparation** : 
    - Understanding data characteristics through visualization, descriptive statistics, and text analysis.
       + The target sentiment distribution is balanced
  ![Gambar](https://github.com/user-attachments/assets/5c6de5d1-722e-47b7-be27-480c956334dd "width="100" height="150"") 
   
       + The text length distribution is in a safe range (between 7-348 characters) with an average length of 74 characters 
       + Data sets are cleaned of null/missing values
    - Dimensionality Data Reduction : reduce some unimportant feature and for computational resource efficiency and data balance. 
       + Changed from 160,000,000 data entries to 160,000 entries and cut 5 features into just 2 features, namely "text" and "target"
    - Remapping label from 0 (negative) and 4 (positive) --> into 0 for neagtive and 1 for postive.
### III. Implementation 
#### 1. **Text Representation and Classification with BERT**:
    - Fine-tuning a pretrained BERT-base-cased model on the sentiment dataset.
    - Input preprocessing: tokenization and attention mask generation using `BertTokenizer` from Hugging Face's `transformers` library.
    - Data splitting : 80% for training, 10% for testing, 10% for validation.
##### **Model classification using BERT**
    - Adding a classification layer for sentiment prediction.
    - Defining a custom classifier:
    - Using `BertModel` as the backbone for encoding input text.
    - Adding a dropout layer (0.5) and a dense classification layer with ReLU activation for binary sentiment prediction.
    - Training the model with cross-entropy loss and optimizer with Adam.
    - Saving the trained model for reuse.
##### **Evaluation**
    - Evaluation metrics : Accuracy, precision, recall, F1-Score.
    - Testing & Analysis : Testing the model with new sentences for sentiment analysis.

#### 2. **Traditional NLP Methods**:
##### Text Cleaning
    - Removing URLs, symbols, stopwords, and applying lemmatization.
##### **Word Embedding**
    - Text vectorization using TF-IDF:
       - Transforming the text data into numerical vectors using `TfidfVectorizer`.
       - Separately transforming training and testing datasets.
##### **Using SVM for Classification Task** :
    - Training an SVM classifier:
       - Hyperparameter optimization using `GridSearchCV` with `C` values [0.1, 1, 10] and kernel types (`linear`, `rbf`).
       - Validation using 5-fold cross-validation.
    - Saving the fine-tuned model for reuse.
##### **Evaluation**
    - Evaluation metrics : Accuracy, precision, recall, F1-Score
    - Testing & Analysis : Testing the classifier with new sentences for performance analysis.
### IV. Analysis
[Analysis Section](#Analysis)
### V. Conclusion
[Conclusion Section](#Conclusion)

---

## Performance Results
| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| BERT           | 83%      | 83%       | 83%    | 83%      |
| TF-IDF + SVM   | 76%      | 76%       | 76%    | 76%      |

- The performance results based on the evaluation metrics (accuracy, precision, recall, F1-score) of **BERT are better overall**. This is in accordance with the theory that BERT's ability to analyze text more deeply by paying attention to contextual relationships gives it an edge over more traditional models like TF-IDF + SVM.

---
## Analysis
To strengthen the justification for the results I obtained, I tested with several sample sentences for a deeper analysis. These examples help demonstrate how both models—BERT and TF-IDF + SVM—handle different types of text, particularly those with complex or nuanced sentiment.


#### Result Predictions:
| Text | BERT Prediction | TF-IDF + SVM Prediction |
|------|----------------|---------------|
| This is a great product! | Positive | Positive |
| I'm really great disappointed with this product. | Negative | Negative |
| The service was amazing; I’m so satisfied and will definitely come back. | Positive | Positive |
| The ticket prices are amazing—amazingly expensive. | Negative | Positive |

### Key Difference in Example Prediction:
- **BERT** successfully captures subtle contextual meanings, such as interpreting "amazingly expensive" as negative, showing its deeper understanding of text context.
- **SVM**, relying more on word frequency, can misclassify nuanced sentences, as seen in the example of "amazingly expensive," where it mistakenly classifies the sentiment as positive.
---

## Conclusion 
### Results and Insights
#### BERT:
- Captures context and nuances in text, making it more robust for handling complex and subtle statements.
- Better suited for datasets with subtle sentiment cues and intricate contextual relationships.
#### TF-IDF + SVM:
- Simpler to train and computationally less intensive.
- Effective for datasets with clear and straightforward sentiment patterns, but may struggle with nuanced or complex examples.
### Overall Comparison:
**BERT** outperforms traditional methods like TF-IDF + SVM, especially on tricky examples where context and nuance matter. However, it requires more computational resources and may not always be the best choice for simpler datasets.

---



