
# Exploring Transformer-Based and Traditional NLP Methods for Text Classification

#### Project By : Rendra Dwi Prasetyo

.

This project presents a comprehensive comparison of sentiment analysis approaches using two distinct methods:
1. **Transformer-Based** approach with **BERT (Bidirectional Encoder Representations from Transformers)**.
2. **Traditional NLP Methods** using **TF-IDF** and **Support Vector Machine (SVM)**.

The study evaluates the performance of these models on sentiment analysis tasks, specifically focusing on Twitter comments. The results demonstrate the strengths and weaknesses of each approach in terms of accuracy, precision, recall, and F1-score.

---

## README Contents 
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#Performance-Results)
- [Analysis](#Analysis)
- [Conclusion](#conclusion)

---

## Project Overview
### Key Features:
- **BERT**: A transformer-based, bidirectional pretrained language model designed to understand context and semantic relationships in text by learning deep representations through tasks like Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
- **TF-IDF + SVM**: A traditional machine learning pipeline where TF-IDF converts text into numerical vectors by weighting terms based on their frequency and importance across documents, and SVM classifies these vectors by finding an optimal hyperplane that separates different sentiment classes.
- Preprocessing steps like tokenization, stopword removal, and lemmatization.
- Comparison of both models on tricky examples to assess their handling of subtle contextual differences.

  
### Objectives:
1. Compare the performance of **BERT** with **TF-IDF** + **SVM** on sentiment analysis tasks.
2. Highlight the trade-offs between accuracy and computational resources for each approach.
3. Highlight model behavior on nuanced examples

---

## Dataset
The dataset contains English-language text data labeled with binary sentiment.

- **Origin Size**: 1,600,000 entries
- **Total Features** : 5


**After Preprocessing :**
*Reduce some unimportant feature and for computational resource efficiency and data balance.
- **Sample Size**: 160,000 entries.
- **Features**:
  - `text`: User-generated text data.
  - `target`: Binary sentiment labels (0 = Negative, 1 = Positive).

Source: [Sentiment140 Dataset from Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## Methodology
This methodology also serves as the step-by-step process I followed in the [notebook.ipynb](https://github.com/rendra7/LLM-BERT/blob/main/Modelling_BERT_Rendra.ipynb) file.
### I. **Import Dataset**
[Sentiment140 Dataset from Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
### II. **Exploration & Dataset Preparation** : 
    - Understanding data characteristics through visualization, descriptive statistics, and text analysis.
    - Dimensionality Data Reduction : reduce some unimportant feature and for computational resource efficiency and data balance.
    
### III. Implementation 
#### 1. **Text Representation and Classification with BERT**:
    - Fine-tuning a pretrained BERT model on the sentiment dataset.
    - Input preprocessing: tokenization and attention mask generation.
    - Data splitting : 80% for training, 10% for testing, 10% for validation. 
##### **Model classification using BERT**
    - Adding a classification layer for sentiment prediction.
    - Training the model.
    - Saving the model for reuse.
##### **Evaluation**
    - Evaluation metrics : Accuracy, precision, recall, F1-Score
    - Testing & Analysis : test with new sentence for analysis

#### 2. **Traditional NLP Methods**:
##### Text Cleaning
    - Removing URLs, symbols, stopwords, and applying lemmatization.
##### **Word Embedding**
    - Text vectorization using TF-IDF.
##### **Using SVM for Classification Task** :
    - Training an SVM classifier with a grid-search optimized linear kernel.
    - Saving the fine-tuned model for reuse.
##### **Evaluation**
    - Evaluation metrics : Accuracy, precision, recall, F1-Score
    - Testing & Analysis : test with new sentence for analysis
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
| The ticket prices are amazing—amazingly expensive. | Negative | Positif |

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



