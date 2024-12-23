
# Performance Comparison of Sentiment Analysis: BERT vs TF-IDF + SVM

This project presents a comprehensive comparison of sentiment analysis approaches using two distinct methods:
1. Large Language Model (LLM) approach with **BERT (Bidirectional Encoder Representations from Transformers)**.
2. A traditional frequency-based model using **TF-IDF** and **Support Vector Machine (SVM)**.

The study evaluates the performance of these models on sentiment analysis tasks, specifically focusing on Twitter comments. The results demonstrate the strengths and weaknesses of each approach in terms of accuracy, precision, recall, and F1-score.

---

## Table of Contents
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
- Comparison of both models on tricky examples to assess their handling of subtle contextual differences.
- Preprocessing steps like tokenization, stopword removal, and lemmatization.
  
### Objectives:
1. Compare the performance of **BERT** with **TF-IDF** + **SVM** on sentiment analysis tasks.
2. Highlight the trade-offs between accuracy and computational resources for each approach.
3. Highlight model behavior on nuanced examples

---

## Dataset
The dataset contains English-language text data labeled with binary sentiment (0 = Negative, 1 = Positive).

- **Sample Size**: 160,000 entries.
- **Features**:
  - `text`: User-generated text data.
  - `target`: Binary sentiment labels (0 = Negative, 1 = Positive).

Source: [Sentiment140 Dataset from Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## Methodology
### 1. **BERT-based Model**:
    - Fine-tuning a pretrained BERT model on the sentiment dataset.
    - Input preprocessing: tokenization and attention mask generation.
    - Adding a classification layer for sentiment prediction.
    - Saving the fine-tuned model for reuse.

### 2. **TF-IDF + SVM**:
    - Preprocessing: Removing URLs, symbols, stopwords, and applying lemmatization.
    - Text vectorization using TF-IDF.
    - Training an SVM classifier with a grid-search optimized linear kernel.

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
| The ticket prices are amazing—amazingly expensive. | Negative | Negative |

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
## Project By : 
Name : Rendra Dwi Prasetyo

NIM  : 2602199960


