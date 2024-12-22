# LLM-BERT
# Exploring LLM Capabilities in Sentiment Analysis: BERT vs TF-IDF + SVM

This project presents a comprehensive comparison of sentiment analysis approaches using two distinct methods:
1. Large language Model (LLM) approach with **BERT**.
2. Traditional model or Frequency-based approach with **TF-IDF** and **Support Vector Machine (SVM)**.

The study evaluates the performance of these models on sentiment analysis tasks, specifically focusing on Twitter comments. The results demonstrate the strengths and weaknesses of each approach in terms of accuracy, precision, recall, and F1-score.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)

---

## Project Overview
This project aims to:
1. Compare **BERT** as a large language model with a traditional frequency-based model, **TF-IDF** + SVM.
2. Analyze the sentiment of tweets.
3. Highlight the trade-offs between accuracy and computational resources for each approach.

The project uses labeled sentiment data to train and evaluate the models, presenting an in-depth performance analysis.

---

## Dataset
The dataset consists of Twitter comments, labeled with sentiment categories:
- Positive
- Negative

Source: [Kaggle Sentiment Analysis Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## Methodology
### 1. **BERT-based Model**:
   - Pre-trained BERT model fine-tuned for sentiment analysis.
   - Features extracted using the Transformers library.
   - Classification layer added for predicting sentiment.

### 2. **TF-IDF + SVM**:
   - Text features extracted using TF-IDF vectorization.
   - SVM classifier trained with a grid-search optimized linear kernel.

---

## Results
| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| BERT           | 83%      | 83%       | 83%    | 83%      |
| TF-IDF + SVM   | 76%      | 76%       | 76%    | 76%      |

- **BERT** outperforms TF-IDF + SVM in terms of accuracy.
- **TF-IDF + SVM** has lower computational requirements, making it suitable for scenarios with limited resources.

---

# Project By :
Rendra Dwi Prasetyo

