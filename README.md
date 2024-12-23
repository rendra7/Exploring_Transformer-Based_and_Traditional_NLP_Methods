# Exploring LLM Capabilities in Sentiment Analysis: BERT vs TF-IDF + SVM

This project presents a comprehensive comparison of sentiment analysis approaches using two distinct methods:
1. Large language Model (LLM) approach with **BERT**.
2. Traditional model or Frequency-based approach with **TF-IDF** and **Support Vector Machine (SVM)**.

The study evaluates the performance of these models on sentiment analysis tasks, specifically focusing on Twitter comments. The results demonstrate the strengths and weaknesses of each approach in terms of accuracy, precision, recall, and F1-score.

### Project By : 
Rendra Dwi Prasetyo

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)

---

## Project Overview
### Key Features:
- **BERT**: A (Large language Model) pretrained transformer-based language model that captures context and semantic relationships in text.
- **TF-IDF + SVM**: A traditional machine learning approach leveraging word frequency for classification.
- Comparison of both models on tricky examples to assess their handling of subtle contextual differences.
- Preprocessing steps like tokenization, stopword removal, and lemmatization.
  
### This project aims to:
1. Compare **BERT** as a large language model with a traditional frequency-based model, **TF-IDF** + SVM.
2. Analyze the sentiment of tweets.
3. Highlight the trade-offs between accuracy and computational resources for each approach.

The project uses labeled sentiment data to train and evaluate the models, presenting an in-depth performance analysis.

---

## Dataset
The dataset contains English-language text data labeled with binary sentiment (0 = Negative, 1 = Positive).

- **Sample Size**: 160,000 entries.
- **Features**:
  - `text`: User-generated text data.
  - `target`: Binary sentiment labels (0 = Negative, 1 = Positive).

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


### 1. Contextual-Based Sentiment Analysis (BERT)

#### Workflow:
- Pretrained BERT model is fine-tuned on the sentiment dataset.
- Saved the model as a pickle file for efficient reuse.
- Prediction includes tokenization and attention mask generation for input text.

#### Example Predictions:
| Text | Predicted Sentiment |
|------|---------------------|
| This is a great product! | Positive |
| I'm really great disappointed with this product. | Negative |
| The service was amazing; I’m so satisfied and will definitely come back. | Positive |
| The ticket prices are amazing—amazingly expensive. | Negative |

### 2. Frequency-Based Sentiment Analysis (TF-IDF + SVM)

#### Workflow:
- Text preprocessing: removal of URLs, symbols, stopwords, and applying lemmatization.
- Vectorization using TF-IDF.
- Classification using SVM with linear kernel.
- Evaluation through accuracy and confusion matrix.


#### Example Predictions:
| Text | BERT Sentiment | SVM Sentiment |
|------|----------------|---------------|
| This is a great product! | Positive | Positive |
| I'm really great disappointed with this product. | Negative | Negative |
| The service was amazing; I’m so satisfied and will definitely come back. | Positive | Positive |
| The ticket prices are amazing—amazingly expensive. | Negative | Negative |

### Key Difference from example prediction:
- BERT successfully understands tricky contextual meanings (e.g., "amazingly expensive").
- SVM relies more on word frequency, which may lead to misclassifications for nuanced text.
---

## Results
| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| BERT           | 83%      | 83%       | 83%    | 83%      |
| TF-IDF + SVM   | 76%      | 76%       | 76%    | 76%      |

- **BERT** outperforms **TF-IDF + SVM** in terms of all evaluation metrics.

---

## Conclusion 
### Results and Insights
#### Strengths of BERT:
- Captures context and nuances in text, making it more robust for complex statements.
- Better suited for data with subtle sentiment cues.
#### Strengths of TF-IDF + SVM:
- Simpler to train.
- Effective for datasets with clear and straightforward sentiment patterns.
### Overall Comparison:
BERT outperforms traditional methods on tricky examples but requires more computational resources.



