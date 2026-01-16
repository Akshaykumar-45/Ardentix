# Ardentix
# Sentiment Analysis System  
### AI/ML Engineer Intern – Technical Assignment (Ardentix)

## 📌 Project Overview
This project implements an **end-to-end Sentiment Analysis system** that classifies movie reviews as **Positive** or **Negative** using Machine Learning and Natural Language Processing (NLP).

The model was **developed and trained in Google Colab** and later **deployed as a live web application using Streamlit**, allowing real-time sentiment prediction.

---

## 🎯 Objective
To build a machine learning pipeline that:
- Takes raw text as input
- Cleans and preprocesses the text
- Converts text into numerical features using **TF-IDF**
- Trains and evaluates multiple ML models
- Deploys the best-performing model as a web application

---

## 📊 Dataset
**IMDB Movie Reviews Dataset**
- 50,000 labeled reviews
- Classes: Positive, Negative
- Publicly available dataset

---

## 🧠 Machine Learning Pipeline
1. Data loading and exploration  
2. Text preprocessing (lowercasing, punctuation removal, stopword removal)  
3. Feature extraction using **TF-IDF**  
4. Model training and comparison  
   - Naive Bayes  
   - Logistic Regression  
   - Support Vector Machine (SVM)  
5. Model evaluation using accuracy, precision, recall, and F1-score  
6. Selection of the best-performing model (SVM)  
7. Deployment using Streamlit  

---

## 🤖 Model Selection
**Support Vector Machine (SVM)** was selected as the final model because it achieved the **highest accuracy** during comparison with other models.

**Logistic Regression** was used as a strong baseline model for validating the feature extraction and preprocessing pipeline.

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 🖥 Live Web Application
The trained model is deployed using **Streamlit Community Cloud**.

🔗 **Live App Link:**  
👉 *Add your Streamlit app URL here*

---

## 📒 Google Colab Notebook
Model training and experimentation were performed in Google Colab.

🔗 **Colab Notebook Link:**  
👉 *Paste your Google Colab link here*

*(Make sure the Colab file is set to “Anyone with the link → Viewer”)*

---

## 🚀 How to Run the Project Locally

### 1️⃣ Install Dependencies
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn joblib streamlit
