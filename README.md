# 📊 Social Media Virality Prediction & Analytics Dashboard

## 🚀 Project Overview

This project focuses on analyzing social media content and predicting whether a post will go **viral or not** using Machine Learning and Natural Language Processing (NLP).

It replaces guesswork with a **data-driven approach** by analyzing engagement patterns, content types, and sentiment to understand what drives virality.

---

## 🎯 Objectives

* Analyze social media engagement data
* Perform sentiment analysis using NLP
* Build a machine learning model for virality prediction
* Avoid data leakage for realistic predictions
* Visualize insights using an interactive dashboard

---

## 🧠 Key Features

* 📌 Data Cleaning & Preprocessing
* 📊 Feature Engineering
* 🧠 Sentiment Analysis (TextBlob)
* 🤖 Machine Learning Model (Random Forest)
* 📉 Model Evaluation (Accuracy, Precision, Recall)
* 📊 Interactive Dashboard (Streamlit)

---

## 📊 Model Performance

### ✔ Balanced Model

* Accuracy: **0.54**
* Provides realistic predictions without data leakage

### ✔ Tuned Model

* Accuracy: **0.65**
* Improved viral detection using threshold tuning
* Trade-off: Reduced performance for non-viral class

---

## 🔍 Key Insights

* Content type and platform have moderate impact on virality
* Sentiment shows slight correlation with engagement
* Removing engagement features prevents misleading results
* There is a trade-off between accuracy and class balance

---

## 🖥️ Dashboard Features

* 📌 Dataset Overview
* 📊 Sentiment Analysis (Pie Chart)
* 📱 Platform vs Virality Analysis
* 🎥 Content Type Performance
* 🤖 Virality Prediction (Interactive)
* 📈 Model Performance Summary

---

## 🛠️ Tech Stack

* Python (Pandas, NumPy)
* Machine Learning: Scikit-learn
* NLP: TextBlob
* Visualization: Matplotlib, Seaborn
* Dashboard: Streamlit

---

## 📂 Project Structure

```
Social-Media-Virality-Prediction/
│
├── app.py
├── final_processed_dataset.csv
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run Streamlit app

```
streamlit run app.py
```

---

## 🌐 Live Demo

👉(https://dnecessaqukpq7bpoevgrz.streamlit.app/)

---

## 💡 Conclusion

This project demonstrates how machine learning and NLP can be used to analyze social media trends and predict virality. It highlights real-world challenges such as **data leakage, class imbalance, and performance trade-offs**, making it a practical data science implementation.

---

## 👩‍💻 Author

**Monisha Sharma**
