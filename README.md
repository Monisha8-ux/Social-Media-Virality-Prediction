# 📊 Social Media Virality Prediction & Analytics Dashboard

## 🚀 Project Overview

This project analyzes social media content and predicts whether a post will go viral using Machine Learning and Natural Language Processing (NLP).

Instead of relying on assumptions, the system uses engagement patterns, content characteristics, and sentiment signals to make data-driven predictions about virality.

---

## 🎯 Objectives

* Analyze social media engagement data
* Perform sentiment analysis using NLP
* Build a machine learning model for virality prediction
* Avoid data leakage for realistic predictions
* Provide interactive insights through a dashboard

---

## 🧠 Key Features

* 📌 Data Cleaning & Preprocessing
* 📊 Feature Engineering (engagement rate, log views, etc.)
* 🧠 Sentiment Analysis using TextBlob
* 🤖 Machine Learning Model (Random Forest)
* 📉 Model Evaluation (Precision, Recall, F1-score)
* 📊 Interactive Dashboard using Streamlit

---

## ⚠️ Important Note on Accuracy

The dataset is **highly imbalanced**, where non-viral posts dominate.

A naive model can achieve ~99% accuracy simply by predicting all posts as non-viral.
Therefore, **accuracy alone is not a reliable metric for this problem**.

---

## 📊 Model Performance

### ✔ Balanced Model

* Provides more realistic predictions
* Improves detection of viral posts

### ✔ Tuned Model

* Better recall for viral class
* Trade-off: Slight decrease in non-viral prediction accuracy

### 📌 Key Focus

* Recall for viral posts (important due to rarity)
* F1-score for balanced evaluation
* Avoiding misleading results due to data leakage

---

## 🔍 Key Insights

* Engagement rate (likes, shares, comments relative to views) is the strongest predictor
* Content type and platform have moderate influence
* Sentiment has a smaller but noticeable impact
* There is a clear trade-off between accuracy and class balance

---

## 🖥️ Dashboard Features

* 📌 User Input for Post Simulation
* 📊 Real-time Virality Prediction
* 🔍 Debug View (shows model inputs & probability)
* 📈 Engagement-based analysis
* ⚠️ Input validation to prevent unrealistic values

---

## 🛠️ Tech Stack

* **Python** (Pandas, NumPy)
* **Machine Learning:** Scikit-learn
* **NLP:** TextBlob
* **Visualization:** Matplotlib, Seaborn
* **Dashboard:** Streamlit

---

## 📂 Project Structure

```
Social-Media-Virality-Prediction/
│
├── app.py
├── model.pkl
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

👉 https://dnecessaqukpq7bpoevgrz.streamlit.app/

---

## 💡 Conclusion

This project demonstrates how machine learning can be applied to analyze social media trends and predict virality.

It also highlights real-world challenges such as:

* Class imbalance
* Data leakage
* Trade-offs between accuracy and recall

The focus on realistic evaluation makes this a practical and industry-relevant implementation.

---

## 👩‍💻 Author

**Monisha Sharma**
