import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 🔹 Page Config
# -------------------------
st.set_page_config(page_title="Virality Dashboard", layout="wide")

st.title("Social Media Virality Dashboard")

# -------------------------
# 🔹 Load Dataset
# -------------------------
df = pd.read_csv("final_processed_dataset.csv")

# -------------------------
# 🔹 1. Dataset Overview
# -------------------------
st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Posts", len(df))
col2.metric("Viral Posts", int(df["is_viral"].sum()))
col3.metric("Non-Viral Posts", int(len(df) - df["is_viral"].sum()))

# -------------------------
# 🔹 2. Charts Section (Side-by-Side)
# -------------------------
st.header("Insights")

col1, col2 = st.columns(2)

# -------------------------
# 📌 Sentiment Pie Chart
# -------------------------
with col1:
    st.subheader("Sentiment Analysis")

    sentiment_counts = df["predicted_sentiment"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(4,4))

    ax1.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        textprops={'fontsize': 10}
    )

    st.pyplot(fig1)

# -------------------------
# 📌 Platform Bar Chart
# -------------------------
with col2:
    st.subheader("Platform vs Virality")

    platform_data = df.groupby("platform")["is_viral"].mean()

    fig2, ax2 = plt.subplots(figsize=(5,3))

    platform_data.plot(kind='bar', ax=ax2)

    ax2.set_ylabel("Virality Rate")
    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig2)

# -------------------------
# 🔹 3. Content Type Analysis
# -------------------------
st.subheader("Content Type Performance")

content_data = df.groupby("content_type")["is_viral"].mean()

fig3, ax3 = plt.subplots(figsize=(5,3))

content_data.plot(kind='bar', ax=ax3)

ax3.set_ylabel("Virality Rate")
ax3.set_xlabel("Content Type")

plt.xticks(rotation=30)
plt.tight_layout()

st.pyplot(fig3, use_container_width=False)

# -------------------------
# 🔹 4. Virality Prediction (Demo)
# -------------------------
st.header("Virality Prediction")

platform = st.selectbox("Select Platform", df["platform"].unique())
topic = st.selectbox("Select Topic", df["topic"].unique())
content_type = st.selectbox("Select Content Type", df["content_type"].unique())

if st.button("Predict"):
    
    # Simple demo logic
    if platform == "Instagram" and content_type == "video":
        result = "Viral"
    else:
        result = "Not Viral"
    
    st.success(f"Prediction: {result}")

# -------------------------
# 🔹 5. Model Performance
# -------------------------
st.header("Model Performance")

st.write("✔ Balanced Model Accuracy: 0.54")
st.write("✔ Improved Accuracy (Threshold Tuning): 0.65")

st.info("Note: Accuracy improved using threshold tuning, but with trade-off in class balance.")