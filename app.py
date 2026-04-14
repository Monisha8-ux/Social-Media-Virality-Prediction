import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Virality Predictor", layout="wide")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# -------------------------------
# TITLE
# -------------------------------
st.title("Social Media Virality Predictor")

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("Enter Post Details")

col1, col2, col3 = st.columns(3)

with col1:
    platform = st.selectbox("Platform", ["Instagram", "TikTok", "YouTube Shorts", "X"])

with col2:
    content_type = st.selectbox("Content Type", ["image", "video", "text"])

with col3:
    topic = st.selectbox("Topic", ["Entertainment", "Lifestyle", "Politics", "Sports", "Technology"])

st.markdown("---")

# -------------------------------
# SLIDERS
# -------------------------------
st.subheader("Engagement Metrics")

likes = st.slider("Likes", 0, 500000, 1000)
views = st.slider("Views", 1, 2000000, 10000)
shares = st.slider("Shares", 0, 100000, 500)
comments = st.slider("Comments", 0, 100000, 300)
sentiment = st.slider("Sentiment Score", -1.0, 1.0, 0.0)

st.markdown("---")
if likes > views:
    st.error("Likes cannot be greater than Views")
    st.stop()

if shares > views:
    st.error("Shares cannot be greater than Views")
    st.stop()

if comments > views:
    st.error("Comments cannot be greater than Views")
    st.stop()
# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Virality"):

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    views_log = np.log1p(views)

    engagement_rate = (likes + shares + comments) / (views + 1)

    # 🔥 IMPORTANT: match training range
    engagement_rate = min(engagement_rate, 0.2)

    # -------------------------------
    # CREATE INPUT DATAFRAME
    # -------------------------------
    input_data = pd.DataFrame({
        "views_log": [views_log],
        "likes": [likes],
        "shares": [shares],
        "comments": [comments],
        "engagement_rate": [engagement_rate],
        "sentiment_score": [sentiment],
        "platform": [platform],
        "content_type": [content_type],
        "topic": [topic]
    })

    # -------------------------------
    # ENCODING
    # -------------------------------
    input_encoded = pd.get_dummies(input_data)

    # Add missing columns
    for col in model.feature_names_in_:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder
    input_encoded = input_encoded[model.feature_names_in_]

    # -------------------------------
    # PREDICTION
    # -------------------------------
    y_prob = model.predict_proba(input_encoded)[0][1]

    # -------------------------------
    # OUTPUT
    # -------------------------------
    st.markdown("---")

    if y_prob > 0.7:
        st.success(f"High Viral Potential\n\nEstimated Chance: **{round(y_prob*100,2)}%**")

    elif y_prob > 0.4:
        st.info(f"Moderate Viral Potential\n\nEstimated Chance: **{round(y_prob*100,2)}%**")

    else:
        st.warning(f"Low Viral Potential\n\nEstimated Chance: **{round(y_prob*100,2)}%**")

    st.progress(int(y_prob * 100))

    # -------------------------------
    # DEBUG SECTION (VERY IMPORTANT)
    # -------------------------------
    with st.expander("Debug Info"):
        st.write("Views (log):", views_log)
        st.write("Engagement Rate:", engagement_rate)
        st.write("Raw Probability:", y_prob)
        st.write("Final Input:")
        st.dataframe(input_encoded)