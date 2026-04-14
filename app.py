import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model.pkl")

st.title("🚀 Social Media Virality Predictor")

# -----------------------------
# INPUTS
# -----------------------------
likes = st.slider("Likes", 0, 1000000, 1000)
views = st.slider("Views", 1, 1000000, 10000)
shares = st.slider("Shares", 0, 100000, 500)
comments = st.slider("Comments", 0, 50000, 300)
sentiment_score = st.slider("Sentiment Score", -1.0, 1.0, 0.0)

platform = st.selectbox("Platform", ["Instagram", "TikTok", "X", "YouTube Shorts"])
content_type = st.selectbox("Content Type", ["image", "video", "text"])
topic = st.selectbox("Topic", ["Sports", "Technology", "Entertainment", "Politics", "Lifestyle"])

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
views_log = np.log1p(views)
engagement_rate = (likes + shares + comments) / views

# -----------------------------
# CREATE INPUT DATAFRAME
# -----------------------------
input_data = pd.DataFrame({
    "views_log": [views_log],
    "likes": [likes],
    "shares": [shares],
    "comments": [comments],
    "engagement_rate": [engagement_rate],
    "sentiment_score": [sentiment_score],
    "platform": [platform],
    "content_type": [content_type],
    "topic": [topic]
})

# -----------------------------
# ENCODING
# -----------------------------
input_encoded = pd.get_dummies(input_data)

# Match training columns
for col in model.feature_names_in_:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[model.feature_names_in_]

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Predict Virality"):

    prob = model.predict_proba(input_encoded)[0][1]

    # Adjust threshold
    if prob > 0.6:
        st.success(f"🔥 High Viral Potential ({prob:.2%})")
    else:
        st.warning(f"⚠️ Low Viral Potential ({prob:.2%})")

    # -----------------------------
    # RECOMMENDATION SYSTEM (IMPORTANT)
    # -----------------------------
    st.subheader("📊 Recommendations")

    if engagement_rate > 0.15:
        st.write("✅ Strong engagement — continue similar content strategy")

    elif engagement_rate > 0.08:
        st.write("⚠️ Moderate engagement — improve hooks or visuals")

    else:
        st.write("❌ Low engagement — rethink content strategy")

    if sentiment_score > 0.3:
        st.write("😊 Positive sentiment works well — focus on emotional content")

    elif sentiment_score < -0.2:
        st.write("⚠️ Negative sentiment detected — may reduce virality")

    else:
        st.write("😐 Neutral sentiment — try more engaging captions")

    if shares < likes * 0.1:
        st.write("📢 Increase shareability — add CTA like 'Share with friends'")

    if comments < likes * 0.05:
        st.write("💬 Low comments — ask questions in captions")

    # -----------------------------
    # FEATURE IMPORTANCE (INTERVIEW GOLD)
    # -----------------------------
    st.subheader("🧠 What Influenced This Prediction")

    try:
        importances = model.feature_importances_
        feature_names = model.feature_names_in_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(8)

        st.bar_chart(importance_df.set_index("Feature"))

    except:
        st.write("Feature importance not available for this model")

    # -----------------------------
    # DEBUG INFO
    # -----------------------------
    st.subheader("🔍 Debug Info")
    st.write(f"Views (log): {views_log:.2f}")
    st.write(f"Engagement Rate: {engagement_rate:.4f}")
    st.write(f"Raw Probability: {prob:.4f}")