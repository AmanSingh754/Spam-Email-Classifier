#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import streamlit as st

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['Category', 'Message']
data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['Message'],
    data['Spam'],
    test_size=0.25,
    random_state=42
)

# Build and train model
clf = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier")
st.write("Paste an email or message below to check if it’s Spam or Safe (Ham).")

user_input = st.text_area("Enter your email or message here:")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        prediction = clf.predict([user_input])[0]
        probability = clf.predict_proba([user_input])[0][1]

        if prediction == 1:
            st.error(f"🚨 This looks like **Spam!** (Confidence: {probability*100:.2f}%)")
        else:
            st.success(f"✅ This looks like a **Safe Email (Ham).** (Confidence: {(1 - probability)*100:.2f}%)")

# Display model accuracy
st.write("---")
accuracy = clf.score(X_test, y_test) * 100
st.write(f"**Model Accuracy:** {accuracy:.2f}%")
st.caption("Model: TF-IDF + Multinomial Naive Bayes")
