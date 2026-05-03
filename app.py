from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load dataset
df = pd.read_csv("india-news-small.csv")

# Combine text columns
df['content'] = df['headline_text'] + " " + df['headline_category']

# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

df['clean_content'] = df['content'].apply(clean_text)

# -------------------------------
# TF-IDF MODEL (existing)
# -------------------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df['clean_content'])

# -------------------------------
# WORD2VEC MODEL (NEW)
# -------------------------------

# Tokenize
tokenized_data = df['clean_content'].apply(lambda x: x.split())

# Train Word2Vec
w2v_model = Word2Vec(
    sentences=tokenized_data,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# Sentence to vector
def get_sentence_vector(sentence):
    words = sentence.split()
    vectors = []

    for word in words:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])

    if len(vectors) == 0:
        return np.zeros(100)

    return np.mean(vectors, axis=0)

# Store vectors
df['w2v_vector'] = df['clean_content'].apply(get_sentence_vector)

# -------------------------------
# RECOMMENDATION FUNCTION (Word2Vec based)
# -------------------------------
def recommend_news(user_input):
    user_input = clean_text(user_input)
    user_vec = get_sentence_vector(user_input)

    similarities = []

    for i in range(len(df)):

        sim = cosine_similarity(
            [user_vec],
            [df.iloc[i]['w2v_vector']]
        )[0][0]

        # Score filter
        if sim > 0.2:
            similarities.append((i, sim))

    # Sort results
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Top 5
    top_articles = similarities[:5]

    # Handle no results
    if len(top_articles) == 0:
        return [{"headline": "No relevant news found", "score": 0}]

    results = []
    for i in top_articles:
        results.append({
            "headline": df.iloc[i[0]]['headline_text'],
            "score": round(i[1], 3)
        })

    return results
# -------------------------------
# FLASK ROUTE
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    results = []

    if request.method == "POST":
        user_input = request.form["news"] 
        results = recommend_news(user_input)
    return render_template("index.html", results=results)

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)