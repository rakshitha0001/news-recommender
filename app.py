from flask import Flask, render_template, request
import pandas as pd
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize
app = Flask(__name__)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load dataset
df = pd.read_csv("india-news-small.csv")

# Combine columns
df['content'] = df['headline_text'] + " " + df['headline_category']

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

df['clean_content'] = df['content'].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df['clean_content'])

# ROUTES
@app.route("/", methods=["GET", "POST"])
def home():
    results = []

    if request.method == "POST":
        user_input = request.form["news"]

        user_input = clean_text(user_input)
        user_vector = tfidf.transform([user_input])

        similarity_scores = linear_kernel(user_vector, tfidf_matrix).flatten()
        scores = list(enumerate(similarity_scores))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        top_articles = scores[:5]

        for i in top_articles:
            results.append({
                "headline": df.iloc[i[0]]['headline_text'],
                "score": round(i[1], 3)
            })

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)