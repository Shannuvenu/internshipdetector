import pandas as pd
import nltk
import string
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Download NLTK
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset/fake_job_postings.csv")
df.fillna("", inplace=True)

# Combine text
df["text"] = df["title"] + " " + df["description"] + " " + df["requirements"] + " " + df["company_profile"]

# Preprocess
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(preprocess)

X = df["clean_text"]
y = df["fraudulent"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save
with open("model/naive_bayes_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved!")