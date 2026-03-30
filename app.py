import streamlit as st
import pickle
import os
import nltk
import string
import re
from PIL import Image
import PyPDF2

# ---------- SAFE OCR ----------
try:
    import pytesseract
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Job Detector", layout="wide")

MODEL_PATH = "model/naive_bayes_model.pkl"
VEC_PATH = "model/tfidf_vectorizer.pkl"

# ---------- LOAD MODEL ----------
if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
    st.warning("Training model... please wait")
    import train_model

model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VEC_PATH, "rb"))

# ---------- NLP ----------
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join([stemmer.stem(w) for w in text.split() if w not in stop_words])

# ---------- VALIDATION ----------
def is_meaningful_text(text):
    words = text.split()
    if len(words) < 5:
        return False
    return len(set(words)) / len(words) > 0.3

# ---------- RULE BASE ----------
scam_signals = [
    "no experience", "easy money", "pay fee",
    "registration fee", "earn money", "work from home",
    "limited slots", "urgent hiring", "guaranteed job"
]

def rule_check(text):
    text = text.lower()
    return [s for s in scam_signals if s in text]

# ---------- EMAIL ----------
def detect_email(text):
    emails = list(set(re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", text)))
    suspicious = []

    for email in emails:
        domain = email.split("@")[1]

        if domain in ["gmail.com", "yahoo.com", "hotmail.com"]:
            suspicious.append(f"{email} (Free domain)")

        if not any(x in email.lower() for x in ["hr", "career", "jobs"]):
            suspicious.append(f"{email} (Unprofessional)")

    return emails, list(set(suspicious))

# ---------- LINKS ----------
def detect_links(text):
    urls = re.findall(r"https?://\S+|www\.\S+", text)
    suspicious = [u for u in urls if any(x in u for x in [".xyz",".click","bit.ly"])]
    return urls, suspicious

# ---------- SCORE ----------
def calculate_score(matches, prob_fake, email_susp, link_susp):
    score = 0
    score += min(len(matches) * 15, 40)
    score += int(prob_fake * 40)

    if email_susp:
        score += 10
    if link_susp:
        score += 10

    return min(score, 100)

# ---------- EXPLANATION ----------
def explain(score):
    if score > 80:
        return "🚨 Strong scam indicators"
    elif score > 50:
        return "⚠ Moderate risk"
    else:
        return "✅ Low risk"

# ---------- HIGHLIGHT ----------
def highlight(text, matches):
    for m in matches:
        text = re.sub(m, f"🔴{m.upper()}🔴", text, flags=re.IGNORECASE)
    return text

# ---------- KEYWORDS ----------
def get_keywords(vector):
    feature_names = vectorizer.get_feature_names_out()
    scores = vector.toarray()[0]
    top = scores.argsort()[-10:][::-1]
    return [feature_names[i] for i in top if scores[i] > 0]

# ---------- TEXT EXTRACTION ----------
def extract_text(file):
    try:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            return " ".join([p.extract_text() or "" for p in reader.pages])

        elif "image" in file.type:
            if OCR_AVAILABLE:
                return pytesseract.image_to_string(Image.open(file))
            else:
                return ""

        elif file.type == "text/plain":
            return file.read().decode("utf-8")

    except:
        return ""

    return ""

# ---------- UI ----------
st.markdown("""
<style>
body {background: #020617; color: white;}
.card {
    background: #0f172a;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.result-good {background:#064e3b;padding:15px;border-radius:10px;}
.result-bad {background:#7f1d1d;padding:15px;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.title("🕵️ AI Fake Job & Document Detector")

col1, col2 = st.columns([2,1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    text_input = st.text_area("Paste job description or resume", height=200)
    analyze = st.button("Analyze")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Tips")
    st.write("• Avoid paying fees")
    st.write("• Check official emails")
    st.write("• Verify company website")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- ANALYSIS ----------
def run_analysis(text):
    matches = rule_check(text)
    clean = preprocess(text)

    vector = vectorizer.transform([clean])
    prob = model.predict_proba(vector)[0][1]

    if len(matches) >= 3:
        prob = max(prob, 0.85)

    emails, email_susp = detect_email(text)
    links, link_susp = detect_links(text)

    score = calculate_score(matches, prob, email_susp, link_susp)

    return score, matches, email_susp, link_susp, vector

# ---------- TEXT ANALYSIS ----------
if analyze:

    if len(text_input) < 50 or not is_meaningful_text(text_input):
        st.warning("Enter meaningful text (min 50 chars)")

    else:
        score, matches, email_susp, link_susp, vector = run_analysis(text_input)

        if score > 60:
            st.markdown(f'<div class="result-bad">🚨 FAKE | Risk: {score}/100</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-good">✅ REAL | Risk: {score}/100</div>', unsafe_allow_html=True)

        st.progress(score)
        st.info(explain(score))

        if matches:
            st.subheader("Suspicious phrases")
            st.write(matches)

        if email_susp:
            st.subheader("Email issues")
            st.write(email_susp)

        if link_susp:
            st.subheader("Suspicious links")
            st.write(link_susp)

        st.subheader("Highlighted Text")
        st.write(highlight(text_input, matches))

        st.subheader("Keywords")
        st.write(get_keywords(vector))

# ---------- FILE ----------
st.subheader("Upload Document")

uploaded = st.file_uploader("Upload file", type=["pdf","png","jpg","jpeg","txt"])

if uploaded:
    text = extract_text(uploaded)

    if len(text.strip()) < 50:
        st.warning("Not enough readable content")

    else:
        score, matches, email_susp, link_susp, vector = run_analysis(text)

        if score > 60:
            st.markdown(f'<div class="result-bad">🚨 FAKE DOCUMENT | Risk: {score}/100</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-good">✅ REAL DOCUMENT | Risk: {score}/100</div>', unsafe_allow_html=True)

        st.progress(score)
        st.info(explain(score))

        if email_susp:
            st.write(email_susp)

        if link_susp:
            st.write(link_susp)

        st.subheader("Preview")
        st.write(highlight(text[:1500], matches))

        st.subheader("Keywords")
        st.write(get_keywords(vector))     