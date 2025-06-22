import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from PyPDF2 import PdfReader
import docx
import os
import json
import requests

# Load secret API key from Streamlit secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Load model & tokenizer (CPU, float16, low mem)
@st.cache_resource
def load_model():
    model_name = "Irfanshareef05/legal-contract-label"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("cpu")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# File type checker
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {"pdf", "docx", "txt"}

# Text extractor
def extract_text(file):
    ext = file.name.rsplit('.', 1)[-1].lower()
    if ext == "pdf":
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == "docx":
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == "txt":
        return file.read().decode("utf-8")
    return ""

# Clause label predictor
def predict_labels(text):
    max_tokens = 256
    words = text.split()
    chunks = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

    label_counts = {}
    id2label = model.config.id2label

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=max_tokens)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
            predicted = (probs > 0.5).nonzero(as_tuple=True)[1].tolist()
            for idx in predicted:
                label = id2label[idx]
                label_counts[label] = label_counts.get(label, 0) + 1

    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    return [label for label, _ in sorted_labels]

# Gemini summary
def generate_summary(text):
    try:
        text = text[:3000]  # Trim for safety
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            params={"key": GOOGLE_API_KEY},
            json={
                "contents": [{
                    "role": "user",
                    "parts": [{"text": f"Summarize the following legal contract and highlight key clauses:\n{text}"}]
                }]
            },
            headers={"Content-Type": "application/json"},
            timeout=20
        )
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            return "Gemini summarization failed."
    except Exception as e:
        return f"Summary error: {str(e)}"

# Gemini label explanation
def generate_label_analyses(labels):
    try:
        if not labels:
            return []
        prompt = "\n".join([f"Explain the '{label}' clause in a legal contract." for label in labels])
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            params={"key": GOOGLE_API_KEY},
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
            headers={"Content-Type": "application/json"},
            timeout=20
        )
        if response.status_code == 200:
            full_text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            raw_analyses = full_text.split("\n")
            filtered = []
            for label, analysis in zip(labels, raw_analyses):
                if analysis.strip():
                    filtered.append((label, analysis.strip()))
            return filtered
        else:
            return []
    except Exception:
        return []

# Streamlit UI
st.set_page_config(page_title="Legal Clause Analyzer", layout="centered")
st.title("📄 Legal Contract Clause Analyzer")
st.write("Upload a legal contract to classify its clauses and get insights.")

uploaded_file = st.file_uploader("Upload a contract file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file and allowed_file(uploaded_file.name):
    with st.spinner("Extracting and analyzing clauses..."):
        text = extract_text(uploaded_file)
        if not text.strip():
            st.error("No readable text found.")
        else:
            labels = predict_labels(text)
            summary = generate_summary(text)
            analyses = generate_label_analyses(labels)

            st.subheader("🔍 Contract Summary")
            st.write(summary)

            st.subheader("📌 Detected Clause Labels")
            st.write(", ".join(labels) if labels else "No clauses detected.")

            if analyses:
                st.subheader("📘 Clause Explanations")
                for label, desc in analyses:
                    st.markdown(f"**{label}**")
                    st.write(desc)
            else:
                st.warning("No detailed clause explanations available.")
else:
    st.info("Please upload a valid legal document.")
