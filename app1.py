import streamlit as st
import os
import docx
import json
import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

model_name = "Irfanshareef05/legal-contract-label"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

st.set_page_config(page_title="Legal Contract Analyzer", layout="wide")
st.title("📄 Legal Contract Analyzer")
st.markdown("Upload a legal document to extract and explain key clauses using AI.")

# Helper Functions

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    return ""

def predict_labels(text):
    max_tokens = 512
    words = text.split()
    chunks = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

    label_counts = {}
    id2label = model.config.id2label

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
            predicted = (probs > 0.5).nonzero(as_tuple=True)[1].tolist()
            for idx in predicted:
                label = id2label[idx]
                label_counts[label] = label_counts.get(label, 0) + 1

    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    return [label for label, count in sorted_labels]

def generate_summary(text):
    api_key = st.secrets["GOOGLE_API_KEY"]
    if not api_key:
        return "Missing Gemini API key."

    text = text[:4000]
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            params={"key": api_key},
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
        return f"Gemini failed: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def generate_label_analyses(labels):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return []

    prompt = "\n".join([f"Explain the '{label}' clause in a legal contract." for label in labels])
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            params={"key": api_key},
            json={
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt}]
                }]
            },
            headers={"Content-Type": "application/json"},
            timeout=20
        )
        if response.status_code == 200:
            full_text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            raw_analyses = full_text.split("\n")
            return [(label, desc.strip()) for label, desc in zip(labels, raw_analyses)]
        return []
    except:
        return []

# Upload and Analyze
uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
if uploaded_file and st.button("Analyze Document"):
    text = extract_text_from_file(uploaded_file)
    if text:
        with st.spinner("Predicting clause labels..."):
            labels = predict_labels(text)
        with st.spinner("Generating summary..."):
            summary = generate_summary(text)
        with st.spinner("Analyzing clauses using Gemini..."):
            analyses = generate_label_analyses(labels)

        st.subheader("📋 Contract Summary")
        st.write(summary)

        # ✅ Only show clauses if analyses are returned and valid
        valid_analyses = [(label, explanation) for label, explanation in analyses if explanation.strip()]
        if valid_analyses:
            st.subheader("📑 Key Clauses")
            for label, explanation in valid_analyses:
                st.markdown(f"**{label}**")
                st.info(explanation)
        else:
            st.info("key clauses were extracted or explained from the document.")
    else:
        st.error("Could not extract text from the uploaded file.")
