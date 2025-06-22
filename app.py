# app.py
import os
from flask import Flask, request, render_template, redirect, flash, send_file
from werkzeug.utils import secure_filename
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests
from PyPDF2 import PdfReader
import docx
import json
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model_path = "legalbert-cuad"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Setup GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(filepath):
    ext = filepath.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif ext == 'docx':
        doc = docx.Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
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
    try:
        if not isinstance(text, str) or not text.strip():
            return "Gemini failed: Empty or invalid input."

        text = text[:4000]

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
            return f"Gemini failed: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def generate_label_analyses(labels):
    try:
        if not labels:
            return []

        prompt = "\n".join([f"Explain the '{label}' clause in a legal contract." for label in labels])
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            params={"key": GOOGLE_API_KEY},
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
            filtered = []
            for label, analysis in zip(labels, raw_analyses):
                analysis = analysis.strip()
                if not analysis or "Gemini failed" in analysis or "Error" in analysis:
                    continue
                filtered.append((label, analysis))
            return filtered
        else:
            return []
    except Exception as e:
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            flash("Upload a valid PDF, DOCX, or TXT file.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        text = extract_text(filepath)
        if not text.strip():
            flash("No text found in document.")
            return redirect(request.url)

        labels = predict_labels(text)
        summary = generate_summary(text)
        analyses = generate_label_analyses(labels)

        # Save JSON output
        output_json = {
            "filename": filename,
            "summary": summary,
            "clauses": [{"label": lbl, "analysis": desc} for lbl, desc in analyses]
        }
        json_path = os.path.join("uploads", f"{filename}_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        return render_template("result.html", analyses=analyses, summary=summary, filename=filename, json_file=json_path)

    return render_template("index.html")

@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join("uploads", filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
