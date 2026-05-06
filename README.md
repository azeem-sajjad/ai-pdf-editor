# 📄 AI PDF Editor & RAG Chatbot

An advanced AI-powered PDF assistant that allows users to:

- Chat with PDFs using natural language
- Perform semantic document search
- Edit PDFs with AI instructions
- Rewrite entire documents intelligently
- Preserve original formatting during edits
- Run fully locally with Ollama

Built with Python, Streamlit, FAISS, Sentence Transformers, and local LLMs.

---

# ✨ Features

## 💬 AI PDF Chat

Ask questions about uploaded PDFs using Retrieval-Augmented Generation (RAG).

Examples:
- "Summarize this contract"
- "What are the payment terms?"
- "Find termination clauses"
- "Who are the parties involved?"

The system retrieves the most relevant document chunks and sends them to the LLM as context.

---

## ✏️ AI PDF Editing

Edit documents using natural language instructions.

Examples:
- "Replace Contractor with Service Provider"
- "Rewrite the introduction professionally"
- "Change payment terms to Net-30"
- "Translate the document into French"

---

## 🎯 Precise Inline Editing

The application can:
- Find exact text inside PDFs
- Replace only targeted content
- Preserve:
  - formatting
  - layout
  - spacing
  - tables
  - images

Powered by PyMuPDF redaction and coordinate-based text replacement.

---

## 🔄 Full Document Rewrite

Supports complete AI-powered document rewriting with:
- professional formatting
- styled PDF generation
- intelligent restructuring

---

## 🔍 Semantic Search

Uses vector embeddings to search documents by meaning instead of keywords.

Powered by:
- Sentence Transformers
- FAISS vector database

---

## 🖥️ Modern Streamlit UI

- Real-time streaming responses
- Interactive PDF editing
- Download edited PDFs instantly
- Fully local workflow

---

# 🧠 System Architecture

```text
PDF Upload
    ↓
Text Extraction (PyMuPDF)
    ↓
Sentence Chunking
    ↓
Embeddings (Sentence Transformers)
    ↓
FAISS Vector Search
    ↓
Relevant Context Retrieval
    ↓
LLM (Ollama / Llama3)
    ↓
Answer Generation or Document Editing
```

---

# 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM Runtime | Ollama |
| Models | Llama 3 / Mistral / Gemma / Phi3 |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| PDF Processing | PyMuPDF |
| PDF Generation | ReportLab |
| Language | Python |

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/azeem-sajjad/ai-pdf-editor.git
cd ai-pdf-editor
```

---

## 2. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

Download Ollama:

https://ollama.com

Start Ollama:

```bash
ollama serve
```

---

## 5. Pull AI Model

```bash
ollama pull llama3
```

Optional supported models:
- llama3.2
- mistral
- gemma2
- phi3

---

## 6. Run Application

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

# 📁 Project Structure

```text
ai-pdf-editor/
│
├── app.py                 # Streamlit frontend
├── utils.py               # Core AI + PDF utilities
├── requirements.txt
├── README.md
├── .gitignore
│
├── PDF Processing
│   ├── text extraction
│   ├── inline editing
│   ├── rewrite engine
│   └── styled PDF export
│
├── RAG Pipeline
│   ├── chunking
│   ├── embeddings
│   ├── FAISS indexing
│   └── semantic retrieval
```

---

# 🔒 Privacy

This application runs fully locally.

Your:
- PDFs
- embeddings
- vector database
- AI processing

remain on your machine.

No cloud upload required.

---

# ⚡ Supported Capabilities

- PDF Question Answering
- AI Document Editing
- Semantic Search
- Local LLM Inference
- Vector Search
- PDF Rewriting
- Styled PDF Export
- Real-Time AI Streaming

---

# 🗺️ Roadmap

- [ ] Multi-PDF support
- [ ] Chat memory
- [ ] DOCX support
- [ ] Source citations
- [ ] OCR support
- [ ] Cloud deployment
- [ ] Team collaboration
- [ ] Highlight edited sections visually

---

# 👨‍💻 Author

Azeem Sajjad

---

# ⭐ Acknowledgements

Built using:
- Streamlit
- Ollama
- Sentence Transformers
- FAISS
- PyMuPDF
- ReportLab
