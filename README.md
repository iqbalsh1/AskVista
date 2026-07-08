# AskVista - Company Policy Assistant

A local, privacy-friendly **RAG (Retrieval-Augmented Generation)** chatbot that answers questions about a company policy document. Everything runs on your machine - no API keys, no data leaves your computer.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2-black)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Ask in plain English** - query a policy PDF conversationally (leave, reimbursement, working hours, etc.)
- **Fully local** - powered by Ollama's `llama3.2`, so no cloud API costs
- **Persistent vector store** - the PDF is embedded once into ChromaDB and reused across restarts
- **Grounded answers** - the model is instructed to say "I don't know" instead of hallucinating
- **Clean custom UI** - modern card-based Streamlit interface with an external stylesheet

## How it works

```
PDF ──▶ PyPDFLoader ──▶ Text splitter (500-char chunks)
                              │
                              ▼
                  mxbai-embed-large embeddings
                              │
                              ▼
                     ChromaDB (persistent)
                              │
                              ▼

        question ──▶ retriever ──▶ prompt ──▶ llama3.2 ──▶ answer
```

### Prerequisites

- Python **3.10+**
- [Ollama](https://ollama.com) installed and running

Pull the required models:

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/askvista.git
cd askvista

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Add your document

Place your policy PDF at:

```
data/company_policy.pdf
```

### Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. The first question triggers a one-time indexing of the PDF; subsequent runs load the existing ChromaDB store instantly.

## Project structure

```
askvista/
├── app.py               # Streamlit app + RAG pipeline
├── assets/
│   └── styles.css       # UI theme
│    └── demo1.png
│    └── demo2.png
│    └── demo3.png
├── data/
│   └── company_policy.pdf  
├── requirements.txt     # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Configuration

All tunable values live at the top of `app.py`:

| Constant | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Ollama chat model |
| `EMBEDDING_MODEL` | `mxbai-embed-large` | Ollama embedding model |
| `CHUNK_SIZE` | `500` | Characters per document chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `PDF_PATH` | `data/company_policy.pdf` | Source document |

> **Tip:** if you swap in a new PDF, delete the `chromaDB/` folder so the index is rebuilt.

## Tech stack

Streamlit · LangChain · Ollama (llama3.2, mxbai-embed-large) · ChromaDB · PyPDF

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.