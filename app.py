"""
AskVista - Company Policy Assistant
===================================

A Streamlit RAG (Retrieval-Augmented Generation) application that answers
questions about a company policy document using:

- LangChain for the retrieval pipeline
- Ollama (llama3.2) as the local LLM
- mxbai-embed-large for embeddings
- ChromaDB as the persistent vector store

Usage:
    streamlit run app.py

Requirements:
    - Ollama running locally with `llama3.2` and `mxbai-embed-large` pulled
    - A policy PDF at the path defined in `PDF_PATH`

Author: <iqbalsh>
"""

import html
import logging
from pathlib import Path

import streamlit as st

# -----------------------------
# Configuration
# -----------------------------
PDF_PATH = Path("data/company_policy.pdf")
CHROMA_DIR = Path("chromaDB")
CSS_PATH = Path("assets/styles.css")

LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "mxbai-embed-large"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
LLM_TEMPERATURE = 0

PROMPT_TEMPLATE = """
Use the following context to answer the question.
If you don't know the answer, just say you don't know.

Context: {context}
Question: {question}

Answer:
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("askvista")

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="AskVista",
    page_icon="📋",
    layout="centered",
)


def load_css(css_path: Path) -> None:
    """Inject the external stylesheet into the Streamlit page."""
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
    else:
        logger.warning("Stylesheet not found at %s — using default theme.", css_path)


load_css(CSS_PATH)


# ----------------------------------------------
# RAG pipeline (cached so it only builds once per session)
# ----------------------------------------------
@st.cache_resource(show_spinner="Loading the policy knowledge base…")
def load_chain():
    """
    Build (or load) the retrieval chain.

    On first run, the policy PDF is split into chunks, embedded, and stored
    in a persistent Chroma database. On subsequent runs, the existing
    database is loaded directly, which avoids re-embedding the document
    every time the app restarts.

    Returns:
        A LangChain runnable: question (str) -> answer (str).

    Raises:
        FileNotFoundError: If the policy PDF is missing and no existing
            vector store is available.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Load the existing vector store if it was already built; otherwise
    # ingest the PDF and persist a new one.
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        logger.info("Loading existing Chroma store from %s", CHROMA_DIR)
        db = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embedding,
        )
    else:
        if not PDF_PATH.exists():
            raise FileNotFoundError(
                f"Policy document not found at '{PDF_PATH}'. "
                "Place your PDF there and restart the app."
            )
        logger.info("Building new Chroma store from %s", PDF_PATH)
        documents = PyPDFLoader(str(PDF_PATH)).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
        db = Chroma.from_documents(
            chunks,
            embedding,
            persist_directory=str(CHROMA_DIR),
        )

    retriever = db.as_retriever()
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def retrieve_context(question: str) -> str:
        """Fetch the most relevant chunks and join them into one string."""
        docs = retriever.invoke(question)
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableLambda(lambda q: {"context": retrieve_context(q), "question": q})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ------------------------------------
# UI - header, suggestion chips, input
# ------------------------------------
st.markdown(
    """
    <div class="header-card">
        <div class="header-icon">📋</div>
        <div class="header-text">
            <h1>AskVista: Company Policy Assistant</h1>
            <p>Ask anything about our company policy document</p>
        </div>
        <div class="status-pill">
            <div class="dot-green"></div> Model ready
        </div>
    </div>

    <div class="chips-row">
        <div class="chip">💼 Reimbursement policy</div>
        <div class="chip">🏖️ Leave entitlement</div>
        <div class="chip">🔄 Return &amp; refund rules</div>
        <div class="chip">📅 Working hours</div>
    </div>
    """,
    unsafe_allow_html=True,
)

question = st.text_input(
    label="Question",
    placeholder="e.g. What is the reimbursement limit for business travel?",
    label_visibility="collapsed",
)

# ----------------------------
# Answer
# ----------------------------
if question:
    try:
        qa_chain = load_chain()
        with st.spinner("Looking through the policy document…"):
            response = qa_chain.invoke(question)

        # Escape the model output before injecting it into raw HTML so that
        # any <, >, or & characters cannot break (or inject into) the layout.
        safe_response = html.escape(response).replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="answer-card">
                <div class="answer-label">📄 Answer</div>
                <div class="answer-body">{safe_response}</div>
                <div class="source-badge">📎 {PDF_PATH.name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError as exc:
        logger.error("Missing document: %s", exc)
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001 — surface any backend failure to the user
        logger.exception("Failed to answer question")
        st.error(
            "Something went wrong while querying the model. "
            "Make sure Ollama is running (`ollama serve`) and the "
            f"`{LLM_MODEL}` and `{EMBEDDING_MODEL}` models are pulled."
        )

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    '<div class="footer-note">Powered by LangChain · Ollama llama3.2 · ChromaDB</div>',
    unsafe_allow_html=True,
)
