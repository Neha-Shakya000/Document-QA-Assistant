import hashlib
import html
from pathlib import Path

import streamlit as st

from document_reader import extract_pages
from text_processor import create_chunks
from retriever import (
    load_embedding_model,
    build_vector_store,
    retrieve_relevant_chunks,
)
from answer_generator import load_qa_model, generate_answer


# -------------------- App configuration --------------------
st.set_page_config(
    page_title="DocuMind | Document AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent
DOCUMENT_DIR = APP_DIR / "uploaded_documents"
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)

NOT_FOUND = "Information not found in provided document"


# -------------------- Styling --------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #eaf0ff;
        --muted: #9aa8c7;
        --line: rgba(157, 177, 220, .16);
        --panel: rgba(20, 29, 51, .78);
        --accent: #8b7cff;
        --accent2: #42d6c5;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
          radial-gradient(ellipse at 8% 0%, rgba(91, 79, 190, .22), transparent 34%),
          radial-gradient(ellipse at 92% 12%, rgba(35, 177, 166, .12), transparent 28%),
          #0a1020;
        color: var(--ink);
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: rgba(13, 20, 38, .96);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.4rem; }

    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: .4px;
        color: #f4f6ff;
        margin-bottom: 1.6rem;
    }
    .brand span { color: #9c91ff; }

    .hero {
        padding: 1.7rem 1.9rem;
        border: 1px solid rgba(161, 169, 255, .2);
        border-radius: 22px;
        background: linear-gradient(115deg, rgba(92, 78, 190, .23), rgba(30, 159, 153, .10));
        box-shadow: 0 18px 60px rgba(0,0,0,.18);
        margin: .2rem 0 1.4rem 0;
    }
    .eyebrow {
        color: #aaa1ff;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: .72rem;
        font-weight: 700;
        margin-bottom: .55rem;
    }
    .hero h1 {
        font-family: 'Manrope', sans-serif;
        color: #f4f6ff;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.12;
        margin: 0;
        font-weight: 800;
    }
    .hero p {
        color: #b7c3df;
        font-size: 1rem;
        margin: .75rem 0 0 0;
        max-width: 720px;
    }

    .section-label {
        color: #cbd5ee;
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        margin: .5rem 0 .7rem 0;
    }

    .metric {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 1rem 1.1rem;
        min-height: 94px;
    }
    .metric .value {
        color: #f3f5ff;
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
    }
    .metric .label {
        color: var(--muted);
        font-size: .82rem;
        margin-top: .12rem;
    }

    .hint-card {
        border: 1px solid var(--line);
        background: rgba(20, 29, 51, .62);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        color: #b8c5e2;
        min-height: 115px;
    }
    .hint-card strong { color: #eef1ff; }

    .answer-card {
        background: linear-gradient(135deg, rgba(40, 157, 145, .15), rgba(100, 86, 203, .14));
        border: 1px solid rgba(87, 210, 190, .25);
        border-radius: 18px;
        padding: 1.2rem 1.35rem;
        margin: .7rem 0;
    }
    .answer-kicker {
        color: #6fe0cc;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: .7rem;
        font-weight: 700;
        margin-bottom: .45rem;
    }
    .answer-text {
        color: #f3f6ff;
        font-family: 'Manrope', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.5;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #7567e8, #4e9eaa);
        border: 0;
        border-radius: 11px;
        color: white;
        font-weight: 700;
        min-height: 2.8rem;
    }
    div.stButton > button {
        border-radius: 10px;
    }
    div[data-testid="stTextInput"] input {
        background: rgba(13, 21, 39, .8);
        border: 1px solid var(--line);
        border-radius: 11px;
        color: #f4f6ff;
    }
    [data-testid="stFileUploader"] {
        background: rgba(20, 29, 51, .45);
        border: 1px dashed rgba(156, 145, 255, .45);
        border-radius: 14px;
        padding: .5rem;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        background: rgba(20, 29, 51, .45);
    }
    .small-muted { color: #8f9dbb; font-size: .8rem; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- Cached models --------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return load_embedding_model()


@st.cache_resource(show_spinner=False)
def get_qa_model():
    return load_qa_model()


# -------------------- Session state --------------------
defaults = {
    "active_doc_path": None,
    "active_doc_hash": None,
    "vector_store": None,
    "total_pages": 0,
    "total_chunks": 0,
    "chat_history": [],
    "last_uploaded_hash": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def get_recent_documents():
    files = [
        path for path in DOCUMENT_DIR.glob("*.pdf")
        if path.is_file()
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def friendly_name(path):
    # Files are stored as <sha256>__<original-name>.pdf
    name = path.name.split("__", 1)[-1]
    return name[:55]


def save_uploaded_pdf(upload):
    data = upload.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    safe_name = Path(upload.name).name.replace("/", "_").replace("\\", "_")
    target = DOCUMENT_DIR / f"{digest}__{safe_name}"
    if not target.exists():
        target.write_bytes(data)
    return target, digest


def load_document(path):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()

    if st.session_state.active_doc_hash == digest and st.session_state.vector_store is not None:
        st.session_state.active_doc_path = str(path)
        return

    with st.spinner("Preparing your document — extracting text and building search index..."):
        pages = extract_pages(data)
        if not pages:
            raise ValueError(
                "No readable text was found. This PDF may be scanned or image-based."
            )

        chunks = create_chunks(pages)
        if not chunks:
            raise ValueError("No text chunks could be created from this PDF.")

        embeddings = get_embeddings()
        store = build_vector_store(chunks, embeddings)

    st.session_state.active_doc_path = str(path)
    st.session_state.active_doc_hash = digest
    st.session_state.vector_store = store
    st.session_state.total_pages = len(pages)
    st.session_state.total_chunks = len(chunks)
    st.session_state.chat_history = []


# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown('<div class="brand">📄 DOCU<span>MIND</span></div>', unsafe_allow_html=True)
    st.markdown("### Your workspace")
    st.caption("Upload a new PDF or reopen a document from your recent list.")

    uploaded_file = st.file_uploader(
        "Add a PDF document",
        type=["pdf"],
        help="Your uploaded PDF is saved in this app's document folder for quick reopening.",
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        new_path, new_hash = save_uploaded_pdf(uploaded_file)
        if st.session_state.last_uploaded_hash != new_hash:
            st.session_state.last_uploaded_hash = new_hash
            st.session_state["recent_document_select"] = str(new_path)
            st.session_state.active_doc_path = str(new_path)
            st.session_state.active_doc_hash = None
            st.session_state.vector_store = None
            st.session_state.chat_history = []
        st.caption(f"Added: {uploaded_file.name}")

    recent_files = get_recent_documents()
    st.markdown("---")
    st.markdown("### 🕘 Recent documents")

    if recent_files:
        paths = [str(p) for p in recent_files]
        current_path = st.session_state.active_doc_path
        default_index = paths.index(current_path) if current_path in paths else 0

        selected_path = st.selectbox(
            "Open a previously uploaded PDF",
            options=paths,
            index=default_index,
            format_func=lambda p: f"📄 {friendly_name(Path(p))}",
            key="recent_document_select",
        )

        # A newly uploaded file takes priority on the first run. After that,
        # selecting another item in this list opens that saved document.
        if uploaded_file is None and selected_path:
            if st.session_state.active_doc_path != selected_path:
                st.session_state.active_doc_path = selected_path
                st.session_state.active_doc_hash = None
                st.session_state.vector_store = None
                st.session_state.chat_history = []
    else:
        st.info("Your recent documents will appear here.")

    st.markdown("---")
    st.markdown(
        '<div class="small-muted">🔒 Answers are based on the PDF you select. '
        'Unsupported questions return the not-found message.</div>',
        unsafe_allow_html=True,
    )


# -------------------- Main interface --------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Your personal document assistant</div>
      <h1>Ask your documents.<br>Find answers faster.</h1>
      <p>Upload a PDF, ask a question in simple language, and get an answer with
      the supporting passage and page reference.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.active_doc_path:
    active_path = Path(st.session_state.active_doc_path)
    if active_path.exists():
        try:
            load_document(active_path)
        except Exception as error:
            st.error(f"Document processing failed: {error}")
            st.stop()

        st.markdown('<div class="section-label">Your active document</div>', unsafe_allow_html=True)
        info_col, page_col, chunk_col = st.columns([2.2, 1, 1])
        with info_col:
            st.markdown(
                f"""
                <div class="metric">
                  <div class="value" style="font-size:1.08rem">📄 {friendly_name(active_path)}</div>
                  <div class="label">Selected PDF document</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with page_col:
            st.markdown(
                f'<div class="metric"><div class="value">{st.session_state.total_pages}</div><div class="label">Pages read</div></div>',
                unsafe_allow_html=True,
            )
        with chunk_col:
            st.markdown(
                f'<div class="metric"><div class="value">{st.session_state.total_chunks}</div><div class="label">Search chunks</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown('<div class="section-label">💬 Ask a question</div>', unsafe_allow_html=True)
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input(
                "Question",
                placeholder="Example: What is the main objective of this document?",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Find answer  →", type="primary", use_container_width=True)

        if submitted:
            if not question.strip():
                st.warning("Please enter a question first.")
            else:
                try:
                    with st.spinner("Searching the document and preparing an answer..."):
                        results = retrieve_relevant_chunks(
                            st.session_state.vector_store,
                            question,
                            top_k=4,
                        )
                        qa_model = get_qa_model()
                        response = generate_answer(question, results, qa_model)

                    st.session_state.chat_history.append({
                        "question": question,
                        "response": response,
                        "retrieved": results,
                    })
                except Exception as error:
                    st.error(f"Question processing failed: {error}")

        if st.session_state.chat_history:
            st.markdown('<div class="section-label">Answer history</div>', unsafe_allow_html=True)
            for item in reversed(st.session_state.chat_history):
                st.markdown(f"**You asked:** {item['question']}")
                response = item["response"]

                if response.get("found"):
                    st.markdown(
                        f"""
                        <div class="answer-card">
                          <div class="answer-kicker">Answer found · Page {response['page']}</div>
                          <div class="answer-text">{html.escape(str(response['answer']))}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander("📌 View supporting passage"):
                        st.write(response["source"])
                        st.caption(f"Source page: {response['page']}")
                else:
                    st.warning(NOT_FOUND)

                with st.expander("🔎 View retrieved passages"):
                    for passage in item["retrieved"]:
                        st.markdown(
                            f"**Chunk {passage['chunk_id']} · Page {passage['page']} · Distance {passage['distance']:.4f}**"
                        )
                        st.write(passage["chunk"])
                        st.divider()
                st.markdown("---")

        else:
            st.markdown(
                """
                <div class="hint-card">
                  <strong>✨ Try asking</strong><br><br>
                  • What is the main objective of this document?<br>
                  • What are the key features or recommendations?<br>
                  • Which page explains the implementation?
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("This saved PDF could not be found. Please upload it again.")
else:
    st.markdown(
        """
        <div class="hint-card">
          <strong>👋 Welcome to DocuMind</strong><br><br>
          Start by uploading a PDF from the left panel. Once it is processed,
          you can ask questions and revisit your recent documents from the same panel.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="small-muted" style="text-align:center;margin-top:3rem;">DocuMind · AI-powered document question answering</p>',
    unsafe_allow_html=True,
)
