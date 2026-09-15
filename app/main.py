"""
DocMind - RAG Knowledge Assistant

Streamlit application for asking grounded questions
against the indexed document knowledge base.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from app.generation.answer_service import (  # noqa: E402
    ABSTENTION_MESSAGE,
    AnswerService,
)

from app.vectorstore.chroma_store import ChromaStore  # noqa: E402


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocMind - RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

        .main-header {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .sub-header {
            color: #666;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "answer" not in st.session_state:
    st.session_state.answer = None

if "sources" not in st.session_state:
    st.session_state.sources = []

if "question" not in st.session_state:
    st.session_state.question = ""


# ============================================================
# SERVICES
# ============================================================

@st.cache_resource
def get_answer_service() -> AnswerService:
    """
    Create and cache the RAG answer service.
    """
    return AnswerService(top_k=3)


@st.cache_resource
def get_vector_store() -> ChromaStore:
    """
    Connect to the existing persistent ChromaDB database.
    """
    chroma_directory = PROJECT_ROOT / "data" / "chroma"

    return ChromaStore(
        str(chroma_directory)
    )


answer_service = get_answer_service()
vector_store = get_vector_store()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 DocMind")

    st.caption(
        "RAG Knowledge Assistant"
    )

    st.divider()

    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------

    st.subheader("📚 Knowledge Base")

    # --------------------------------------------------------
    # Indexed chunks
    # --------------------------------------------------------

    try:

        chunk_count = vector_store.count()

    except Exception:

        chunk_count = 0

    st.metric(
        label="Indexed Chunks",
        value=chunk_count,
    )

    # --------------------------------------------------------
    # Documents
    # --------------------------------------------------------

    try:

        collection = vector_store.collection

        all_data = collection.get(
            include=["metadatas"]
        )

        metadatas = (
            all_data.get("metadatas")
            or []
        )

        filenames = {
            metadata.get("filename")
            for metadata in metadatas
            if metadata
            and metadata.get("filename")
        }

        document_count = len(filenames)

    except Exception:

        document_count = 0

    st.metric(
        label="Documents",
        value=document_count,
    )

    st.divider()

    # --------------------------------------------------------
    # RETRIEVAL CONFIGURATION
    # --------------------------------------------------------

    st.subheader("⚙️ Retrieval")

    st.write("Embedding model")

    st.code(
        "all-MiniLM-L6-v2",
        language="text",
    )

    st.write("Vector database")

    st.code(
        "ChromaDB",
        language="text",
    )

    st.write("Retrieval mode")

    st.code(
        "Dense Semantic Search",
        language="text",
    )

    st.write("Top-K")

    st.code(
        "3",
        language="text",
    )

    st.divider()

    st.caption(
        "Answers are generated only from "
        "retrieved document context."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-header">🧠 DocMind</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-header">
        RAG Knowledge Assistant —
        Ask questions about your indexed documents.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# QUESTION SECTION
# ============================================================

st.subheader("Ask a question")

question = st.text_area(
    "Question",
    value=st.session_state.question,
    placeholder=(
        "Example: How many annual leave days "
        "do full-time employees receive?"
    ),
    height=100,
    label_visibility="collapsed",
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Ask DocMind",
    type="primary",
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    st.session_state.question = (
        question.strip()
    )

    st.session_state.answer = None

    st.session_state.sources = []

    with st.spinner(
        "Searching documents and "
        "generating a grounded answer..."
    ):

        try:

            result = answer_service.ask(
                question.strip()
            )

            st.session_state.answer = (
                result["answer"]
            )

            st.session_state.sources = (
                result["sources"]
            )

        except Exception as exc:

            st.error(
                "An error occurred while "
                "processing your question."
            )

            st.exception(exc)

            st.stop()


# ============================================================
# DISPLAY ANSWER
# ============================================================

if st.session_state.answer:

    st.divider()

    st.subheader("📝 Answer")

    answer = st.session_state.answer

    # --------------------------------------------------------
    # Abstention response
    # --------------------------------------------------------

    if answer.strip() == ABSTENTION_MESSAGE:

        st.info(
            "I don't know based on the available documents."
        )

    # --------------------------------------------------------
    # Normal grounded response
    # --------------------------------------------------------

    else:

        st.markdown(
            answer
        )

    # ========================================================
    # SOURCES
    # ========================================================

    sources = st.session_state.sources

    if sources:

        st.divider()

        st.subheader("📚 Sources")

        st.caption(
            "These sources come directly from "
            "the retrieval metadata."
        )

        # ----------------------------------------------------
        # Display every retrieved source
        # ----------------------------------------------------

        for source in sources:

            source_number = source.get(
                "source_number",
                "?",
            )

            filename = source.get(
                "filename",
                "Unknown document",
            )

            page = source.get(
                "page"
            )

            category = source.get(
                "category",
                "unknown",
            )

            chunk_number = source.get(
                "chunk_number"
            )

            distance = source.get(
                "distance"
            )

            # ------------------------------------------------
            # Format values
            # ------------------------------------------------

            page_text = (
                "N/A"
                if page is None
                else str(page)
            )

            chunk_text = (
                "N/A"
                if chunk_number is None
                else str(chunk_number)
            )

            if isinstance(
                distance,
                (int, float),
            ):

                distance_text = (
                    f"{distance:.4f}"
                )

            else:

                distance_text = str(
                    distance
                )

            # ------------------------------------------------
            # Native Streamlit source card
            # ------------------------------------------------

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### 📄 {source_number}. {filename}"
                )

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                with col1:

                    st.write(
                        "**Category**"
                    )

                    st.write(
                        category
                    )

                with col2:

                    st.write(
                        "**Page**"
                    )

                    st.write(
                        page_text
                    )

                with col3:

                    st.write(
                        "**Chunk**"
                    )

                    st.write(
                        chunk_text
                    )

                with col4:

                    st.write(
                        "**Distance**"
                    )

                    st.write(
                        distance_text
                    )

    else:

        st.caption(
            "No source metadata was returned."
        )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "Ask a question above to search "
        "the knowledge base."
    )

    st.markdown(
        """
        ### Example questions

        **HR**

        - How many annual leave days do
          full-time employees receive?
        - How far in advance should annual
          leave be requested?

        **Technical**

        - What should I do if NovaDesk
          fails to start?
        - How do I configure NovaDesk?

        **Products**

        - What features does NovaDesk provide?
        - What is NovaFlow used for?

        **Unknown information**

        - What is the capital city of France?

        ---

        DocMind should answer only when the
        information is supported by the
        indexed documents.
        """
    )