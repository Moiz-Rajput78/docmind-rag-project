from pathlib import Path
import sys

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# IMPORTS
# ============================================================

from app.generation.answer_service import (
    AnswerService,
)

from app.vectorstore.chroma_store import (
    ChromaStore,
)

from app.utils.document_manager import (
    DocumentManager,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocMind — RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .conversation-question {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.20);
        margin-bottom: 0.6rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "action_message" not in st.session_state:
    st.session_state.action_message = None

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# ============================================================
# CACHED SERVICES
# ============================================================

@st.cache_resource
def get_answer_service(
    top_k: int,
):
    return AnswerService(
        top_k=top_k
    )


@st.cache_resource
def get_vector_store():

    return ChromaStore(
        PROJECT_ROOT
        / "data"
        / "chroma"
    )


@st.cache_resource
def get_document_manager():

    return DocumentManager()


# ============================================================
# HELPER — ADD CONVERSATION
# ============================================================

def add_to_conversation(result):
    """
    Store a completed RAG interaction in Streamlit
    session state.

    Conversation history is UI/session history only.
    It is NOT used as a replacement for document retrieval.
    """

    if not result:
        return

    question = result.get(
        "question",
        "",
    ).strip()

    if not question:
        return

    st.session_state.conversation_history.append(
        {
            "question": question,
            "answer": result.get(
                "answer",
                "No answer generated.",
            ),
            "sources": result.get(
                "sources",
                [],
            ),
            "retrieved_count": result.get(
                "retrieved_count",
                0,
            ),
            "top_k": result.get(
                "top_k",
                3,
            ),
        }
    )


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

    try:

        vector_store = (
            get_vector_store()
        )

        indexed_chunks = (
            vector_store.count()
        )

        try:

            collection_data = (
                vector_store.collection.get(
                    include=["metadatas"]
                )
            )

            metadatas = (
                collection_data.get(
                    "metadatas",
                    [],
                )
            )

            indexed_documents = len(
                {
                    metadata.get(
                        "source"
                    )
                    for metadata in metadatas
                    if metadata
                    and metadata.get(
                        "source"
                    )
                }
            )

        except Exception:

            indexed_documents = 0

        st.subheader(
            "📊 Knowledge Base"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Documents",
                indexed_documents,
            )

        with col2:

            st.metric(
                "Chunks",
                indexed_chunks,
            )

    except Exception as exc:

        st.error(
            f"Knowledge base error: {exc}"
        )

    st.divider()

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    st.subheader(
        "⚙️ Retrieval"
    )

    top_k = st.slider(
        "Top-K chunks",
        min_value=1,
        max_value=10,
        value=3,
        help=(
            "Number of relevant document "
            "chunks retrieved for each question."
        ),
    )

    st.caption(
        f"Retrieving the top **{top_k}** "
        "relevant chunks."
    )

    st.divider()

    # --------------------------------------------------------
    # CONVERSATION
    # --------------------------------------------------------

    st.subheader(
        "💬 Conversation"
    )

    conversation_count = len(
        st.session_state.conversation_history
    )

    st.metric(
        "Questions asked",
        conversation_count,
    )

    if conversation_count > 0:

        if st.button(
            "🗑️ Clear Conversation",
            use_container_width=True,
        ):

            st.session_state.conversation_history = []
            st.session_state.last_result = None

            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.subheader(
        "🟢 System Status"
    )

    st.success(
        "RAG Pipeline Ready"
    )

    st.caption(
        "✓ Local embeddings"
    )

    st.caption(
        "✓ ChromaDB vector store"
    )

    st.caption(
        "✓ Semantic retrieval"
    )

    st.caption(
        "✓ Grounded generation"
    )

    st.caption(
        "✓ Source citations"
    )

    st.caption(
        "✓ Conversation history"
    )

    st.divider()

    st.info(
        "DocMind answers questions using "
        "retrieved document content rather than "
        "general model knowledge."
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧠 DocMind — RAG Knowledge Assistant"
)

st.write(
    "Upload documents, retrieve relevant knowledge, "
    "and ask grounded questions with source citations."
)


# ============================================================
# TOP METRICS
# ============================================================

try:

    document_manager = (
        get_document_manager()
    )

    manager_stats = (
        document_manager.get_statistics()
    )

    managed_documents = (
        manager_stats.get(
            "document_count",
            0,
        )
    )

    managed_chunks = (
        manager_stats.get(
            "indexed_chunks",
            0,
        )
    )

    storage_bytes = (
        manager_stats.get(
            "total_size_bytes",
            0,
        )
    )

    storage_mb = (
        storage_bytes
        / (1024 * 1024)
    )

except Exception:

    managed_documents = 0
    managed_chunks = 0
    storage_mb = 0


metric1, metric2, metric3 = (
    st.columns(3)
)

with metric1:

    st.metric(
        "📄 Managed Documents",
        managed_documents,
    )

with metric2:

    st.metric(
        "🧩 Indexed Chunks",
        managed_chunks,
    )

with metric3:

    st.metric(
        "💾 Storage",
        f"{storage_mb:.2f} MB",
    )


# ============================================================
# TABS
# ============================================================

ask_tab, documents_tab = st.tabs(
    [
        "💬 Ask DocMind",
        "📚 Document Management",
    ]
)


# ============================================================
# ASK DOCMIND
# ============================================================

with ask_tab:

    # ========================================================
    # CONVERSATION HISTORY
    # ========================================================

    if st.session_state.conversation_history:

        st.subheader(
            "💬 Conversation History"
        )

        st.caption(
            "Previous questions and grounded answers "
            "from this browser session."
        )

        for index, conversation in enumerate(
            st.session_state.conversation_history
        ):

            question_text = conversation.get(
                "question",
                "",
            )

            answer_text = conversation.get(
                "answer",
                "No answer generated.",
            )

            sources = conversation.get(
                "sources",
                [],
            )

            with st.chat_message("user"):

                st.markdown(
                    question_text
                )

            with st.chat_message("assistant"):

                if answer_text.strip() == (
                    "I don't know based on the available documents."
                ):

                    st.warning(
                        answer_text
                    )

                else:

                    st.markdown(
                        answer_text
                    )

                # ------------------------------------------------
                # Conversation Sources
                # ------------------------------------------------

                if sources:

                    with st.expander(
                        f"📚 Sources ({len(sources)})"
                    ):

                        for source in sources:

                            source_number = (
                                source.get(
                                    "source_number",
                                    "?",
                                )
                            )

                            filename = (
                                source.get(
                                    "filename",
                                    "Unknown",
                                )
                            )

                            page = source.get(
                                "page"
                            )

                            category = (
                                source.get(
                                    "category",
                                    "unknown",
                                )
                            )

                            chunk_number = (
                                source.get(
                                    "chunk_number",
                                    "?",
                                )
                            )

                            distance = (
                                source.get(
                                    "distance"
                                )
                            )

                            if (
                                page is None
                                or page == -1
                            ):

                                page_text = (
                                    "Page: N/A"
                                )

                            else:

                                page_text = (
                                    f"Page: {page}"
                                )

                            if distance is not None:

                                distance_text = (
                                    f"{float(distance):.4f}"
                                )

                            else:

                                distance_text = "N/A"
                            st.markdown(
                                f"**[{source_number}] 📄 {filename}**"
                            )

                            st.caption(
                                f"Category: {category} • "
                                f"{page_text} • "
                                f"Chunk: {chunk_number} • "
                                f"Distance: {distance_text}"
                            )

            if index < len(
                st.session_state.conversation_history
            ) - 1:

                st.divider()

        st.divider()


    # ========================================================
    # QUESTION INPUT
    # ========================================================

    st.subheader(
        "Ask a question"
    )

    question = st.text_area(
        "Question",
        placeholder=(
            "Example: How many annual leave "
            "days do full-time employees receive?"
        ),
        height=100,
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns(
        [1, 1, 4]
    )

    with col1:

        ask_button = st.button(
            "🔍 Ask",
            type="primary",
            use_container_width=True,
        )

    with col2:

        clear_button = st.button(
            "🧹 Clear Input",
            use_container_width=True,
        )

    with col3:

        clear_history_button = st.button(
            "🗑️ Clear Conversation",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # CLEAR INPUT
    # --------------------------------------------------------

    if clear_button:

        st.rerun()


    # --------------------------------------------------------
    # CLEAR CONVERSATION
    # --------------------------------------------------------

    if clear_history_button:

        st.session_state.conversation_history = []
        st.session_state.last_result = None

        st.rerun()


    # --------------------------------------------------------
    # ASK
    # --------------------------------------------------------

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner(
                "Retrieving relevant documents "
                "and generating answer..."
            ):

                try:

                    answer_service = (
                        get_answer_service(
                            top_k
                        )
                    )

                    result = (
                        answer_service.ask(
                            question.strip()
                        )
                    )

                    st.session_state.last_result = (
                        result
                    )

                    add_to_conversation(
                        result
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Unable to answer the "
                        f"question: {exc}"
                    )


    # --------------------------------------------------------
    # CURRENT ANSWER
    # --------------------------------------------------------

    result = (
        st.session_state.last_result
    )

    if result:

        st.divider()

        st.subheader(
            "💡 Latest Answer"
        )

        answer = result.get(
            "answer",
            "No answer generated.",
        )

        if answer.strip() == (
            "I don't know based on the available documents."
        ):

            st.warning(
                answer
            )

        else:

            st.markdown(
                answer
            )

        st.divider()

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        sources = result.get(
            "sources",
            [],
        )

        st.subheader(
            f"📚 Sources ({len(sources)})"
        )

        if sources:

            for source in sources:

                source_number = (
                    source.get(
                        "source_number",
                        "?",
                    )
                )

                filename = (
                    source.get(
                        "filename",
                        "Unknown",
                    )
                )

                page = source.get(
                    "page"
                )

                category = (
                    source.get(
                        "category",
                        "unknown",
                    )
                )

                chunk_number = (
                    source.get(
                        "chunk_number",
                        "?",
                    )
                )

                distance = (
                    source.get(
                        "distance"
                    )
                )

                if (
                    page is None
                    or page == -1
                ):

                    page_text = (
                        "Page: N/A"
                    )

                else:

                    page_text = (
                        f"Page: {page}"
                    )

                if distance is not None:

                    distance_text = (
                        f"{float(distance):.4f}"
                    )

                else:

                    distance_text = "N/A"

                st.markdown(
                    f"""
                    <div class="source-card">

                        <div class="source-title">
                            [{source_number}]
                            📄 {filename}
                        </div>

                        <div class="small-muted">
                            Category: {category}
                            &nbsp; • &nbsp;
                            {page_text}
                            &nbsp; • &nbsp;
                            Chunk: {chunk_number}
                            &nbsp; • &nbsp;
                            Distance: {distance_text}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.info(
                "No source information was returned."
            )

        # ----------------------------------------------------
        # RETRIEVAL DETAILS
        # ----------------------------------------------------

        with st.expander(
            "🔎 Retrieval Details"
        ):

            st.write(
                f"**Question:** "
                f"{result.get('question', '')}"
            )

            st.write(
                f"**Top-K:** "
                f"{result.get('top_k', top_k)}"
            )

            st.write(
                f"**Retrieved chunks:** "
                f"{result.get('retrieved_count', 0)}"
            )


    # ========================================================
    # EXAMPLES
    # ========================================================

    st.divider()

    st.subheader(
        "💭 Example Questions"
    )

    example_questions = [
        "How many annual leave days do full-time employees receive?",
        "What should I do if NovaDesk fails to start?",
        "What are the main features of NovaDesk?",
        "How do I install NovaDesk?",
        "What is the company's attendance policy?",
    ]

    for example in example_questions:

        if st.button(
            example,
            key=f"example_{example}",
            use_container_width=True,
        ):

            with st.spinner(
                "Retrieving documents and generating answer..."
            ):

                try:

                    result = (
                        get_answer_service(
                            top_k
                        ).ask(
                            example
                        )
                    )

                    st.session_state.last_result = (
                        result
                    )

                    add_to_conversation(
                        result
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Unable to answer example question: "
                        f"{exc}"
                    )


    # ========================================================
    # ABSTENTION TEST
    # ========================================================

    with st.expander(
        "🧪 Test Unknown Question"
    ):

        st.write(
            "This tests whether DocMind refuses "
            "to invent information that is not "
            "present in the knowledge base."
        )

        unknown_question = (
            "What is the capital city of France?"
        )

        if st.button(
            "Run Abstention Test",
            use_container_width=True,
        ):

            with st.spinner(
                "Testing..."
            ):

                try:

                    result = (
                        get_answer_service(
                            top_k
                        ).ask(
                            unknown_question
                        )
                    )

                    st.session_state.last_result = (
                        result
                    )

                    add_to_conversation(
                        result
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Abstention test failed: {exc}"
                    )


    # ========================================================
    # ARCHITECTURE
    # ========================================================

    with st.expander(
        "🏗️ How DocMind Works"
    ):

        st.markdown(
            """
            **1. Document ingestion**

            PDF, DOCX and TXT documents are loaded and cleaned.

            **2. Chunking**

            Documents are divided into smaller overlapping chunks.

            **3. Embeddings**

            Each chunk is converted into a semantic vector using
            `all-MiniLM-L6-v2`.

            **4. Vector storage**

            Embeddings and metadata are stored in ChromaDB.

            **5. Retrieval**

            Your question is embedded and the most relevant chunks
            are retrieved.

            **6. Context building**

            Retrieved chunks are combined into grounded context.

            **7. Generation**

            Ollama Cloud generates an answer using the retrieved
            context.

            **8. Citations**

            Source filename, category, page and chunk information
            are displayed with the answer.

            **9. Abstention**

            If the information cannot be supported by the available
            documents, DocMind responds:

            > I don't know based on the available documents.

            **10. Conversation history**

            Previous questions and answers are displayed for the
            current Streamlit session. Each new question still
            performs document retrieval independently.
            """
        )


# ============================================================
# DOCUMENT MANAGEMENT
# ============================================================

with documents_tab:

    st.subheader(
        "📚 Document Management"
    )

    document_manager = (
        get_document_manager()
    )

    # ========================================================
    # ACTION MESSAGE
    # ========================================================

    if st.session_state.action_message:

        message = (
            st.session_state.action_message
        )

        message_type = (
            message.get(
                "type",
                "info",
            )
        )

        message_text = (
            message.get(
                "text",
                "",
            )
        )

        if message_type == "success":

            st.success(
                message_text
            )

        elif message_type == "error":

            st.error(
                message_text
            )

        elif message_type == "warning":

            st.warning(
                message_text
            )

        else:

            st.info(
                message_text
            )

        st.session_state.action_message = None


    # ========================================================
    # STATISTICS
    # ========================================================

    try:

        stats = (
            document_manager.get_statistics()
        )

        document_count = (
            stats.get(
                "document_count",
                0,
            )
        )

        indexed_chunks = (
            stats.get(
                "indexed_chunks",
                0,
            )
        )

        total_size = (
            stats.get(
                "total_size_bytes",
                0,
            )
        )

        total_size_mb = (
            total_size
            / (1024 * 1024)
        )

    except Exception as exc:

        st.error(
            f"Could not load document statistics: "
            f"{exc}"
        )

        document_count = 0
        indexed_chunks = 0
        total_size_mb = 0


    stat1, stat2, stat3 = (
        st.columns(3)
    )

    with stat1:

        st.metric(
            "📄 Documents",
            document_count,
        )

    with stat2:

        st.metric(
            "🧩 Indexed Chunks",
            indexed_chunks,
        )

    with stat3:

        st.metric(
            "💾 Storage",
            f"{total_size_mb:.2f} MB",
        )


    st.divider()


    # ========================================================
    # UPLOAD
    # ========================================================

    st.subheader(
        "⬆️ Upload a Document"
    )

    st.write(
        "Upload a PDF, DOCX or TXT document. "
        "The document will be saved, chunked, embedded, "
        "and indexed automatically."
    )

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=[
            "pdf",
            "docx",
            "txt",
        ],
        help=(
            "Supported formats: "
            "PDF, DOCX and TXT."
        ),
    )

    if uploaded_file:

        file_size_mb = (
            uploaded_file.size
            / (1024 * 1024)
        )

        st.info(
            f"📄 **Selected:** "
            f"`{uploaded_file.name}`  \n"
            f"📦 **Size:** "
            f"{file_size_mb:.2f} MB"
        )

        upload_button = st.button(
            "⬆️ Upload & Index",
            type="primary",
            use_container_width=True,
        )

        if upload_button:

            with st.spinner(
                f"Uploading and indexing "
                f"`{uploaded_file.name}`..."
            ):

                try:

                    result = (
                        document_manager
                        .upload_and_index(
                            uploaded_file
                        )
                    )

                    filename = result.get(
                        "filename",
                        uploaded_file.name,
                    )

                    chunks = result.get(
                        "chunks",
                        0,
                    )

                    indexed = result.get(
                        "indexed",
                        chunks,
                    )

                    status = result.get(
                        "status",
                        "success",
                    )

                    if status == "success":

                        st.session_state.action_message = {
                            "type": "success",
                            "text": (
                                f"✅ Document uploaded "
                                f"and indexed successfully!\n\n"
                                f"**File:** `{filename}`\n\n"
                                f"**Chunks created:** "
                                f"{chunks}\n\n"
                                f"**Chunks indexed:** "
                                f"{indexed}"
                            ),
                        }

                    else:

                        st.session_state.action_message = {
                            "type": "warning",
                            "text": (
                                f"⚠️ Document processing "
                                f"finished with status: "
                                f"`{status}`"
                            ),
                        }

                    st.rerun()

                except FileExistsError as exc:

                    st.session_state.action_message = {
                        "type": "warning",
                        "text": (
                            f"⚠️ {exc}"
                        ),
                    }

                    st.rerun()

                except Exception as exc:

                    st.session_state.action_message = {
                        "type": "error",
                        "text": (
                            f"❌ Upload failed for "
                            f"`{uploaded_file.name}`.\n\n"
                            f"**Reason:** {exc}"
                        ),
                    }

                    st.rerun()


    st.divider()


    # ========================================================
    # RE-INDEX ALL
    # ========================================================

    st.subheader(
        "🔄 Re-index Documents"
    )

    st.write(
        "Re-process all managed documents and rebuild "
        "their embeddings in ChromaDB."
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔄 Re-index All",
            use_container_width=True,
        ):

            with st.spinner(
                "Re-indexing all documents..."
            ):

                try:

                    result = (
                        document_manager
                        .reindex_all()
                    )

                    total = result.get(
                        "total_documents",
                        0,
                    )

                    success_count = result.get(
                        "success_count",
                        0,
                    )

                    failure_count = result.get(
                        "failure_count",
                        0,
                    )

                    failed = result.get(
                        "failed",
                        [],
                    )

                    status = result.get(
                        "status",
                        "success",
                    )

                    if status == "success":

                        st.session_state.action_message = {
                            "type": "success",
                            "text": (
                                f"✅ All documents "
                                f"re-indexed successfully!\n\n"
                                f"**Documents:** {total}\n\n"
                                f"**Successful:** "
                                f"{success_count}"
                            ),
                        }

                    else:

                        failed_names = ", ".join(
                            item.get(
                                "filename",
                                "Unknown",
                            )
                            for item in failed
                        )

                        st.session_state.action_message = {
                            "type": "warning",
                            "text": (
                                f"⚠️ Re-indexing "
                                f"completed partially.\n\n"
                                f"**Total:** {total}\n\n"
                                f"**Successful:** "
                                f"{success_count}\n\n"
                                f"**Failed:** "
                                f"{failure_count}\n\n"
                                f"**Failed files:** "
                                f"{failed_names}"
                            ),
                        }

                    st.rerun()

                except Exception as exc:

                    st.session_state.action_message = {
                        "type": "error",
                        "text": (
                            f"❌ Re-index all failed.\n\n"
                            f"**Reason:** {exc}"
                        ),
                    }

                    st.rerun()


    with col2:

        if st.button(
            "🔃 Refresh",
            use_container_width=True,
        ):

            st.rerun()


    st.divider()


    # ========================================================
    # DOCUMENT LIST
    # ========================================================

    st.subheader(
        "📄 Your Documents"
    )

    try:

        documents = (
            document_manager.list_documents()
        )

    except Exception as exc:

        documents = []

        st.error(
            f"Could not load documents: {exc}"
        )


    if not documents:

        st.info(
            "No managed documents found. "
            "Upload your first document above."
        )

    else:

        for index, document in enumerate(
            documents
        ):

            filename = document.get(
                "filename",
                "Unknown",
            )

            extension = document.get(
                "extension",
                "",
            )

            size_bytes = document.get(
                "size_bytes",
                0,
            )

            size_kb = (
                size_bytes / 1024
            )

            st.markdown(
                f"### 📄 {filename}"
            )

            st.caption(
                f"{extension.upper()} • "
                f"{size_kb:.1f} KB"
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "🔄 Re-index",
                    key=(
                        f"reindex_"
                        f"{index}_"
                        f"{filename}"
                    ),
                    use_container_width=True,
                ):

                    with st.spinner(
                        f"Re-indexing {filename}..."
                    ):

                        try:

                            result = (
                                document_manager
                                .reindex_document(
                                    filename
                                )
                            )

                            chunks = result.get(
                                "chunks",
                                0,
                            )

                            removed = result.get(
                                "previous_chunks_removed",
                                0,
                            )

                            st.session_state.action_message = {
                                "type": "success",
                                "text": (
                                    f"✅ `{filename}` "
                                    f"re-indexed successfully!\n\n"
                                    f"**Chunks indexed:** "
                                    f"{chunks}\n\n"
                                    f"**Previous chunks "
                                    f"removed:** {removed}"
                                ),
                            }

                            st.rerun()

                        except Exception as exc:

                            st.session_state.action_message = {
                                "type": "error",
                                "text": (
                                    f"❌ Re-indexing "
                                    f"`{filename}` failed.\n\n"
                                    f"**Reason:** {exc}"
                                ),
                            }

                            st.rerun()

            with col2:

                if st.button(
                    "🗑️ Delete",
                    key=(
                        f"delete_"
                        f"{index}_"
                        f"{filename}"
                    ),
                    use_container_width=True,
                ):

                    with st.spinner(
                        f"Deleting {filename}..."
                    ):

                        try:

                            result = (
                                document_manager
                                .delete_document(
                                    filename
                                )
                            )

                            deleted_chunks = (
                                result.get(
                                    "deleted_chunks",
                                    0,
                                )
                            )

                            st.session_state.action_message = {
                                "type": "success",
                                "text": (
                                    f"🗑️ `{filename}` "
                                    f"was deleted successfully!\n\n"
                                    f"**Vectors removed:** "
                                    f"{deleted_chunks}"
                                ),
                            }

                            st.rerun()

                        except Exception as exc:

                            st.session_state.action_message = {
                                "type": "error",
                                "text": (
                                    f"❌ Deleting "
                                    f"`{filename}` failed.\n\n"
                                    f"**Reason:** {exc}"
                                ),
                            }

                            st.rerun()

            st.divider()


    # ========================================================
    # SUPPORTED FORMATS
    # ========================================================

    with st.expander(
        "📋 Supported Formats"
    ):

        st.markdown(
            """
            **PDF**

            `.pdf`

            **Microsoft Word**

            `.docx`

            **Plain Text**

            `.txt`

            Documents are automatically:

            1. Saved to the document directory
            2. Parsed
            3. Cleaned
            4. Chunked
            5. Embedded
            6. Stored in ChromaDB
            7. Available for RAG questions
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DocMind — RAG Knowledge Assistant | "
    "Semantic Retrieval + Grounded Generation + "
    "Source Attribution"
)