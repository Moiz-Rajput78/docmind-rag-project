from pathlib import Path
import sys
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

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

from app.analytics.evaluation_data import (
    get_evaluation_data,
    percent,
    render_metric_delta,
    safe_float,
    available_columns,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocMind — RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)


# ============================================================
# THEME STATE + CUSTOM FRONTEND CSS
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "dark"


def toggle_theme():
    st.session_state.theme = (
        "dark" if st.session_state.theme == "light" else "light"
    )


def set_theme_from_documents_toggle():
    """Keep the Documents-panel theme switch in sync with the app theme."""
    st.session_state.theme = (
        "light"
        if st.session_state.documents_theme_toggle
        else "dark"
    )


if "documents_theme_toggle" not in st.session_state:
    st.session_state.documents_theme_toggle = (
        st.session_state.theme == "light"
    )


if st.session_state.theme == "dark":
    THEME = {
        "bg": "#101412",
        "surface": "#171d1a",
        "surface2": "#1d2521",
        "surface3": "#232d28",
        "text": "#e9f1ed",
        "heading": "#f7fbf9",
        "muted": "#9eaea5",
        "border": "#2d3933",
        "primary": "#35c98a",
        "primary2": "#1ea86f",
        "soft": "#183128",
        "success": "#42d79a",
        "warning": "#e9b85f",
        "danger": "#ef7373",
        "input": "#121815",
        "input_text": "#f4f8f6",
        "placeholder": "#84968c",
        "sidebar": "#111714",
        "sidebar2": "#1a241f",
    }
else:
    THEME = {
        "bg": "#ffffff",
        "surface": "#ffffff",
        "surface2": "#f7faf8",
        "surface3": "#eef5f1",
        "text": "#24312b",
        "heading": "#111a16",
        "muted": "#65756d",
        "border": "#d9e2dd",
        "primary": "#159a68",
        "primary2": "#0f7f55",
        "soft": "#eaf7f1",
        "success": "#12845c",
        "warning": "#a86d12",
        "danger": "#bd4747",
        "input": "#ffffff",
        "input_text": "#24312b",
        "placeholder": "#788a81",
        "sidebar": "#ffffff",
        "sidebar2": "#f3f7f5",
    }

st.markdown(
    f"""
    <style>
    :root {{
        --dm-bg:{THEME['bg']};
        --dm-surface:{THEME['surface']};
        --dm-surface2:{THEME['surface2']};
        --dm-surface3:{THEME['surface3']};
        --dm-text:{THEME['text']};
        --dm-heading:{THEME['heading']};
        --dm-muted:{THEME['muted']};
        --dm-border:{THEME['border']};
        --dm-primary:{THEME['primary']};
        --dm-primary2:{THEME['primary2']};
        --dm-soft:{THEME['soft']};
        --dm-success:{THEME['success']};
        --dm-warning:{THEME['warning']};
        --dm-danger:{THEME['danger']};
        --dm-input:{THEME['input']};
        --dm-input-text:{THEME['input_text']};
        --dm-placeholder:{THEME['placeholder']};
        --dm-sidebar-bg:{'#ffffff' if st.session_state.theme == 'light' else THEME['sidebar']};
        --dm-sidebar-text:{'#24312b' if st.session_state.theme == 'light' else '#edf6f1'};
        --dm-sidebar-muted:{'#66776e' if st.session_state.theme == 'light' else '#9fb0a7'};
        --dm-sidebar-value:{'#111a16' if st.session_state.theme == 'light' else '#ffffff'};
        --dm-sidebar-border:{'#dde5e1' if st.session_state.theme == 'light' else 'rgba(255,255,255,.09)'};
        --dm-sidebar-hover:{'#eef8f3' if st.session_state.theme == 'light' else '#1a241f'};

    }}

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {{
        background:var(--dm-bg) !important;
        color:var(--dm-text) !important;
    }}

    [data-testid="stHeader"] {{
        background:var(--dm-bg) !important;
        border-bottom:1px solid var(--dm-border) !important;
    }}

    /* Keep ALL DocMind content below Streamlit's fixed top toolbar/header. */
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] .block-container,
    .main .block-container,
    .block-container {{
        max-width:1600px !important;
        width:100% !important;
        box-sizing:border-box !important;
        padding-top:4.4rem !important;
        padding-bottom:3rem !important;
        margin-top:0 !important;
    }}

    /* Never allow the custom top area to collapse behind the toolbar. */
    .dm-main-top-spacer {{
        display:block !important;
        height:0 !important;
        min-height:0 !important;
    }}

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {{
        background:var(--dm-sidebar-bg) !important;
        border-right:1px solid var(--dm-sidebar-border) !important;
    }}

    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] > div {{
        background:var(--dm-sidebar-bg) !important;
    }}

    /* Sidebar uses a true light palette in Light mode and a dark palette in Dark mode. */
    [data-testid="stSidebar"] * {{
        color:var(--dm-sidebar-text) !important;
    }}

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {{
        color:var(--dm-sidebar-muted) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMetricValue"] {{
        color:var(--dm-sidebar-value) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {{
        color:var(--dm-sidebar-muted) !important;
    }}

    .dm-brand {{
        display:flex;
        align-items:center;
        gap:12px;
        margin:3px 0 18px 1px;
    }}

    .dm-brand-icon {{
        width:42px;height:42px;border-radius:12px;
        display:flex;align-items:center;justify-content:center;
        background:linear-gradient(135deg,#35c98a,#159a68);
        color:#fff !important;font-size:20px;
        box-shadow:0 8px 24px rgba(21,154,104,.24);
    }}

    .dm-brand-title {{color:var(--dm-sidebar-text) !important;font-size:1.18rem;font-weight:800;line-height:1.05;}}
    .dm-brand-sub {{color:var(--dm-sidebar-muted) !important;font-size:.71rem;margin-top:4px;}}

    .dm-nav-label {{
        color:var(--dm-sidebar-muted) !important;font-size:.68rem;font-weight:800;
        text-transform:uppercase;letter-spacing:.08em;margin:10px 0 5px;
    }}

    /* Streamlit radio navigation */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {{
        color:var(--dm-sidebar-muted) !important;font-size:.7rem;font-weight:800 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{
        gap:4px !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] {{
        min-height:38px !important;
        padding:7px 10px !important;
        border-radius:9px !important;
        background:transparent !important;
        color:var(--dm-sidebar-text) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] span,
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] div {{
        color:var(--dm-sidebar-text) !important;
        font-size:.82rem !important;
        font-weight:650 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"][aria-checked="true"] {{
        background:var(--dm-sidebar-hover) !important;
        border:1px solid var(--dm-sidebar-border) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"]:hover {{
        background:var(--dm-sidebar-hover) !important;
    }}

    [data-testid="stSidebar"] button {{
        color:var(--dm-sidebar-text) !important;
        border-color:var(--dm-sidebar-border) !important;
    }}

    [data-testid="stSidebar"] button:hover {{
        border-color:var(--dm-primary) !important;
    }}

    .dm-theme-card {{
        background:var(--dm-sidebar-hover);
        border:1px solid var(--dm-sidebar-border);
        border-radius:12px;padding:10px 11px 4px;margin:7px 0 13px;
    }}

    /* ---------- Header ---------- */
    .dm-topbar {{display:flex;align-items:center;gap:12px;margin-bottom:13px;}}

    .dm-search-box {{
        background:var(--dm-surface) !important;
        border:1px solid var(--dm-border) !important;
        color:var(--dm-muted) !important;
        border-radius:12px;padding:11px 15px;font-size:.83rem;
        box-shadow:0 4px 18px rgba(35,61,110,.04);
    }}
/* ---------- Boxed AI chat ---------- */
    .dm-chat-box-title {{
        color: var(--dm-heading);
        font-size: 1.12rem;
        font-weight: 800;
        margin: 0 0 2px 0;
    }}

    .dm-chat-box-subtitle {{
        color: var(--dm-muted);
        font-size: .78rem;
        margin-bottom: 10px;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker) {{
        border: 1px solid var(--dm-border) !important;
        border-radius: 16px !important;
        background: var(--dm-surface) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,.08);
    }}

    [data-testid="stChatMessage"] {{background:var(--dm-surface) !important;border:1px solid var(--dm-border) !important;border-radius:13px !important;}}
    [data-testid="stChatMessage"] p {{color:var(--dm-text) !important;}}

    /* Theme toggle */
    .dm-theme-note {{color:#a9b9d5 !important;font-size:.7rem;line-height:1.4;}}

    /* High-contrast safeguards for Streamlit's light theme. */
    [data-testid="stMain"] .stMarkdown,
    [data-testid="stMain"] .stMarkdown p,
    [data-testid="stMain"] .stMarkdown li,
    [data-testid="stMain"] .stText,
    [data-testid="stMain"] label,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p {{
        color:var(--dm-text) !important;
        opacity:1 !important;
        visibility:visible !important;
    }}
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] h5,
    [data-testid="stMain"] h6 {{
        color:var(--dm-heading) !important;
        opacity:1 !important;
        visibility:visible !important;
    }}
    [data-testid="stMain"] [data-testid="stAlert"] {{
        opacity:1 !important;visibility:visible !important;
    }}

    @media (max-width:900px) {{
        [data-testid="stAppViewContainer"] .block-container,
        [data-testid="stMainBlockContainer"],
        .block-container {{
            padding-top:5.5rem !important;
            padding-left:1rem !important;
            padding-right:1rem !important;
        }}
        .dm-welcome {{font-size:1.4rem;}}
    }}
    
    /* ===== FINAL VISIBILITY + RESPONSIVE LAYOUT FIXES ===== */

    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] .block-container,
    .main .block-container,
    .block-container {{
        max-width: 1600px !important;
        width: 100% !important;
        padding-top: 4.8rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        padding-bottom: 1rem !important;
    }}

    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {{
        gap: 1rem !important;
        align-items: stretch !important;
    }}

    /* Right Documents panel: wider and readable. */
    div[data-testid="column"]:has(.dm-right-panel-marker) {{
        min-width: 340px !important;
        width: 340px !important;
        flex: 0 0 340px !important;
        padding: 8px 8px 10px 14px !important;
    }}

    /* Upload box uses the actual app palette instead of a white slab. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] {{
        background: var(--dm-surface2) !important;
        border: 1px dashed var(--dm-border) !important;
        border-radius: 12px !important;
        min-height: 116px !important;
        padding: 14px !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] *,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploader"] * {{
        color: var(--dm-text) !important;
        opacity: 1 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button {{
        background: var(--dm-primary) !important;
        color: #ffffff !important;
        border: 1px solid var(--dm-primary) !important;
        border-radius: 9px !important;
        min-height: 38px !important;
        opacity: 1 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button * {{
        color: #ffffff !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker) .stButton > button {{
        background: var(--dm-surface2) !important;
        color: var(--dm-text) !important;
        border: 1px solid var(--dm-border) !important;
        min-height: 38px !important;
        opacity: 1 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker) .stButton > button:hover {{
        border-color: var(--dm-primary) !important;
        color: var(--dm-primary) !important;
    }}

    /* Chat: remove the oversized empty area. */
    .dm-empty-state {{
        min-height: 235px !important;
        padding: 24px 22px !important;
    }}

    .dm-empty-icon {{
        margin-bottom: 10px !important;
    }}

    /* Example questions must be readable in full. */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker)
    .stButton > button {{
        min-height: 48px !important;
        height: auto !important;
        white-space: normal !important;
        line-height: 1.25 !important;
        padding: 9px 11px !important;
        background: var(--dm-surface2) !important;
        color: var(--dm-text) !important;
        border: 1px solid var(--dm-border) !important;
        opacity: 1 !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker)
    .stButton > button p {{
        color: var(--dm-text) !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }}

    /* Chat input follows dark/light palette and stays clearly visible. */
    [data-testid="stChatInput"] {{
        background: var(--dm-input) !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        overflow: hidden !important;
        margin-top: 10px !important;
    }}

    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus {{
        background: var(--dm-input) !important;
        color: var(--dm-input-text) !important;
        caret-color: var(--dm-primary) !important;
    }}

    [data-testid="stChatInput"] textarea {{
        min-height: 52px !important;
        padding: 14px 52px 12px 14px !important;
        border: 0 !important;
        outline: none !important;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: var(--dm-placeholder) !important;
        opacity: 1 !important;
    }}

    [data-testid="stChatInput"] button {{
        background: var(--dm-primary) !important;
        color: #ffffff !important;
        border-radius: 9px !important;
        opacity: 1 !important;
    }}

    [data-testid="stChatInput"] button svg {{
        color: #ffffff !important;
        fill: #ffffff !important;
    }}

    [data-testid="stChatMessage"] {{
        padding: 12px 14px !important;
        margin-bottom: 9px !important;
    }}

    @media (max-width: 1200px) {{
        div[data-testid="column"]:has(.dm-right-panel-marker) {{
            min-width: 300px !important;
            width: 300px !important;
            flex-basis: 300px !important;
        }}
    }}

    @media (max-width: 900px) {{
        div[data-testid="column"]:has(.dm-right-panel-marker) {{
            min-width: 100% !important;
            width: 100% !important;
            flex: 1 1 100% !important;
            position: static !important;
        }}

        .dm-empty-state {{
            min-height: 200px !important;
        }}
    }}

</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================
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

def add_to_conversation(
    result,
):
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

            "retrieval_mode": result.get(
                "retrieval_mode",
                "Unknown",
            ),

            "timings_ms": result.get(
                "timings_ms",
                {},
            ),

            "observability": result.get(
                "observability",
                {},
            ),
        }
    )


# ============================================================
# HELPER — FORMAT TIME
# ============================================================

def format_duration_ms(
    value,
):
    """
    Convert milliseconds into a human-readable value.
    """

    if value is None:
        return "N/A"

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return "N/A"

    if value >= 1000:
        return f"{value / 1000:.2f} s"

    return f"{value:.2f} ms"


# ============================================================
# HELPER — SOURCE CARD
# ============================================================

def render_source_card(
    source,
):
    """
    Render a source using native Streamlit components.
    """

    source_number = source.get(
        "source_number",
        "?",
    )

    filename = source.get(
        "filename",
        "Unknown",
    )

    page = source.get(
        "page"
    )

    category = source.get(
        "category",
        "unknown",
    )

    chunk_number = source.get(
        "chunk_number",
        "?",
    )

    distance = source.get(
        "distance"
    )

    if (
        page is None
        or page == -1
    ):
        page_text = "Page: N/A"

    else:
        page_text = f"Page: {page}"

    if distance is not None:

        try:

            distance_text = (
                f"{float(distance):.4f}"
            )

        except (
            TypeError,
            ValueError,
        ):

            distance_text = "N/A"

    else:

        distance_text = "N/A"

    with st.container(
        border=True
    ):

        st.markdown(
            f"**[{source_number}] "
            f"📄 {filename}**"
        )

        st.caption(
            f"Category: {category} • "
            f"{page_text} • "
            f"Chunk: {chunk_number} • "
            f"Distance: {distance_text}"
        )


# ============================================================
# HELPER — RETRIEVAL DETAILS
# ============================================================

def render_retrieval_details(
    result,
    expanded=False,
):
    """
    Render professional retrieval and performance
    information for a completed RAG response.

    Uses only native Streamlit components.
    """

    if not result:
        return

    timings = result.get(
        "timings_ms",
        {},
    ) or {}

    observability = result.get(
        "observability",
        {},
    ) or {}

    retrieval_mode = result.get(
        "retrieval_mode",
        "Unknown",
    )

    top_k_value = result.get(
        "top_k",
        3,
    )

    retrieved_count = result.get(
        "retrieved_count",
        observability.get(
            "retrieved_count",
            0,
        ),
    )

    unique_source_count = observability.get(
        "unique_source_count",
        0,
    )

    top_hybrid_score = observability.get(
        "top_hybrid_score"
    )

    top_reranker_score = observability.get(
        "top_reranker_score"
    )

    abstained = observability.get(
        "abstained",
        False,
    )

    with st.expander(
        "🔎 Retrieval Details",
        expanded=expanded,
    ):

        st.markdown(
            "### 🔄 Retrieval Pipeline"
        )

        st.info(
            retrieval_mode
        )

        st.markdown(
            "### 📊 Retrieval Overview"
        )

        (
            overview_col1,
            overview_col2,
            overview_col3,
            overview_col4,
        ) = st.columns(4)

        with overview_col1:

            st.metric(
                "Top-K",
                top_k_value,
            )

        with overview_col2:

            st.metric(
                "Retrieved Chunks",
                retrieved_count,
            )

        with overview_col3:

            st.metric(
                "Unique Sources",
                unique_source_count,
            )

        with overview_col4:

            st.metric(
                "Abstained",
                "Yes" if abstained else "No",
            )

        st.markdown(
            "### ⚡ Performance"
        )

        (
            performance_col1,
            performance_col2,
            performance_col3,
            performance_col4,
        ) = st.columns(4)

        with performance_col1:

            st.metric(
                "Retrieval",
                format_duration_ms(
                    timings.get(
                        "retrieval"
                    )
                ),
            )

        with performance_col2:

            st.metric(
                "Context Build",
                format_duration_ms(
                    timings.get(
                        "context_build"
                    )
                ),
            )

        with performance_col3:

            st.metric(
                "Generation",
                format_duration_ms(
                    timings.get(
                        "generation"
                    )
                ),
            )

        with performance_col4:

            st.metric(
                "Total",
                format_duration_ms(
                    timings.get(
                        "total"
                    )
                ),
            )

        st.caption(
            "Retrieval includes dense/BM25 candidate retrieval "
            "and cross-encoder reranking when enabled."
        )

        st.markdown(
            "### 🎯 Retrieval Quality Signals"
        )

        score_col1, score_col2 = st.columns(2)

        with score_col1:

            if top_hybrid_score is None:

                hybrid_display = "N/A"

            else:

                try:

                    hybrid_display = (
                        f"{float(top_hybrid_score):.4f}"
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    hybrid_display = "N/A"

            st.metric(
                "Top Hybrid Score",
                hybrid_display,
            )

        with score_col2:

            if top_reranker_score is None:

                reranker_display = "N/A"

            else:

                try:

                    reranker_display = (
                        f"{float(top_reranker_score):.4f}"
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    reranker_display = "N/A"

            st.metric(
                "Top Reranker Score",
                reranker_display,
            )

        st.markdown(
            "### ❓ Query"
        )

        st.info(
            result.get(
                "question",
                "",
            )
        )


# ============================================================
# HELPER — MISSING EVALUATION DATA
# ============================================================

def render_missing_evaluation_message(
    filename,
):

    st.info(
        f"Evaluation data `{filename}` "
        "was not found in the `docs/` directory."
    )


# ============================================================
# ANSWER QUALITY — LOAD RESULTS
# ============================================================

ANSWER_QUALITY_RESULTS_PATH = (
    PROJECT_ROOT
    / "docs"
    / "answer_evaluation_results.csv"
)


def load_answer_quality_results():
    """
    Load the existing answer-quality evaluation CSV.

    This function only reads previously generated results.
    It does NOT call Ollama or run the 30-question evaluation.
    """

    if not ANSWER_QUALITY_RESULTS_PATH.exists():
        return pd.DataFrame()

    try:

        dataframe = pd.read_csv(
            ANSWER_QUALITY_RESULTS_PATH
        )

    except Exception:

        return pd.DataFrame()

    return dataframe


# ============================================================
# ANSWER QUALITY — SAFE NUMERIC HELPERS
# ============================================================

def answer_quality_numeric(
    dataframe,
    column,
):
    """
    Convert a dataframe column to numeric safely.
    """

    if column not in dataframe.columns:
        return pd.Series(
            dtype="float64"
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def answer_quality_mean(
    dataframe,
    column,
    default=0.0,
):
    """
    Calculate a safe mean for an answer-quality metric.
    """

    values = answer_quality_numeric(
        dataframe,
        column,
    ).dropna()

    if values.empty:
        return default

    return float(
        values.mean()
    )


def answer_quality_percentage(
    dataframe,
    column,
):
    """
    Calculate the percentage of truthy values.
    """

    if column not in dataframe.columns:
        return None

    values = dataframe[column]

    if values.empty:
        return None

    normalized = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )

    truthy = normalized.isin(
        [
            "true",
            "1",
            "yes",
        ]
    )

    return float(
        truthy.mean()
    )


# ============================================================
# ANSWER QUALITY EVALUATION DASHBOARD
# ============================================================

def render_answer_quality_dashboard():
    """
    Render the answer-quality evaluation dashboard.

    Results are loaded from:
        docs/answer_evaluation_results.csv

    The dashboard does not execute the evaluation itself.
    """

    st.markdown(
        "## 📝 Answer Quality Evaluation"
    )

    st.write(
        "Analyze the quality of DocMind's generated answers "
        "using the completed 30-question answer evaluation."
    )

    st.caption(
        "This dashboard reads the existing "
        "`docs/answer_evaluation_results.csv` file. "
        "Opening or refreshing this page does not trigger "
        "new Ollama Cloud generations."
    )

    dataframe = load_answer_quality_results()

    if dataframe.empty:

        st.warning(
            "No answer-quality evaluation results were found."
        )

        st.info(
            "Run the answer evaluation first with:\n\n"
            "`python -m app.evaluation.answer_evaluator`"
        )

        return

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.markdown(
        "### 📋 Evaluation Dataset"
    )

    total_questions = len(
        dataframe
    )

    difficulty_count = (
        dataframe["difficulty"].nunique()
        if "difficulty" in dataframe.columns
        else 0
    )

    category_count = (
        dataframe["category"].nunique()
        if "category" in dataframe.columns
        else 0
    )

    successful_answers = total_questions

    if "generation_error" in dataframe.columns:

        generation_errors = (
            dataframe["generation_error"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        successful_answers = (
            total_questions
            - generation_errors
        )

    dataset_cols = st.columns(4)

    with dataset_cols[0]:

        st.metric(
            "Questions",
            total_questions,
        )

    with dataset_cols[1]:

        st.metric(
            "Difficulty Levels",
            difficulty_count,
        )

    with dataset_cols[2]:

        st.metric(
            "Categories",
            category_count,
        )

    with dataset_cols[3]:

        st.metric(
            "Successful Evaluations",
            successful_answers,
        )

    st.divider()

    # ========================================================
    # OVERALL QUALITY METRICS
    # ========================================================

    st.markdown(
        "### 🎯 Overall Answer Quality"
    )

    source_hit = answer_quality_percentage(
        dataframe,
        "expected_source_found",
    )

    abstention_accuracy = answer_quality_percentage(
        dataframe,
        "abstention_correct",
    )

    citation_correctness = answer_quality_percentage(
        dataframe,
        "citation_correct",
    )

    suspicious_rate = answer_quality_percentage(
        dataframe,
        "suspicious_content",
    )

    groundedness = answer_quality_mean(
        dataframe,
        "groundedness_score",
    )

    context_relevance = answer_quality_mean(
        dataframe,
        "context_relevance_score",
    )

    answer_length = answer_quality_mean(
        dataframe,
        "answer_length",
    )

    retrieval_ms = answer_quality_mean(
        dataframe,
        "retrieval_ms",
    )

    generation_ms = answer_quality_mean(
        dataframe,
        "generation_ms",
    )

    total_ms = answer_quality_mean(
        dataframe,
        "total_ms",
    )

    quality_cols = st.columns(5)

    with quality_cols[0]:

        st.metric(
            "Expected Source Hit",
            (
                f"{source_hit:.1%}"
                if source_hit is not None
                else "N/A"
            ),
        )

    with quality_cols[1]:

        st.metric(
            "Groundedness",
            f"{groundedness:.3f}",
        )

    with quality_cols[2]:

        st.metric(
            "Context Relevance",
            f"{context_relevance:.3f}",
        )

    with quality_cols[3]:

        st.metric(
            "Citation Correctness",
            (
                f"{citation_correctness:.1%}"
                if citation_correctness is not None
                else "N/A"
            ),
        )

    with quality_cols[4]:

        st.metric(
            "Suspicious Content",
            (
                f"{suspicious_rate:.1%}"
                if suspicious_rate is not None
                else "N/A"
            ),
        )

    quality_cols2 = st.columns(5)

    with quality_cols2[0]:

        st.metric(
            "Abstention Accuracy",
            (
                f"{abstention_accuracy:.1%}"
                if abstention_accuracy is not None
                else "N/A"
            ),
        )

    with quality_cols2[1]:

        st.metric(
            "Avg Answer Length",
            f"{answer_length:.0f} chars",
        )

    with quality_cols2[2]:

        st.metric(
            "Avg Retrieval",
            format_duration_ms(
                retrieval_ms
            ),
        )

    with quality_cols2[3]:

        st.metric(
            "Avg Generation",
            format_duration_ms(
                generation_ms
            ),
        )

    with quality_cols2[4]:

        st.metric(
            "Avg Total",
            format_duration_ms(
                total_ms
            ),
        )

    st.caption(
        "Groundedness and context relevance are evidence-support "
        "metrics produced by the evaluation framework. They should "
        "not be interpreted as a substitute for human factual review."
    )

    st.divider()

    # ========================================================
    # QUALITY METRIC CHART
    # ========================================================

    st.markdown(
        "### 📊 Answer Quality Metrics"
    )

    quality_chart_data = pd.DataFrame(
        {
            "Score": [
                (
                    source_hit
                    if source_hit is not None
                    else 0
                ),
                groundedness,
                context_relevance,
                (
                    citation_correctness
                    if citation_correctness is not None
                    else 0
                ),
                (
                    abstention_accuracy
                    if abstention_accuracy is not None
                    else 0
                ),
            ]
        },
        index=[
            "Expected Source Hit",
            "Groundedness",
            "Context Relevance",
            "Citation Correctness",
            "Abstention Accuracy",
        ],
    )

    st.bar_chart(
        quality_chart_data
    )

    st.caption(
        "Percentage-based metrics are shown on a 0–1 scale "
        "in the chart."
    )

    st.divider()

    # ========================================================
    # DIFFICULTY ANALYSIS
    # ========================================================

    st.markdown(
        "### 🧠 Answer Quality by Difficulty"
    )

    if "difficulty" not in dataframe.columns:

        st.info(
            "Difficulty information is not available."
        )

    else:

        difficulty_quality = (
            dataframe
            .groupby("difficulty")
            .agg(
                Questions=(
                    "question",
                    "count",
                ),
                Groundedness=(
                    "groundedness_score",
                    "mean",
                ),
                Context_Relevance=(
                    "context_relevance_score",
                    "mean",
                ),
                Citation_Correctness=(
                    "citation_correct",
                    lambda values:
                    (
                        pd.to_numeric(
                            values,
                            errors="coerce",
                        )
                        .mean()
                        if pd.to_numeric(
                            values,
                            errors="coerce",
                        ).notna().any()
                        else None
                    ),
                ),
                Avg_Answer_Length=(
                    "answer_length",
                    "mean",
                ),
                Avg_Total_ms=(
                    "total_ms",
                    "mean",
                ),
            )
        )

        st.dataframe(
            difficulty_quality.style.format(
                {
                    "Groundedness": "{:.3f}",
                    "Context_Relevance": "{:.3f}",
                    "Citation_Correctness": "{:.1%}",
                    "Avg_Answer_Length": "{:.0f}",
                    "Avg_Total_ms": "{:.0f}",
                }
            ),
            width="stretch",
        )

        difficulty_metric = st.selectbox(
            "Difficulty metric",
            [
                "Groundedness",
                "Context_Relevance",
                "Citation_Correctness",
                "Avg_Answer_Length",
                "Avg_Total_ms",
            ],
            format_func=(
                lambda value:
                value.replace(
                    "_",
                    " ",
                )
            ),
            key="answer_quality_difficulty_metric",
        )

        st.bar_chart(
            difficulty_quality[
                [difficulty_metric]
            ]
        )

    st.divider()

    # ========================================================
    # CATEGORY ANALYSIS
    # ========================================================

    st.markdown(
        "### 📚 Answer Quality by Knowledge Category"
    )

    if "category" not in dataframe.columns:

        st.info(
            "Category information is not available."
        )

    else:

        category_quality = (
            dataframe
            .groupby("category")
            .agg(
                Questions=(
                    "question",
                    "count",
                ),
                Groundedness=(
                    "groundedness_score",
                    "mean",
                ),
                Context_Relevance=(
                    "context_relevance_score",
                    "mean",
                ),
                Avg_Answer_Length=(
                    "answer_length",
                    "mean",
                ),
                Avg_Retrieval_ms=(
                    "retrieval_ms",
                    "mean",
                ),
                Avg_Generation_ms=(
                    "generation_ms",
                    "mean",
                ),
                Avg_Total_ms=(
                    "total_ms",
                    "mean",
                ),
            )
        )

        st.dataframe(
            category_quality.style.format(
                {
                    "Groundedness": "{:.3f}",
                    "Context_Relevance": "{:.3f}",
                    "Avg_Answer_Length": "{:.0f}",
                    "Avg_Retrieval_ms": "{:.0f}",
                    "Avg_Generation_ms": "{:.0f}",
                    "Avg_Total_ms": "{:.0f}",
                }
            ),
            width="stretch",
        )

        category_metric = st.selectbox(
            "Category metric",
            [
                "Groundedness",
                "Context_Relevance",
                "Avg_Answer_Length",
                "Avg_Retrieval_ms",
                "Avg_Generation_ms",
                "Avg_Total_ms",
            ],
            format_func=(
                lambda value:
                value.replace(
                    "_",
                    " ",
                )
            ),
            key="answer_quality_category_metric",
        )

        st.bar_chart(
            category_quality[
                [category_metric]
            ]
        )

    st.divider()

    # ========================================================
    # GROUNDEDNESS VS CONTEXT RELEVANCE
    # ========================================================

    st.markdown(
        "### 🔗 Groundedness vs Context Relevance"
    )

    quality_comparison = pd.DataFrame(
        {
            "Groundedness": [
                groundedness
            ],
            "Context Relevance": [
                context_relevance
            ],
        }
    )

    st.bar_chart(
        quality_comparison
    )

    st.caption(
        "Groundedness measures how strongly generated answers "
        "are supported by retrieved evidence according to the "
        "evaluation proxy. Context relevance measures how "
        "relevant the retrieved evidence is to the question."
    )

    st.divider()

    # ========================================================
    # LATENCY ANALYSIS
    # ========================================================

    st.markdown(
        "### ⚡ Answer Generation Latency"
    )

    latency_data = pd.DataFrame(
        {
            "Average Time (ms)": [
                retrieval_ms,
                generation_ms,
                total_ms,
            ]
        },
        index=[
            "Retrieval",
            "Generation",
            "End-to-End",
        ],
    )

    st.bar_chart(
        latency_data
    )

    latency_cols = st.columns(3)

    with latency_cols[0]:

        st.metric(
            "Retrieval",
            format_duration_ms(
                retrieval_ms
            ),
        )

    with latency_cols[1]:

        st.metric(
            "Generation",
            format_duration_ms(
                generation_ms
            ),
        )

    with latency_cols[2]:

        st.metric(
            "End-to-End",
            format_duration_ms(
                total_ms
            ),
        )

    st.caption(
        "End-to-end latency includes retrieval, context building, "
        "and LLM generation."
    )

    st.divider()

    # ========================================================
    # RETRIEVAL MODE ANALYSIS
    # ========================================================

    st.markdown(
        "### 🔀 Retrieval Mode Analysis"
    )

    if "retrieval_mode" in dataframe.columns:

        retrieval_modes = (
            dataframe[
                "retrieval_mode"
            ]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .to_frame(
                "Questions"
            )
        )

        st.dataframe(
            retrieval_modes,
            width="stretch",
        )

        st.bar_chart(
            retrieval_modes
        )

    else:

        st.info(
            "Retrieval mode information is not available."
        )

    st.divider()

    # ========================================================
    # TOP-K ANALYSIS
    # ========================================================

    st.markdown(
        "### 🔢 Top-K Evaluation Analysis"
    )

    if "top_k" in dataframe.columns:

        top_k_df = (
            dataframe
            .groupby("top_k")
            .agg(
                Questions=(
                    "question",
                    "count",
                ),
                Groundedness=(
                    "groundedness_score",
                    "mean",
                ),
                Context_Relevance=(
                    "context_relevance_score",
                    "mean",
                ),
                Avg_Total_ms=(
                    "total_ms",
                    "mean",
                ),
            )
        )

        st.dataframe(
            top_k_df.style.format(
                {
                    "Groundedness": "{:.3f}",
                    "Context_Relevance": "{:.3f}",
                    "Avg_Total_ms": "{:.0f}",
                }
            ),
            width="stretch",
        )

        st.line_chart(
            top_k_df[
                [
                    "Groundedness",
                    "Context_Relevance",
                ]
            ]
        )

    else:

        st.info(
            "Top-K information is not available."
        )

    st.divider()

    # ========================================================
    # WEAKEST QUESTIONS
    # ========================================================

    st.markdown(
        "### ⚠️ Questions Requiring Further Investigation"
    )

    weakest_df = dataframe.copy()

    if (
        "groundedness_score" in weakest_df.columns
        and "context_relevance_score" in weakest_df.columns
    ):

        weakest_df[
            "evidence_quality"
        ] = (
            pd.to_numeric(
                weakest_df[
                    "groundedness_score"
                ],
                errors="coerce",
            ).fillna(0)
            +
            pd.to_numeric(
                weakest_df[
                    "context_relevance_score"
                ],
                errors="coerce",
            ).fillna(0)
        ) / 2

        weakest_df = (
            weakest_df
            .sort_values(
                "evidence_quality"
            )
            .head(10)
        )

        weakest_columns = [
            "question",
            "difficulty",
            "category",
            "groundedness_score",
            "context_relevance_score",
            "citation_correct",
            "answer_length",
        ]

        available_weakest = available_columns(
            weakest_df,
            weakest_columns,
        )

        if available_weakest:

            display_weakest = (
                weakest_df[
                    available_weakest
                ]
                .copy()
            )

            rename_map = {
                "groundedness_score": "Groundedness",
                "context_relevance_score": "Context Relevance",
                "citation_correct": "Citation Correct",
                "answer_length": "Answer Length",
            }

            display_weakest = (
                display_weakest
                .rename(
                    columns=rename_map
                )
            )

            st.dataframe(
                display_weakest,
                width="stretch",
                hide_index=True,
            )

        st.caption(
            "These questions have the lowest combined "
            "groundedness and context-relevance scores. "
            "They are useful candidates for retrieval and "
            "prompt-quality investigation."
        )

    else:

        st.info(
            "Evidence-quality columns are not available."
        )

    st.divider()

    # ========================================================
    # STRONGEST QUESTIONS
    # ========================================================

    st.markdown(
        "### ✅ Strong Evidence-Support Questions"
    )

    strongest_df = dataframe.copy()

    if (
        "groundedness_score" in strongest_df.columns
        and "context_relevance_score" in strongest_df.columns
    ):

        strongest_df[
            "evidence_quality"
        ] = (
            pd.to_numeric(
                strongest_df[
                    "groundedness_score"
                ],
                errors="coerce",
            ).fillna(0)
            +
            pd.to_numeric(
                strongest_df[
                    "context_relevance_score"
                ],
                errors="coerce",
            ).fillna(0)
        ) / 2

        strongest_df = (
            strongest_df
            .sort_values(
                "evidence_quality",
                ascending=False,
            )
            .head(10)
        )

        strongest_columns = [
            "question",
            "difficulty",
            "category",
            "groundedness_score",
            "context_relevance_score",
            "citation_correct",
            "answer_length",
        ]

        available_strongest = available_columns(
            strongest_df,
            strongest_columns,
        )

        if available_strongest:

            display_strongest = (
                strongest_df[
                    available_strongest
                ]
                .copy()
            )

            rename_map = {
                "groundedness_score": "Groundedness",
                "context_relevance_score": "Context Relevance",
                "citation_correct": "Citation Correct",
                "answer_length": "Answer Length",
            }

            display_strongest = (
                display_strongest
                .rename(
                    columns=rename_map
                )
            )

            st.dataframe(
                display_strongest,
                width="stretch",
                hide_index=True,
            )

    else:

        st.info(
            "Evidence-quality columns are not available."
        )

    st.divider()

    # ========================================================
    # ANSWER QUALITY DISTRIBUTION
    # ========================================================

    st.markdown(
        "### 📈 Groundedness Distribution"
    )

    if "groundedness_score" in dataframe.columns:

        groundedness_values = (
            pd.to_numeric(
                dataframe[
                    "groundedness_score"
                ],
                errors="coerce",
            )
            .dropna()
        )

        if not groundedness_values.empty:
            histogram_df = pd.DataFrame(
                {
                    "Groundedness": (
                        groundedness_values
                        .round(2)
                    )
                }
            )

            histogram_df = (
                histogram_df
                .value_counts()
                .sort_index()
                .rename("Count")
                .reset_index()
            )

            st.bar_chart(
                histogram_df,
                x="Groundedness",
                y="Count",
            )
    if "context_relevance_score" in dataframe.columns:

        st.markdown(
            "### 📈 Context Relevance Distribution"
        )

        relevance_values = (
            pd.to_numeric(
                dataframe[
                    "context_relevance_score"
                ],
                errors="coerce",
            )
            .dropna()
        )

        if not relevance_values.empty:
            relevance_histogram = pd.DataFrame(
                {
                    "Context Relevance": (
                        relevance_values
                        .round(2)
                    )
                }
            )

            relevance_histogram = (
                relevance_histogram
                .value_counts()
                .sort_index()
                .rename("Count")
                .reset_index()
            )

            st.bar_chart(
                relevance_histogram,
                x="Context Relevance",
                y="Count",
            )
    st.divider()

    # ========================================================
    # ANSWER QUALITY DETAILED RESULTS
    # ========================================================

    st.markdown(
        "### 🔬 Detailed Question-Level Evaluation"
    )

    detail_columns = [
        "question",
        "difficulty",
        "category",
        "expected_sources",
        "matched_sources",
        "expected_source_found",
        "groundedness_score",
        "context_relevance_score",
        "citation_score",
        "citation_correct",
        "abstention_expected",
        "abstained",
        "abstention_correct",
        "suspicious_content",
        "answer_length",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "retrieval_mode",
        "top_k",
    ]

    available_detail_columns = available_columns(
        dataframe,
        detail_columns,
    )

    if available_detail_columns:

        detailed_df = (
            dataframe[
                available_detail_columns
            ]
            .copy()
        )

        st.dataframe(
            detailed_df,
            width="stretch",
            hide_index=True,
        )

    else:

        st.dataframe(
            dataframe,
            width="stretch",
            hide_index=True,
        )

    st.divider()

    # ========================================================
    # SELECT INDIVIDUAL QUESTION
    # ========================================================

    st.markdown(
        "### 🔍 Inspect an Individual Evaluation"
    )

    if "question" in dataframe.columns:

        question_options = (
            dataframe[
                "question"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        if question_options:

            selected_question = st.selectbox(
                "Select an evaluated question",
                question_options,
                key="answer_quality_question_selector",
            )

            selected_rows = dataframe[
                dataframe[
                    "question"
                ].astype(str)
                == selected_question
            ]

            if not selected_rows.empty:

                selected = (
                    selected_rows
                    .iloc[0]
                )

                st.markdown(
                    "#### ❓ Question"
                )

                st.info(
                    selected_question
                )

                inspect_cols = st.columns(4)

                with inspect_cols[0]:

                    value = selected.get(
                        "groundedness_score"
                    )

                    st.metric(
                        "Groundedness",
                        (
                            f"{float(value):.3f}"
                            if pd.notna(value)
                            else "N/A"
                        ),
                    )

                with inspect_cols[1]:

                    value = selected.get(
                        "context_relevance_score"
                    )

                    st.metric(
                        "Context Relevance",
                        (
                            f"{float(value):.3f}"
                            if pd.notna(value)
                            else "N/A"
                        ),
                    )

                with inspect_cols[2]:

                    citation_value = selected.get(
                        "citation_correct"
                    )

                    citation_text = str(
                        citation_value
                    )

                    st.metric(
                        "Citation Correct",
                        citation_text,
                    )

                with inspect_cols[3]:

                    total_value = selected.get(
                        "total_ms"
                    )

                    st.metric(
                        "Total Latency",
                        format_duration_ms(
                            total_value
                        ),
                    )

                st.markdown(
                    "#### 📝 Generated Answer"
                )

                answer_text = selected.get(
                    "answer",
                    "No answer recorded.",
                )

                st.write(
                    answer_text
                )

                st.markdown(
                    "#### 📚 Expected Sources"
                )

                st.write(
                    selected.get(
                        "expected_sources",
                        "N/A",
                    )
                )

                st.markdown(
                    "#### 🔗 Matched Sources"
                )

                st.write(
                    selected.get(
                        "matched_sources",
                        "N/A",
                    )
                )

                if "suspicious_patterns" in selected.index:

                    suspicious_patterns = selected.get(
                        "suspicious_patterns"
                    )

                    if (
                        pd.notna(
                            suspicious_patterns
                        )
                        and str(
                            suspicious_patterns
                        ).strip()
                    ):

                        st.warning(
                            "Suspicious patterns detected: "
                            f"{suspicious_patterns}"
                        )

    st.divider()

    # ========================================================
    # METHODOLOGY
    # ========================================================

    with st.expander(
        "ℹ️ Answer Quality Evaluation Methodology"
    ):

        st.markdown(
            """
            ### Evaluation Pipeline

            Each evaluation question is processed through the
            same production RAG pipeline:

            **Question → Retrieval → Context Building → "
            "Grounded Prompt → Ollama Cloud → Answer Evaluation**

            The evaluation records:

            - Expected source retrieval
            - Context relevance
            - Evidence-support / groundedness
            - Citation correctness
            - Abstention behavior
            - Suspicious answer patterns
            - Answer length
            - Retrieval latency
            - Generation latency
            - End-to-end latency

            ### Important Interpretation

            The current groundedness score is an automated
            lexical/evidence-support proxy. It is useful for
            comparing system behavior and identifying questions
            that require investigation, but it is not a definitive
            factuality or hallucination detector.

            The current evaluation dataset contains 30 questions:

            - 10 easy
            - 10 medium
            - 10 difficult

            The evaluation results are stored in:

            `docs/answer_evaluation_results.csv`

            Running the dashboard does not regenerate these answers.
            """
        )


# ============================================================
# NORMAL LLM VS RAG
# ============================================================

NORMAL_VS_RAG_RESULTS_PATH = (
    PROJECT_ROOT
    / "docs"
    / "normal_vs_rag_results.csv"
)


def load_normal_vs_rag_results():
    """
    Load the completed Normal LLM vs RAG experiment.

    This function only reads the CSV file.
    It does NOT execute Ollama or rerun the experiment.
    """

    if not NORMAL_VS_RAG_RESULTS_PATH.exists():
        return pd.DataFrame()

    try:

        dataframe = pd.read_csv(
            NORMAL_VS_RAG_RESULTS_PATH
        )

    except Exception:

        return pd.DataFrame()

    return dataframe


def normal_vs_rag_numeric(
    dataframe,
    column,
):
    """
    Safely convert a Normal-vs-RAG column to numeric.
    """

    if column not in dataframe.columns:

        return pd.Series(
            dtype="float64"
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def normal_vs_rag_mean(
    dataframe,
    column,
    default=0.0,
):
    """
    Safely calculate an experiment metric mean.
    """

    values = (
        normal_vs_rag_numeric(
            dataframe,
            column,
        )
        .dropna()
    )

    if values.empty:
        return default

    return float(
        values.mean()
    )


def normal_vs_rag_percentage(
    dataframe,
    column,
):
    """
    Calculate percentage of truthy values.
    """

    if column not in dataframe.columns:
        return None

    values = dataframe[column]

    if values.empty:
        return None

    normalized = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )

    truthy = normalized.isin(
        [
            "true",
            "1",
            "yes",
        ]
    )

    return float(
        truthy.mean()
    )


def render_normal_vs_rag_dashboard():
    """
    Render the completed Normal LLM vs RAG comparison.

    The dashboard reads:
        docs/normal_vs_rag_results.csv

    No new LLM calls are performed.
    """

    st.markdown(
        "## 🆚 Normal LLM vs RAG Comparison"
    )

    st.write(
        "Compare a general LLM response with DocMind's "
        "knowledge-grounded RAG response using the same "
        "30-question evaluation dataset."
    )

    st.caption(
        "This dashboard reads the completed "
        "`docs/normal_vs_rag_results.csv` experiment. "
        "Opening this dashboard does not trigger new Ollama "
        "Cloud generations."
    )

    dataframe = load_normal_vs_rag_results()

    if dataframe.empty:

        st.warning(
            "Normal LLM vs RAG experiment results were not found."
        )

        st.info(
            "Run the experiment first with:\n\n"
            "`python -m app.evaluation.normal_vs_rag`"
        )

        return

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.markdown(
        "### 📋 Experiment Overview"
    )

    total_questions = len(
        dataframe
    )

    difficulty_count = (
        dataframe["difficulty"].nunique()
        if "difficulty" in dataframe.columns
        else 0
    )

    category_count = (
        dataframe["category"].nunique()
        if "category" in dataframe.columns
        else 0
    )

    normal_errors = 0

    if "normal_error" in dataframe.columns:

        normal_errors = (
            dataframe[
                "normal_error"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

    rag_errors = 0

    if "rag_error" in dataframe.columns:

        rag_errors = (
            dataframe[
                "rag_error"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

    successful_comparisons = (
        total_questions
        - max(
            normal_errors,
            rag_errors,
        )
    )

    overview_cols = st.columns(5)

    with overview_cols[0]:

        st.metric(
            "Questions",
            total_questions,
        )

    with overview_cols[1]:

        st.metric(
            "Difficulty Levels",
            difficulty_count,
        )

    with overview_cols[2]:

        st.metric(
            "Categories",
            category_count,
        )

    with overview_cols[3]:

        st.metric(
            "Successful Comparisons",
            successful_comparisons,
        )

    with overview_cols[4]:

        st.metric(
            "RAG Sources",
            (
                f"{normal_vs_rag_percentage(dataframe, 'rag_citation_available'):.1%}"
                if normal_vs_rag_percentage(
                    dataframe,
                    "rag_citation_available",
                ) is not None
                else "N/A"
            ),
        )

    st.divider()

    # ========================================================
    # OVERALL COMPARISON METRICS
    # ========================================================

    st.markdown(
        "### 🎯 Overall Comparison"
    )

    normal_generation = normal_vs_rag_mean(
        dataframe,
        "normal_generation_ms",
    )

    rag_retrieval = normal_vs_rag_mean(
        dataframe,
        "rag_retrieval_ms",
    )

    rag_context = normal_vs_rag_mean(
        dataframe,
        "rag_context_ms",
    )

    rag_generation = normal_vs_rag_mean(
        dataframe,
        "rag_generation_ms",
    )

    rag_total = normal_vs_rag_mean(
        dataframe,
        "rag_total_ms",
    )

    normal_evidence = normal_vs_rag_mean(
        dataframe,
        "normal_evidence_support_proxy",
    )

    rag_groundedness = normal_vs_rag_mean(
        dataframe,
        "rag_groundedness_proxy",
    )

    rag_context_relevance = normal_vs_rag_mean(
        dataframe,
        "rag_context_relevance",
    )

    normal_citation = normal_vs_rag_percentage(
        dataframe,
        "normal_citation_available",
    )

    rag_citation = normal_vs_rag_percentage(
        dataframe,
        "rag_citation_available",
    )

    expected_source = normal_vs_rag_percentage(
        dataframe,
        "expected_source_found",
    )

    normal_abstention = normal_vs_rag_percentage(
        dataframe,
        "normal_abstained",
    )

    rag_abstention = normal_vs_rag_percentage(
        dataframe,
        "rag_abstained",
    )

    metric_cols = st.columns(4)

    with metric_cols[0]:

        st.metric(
            "Normal LLM Generation",
            format_duration_ms(
                normal_generation
            ),
        )

    with metric_cols[1]:

        st.metric(
            "RAG Retrieval",
            format_duration_ms(
                rag_retrieval
            ),
        )

    with metric_cols[2]:

        st.metric(
            "RAG Generation",
            format_duration_ms(
                rag_generation
            ),
        )

    with metric_cols[3]:

        st.metric(
            "RAG End-to-End",
            format_duration_ms(
                rag_total
            ),
        )

    metric_cols2 = st.columns(4)

    with metric_cols2[0]:

        st.metric(
            "Normal Evidence Proxy",
            f"{normal_evidence:.3f}",
        )

    with metric_cols2[1]:

        st.metric(
            "RAG Groundedness Proxy",
            f"{rag_groundedness:.3f}",
        )

    with metric_cols2[2]:

        st.metric(
            "RAG Context Relevance",
            f"{rag_context_relevance:.3f}",
        )
    with metric_cols2[3]:

        st.metric(
            "Expected Source Found",
            (
                f"{expected_source:.1%}"
                if expected_source is not None
                else "N/A"
            ),
        )

    # ========================================================
    # EXECUTIVE ANALYTICS SUMMARY
    # ========================================================

    st.markdown(
        "### 📊 Executive Analytics Summary"
    )

    # --------------------------------------------------------
    # Additional experiment metrics
    # --------------------------------------------------------

    rag_source_count = normal_vs_rag_mean(
        dataframe,
        "rag_source_count",
    )

    normal_answer_length = None

    if "normal_answer" in dataframe.columns:

        normal_lengths = (
            dataframe["normal_answer"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        if not normal_lengths.empty:
            normal_answer_length = (
                normal_lengths.mean()
            )

    rag_answer_length = None

    if "rag_answer" in dataframe.columns:

        rag_lengths = (
            dataframe["rag_answer"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        if not rag_lengths.empty:
            rag_answer_length = (
                rag_lengths.mean()
            )

    latency_difference = None
    latency_change = None

    if (
        normal_generation is not None
        and rag_total is not None
        and normal_generation > 0
    ):

        latency_difference = (
            normal_generation
            - rag_total
        )

        latency_change = (
            latency_difference
            / normal_generation
        )

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    summary_cols = st.columns(3)

    with summary_cols[0]:

        st.metric(
            "Average RAG Sources",
            (
                f"{rag_source_count:.2f}"
                if rag_source_count is not None
                else "N/A"
            ),
        )

    with summary_cols[1]:

        st.metric(
            "Normal Answer Length",
            (
                f"{normal_answer_length:.0f} chars"
                if normal_answer_length is not None
                else "N/A"
            ),
        )

    with summary_cols[2]:

        st.metric(
            "RAG Answer Length",
            (
                f"{rag_answer_length:.0f} chars"
                if rag_answer_length is not None
                else "N/A"
            ),
        )

    summary_cols2 = st.columns(3)

    with summary_cols2[0]:

        st.metric(
            "Normal Abstention",
            (
                f"{normal_abstention:.1%}"
                if normal_abstention is not None
                else "N/A"
            ),
        )

    with summary_cols2[1]:

        st.metric(
            "RAG Abstention",
            (
                f"{rag_abstention:.1%}"
                if rag_abstention is not None
                else "N/A"
            ),
        )

    with summary_cols2[2]:

        if latency_difference is not None:

            st.metric(
                "Generation → RAG E2E Difference",
                format_duration_ms(
                    latency_difference
                ),
                delta=(
                    f"{latency_change:+.1%}"
                    if latency_change is not None
                    else None
                ),
            )

        else:

            st.metric(
                "Generation → RAG E2E Difference",
                "N/A",
            )

    st.caption(
        "These metrics summarize the completed 30-question "
        "experiment. They are calculated from the stored CSV "
        "results and do not trigger new model generations."
    )

    st.divider()
    # ========================================================
    # EVIDENCE SUPPORT COMPARISON
    # ========================================================

    st.markdown(
        "### 🔎 Evidence-Support Comparison"
    )

    evidence_df = pd.DataFrame(
        {
            "Evidence Support": [
                normal_evidence,
                rag_groundedness,
            ]
        },
        index=[
            "Normal LLM",
            "RAG",
        ],
    )

    st.bar_chart(
        evidence_df
    )

    evidence_difference = (
        rag_groundedness
        - normal_evidence
    )

    st.info(
        f"Recorded evidence-support proxy difference: "
        f"{evidence_difference:+.3f} "
        f"({evidence_difference * 100:+.2f} percentage points)."
    )

    st.caption(
        "These values are automated lexical/evidence-support "
        "proxies. They should not be interpreted as definitive "
        "factual correctness scores."
    )

    st.divider()

    # ========================================================
    # LATENCY COMPARISON
    # ========================================================

    st.markdown(
        "### ⚡ Latency Comparison"
    )

    latency_comparison = pd.DataFrame(
        {
            "Average Time (ms)": [
                normal_generation,
                rag_retrieval,
                rag_context,
                rag_generation,
                rag_total,
            ]
        },
        index=[
            "Normal LLM Generation",
            "RAG Retrieval",
            "RAG Context Build",
            "RAG Generation",
            "RAG End-to-End",
        ],
    )

    st.bar_chart(
        latency_comparison
    )

    st.caption(
        "The RAG retrieval measurement includes the retrieval "
        "pipeline and cross-encoder reranking when enabled. "
        "The first retrieval can be substantially slower because "
        "the reranker model may need to load into memory."
    )

    latency_table = pd.DataFrame(
        {
            "Stage": [
                "Normal LLM Generation",
                "RAG Retrieval",
                "RAG Context Build",
                "RAG Generation",
                "RAG End-to-End",
            ],
            "Average ms": [
                normal_generation,
                rag_retrieval,
                rag_context,
                rag_generation,
                rag_total,
            ],
        }
    )
    st.dataframe(
        latency_table.style.format(
            {
                "Average ms": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    # ========================================================
    # ADVANCED LATENCY ANALYTICS
    # ========================================================

    st.markdown(
        "#### 📊 Advanced Latency Analytics"
    )

    latency_columns = [
        "question",
        "difficulty",
        "category",
        "normal_generation_ms",
        "rag_retrieval_ms",
        "rag_context_ms",
        "rag_generation_ms",
        "rag_total_ms",
    ]

    if not all(
        column in dataframe.columns
        for column in latency_columns
    ):

        st.info(
            "Detailed latency fields are not available "
            "in the recorded Normal LLM vs RAG experiment."
        )

    else:

        latency_df = dataframe[
            latency_columns
        ].copy()

        numeric_latency_columns = [
            "normal_generation_ms",
            "rag_retrieval_ms",
            "rag_context_ms",
            "rag_generation_ms",
            "rag_total_ms",
        ]

        for column in numeric_latency_columns:

            latency_df[column] = pd.to_numeric(
                latency_df[column],
                errors="coerce",
            )

        # ----------------------------------------------------
        # Percentile summary
        # ----------------------------------------------------

        st.markdown(
            "##### ⏱️ Latency Percentiles"
        )

        percentile_rows = []

        latency_stage_names = {
            "normal_generation_ms": (
                "Normal LLM Generation"
            ),
            "rag_retrieval_ms": (
                "RAG Retrieval"
            ),
            "rag_context_ms": (
                "RAG Context Build"
            ),
            "rag_generation_ms": (
                "RAG Generation"
            ),
            "rag_total_ms": (
                "RAG End-to-End"
            ),
        }

        for column, stage_name in (
            latency_stage_names.items()
        ):

            values = (
                latency_df[column]
                .dropna()
            )

            if values.empty:
                continue

            percentile_rows.append(
                {
                    "Stage": stage_name,
                    "P50 ms": values.quantile(
                        0.50
                    ),
                    "P90 ms": values.quantile(
                        0.90
                    ),
                    "P95 ms": values.quantile(
                        0.95
                    ),
                    "Maximum ms": values.max(),
                }
            )

        percentile_df = pd.DataFrame(
            percentile_rows
        )

        if not percentile_df.empty:

            st.dataframe(
                percentile_df.style.format(
                    {
                        "P50 ms": "{:.2f}",
                        "P90 ms": "{:.2f}",
                        "P95 ms": "{:.2f}",
                        "Maximum ms": "{:.2f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # Per-question latency distribution
        # ----------------------------------------------------

        st.markdown(
            "##### 📈 Per-Question Latency Distribution"
        )

        latency_metric = st.selectbox(
            "Latency distribution metric",
            [
                "normal_generation_ms",
                "rag_retrieval_ms",
                "rag_context_ms",
                "rag_generation_ms",
                "rag_total_ms",
            ],
            format_func=(
                lambda value:
                value.replace(
                    "_ms",
                    "",
                )
                .replace(
                    "_",
                    " ",
                )
                .title()
            ),
            key="advanced_latency_metric",
        )

        distribution_values = (
            latency_df[
                latency_metric
            ]
            .dropna()
            .reset_index(
                drop=True
            )
        )

        if not distribution_values.empty:

            distribution_df = pd.DataFrame(
                {
                    "Question": (
                        range(
                            1,
                            len(
                                distribution_values
                            ) + 1,
                        )
                    ),
                    "Latency ms": (
                        distribution_values
                    ),
                }
            )

            st.line_chart(
                distribution_df,
                x="Question",
                y="Latency ms",
            )

        # ----------------------------------------------------
        # Slowest questions
        # ----------------------------------------------------

        st.markdown(
            "##### 🐌 Slowest Recorded Questions"
        )

        slowest_df = (
            latency_df[
                [
                    "question",
                    "difficulty",
                    "category",
                    "normal_generation_ms",
                    "rag_retrieval_ms",
                    "rag_context_ms",
                    "rag_generation_ms",
                    "rag_total_ms",
                ]
            ]
            .dropna(
                subset=[
                    "rag_total_ms"
                ]
            )
            .sort_values(
                "rag_total_ms",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        slowest_display = (
            slowest_df.rename(
                columns={
                    "question": "Question",
                    "difficulty": "Difficulty",
                    "category": "Category",
                    "normal_generation_ms": (
                        "Normal Generation ms"
                    ),
                    "rag_retrieval_ms": (
                        "RAG Retrieval ms"
                    ),
                    "rag_context_ms": (
                        "RAG Context ms"
                    ),
                    "rag_generation_ms": (
                        "RAG Generation ms"
                    ),
                    "rag_total_ms": (
                        "RAG Total ms"
                    ),
                }
            )
        )

        st.dataframe(
            slowest_display.style.format(
                {
                    "Normal Generation ms": "{:.0f}",
                    "RAG Retrieval ms": "{:.0f}",
                    "RAG Context ms": "{:.0f}",
                    "RAG Generation ms": "{:.0f}",
                    "RAG Total ms": "{:.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # Latency by difficulty
        # ----------------------------------------------------

        if "difficulty" in latency_df.columns:

            st.markdown(
                "##### 🎚️ Latency by Question Difficulty"
            )

            latency_difficulty_df = (
                latency_df
                .groupby("difficulty")
                .agg(
                    Questions=(
                        "question",
                        "count",
                    ),
                    Normal_Generation_ms=(
                        "normal_generation_ms",
                        "mean",
                    ),
                    RAG_Retrieval_ms=(
                        "rag_retrieval_ms",
                        "mean",
                    ),
                    RAG_Generation_ms=(
                        "rag_generation_ms",
                        "mean",
                    ),
                    RAG_Total_ms=(
                        "rag_total_ms",
                        "mean",
                    ),
                )
                .reset_index()
            )

            st.dataframe(
                latency_difficulty_df.style.format(
                    {
                        "Normal_Generation_ms": "{:.0f}",
                        "RAG_Retrieval_ms": "{:.0f}",
                        "RAG_Generation_ms": "{:.0f}",
                        "RAG_Total_ms": "{:.0f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            latency_difficulty_metric = (
                st.selectbox(
                    "Difficulty latency metric",
                    [
                        "Normal_Generation_ms",
                        "RAG_Retrieval_ms",
                        "RAG_Generation_ms",
                        "RAG_Total_ms",
                    ],
                    format_func=(
                        lambda value:
                        value.replace(
                            "_ms",
                            "",
                        )
                        .replace(
                            "_",
                            " ",
                        )
                        .title()
                    ),
                    key="latency_difficulty_metric",
                )
            )

            difficulty_latency_chart = (
                latency_difficulty_df[
                    [
                        "difficulty",
                        latency_difficulty_metric,
                    ]
                ]
                .copy()
            )

            difficulty_latency_chart.columns = [
                "Difficulty",
                "Latency ms",
            ]

            st.bar_chart(
                difficulty_latency_chart,
                x="Difficulty",
                y="Latency ms",
            )

        st.caption(
            "P50, P90, and P95 describe the recorded latency "
            "distribution rather than only the average. The "
            "slowest-question table helps identify individual "
            "queries that contributed disproportionately to "
            "end-to-end RAG latency."
        )

    st.divider()

    # ========================================================
    # CITATION COMPARISON
    # ========================================================
    st.markdown(
        "### 📚 Source Citation Availability"
    )

    citation_df = pd.DataFrame(
        {
            "Citation Availability": [
                (
                    normal_citation
                    if normal_citation is not None
                    else 0
                ),
                (
                    rag_citation
                    if rag_citation is not None
                    else 0
                ),
            ]
        },
        index=[
            "Normal LLM",
            "RAG",
        ],
    )

    st.bar_chart(
        citation_df
    )

    citation_cols = st.columns(3)

    with citation_cols[0]:

        st.metric(
            "Normal LLM Citations",
            (
                f"{normal_citation:.1%}"
                if normal_citation is not None
                else "N/A"
            ),
        )

    with citation_cols[1]:

        st.metric(
            "RAG Citations",
            (
                f"{rag_citation:.1%}"
                if rag_citation is not None
                else "N/A"
            ),
        )

    with citation_cols[2]:

        st.metric(
            "RAG Expected Source Hit",
            (
                f"{expected_source:.1%}"
                if expected_source is not None
                else "N/A"
            ),
        )

    st.caption(
        "Citation availability indicates whether source "
        "information was available in the experiment output. "
        "It does not independently establish factual correctness."
    )

    st.divider()

    # ========================================================
    # DIFFICULTY ANALYSIS
    # ========================================================

    st.markdown(
        "### 🧠 Comparison by Question Difficulty"
    )

    if "difficulty" not in dataframe.columns:

        st.info(
            "Difficulty information is not available."
        )

    else:

        difficulty_comparison = (
            dataframe
            .groupby("difficulty")
            .agg(
                Questions=(
                    "question",
                    "count",
                ),
                Normal_Evidence=(
                    "normal_evidence_support_proxy",
                    "mean",
                ),
                RAG_Groundedness=(
                    "rag_groundedness_proxy",
                    "mean",
                ),
                RAG_Context_Relevance=(
                    "rag_context_relevance",
                    "mean",
                ),
                Normal_Generation_ms=(
                    "normal_generation_ms",
                    "mean",
                ),
                RAG_Total_ms=(
                    "rag_total_ms",
                    "mean",
                ),
            )
        )

        st.dataframe(
            difficulty_comparison.style.format(
                {
                    "Normal_Evidence": "{:.3f}",
                    "RAG_Groundedness": "{:.3f}",
                    "RAG_Context_Relevance": "{:.3f}",
                    "Normal_Generation_ms": "{:.0f}",
                    "RAG_Total_ms": "{:.0f}",
                }
            ),
            width="stretch",
        )

        difficulty_metric = st.selectbox(
            "Difficulty comparison metric",
            [
                "Normal_Evidence",
                "RAG_Groundedness",
                "RAG_Context_Relevance",
                "Normal_Generation_ms",
                "RAG_Total_ms",
            ],
            format_func=(
                lambda value:
                value.replace(
                    "_",
                    " ",
                )
            ),
            key="normal_rag_difficulty_metric",
        )

        st.bar_chart(
            difficulty_comparison[
                [difficulty_metric]
            ]
        )

    st.divider()

    # ========================================================
    # CATEGORY ANALYSIS
    # ========================================================

    st.markdown(
        "### 📚 Comparison by Knowledge Category"
    )

    if "category" not in dataframe.columns:

        st.info(
            "Category information is not available."
        )

    else:

        category_comparison = (
            dataframe
            .groupby("category")
            .agg(
                Questions=(
                    "question",
                    "count",
                ),
                Normal_Evidence=(
                    "normal_evidence_support_proxy",
                    "mean",
                ),
                RAG_Groundedness=(
                    "rag_groundedness_proxy",
                    "mean",
                ),
                RAG_Context_Relevance=(
                    "rag_context_relevance",
                    "mean",
                ),
                Normal_Generation_ms=(
                    "normal_generation_ms",
                    "mean",
                ),
                RAG_Generation_ms=(
                    "rag_generation_ms",
                    "mean",
                ),
                RAG_Total_ms=(
                    "rag_total_ms",
                    "mean",
                ),
            )
        )

        st.dataframe(
            category_comparison.style.format(
                {
                    "Normal_Evidence": "{:.3f}",
                    "RAG_Groundedness": "{:.3f}",
                    "RAG_Context_Relevance": "{:.3f}",
                    "Normal_Generation_ms": "{:.0f}",
                    "RAG_Generation_ms": "{:.0f}",
                    "RAG_Total_ms": "{:.0f}",
                }
            ),
            width="stretch",
        )

        category_metric = st.selectbox(
            "Category comparison metric",
            [
                "Normal_Evidence",
                "RAG_Groundedness",
                "RAG_Context_Relevance",
                "Normal_Generation_ms",
                "RAG_Generation_ms",
                "RAG_Total_ms",
            ],
            format_func=(
                lambda value:
                value.replace(
                    "_",
                    " ",
                )
            ),
            key="normal_rag_category_metric",
        )

        st.bar_chart(
            category_comparison[
                [category_metric]
            ]
        )

    st.divider()

    # ========================================================
    # QUESTION-LEVEL EVIDENCE DIFFERENCE
    # ========================================================

    st.markdown(
        "### 📈 Question-Level Evidence Comparison"
    )

    question_comparison = dataframe.copy()

    if (
        "normal_evidence_support_proxy"
        in question_comparison.columns
        and
        "rag_groundedness_proxy"
        in question_comparison.columns
    ):

        question_comparison[
            "Evidence Difference"
        ] = (
            pd.to_numeric(
                question_comparison[
                    "rag_groundedness_proxy"
                ],
                errors="coerce",
            )
            -
            pd.to_numeric(
                question_comparison[
                    "normal_evidence_support_proxy"
                ],
                errors="coerce",
            )
        )

        question_comparison[
            "Question"
        ] = (
            question_comparison[
                "question"
            ]
            .astype(str)
            .str.slice(
                0,
                80,
            )
        )

        evidence_question_chart = (
            question_comparison[
                [
                    "Question",
                    "Evidence Difference",
                ]
            ]
            .set_index(
                "Question"
            )
            .sort_values(
                "Evidence Difference"
            )
        )

        st.bar_chart(
            evidence_question_chart
        )

        st.caption(
            "Positive values indicate a higher recorded "
            "RAG groundedness proxy than the normal LLM "
            "evidence-support proxy for that question."
        )

    else:

        st.info(
            "Question-level evidence columns are not available."
        )

    st.divider()

    # ========================================================
    # DETAILED RESULTS
    # ========================================================

    st.markdown(
        "### 🔬 Detailed Normal LLM vs RAG Results"
    )

    detail_columns = [
        "question",
        "difficulty",
        "category",
        "normal_generation_ms",
        "rag_retrieval_ms",
        "rag_context_ms",
        "rag_generation_ms",
        "rag_total_ms",
        "normal_evidence_support_proxy",
        "rag_groundedness_proxy",
        "rag_context_relevance",
        "normal_citation_available",
        "rag_citation_available",
        "expected_source_found",
        "normal_abstained",
        "rag_abstained",
        "rag_source_count",
        "normal_error",
        "rag_error",
    ]

    available_detail = available_columns(
        dataframe,
        detail_columns,
    )

    if available_detail:

        detailed_df = (
            dataframe[
                available_detail
            ]
            .copy()
        )

        st.dataframe(
            detailed_df,
            width="stretch",
            hide_index=True,
        )

    else:

        st.dataframe(
            dataframe,
            width="stretch",
            hide_index=True,
        )

    st.divider()

    # ========================================================
    # INSPECT INDIVIDUAL QUESTION
    # ========================================================

    st.markdown(
        "### 🔍 Inspect an Individual Comparison"
    )

    if "question" in dataframe.columns:

        question_options = (
            dataframe[
                "question"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        if question_options:

            selected_question = st.selectbox(
                "Select a comparison question",
                question_options,
                key="normal_rag_question_selector",
            )

            selected_rows = dataframe[
                dataframe[
                    "question"
                ].astype(str)
                == selected_question
            ]

            if not selected_rows.empty:

                selected = (
                    selected_rows
                    .iloc[0]
                )

                st.info(
                    selected_question
                )

                comparison_cols = st.columns(4)

                with comparison_cols[0]:

                    value = selected.get(
                        "normal_evidence_support_proxy"
                    )

                    st.metric(
                        "Normal Evidence Proxy",
                        (
                            f"{float(value):.3f}"
                            if pd.notna(value)
                            else "N/A"
                        ),
                    )

                with comparison_cols[1]:

                    value = selected.get(
                        "rag_groundedness_proxy"
                    )

                    st.metric(
                        "RAG Groundedness Proxy",
                        (
                            f"{float(value):.3f}"
                            if pd.notna(value)
                            else "N/A"
                        ),
                    )

                with comparison_cols[2]:

                    value = selected.get(
                        "rag_context_relevance"
                    )

                    st.metric(
                        "RAG Context Relevance",
                        (
                            f"{float(value):.3f}"
                            if pd.notna(value)
                            else "N/A"
                        ),
                    )

                with comparison_cols[3]:

                    value = selected.get(
                        "rag_total_ms"
                    )

                    st.metric(
                        "RAG Total",
                        format_duration_ms(
                            value
                        ),
                    )

                st.markdown(
                    "#### 🤖 Normal LLM Answer"
                )

                st.write(
                    selected.get(
                        "normal_answer",
                        "No normal LLM answer recorded.",
                    )
                )

                st.markdown(
                    "#### 🧠 RAG Answer"
                )

                st.write(
                    selected.get(
                        "rag_answer",
                        "No RAG answer recorded.",
                    )
                )

                answer_detail_cols = st.columns(4)

                with answer_detail_cols[0]:

                    st.metric(
                        "Normal Generation",
                        format_duration_ms(
                            selected.get(
                                "normal_generation_ms"
                            )
                        ),
                    )

                with answer_detail_cols[1]:

                    st.metric(
                        "RAG Retrieval",
                        format_duration_ms(
                            selected.get(
                                "rag_retrieval_ms"
                            )
                        ),
                    )

                with answer_detail_cols[2]:

                    st.metric(
                        "RAG Generation",
                        format_duration_ms(
                            selected.get(
                                "rag_generation_ms"
                            )
                        ),
                    )

                with answer_detail_cols[3]:

                    st.metric(
                        "Expected Source",
                        (
                            "Yes"
                            if str(
                                selected.get(
                                    "expected_source_found",
                                    ""
                                )
                            ).strip().lower()
                            in [
                                "true",
                                "1",
                                "yes",
                            ]
                            else "No"
                        ),
                    )

                st.markdown(
                    "#### 📚 Source Information"
                )

                st.write(
                    f"RAG source count: "
                    f"{selected.get('rag_source_count', 'N/A')}"
                )

                st.write(
                    f"RAG citation available: "
                    f"{selected.get('rag_citation_available', 'N/A')}"
                )
    st.divider()

    # ========================================================
    # FAILURE ANALYSIS
    # ========================================================

    st.markdown(
        "### 🚨 RAG Failure Analysis"
    )

    failure_required = [
        "question",
        "difficulty",
        "category",
        "rag_groundedness_proxy",
        "rag_context_relevance",
        "expected_source_found",
        "rag_abstained",
        "rag_source_count",
    ]

    if not all(
        column in dataframe.columns
        for column in failure_required
    ):

        st.info(
            "The stored Normal LLM vs RAG dataset does not "
            "contain all fields required for failure analysis."
        )

    else:

        failure_df = dataframe.copy()

        # ----------------------------------------------------
        # Normalize numeric metrics
        # ----------------------------------------------------

        numeric_failure_columns = [
            "rag_groundedness_proxy",
            "rag_context_relevance",
            "rag_source_count",
        ]

        for column in numeric_failure_columns:

            failure_df[column] = pd.to_numeric(
                failure_df[column],
                errors="coerce",
            )

        # ----------------------------------------------------
        # Normalize boolean fields
        # ----------------------------------------------------

        def normalize_bool(value):

            if pd.isna(value):
                return False

            return str(value).strip().lower() in [
                "true",
                "1",
                "yes",
                "y",
            ]

        failure_df[
            "Expected_Source_Found"
        ] = (
            failure_df[
                "expected_source_found"
            ]
            .apply(normalize_bool)
        )

        failure_df[
            "RAG_Abstained"
        ] = (
            failure_df[
                "rag_abstained"
            ]
            .apply(normalize_bool)
        )

        # ----------------------------------------------------
        # Define recorded failure signals
        # ----------------------------------------------------

        failure_df[
            "Low_Groundedness"
        ] = (
            failure_df[
                "rag_groundedness_proxy"
            ]
            < 0.50
        )

        failure_df[
            "Low_Context_Relevance"
        ] = (
            failure_df[
                "rag_context_relevance"
            ]
            < 0.50
        )

        failure_df[
            "Retrieval_Miss"
        ] = (
            ~failure_df[
                "Expected_Source_Found"
            ]
        )

        failure_df[
            "No_Retrieved_Sources"
        ] = (
            failure_df[
                "rag_source_count"
            ].fillna(0)
            <= 0
        )

        # ----------------------------------------------------
        # Failure counts
        # ----------------------------------------------------

        low_groundedness_count = int(
            failure_df[
                "Low_Groundedness"
            ].sum()
        )

        low_context_count = int(
            failure_df[
                "Low_Context_Relevance"
            ].sum()
        )

        retrieval_miss_count = int(
            failure_df[
                "Retrieval_Miss"
            ].sum()
        )

        abstention_count = int(
            failure_df[
                "RAG_Abstained"
            ].sum()
        )

        no_source_count = int(
            failure_df[
                "No_Retrieved_Sources"
            ].sum()
        )

        total_failure_questions = len(
            failure_df
        )

        # ----------------------------------------------------
        # Failure summary metrics
        # ----------------------------------------------------

        failure_cols = st.columns(5)

        with failure_cols[0]:

            st.metric(
                "Low Groundedness",
                low_groundedness_count,
                delta=(
                    f"{low_groundedness_count / total_failure_questions:.1%}"
                    if total_failure_questions
                    else None
                ),
            )

        with failure_cols[1]:

            st.metric(
                "Low Context Relevance",
                low_context_count,
                delta=(
                    f"{low_context_count / total_failure_questions:.1%}"
                    if total_failure_questions
                    else None
                ),
            )

        with failure_cols[2]:

            st.metric(
                "Retrieval Misses",
                retrieval_miss_count,
                delta=(
                    f"{retrieval_miss_count / total_failure_questions:.1%}"
                    if total_failure_questions
                    else None
                ),
            )

        with failure_cols[3]:

            st.metric(
                "RAG Abstentions",
                abstention_count,
                delta=(
                    f"{abstention_count / total_failure_questions:.1%}"
                    if total_failure_questions
                    else None
                ),
            )

        with failure_cols[4]:

            st.metric(
                "No Sources Retrieved",
                no_source_count,
                delta=(
                    f"{no_source_count / total_failure_questions:.1%}"
                    if total_failure_questions
                    else None
                ),
            )

        # ----------------------------------------------------
        # Failure-type overview
        # ----------------------------------------------------

        st.markdown(
            "#### 📊 Recorded Failure Signals"
        )

        failure_summary_df = pd.DataFrame(
            {
                "Failure Type": [
                    "Low Groundedness",
                    "Low Context Relevance",
                    "Retrieval Miss",
                    "RAG Abstention",
                    "No Sources Retrieved",
                ],
                "Questions": [
                    low_groundedness_count,
                    low_context_count,
                    retrieval_miss_count,
                    abstention_count,
                    no_source_count,
                ],
            }
        )

        st.bar_chart(
            failure_summary_df,
            x="Failure Type",
            y="Questions",
        )

        # ----------------------------------------------------
        # Weakest RAG responses
        # ----------------------------------------------------

        st.markdown(
            "#### 🔎 Weakest Recorded RAG Responses"
        )

        weakness_columns = [
            "question",
            "difficulty",
            "category",
            "rag_groundedness_proxy",
            "rag_context_relevance",
            "expected_source_found",
            "rag_source_count",
            "rag_abstained",
        ]

        available_weakness_columns = [
            column
            for column in weakness_columns
            if column in failure_df.columns
        ]

        weakest_df = (
            failure_df[
                available_weakness_columns
            ]
            .copy()
            .sort_values(
                [
                    "rag_groundedness_proxy",
                    "rag_context_relevance",
                ],
                ascending=[
                    True,
                    True,
                ],
                na_position="last",
            )
            .head(10)
        )

        weakest_display = (
            weakest_df.rename(
                columns={
                    "question": "Question",
                    "difficulty": "Difficulty",
                    "category": "Category",
                    "rag_groundedness_proxy": (
                        "Groundedness"
                    ),
                    "rag_context_relevance": (
                        "Context Relevance"
                    ),
                    "expected_source_found": (
                        "Expected Source Found"
                    ),
                    "rag_source_count": (
                        "RAG Sources"
                    ),
                    "rag_abstained": (
                        "RAG Abstained"
                    ),
                }
            )
        )

        st.dataframe(
            weakest_display.style.format(
                {
                    "Groundedness": "{:.3f}",
                    "Context Relevance": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # Failure analysis by difficulty
        # ----------------------------------------------------

        st.markdown(
            "#### 🎚️ Failure Signals by Difficulty"
        )

        difficulty_failure_df = (
            failure_df
            .groupby("difficulty")
            .agg(
                Questions=("question", "count"),
                Low_Groundedness=(
                    "Low_Groundedness",
                    "mean",
                ),
                Low_Context_Relevance=(
                    "Low_Context_Relevance",
                    "mean",
                ),
                Retrieval_Miss=(
                    "Retrieval_Miss",
                    "mean",
                ),
            )
            .reset_index()
        )

        st.dataframe(
            difficulty_failure_df.style.format(
                {
                    "Low_Groundedness": "{:.1%}",
                    "Low_Context_Relevance": "{:.1%}",
                    "Retrieval_Miss": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        difficulty_failure_metric = st.selectbox(
            "Failure metric by difficulty",
            [
                "Low_Groundedness",
                "Low_Context_Relevance",
                "Retrieval_Miss",
            ],
            format_func=(
                lambda value:
                value.replace(
                    "_",
                    " ",
                )
            ),
            key="failure_difficulty_metric",
        )

        difficulty_failure_chart = (
            difficulty_failure_df[
                [
                    "difficulty",
                    difficulty_failure_metric,
                ]
            ]
            .copy()
        )

        difficulty_failure_chart.columns = [
            "Difficulty",
            "Failure Rate",
        ]

        st.bar_chart(
            difficulty_failure_chart,
            x="Difficulty",
            y="Failure Rate",
        )

        # ----------------------------------------------------
        # Failure analysis by category
        # ----------------------------------------------------

        st.markdown(
            "#### 🗂️ Failure Signals by Category"
        )

        category_failure_df = (
            failure_df
            .groupby("category")
            .agg(
                Questions=("question", "count"),
                Low_Groundedness=(
                    "Low_Groundedness",
                    "mean",
                ),
                Low_Context_Relevance=(
                    "Low_Context_Relevance",
                    "mean",
                ),
                Retrieval_Miss=(
                    "Retrieval_Miss",
                    "mean",
                ),
            )
            .reset_index()
        )

        st.dataframe(
            category_failure_df.style.format(
                {
                    "Low_Groundedness": "{:.1%}",
                    "Low_Context_Relevance": "{:.1%}",
                    "Retrieval_Miss": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        category_failure_metric = st.selectbox(
            "Failure metric by category",
            [
                "Low_Groundedness",
                "Low_Context_Relevance",
                "Retrieval_Miss",
            ],
            format_func=(
                lambda value:
                value.replace(
                    "_",
                    " ",
                )
            ),
            key="failure_category_metric",
        )

        category_failure_chart = (
            category_failure_df[
                [
                    "category",
                    category_failure_metric,
                ]
            ]
            .copy()
        )

        category_failure_chart.columns = [
            "Category",
            "Failure Rate",
        ]

        st.bar_chart(
            category_failure_chart,
            x="Category",
            y="Failure Rate",
        )

        st.caption(
            "Failure signals are rule-based diagnostics derived "
            "from the recorded experiment. Low groundedness and "
            "low context relevance use a 0.50 threshold, while "
            "retrieval misses are questions where the expected "
            "source was not found. These signals are diagnostic "
            "proxies rather than definitive measures of factual "
            "correctness."
        )

    st.divider()

    # ========================================================
    # EXPERIMENT INTERPRETATION
    # ========================================================

    with st.expander(
        "ℹ️ Normal LLM vs RAG Methodology & Limitations"
    ):

        st.markdown(
            """
            ### Experiment Design

            The same 30-question evaluation dataset was used for
            both response paths:

            **Normal LLM**

            Question → Ollama Cloud → Answer

            **DocMind RAG**

            Question → Hybrid Retrieval → Reranking →
            Retrieved Context → Grounded Prompt →
            Ollama Cloud → Answer + Sources

            The experiment records:

            - Normal LLM generation latency
            - RAG retrieval latency
            - RAG context-building latency
            - RAG generation latency
            - RAG end-to-end latency
            - Normal LLM evidence-support proxy
            - RAG groundedness proxy
            - RAG context relevance
            - Citation availability
            - Expected-source retrieval
            - Abstention behavior
            - Errors

            ### Interpretation

            The evidence-support and groundedness values are
            automated lexical/evidence-support proxies. They are
            useful for comparing the recorded experiment behavior,
            but they are not definitive factual correctness or
            hallucination scores.

            The experiment therefore should not be interpreted as
            proving that one response is factually correct solely
            from these proxy metrics.

            ### Latency

            RAG retrieval can have a higher first-request latency
            when the cross-encoder reranker model is loaded for the
            first time. Subsequent warm retrieval requests are
            substantially faster.

            The recorded experiment keeps this behavior visible
            instead of artificially removing the cold-start result.

            ### Citations

            The normal LLM path does not receive the knowledge-base
            retrieval metadata, so it does not provide DocMind-style
            source attribution.

            The RAG path returns source metadata associated with
            retrieved chunks.

            ### Reproducibility

            The dashboard reads the already-generated CSV file:

            `docs/normal_vs_rag_results.csv`

            Refreshing the Streamlit dashboard does not rerun the
            30-question experiment or generate additional Ollama
            Cloud responses.
            """
        )


    # ========================================================
    # FINAL EVALUATION SUMMARY
    # ========================================================

    st.markdown(
        "## 🏁 Final Evaluation Summary"
    )

    st.caption(
        "Consolidated view of the recorded Normal LLM vs RAG "
        "experiment. All values are calculated from the existing "
        "30-question evaluation dataset."
    )

    # --------------------------------------------------------
    # Dataset overview
    # --------------------------------------------------------

    st.markdown(
        "### 📋 Evaluation Dataset"
    )

    summary_total_questions = len(dataframe)

    summary_difficulties = (
        dataframe["difficulty"]
        .nunique()
        if "difficulty" in dataframe.columns
        else None
    )

    summary_categories = (
        dataframe["category"]
        .nunique()
        if "category" in dataframe.columns
        else None
    )

    summary_successful = None

    if "normal_error" in dataframe.columns:
        normal_errors = dataframe[
            "normal_error"
        ].fillna("").astype(str).str.strip()

        summary_normal_success = int(
            (normal_errors == "").sum()
        )
    else:
        summary_normal_success = None

    if "rag_error" in dataframe.columns:
        rag_errors = dataframe[
            "rag_error"
        ].fillna("").astype(str).str.strip()

        summary_rag_success = int(
            (rag_errors == "").sum()
        )
    else:
        summary_rag_success = None

    dataset_cols = st.columns(4)

    with dataset_cols[0]:

        st.metric(
            "Questions",
            summary_total_questions,
        )

    with dataset_cols[1]:

        st.metric(
            "Difficulty Levels",
            (
                summary_difficulties
                if summary_difficulties is not None
                else "N/A"
            ),
        )

    with dataset_cols[2]:

        st.metric(
            "Categories",
            (
                summary_categories
                if summary_categories is not None
                else "N/A"
            ),
        )

    with dataset_cols[3]:

        if (
            summary_normal_success is not None
            and summary_rag_success is not None
        ):

            st.metric(
                "Successful Comparisons",
                min(
                    summary_normal_success,
                    summary_rag_success,
                ),
            )

        else:

            st.metric(
                "Successful Comparisons",
                "N/A",
            )

    st.divider()

    # --------------------------------------------------------
    # Overall recorded metrics
    # --------------------------------------------------------

    st.markdown(
        "### 📊 Overall Recorded Metrics"
    )

    final_normal_evidence = normal_vs_rag_mean(
        dataframe,
        "normal_evidence_support_proxy",
    )

    final_rag_groundedness = normal_vs_rag_mean(
        dataframe,
        "rag_groundedness_proxy",
    )

    final_rag_relevance = normal_vs_rag_mean(
        dataframe,
        "rag_context_relevance",
    )

    final_expected_source = normal_vs_rag_mean(
        dataframe,
        "expected_source_found",
    )

    final_rag_citation = normal_vs_rag_mean(
        dataframe,
        "rag_citation_available",
    )

    final_normal_citation = normal_vs_rag_mean(
        dataframe,
        "normal_citation_available",
    )

    final_rag_abstention = normal_vs_rag_mean(
        dataframe,
        "rag_abstained",
    )

    final_normal_abstention = normal_vs_rag_mean(
        dataframe,
        "normal_abstained",
    )

    final_rag_sources = normal_vs_rag_mean(
        dataframe,
        "rag_source_count",
    )

    final_metric_cols = st.columns(4)

    with final_metric_cols[0]:

        st.metric(
            "Normal Evidence Proxy",
            (
                f"{final_normal_evidence:.3f}"
                if final_normal_evidence is not None
                else "N/A"
            ),
        )

    with final_metric_cols[1]:

        st.metric(
            "RAG Groundedness Proxy",
            (
                f"{final_rag_groundedness:.3f}"
                if final_rag_groundedness is not None
                else "N/A"
            ),
        )

    with final_metric_cols[2]:

        st.metric(
            "RAG Context Relevance",
            (
                f"{final_rag_relevance:.3f}"
                if final_rag_relevance is not None
                else "N/A"
            ),
        )

    with final_metric_cols[3]:

        st.metric(
            "Expected Source Found",
            (
                f"{final_expected_source:.1%}"
                if final_expected_source is not None
                else "N/A"
            ),
        )

    final_metric_cols2 = st.columns(4)

    with final_metric_cols2[0]:

        st.metric(
            "Normal Citations",
            (
                f"{final_normal_citation:.1%}"
                if final_normal_citation is not None
                else "N/A"
            ),
        )

    with final_metric_cols2[1]:

        st.metric(
            "RAG Citations",
            (
                f"{final_rag_citation:.1%}"
                if final_rag_citation is not None
                else "N/A"
            ),
        )

    with final_metric_cols2[2]:

        st.metric(
            "RAG Abstention",
            (
                f"{final_rag_abstention:.1%}"
                if final_rag_abstention is not None
                else "N/A"
            ),
        )

    with final_metric_cols2[3]:

        st.metric(
            "Average RAG Sources",
            (
                f"{final_rag_sources:.2f}"
                if final_rag_sources is not None
                else "N/A"
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # Latency summary
    # --------------------------------------------------------

    st.markdown(
        "### ⏱️ Latency Summary"
    )

    final_latency_columns = [
        "normal_generation_ms",
        "rag_retrieval_ms",
        "rag_context_ms",
        "rag_generation_ms",
        "rag_total_ms",
    ]

    final_latency_values = {}

    for column in final_latency_columns:

        if column in dataframe.columns:

            values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).dropna()

            if not values.empty:

                final_latency_values[column] = {
                    "Average": values.mean(),
                    "P50": values.quantile(0.50),
                    "P95": values.quantile(0.95),
                    "Maximum": values.max(),
                }

    if final_latency_values:

        final_latency_rows = []

        final_latency_names = {
            "normal_generation_ms": (
                "Normal LLM Generation"
            ),
            "rag_retrieval_ms": (
                "RAG Retrieval"
            ),
            "rag_context_ms": (
                "RAG Context Build"
            ),
            "rag_generation_ms": (
                "RAG Generation"
            ),
            "rag_total_ms": (
                "RAG End-to-End"
            ),
        }

        for column, values in final_latency_values.items():

            final_latency_rows.append(
                {
                    "Stage": final_latency_names.get(
                        column,
                        column,
                    ),
                    "Average ms": values["Average"],
                    "P50 ms": values["P50"],
                    "P95 ms": values["P95"],
                    "Maximum ms": values["Maximum"],
                }
            )

        final_latency_df = pd.DataFrame(
            final_latency_rows
        )

        st.dataframe(
            final_latency_df.style.format(
                {
                    "Average ms": "{:.0f}",
                    "P50 ms": "{:.0f}",
                    "P95 ms": "{:.0f}",
                    "Maximum ms": "{:.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "Recorded latency metrics are not available."
        )

    st.divider()

    # --------------------------------------------------------
    # Failure signal summary
    # --------------------------------------------------------

    st.markdown(
        "### 🚨 Recorded Failure Signals"
    )

    final_failure_available = all(
        column in dataframe.columns
        for column in [
            "rag_groundedness_proxy",
            "rag_context_relevance",
            "expected_source_found",
            "rag_abstained",
            "rag_source_count",
        ]
    )

    if final_failure_available:

        final_failure_df = dataframe.copy()

        final_failure_df[
            "final_groundedness"
        ] = pd.to_numeric(
            final_failure_df[
                "rag_groundedness_proxy"
            ],
            errors="coerce",
        )

        final_failure_df[
            "final_relevance"
        ] = pd.to_numeric(
            final_failure_df[
                "rag_context_relevance"
            ],
            errors="coerce",
        )

        final_failure_df[
            "final_sources"
        ] = pd.to_numeric(
            final_failure_df[
                "rag_source_count"
            ],
            errors="coerce",
        )

        final_failure_df[
            "final_expected_source"
        ] = (
            final_failure_df[
                "expected_source_found"
            ]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                    "y",
                ]
            )
        )

        final_failure_df[
            "final_abstained"
        ] = (
            final_failure_df[
                "rag_abstained"
            ]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                    "y",
                ]
            )
        )

        final_low_groundedness = int(
            (
                final_failure_df[
                    "final_groundedness"
                ] < 0.50
            ).sum()
        )

        final_low_relevance = int(
            (
                final_failure_df[
                    "final_relevance"
                ] < 0.50
            ).sum()
        )

        final_retrieval_misses = int(
            (
                ~final_failure_df[
                    "final_expected_source"
                ]
            ).sum()
        )

        final_abstentions = int(
            final_failure_df[
                "final_abstained"
            ].sum()
        )

        final_no_sources = int(
            (
                final_failure_df[
                    "final_sources"
                ].fillna(0)
                <= 0
            ).sum()
        )

        final_failure_summary = pd.DataFrame(
            {
                "Signal": [
                    "Low Groundedness",
                    "Low Context Relevance",
                    "Retrieval Miss",
                    "RAG Abstention",
                    "No Sources Retrieved",
                ],
                "Questions": [
                    final_low_groundedness,
                    final_low_relevance,
                    final_retrieval_misses,
                    final_abstentions,
                    final_no_sources,
                ],
            }
        )

        st.bar_chart(
            final_failure_summary,
            x="Signal",
            y="Questions",
        )

        st.dataframe(
            final_failure_summary,
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "Failure-signal fields are not available "
            "in the recorded dataset."
        )

    st.divider()

    # --------------------------------------------------------
    # What the experiment records
    # --------------------------------------------------------

    st.markdown(
        "### 🧠 What the Recorded Experiment Shows"
    )

    interpretation_items = []

    if (
        final_rag_groundedness is not None
        and final_normal_evidence is not None
    ):

        evidence_difference = (
            final_rag_groundedness
            - final_normal_evidence
        )

        interpretation_items.append(
            f"- The recorded RAG groundedness proxy "
            f"was {final_rag_groundedness:.3f}, while the "
            f"normal LLM evidence-support proxy was "
            f"{final_normal_evidence:.3f}. The recorded "
            f"difference was {evidence_difference:+.3f}."
        )

    if final_rag_relevance is not None:

        interpretation_items.append(
            f"- The recorded average RAG context relevance "
            f"was {final_rag_relevance:.3f}."
        )

    if final_expected_source is not None:

        interpretation_items.append(
            f"- The expected source was found in "
            f"{final_expected_source:.1%} of the recorded "
            f"questions."
        )

    if (
        final_rag_citation is not None
        and final_normal_citation is not None
    ):

        interpretation_items.append(
            f"- Recorded citation availability was "
            f"{final_normal_citation:.1%} for the normal "
            f"LLM path and {final_rag_citation:.1%} for "
            f"the RAG path."
        )

    if (
        normal_generation is not None
        and rag_total is not None
    ):

        interpretation_items.append(
            f"- Average recorded latency was "
            f"{format_duration_ms(normal_generation)} "
            f"for normal LLM generation and "
            f"{format_duration_ms(rag_total)} for "
            f"RAG end-to-end processing."
        )

    if final_rag_sources is not None:

        interpretation_items.append(
            f"- The RAG pipeline returned an average of "
            f"{final_rag_sources:.2f} recorded sources "
            f"per question."
        )

    if interpretation_items:

        for item in interpretation_items:

            st.markdown(item)

    else:

        st.info(
            "No consolidated interpretation metrics "
            "are available."
        )

    st.caption(
        "These statements describe recorded experimental "
        "measurements. Evidence-support and groundedness "
        "values are automated proxies and should not be "
        "treated as definitive factual-correctness scores. "
        "The dashboard does not assign an overall winner "
        "or ranking."
    )

    st.divider()


# ============================================================
# MAIN EVALUATION DASHBOARD
# ============================================================
def render_evaluation_dashboard():
    """
    Render the complete RAG Evaluation & Analytics dashboard.

    All benchmark values are loaded dynamically from CSV files.
    """

    st.header(
        "📊 RAG Evaluation & Analytics"
    )

    st.write(
        "Analyze DocMind's retrieval, hybrid search, "
        "reranking, chunking, answer quality, "
        "Normal LLM vs RAG comparison, and "
        "question-level evaluation experiments."
    )

    st.caption(
        "Benchmark results are loaded dynamically from "
        "the project's `docs/` evaluation CSV files."
    )

    # ========================================================
    # ANSWER QUALITY
    # ========================================================

    render_answer_quality_dashboard()

    st.divider()

    # ========================================================
    # NORMAL LLM VS RAG
    # ========================================================

    render_normal_vs_rag_dashboard()

    st.divider()

    # ========================================================
    # EXISTING RETRIEVAL EVALUATION DATA
    # ========================================================

    data = get_evaluation_data()

    hybrid_df = data.get(
        "hybrid",
        pd.DataFrame(),
    )

    weights_df = data.get(
        "hybrid_weights",
        pd.DataFrame(),
    )

    reranker_df = data.get(
        "reranker",
        pd.DataFrame(),
    )

    reranker_summary_df = data.get(
        "reranker_summary",
        pd.DataFrame(),
    )

    distance_df = data.get(
        "distance",
        pd.DataFrame(),
    )

    chunking_df = data.get(
        "chunking",
        pd.DataFrame(),
    )

    # ========================================================
    # DASHBOARD STATUS
    # ========================================================

    available_experiments = sum(
        not dataframe.empty
        for dataframe in [
            hybrid_df,
            weights_df,
            reranker_df,
            distance_df,
            chunking_df,
        ]
    )

    st.success(
        f"📈 {available_experiments} of 5 evaluation "
        "experiment datasets are available."
    )
    # ========================================================
    # BENCHMARK OVERVIEW
    # ========================================================

    st.markdown(
        "### 📌 Benchmark Overview"
    )

    st.caption(
        "Summary of the recorded evaluation experiments "
        "currently available to the DocMind dashboard."
    )

    # --------------------------------------------------------
    # Experiment availability
    # --------------------------------------------------------

    experiment_datasets = [
        (
            "Hybrid Retrieval",
            hybrid_df,
            "Dense vs BM25 vs Hybrid retrieval",
        ),
        (
            "Retrieval Weights",
            weights_df,
            "Hybrid retrieval weight configurations",
        ),
        (
            "Reranker",
            reranker_df,
            "Reranking impact on retrieval quality",
        ),
        (
            "Distance / Similarity",
            distance_df,
            "Vector similarity / distance comparison",
        ),
        (
            "Chunking",
            chunking_df,
            "Chunk-size and overlap experiments",
        ),
    ]

    available_experiment_count = sum(
        not dataframe.empty
        for _, dataframe, _ in experiment_datasets
    )

    total_experiment_count = len(
        experiment_datasets
    )

    # --------------------------------------------------------
    # KPI cards
    # --------------------------------------------------------

    overview_cols = st.columns(5)

    with overview_cols[0]:

        st.metric(
            "Experiments Available",
            (
                f"{available_experiment_count}/"
                f"{total_experiment_count}"
            ),
        )

    with overview_cols[1]:

        st.metric(
            "Evaluation Questions",
            (
                len(hybrid_df)
                if not hybrid_df.empty
                else "N/A"
            ),
        )

    with overview_cols[2]:

        st.metric(
            "Reranker Questions",
            (
                len(reranker_df)
                if not reranker_df.empty
                else "N/A"
            ),
        )

    with overview_cols[3]:

        st.metric(
            "Weight Configurations",
            (
                len(weights_df)
                if not weights_df.empty
                else "N/A"
            ),
        )

    with overview_cols[4]:

        st.metric(
            "Chunk Configurations",
            (
                len(chunking_df)
                if not chunking_df.empty
                else "N/A"
            ),
        )

    st.markdown(
        "#### 🧪 Experiment Coverage"
    )

    # --------------------------------------------------------
    # Build experiment coverage table
    # --------------------------------------------------------

    coverage_rows = []

    for (
        experiment_name,
        experiment_dataframe,
        experiment_description,
    ) in experiment_datasets:

        is_available = (
            not experiment_dataframe.empty
        )

        coverage_rows.append(
            {
                "Experiment": experiment_name,
                "Status": (
                    "Available"
                    if is_available
                    else "Missing"
                ),
                "Recorded Rows": (
                    len(experiment_dataframe)
                    if is_available
                    else 0
                ),
                "Purpose": experiment_description,
            }
        )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    st.dataframe(
        coverage_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # Evaluation distribution
    # --------------------------------------------------------

    if not hybrid_df.empty:

        st.markdown(
            "#### 📊 Evaluation Distribution"
        )

        distribution_cols = st.columns(2)

        with distribution_cols[0]:

            if "difficulty" in hybrid_df.columns:

                benchmark_difficulty = (
                    hybrid_df[
                        "difficulty"
                    ]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .rename_axis(
                        "Difficulty"
                    )
                    .reset_index(
                        name="Questions"
                    )
                )

                st.markdown(
                    "**Questions by Difficulty**"
                )

                st.bar_chart(
                    benchmark_difficulty,
                    x="Difficulty",
                    y="Questions",
                )

        with distribution_cols[1]:

            if "category" in hybrid_df.columns:

                benchmark_category = (
                    hybrid_df[
                        "category"
                    ]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .rename_axis(
                        "Category"
                    )
                    .reset_index(
                        name="Questions"
                    )
                )

                st.markdown(
                    "**Questions by Category**"
                )

                st.bar_chart(
                    benchmark_category,
                    x="Category",
                    y="Questions",
                )

    st.caption(
        "The dashboard reads these benchmark files from the "
        "existing evaluation results. No new retrieval, "
        "embedding, reranking, or generation experiments are "
        "performed when this dashboard is refreshed."
    )

    st.divider()
    # ========================================================
    # RETRIEVAL METHOD COMPARISON
    # ========================================================

    st.markdown(
        "### 🔍 Retrieval Method Comparison"
    )

    st.caption(
        "Comparison of Dense, BM25, and Hybrid retrieval "
        "using the recorded evaluation results."
    )

    required_hybrid_columns = [
        "dense_hit_at_3",
        "bm25_hit_at_3",
        "hybrid_hit_at_3",
        "dense_mrr",
        "bm25_mrr",
        "hybrid_mrr",
    ]

    if (
        hybrid_df.empty
        or not all(
            column in hybrid_df.columns
            for column in required_hybrid_columns
        )
    ):

        render_missing_evaluation_message(
            "hybrid_retrieval_experiment.csv"
        )

    else:

        # ----------------------------------------------------
        # Calculate recorded averages
        # ----------------------------------------------------

        dense_hit3 = (
            hybrid_df[
                "dense_hit_at_3"
            ].mean()
        )

        bm25_hit3 = (
            hybrid_df[
                "bm25_hit_at_3"
            ].mean()
        )

        hybrid_hit3 = (
            hybrid_df[
                "hybrid_hit_at_3"
            ].mean()
        )

        dense_mrr = (
            hybrid_df[
                "dense_mrr"
            ].mean()
        )

        bm25_mrr = (
            hybrid_df[
                "bm25_mrr"
            ].mean()
        )

        hybrid_mrr = (
            hybrid_df[
                "hybrid_mrr"
            ].mean()
        )

        # ----------------------------------------------------
        # KPI cards
        # ----------------------------------------------------

        method_cols = st.columns(3)

        with method_cols[0]:

            st.markdown(
                "#### 🧠 Dense"
            )

            st.metric(
                "Hit@3",
                percent(dense_hit3),
            )

            st.metric(
                "MRR",
                f"{dense_mrr:.3f}",
            )

        with method_cols[1]:

            st.markdown(
                "#### 🔤 BM25"
            )

            st.metric(
                "Hit@3",
                percent(bm25_hit3),
            )

            st.metric(
                "MRR",
                f"{bm25_mrr:.3f}",
            )

        with method_cols[2]:

            st.markdown(
                "#### 🔀 Hybrid"
            )

            st.metric(
                "Hit@3",
                percent(hybrid_hit3),
            )

            st.metric(
                "MRR",
                f"{hybrid_mrr:.3f}",
            )

        st.markdown(
            "#### 📋 Recorded Retrieval Scores"
        )

        comparison_df = pd.DataFrame(
            {
                "Method": [
                    "Dense",
                    "BM25",
                    "Hybrid",
                ],
                "Hit@3": [
                    dense_hit3,
                    bm25_hit3,
                    hybrid_hit3,
                ],
                "MRR": [
                    dense_mrr,
                    bm25_mrr,
                    hybrid_mrr,
                ],
            }
        )

        st.dataframe(
            comparison_df.style.format(
                {
                    "Hit@3": "{:.1%}",
                    "MRR": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown(
            "#### 📊 Hit@3 Comparison"
        )

        hit3_chart = pd.DataFrame(
            {
                "Method": [
                    "Dense",
                    "BM25",
                    "Hybrid",
                ],
                "Hit@3": [
                    dense_hit3,
                    bm25_hit3,
                    hybrid_hit3,
                ],
            }
        )

        st.bar_chart(
            hit3_chart,
            x="Method",
            y="Hit@3",
        )

        st.markdown(
            "#### 🎯 MRR Comparison"
        )

        mrr_chart = pd.DataFrame(
            {
                "Method": [
                    "Dense",
                    "BM25",
                    "Hybrid",
                ],
                "MRR": [
                    dense_mrr,
                    bm25_mrr,
                    hybrid_mrr,
                ],
            }
        )

        st.bar_chart(
            mrr_chart,
            x="Method",
            y="MRR",
        )

        # ----------------------------------------------------
        # Metric selector
        # ----------------------------------------------------

        st.markdown(
            "#### 🎚️ Interactive Retrieval Metric"
        )

        retrieval_metric = st.selectbox(
            "Select retrieval metric",
            [
                "Hit@3",
                "MRR",
            ],
            key="retrieval_method_metric",
        )

        if retrieval_metric == "Hit@3":

            interactive_values = [
                dense_hit3,
                bm25_hit3,
                hybrid_hit3,
            ]

        else:

            interactive_values = [
                dense_mrr,
                bm25_mrr,
                hybrid_mrr,
            ]

        interactive_chart = pd.DataFrame(
            {
                "Method": [
                    "Dense",
                    "BM25",
                    "Hybrid",
                ],
                "Score": interactive_values,
            }
        )

        st.bar_chart(
            interactive_chart,
            x="Method",
            y="Score",
        )

        # ----------------------------------------------------
        # Recorded differences
        # ----------------------------------------------------

        st.markdown(
            "#### 🔀 Recorded Hybrid Differences"
        )

        difference_cols = st.columns(2)

        with difference_cols[0]:

            hybrid_dense_hit_difference = (
                hybrid_hit3
                - dense_hit3
            )

            st.metric(
                "Hybrid vs Dense Hit@3",
                f"{hybrid_dense_hit_difference:+.3f}",
            )

        with difference_cols[1]:

            hybrid_bm25_hit_difference = (
                hybrid_hit3
                - bm25_hit3
            )

            st.metric(
                "Hybrid vs BM25 Hit@3",
                f"{hybrid_bm25_hit_difference:+.3f}",
            )

        difference_cols2 = st.columns(2)

        with difference_cols2[0]:

            hybrid_dense_mrr_difference = (
                hybrid_mrr
                - dense_mrr
            )

            st.metric(
                "Hybrid vs Dense MRR",
                f"{hybrid_dense_mrr_difference:+.3f}",
            )

        with difference_cols2[1]:

            hybrid_bm25_mrr_difference = (
                hybrid_mrr
                - bm25_mrr
            )

            st.metric(
                "Hybrid vs BM25 MRR",
                f"{hybrid_bm25_mrr_difference:+.3f}",
            )

        st.caption(
            "Hit@3 measures whether an expected source was "
            "retrieved within the top three results. MRR "
            "(Mean Reciprocal Rank) measures how highly the "
            "first relevant result was ranked. The difference "
            "cards show recorded numerical differences only "
            "and do not represent an overall system ranking."
        )

    st.divider()
    # ========================================================
    # RETRIEVAL QUALITY DEEP DIVE
    # ========================================================

    st.markdown(
        "### 📐 Retrieval Quality Deep Dive"
    )

    st.caption(
        "Detailed Precision@3, Recall@3, and Coverage@3 "
        "analysis across the recorded Dense, BM25, and "
        "Hybrid retrieval results."
    )

    quality_columns = [
        "dense_precision_at_3",
        "bm25_precision_at_3",
        "hybrid_precision_at_3",
        "dense_recall_at_3",
        "bm25_recall_at_3",
        "hybrid_recall_at_3",
        "dense_coverage_at_3",
        "bm25_coverage_at_3",
        "hybrid_coverage_at_3",
    ]

    if (
        hybrid_df.empty
        or not all(
            column in hybrid_df.columns
            for column in quality_columns
        )
    ):

        st.info(
            "Precision, recall, or coverage metrics "
            "are not available in the recorded "
            "hybrid retrieval experiment."
        )

    else:

        # ----------------------------------------------------
        # Calculate average quality metrics
        # ----------------------------------------------------

        dense_precision = (
            hybrid_df[
                "dense_precision_at_3"
            ].mean()
        )

        bm25_precision = (
            hybrid_df[
                "bm25_precision_at_3"
            ].mean()
        )

        hybrid_precision = (
            hybrid_df[
                "hybrid_precision_at_3"
            ].mean()
        )

        dense_recall = (
            hybrid_df[
                "dense_recall_at_3"
            ].mean()
        )

        bm25_recall = (
            hybrid_df[
                "bm25_recall_at_3"
            ].mean()
        )

        hybrid_recall = (
            hybrid_df[
                "hybrid_recall_at_3"
            ].mean()
        )

        dense_coverage = (
            hybrid_df[
                "dense_coverage_at_3"
            ].mean()
        )

        bm25_coverage = (
            hybrid_df[
                "bm25_coverage_at_3"
            ].mean()
        )

        hybrid_coverage = (
            hybrid_df[
                "hybrid_coverage_at_3"
            ].mean()
        )

        # ----------------------------------------------------
        # Quality summary table
        # ----------------------------------------------------

        st.markdown(
            "#### 📋 Recorded Retrieval Quality Scores"
        )

        quality_df = pd.DataFrame(
            {
                "Metric": [
                    "Precision@3",
                    "Recall@3",
                    "Coverage@3",
                ],
                "Dense": [
                    dense_precision,
                    dense_recall,
                    dense_coverage,
                ],
                "BM25": [
                    bm25_precision,
                    bm25_recall,
                    bm25_coverage,
                ],
                "Hybrid": [
                    hybrid_precision,
                    hybrid_recall,
                    hybrid_coverage,
                ],
            }
        )

        st.dataframe(
            quality_df.style.format(
                {
                    "Dense": "{:.3f}",
                    "BM25": "{:.3f}",
                    "Hybrid": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # Separate metric visualizations
        # ----------------------------------------------------

        st.markdown(
            "#### 📊 Precision@3 Comparison"
        )

        precision_chart = pd.DataFrame(
            {
                "Method": [
                    "Dense",
                    "BM25",
                    "Hybrid",
                ],
                "Precision@3": [
                    dense_precision,
                    bm25_precision,
                    hybrid_precision,
                ],
            }
        )

        st.bar_chart(
            precision_chart,
            x="Method",
            y="Precision@3",
        )

        st.markdown(
            "#### 📊 Recall@3 Comparison"
        )

        recall_chart = pd.DataFrame(
            {
                "Method": [
                    "Dense",
                    "BM25",
                    "Hybrid",
                ],
                "Recall@3": [
                    dense_recall,
                    bm25_recall,
                    hybrid_recall,
                ],
            }
        )

        st.bar_chart(
            recall_chart,
            x="Method",
            y="Recall@3",
        )

        st.markdown(
            "#### 📊 Coverage@3 Comparison"
        )

        coverage_chart = pd.DataFrame(
            {
                "Method": [
                    "Dense",
                    "BM25",
                    "Hybrid",
                ],
                "Coverage@3": [
                    dense_coverage,
                    bm25_coverage,
                    hybrid_coverage,
                ],
            }
        )

        st.bar_chart(
            coverage_chart,
            x="Method",
            y="Coverage@3",
        )

        # ----------------------------------------------------
        # Interactive metric selector
        # ----------------------------------------------------

        st.markdown(
            "#### 🎚️ Interactive Quality Metric"
        )

        quality_metric = st.selectbox(
            "Select retrieval quality metric",
            [
                "Precision@3",
                "Recall@3",
                "Coverage@3",
            ],
            key="hybrid_quality_metric",
        )

        quality_values = {
            "Precision@3": [
                dense_precision,
                bm25_precision,
                hybrid_precision,
            ],
            "Recall@3": [
                dense_recall,
                bm25_recall,
                hybrid_recall,
            ],
            "Coverage@3": [
                dense_coverage,
                bm25_coverage,
                hybrid_coverage,
            ],
        }

        interactive_quality_chart = pd.DataFrame(
            {
                "Method": [
                    "Dense",
                    "BM25",
                    "Hybrid",
                ],
                "Score": quality_values[
                    quality_metric
                ],
            }
        )

        st.bar_chart(
            interactive_quality_chart,
            x="Method",
            y="Score",
        )

        # ----------------------------------------------------
        # Hybrid difference analysis
        # ----------------------------------------------------

        st.markdown(
            "#### 🔀 Recorded Hybrid Differences"
        )

        precision_dense_difference = (
            hybrid_precision
            - dense_precision
        )

        precision_bm25_difference = (
            hybrid_precision
            - bm25_precision
        )

        recall_dense_difference = (
            hybrid_recall
            - dense_recall
        )

        recall_bm25_difference = (
            hybrid_recall
            - bm25_recall
        )

        coverage_dense_difference = (
            hybrid_coverage
            - dense_coverage
        )

        coverage_bm25_difference = (
            hybrid_coverage
            - bm25_coverage
        )

        difference_cols = st.columns(3)

        with difference_cols[0]:

            st.metric(
                "Hybrid vs Dense Precision",
                f"{precision_dense_difference:+.3f}",
            )

            st.metric(
                "Hybrid vs BM25 Precision",
                f"{precision_bm25_difference:+.3f}",
            )

        with difference_cols[1]:

            st.metric(
                "Hybrid vs Dense Recall",
                f"{recall_dense_difference:+.3f}",
            )

            st.metric(
                "Hybrid vs BM25 Recall",
                f"{recall_bm25_difference:+.3f}",
            )

        with difference_cols[2]:

            st.metric(
                "Hybrid vs Dense Coverage",
                f"{coverage_dense_difference:+.3f}",
            )

            st.metric(
                "Hybrid vs BM25 Coverage",
                f"{coverage_bm25_difference:+.3f}",
            )

        # ----------------------------------------------------
        # Relative improvement analysis
        # ----------------------------------------------------

        st.markdown(
            "#### 📈 Recorded Relative Differences"
        )

        def relative_difference(
            hybrid_value,
            baseline_value,
        ):

            if (
                baseline_value is None
                or pd.isna(baseline_value)
                or baseline_value == 0
            ):

                return None

            return (
                hybrid_value
                - baseline_value
            ) / baseline_value

        precision_dense_relative = (
            relative_difference(
                hybrid_precision,
                dense_precision,
            )
        )

        precision_bm25_relative = (
            relative_difference(
                hybrid_precision,
                bm25_precision,
            )
        )

        recall_dense_relative = (
            relative_difference(
                hybrid_recall,
                dense_recall,
            )
        )

        recall_bm25_relative = (
            relative_difference(
                hybrid_recall,
                bm25_recall,
            )
        )

        coverage_dense_relative = (
            relative_difference(
                hybrid_coverage,
                dense_coverage,
            )
        )

        coverage_bm25_relative = (
            relative_difference(
                hybrid_coverage,
                bm25_coverage,
            )
        )

        relative_df = pd.DataFrame(
            {
                "Metric": [
                    "Precision@3",
                    "Recall@3",
                    "Coverage@3",
                ],
                "Hybrid vs Dense": [
                    precision_dense_relative,
                    recall_dense_relative,
                    coverage_dense_relative,
                ],
                "Hybrid vs BM25": [
                    precision_bm25_relative,
                    recall_bm25_relative,
                    coverage_bm25_relative,
                ],
            }
        )

        st.dataframe(
            relative_df.style.format(
                {
                    "Hybrid vs Dense": (
                        lambda value:
                        "N/A"
                        if pd.isna(value)
                        else f"{value:+.1%}"
                    ),
                    "Hybrid vs BM25": (
                        lambda value:
                        "N/A"
                        if pd.isna(value)
                        else f"{value:+.1%}"
                    ),
                }
            ),
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # Evaluation metadata
        # ----------------------------------------------------

        metadata_cols = st.columns(2)

        with metadata_cols[0]:

            st.metric(
                "Recorded Questions",
                len(hybrid_df),
            )

        with metadata_cols[1]:

            if "difficulty" in hybrid_df.columns:

                st.metric(
                    "Difficulty Levels",
                    hybrid_df[
                        "difficulty"
                    ]
                    .dropna()
                    .nunique(),
                )

            else:

                st.metric(
                    "Difficulty Levels",
                    "N/A",
                )

        st.caption(
            "Precision@3 measures the proportion of the "
            "top-three retrieved results that match expected "
            "sources. Recall@3 measures how much of the "
            "expected source set was retrieved within the "
            "top three. Coverage@3 indicates whether the "
            "expected source information was covered by the "
            "retrieved results. Relative differences are "
            "calculated against the corresponding recorded "
            "baseline and are not an overall system ranking."
        )
    # ========================================================
    # RETRIEVAL PERFORMANCE BY DIFFICULTY & CATEGORY
    # ========================================================

    st.markdown(
        "### 🧩 Retrieval Performance by Difficulty & Category"
    )

    segmentation_required = [
        "difficulty",
        "category",
        "hybrid_hit_at_3",
        "hybrid_precision_at_3",
        "hybrid_recall_at_3",
        "hybrid_coverage_at_3",
        "hybrid_mrr",
    ]

    if not all(
        column in hybrid_df.columns
        for column in segmentation_required
    ):

        st.info(
            "Difficulty/category retrieval metrics are "
            "not available in the recorded hybrid "
            "retrieval experiment."
        )

    else:

        # ----------------------------------------------------
        # Difficulty analysis
        # ----------------------------------------------------

        st.markdown(
            "#### 🎚️ Performance by Question Difficulty"
        )

        difficulty_df = (
            hybrid_df
            .groupby("difficulty")
            .agg(
                Questions=("question", "count"),
                Hit_at_3=("hybrid_hit_at_3", "mean"),
                Precision_at_3=(
                    "hybrid_precision_at_3",
                    "mean",
                ),
                Recall_at_3=(
                    "hybrid_recall_at_3",
                    "mean",
                ),
                Coverage_at_3=(
                    "hybrid_coverage_at_3",
                    "mean",
                ),
                MRR=("hybrid_mrr", "mean"),
            )
            .reset_index()
        )

        st.dataframe(
            difficulty_df.style.format(
                {
                    "Hit_at_3": "{:.3f}",
                    "Precision_at_3": "{:.3f}",
                    "Recall_at_3": "{:.3f}",
                    "Coverage_at_3": "{:.3f}",
                    "MRR": "{:.3f}",
                }
            ),
            width="stretch",
        )

        difficulty_metric = st.selectbox(
            "Difficulty analysis metric",
            [
                "Hit_at_3",
                "Precision_at_3",
                "Recall_at_3",
                "Coverage_at_3",
                "MRR",
            ],
            key="retrieval_difficulty_metric",
        )

        difficulty_chart = difficulty_df[
            [
                "difficulty",
                difficulty_metric,
            ]
        ].copy()

        difficulty_chart.columns = [
            "Difficulty",
            "Score",
        ]

        st.bar_chart(
            difficulty_chart,
            x="Difficulty",
            y="Score",
        )

        # ----------------------------------------------------
        # Category analysis
        # ----------------------------------------------------

        st.markdown(
            "#### 🗂️ Performance by Question Category"
        )

        category_df = (
            hybrid_df
            .groupby("category")
            .agg(
                Questions=("question", "count"),
                Hit_at_3=("hybrid_hit_at_3", "mean"),
                Precision_at_3=(
                    "hybrid_precision_at_3",
                    "mean",
                ),
                Recall_at_3=(
                    "hybrid_recall_at_3",
                    "mean",
                ),
                Coverage_at_3=(
                    "hybrid_coverage_at_3",
                    "mean",
                ),
                MRR=("hybrid_mrr", "mean"),
            )
            .reset_index()
        )

        st.dataframe(
            category_df.style.format(
                {
                    "Hit_at_3": "{:.3f}",
                    "Precision_at_3": "{:.3f}",
                    "Recall_at_3": "{:.3f}",
                    "Coverage_at_3": "{:.3f}",
                    "MRR": "{:.3f}",
                }
            ),
            width="stretch",
        )

        category_metric = st.selectbox(
            "Category analysis metric",
            [
                "Hit_at_3",
                "Precision_at_3",
                "Recall_at_3",
                "Coverage_at_3",
                "MRR",
            ],
            key="retrieval_category_metric",
        )

        category_chart = category_df[
            [
                "category",
                category_metric,
            ]
        ].copy()

        category_chart.columns = [
            "Category",
            "Score",
        ]

        st.bar_chart(
            category_chart,
            x="Category",
            y="Score",
        )

        # ----------------------------------------------------
        # Question-count overview
        # ----------------------------------------------------

        st.markdown(
            "#### 📊 Evaluation Distribution"
        )

        distribution_cols = st.columns(2)

        with distribution_cols[0]:

            st.markdown(
                "**Questions by Difficulty**"
            )

            difficulty_counts = (
                hybrid_df[
                    "difficulty"
                ]
                .value_counts()
                .rename_axis("Difficulty")
                .reset_index(
                    name="Questions"
                )
            )

            st.bar_chart(
                difficulty_counts,
                x="Difficulty",
                y="Questions",
            )

        with distribution_cols[1]:

            st.markdown(
                "**Questions by Category**"
            )

            category_counts = (
                hybrid_df[
                    "category"
                ]
                .value_counts()
                .rename_axis("Category")
                .reset_index(
                    name="Questions"
                )
            )

            st.bar_chart(
                category_counts,
                x="Category",
                y="Questions",
            )

        st.caption(
            "Difficulty and category results are calculated "
            "from the recorded hybrid retrieval evaluation. "
            "They show how retrieval quality varies across "
            "different question types and difficulty levels."
        )

    st.divider()

    # ========================================================
    # RERANKER IMPACT
    # ========================================================

    st.markdown(
        "### 🎯 Cross-Encoder Reranker Impact"
    )

    st.caption(
        "Comparison of the recorded baseline retrieval "
        "results with cross-encoder reranked results."
    )

    reranker_required = [
        "baseline_hit_at_1",
        "reranker_hit_at_1",
        "baseline_mrr",
        "reranker_mrr",
        "baseline_source_coverage_at_3",
        "reranker_source_coverage_at_3",
    ]

    if (
        reranker_df.empty
        or not all(
            column in reranker_df.columns
            for column in reranker_required
        )
    ):

        render_missing_evaluation_message(
            "reranker_metrics_experiment.csv"
        )

    else:

        # ----------------------------------------------------
        # Calculate recorded averages
        # ----------------------------------------------------

        baseline_hit1 = (
            reranker_df[
                "baseline_hit_at_1"
            ].mean()
        )

        reranked_hit1 = (
            reranker_df[
                "reranker_hit_at_1"
            ].mean()
        )

        baseline_mrr = (
            reranker_df[
                "baseline_mrr"
            ].mean()
        )

        reranked_mrr = (
            reranker_df[
                "reranker_mrr"
            ].mean()
        )

        baseline_coverage = (
            reranker_df[
                "baseline_source_coverage_at_3"
            ].mean()
        )

        reranked_coverage = (
            reranker_df[
                "reranker_source_coverage_at_3"
            ].mean()
        )

        # ----------------------------------------------------
        # KPI summary
        # ----------------------------------------------------

        st.markdown(
            "#### 📌 Reranker Summary"
        )

        reranker_cols = st.columns(3)

        with reranker_cols[0]:

            st.metric(
                "Hit@1",
                percent(
                    reranked_hit1
                ),
                delta=render_metric_delta(
                    reranked_hit1,
                    baseline_hit1,
                ),
            )

            st.caption(
                f"Baseline: "
                f"{percent(baseline_hit1)}"
            )

        with reranker_cols[1]:

            st.metric(
                "MRR",
                f"{reranked_mrr:.3f}",
                delta=(
                    f"{reranked_mrr - baseline_mrr:+.3f}"
                    if (
                        baseline_mrr is not None
                        and reranked_mrr is not None
                    )
                    else None
                ),
            )

            st.caption(
                f"Baseline: "
                f"{baseline_mrr:.3f}"
            )

        with reranker_cols[2]:

            st.metric(
                "Source Coverage@3",
                percent(
                    reranked_coverage
                ),
                delta=render_metric_delta(
                    reranked_coverage,
                    baseline_coverage,
                ),
            )

            st.caption(
                f"Baseline: "
                f"{percent(baseline_coverage)}"
            )

        # ----------------------------------------------------
        # Comparison table
        # ----------------------------------------------------

        st.markdown(
            "#### 📋 Baseline vs Reranker"
        )

        reranker_comparison_df = pd.DataFrame(
            {
                "Metric": [
                    "Hit@1",
                    "MRR",
                    "Source Coverage@3",
                ],
                "Baseline": [
                    baseline_hit1,
                    baseline_mrr,
                    baseline_coverage,
                ],
                "Reranker": [
                    reranked_hit1,
                    reranked_mrr,
                    reranked_coverage,
                ],
                "Difference": [
                    reranked_hit1 - baseline_hit1,
                    reranked_mrr - baseline_mrr,
                    reranked_coverage - baseline_coverage,
                ],
            }
        )

        st.dataframe(
            reranker_comparison_df.style.format(
                {
                    "Baseline": "{:.3f}",
                    "Reranker": "{:.3f}",
                    "Difference": "{:+.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # Individual metric charts
        # ----------------------------------------------------

        st.markdown(
            "#### 📊 Hit@1 Comparison"
        )

        hit1_chart = pd.DataFrame(
            {
                "Method": [
                    "Baseline",
                    "Reranker",
                ],
                "Hit@1": [
                    baseline_hit1,
                    reranked_hit1,
                ],
            }
        )

        st.bar_chart(
            hit1_chart,
            x="Method",
            y="Hit@1",
        )

        st.markdown(
            "#### 🎯 MRR Comparison"
        )

        mrr_chart = pd.DataFrame(
            {
                "Method": [
                    "Baseline",
                    "Reranker",
                ],
                "MRR": [
                    baseline_mrr,
                    reranked_mrr,
                ],
            }
        )

        st.bar_chart(
            mrr_chart,
            x="Method",
            y="MRR",
        )

        st.markdown(
            "#### 📚 Source Coverage@3 Comparison"
        )

        coverage_chart = pd.DataFrame(
            {
                "Method": [
                    "Baseline",
                    "Reranker",
                ],
                "Source Coverage@3": [
                    baseline_coverage,
                    reranked_coverage,
                ],
            }
        )

        st.bar_chart(
            coverage_chart,
            x="Method",
            y="Source Coverage@3",
        )

        # ----------------------------------------------------
        # Interactive metric selector
        # ----------------------------------------------------

        st.markdown(
            "#### 🎚️ Interactive Reranker Metric"
        )

        reranker_metric = st.selectbox(
            "Select reranker metric",
            [
                "Hit@1",
                "MRR",
                "Source Coverage@3",
            ],
            key="reranker_impact_metric",
        )

        reranker_metric_values = {
            "Hit@1": [
                baseline_hit1,
                reranked_hit1,
            ],
            "MRR": [
                baseline_mrr,
                reranked_mrr,
            ],
            "Source Coverage@3": [
                baseline_coverage,
                reranked_coverage,
            ],
        }

        interactive_reranker_chart = pd.DataFrame(
            {
                "Method": [
                    "Baseline",
                    "Reranker",
                ],
                "Score": reranker_metric_values[
                    reranker_metric
                ],
            }
        )

        st.bar_chart(
            interactive_reranker_chart,
            x="Method",
            y="Score",
        )

        # ----------------------------------------------------
        # Absolute difference analysis
        # ----------------------------------------------------

        st.markdown(
            "#### 🔀 Recorded Reranker Differences"
        )

        difference_cols = st.columns(3)

        hit1_difference = (
            reranked_hit1
            - baseline_hit1
        )

        mrr_difference = (
            reranked_mrr
            - baseline_mrr
        )

        coverage_difference = (
            reranked_coverage
            - baseline_coverage
        )

        with difference_cols[0]:

            st.metric(
                "Hit@1 Difference",
                f"{hit1_difference:+.3f}",
            )

        with difference_cols[1]:

            st.metric(
                "MRR Difference",
                f"{mrr_difference:+.3f}",
            )

        with difference_cols[2]:

            st.metric(
                "Coverage@3 Difference",
                f"{coverage_difference:+.3f}",
            )

        # ----------------------------------------------------
        # Relative difference analysis
        # ----------------------------------------------------

        st.markdown(
            "#### 📈 Recorded Relative Differences"
        )

        def reranker_relative_difference(
            reranked_value,
            baseline_value,
        ):

            if (
                baseline_value is None
                or pd.isna(baseline_value)
                or baseline_value == 0
            ):

                return None

            return (
                reranked_value
                - baseline_value
            ) / baseline_value

        hit1_relative = (
            reranker_relative_difference(
                reranked_hit1,
                baseline_hit1,
            )
        )

        mrr_relative = (
            reranker_relative_difference(
                reranked_mrr,
                baseline_mrr,
            )
        )

        coverage_relative = (
            reranker_relative_difference(
                reranked_coverage,
                baseline_coverage,
            )
        )

        relative_reranker_df = pd.DataFrame(
            {
                "Metric": [
                    "Hit@1",
                    "MRR",
                    "Source Coverage@3",
                ],
                "Relative Difference": [
                    hit1_relative,
                    mrr_relative,
                    coverage_relative,
                ],
            }
        )

        st.dataframe(
            relative_reranker_df.style.format(
                {
                    "Relative Difference": (
                        lambda value:
                        "N/A"
                        if pd.isna(value)
                        else f"{value:+.1%}"
                    ),
                }
            ),
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # Evaluation metadata
        # ----------------------------------------------------

        metadata_cols = st.columns(2)

        with metadata_cols[0]:

            st.metric(
                "Recorded Questions",
                len(reranker_df),
            )

        with metadata_cols[1]:

            if "difficulty" in reranker_df.columns:

                st.metric(
                    "Difficulty Levels",
                    reranker_df[
                        "difficulty"
                    ]
                    .dropna()
                    .nunique(),
                )

            else:

                st.metric(
                    "Difficulty Levels",
                    "N/A",
                )

        # ----------------------------------------------------
        # Existing experiment configuration
        # ----------------------------------------------------

        if not reranker_summary_df.empty:

            with st.expander(
                "ℹ️ Reranker Experiment Configuration"
            ):

                st.dataframe(
                    reranker_summary_df,
                    width="stretch",
                    hide_index=True,
                )

        st.caption(
            "Hit@1 measures whether the expected source was "
            "retrieved at the first position. MRR measures "
            "the reciprocal rank of the first relevant result. "
            "Source Coverage@3 measures expected-source "
            "coverage within the top three results. All "
            "differences shown here are calculated from the "
            "recorded experiment and do not trigger a new "
            "reranking run."
        )

    st.divider()
    # ========================================================
    # HYBRID WEIGHT ANALYSIS
    # ========================================================

    st.markdown(
        "### ⚖️ Hybrid Retrieval Weight Analysis"
    )

    st.caption(
        "Analysis of the recorded Dense/BM25 hybrid retrieval "
        "weight configurations and their measured retrieval "
        "metrics."
    )

    weight_required = [
        "dense_weight",
        "bm25_weight",
        "hit_at_1",
        "hit_at_3",
        "precision_at_3",
        "recall_at_3",
        "mrr",
        "coverage_at_3",
        "all_expected_at_3",
    ]

    if (
        weights_df.empty
        or not all(
            column in weights_df.columns
            for column in weight_required
        )
    ):

        render_missing_evaluation_message(
            "hybrid_weight_experiment.csv"
        )

    else:

        weights_analysis = (
            weights_df.copy()
        )

        # ----------------------------------------------------
        # NORMALIZE NUMERIC COLUMNS
        # ----------------------------------------------------

        numeric_weight_columns = [
            "dense_weight",
            "bm25_weight",
            "hit_at_1",
            "hit_at_3",
            "precision_at_3",
            "recall_at_3",
            "mrr",
            "coverage_at_3",
            "all_expected_at_3",
        ]

        for column in numeric_weight_columns:

            weights_analysis[column] = pd.to_numeric(
                weights_analysis[column],
                errors="coerce",
            )

        weights_analysis = (
            weights_analysis
            .dropna(
                subset=[
                    "dense_weight",
                    "bm25_weight",
                ]
            )
            .copy()
        )

        # ----------------------------------------------------
        # CONFIGURATION LABEL
        # ----------------------------------------------------

        weights_analysis[
            "Configuration"
        ] = (
            weights_analysis[
                "dense_weight"
            ].map(
                lambda value:
                f"Dense {value:.1f}"
            )
            + " / "
            + weights_analysis[
                "bm25_weight"
            ].map(
                lambda value:
                f"BM25 {value:.1f}"
            )
        )

        # ----------------------------------------------------
        # CONFIGURATION COUNT
        # ----------------------------------------------------

        st.markdown(
            "#### 🧪 Recorded Weight Configurations"
        )

        weight_configuration_count = (
            len(weights_analysis)
        )

        weight_summary_cols = st.columns(4)

        with weight_summary_cols[0]:

            st.metric(
                "Configurations",
                weight_configuration_count,
            )

        with weight_summary_cols[1]:

            dense_min = (
                weights_analysis[
                    "dense_weight"
                ].min()
            )

            dense_max = (
                weights_analysis[
                    "dense_weight"
                ].max()
            )

            st.metric(
                "Dense Weight Range",
                (
                    f"{dense_min:.1f} → "
                    f"{dense_max:.1f}"
                ),
            )

        with weight_summary_cols[2]:

            bm25_min = (
                weights_analysis[
                    "bm25_weight"
                ].min()
            )

            bm25_max = (
                weights_analysis[
                    "bm25_weight"
                ].max()
            )

            st.metric(
                "BM25 Weight Range",
                (
                    f"{bm25_min:.1f} → "
                    f"{bm25_max:.1f}"
                ),
            )

        with weight_summary_cols[3]:

            current_dense_weight = 0.6
            current_bm25_weight = 0.4

            st.metric(
                "Current AnswerService",
                (
                    f"Dense {current_dense_weight:.1f} / "
                    f"BM25 {current_bm25_weight:.1f}"
                ),
            )

        # ----------------------------------------------------
        # WEIGHT CONFIGURATION TABLE
        # ----------------------------------------------------

        st.markdown(
            "#### 📋 Configuration Results"
        )

        configuration_columns = [
            "Configuration",
            "dense_weight",
            "bm25_weight",
            "hit_at_1",
            "hit_at_3",
            "precision_at_3",
            "recall_at_3",
            "mrr",
            "coverage_at_3",
            "all_expected_at_3",
        ]

        st.dataframe(
            weights_analysis[
                configuration_columns
            ],
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # INTERACTIVE METRIC ANALYSIS
        # ----------------------------------------------------

        st.markdown(
            "#### 📈 Weight Configuration vs Metric"
        )

        weight_metric = st.selectbox(
            "Select retrieval metric",
            [
                "hit_at_1",
                "hit_at_3",
                "precision_at_3",
                "recall_at_3",
                "mrr",
                "coverage_at_3",
                "all_expected_at_3",
            ],
            format_func=(
                lambda value:
                value.replace(
                    "_",
                    " ",
                ).title()
            ),
            key="evaluation_weight_metric",
        )

        weight_metric_chart = (
            weights_analysis[
                [
                    "Configuration",
                    weight_metric,
                ]
            ]
            .copy()
        )

        weight_metric_chart[
            weight_metric
        ] = pd.to_numeric(
            weight_metric_chart[
                weight_metric
            ],
            errors="coerce",
        )

        weight_metric_chart = (
            weight_metric_chart
            .dropna(
                subset=[
                    weight_metric
                ]
            )
        )

        st.line_chart(
            weight_metric_chart,
            x="Configuration",
            y=weight_metric,
        )

        # ----------------------------------------------------
        # DENSE WEIGHT RELATIONSHIP
        # ----------------------------------------------------

        st.markdown(
            "#### 🔵 Dense Weight vs Selected Metric"
        )

        dense_weight_chart = (
            weights_analysis[
                [
                    "dense_weight",
                    weight_metric,
                ]
            ]
            .dropna()
            .sort_values(
                "dense_weight"
            )
            .copy()
        )

        st.line_chart(
            dense_weight_chart,
            x="dense_weight",
            y=weight_metric,
        )

        # ----------------------------------------------------
        # BM25 WEIGHT RELATIONSHIP
        # ----------------------------------------------------

        st.markdown(
            "#### 🟠 BM25 Weight vs Selected Metric"
        )

        bm25_weight_chart = (
            weights_analysis[
                [
                    "bm25_weight",
                    weight_metric,
                ]
            ]
            .dropna()
            .sort_values(
                "bm25_weight"
            )
            .copy()
        )

        st.line_chart(
            bm25_weight_chart,
            x="bm25_weight",
            y=weight_metric,
        )

        # ----------------------------------------------------
        # ALL RETRIEVAL METRICS
        # ----------------------------------------------------

        st.markdown(
            "#### 📊 All Retrieval Metrics by Configuration"
        )

        all_metrics_chart_df = (
            weights_analysis[
                [
                    "Configuration",
                    "hit_at_1",
                    "hit_at_3",
                    "precision_at_3",
                    "recall_at_3",
                    "mrr",
                    "coverage_at_3",
                    "all_expected_at_3",
                ]
            ]
            .set_index(
                "Configuration"
            )
        )

        st.line_chart(
            all_metrics_chart_df
        )

        # ----------------------------------------------------
        # CONFIGURATION DIFFERENCES
        # ----------------------------------------------------

        st.markdown(
            "#### 📐 Recorded Configuration Differences"
        )

        metric_difference_rows = []

        for metric in [
            "hit_at_1",
            "hit_at_3",
            "precision_at_3",
            "recall_at_3",
            "mrr",
            "coverage_at_3",
            "all_expected_at_3",
        ]:

            metric_values = (
                pd.to_numeric(
                    weights_analysis[
                        metric
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            if metric_values.empty:
                continue

            metric_difference_rows.append(
                {
                    "Metric": (
                        metric
                        .replace(
                            "_",
                            " ",
                        )
                        .title()
                    ),
                    "Minimum Recorded": (
                        metric_values.min()
                    ),
                    "Maximum Recorded": (
                        metric_values.max()
                    ),
                    "Range": (
                        metric_values.max()
                        - metric_values.min()
                    ),
                }
            )

        if metric_difference_rows:

            weight_difference_df = pd.DataFrame(
                metric_difference_rows
            )

            st.dataframe(
                weight_difference_df,
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # CURRENT CONFIGURATION LOOKUP
        # ----------------------------------------------------

        st.markdown(
            "#### 🔎 Current AnswerService Configuration"
        )

        current_configuration = (
            weights_analysis[
                (
                    weights_analysis[
                        "dense_weight"
                    ].sub(
                        current_dense_weight
                    ).abs()
                    < 1e-9
                )
                & (
                    weights_analysis[
                        "bm25_weight"
                    ].sub(
                        current_bm25_weight
                    ).abs()
                    < 1e-9
                )
            ]
        )

        if not current_configuration.empty:

            st.dataframe(
                current_configuration[
                    [
                        "Configuration",
                        "hit_at_1",
                        "hit_at_3",
                        "precision_at_3",
                        "recall_at_3",
                        "mrr",
                        "coverage_at_3",
                        "all_expected_at_3",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        else:

            st.info(
                "The current AnswerService weight configuration "
                "(Dense 0.6 / BM25 0.4) is not present in the "
                "recorded experiment configurations."
            )

        # ----------------------------------------------------
        # WEIGHT EXPERIMENT METADATA
        # ----------------------------------------------------

        weight_metadata_cols = st.columns(3)

        with weight_metadata_cols[0]:

            st.metric(
                "Recorded Rows",
                len(weights_analysis),
            )

        with weight_metadata_cols[1]:

            if "difficulty" in weights_analysis.columns:

                weight_difficulties = (
                    weights_analysis[
                        "difficulty"
                    ]
                    .dropna()
                    .astype(str)
                    .nunique()
                )

                st.metric(
                    "Difficulty Levels",
                    weight_difficulties,
                )

            else:

                st.metric(
                    "Difficulty Levels",
                    "N/A",
                )

        with weight_metadata_cols[2]:

            if "question" in weights_analysis.columns:

                weight_questions = (
                    weights_analysis[
                        "question"
                    ]
                    .dropna()
                    .astype(str)
                    .nunique()
                )

                st.metric(
                    "Unique Questions",
                    weight_questions,
                )

            else:

                st.metric(
                    "Unique Questions",
                    "N/A",
                )

        st.caption(
            "All values shown here come from the existing "
            "hybrid weight experiment CSV. Refreshing the "
            "dashboard does not rerun retrieval, embeddings, "
            "or model generation."
        )

    st.divider()
    # ========================================================
    # DISTANCE THRESHOLD
    # ========================================================

    st.markdown(
        "### 📏 Retrieval Distance Threshold Analysis"
    )

    st.caption(
        "Analysis of the recorded retrieval distance "
        "threshold experiment and its effect on retrieval "
        "quality and result counts."
    )

    distance_required = [
        "threshold",
        "hit_rate",
        "precision",
        "recall",
        "average_results",
    ]

    if (
        distance_df.empty
        or not all(
            column in distance_df.columns
            for column in distance_required
        )
    ):

        render_missing_evaluation_message(
            "retrieval_distance_experiment.csv"
        )

    else:

        distance_analysis = (
            distance_df.copy()
        )

        # ----------------------------------------------------
        # NORMALIZE NUMERIC COLUMNS
        # ----------------------------------------------------

        distance_numeric_columns = [
            "threshold",
            "hit_rate",
            "precision",
            "recall",
            "average_results",
        ]

        for column in distance_numeric_columns:

            distance_analysis[column] = pd.to_numeric(
                distance_analysis[column],
                errors="coerce",
            )

        distance_analysis = (
            distance_analysis
            .dropna(
                subset=[
                    "threshold"
                ]
            )
            .sort_values(
                "threshold"
            )
            .copy()
        )

        # ----------------------------------------------------
        # EXPERIMENT OVERVIEW
        # ----------------------------------------------------

        st.markdown(
            "#### 🧪 Experiment Overview"
        )

        distance_overview_cols = st.columns(4)

        with distance_overview_cols[0]:

            st.metric(
                "Threshold Configurations",
                len(distance_analysis),
            )

        with distance_overview_cols[1]:

            threshold_min = (
                distance_analysis[
                    "threshold"
                ].min()
            )

            threshold_max = (
                distance_analysis[
                    "threshold"
                ].max()
            )

            st.metric(
                "Threshold Range",
                (
                    f"{threshold_min:.3f} → "
                    f"{threshold_max:.3f}"
                ),
            )

        with distance_overview_cols[2]:

            average_result_mean = (
                distance_analysis[
                    "average_results"
                ].mean()
            )

            st.metric(
                "Average Results",
                (
                    f"{average_result_mean:.2f}"
                ),
            )

        with distance_overview_cols[3]:

            hit_rate_mean = (
                distance_analysis[
                    "hit_rate"
                ].mean()
            )

            st.metric(
                "Average Hit Rate",
                (
                    f"{hit_rate_mean:.1%}"
                ),
            )

        # ----------------------------------------------------
        # INTERACTIVE METRIC ANALYSIS
        # ----------------------------------------------------

        st.markdown(
            "#### 📈 Threshold vs Retrieval Metric"
        )

        distance_chart_metric = st.selectbox(
            "Select distance experiment metric",
            [
                "hit_rate",
                "precision",
                "recall",
                "average_results",
            ],
            format_func=(
                lambda value:
                value.replace(
                    "_",
                    " ",
                ).title()
            ),
            key="evaluation_distance_metric",
        )

        distance_plot_df = (
            distance_analysis[
                [
                    "threshold",
                    distance_chart_metric,
                ]
            ]
            .dropna()
            .sort_values(
                "threshold"
            )
            .copy()
        )

        st.line_chart(
            distance_plot_df,
            x="threshold",
            y=distance_chart_metric,
        )

        # ----------------------------------------------------
        # INDIVIDUAL QUALITY METRIC CHARTS
        # ----------------------------------------------------

        st.markdown(
            "#### 📊 Retrieval Quality Across Thresholds"
        )

        distance_hit_chart = (
            distance_analysis[
                [
                    "threshold",
                    "hit_rate",
                ]
            ]
            .dropna()
            .copy()
        )

        st.markdown(
            "**Hit Rate**"
        )

        st.line_chart(
            distance_hit_chart,
            x="threshold",
            y="hit_rate",
        )

        distance_precision_chart = (
            distance_analysis[
                [
                    "threshold",
                    "precision",
                ]
            ]
            .dropna()
            .copy()
        )

        st.markdown(
            "**Precision**"
        )

        st.line_chart(
            distance_precision_chart,
            x="threshold",
            y="precision",
        )

        distance_recall_chart = (
            distance_analysis[
                [
                    "threshold",
                    "recall",
                ]
            ]
            .dropna()
            .copy()
        )

        st.markdown(
            "**Recall**"
        )

        st.line_chart(
            distance_recall_chart,
            x="threshold",
            y="recall",
        )

        distance_results_chart = (
            distance_analysis[
                [
                    "threshold",
                    "average_results",
                ]
            ]
            .dropna()
            .copy()
        )

        st.markdown(
            "**Average Retrieved Results**"
        )

        st.line_chart(
            distance_results_chart,
            x="threshold",
            y="average_results",
        )

        # ----------------------------------------------------
        # QUALITY METRIC SUMMARY
        # ----------------------------------------------------

        st.markdown(
            "#### 📋 Threshold Experiment Summary"
        )

        distance_summary_df = (
            distance_analysis[
                [
                    "threshold",
                    "hit_rate",
                    "precision",
                    "recall",
                    "average_results",
                ]
            ]
            .copy()
        )

        distance_summary_df.columns = [
            "Threshold",
            "Hit Rate",
            "Precision",
            "Recall",
            "Average Results",
        ]

        st.dataframe(
            distance_summary_df,
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # METRIC RANGES
        # ----------------------------------------------------

        st.markdown(
            "#### 📐 Recorded Metric Ranges"
        )

        distance_range_rows = []

        for metric in [
            "hit_rate",
            "precision",
            "recall",
            "average_results",
        ]:

            metric_values = (
                pd.to_numeric(
                    distance_analysis[
                        metric
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            if metric_values.empty:
                continue

            distance_range_rows.append(
                {
                    "Metric": (
                        metric
                        .replace(
                            "_",
                            " ",
                        )
                        .title()
                    ),
                    "Minimum": (
                        metric_values.min()
                    ),
                    "Maximum": (
                        metric_values.max()
                    ),
                    "Range": (
                        metric_values.max()
                        - metric_values.min()
                    ),
                    "Mean": (
                        metric_values.mean()
                    ),
                }
            )

        if distance_range_rows:

            distance_range_df = pd.DataFrame(
                distance_range_rows
            )

            st.dataframe(
                distance_range_df,
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # THRESHOLD CONFIGURATION DETAILS
        # ----------------------------------------------------

        st.markdown(
            "#### 🔎 Recorded Threshold Configurations"
        )

        st.dataframe(
            distance_analysis,
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # EXPERIMENT INTERPRETATION
        # ----------------------------------------------------

        st.markdown(
            "#### 📝 Experiment Interpretation"
        )

        st.info(
            "A retrieval distance threshold controls which "
            "vector matches are accepted by the retrieval "
            "stage. Changing the threshold can affect both "
            "retrieval quality metrics and the number of "
            "returned results. The charts above describe the "
            "recorded experiment only; they do not rerun "
            "retrieval or make assumptions about unseen "
            "queries."
        )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        distance_metadata_cols = st.columns(3)

        with distance_metadata_cols[0]:

            st.metric(
                "Recorded Configurations",
                len(distance_analysis),
            )

        with distance_metadata_cols[1]:

            if "question" in distance_analysis.columns:

                distance_questions = (
                    distance_analysis[
                        "question"
                    ]
                    .dropna()
                    .astype(str)
                    .nunique()
                )

                st.metric(
                    "Unique Questions",
                    distance_questions,
                )

            else:

                st.metric(
                    "Unique Questions",
                    "N/A",
                )

        with distance_metadata_cols[2]:

            if "difficulty" in distance_analysis.columns:

                distance_difficulties = (
                    distance_analysis[
                        "difficulty"
                    ]
                    .dropna()
                    .astype(str)
                    .nunique()
                )

                st.metric(
                    "Difficulty Levels",
                    distance_difficulties,
                )

            else:

                st.metric(
                    "Difficulty Levels",
                    "N/A",
                )

        st.caption(
            "All values are calculated from the existing "
            "retrieval_distance_experiment.csv file. "
            "Refreshing this dashboard does not rerun "
            "retrieval, embedding, or generation experiments."
        )

    st.divider()
    # ========================================================
    # CHUNKING ANALYSIS
    # ========================================================

    st.markdown(
        "### 🧩 Chunking Experiment Analysis"
    )

    st.caption(
        "Analysis of the recorded chunk-size and chunk-overlap "
        "experiments and their effect on knowledge-base "
        "structure and text fragmentation."
    )

    chunk_required = [
        "chunk_size",
        "chunk_overlap",
        "chunk_count",
        "average_length",
        "minimum_length",
        "maximum_length",
        "total_characters",
    ]

    if (
        chunking_df.empty
        or not all(
            column in chunking_df.columns
            for column in chunk_required
        )
    ):

        render_missing_evaluation_message(
            "chunking_experiment_results.csv"
        )

    else:

        chunk_analysis = (
            chunking_df.copy()
        )

        # ----------------------------------------------------
        # NORMALIZE NUMERIC COLUMNS
        # ----------------------------------------------------

        chunk_numeric_columns = [
            "chunk_size",
            "chunk_overlap",
            "chunk_count",
            "average_length",
            "minimum_length",
            "maximum_length",
            "total_characters",
        ]

        for column in chunk_numeric_columns:

            chunk_analysis[column] = pd.to_numeric(
                chunk_analysis[column],
                errors="coerce",
            )

        chunk_analysis = (
            chunk_analysis
            .dropna(
                subset=[
                    "chunk_size",
                    "chunk_overlap",
                ]
            )
            .sort_values(
                [
                    "chunk_size",
                    "chunk_overlap",
                ]
            )
            .copy()
        )

        # ----------------------------------------------------
        # EXPERIMENT OVERVIEW
        # ----------------------------------------------------

        st.markdown(
            "#### 🧪 Experiment Overview"
        )

        chunk_overview_cols = st.columns(5)

        with chunk_overview_cols[0]:

            st.metric(
                "Configurations",
                len(chunk_analysis),
            )

        with chunk_overview_cols[1]:

            unique_chunk_sizes = (
                chunk_analysis[
                    "chunk_size"
                ]
                .nunique()
            )

            st.metric(
                "Chunk Sizes",
                unique_chunk_sizes,
            )

        with chunk_overview_cols[2]:

            unique_overlaps = (
                chunk_analysis[
                    "chunk_overlap"
                ]
                .nunique()
            )

            st.metric(
                "Overlap Values",
                unique_overlaps,
            )

        with chunk_overview_cols[3]:

            total_chunk_count = (
                chunk_analysis[
                    "chunk_count"
                ].sum()
            )

            st.metric(
                "Total Chunks",
                f"{total_chunk_count:,.0f}",
            )

        with chunk_overview_cols[4]:

            mean_chunk_length = (
                chunk_analysis[
                    "average_length"
                ].mean()
            )

            st.metric(
                "Mean Chunk Length",
                f"{mean_chunk_length:,.1f}",
            )

        # ----------------------------------------------------
        # CHUNK SIZE RANGE
        # ----------------------------------------------------

        chunk_range_cols = st.columns(3)

        with chunk_range_cols[0]:

            st.metric(
                "Chunk Size Range",
                (
                    f"{chunk_analysis['chunk_size'].min():,.0f}"
                    f" → "
                    f"{chunk_analysis['chunk_size'].max():,.0f}"
                ),
            )

        with chunk_range_cols[1]:

            st.metric(
                "Overlap Range",
                (
                    f"{chunk_analysis['chunk_overlap'].min():,.0f}"
                    f" → "
                    f"{chunk_analysis['chunk_overlap'].max():,.0f}"
                ),
            )

        with chunk_range_cols[2]:

            st.metric(
                "Total Characters",
                (
                    f"{chunk_analysis['total_characters'].sum():,.0f}"
                ),
            )

        # ----------------------------------------------------
        # INTERACTIVE CHUNK SIZE / OVERLAP ANALYSIS
        # ----------------------------------------------------

        st.markdown(
            "#### 📈 Chunk Size and Overlap Analysis"
        )

        chunk_selector_cols = st.columns(2)

        with chunk_selector_cols[0]:

            chunk_sizes = sorted(
                chunk_analysis[
                    "chunk_size"
                ]
                .dropna()
                .unique()
            )

            selected_chunk_size = st.selectbox(
                "Select chunk size",
                chunk_sizes,
                key="evaluation_chunk_size",
            )

        with chunk_selector_cols[1]:

            chunk_metric = st.selectbox(
                "Select chunking metric",
                [
                    "chunk_count",
                    "average_length",
                    "minimum_length",
                    "maximum_length",
                    "total_characters",
                ],
                format_func=(
                    lambda value:
                    value.replace(
                        "_",
                        " ",
                    ).title()
                ),
                key="evaluation_chunk_metric",
            )

        selected_chunk_df = (
            chunk_analysis[
                chunk_analysis[
                    "chunk_size"
                ] == selected_chunk_size
            ]
            .copy()
            .sort_values(
                "chunk_overlap"
            )
        )

        if not selected_chunk_df.empty:

            selected_chunk_plot = (
                selected_chunk_df[
                    [
                        "chunk_overlap",
                        chunk_metric,
                    ]
                ]
                .dropna()
                .copy()
            )

            st.line_chart(
                selected_chunk_plot,
                x="chunk_overlap",
                y=chunk_metric,
            )

        # ----------------------------------------------------
        # CHUNK COUNT BY CHUNK SIZE
        # ----------------------------------------------------

        st.markdown(
            "#### 📦 Chunk Count by Chunk Size"
        )

        chunk_count_by_size = (
            chunk_analysis
            .groupby(
                "chunk_size",
                as_index=False,
            )[
                "chunk_count"
            ]
            .mean()
            .sort_values(
                "chunk_size"
            )
        )

        st.line_chart(
            chunk_count_by_size,
            x="chunk_size",
            y="chunk_count",
        )

        # ----------------------------------------------------
        # AVERAGE CHUNK LENGTH
        # ----------------------------------------------------

        st.markdown(
            "#### 📏 Average Chunk Length"
        )

        average_length_by_size = (
            chunk_analysis[
                [
                    "chunk_size",
                    "average_length",
                ]
            ]
            .dropna()
            .groupby(
                "chunk_size",
                as_index=False,
            )[
                "average_length"
            ]
            .mean()
            .sort_values(
                "chunk_size"
            )
        )

        st.line_chart(
            average_length_by_size,
            x="chunk_size",
            y="average_length",
        )

        # ----------------------------------------------------
        # TOTAL CHARACTERS
        # ----------------------------------------------------

        st.markdown(
            "#### 🔤 Total Characters by Configuration"
        )

        total_characters_chart = (
            chunk_analysis[
                [
                    "chunk_size",
                    "chunk_overlap",
                    "total_characters",
                ]
            ]
            .copy()
        )

        total_characters_chart[
            "Configuration"
        ] = (
            total_characters_chart[
                "chunk_size"
            ].map(
                lambda value:
                f"Size {value:.0f}"
            )
            + " / "
            + total_characters_chart[
                "chunk_overlap"
            ].map(
                lambda value:
                f"Overlap {value:.0f}"
            )
        )

        total_characters_chart = (
            total_characters_chart[
                [
                    "Configuration",
                    "total_characters",
                ]
            ]
            .dropna()
        )

        st.bar_chart(
            total_characters_chart,
            x="Configuration",
            y="total_characters",
        )

        # ----------------------------------------------------
        # CHUNK SIZE vs OVERLAP TABLE
        # ----------------------------------------------------

        st.markdown(
            "#### 🗂️ Chunking Configuration Matrix"
        )

        chunk_matrix = (
            chunk_analysis[
                [
                    "chunk_size",
                    "chunk_overlap",
                    "chunk_count",
                    "average_length",
                    "minimum_length",
                    "maximum_length",
                    "total_characters",
                ]
            ]
            .copy()
        )

        chunk_matrix.columns = [
            "Chunk Size",
            "Chunk Overlap",
            "Chunk Count",
            "Average Length",
            "Minimum Length",
            "Maximum Length",
            "Total Characters",
        ]

        st.dataframe(
            chunk_matrix,
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # METRIC RANGE ANALYSIS
        # ----------------------------------------------------

        st.markdown(
            "#### 📐 Recorded Chunking Metric Ranges"
        )

        chunk_range_rows = []

        for metric in [
            "chunk_count",
            "average_length",
            "minimum_length",
            "maximum_length",
            "total_characters",
        ]:

            metric_values = (
                pd.to_numeric(
                    chunk_analysis[
                        metric
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            if metric_values.empty:
                continue

            chunk_range_rows.append(
                {
                    "Metric": (
                        metric
                        .replace(
                            "_",
                            " ",
                        )
                        .title()
                    ),
                    "Minimum": (
                        metric_values.min()
                    ),
                    "Maximum": (
                        metric_values.max()
                    ),
                    "Range": (
                        metric_values.max()
                        - metric_values.min()
                    ),
                    "Mean": (
                        metric_values.mean()
                    ),
                }
            )

        if chunk_range_rows:

            chunk_range_df = pd.DataFrame(
                chunk_range_rows
            )

            st.dataframe(
                chunk_range_df,
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # CHUNKING CONFIGURATION DETAILS
        # ----------------------------------------------------

        st.markdown(
            "#### 🔎 Recorded Chunking Configurations"
        )

        st.dataframe(
            chunk_analysis,
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # CHUNKING INTERPRETATION
        # ----------------------------------------------------

        st.markdown(
            "#### 📝 Experiment Interpretation"
        )

        st.info(
            "Chunk size determines the approximate amount of "
            "text placed into each knowledge-base chunk, while "
            "chunk overlap controls how much neighboring text "
            "is repeated between chunks. Smaller chunks can "
            "increase the number of chunks, while larger "
            "chunks can increase the amount of text contained "
            "in each chunk. The recorded metrics above describe "
            "the stored experiment results only."
        )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        chunk_metadata_cols = st.columns(3)

        with chunk_metadata_cols[0]:

            st.metric(
                "Recorded Configurations",
                len(chunk_analysis),
            )

        with chunk_metadata_cols[1]:

            st.metric(
                "Unique Chunk Sizes",
                (
                    chunk_analysis[
                        "chunk_size"
                    ]
                    .nunique()
                ),
            )

        with chunk_metadata_cols[2]:

            st.metric(
                "Unique Overlap Values",
                (
                    chunk_analysis[
                        "chunk_overlap"
                    ]
                    .nunique()
                ),
            )

        st.caption(
            "All values are calculated from the existing "
            "chunking_experiment_results.csv file. "
            "Refreshing the dashboard does not rerun document "
            "ingestion, chunking, embedding, or retrieval."
        )

    st.divider()

    # ========================================================
    # DIFFICULTY ANALYSIS
    # ========================================================

    st.markdown(
        "### 🧠 Retrieval Performance by Question Difficulty"
    )

    difficulty_required = [
        "difficulty",
        "id",
        "dense_hit_at_3",
        "bm25_hit_at_3",
        "hybrid_hit_at_3",
        "dense_mrr",
        "bm25_mrr",
        "hybrid_mrr",
    ]

    if (
        hybrid_df.empty
        or not all(
            column in hybrid_df.columns
            for column in difficulty_required
        )
    ):

        render_missing_evaluation_message(
            "hybrid_retrieval_experiment.csv"
        )

    else:

        difficulty_df = (
            hybrid_df
            .groupby(
                "difficulty"
            )
            .agg(
                Questions=(
                    "id",
                    "count",
                ),
                Dense_Hit_at_3=(
                    "dense_hit_at_3",
                    "mean",
                ),
                BM25_Hit_at_3=(
                    "bm25_hit_at_3",
                    "mean",
                ),
                Hybrid_Hit_at_3=(
                    "hybrid_hit_at_3",
                    "mean",
                ),
                Dense_MRR=(
                    "dense_mrr",
                    "mean",
                ),
                BM25_MRR=(
                    "bm25_mrr",
                    "mean",
                ),
                Hybrid_MRR=(
                    "hybrid_mrr",
                    "mean",
                ),
            )
        )

        st.dataframe(
            difficulty_df.style.format(
                {
                    "Dense_Hit_at_3": "{:.1%}",
                    "BM25_Hit_at_3": "{:.1%}",
                    "Hybrid_Hit_at_3": "{:.1%}",
                    "Dense_MRR": "{:.3f}",
                    "BM25_MRR": "{:.3f}",
                    "Hybrid_MRR": "{:.3f}",
                }
            ),
            width="stretch",
        )

        difficulty_metric = st.selectbox(
            "Difficulty chart metric",
            [
                "Dense_Hit_at_3",
                "BM25_Hit_at_3",
                "Hybrid_Hit_at_3",
                "Dense_MRR",
                "BM25_MRR",
                "Hybrid_MRR",
            ],
            format_func=(
                lambda value:
                value.replace(
                    "_",
                    " ",
                )
            ),
            key="evaluation_difficulty_metric",
        )

        difficulty_chart_df = (
            difficulty_df[
                [difficulty_metric]
            ]
        )

        st.bar_chart(
            difficulty_chart_df
        )

    st.divider()

    # ========================================================
    # CATEGORY ANALYSIS
    # ========================================================

    st.markdown(
        "### 📚 Retrieval Performance by Knowledge Category"
    )

    category_required = [
        "category",
        "id",
        "hybrid_hit_at_3",
        "hybrid_precision_at_3",
        "hybrid_recall_at_3",
        "hybrid_mrr",
        "hybrid_coverage_at_3",
    ]

    if (
        hybrid_df.empty
        or not all(
            column in hybrid_df.columns
            for column in category_required
        )
    ):

        render_missing_evaluation_message(
            "hybrid_retrieval_experiment.csv"
        )

    else:

        category_df = (
            hybrid_df
            .groupby(
                "category"
            )
            .agg(
                Questions=(
                    "id",
                    "count",
                ),
                Hybrid_Hit_at_3=(
                    "hybrid_hit_at_3",
                    "mean",
                ),
                Hybrid_Precision_at_3=(
                    "hybrid_precision_at_3",
                    "mean",
                ),
                Hybrid_Recall_at_3=(
                    "hybrid_recall_at_3",
                    "mean",
                ),
                Hybrid_MRR=(
                    "hybrid_mrr",
                    "mean",
                ),
                Hybrid_Coverage_at_3=(
                    "hybrid_coverage_at_3",
                    "mean",
                ),
            )
        )

        st.dataframe(
            category_df.style.format(
                {
                    "Hybrid_Hit_at_3": "{:.1%}",
                    "Hybrid_Precision_at_3": "{:.1%}",
                    "Hybrid_Recall_at_3": "{:.1%}",
                    "Hybrid_MRR": "{:.3f}",
                    "Hybrid_Coverage_at_3": "{:.1%}",
                }
            ),
            width="stretch",
        )

        category_chart_metric = st.selectbox(
            "Category chart metric",
            [
                "Hybrid_Hit_at_3",
                "Hybrid_Precision_at_3",
                "Hybrid_Recall_at_3",
                "Hybrid_MRR",
                "Hybrid_Coverage_at_3",
            ],
            format_func=(
                lambda value:
                value.replace(
                    "_",
                    " ",
                )
            ),
            key="evaluation_category_metric",
        )

        st.bar_chart(
            category_df[
                [category_chart_metric]
            ]
        )

    st.divider()

    # ========================================================
    # DETAILED RERANKER RESULTS
    # ========================================================

    st.markdown(
        "### 🔎 Detailed Reranker Evaluation"
    )

    if reranker_df.empty:

        render_missing_evaluation_message(
            "reranker_metrics_experiment.csv"
        )

    else:

        display_reranker = (
            reranker_df.copy()
        )

        columns_to_show = [
            "id",
            "difficulty",
            "category",
            "baseline_hit_at_1",
            "baseline_hit_at_3",
            "reranker_hit_at_1",
            "reranker_hit_at_3",
            "baseline_mrr",
            "reranker_mrr",
            "rank_change",
        ]

        available = available_columns(
            display_reranker,
            columns_to_show,
        )

        if available:

            st.dataframe(
                display_reranker[
                    available
                ],
                width="stretch",
                hide_index=True,
            )

    # ========================================================
    # DETAILED HYBRID RESULTS
    # ========================================================

    st.markdown(
        "### 🔬 Detailed Hybrid Retrieval Evaluation"
    )

    if hybrid_df.empty:

        render_missing_evaluation_message(
            "hybrid_retrieval_experiment.csv"
        )

    else:

        hybrid_columns = [
            "id",
            "difficulty",
            "category",
            "dense_hit_at_3",
            "bm25_hit_at_3",
            "hybrid_hit_at_3",
            "dense_precision_at_3",
            "bm25_precision_at_3",
            "hybrid_precision_at_3",
            "dense_recall_at_3",
            "bm25_recall_at_3",
            "hybrid_recall_at_3",
            "dense_mrr",
            "bm25_mrr",
            "hybrid_mrr",
        ]

        available_hybrid_columns = (
            available_columns(
                hybrid_df,
                hybrid_columns,
            )
        )

        if available_hybrid_columns:

            st.dataframe(
                hybrid_df[
                    available_hybrid_columns
                ],
                width="stretch",
                hide_index=True,
            )

    # ========================================================
    # RAW DATA
    # ========================================================

    st.markdown(
        "### 📋 Raw Experiment Data"
    )

    raw_experiment = st.selectbox(
        "Select experiment",
        [
            "Hybrid Retrieval",
            "Hybrid Weights",
            "Reranker",
            "Distance Threshold",
            "Chunking",
        ],
        key="evaluation_raw_experiment",
    )

    raw_map = {
        "Hybrid Retrieval": (
            hybrid_df,
            "hybrid_retrieval_experiment.csv",
        ),
        "Hybrid Weights": (
            weights_df,
            "hybrid_weight_experiment.csv",
        ),
        "Reranker": (
            reranker_df,
            "reranker_metrics_experiment.csv",
        ),
        "Distance Threshold": (
            distance_df,
            "retrieval_distance_experiment.csv",
        ),
        "Chunking": (
            chunking_df,
            "chunking_experiment_results.csv",
        ),
    }

    raw_df, raw_filename = raw_map[
        raw_experiment
    ]

    if raw_df.empty:

        render_missing_evaluation_message(
            raw_filename
        )

    else:

        st.dataframe(
            raw_df,
            width="stretch",
            hide_index=True,
        )



    # ========================================================
    # STEP 7.8 — EVALUATION DATASET COVERAGE
    # ========================================================

    st.markdown(
        "### 🗂️ Evaluation Dataset Coverage & Quality"
    )

    st.caption(
        "Analysis of the recorded evaluation question set, "
        "including difficulty, category, expected-source "
        "coverage, completeness, and duplicate checks."
    )

    if hybrid_df.empty:

        render_missing_evaluation_message(
            "hybrid_retrieval_experiment.csv"
        )

    else:

        coverage_analysis = (
            hybrid_df.copy()
        )

        # ----------------------------------------------------
        # DATASET OVERVIEW
        # ----------------------------------------------------

        st.markdown(
            "#### 📊 Dataset Overview"
        )

        total_questions = (
            len(coverage_analysis)
        )

        unique_questions = 0

        if "question" in coverage_analysis.columns:

            unique_questions = (
                coverage_analysis[
                    "question"
                ]
                .dropna()
                .astype(str)
                .nunique()
            )

        duplicate_questions = (
            total_questions
            - unique_questions
        )

        difficulty_count = 0

        if "difficulty" in coverage_analysis.columns:

            difficulty_count = (
                coverage_analysis[
                    "difficulty"
                ]
                .dropna()
                .astype(str)
                .nunique()
            )

        category_count = 0

        if "category" in coverage_analysis.columns:

            category_count = (
                coverage_analysis[
                    "category"
                ]
                .dropna()
                .astype(str)
                .nunique()
            )

        overview_cols = st.columns(5)

        with overview_cols[0]:

            st.metric(
                "Evaluation Questions",
                total_questions,
            )

        with overview_cols[1]:

            st.metric(
                "Unique Questions",
                unique_questions,
            )

        with overview_cols[2]:

            st.metric(
                "Difficulty Levels",
                difficulty_count,
            )

        with overview_cols[3]:

            st.metric(
                "Categories",
                category_count,
            )

        with overview_cols[4]:

            st.metric(
                "Duplicate Questions",
                duplicate_questions,
            )

        # ----------------------------------------------------
        # DIFFICULTY DISTRIBUTION
        # ----------------------------------------------------

        if "difficulty" in coverage_analysis.columns:

            st.markdown(
                "#### 🎯 Questions by Difficulty"
            )

            difficulty_distribution = (
                coverage_analysis[
                    "difficulty"
                ]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis(
                    "Difficulty"
                )
                .reset_index(
                    name="Questions"
                )
            )

            st.bar_chart(
                difficulty_distribution,
                x="Difficulty",
                y="Questions",
            )

            st.dataframe(
                difficulty_distribution,
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # CATEGORY DISTRIBUTION
        # ----------------------------------------------------

        if "category" in coverage_analysis.columns:

            st.markdown(
                "#### 📚 Questions by Knowledge Category"
            )

            category_distribution = (
                coverage_analysis[
                    "category"
                ]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis(
                    "Category"
                )
                .reset_index(
                    name="Questions"
                )
            )

            st.bar_chart(
                category_distribution,
                x="Category",
                y="Questions",
            )

            st.dataframe(
                category_distribution,
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # DIFFICULTY × CATEGORY COVERAGE
        # ----------------------------------------------------

        if (
            "difficulty" in coverage_analysis.columns
            and "category" in coverage_analysis.columns
        ):

            st.markdown(
                "#### 🧩 Difficulty × Category Coverage"
            )

            difficulty_category_matrix = (
                pd.crosstab(
                    coverage_analysis[
                        "difficulty"
                    ]
                    .fillna("Unknown")
                    .astype(str),
                    coverage_analysis[
                        "category"
                    ]
                    .fillna("Unknown")
                    .astype(str),
                )
            )

            st.dataframe(
                difficulty_category_matrix,
                width="stretch",
            )

            matrix_chart = (
                difficulty_category_matrix
                .reset_index()
                .melt(
                    id_vars=[
                        "difficulty"
                    ],
                    var_name="category",
                    value_name="questions",
                )
            )

            st.bar_chart(
                matrix_chart,
                x="category",
                y="questions",
            )

        # ----------------------------------------------------
        # EXPECTED SOURCE COVERAGE
        # ----------------------------------------------------

        if "expected_sources" in coverage_analysis.columns:

            st.markdown(
                "#### 📚 Expected Source Coverage"
            )

            expected_source_series = (
                coverage_analysis[
                    "expected_sources"
                ]
            )

            expected_source_available = (
                expected_source_series
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
            )

            expected_source_count = (
                int(
                    expected_source_available.sum()
                )
            )

            expected_source_missing = (
                total_questions
                - expected_source_count
            )

            source_coverage_cols = (
                st.columns(3)
            )

            with source_coverage_cols[0]:

                st.metric(
                    "Questions With Expected Sources",
                    expected_source_count,
                )

            with source_coverage_cols[1]:

                st.metric(
                    "Questions Without Expected Sources",
                    expected_source_missing,
                )

            with source_coverage_cols[2]:

                source_coverage_rate = (
                    (
                        expected_source_count
                        / total_questions
                    )
                    if total_questions
                    else 0
                )

                st.metric(
                    "Expected Source Coverage",
                    f"{source_coverage_rate:.1%}",
                )

            source_coverage_chart = pd.DataFrame(
                {
                    "Status": [
                        "Available",
                        "Missing",
                    ],
                    "Questions": [
                        expected_source_count,
                        expected_source_missing,
                    ],
                }
            )

            st.bar_chart(
                source_coverage_chart,
                x="Status",
                y="Questions",
            )

        # ----------------------------------------------------
        # DATA COMPLETENESS
        # ----------------------------------------------------

        st.markdown(
            "#### 🔍 Dataset Completeness"
        )

        completeness_rows = []

        for column in coverage_analysis.columns:

            missing_count = int(
                coverage_analysis[
                    column
                ]
                .isna()
                .sum()
            )

            empty_count = 0

            if (
                coverage_analysis[
                    column
                ].dtype == "object"
            ):

                empty_count = int(
                    coverage_analysis[
                        column
                    ]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq("")
                    .sum()
                )

            total_missing = (
                missing_count
                + empty_count
            )

            completeness_rows.append(
                {
                    "Column": column,
                    "Missing / Empty": total_missing,
                    "Completeness": (
                        (
                            total_questions
                            - total_missing
                        )
                        / total_questions
                        if total_questions
                        else 0
                    ),
                }
            )

        completeness_df = (
            pd.DataFrame(
                completeness_rows
            )
            .sort_values(
                "Completeness"
            )
        )

        st.dataframe(
            completeness_df.style.format(
                {
                    "Completeness": "{:.1%}"
                }
            ),
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # DUPLICATE QUESTIONS
        # ----------------------------------------------------

        if "question" in coverage_analysis.columns:

            st.markdown(
                "#### 🔁 Duplicate Question Check"
            )

            normalized_questions = (
                coverage_analysis[
                    "question"
                ]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )

            duplicate_mask = (
                normalized_questions
                .duplicated(
                    keep=False
                )
            )

            duplicate_count = int(
                duplicate_mask.sum()
            )

            if duplicate_count == 0:

                st.success(
                    "No duplicate evaluation questions "
                    "were detected in the recorded dataset."
                )

            else:

                st.warning(
                    f"{duplicate_count} evaluation rows "
                    "belong to duplicated question text."
                )

                duplicate_questions_df = (
                    coverage_analysis[
                        duplicate_mask
                    ]
                    .copy()
                )

                available_duplicate_columns = (
                    available_columns(
                        duplicate_questions_df,
                        [
                            "id",
                            "question",
                            "difficulty",
                            "category",
                            "expected_sources",
                        ],
                    )
                )

                if available_duplicate_columns:

                    st.dataframe(
                        duplicate_questions_df[
                            available_duplicate_columns
                        ],
                        width="stretch",
                        hide_index=True,
                    )

        # ----------------------------------------------------
        # EVALUATION DATASET PREVIEW
        # ----------------------------------------------------

        st.markdown(
            "#### 📝 Evaluation Dataset Preview"
        )

        preview_columns = [
            "id",
            "question",
            "difficulty",
            "category",
            "expected_sources",
        ]

        available_preview_columns = (
            available_columns(
                coverage_analysis,
                preview_columns,
            )
        )

        if available_preview_columns:

            st.dataframe(
                coverage_analysis[
                    available_preview_columns
                ],
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # COVERAGE SUMMARY
        # ----------------------------------------------------

        st.markdown(
            "#### 📌 Coverage Summary"
        )

        coverage_summary_rows = [
            {
                "Dataset Property": (
                    "Total Evaluation Rows"
                ),
                "Value": total_questions,
            },
            {
                "Dataset Property": (
                    "Unique Questions"
                ),
                "Value": unique_questions,
            },
            {
                "Dataset Property": (
                    "Difficulty Levels"
                ),
                "Value": difficulty_count,
            },
            {
                "Dataset Property": (
                    "Knowledge Categories"
                ),
                "Value": category_count,
            },
            {
                "Dataset Property": (
                    "Duplicate Questions"
                ),
                "Value": duplicate_questions,
            },
        ]

        if "expected_sources" in coverage_analysis.columns:

            coverage_summary_rows.append(
                {
                    "Dataset Property": (
                        "Expected Source Coverage"
                    ),
                    "Value": (
                        f"{source_coverage_rate:.1%}"
                    ),
                }
            )
        coverage_summary_df = pd.DataFrame(
            coverage_summary_rows
        )

        # Keep the display column type consistent for
        # Streamlit / PyArrow serialization.
        if "Value" in coverage_summary_df.columns:
            coverage_summary_df["Value"] = (
                coverage_summary_df["Value"].astype(str)
            )

        st.dataframe(
            coverage_summary_df,
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "This section analyzes the existing evaluation "
            "dataset only. It does not generate new questions "
            "or run new retrieval, reranking, embedding, or "
            "LLM experiments."
        )

    st.divider()



    # ========================================================
    # STEP 7.9 — KNOWLEDGE BASE / SOURCE ANALYTICS
    # ========================================================

    st.markdown(
        "### 📚 Knowledge Base & Source Analytics"
    )

    st.caption(
        "Read-only analysis of the documents and chunks "
        "currently stored in the DocMind knowledge base "
        "and ChromaDB vector store."
    )

    try:

        document_manager = (
            get_document_manager()
        )

        vector_store = (
            get_vector_store()
        )

        knowledge_documents = (
            document_manager.list_documents()
        )

        indexed_chunk_count = (
            vector_store.count()
        )

        # ----------------------------------------------------
        # LOAD CHROMA METADATA
        # ----------------------------------------------------

        chroma_records = (
            vector_store.collection.get(
                include=[
                    "metadatas"
                ]
            )
        )

        knowledge_metadatas = (
            chroma_records.get(
                "metadatas",
                []
            )
        )

        knowledge_metadata_df = (
            pd.DataFrame(
                knowledge_metadatas
            )
        )

        # ----------------------------------------------------
        # KNOWLEDGE BASE OVERVIEW
        # ----------------------------------------------------

        st.markdown(
            "#### 📊 Knowledge Base Overview"
        )

        physical_document_count = (
            len(knowledge_documents)
        )

        indexed_document_count = 0

        if (
            not knowledge_metadata_df.empty
            and "filename"
            in knowledge_metadata_df.columns
        ):

            indexed_document_count = (
                knowledge_metadata_df[
                    "filename"
                ]
                .dropna()
                .astype(str)
                .nunique()
            )

        category_count = 0

        if (
            not knowledge_metadata_df.empty
            and "category"
            in knowledge_metadata_df.columns
        ):

            category_count = (
                knowledge_metadata_df[
                    "category"
                ]
                .dropna()
                .astype(str)
                .nunique()
            )

        file_type_count = 0

        if (
            not knowledge_metadata_df.empty
            and "file_type"
            in knowledge_metadata_df.columns
        ):

            file_type_count = (
                knowledge_metadata_df[
                    "file_type"
                ]
                .dropna()
                .astype(str)
                .nunique()
            )

        overview_cols = st.columns(5)

        with overview_cols[0]:

            st.metric(
                "Indexed Chunks",
                indexed_chunk_count,
            )

        with overview_cols[1]:

            st.metric(
                "Physical Documents",
                physical_document_count,
            )

        with overview_cols[2]:

            st.metric(
                "Indexed Documents",
                indexed_document_count,
            )

        with overview_cols[3]:

            st.metric(
                "Categories",
                category_count,
            )

        with overview_cols[4]:

            st.metric(
                "File Types",
                file_type_count,
            )

        # ----------------------------------------------------
        # INDEXING COVERAGE
        # ----------------------------------------------------

        st.markdown(
            "#### 🔎 Document Indexing Coverage"
        )

        if physical_document_count > 0:

            indexing_difference = (
                physical_document_count
                - indexed_document_count
            )

            indexing_rate = (
                indexed_document_count
                / physical_document_count
            )

        else:

            indexing_difference = 0
            indexing_rate = 0

        coverage_cols = st.columns(3)

        with coverage_cols[0]:

            st.metric(
                "Physical Documents",
                physical_document_count,
            )

        with coverage_cols[1]:

            st.metric(
                "Documents Represented in Chroma",
                indexed_document_count,
            )

        with coverage_cols[2]:

            st.metric(
                "Document Indexing Coverage",
                f"{indexing_rate:.1%}",
            )

        indexing_coverage_df = pd.DataFrame(
            {
                "Status": [
                    "Indexed",
                    "Not Represented",
                ],
                "Documents": [
                    indexed_document_count,
                    max(
                        indexing_difference,
                        0,
                    ),
                ],
            }
        )

        st.bar_chart(
            indexing_coverage_df,
            x="Status",
            y="Documents",
        )

        # ----------------------------------------------------
        # DOCUMENT INVENTORY
        # ----------------------------------------------------

        st.markdown(
            "#### 📄 Knowledge Base Document Inventory"
        )

        if knowledge_documents:

            document_inventory_rows = []

            for document in knowledge_documents:

                filename = document.get(
                    "filename",
                    "Unknown",
                )

                document_chunks = 0

                if (
                    not knowledge_metadata_df.empty
                    and "filename"
                    in knowledge_metadata_df.columns
                ):

                    document_chunks = int(
                        (
                            knowledge_metadata_df[
                                "filename"
                            ]
                            .astype(str)
                            == str(filename)
                        )
                        .sum()
                    )

                size_bytes = document.get(
                    "size_bytes",
                    0,
                )

                try:
                    size_kb = (
                        float(size_bytes)
                        / 1024
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    size_kb = 0

                document_inventory_rows.append(
                    {
                        "Filename": filename,
                        "Type": document.get(
                            "extension",
                            "",
                        ),
                        "Size (KB)": round(
                            size_kb,
                            2,
                        ),
                        "Indexed Chunks": (
                            document_chunks
                        ),
                        "Status": (
                            "Indexed"
                            if document_chunks > 0
                            else "Not Represented"
                        ),
                    }
                )

            document_inventory_df = (
                pd.DataFrame(
                    document_inventory_rows
                )
                .sort_values(
                    [
                        "Status",
                        "Filename",
                    ]
                )
            )

            st.dataframe(
                document_inventory_df,
                width="stretch",
                hide_index=True,
            )

        else:

            st.info(
                "No supported documents are currently "
                "present in the knowledge-base documents "
                "directory."
            )

        # ----------------------------------------------------
        # CHUNKS BY DOCUMENT
        # ----------------------------------------------------

        if (
            not knowledge_metadata_df.empty
            and "filename"
            in knowledge_metadata_df.columns
        ):

            st.markdown(
                "#### 🧩 Chunks by Document"
            )

            chunks_by_document = (
                knowledge_metadata_df[
                    "filename"
                ]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis(
                    "Filename"
                )
                .reset_index(
                    name="Chunks"
                )
            )

            st.bar_chart(
                chunks_by_document,
                x="Filename",
                y="Chunks",
            )

            st.dataframe(
                chunks_by_document,
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # CHUNKS BY CATEGORY
        # ----------------------------------------------------

        if (
            not knowledge_metadata_df.empty
            and "category"
            in knowledge_metadata_df.columns
        ):

            st.markdown(
                "#### 🗂️ Chunks by Knowledge Category"
            )

            chunks_by_category = (
                knowledge_metadata_df[
                    "category"
                ]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis(
                    "Category"
                )
                .reset_index(
                    name="Chunks"
                )
            )

            st.bar_chart(
                chunks_by_category,
                x="Category",
                y="Chunks",
            )

            st.dataframe(
                chunks_by_category,
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # FILE TYPE DISTRIBUTION
        # ----------------------------------------------------

        if (
            not knowledge_metadata_df.empty
            and "file_type"
            in knowledge_metadata_df.columns
        ):

            st.markdown(
                "#### 📁 File Type Distribution"
            )

            file_type_distribution = (
                knowledge_metadata_df[
                    "file_type"
                ]
                .fillna("Unknown")
                .astype(str)
                .str.lower()
                .value_counts()
                .rename_axis(
                    "File Type"
                )
                .reset_index(
                    name="Chunks"
                )
            )

            st.bar_chart(
                file_type_distribution,
                x="File Type",
                y="Chunks",
            )

            st.dataframe(
                file_type_distribution,
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # PAGE COVERAGE
        # ----------------------------------------------------

        if (
            not knowledge_metadata_df.empty
            and "page"
            in knowledge_metadata_df.columns
        ):

            st.markdown(
                "#### 📑 Page Coverage"
            )

            page_values = pd.to_numeric(
                knowledge_metadata_df[
                    "page"
                ],
                errors="coerce",
            )

            valid_pages = (
                page_values[
                    page_values >= 1
                ]
            )

            unique_page_count = (
                int(
                    valid_pages
                    .dropna()
                    .nunique()
                )
            )

            maximum_page = (
                int(
                    valid_pages.max()
                )
                if not valid_pages.empty
                else 0
            )

            page_cols = st.columns(3)

            with page_cols[0]:

                st.metric(
                    "Pages Represented",
                    unique_page_count,
                )

            with page_cols[1]:

                st.metric(
                    "Highest Page Number",
                    maximum_page,
                )

            with page_cols[2]:

                st.metric(
                    "Chunks With Valid Page",
                    int(
                        valid_pages
                        .notna()
                        .sum()
                    ),
                )

        # ----------------------------------------------------
        # CHUNK CONFIGURATION
        # ----------------------------------------------------

        if (
            not knowledge_metadata_df.empty
            and "chunk_size"
            in knowledge_metadata_df.columns
            and "chunk_overlap"
            in knowledge_metadata_df.columns
        ):

            st.markdown(
                "#### 📏 Chunk Configuration in Index"
            )

            chunk_size_values = pd.to_numeric(
                knowledge_metadata_df[
                    "chunk_size"
                ],
                errors="coerce",
            )

            chunk_overlap_values = pd.to_numeric(
                knowledge_metadata_df[
                    "chunk_overlap"
                ],
                errors="coerce",
            )

            chunk_configuration_df = pd.DataFrame(
                {
                    "Chunk Size": (
                        chunk_size_values
                        .fillna(0)
                        .astype(int)
                    ),
                    "Chunk Overlap": (
                        chunk_overlap_values
                        .fillna(0)
                        .astype(int)
                    ),
                }
            )

            configuration_summary = (
                chunk_configuration_df
                .value_counts()
                .rename(
                    "Chunks"
                )
                .reset_index()
            )

            st.dataframe(
                configuration_summary,
                width="stretch",
                hide_index=True,
            )

            chunk_size_cols = st.columns(3)

            with chunk_size_cols[0]:

                st.metric(
                    "Average Chunk Size",
                    f"{chunk_size_values.mean():.0f}",
                )

            with chunk_size_cols[1]:

                st.metric(
                    "Average Chunk Overlap",
                    f"{chunk_overlap_values.mean():.0f}",
                )

            with chunk_size_cols[2]:

                st.metric(
                    "Unique Configurations",
                    len(
                        configuration_summary
                    ),
                )

        # ----------------------------------------------------
        # METADATA COMPLETENESS
        # ----------------------------------------------------

        st.markdown(
            "#### 🏷️ Chroma Metadata Completeness"
        )

        metadata_columns = [
            "source",
            "filename",
            "file_type",
            "document_type",
            "page",
            "chunk_number",
            "chunk_size",
            "chunk_overlap",
            "category",
        ]

        metadata_completeness_rows = []

        if knowledge_metadata_df.empty:

            st.info(
                "No Chroma metadata records are currently "
                "available."
            )

        else:

            for column in metadata_columns:

                if (
                    column
                    not in knowledge_metadata_df.columns
                ):

                    metadata_completeness_rows.append(
                        {
                            "Metadata Field": column,
                            "Present Records": 0,
                            "Missing / Empty": (
                                indexed_chunk_count
                            ),
                            "Completeness": 0.0,
                        }
                    )

                    continue

                series = (
                    knowledge_metadata_df[
                        column
                    ]
                )

                missing = int(
                    series.isna().sum()
                )

                if (
                    series.dtype == "object"
                ):

                    missing += int(
                        series
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .eq("")
                        .sum()
                    )

                missing = min(
                    missing,
                    indexed_chunk_count,
                )

                present = (
                    indexed_chunk_count
                    - missing
                )

                completeness = (
                    present
                    / indexed_chunk_count
                    if indexed_chunk_count
                    else 0
                )

                metadata_completeness_rows.append(
                    {
                        "Metadata Field": column,
                        "Present Records": present,
                        "Missing / Empty": missing,
                        "Completeness": completeness,
                    }
                )

            metadata_completeness_df = (
                pd.DataFrame(
                    metadata_completeness_rows
                )
                .sort_values(
                    "Completeness"
                )
            )

            st.dataframe(
                metadata_completeness_df.style.format(
                    {
                        "Completeness": "{:.1%}"
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # SOURCE / PATH DISTRIBUTION
        # ----------------------------------------------------

        if (
            not knowledge_metadata_df.empty
            and "source"
            in knowledge_metadata_df.columns
        ):

            st.markdown(
                "#### 🗃️ Source Path Distribution"
            )

            source_distribution = (
                knowledge_metadata_df[
                    "source"
                ]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis(
                    "Source"
                )
                .reset_index(
                    name="Chunks"
                )
            )

            st.dataframe(
                source_distribution,
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # KNOWLEDGE BASE HEALTH
        # ----------------------------------------------------

        st.markdown(
            "#### 🩺 Knowledge Base Health"
        )

        health_rows = []

        health_rows.append(
            {
                "Check": (
                    "Indexed Chunks"
                ),
                "Value": indexed_chunk_count,
                "Status": (
                    "Healthy"
                    if indexed_chunk_count > 0
                    else "Empty"
                ),
            }
        )

        health_rows.append(
            {
                "Check": (
                    "Physical Documents"
                ),
                "Value": physical_document_count,
                "Status": (
                    "Available"
                    if physical_document_count > 0
                    else "Empty"
                ),
            }
        )

        health_rows.append(
            {
                "Check": (
                    "Document Indexing Coverage"
                ),
                "Value": f"{indexing_rate:.1%}",
                "Status": (
                    "Complete"
                    if indexing_rate >= 1
                    else "Partial"
                ),
            }
        )

        if not knowledge_metadata_df.empty:

            required_metadata = [
                column
                for column in metadata_columns
                if column
                in knowledge_metadata_df.columns
            ]

            if required_metadata:

                metadata_completeness_average = (
                    metadata_completeness_df[
                        "Completeness"
                    ]
                    .mean()
                )

            else:

                metadata_completeness_average = 0

        else:

            metadata_completeness_average = 0

        health_rows.append(
            {
                "Check": (
                    "Average Metadata Completeness"
                ),
                "Value": (
                    f"{metadata_completeness_average:.1%}"
                ),
                "Status": (
                    "Healthy"
                    if metadata_completeness_average
                    >= 0.95
                    else "Review"
                ),
            }
        )
        health_df = pd.DataFrame(
            health_rows
        )

        # Keep the display column type consistent for
        # Streamlit / PyArrow serialization.
        if "Value" in health_df.columns:
            health_df["Value"] = health_df[
                "Value"
            ].astype(str)
        st.dataframe(
            health_df,
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # RAW CHROMA METADATA
        # ----------------------------------------------------

        with st.expander(
            "🔬 Inspect Raw Chroma Metadata"
        ):

            if knowledge_metadata_df.empty:

                st.info(
                    "No Chroma metadata is available."
                )

            else:

                st.dataframe(
                    knowledge_metadata_df,
                    width="stretch",
                    hide_index=True,
                )

        st.caption(
            f"Knowledge-base analytics are read-only. "
            f"The current ChromaDB collection contains "
            f"{indexed_chunk_count} indexed chunks. "
            "Refreshing this dashboard does not re-index "
            "documents or regenerate embeddings."
        )

    except Exception as knowledge_error:

        st.error(
            "Knowledge-base analytics could not be loaded."
        )

        st.caption(
            f"Details: {knowledge_error}"
        )

    st.divider()






# ============================================================
# DOCMIND WORKSPACE STATE
# ============================================================

if "docmind_view" not in st.session_state:
    st.session_state.docmind_view = "AI Chat"

if "conversation_title" not in st.session_state:
    st.session_state.conversation_title = "New conversation"

if "last_error" not in st.session_state:
    st.session_state.last_error = None

if "scroll_chat_to_bottom" not in st.session_state:
    st.session_state.scroll_chat_to_bottom = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "docmind_is_thinking" not in st.session_state:
    st.session_state.docmind_is_thinking = False

if "show_clear_session_confirmation" not in st.session_state:
    st.session_state.show_clear_session_confirmation = False

# ------------------------------------------------------------
# MULTI-CHAT SESSION STATE
# ------------------------------------------------------------
# Each sidebar chat is a separate conversation thread.
# The existing conversation_history / conversation_title / last_result
# variables remain the "active chat" working state so the rest of the
# application does not need to be rewritten.
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "chat_order" not in st.session_state:
    st.session_state.chat_order = []

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0


def _next_chat_id():
    st.session_state.chat_counter += 1
    return f"chat_{st.session_state.chat_counter}"


def _empty_chat_state():
    return {
        "title": "New conversation",
        "history": [],
        "last_result": None,
    }


def save_active_chat():
    """Persist the current working conversation into the multi-chat store."""
    chat_id = st.session_state.active_chat_id

    if not chat_id:
        return

    st.session_state.chat_sessions[chat_id] = {
        "title": st.session_state.conversation_title or "New conversation",
        "history": list(st.session_state.conversation_history),
        "last_result": st.session_state.last_result,
    }

    if chat_id not in st.session_state.chat_order:
        st.session_state.chat_order.append(chat_id)


def load_chat(chat_id):
    """Switch the workspace to an existing chat thread."""
    if chat_id not in st.session_state.chat_sessions:
        return False

    save_active_chat()

    chat = st.session_state.chat_sessions[chat_id]

    st.session_state.active_chat_id = chat_id
    st.session_state.conversation_title = (
        chat.get("title") or "New conversation"
    )
    st.session_state.conversation_history = list(
        chat.get("history", [])
    )
    st.session_state.last_result = chat.get("last_result")

    if (
        st.session_state.last_result is None
        and st.session_state.conversation_history
    ):
        st.session_state.last_result = (
            st.session_state.conversation_history[-1]
        )

    st.session_state.last_error = None
    st.session_state.scroll_chat_to_bottom = False
    st.session_state.pending_question = None
    st.session_state.docmind_is_thinking = False
    st.session_state.docmind_view = "AI Chat"
    return True


def create_new_chat():
    """
    Start a new empty chat without deleting any previous conversations.
    Previous chats stay available in the left sidebar.
    """
    save_active_chat()

    chat_id = _next_chat_id()
    st.session_state.chat_sessions[chat_id] = _empty_chat_state()
    st.session_state.chat_order.append(chat_id)
    st.session_state.active_chat_id = chat_id

    st.session_state.conversation_history = []
    st.session_state.last_result = None
    st.session_state.conversation_title = "New conversation"
    st.session_state.last_error = None
    st.session_state.scroll_chat_to_bottom = False
    st.session_state.pending_question = None
    st.session_state.docmind_is_thinking = False
    st.session_state.docmind_view = "AI Chat"


def reset_chat():
    """
    Reset only the currently active working conversation.

    This does NOT delete other stored chat threads. It is kept as a helper
    for flows that need a clean active workspace.
    """
    st.session_state.conversation_history = []
    st.session_state.last_result = None
    st.session_state.conversation_title = "New conversation"
    st.session_state.last_error = None
    st.session_state.scroll_chat_to_bottom = False
    st.session_state.pending_question = None
    st.session_state.docmind_is_thinking = False
    st.session_state.docmind_view = "AI Chat"

    if st.session_state.active_chat_id:
        st.session_state.chat_sessions[
            st.session_state.active_chat_id
        ] = _empty_chat_state()


def delete_chat(chat_id):
    """Delete one chat thread and activate another remaining chat if possible."""
    if chat_id in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[chat_id]

    st.session_state.chat_order = [
        item
        for item in st.session_state.chat_order
        if item != chat_id
    ]

    if st.session_state.active_chat_id == chat_id:
        remaining = [
            item
            for item in st.session_state.chat_order
            if item in st.session_state.chat_sessions
            and st.session_state.chat_sessions[item].get("history")
        ]

        if remaining:
            load_chat(remaining[-1])
        else:
            st.session_state.active_chat_id = None
            st.session_state.conversation_history = []
            st.session_state.last_result = None
            st.session_state.conversation_title = "New conversation"
            st.session_state.last_error = None
            st.session_state.pending_question = None
            st.session_state.docmind_is_thinking = False
            st.session_state.docmind_view = "AI Chat"


def clear_all_chats():
    """Clear every chat thread while keeping uploaded/indexed documents."""
    st.session_state.chat_sessions = {}
    st.session_state.chat_order = []
    st.session_state.active_chat_id = None

    st.session_state.conversation_history = []
    st.session_state.last_result = None
    st.session_state.conversation_title = "New conversation"
    st.session_state.last_error = None
    st.session_state.scroll_chat_to_bottom = False
    st.session_state.pending_question = None
    st.session_state.docmind_is_thinking = False
    st.session_state.docmind_view = "AI Chat"

    st.session_state.show_clear_session_confirmation = False
    st.session_state.action_message = {
        "type": "success",
        "text": "All chats were cleared. Uploaded documents were kept.",
    }


def clear_all_data():
    """
    Clear all uploaded/indexed documents and all chat threads.

    Each document is deleted through DocumentManager so its stored file and
    vector/index data are removed through the project's existing delete flow.
    """
    document_manager = get_document_manager()

    try:
        documents = document_manager.list_documents()
    except Exception as exc:
        raise RuntimeError(
            f"Could not read the uploaded document list: {exc}"
        ) from exc

    failures = []

    for document in documents:
        filename = document.get("filename")
        if not filename:
            continue

        try:
            document_manager.delete_document(filename)
        except Exception as exc:
            failures.append(f"{filename}: {exc}")

    if failures:
        raise RuntimeError(
            "Some documents could not be deleted:\n" + "\n".join(failures)
        )

    st.session_state.chat_sessions = {}
    st.session_state.chat_order = []
    st.session_state.active_chat_id = None

    st.session_state.conversation_history = []
    st.session_state.last_result = None
    st.session_state.conversation_title = "New conversation"
    st.session_state.last_error = None
    st.session_state.scroll_chat_to_bottom = False
    st.session_state.pending_question = None
    st.session_state.docmind_is_thinking = False
    st.session_state.docmind_view = "AI Chat"

    st.session_state.show_clear_session_confirmation = False
    st.session_state.action_message = {
        "type": "success",
        "text": "All chats, uploaded files, and indexed document data were cleared.",
    }


# ------------------------------------------------------------
# MIGRATE THE CURRENT SINGLE CHAT INTO THE MULTI-CHAT STORE
# ------------------------------------------------------------
# This runs once after upgrading from the previous one-chat implementation.
if (
    st.session_state.active_chat_id is None
    and st.session_state.conversation_history
):
    migrated_chat_id = _next_chat_id()

    st.session_state.chat_sessions[migrated_chat_id] = {
        "title": st.session_state.conversation_title or "New conversation",
        "history": list(st.session_state.conversation_history),
        "last_result": st.session_state.last_result,
    }
    st.session_state.chat_order.append(migrated_chat_id)
    st.session_state.active_chat_id = migrated_chat_id


def ask_question(question: str, top_k_value: int):
    """Run the existing RAG pipeline once for one submitted question."""
    question = (question or "").strip()
    if not question:
        return False

    try:
        # The visible thinking indicator is rendered directly above the composer
        # by the chat workspace so it appears exactly where the next answer will go.
        answer_service = get_answer_service(top_k_value)
        result = answer_service.ask(question)

        st.session_state.last_result = result
        st.session_state.last_error = None
        add_to_conversation(result)
        st.session_state.scroll_chat_to_bottom = True

        if st.session_state.conversation_title == "New conversation":
            compact = " ".join(question.split())
            st.session_state.conversation_title = (
                compact[:42] + "…" if len(compact) > 42 else compact
            )

        # If this is the first message in a brand-new workspace, create a
        # persistent chat thread now. Otherwise update the existing thread.
        if st.session_state.active_chat_id is None:
            chat_id = _next_chat_id()
            st.session_state.active_chat_id = chat_id
            st.session_state.chat_order.append(chat_id)

        save_active_chat()

        return True

    except Exception as exc:
        st.session_state.last_error = str(exc)
        st.session_state.action_message = {
            "type": "error",
            "text": (
                "Something went wrong while generating the answer. "
                "Please try again."
            ),
        }
        return False



def scroll_chat_history_to_bottom():
    """
    Show the newest user question first, followed by the DocMind answer.

    The function name is preserved so the existing session-state flow does
    not need to change. It scrolls only the internal conversation viewport.
    """
    components.html(
        """
        <script>
        (() => {
            const win = window.parent;
            const doc = win.document;

            const positionAtLatestAnswerStart = () => {
                const viewport = doc.querySelector(
                    ".st-key-dm_conversation_viewport"
                );
                const marker = doc.querySelector(
                    "#dm-latest-turn-start"
                );

                if (!viewport || !marker) return;

                const viewportRect = viewport.getBoundingClientRect();
                const markerRect = marker.getBoundingClientRect();

                const target =
                    viewport.scrollTop +
                    (markerRect.top - viewportRect.top) -
                    8;

                viewport.scrollTop = Math.max(0, target);
            };

            [30, 100, 220, 420, 700].forEach((delay) => {
                win.setTimeout(positionAtLatestAnswerStart, delay);
            });
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def render_chat_message(
    question,
    answer,
    sources,
    result=None,
    is_latest=False,
):
    """Render one clean conversation turn without retrieval UI inside chat."""

    # A stable marker immediately BEFORE the newest user question.
    # This makes the newest question appear first, followed by the answer.
    if is_latest:
        st.markdown(
            '<div id="dm-latest-turn-start" '
            'style="height:1px; margin:0; padding:0;"></div>',
            unsafe_allow_html=True,
        )

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🧠"):
        st.markdown(answer)


def render_right_upload_controls(document_manager):
    """Render the fixed upload controls for the Documents sidebar."""

    st.markdown(
        '<div class="dm-right-section-title">Upload</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt"],
        key="workspace_uploader",
        help="Supported formats: PDF, DOCX and TXT.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        size_mb = uploaded_file.size / (1024 * 1024)

        st.markdown(
            f"""
            <div class="dm-upload-selected">
                <div class="dm-upload-file">
                    <span>📄</span>
                    <span>{uploaded_file.name}</span>
                </div>
                <div class="dm-upload-meta">
                    {size_mb:.2f} MB • Ready to index
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    submit_upload = st.button(
        "Submit",
        width="stretch",
        key="workspace_upload_submit",
        disabled=uploaded_file is None,
    )

    if submit_upload and uploaded_file is not None:
        try:
            with st.status(
                "Processing document…",
                expanded=False,
            ) as status:
                result = document_manager.upload_and_index(uploaded_file)

                status.update(
                    label="Document indexed",
                    state="complete",
                )

            chunks = result.get("chunks", 0)
            indexed = result.get("indexed", chunks)
            status_value = result.get("status", "success")

            if status_value == "success":
                st.session_state.action_message = {
                    "type": "success",
                    "text": (
                        f"**{result.get('filename', uploaded_file.name)}** "
                        f"is ready. {chunks} chunks created, "
                        f"{indexed} indexed."
                    ),
                }
            else:
                st.session_state.action_message = {
                    "type": "warning",
                    "text": (
                        "Document processing finished with status "
                        f"`{status_value}`."
                    ),
                }

            st.rerun()

        except FileExistsError as exc:
            st.warning(str(exc))

        except Exception as exc:
            st.error(
                "Couldn't process this document. "
                "Please check that the file is supported and try again."
            )
            with st.expander("Technical details"):
                st.code(str(exc))


def render_right_documents_content(document_manager, current_result):
    """Render only the independently scrollable Documents sidebar content."""

    st.markdown(
        '<div class="dm-right-section-title">Uploaded Documents</div>',
        unsafe_allow_html=True,
    )

    # Use the real document manager as the single source of truth.
    try:
        documents = document_manager.list_documents()
    except Exception as exc:
        documents = []
        st.caption(f"Document list unavailable: {exc}")

    if not documents:
        st.markdown(
            '<div class="dm-right-empty">No documents uploaded yet.</div>',
            unsafe_allow_html=True,
        )
    else:
        for idx, document in enumerate(documents):
            filename = document.get("filename", "Unknown")
            extension = document.get("extension", "").upper()
            size_bytes = document.get("size_bytes", 0) or 0

            if size_bytes >= 1024 * 1024:
                size_text = f"{size_bytes / (1024 * 1024):.2f} MB"
            else:
                size_text = f"{size_bytes / 1024:.1f} KB"

            st.markdown(
                f"""
                <div class="dm-doc-row">
                    <div class="dm-doc-main">
                        <span class="dm-doc-icon">▤</span>
                        <div class="dm-doc-name-wrap">
                            <div class="dm-doc-name">{filename}</div>
                            <div class="dm-doc-meta">{extension or 'FILE'} • {size_text}</div>
                        </div>
                    </div>
                    <span class="dm-status ready">Ready</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="dm-right-section-title">Indexed Knowledge Base</div>',
        unsafe_allow_html=True,
    )

    try:
        stats = document_manager.get_statistics()
        document_count = stats.get("document_count", len(documents))
        indexed_chunks = stats.get("indexed_chunks", 0)
    except Exception:
        document_count = len(documents)
        indexed_chunks = 0

    st.markdown(
        f"""
        <div class="dm-kb-summary">
            <span>{document_count} document{"s" if document_count != 1 else ""}</span>
            <span>{indexed_chunks} chunks</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not documents:
        st.markdown(
            '<div class="dm-right-empty">Upload a document to build your index.</div>',
            unsafe_allow_html=True,
        )
    else:
        for idx, document in enumerate(documents):
            filename = document.get("filename", "Unknown")

            chunk_value = document.get("chunks")
            if chunk_value is None:
                chunk_value = document.get("chunk_count")

            chunk_text = (
                f"{chunk_value} chunks"
                if chunk_value is not None
                else "Indexed"
            )

            st.markdown(
                f"""
                <div class="dm-indexed-row">
                    <div class="dm-indexed-main">
                        <span class="dm-index-check">✓</span>
                        <div>
                            <div class="dm-doc-name">{filename}</div>
                            <div class="dm-doc-meta">{chunk_text}</div>
                        </div>
                    </div>
                    <span class="dm-status indexed">Indexed</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            action_col1, action_col2 = st.columns(2)

            with action_col1:
                if st.button(
                    "Re-index",
                    key=f"workspace_reindex_{idx}_{filename}",
                    width="stretch",
                ):
                    try:
                        with st.spinner(f"Re-indexing {filename}…"):
                            document_manager.reindex_document(filename)

                        st.rerun()

                    except Exception as exc:
                        st.error(
                            f"Re-indexing failed: {exc}"
                        )

            with action_col2:
                if st.button(
                    "Delete",
                    key=f"workspace_delete_{idx}_{filename}",
                    width="stretch",
                ):
                    try:
                        with st.spinner(f"Deleting {filename}…"):
                            document_manager.delete_document(filename)

                        st.rerun()

                    except Exception as exc:
                        st.error(
                            f"Couldn't delete this document: {exc}"
                        )

    st.markdown(
        '<div class="dm-right-section-title">Retrieved for This Answer</div>',
        unsafe_allow_html=True,
    )

    if not current_result:
        st.markdown(
            '<div class="dm-right-empty">'
            'Ask a question to see the sources used for the answer.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    retrieved_sources = current_result.get("sources", []) or []

    if not retrieved_sources:
        st.markdown(
            '<div class="dm-right-empty">'
            'No source information was returned for this answer.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    for source in retrieved_sources:
        filename = source.get("filename", "Unknown")
        page = source.get("page")
        chunk = source.get("chunk_number", "?")
        distance = source.get("distance")

        page_text = (
            "Page N/A"
            if page in (None, -1)
            else f"Page {page}"
        )

        try:
            distance_text = f"{float(distance):.2f}"
        except (TypeError, ValueError):
            distance_text = "N/A"

        st.markdown(
            f"""
            <div class="dm-retrieved-card">
                <div class="dm-retrieved-title">
                    <span class="dm-retrieved-check">●</span>
                    <span>{filename}</span>
                </div>
                <div class="dm-retrieved-meta">
                    {page_text} • Chunk {chunk}
                </div>
                <div class="dm-retrieved-score">
                    <span>Relevance: <b>High</b></span>
                    <span>Distance: <b>{distance_text}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SIDEBAR — CONVERSATIONS + NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <span class="dm-sidebar-brand-sticky-marker"></span>
        <div class="dm-brand">
            <div class="dm-brand-icon">◆</div>
            <div>
                <div class="dm-brand-title">DocMind</div>
                <div class="dm-brand-sub">AI document intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<span class="dm-sidebar-newchat-sticky-marker"></span>',
        unsafe_allow_html=True,
    )

    if st.button(
        "✎ New chat",
        type="primary",
        width="stretch",
        key="new_chat_button",
    ):
        create_new_chat()
        st.rerun()

    st.markdown(
        '<div class="dm-nav-label">Recents</div>',
        unsafe_allow_html=True,
    )

    # Persist the active workspace before drawing the chat list.
    save_active_chat()

    visible_chat_ids = [
        chat_id
        for chat_id in st.session_state.chat_order
        if (
            chat_id in st.session_state.chat_sessions
            and st.session_state.chat_sessions[chat_id].get("history")
        )
    ]

    if visible_chat_ids:
        # Newest chats appear first, similar to modern chat applications.
        for chat_id in reversed(visible_chat_ids):
            chat_data = st.session_state.chat_sessions[chat_id]

            title = chat_data.get("title") or "New conversation"
            title = " ".join(str(title).split())

            if len(title) > 31:
                title = title[:31] + "…"

            is_active = (
                chat_id == st.session_state.active_chat_id
            )

            # One clean visual chat row.
            # The visible delete icon is injected INSIDE the chat button.
            # A hidden Streamlit delete button remains underneath so the
            # existing Python delete functionality stays reliable.
            with st.container(
                key=f"chat_row_{chat_id}",
                border=False,
            ):
                if st.button(
                    title,
                    key=f"chat_nav_{chat_id}",
                    width="stretch",
                    help=(
                        "Current conversation."
                        if is_active
                        else "Open this conversation."
                    ),
                    type="primary" if is_active else "secondary",
                ):
                    if load_chat(chat_id):
                        st.rerun()

                if st.button(
                    "",
                    key=f"delete_chat_{chat_id}",
                    help=f"Delete '{title}'.",
                ):
                    delete_chat(chat_id)
                    st.rerun()

    else:
        st.caption("No conversations in this session yet.")

    # Visual separator between recent chats and the navigation/tools below.
    st.markdown(
        '<div class="dm-recents-separator" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "📊 Analytics & Evaluation",
        width="stretch",
        key="analytics_eval_nav",
    ):
        st.session_state.docmind_view = "Analytics & Evaluation"
        st.rerun()

    if st.button(
        "📚 Knowledge Base",
        width="stretch",
        key="knowledge_nav",
    ):
        st.session_state.docmind_view = "Knowledge Base"
        st.rerun()

    top_k = st.slider(
        "Top-K chunks",
        min_value=1,
        max_value=10,
        value=3,
        key="workspace_top_k",
        help="Number of relevant chunks retrieved for each question.",
    )

    st.caption(f"Retrieving top {top_k} relevant chunks.")

    st.divider()

    if st.button(
        "🗑️ Clear session",
        width="stretch",
        key="clear_session_sidebar",
    ):
        st.session_state.show_clear_session_confirmation = True
        st.rerun()

    if st.session_state.show_clear_session_confirmation:
        st.markdown(
            """
            <div class="dm-clear-confirm-box">
                <div class="dm-clear-confirm-title">Clear session?</div>
                <div class="dm-clear-confirm-text">
                    Choose what you want to remove.
                </div>
                <div class="dm-clear-confirm-warning">
                    <b>Clear all data</b> permanently removes all uploaded
                    documents, indexed document data, and all chats.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        clear_chats_col, clear_all_col = st.columns(2, gap="small")

        with clear_chats_col:
            if st.button(
                "Clear chats",
                width="stretch",
                key="confirm_clear_chats_only",
                help="Delete all chats but keep uploaded documents and the knowledge base.",
            ):
                clear_all_chats()
                st.rerun()

        with clear_all_col:
            if st.button(
                "Clear all data",
                width="stretch",
                key="confirm_clear_everything",
                help="Delete all uploaded files, indexed data, and all chats.",
            ):
                try:
                    with st.spinner("Clearing all data…"):
                        clear_all_data()
                    st.rerun()

                except Exception as exc:
                    st.session_state.action_message = {
                        "type": "error",
                        "text": f"Could not clear all data: {exc}",
                    }
                    st.session_state.show_clear_session_confirmation = False
                    st.rerun()

        if st.button(
            "Cancel",
            width="stretch",
            key="cancel_clear_session",
        ):
            st.session_state.show_clear_session_confirmation = False
            st.rerun()

    st.divider()

    st.caption(
        "Session history is temporary and is not used "
        "as a replacement for document retrieval."
    )


# ============================================================
# THREE-PANEL WORKSPACE
# ============================================================

try:
    workspace_document_manager = get_document_manager()
except Exception:
    workspace_document_manager = None

# ============================================================
# GLOBAL THEME CONTROL
# Separate from the Documents panel so mobile stacking cannot
# push it into the middle of the page.
# ============================================================
with st.container(key="dm_global_theme_control", border=False):
    selected_theme = st.radio(
        "Theme",
        options=["🌙", "☀️"],
        index=0 if st.session_state.theme == "dark" else 1,
        horizontal=True,
        key="documents_theme_segmented_toggle",
        label_visibility="collapsed",
    )

    selected_theme_value = (
        "light"
        if selected_theme == "☀️"
        else "dark"
    )

    if selected_theme_value != st.session_state.theme:
        st.session_state.theme = selected_theme_value
        st.session_state.documents_theme_toggle = (
            selected_theme_value == "light"
        )
        st.rerun()


center_panel, right_panel = st.columns(
    [3.15, 1.15],
    gap="small",
)


# ============================================================
# CENTER PANEL
# ============================================================

with center_panel:
    if st.session_state.docmind_view == "Analytics & Evaluation":
        with st.container(
            height=700,
            border=False,
            key="analytics_scroll_container",
        ):
            st.markdown(
                '<div class="dm-analytics-scroll-marker" aria-hidden="true"></div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="dm-center-title">'
                'Analytics & Evaluation'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="dm-center-subtitle">'
                'Measure retrieval, generation, answer quality, '
                'and recorded RAG experiments.'
                '</div>',
                unsafe_allow_html=True,
            )

            render_evaluation_dashboard()

    elif st.session_state.docmind_view == "Knowledge Base":
        with st.container(
            height=700,
            border=False,
            key="knowledge_scroll_container",
        ):
            st.markdown(
                '<div class="dm-kb-scroll-marker" aria-hidden="true"></div>',
                unsafe_allow_html=True,
            )
    
            st.markdown(
                '<div class="dm-center-title">Knowledge Base</div>',
                unsafe_allow_html=True,
            )
    
            st.markdown(
                '<div class="dm-center-subtitle">'
                'Inspect the indexed document collection without '
                're-running embeddings.'
                '</div>',
                unsafe_allow_html=True,
            )
    
            try:
                kb_stats = workspace_document_manager.get_statistics()
    
                kb_docs = kb_stats.get(
                    "document_count",
                    0,
                )
    
                kb_chunks = kb_stats.get(
                    "indexed_chunks",
                    0,
                )
    
                kb_size = (
                    kb_stats.get(
                        "total_size_bytes",
                        0,
                    )
                    / (1024 * 1024)
                )
    
            except Exception:
                kb_docs = 0
                kb_chunks = 0
                kb_size = 0
    
            c1, c2, c3 = st.columns(3)
    
            c1.metric("Documents", kb_docs)
            c2.metric("Indexed chunks", kb_chunks)
            c3.metric("Storage", f"{kb_size:.2f} MB")
    
            st.markdown("#### Knowledge-base status")
    
            st.info(
                "DocMind uses local embeddings, persistent ChromaDB, "
                "hybrid dense + BM25 retrieval, optional cross-encoder "
                "reranking, and grounded Ollama Cloud generation."
            )
    
            if workspace_document_manager:
                try:
                    kb_documents = (
                        workspace_document_manager
                        .list_documents()
                    )
                except Exception:
                    kb_documents = []
    
                if kb_documents:
                    for document in kb_documents:
                        filename = document.get(
                            "filename",
                            "Unknown",
                        )
    
                        extension = document.get(
                            "extension",
                            "",
                        ).upper()
    
                        size = (
                            document.get(
                                "size_bytes",
                                0,
                            )
                            / 1024
                        )
    
                        with st.container(border=True):
                            st.markdown(
                                f"📄 **{filename}**"
                            )
    
                            st.caption(
                                f"{extension} • "
                                f"{size:.1f} KB • Ready"
                            )
                else:
                    st.info(
                        "No managed documents are available yet."
                    )
    else:
        history = st.session_state.conversation_history

        # Isolated marker: older chat CSS does not target this version.
        st.markdown(
            '<div class="dm-agent-v2-column-marker" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )

        with st.container(key="dm_center_chat", border=False):
            with st.container(key="dm_center_header", border=False):
                # --------------------------------------------------------
                # Fixed DocMind identity area
                # --------------------------------------------------------
                st.markdown(
                    """
                    <div class="dm-agent-v2-header">
                        <div class="dm-agent-v2-brand">
                            <div class="dm-agent-v2-logo">◆</div>
                            <div class="dm-agent-v2-title">DocMind</div>
                        </div>
                        <div class="dm-agent-v2-subtitle">
                            AI-powered document intelligence.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # --------------------------------------------------------
            # CENTER CONTENT
            # --------------------------------------------------------
            # Use Streamlit's native fixed-height containers for reliable
            # internal scrolling. This avoids the conflicting CSS/JS scroll
            # behavior that previously hid the empty state and chat history.
            if history:
                with st.container(
                    height=520,
                    border=False,
                    key="dm_conversation_viewport",
                ):
                    for idx, conversation in enumerate(history):
                        render_chat_message(
                            conversation.get("question", ""),
                            conversation.get(
                                "answer",
                                "No answer generated.",
                            ),
                            conversation.get("sources", []) or [],
                            conversation,
                            is_latest=(idx == len(history) - 1),
                        )

                        if idx < len(history) - 1:
                            st.markdown(
                                '<div class="dm-agent-v2-turn-gap"></div>',
                                unsafe_allow_html=True,
                            )

                    if st.session_state.scroll_chat_to_bottom:
                        scroll_chat_history_to_bottom()
                        st.session_state.scroll_chat_to_bottom = False

                    if st.session_state.last_error:
                        st.error(
                            "Something went wrong while generating the answer. "
                            "Please try again."
                        )
                        with st.expander("Technical details"):
                            st.code(st.session_state.last_error)

                    # The thinking indicator is rendered outside this
                    # scrollable history container, directly above the composer.

            else:
                with st.container(
                    height=520,
                    border=False,
                    key="dm_empty_state_viewport",
                ):
                    st.markdown(
                        """
                        <div class="dm-agent-v2-empty">
                            <div class="dm-agent-v2-empty-icon">🧠</div>
                            <div class="dm-agent-v2-empty-title">
                                Ask questions about your documents
                            </div>
                            <div class="dm-agent-v2-empty-text">
                                Upload your knowledge base, then ask grounded
                                questions. DocMind retrieves evidence before
                                generating an answer.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        '<div class="dm-agent-v2-example-title">Try a question</div>',
                        unsafe_allow_html=True,
                    )

                    examples = [
                        "What is the main topic of the project?",
                        "Summarize the key findings.",
                        "List the important requirements.",
                    ]

                    example_cols = st.columns(3)

                    for idx, example in enumerate(examples):
                        with example_cols[idx]:
                            if st.button(
                                example,
                                key=f"workspace_example_{idx}",
                                width="stretch",
                                disabled=st.session_state.docmind_is_thinking,
                            ):
                                st.session_state.pending_question = example
                                st.session_state.docmind_is_thinking = True
                                st.rerun()

                    if st.session_state.last_error:
                        st.error(
                            "Something went wrong while generating the answer. "
                            "Please try again."
                        )
                        with st.expander("Technical details"):
                            st.code(st.session_state.last_error)

            # --------------------------------------------------------
            # COMPOSER + THINKING ROW
            # --------------------------------------------------------
            # The thinking indicator lives INSIDE the composer container,
            # immediately above the chat input. This guarantees that it is
            # visible exactly where the user expects it.
            with st.container(key="dm_center_composer", border=False):

                thinking_slot = st.empty()

                if st.session_state.docmind_is_thinking:
                    thinking_slot.markdown(
                        """
                        <div class="dm-thinking-inline">
                            <span class="dm-thinking-pulse"></span>
                            <span>🧠 DocMind is thinking…</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                submitted_question = st.chat_input(
                    "Ask anything about your documents…",
                    key="agent_v2_input",
                    max_chars=4000,
                    disabled=st.session_state.docmind_is_thinking,
                )

                st.markdown(
                    """
                    <div class="dm-agent-v2-disclaimer">
                        DocMind can make mistakes. Verify important information.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ------------------------------------------------------------
        # TWO-PASS SUBMISSION FLOW
        # ------------------------------------------------------------
        # Pass 1: store the question + set thinking=True + rerun.
        # That rerun renders "DocMind is thinking…" ABOVE the input.
        if submitted_question and not st.session_state.docmind_is_thinking:
            st.session_state.pending_question = submitted_question
            st.session_state.docmind_is_thinking = True
            st.rerun()

        # Pass 2: the thinking row is already rendered. Give the browser a
        # brief moment to paint it, then execute the blocking RAG/LLM call.
        if (
            st.session_state.docmind_is_thinking
            and st.session_state.pending_question
        ):
            pending_question = st.session_state.pending_question

            # Small paint window so the visible thinking row appears before
            # the blocking retrieval/generation work starts.
            time.sleep(0.30)

            try:
                ask_question(
                    pending_question,
                    top_k,
                )
            finally:
                st.session_state.pending_question = None
                st.session_state.docmind_is_thinking = False

            st.rerun()


# ============================================================
# RIGHT DOCUMENT INTELLIGENCE PANEL
# ============================================================

with right_panel:
    # This marker stays in the right column so all existing column-level
    # styling continues to target the Documents sidebar correctly.
    st.markdown(
        '<div class="dm-right-panel-marker" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )

    with st.container(key="dm_documents_dock", border=False):

        # ------------------------------------------------------------
        # FIXED DOCUMENTS HEADER
        # Heading + theme toggle + upload controls do NOT scroll.
        # ------------------------------------------------------------
        with st.container(key="dm_documents_fixed_header", border=False):
            st.markdown(
                '<div class="dm-right-panel-title">Documents</div>',
                unsafe_allow_html=True,
            )

            if workspace_document_manager:
                render_right_upload_controls(
                    workspace_document_manager,
                )

        # ------------------------------------------------------------
        # SCROLLABLE DOCUMENT CONTENT
        # Only this lower section receives its own vertical scrollbar.
        # ------------------------------------------------------------
        if workspace_document_manager:
            # The entire Documents dock behaves like a sidebar.
            # This lower container is intentionally NOT given its own height
            # or scrollbar; the outer Documents dock owns scrolling.
            with st.container(
                border=False,
                key="right_documents_scroll_container",
            ):
                render_right_documents_content(
                    workspace_document_manager,
                    st.session_state.last_result,
                )
        else:
            st.error(
                "Document services are unavailable right now. "
                "Please refresh the application."
            )


# ============================================================
# WORKSPACE-ONLY STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main layout */
    [data-testid="stMainBlockContainer"] .block-container {
        padding-left:1.25rem !important;
        padding-right:1.25rem !important;
    }

    /* Main buttons: readable in both themes. */
    [data-testid="stMain"] .stButton > button {
        min-height:40px !important;
        border-radius:10px !important;
        border:1px solid var(--dm-border) !important;
        background:var(--dm-surface2) !important;
        color:var(--dm-text) !important;
        font-weight:650 !important;
        box-shadow:none !important;
    }
    [data-testid="stMain"] .stButton > button:hover {
        border-color:var(--dm-primary) !important;
        background:var(--dm-soft) !important;
        color:var(--dm-heading) !important;
    }
    [data-testid="stMain"] .stButton > button[kind="primary"],
    [data-testid="stMain"] .stButton > button[data-testid="baseButton-primary"] {
        background:linear-gradient(135deg,var(--dm-primary),var(--dm-primary2)) !important;
        color:#fff !important;
        border-color:var(--dm-primary) !important;
    }

    /* Sidebar compactness and contrast */
    [data-testid="stSidebar"] .stButton > button {
        background:transparent !important;
        color:var(--dm-sidebar-text) !important;
        border:1px solid var(--dm-sidebar-border) !important;
        min-height:38px !important;
        border-radius:9px !important;
        font-weight:650 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background:var(--dm-sidebar-hover) !important;
        border-color:var(--dm-primary) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
        background:linear-gradient(135deg,var(--dm-primary),var(--dm-primary2)) !important;
        color:#fff !important;
        border-color:transparent !important;
    }
    [data-testid="stSidebar"] hr { margin:.8rem 0 !important; }

    /* Center conversation workspace */
    .dm-chat-box-title {
        color:var(--dm-heading) !important;
        font-size:1.12rem;
        font-weight:800;
        margin:0 0 2px 0;
    }
    .dm-chat-box-subtitle {
        color:var(--dm-muted) !important;
        font-size:.78rem;
        margin-bottom:8px;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker) {
        border:1px solid var(--dm-border) !important;
        border-radius:14px !important;
        background:var(--dm-surface) !important;
        box-shadow:0 8px 28px rgba(0,0,0,.06) !important;
        padding:2px !important;
    }
    .dm-empty-state {
        min-height:215px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        text-align:center;
        padding:24px 24px 18px;
        border:0;
        border-radius:12px;
        background:transparent;
    }
    .dm-empty-icon {
        width:52px;height:52px;border-radius:14px;
        display:flex;align-items:center;justify-content:center;
        background:var(--dm-soft);
        border:1px solid var(--dm-border);
        font-size:23px;margin-bottom:12px;
    }
    .dm-empty-title {
        color:var(--dm-heading);font-size:1.08rem;font-weight:800;
    }
    .dm-empty-text {
        color:var(--dm-muted);max-width:560px;line-height:1.5;
        margin-top:5px;font-size:.86rem;
    }
    [data-testid="stChatMessage"] {
        border:0 !important;
        border-bottom:1px solid var(--dm-border) !important;
        border-radius:0 !important;
        background:transparent !important;
        padding:12px 4px !important;
    }
    [data-testid="stChatMessage"]:last-child { border-bottom:0 !important; }
    [data-testid="stChatMessageContent"],
    [data-testid="stChatMessageContent"] p { color:var(--dm-text) !important; }
    [data-testid="stChatInput"] {
        border:1px solid var(--dm-border) !important;
        background:var(--dm-input) !important;
        border-radius:12px !important;
        box-shadow:none !important;
    }
    [data-testid="stChatInput"]:focus-within { border-color:var(--dm-primary) !important; }
    [data-testid="stChatInput"] textarea {
        background:transparent !important;color:var(--dm-input-text) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color:var(--dm-placeholder) !important; }

    /* Right Documents sidebar */
    .dm-right-panel-title {
        color:var(--dm-sidebar-text) !important;
        font-size:1.05rem;font-weight:800;padding:1px 0 7px 2px;
    }
    div[data-testid="column"]:has(.dm-right-panel-marker) {
        background:var(--dm-sidebar-bg) !important;
        border-left:1px solid var(--dm-sidebar-border) !important;
        border-radius:0 !important;
        padding:12px 12px 16px 16px !important;
        min-height:calc(100vh - 5.5rem) !important;
        position:sticky !important;
        top:4.2rem !important;
        box-sizing:border-box !important;
    }
    .dm-right-panel-marker { display:none !important; }
    .dm-right-section-title {
        color:var(--dm-sidebar-text) !important;
        font-size:.78rem;font-weight:800;margin:14px 0 7px;
    }
    .dm-right-empty {
        color:var(--dm-sidebar-muted) !important;font-size:.7rem;
        line-height:1.45;padding:7px 0 2px;
    }
    .dm-upload-selected {
        background:var(--dm-sidebar-hover) !important;
        border:1px solid var(--dm-sidebar-border) !important;
        border-radius:9px;padding:8px 9px;margin:6px 0;
    }
    .dm-upload-file {display:flex;gap:7px;align-items:center;color:var(--dm-sidebar-text)!important;font-size:.72rem;font-weight:700;overflow:hidden;}
    .dm-upload-file span:last-child {overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
    .dm-upload-meta {color:var(--dm-sidebar-muted)!important;font-size:.63rem;margin-top:3px;}
    .dm-doc-row,.dm-indexed-row {display:flex;align-items:center;justify-content:space-between;gap:7px;padding:8px 0;border-bottom:1px solid var(--dm-sidebar-border);}
    .dm-doc-main,.dm-indexed-main {display:flex;align-items:center;gap:7px;min-width:0;}
    .dm-doc-icon {color:var(--dm-sidebar-muted)!important;font-size:15px;flex:0 0 auto;}
    .dm-index-check {color:var(--dm-success)!important;font-size:14px;flex:0 0 auto;}
    .dm-doc-name-wrap {min-width:0;}
    .dm-doc-name {color:var(--dm-sidebar-text)!important;font-size:.7rem;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:175px;}
    .dm-doc-meta {color:var(--dm-sidebar-muted)!important;font-size:.61rem;margin-top:2px;}
    .dm-status {flex:0 0 auto;border-radius:999px;padding:3px 7px;font-size:.56rem;font-weight:800;border:1px solid transparent;}
    .dm-status.ready,.dm-status.indexed {color:var(--dm-success)!important;background:rgba(53,201,138,.10);border-color:rgba(53,201,138,.28);}
    .dm-kb-summary {display:flex;justify-content:space-between;gap:8px;color:var(--dm-sidebar-muted)!important;font-size:.63rem;margin:-1px 0 3px;}
    .dm-retrieved-card {background:rgba(53,201,138,.08)!important;border:1px solid rgba(53,201,138,.32)!important;border-radius:10px;padding:9px;margin:7px 0;}
    .dm-retrieved-title {display:flex;align-items:center;gap:7px;color:var(--dm-sidebar-text)!important;font-size:.71rem;font-weight:800;}
    .dm-retrieved-check {color:var(--dm-success)!important;font-size:.76rem;}
    .dm-retrieved-meta {color:var(--dm-sidebar-text)!important;opacity:.86;font-size:.63rem;margin-top:4px;}
    .dm-retrieved-score {display:flex;flex-wrap:wrap;gap:7px;color:var(--dm-sidebar-muted)!important;font-size:.59rem;margin-top:5px;}
    .dm-retrieved-score b {color:var(--dm-success)!important;}

    div[data-testid="column"]:has(.dm-right-panel-marker) [data-testid="stFileUploader"] {background:transparent!important;margin:0!important;}
    div[data-testid="column"]:has(.dm-right-panel-marker) [data-testid="stFileUploaderDropzone"] {
        background:var(--dm-sidebar-hover)!important;
        border:1px dashed var(--dm-sidebar-border)!important;
        border-radius:10px!important;min-height:76px!important;padding:8px!important;
    }
    div[data-testid="column"]:has(.dm-right-panel-marker) [data-testid="stFileUploaderDropzone"] * {color:var(--dm-sidebar-muted)!important;}
    div[data-testid="column"]:has(.dm-right-panel-marker) [data-testid="stFileUploaderDropzone"] button {
        background:var(--dm-soft)!important;border:1px solid var(--dm-border)!important;
        color:var(--dm-primary)!important;padding:5px 9px!important;min-height:30px!important;
    }
    div[data-testid="column"]:has(.dm-right-panel-marker) .stButton > button {
        min-height:32px!important;font-size:.65rem!important;
        background:var(--dm-sidebar-hover)!important;color:var(--dm-sidebar-text)!important;
        border-color:var(--dm-sidebar-border)!important;box-shadow:none!important;
    }
    div[data-testid="column"]:has(.dm-right-panel-marker) .stButton > button:hover {
        border-color:var(--dm-primary)!important;color:var(--dm-primary)!important;
    }

    /* Independent scroll containers */
    div[data-testid="stVerticalBlockBorderWrapper"] { scrollbar-width:thin; scrollbar-color:var(--dm-border) transparent; }
    div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar {width:7px;}
    div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-track {background:transparent;}
    div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb {background:var(--dm-border);border-radius:999px;}

    .dm-center-title {font-size:1.45rem;font-weight:800;color:var(--dm-heading);margin:0 0 3px;}
    .dm-center-subtitle {color:var(--dm-muted);font-size:.82rem;margin-bottom:15px;}

    @media (max-width:1100px) {
        div[data-testid="column"]:has(.dm-right-panel-marker) {position:static!important;min-height:0!important;margin-top:14px;}
        .dm-empty-state {min-height:190px;}
    }

    /* =========================================================
       FINAL DOCUMENT UPLOADER THEME
       Keep the uploader visually integrated with the right sidebar.
       ========================================================= */

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploader"] {{
        background: transparent !important;
        border: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] {{
        background: var(--dm-sidebar-hover) !important;
        border: 1px dashed var(--dm-sidebar-border) !important;
        border-radius: 12px !important;
        min-height: 96px !important;
        padding: 12px !important;
        box-shadow: none !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"]:hover {{
        background: var(--dm-surface2) !important;
        border-color: var(--dm-primary) !important;
    }}

    /* Force every nested uploader surface to inherit the dark sidebar
       instead of Streamlit's default white/light surface. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] > div,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] section,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        background: transparent !important;
        color: var(--dm-sidebar-text) !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] span,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] small,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] p,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] div {{
        color: var(--dm-sidebar-muted) !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] svg {{
        color: var(--dm-primary) !important;
        fill: none !important;
        opacity: 1 !important;
    }}

    /* Emerald browse/upload button in dark mode. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button {{
        background: var(--dm-primary) !important;
        border: 1px solid var(--dm-primary) !important;
        color: #ffffff !important;
        border-radius: 9px !important;
        min-height: 34px !important;
        padding: 6px 12px !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button *,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button p,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button span {{
        color: #ffffff !important;
        opacity: 1 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button:hover {{
        background: var(--dm-primary-hover) !important;
        border-color: var(--dm-primary-hover) !important;
        color: #ffffff !important;
    }}

    /* Uploaded-file rows generated by Streamlit should also stay dark. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderFile"] {{
        background: var(--dm-sidebar-hover) !important;
        border: 1px solid var(--dm-sidebar-border) !important;
        color: var(--dm-sidebar-text) !important;
        border-radius: 10px !important;
    }}

    /* Light mode: white uploader, neutral border, emerald action. */

    [data-testid="stFileUploaderDropzone"] {{
        background: #ffffff !important;
        border-color: #d9e2df !important;
    }}

    [data-testid="stFileUploaderDropzone"]:hover {{
        background: #f7faf9 !important;
        border-color: var(--dm-primary) !important;
    }}


    /* =========================================================
       COMPACT UPLOADER — MATCH RE-INDEX BUTTON / SIDEBAR THEME
       ========================================================= */

    /* Remove the large white drop-zone appearance. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    div[data-testid="stFileUploader"] {{
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
        margin: 0 0 8px 0 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    section[data-testid="stFileUploaderDropzone"],
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {THEME['surface2']} !important;
        background: {THEME['surface2']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 10px !important;
        min-height: 46px !important;
        height: auto !important;
        padding: 6px 8px !important;
        box-shadow: none !important;
    }}

    /* Collapse the large instruction area so this looks like an action control,
       not a white upload card. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        display: none !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    section[data-testid="stFileUploaderDropzone"] > div {{
        background: transparent !important;
        min-height: 0 !important;
        padding: 0 !important;
    }}

    /* Native file-picker button — same visual language as Re-index. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button {{
        width: 100% !important;
        min-height: 38px !important;
        margin: 0 !important;
        padding: 7px 12px !important;
        background: {THEME['surface2']} !important;
        color: {THEME['text']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 9px !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button *,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button p,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button span {{
        color: {THEME['text']} !important;
        opacity: 1 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button:hover {{
        background: {THEME['surface3']} !important;
        color: {THEME['primary']} !important;
        border-color: {THEME['primary']} !important;
    }}

    /* Selected-file row follows the same compact dark surface. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderFile"] {{
        background: {THEME['surface2']} !important;
        color: {THEME['text']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 9px !important;
        box-shadow: none !important;
    }}

    /* Submit button directly after the uploader matches Re-index too. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    .stButton > button[kind="secondary"],
    div[data-testid="column"]:has(.dm-right-panel-marker)
    .stButton > button {{
        background: {THEME['surface2']} !important;
        color: {THEME['text']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 9px !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    .stButton > button:not(:disabled):hover {{
        background: {THEME['surface3']} !important;
        color: {THEME['primary']} !important;
        border-color: {THEME['primary']} !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    .stButton > button:disabled {{
        background: {THEME['surface2']} !important;
        color: {THEME['muted']} !important;
        border-color: {THEME['border']} !important;
        opacity: .55 !important;
    }}

    .dm-upload-selected {{
        background: {THEME['surface2']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 9px !important;
        padding: 8px 10px !important;
        margin: 6px 0 8px !important;
    }}


    /* =========================================================
       FINAL UPLOAD STYLE REQUEST
       Dark mode: black upload box + New Chat-style green button.
       Light mode: white upload box + same green action button.
       ========================================================= */

    div[data-testid="column"]:has(.dm-right-panel-marker)
    div[data-testid="stFileUploader"] {{
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
        margin: 0 0 8px 0 !important;
    }}

    /* Force the entire native dropzone to black in dark mode. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    section[data-testid="stFileUploaderDropzone"],
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] {{
        background-color: #050806 !important;
        background: #050806 !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 10px !important;
        min-height: 56px !important;
        height: auto !important;
        padding: 8px !important;
        box-shadow: none !important;
    }}

    /* Also force nested Streamlit uploader surfaces to black. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    section[data-testid="stFileUploaderDropzone"] > div,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] > div,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        background-color: #050806 !important;
        background: #050806 !important;
    }}

    /* Keep the large helper text hidden for the compact layout. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        display: none !important;
    }}

    /* Upload/Browse button = same visual treatment as + New Chat. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button {{
        width: 100% !important;
        min-height: 40px !important;
        margin: 0 !important;
        padding: 8px 14px !important;
        background: {THEME['primary']} !important;
        background-color: {THEME['primary']} !important;
        color: #ffffff !important;
        border: 1px solid {THEME['primary']} !important;
        border-radius: 9px !important;
        box-shadow: none !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button *,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button p,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button span,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button svg {{
        color: #ffffff !important;
        fill: #ffffff !important;
        opacity: 1 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button:hover {{
        background: {THEME['primary_hover']} !important;
        background-color: {THEME['primary_hover']} !important;
        border-color: {THEME['primary_hover']} !important;
        color: #ffffff !important;
    }}

    /* Light mode stays white, while its Upload button remains green. */

    [data-testid="stFileUploaderDropzone"],

    [data-testid="stFileUploaderDropzone"] > div {{
        background: #ffffff !important;
        background-color: #ffffff !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL NON-OVERLAPPING SPACING SAFEGUARD
# ============================================================

st.markdown(
    """
    <style>
    /*
      Keep the existing visual design intact while making every panel size
      itself from its border box. This prevents padding from being added on
      top of the declared width/height and pushing content into a neighbour.
    */
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] *,
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        box-sizing: border-box !important;
    }

    /* Reserve real space for Streamlit's fixed toolbar. */
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] .block-container,
    .main .block-container,
    .block-container {
        width: 100% !important;
        max-width: 1600px !important;
        padding-top: 4.65rem !important;
        padding-bottom: .75rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Use the visible dynamic viewport so the bottom composer cannot escape. */
    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {
        height: calc(100dvh - 5.4rem) !important;
        max-height: calc(100dvh - 5.4rem) !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        align-items: stretch !important;
        overflow: hidden !important;
    }

    /* Allow Streamlit's flex columns to shrink without overlapping. */
    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker),
    div[data-testid="column"]:has(.dm-agent-v2-column-marker),
    div[data-testid="column"]:has(.dm-right-panel-marker),
    div[data-testid="column"]:has(.dm-analytics-scroll-marker),
    div[data-testid="column"]:has(.dm-kb-scroll-marker) {
        min-width: 0 !important;
    }

    div[data-testid="column"]:has(.dm-agent-v2-column-marker),
    div[data-testid="column"]:has(.dm-right-panel-marker) {
        padding-bottom: 0 !important;
    }

    /* Keep the AI heading, conversation, and composer in separate regions. */
    .dm-agent-v2-header {
        flex-shrink: 0 !important;
        padding-left: .5rem !important;
        padding-right: .5rem !important;
    }

    .st-key-agent_v2_scroll,
    div[class*="st-key-agent_v2_scroll"] {
        min-width: 0 !important;
        padding: .25rem .75rem .8rem .35rem !important;
        scroll-padding-bottom: .8rem !important;
    }

    .st-key-agent_v2_scroll > div,
    div[class*="st-key-agent_v2_scroll"] > div,
    .st-key-agent_v2_scroll [data-testid="stVerticalBlock"],
    div[class*="st-key-agent_v2_scroll"] [data-testid="stVerticalBlock"] {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
    }

    .st-key-agent_v2_scroll p,
    .st-key-agent_v2_scroll li,
    div[class*="st-key-agent_v2_scroll"] p,
    div[class*="st-key-agent_v2_scroll"] li,
    .dm-doc-name,
    .dm-retrieved-title span {
        overflow-wrap: anywhere !important;
        word-break: normal !important;
    }

    .st-key-agent_v2_scroll pre,
    .st-key-agent_v2_scroll table,
    div[class*="st-key-agent_v2_scroll"] pre,
    div[class*="st-key-agent_v2_scroll"] table {
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    /*
      The composer remains at the bottom because it is the non-shrinking final
      flex item. Relative positioning prevents it from covering chat content.
    */
    .st-key-agent_v2_input,
    div[class*="st-key-agent_v2_input"] {
        flex: 0 0 auto !important;
        position: relative !important;
        inset: auto !important;
        width: auto !important;
        max-width: calc(100% - 1.3rem) !important;
        margin: .35rem .65rem 0 !important;
        padding: .4rem 0 .1rem !important;
    }

    .st-key-agent_v2_input [data-testid="stChatInput"],
    div[class*="st-key-agent_v2_input"] [data-testid="stChatInput"],
    .st-key-agent_v2_input textarea,
    div[class*="st-key-agent_v2_input"] textarea {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    .st-key-agent_v2_input textarea,
    div[class*="st-key-agent_v2_input"] textarea {
        padding: 15px 56px 13px 15px !important;
    }

    .dm-agent-v2-disclaimer {
        flex-shrink: 0 !important;
        padding-left: .65rem !important;
        padding-right: .65rem !important;
    }

    /* Preserve comfortable spacing in the independently scrolling panel. */
    .st-key-right_documents_scroll_container,
    div[class*="st-key-right_documents_scroll_container"] {
        min-width: 0 !important;
        width: 100% !important;
        padding-left: .1rem !important;
        padding-right: .45rem !important;
        padding-bottom: .75rem !important;
    }

    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"],
        [data-testid="stMainBlockContainer"] .block-container,
        .main .block-container,
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        div[data-testid="column"]:has(.dm-agent-v2-column-marker),
        div[data-testid="column"]:has(.dm-right-panel-marker) {
            width: 100% !important;
            min-width: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        .st-key-agent_v2_scroll,
        div[class*="st-key-agent_v2_scroll"] {
            padding-left: .15rem !important;
            padding-right: .35rem !important;
        }

        .st-key-agent_v2_input,
        div[class*="st-key-agent_v2_input"] {
            max-width: 100% !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
    }

    @media (max-width: 600px) {
        [data-testid="stMainBlockContainer"],
        [data-testid="stMainBlockContainer"] .block-container,
        .main .block-container,
        .block-container {
            padding-left: .7rem !important;
            padding-right: .7rem !important;
        }

        .dm-agent-v2-header {
            padding-left: .25rem !important;
            padding-right: .25rem !important;
        }

        .st-key-agent_v2_input textarea,
        div[class*="st-key-agent_v2_input"] textarea {
            padding-right: 52px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINAL UI OVERRIDES
# ============================================================

UPLOAD_BG = "#050806" if st.session_state.theme == "dark" else "#ffffff"
UPLOAD_BORDER = "#26352f" if st.session_state.theme == "dark" else "#d8e0dc"
UPLOAD_TEXT = "#f4faf7" if st.session_state.theme == "dark" else "#172b25"

st.markdown(
    f"""
    <style>
    /* Remove leftover top spacing after deleting RAG Assistant. */
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] .block-container,
    .main .block-container,
    .block-container {{
        padding-top: 4.1rem !important;
    }}

    /* ========================================================
       CHATGPT-LIKE FIXED ASSISTANT
       The overall bot area stays fixed.
       Only the dedicated conversation history scrolls.
       ======================================================== */

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker) {{
        height: 610px !important;
        min-height: 610px !important;
        max-height: 610px !important;
        overflow: hidden !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 14px !important;
        background: var(--dm-surface) !important;
        box-shadow: none !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker)
    > div {{
        height: 100% !important;
        max-height: 100% !important;
        overflow: hidden !important;
    }}

    .dm-empty-state {{
        min-height: 250px !important;
        max-height: 250px !important;
        padding: 32px 24px 20px !important;
        overflow: hidden !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker)
    [data-testid="stChatInput"] {{
        margin-top: 10px !important;
        margin-bottom: 0 !important;
        flex-shrink: 0 !important;
    }}

    /* Dedicated conversation scroller. */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-message-scroll-marker) {{
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-message-scroll-marker)
   ::-webkit-scrollbar {{
        width: 7px !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-message-scroll-marker)
   ::-webkit-scrollbar-thumb {{
        background: var(--dm-border) !important;
        border-radius: 999px !important;
    }}

    /* ========================================================
       UPLOAD AREA
       Dark mode: TRUE BLACK.
       Light mode: WHITE.
       Upload button: same green treatment as New Chat.
       ======================================================== */

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploader"] {{
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
        margin: 0 0 8px 0 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    section[data-testid="stFileUploaderDropzone"],
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"],
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] > div,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] section {{
        background: {UPLOAD_BG} !important;
        background-color: {UPLOAD_BG} !important;
        box-shadow: none !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    section[data-testid="stFileUploaderDropzone"],
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] {{
        border: 1px solid {UPLOAD_BORDER} !important;
        border-radius: 10px !important;
        min-height: 58px !important;
        height: auto !important;
        padding: 8px !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        display: none !important;
    }}

    /* Green Upload button — same family as + New Chat. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button {{
        width: 100% !important;
        min-height: 40px !important;
        margin: 0 !important;
        padding: 8px 14px !important;
        background: linear-gradient(
            135deg,
            var(--dm-primary),
            var(--dm-primary2)
        ) !important;
        background-color: var(--dm-primary) !important;
        color: #ffffff !important;
        border: 1px solid var(--dm-primary) !important;
        border-radius: 9px !important;
        box-shadow: none !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button *,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button p,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button span {{
        color: #ffffff !important;
        opacity: 1 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button svg {{
        color: #ffffff !important;
        stroke: #ffffff !important;
        opacity: 1 !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] button:hover {{
        filter: brightness(.94) !important;
        border-color: var(--dm-primary) !important;
        color: #ffffff !important;
    }}

    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderFile"],
    .dm-upload-selected {{
        background: {UPLOAD_BG} !important;
        color: {UPLOAD_TEXT} !important;
        border: 1px solid {UPLOAD_BORDER} !important;
        border-radius: 9px !important;
        box-shadow: none !important;
    }}

    /* Make sure nested uploader text/icons never create a white surface. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] div,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] span,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploaderDropzone"] p {{
        background-color: transparent !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINAL FIXED-VIEWPORT + UPLOADER OVERRIDES
# ============================================================

DOCMIND_UPLOAD_BG = (
    "#050806"
    if st.session_state.theme == "dark"
    else "#ffffff"
)
DOCMIND_UPLOAD_TEXT = (
    "#f4faf7"
    if st.session_state.theme == "dark"
    else "#172b25"
)
DOCMIND_UPLOAD_MUTED = (
    "#93a59e"
    if st.session_state.theme == "dark"
    else "#63756e"
)
DOCMIND_UPLOAD_BORDER = (
    "#25342e"
    if st.session_state.theme == "dark"
    else "#d5dfda"
)

st.markdown(
    f"""
    <style>
    /* ========================================================
       1. FIX THE COMPLETE APPLICATION TO THE VIEWPORT
       No browser/page scrolling.
       ======================================================== */

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"] {{
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }}

    [data-testid="stMain"] {{
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }}

    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] .block-container,
    .main .block-container,
    .block-container {{
        height: calc(100vh - 3.9rem) !important;
        max-height: calc(100vh - 3.9rem) !important;
        min-height: 0 !important;
        overflow: hidden !important;
        padding-top: 1.35rem !important;
        padding-bottom: .8rem !important;
        box-sizing: border-box !important;
    }}

    /* Left sidebar remains fixed, but its CONTENT scrolls independently. */
    [data-testid="stSidebar"] {{
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }}

    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {{
        height: 100% !important;
        max-height: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
    }}

    /* The two-column workspace itself never scrolls the page. */
    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {{
        height: calc(100vh - 6rem) !important;
        max-height: calc(100vh - 6rem) !important;
        min-height: 0 !important;
        overflow: hidden !important;
        align-items: stretch !important;
    }}

    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker)
    > div[data-testid="column"] {{
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }}

    /* ========================================================
       2. CENTER CONTENT SCROLL RULES
       Analytics/Evaluation and Knowledge Base scroll internally.
       Chatbot shell stays fixed.
       ======================================================== */

    div[data-testid="column"]:has(.dm-analytics-scroll-marker),
    div[data-testid="column"]:has(.dm-kb-scroll-marker) {{
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 8px !important;
        scrollbar-width: thin !important;
    }}

    div[data-testid="column"]:has(.dm-analytics-scroll-marker)
    ::-webkit-scrollbar,
    div[data-testid="column"]:has(.dm-kb-scroll-marker)
    ::-webkit-scrollbar {{
        width: 7px !important;
    }}

    div[data-testid="column"]:has(.dm-analytics-scroll-marker)
    ::-webkit-scrollbar-thumb,
    div[data-testid="column"]:has(.dm-kb-scroll-marker)
    ::-webkit-scrollbar-thumb {{
        background: var(--dm-border) !important;
        border-radius: 999px !important;
    }}

    /* Fixed ChatGPT-like chatbot shell. */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker) {{
        height: calc(100vh - 6.35rem) !important;
        max-height: calc(100vh - 6.35rem) !important;
        min-height: 0 !important;
        overflow: hidden !important;
        display: block !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 14px !important;
        background: var(--dm-surface) !important;
    }}

    /* Never let the empty assistant/welcome state create a scrollbar. */
    .dm-empty-state {{
        min-height: 225px !important;
        max-height: 225px !important;
        padding: 28px 22px 18px !important;
        overflow: hidden !important;
    }}

    /* ONLY the real completed conversation container can scroll. */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-message-scroll-marker) {{
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-message-scroll-marker)
    ::-webkit-scrollbar {{
        width: 7px !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-message-scroll-marker)
    ::-webkit-scrollbar-thumb {{
        background: var(--dm-border) !important;
        border-radius: 999px !important;
    }}

    /* Keep the chat composer visible as a fixed part of the bot shell. */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker)
    [data-testid="stChatInput"] {{
        flex: 0 0 auto !important;
        margin-top: 8px !important;
        margin-bottom: 0 !important;
    }}

    /* ========================================================
       3. RIGHT DOCUMENT SIDEBAR
       The column is fixed; its native container scrolls.
       ======================================================== */

    div[data-testid="column"]:has(.dm-right-panel-marker) {{
        height: 100% !important;
        max-height: 100% !important;
        overflow: hidden !important;
        position: relative !important;
        top: auto !important;
    }}

    .st-key-right_documents_scroll_container,
    div[class*="st-key-right_documents_scroll_container"] {{
        height: calc(100vh - 9.2rem) !important;
        max-height: calc(100vh - 9.2rem) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
    }}

    /* ========================================================
       4. FILE UPLOADER — FORCE BLACK IN DARK MODE
       This deliberately targets several Streamlit DOM variants.
       ======================================================== */

    [data-testid="stFileUploader"],
    .stFileUploader,
    div[class*="stFileUploader"] {{
        background: transparent !important;
        background-color: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }}

    /* Force the actual native uploader/dropzone AND every internal
       non-button surface to the DocMind upload background. */
    [data-testid="stFileUploader"] section,
    .stFileUploader section,
    div[class*="stFileUploader"] section {{
        background: {DOCMIND_UPLOAD_BG} !important;
        background-color: {DOCMIND_UPLOAD_BG} !important;
        border: 1px solid {DOCMIND_UPLOAD_BORDER} !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        color: {DOCMIND_UPLOAD_TEXT} !important;
    }}

    [data-testid="stFileUploader"] section > div,
    [data-testid="stFileUploader"] section > div > div,
    .stFileUploader section > div,
    div[class*="stFileUploader"] section > div {{
        background: {DOCMIND_UPLOAD_BG} !important;
        background-color: {DOCMIND_UPLOAD_BG} !important;
        color: {DOCMIND_UPLOAD_TEXT} !important;
    }}

    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        background: {DOCMIND_UPLOAD_BG} !important;
        background-color: {DOCMIND_UPLOAD_BG} !important;
        color: {DOCMIND_UPLOAD_TEXT} !important;
    }}

    [data-testid="stFileUploader"] section p,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small {{
        color: {DOCMIND_UPLOAD_MUTED} !important;
        opacity: 1 !important;
    }}

    /* Upload/Browse button = EXACT green action family used by New Chat. */
    [data-testid="stFileUploader"] section button,
    .stFileUploader section button,
    div[class*="stFileUploader"] section button,
    [data-testid="stFileUploader"] button[data-testid^="stBaseButton"] {{
        background: linear-gradient(
            135deg,
            var(--dm-primary),
            var(--dm-primary2)
        ) !important;
        background-color: var(--dm-primary) !important;
        color: #ffffff !important;
        border: 1px solid var(--dm-primary) !important;
        border-radius: 9px !important;
        min-height: 40px !important;
        padding: 8px 15px !important;
        box-shadow: none !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }}

    [data-testid="stFileUploader"] section button *,
    .stFileUploader section button *,
    div[class*="stFileUploader"] section button * {{
        color: #ffffff !important;
        stroke: #ffffff !important;
        opacity: 1 !important;
    }}

    [data-testid="stFileUploader"] section button:hover,
    .stFileUploader section button:hover {{
        filter: brightness(.93) !important;
        border-color: var(--dm-primary) !important;
        color: #ffffff !important;
    }}

    /* Native selected-file row also follows the same theme. */
    [data-testid="stFileUploaderFile"],
    .dm-upload-selected {{
        background: {DOCMIND_UPLOAD_BG} !important;
        background-color: {DOCMIND_UPLOAD_BG} !important;
        color: {DOCMIND_UPLOAD_TEXT} !important;
        border: 1px solid {DOCMIND_UPLOAD_BORDER} !important;
        border-radius: 9px !important;
        box-shadow: none !important;
    }}

    /* Remove any residual Streamlit white background from uploader wrappers. */
    div[data-testid="column"]:has(.dm-right-panel-marker)
    [data-testid="stFileUploader"] div:not([role="progressbar"]) {{
        box-shadow: none !important;
    }}

    /* Responsive safety: allow page flow only on genuinely small screens. */
    @media (max-width: 900px) {{
        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {{
            height: auto !important;
            max-height: none !important;
            overflow: auto !important;
        }}

        [data-testid="stMainBlockContainer"],
        .block-container {{
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }}

        div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {{
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FULL-HEIGHT CENTER PANEL OVERRIDES
# ============================================================

st.markdown(
    """
    <style>
    /* Use almost the complete available desktop height. */
    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {
        height: calc(100vh - 4.75rem) !important;
        max-height: calc(100vh - 4.75rem) !important;
        min-height: 0 !important;
        overflow: hidden !important;
        align-items: stretch !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker)
    > div[data-testid="column"] {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
    }

    /* The center Chat column itself is fixed and fills the workspace height. */
    div[data-testid="column"]:has(.dm-chat-column-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }

    /* Stretch the keyed chatbot container to the bottom of the center panel. */
    .st-key-ai_chat_workspace_box,
    div[class*="st-key-ai_chat_workspace_box"] {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }

    .st-key-ai_chat_workspace_box
    [data-testid="stVerticalBlockBorderWrapper"],
    div[class*="st-key-ai_chat_workspace_box"]
    [data-testid="stVerticalBlockBorderWrapper"] {
        max-height: 100% !important;
    }

    /* The visible assistant box fills the available center height. */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
    }

    /* Make the main vertical content inside the assistant behave like
       a fixed chat app so the composer can remain at the bottom. */
    .st-key-ai_chat_workspace_box > div,
    div[class*="st-key-ai_chat_workspace_box"] > div {
        height: 100% !important;
        min-height: 0 !important;
    }

    /* Give the empty state more vertical room so the chat section
       visually fills the blank area instead of ending too early. */
    .dm-empty-state {
        min-height: 300px !important;
        max-height: 300px !important;
        padding-top: 48px !important;
        padding-bottom: 28px !important;
    }

    /* Keep chat input at the bottom area of the fixed assistant. */
    .st-key-ai_chat_workspace_box [data-testid="stChatInput"],
    div[class*="st-key-ai_chat_workspace_box"] [data-testid="stChatInput"] {
        margin-top: 12px !important;
        margin-bottom: 4px !important;
        flex-shrink: 0 !important;
    }

    /* Only the actual conversation history scrolls. */
    .st-key-chat_history_scroll_container,
    div[class*="st-key-agent_v2_scroll"] {
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
    }

    /* Analytics & Evaluation gets its own independent scrollbar. */
    .st-key-analytics_scroll_container,
    div[class*="st-key-analytics_scroll_container"] {
        height: calc(100vh - 5.15rem) !important;
        max-height: calc(100vh - 5.15rem) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 8px !important;
        scrollbar-width: thin !important;
    }

    /* Knowledge Base gets its own independent scrollbar. */
    .st-key-knowledge_scroll_container,
    div[class*="st-key-knowledge_scroll_container"] {
        height: calc(100vh - 5.15rem) !important;
        max-height: calc(100vh - 5.15rem) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 8px !important;
        scrollbar-width: thin !important;
    }

    .st-key-analytics_scroll_container::-webkit-scrollbar,
    .st-key-knowledge_scroll_container::-webkit-scrollbar,
    .st-key-chat_history_scroll_container::-webkit-scrollbar {
        width: 7px !important;
    }

    .st-key-analytics_scroll_container::-webkit-scrollbar-thumb,
    .st-key-knowledge_scroll_container::-webkit-scrollbar-thumb,
    .st-key-chat_history_scroll_container::-webkit-scrollbar-thumb {
        background: var(--dm-border) !important;
        border-radius: 999px !important;
    }

    /* Right sidebar stays independently scrollable while matching
       the increased center height. */
    .st-key-right_documents_scroll_container,
    div[class*="st-key-right_documents_scroll_container"] {
        height: calc(100vh - 7.4rem) !important;
        max-height: calc(100vh - 7.4rem) !important;
    }

    @media (max-width: 900px) {
        .st-key-ai_chat_workspace_box,
        div[class*="st-key-ai_chat_workspace_box"],
        .st-key-analytics_scroll_container,
        div[class*="st-key-analytics_scroll_container"],
        .st-key-knowledge_scroll_container,
        div[class*="st-key-knowledge_scroll_container"] {
            height: auto !important;
            max-height: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINAL CHAT UX + SIDEBAR HISTORY FIXES
# ============================================================

st.markdown(
    """
    <style>
    /* --------------------------------------------------------
       LEFT SIDEBAR CHAT HISTORY
       Prevent Streamlit focus/hover states from turning history
       buttons white with unreadable text.
       -------------------------------------------------------- */

    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button:hover,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button:focus,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button:focus-visible,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button:active {
        background: var(--dm-sidebar-hover) !important;
        background-color: var(--dm-sidebar-hover) !important;
        color: var(--dm-sidebar-text) !important;
        border: 1px solid var(--dm-sidebar-border) !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button:hover {
        border-color: var(--dm-primary) !important;
    }

    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button *,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button p,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button span,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button div {
        color: var(--dm-sidebar-text) !important;
        opacity: 1 !important;
        visibility: visible !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* --------------------------------------------------------
       CHATGPT-LIKE CENTER PANEL
       Outer bot is fixed.
       Header is fixed.
       Only message history scrolls.
       Composer remains fixed at bottom.
       -------------------------------------------------------- */

    div[data-testid="column"]:has(.dm-chat-column-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }

    .st-key-ai_chat_workspace_box,
    div[class*="st-key-ai_chat_workspace_box"] {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }

    /* The bordered assistant wrapper fills the center panel. */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
    }

    /* Make the assistant's inner vertical block a flex column. */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker)
    > div > [data-testid="stVerticalBlock"] {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
    }

    /* Conversation history expands to all remaining space. */
    .st-key-chat_history_scroll_container,
    div[class*="st-key-agent_v2_scroll"] {
        flex: 1 1 auto !important;
        height: auto !important;
        max-height: none !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
        padding-right: 5px !important;
        overscroll-behavior: contain !important;
    }

    .st-key-chat_history_scroll_container::-webkit-scrollbar,
    div[class*="st-key-agent_v2_scroll"]::-webkit-scrollbar {
        width: 7px !important;
    }

    .st-key-chat_history_scroll_container::-webkit-scrollbar-thumb,
    div[class*="st-key-agent_v2_scroll"]::-webkit-scrollbar-thumb {
        background: var(--dm-border) !important;
        border-radius: 999px !important;
    }

    /* Chat turns flow normally from top to bottom.
       Auto-scroll brings the latest turn into view, so older chats move up. */
    .st-key-chat_history_scroll_container
    [data-testid="stVerticalBlock"] {
        height: auto !important;
        min-height: min-content !important;
        overflow: visible !important;
    }

    /* The prompt composer must never join the scrolling message pane. */
    .st-key-docmind_chat_input,
    div[class*="st-key-docmind_chat_input"] {
        flex: 0 0 auto !important;
        position: sticky !important;
        bottom: 0 !important;
        z-index: 20 !important;
        background: var(--dm-surface) !important;
        padding-top: 8px !important;
        margin-top: 4px !important;
    }

    .st-key-docmind_chat_input [data-testid="stChatInput"],
    div[class*="st-key-docmind_chat_input"] [data-testid="stChatInput"] {
        margin: 0 !important;
    }

    /* Keep header content from shrinking when chat grows. */
    .dm-chat-box-marker,
    .dm-chat-box-title,
    .dm-chat-box-subtitle {
        flex: 0 0 auto !important;
    }

    /* Empty-state layout stays static; no unnecessary scrollbar. */
    .dm-empty-state {
        flex: 0 0 auto !important;
        overflow: hidden !important;
    }

    @media (max-width: 900px) {
        .st-key-docmind_chat_input,
        div[class*="st-key-docmind_chat_input"] {
            position: static !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINAL SIDEBAR + CHAT LAYOUT RECOVERY
# ============================================================

st.markdown(
    """
    <style>
    /* Sidebar conversation buttons must always be readable. */
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button {
        background: var(--dm-sidebar-hover) !important;
        background-color: var(--dm-sidebar-hover) !important;
        border: 1px solid var(--dm-sidebar-border) !important;
        color: var(--dm-sidebar-text) !important;
        opacity: 1 !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button:hover,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button:focus,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button:active {
        background: var(--dm-sidebar-hover) !important;
        color: var(--dm-sidebar-text) !important;
        border-color: var(--dm-primary) !important;
    }

    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button *,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button p,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button span {
        color: var(--dm-sidebar-text) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* ========================================================
       CHATBOT
       Reset the flex rules that hid DocMind/title/chat content.
       ======================================================== */

    div[data-testid="column"]:has(.dm-chat-column-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }

    .st-key-ai_chat_workspace_box,
    div[class*="st-key-ai_chat_workspace_box"] {
        height: calc(100vh - 5.35rem) !important;
        max-height: calc(100vh - 5.35rem) !important;
        min-height: 0 !important;
        overflow: hidden !important;
        display: block !important;
    }

    /* Restore normal block flow inside the assistant.
       Header -> scrollable messages -> prompt input. */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker),
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker) > div,
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker)
    > div > [data-testid="stVerticalBlock"] {
        display: block !important;
        flex: none !important;
        min-height: 0 !important;
        overflow: visible !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker) {
        height: 100% !important;
        max-height: 100% !important;
        overflow: hidden !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 14px !important;
        background: var(--dm-surface) !important;
    }

    /* DocMind heading and subtitle remain visible at the top. */
    .dm-chat-box-marker,
    .dm-chat-box-title,
    .dm-chat-box-subtitle {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: relative !important;
        z-index: 2 !important;
    }

    /* ========================================================
       ONLY CHAT HISTORY SCROLLS
       ======================================================== */

    .st-key-chat_history_scroll_container,
    div[class*="st-key-agent_v2_scroll"] {
        height: calc(100vh - 14rem) !important;
        max-height: calc(100vh - 14rem) !important;
        min-height: 240px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        overscroll-behavior: contain !important;
        scrollbar-width: thin !important;
        padding-right: 5px !important;
    }

    .st-key-chat_history_scroll_container
    [data-testid="stVerticalBlock"],
    div[class*="st-key-agent_v2_scroll"]
    [data-testid="stVerticalBlock"] {
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        display: block !important;
    }

    .st-key-chat_history_scroll_container::-webkit-scrollbar,
    div[class*="st-key-agent_v2_scroll"]::-webkit-scrollbar {
        width: 7px !important;
    }

    .st-key-chat_history_scroll_container::-webkit-scrollbar-thumb,
    div[class*="st-key-agent_v2_scroll"]::-webkit-scrollbar-thumb {
        background: var(--dm-border) !important;
        border-radius: 999px !important;
    }

    /* ========================================================
       PROMPT COMPOSER — FIXED OUTSIDE THE CHAT SCROLLER
       ======================================================== */

    .st-key-docmind_chat_input,
    div[class*="st-key-docmind_chat_input"] {
        display: block !important;
        visibility: visible !important;
        position: relative !important;
        bottom: auto !important;
        flex: none !important;
        z-index: 5 !important;
        margin-top: 8px !important;
        padding-top: 6px !important;
        background: var(--dm-surface) !important;
    }

    .st-key-docmind_chat_input [data-testid="stChatInput"],
    div[class*="st-key-docmind_chat_input"] [data-testid="stChatInput"] {
        position: relative !important;
        bottom: auto !important;
        margin: 0 !important;
    }

    /* Empty/new chat state stays visible and does not scroll. */
    .dm-empty-state {
        display: block !important;
        visibility: visible !important;
        min-height: 275px !important;
        max-height: 275px !important;
        overflow: hidden !important;
    }

    /* Analytics and Knowledge Base remain independently scrollable. */
    .st-key-analytics_scroll_container,
    div[class*="st-key-analytics_scroll_container"],
    .st-key-knowledge_scroll_container,
    div[class*="st-key-knowledge_scroll_container"] {
        overflow-y: auto !important;
        overflow-x: hidden !important;
        height: calc(100vh - 5.2rem) !important;
        max-height: calc(100vh - 5.2rem) !important;
    }

    /* Right Documents sidebar remains independent too. */
    .st-key-right_documents_scroll_container,
    div[class*="st-key-right_documents_scroll_container"] {
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }

    @media (max-width: 900px) {
        section[data-testid="stSidebar"],
        [data-testid="stSidebar"] {
            min-width: unset !important;
            width: unset !important;
            max-width: unset !important;
        }

        .st-key-ai_chat_workspace_box,
        div[class*="st-key-ai_chat_workspace_box"],
        .st-key-chat_history_scroll_container,
        div[class*="st-key-agent_v2_scroll"] {
            height: auto !important;
            max-height: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINAL CHAT LAYOUT — RESTORE CONTENT + FIX COMPOSER AT BOTTOM
# ============================================================

st.markdown(
    """
    <style>
    /* ========================================================
       CHAT SHELL
       Header + welcome/messages stay visible.
       Only message history scrolls.
       Composer stays fixed at the bottom of the chatbot.
       ======================================================== */

    div[data-testid="column"]:has(.dm-chat-column-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }

    .st-key-ai_chat_workspace_box,
    div[class*="st-key-ai_chat_workspace_box"] {
        height: calc(100vh - 5.25rem) !important;
        max-height: calc(100vh - 5.25rem) !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 14px !important;
        background: var(--dm-surface) !important;
        box-sizing: border-box !important;
    }

    /*
      Streamlit places the chatbot elements inside this vertical block.
      Make THAT block the flex column:
        title
        subtitle
        welcome OR scrollable messages
        errors if any
        prompt input
    */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker)
    > div
    > [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
        flex: 1 1 auto !important;
    }

    /* Restore the content that was being clipped. */
    .dm-chat-box-marker,
    .dm-chat-box-title,
    .dm-chat-box-subtitle,
    .dm-empty-state,
    .dm-empty-icon,
    .dm-empty-title,
    .dm-empty-text {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    .dm-chat-box-title {
        flex: 0 0 auto !important;
        margin-top: 0 !important;
        margin-bottom: 2px !important;
    }

    .dm-chat-box-subtitle {
        flex: 0 0 auto !important;
        margin-bottom: 8px !important;
    }

    /* New-chat welcome content is visible and occupies the middle area. */
    .dm-empty-state {
        flex: 0 0 auto !important;
        min-height: 255px !important;
        max-height: 255px !important;
        padding: 38px 24px 20px !important;
        overflow: hidden !important;
    }

    /* ========================================================
       MESSAGE HISTORY
       This is the ONLY part of the chatbot allowed to scroll.
       ======================================================== */

    .st-key-chat_history_scroll_container,
    div[class*="st-key-agent_v2_scroll"] {
        flex: 1 1 auto !important;
        height: auto !important;
        max-height: none !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        overscroll-behavior: contain !important;
        scrollbar-width: thin !important;
        padding-right: 6px !important;
        margin-bottom: 6px !important;
    }

    .st-key-chat_history_scroll_container
    > div,
    div[class*="st-key-agent_v2_scroll"]
    > div {
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
        overflow: visible !important;
    }

    .st-key-chat_history_scroll_container
    [data-testid="stVerticalBlock"],
    div[class*="st-key-agent_v2_scroll"]
    [data-testid="stVerticalBlock"] {
        display: block !important;
        height: auto !important;
        min-height: min-content !important;
        max-height: none !important;
        overflow: visible !important;
    }

    .st-key-chat_history_scroll_container::-webkit-scrollbar,
    div[class*="st-key-agent_v2_scroll"]::-webkit-scrollbar {
        width: 7px !important;
    }

    .st-key-chat_history_scroll_container::-webkit-scrollbar-thumb,
    div[class*="st-key-agent_v2_scroll"]::-webkit-scrollbar-thumb {
        background: var(--dm-border) !important;
        border-radius: 999px !important;
    }

    /* ========================================================
       PROMPT INPUT
       It is NOT inside the chat-history scroller.
       Push it to the bottom of the fixed chatbot.
       ======================================================== */

    .st-key-docmind_chat_input,
    div[class*="st-key-docmind_chat_input"] {
        flex: 0 0 auto !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: relative !important;
        inset: auto !important;
        margin-top: auto !important;
        margin-bottom: 0 !important;
        padding-top: 8px !important;
        background: var(--dm-surface) !important;
        z-index: 30 !important;
    }

    .st-key-docmind_chat_input [data-testid="stChatInput"],
    div[class*="st-key-docmind_chat_input"] [data-testid="stChatInput"] {
        position: relative !important;
        inset: auto !important;
        margin: 0 !important;
        width: 100% !important;
    }

    /* Keep the example questions visible above the fixed composer. */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker)
    .stButton {
        flex: 0 0 auto !important;
    }

    /* ========================================================
       SIDEBAR HISTORY — readable in dark and light modes.
       ======================================================== */

    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button:hover,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button:focus,
    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button:active {
        background: var(--dm-sidebar-hover) !important;
        background-color: var(--dm-sidebar-hover) !important;
        border: 1px solid var(--dm-sidebar-border) !important;
        color: var(--dm-sidebar-text) !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button * {
        color: var(--dm-sidebar-text) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stSidebar"] [class*="st-key-history_nav_"] button:hover {
        border-color: var(--dm-primary) !important;
    }

    /* Analytics / Knowledge Base / Documents keep their own scrollbars. */
    .st-key-analytics_scroll_container,
    div[class*="st-key-analytics_scroll_container"],
    .st-key-knowledge_scroll_container,
    div[class*="st-key-knowledge_scroll_container"],
    .st-key-right_documents_scroll_container,
    div[class*="st-key-right_documents_scroll_container"] {
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }

    @media (max-width: 900px) {
        .st-key-ai_chat_workspace_box,
        div[class*="st-key-ai_chat_workspace_box"] {
            height: auto !important;
            max-height: none !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker)
        > div
        > [data-testid="stVerticalBlock"] {
            display: block !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        .st-key-docmind_chat_input,
        div[class*="st-key-docmind_chat_input"] {
            margin-top: 10px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINAL CHATGPT-LIKE AI CHAT OVERRIDES
# ============================================================

st.markdown(
    """
    <style>
    /* Scope every rule to the AI chat workspace only. */

    .st-key-ai_chat_workspace_box,
    div[class*="st-key-ai_chat_workspace_box"] {
        height: calc(100vh - 5.4rem) !important;
        max-height: calc(100vh - 5.4rem) !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 14px !important;
        background: var(--dm-surface) !important;
        box-sizing: border-box !important;
    }

    /* Restore normal Streamlit flow inside the outer chat shell.
       The middle native container handles scrolling. */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker)
    > div,
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker)
    > div > [data-testid="stVerticalBlock"] {
        height: auto !important;
        max-height: none !important;
        min-height: 0 !important;
        overflow: visible !important;
        display: block !important;
    }

    .dm-chat-box-title,
    .dm-chat-box-subtitle,
    .dm-empty-state,
    .dm-empty-icon,
    .dm-empty-title,
    .dm-empty-text {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    .dm-chat-box-title {
        margin-bottom: 2px !important;
    }

    .dm-chat-box-subtitle {
        margin-bottom: 8px !important;
    }

    /* This is the ONLY scrollable area in the chatbot. */
    .st-key-chat_history_scroll_container,
    div[class*="st-key-agent_v2_scroll"] {
        height: calc(100vh - 14.1rem) !important;
        max-height: calc(100vh - 14.1rem) !important;
        min-height: 300px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        overscroll-behavior: contain !important;
        scrollbar-width: thin !important;
        padding-right: 5px !important;
        margin-bottom: 8px !important;
    }

    .st-key-chat_history_scroll_container
    [data-testid="stVerticalBlock"],
    div[class*="st-key-agent_v2_scroll"]
    [data-testid="stVerticalBlock"] {
        height: auto !important;
        max-height: none !important;
        min-height: min-content !important;
        overflow: visible !important;
        display: block !important;
    }

    .st-key-chat_history_scroll_container::-webkit-scrollbar,
    div[class*="st-key-agent_v2_scroll"]::-webkit-scrollbar {
        width: 7px !important;
    }

    .st-key-chat_history_scroll_container::-webkit-scrollbar-thumb,
    div[class*="st-key-agent_v2_scroll"]::-webkit-scrollbar-thumb {
        background: var(--dm-border) !important;
        border-radius: 999px !important;
    }

    /* Welcome screen sits naturally inside the message viewport. */
    .dm-empty-state {
        min-height: 270px !important;
        max-height: none !important;
        padding: 48px 24px 24px !important;
        overflow: visible !important;
    }

    /* Prompt stays below the scroller and never scrolls with messages. */
    .st-key-docmind_chat_input,
    div[class*="st-key-docmind_chat_input"] {
        position: relative !important;
        inset: auto !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
        background: var(--dm-surface) !important;
        z-index: 10 !important;
    }

    .st-key-docmind_chat_input [data-testid="stChatInput"],
    div[class*="st-key-docmind_chat_input"] [data-testid="stChatInput"] {
        position: relative !important;
        inset: auto !important;
        width: 100% !important;
        margin: 0 !important;
    }

    @media (max-width: 900px) {
        .st-key-ai_chat_workspace_box,
        div[class*="st-key-ai_chat_workspace_box"],
        .st-key-chat_history_scroll_container,
        div[class*="st-key-agent_v2_scroll"] {
            height: auto !important;
            max-height: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# REFERENCE-STYLE AI AGENT — AI SECTION ONLY
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Fixed center AI workspace ---------- */
    .st-key-ai_chat_workspace_box,
    div[class*="st-key-ai_chat_workspace_box"] {
        height: calc(100vh - 5.25rem) !important;
        max-height: calc(100vh - 5.25rem) !important;
        min-height: 0 !important;
        overflow: hidden !important;
        background: transparent !important;
    }

    .st-key-ai_chat_workspace_box > div,
    div[class*="st-key-ai_chat_workspace_box"] > div {
        height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }

    /* Neutralize old bordered-chat rules only in this AI workspace. */
    .st-key-ai_chat_workspace_box
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker),
    div[class*="st-key-ai_chat_workspace_box"]
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-box-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* ---------- Centered DocMind identity ---------- */
    .dm-agent-header {
        flex: 0 0 auto !important;
        text-align: center !important;
        padding: 2px 0 14px !important;
    }

    .dm-agent-brand-row {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 10px !important;
    }

    .dm-agent-logo {
        width: 38px !important;
        height: 38px !important;
        border-radius: 11px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: linear-gradient(
            135deg,
            var(--dm-primary),
            var(--dm-primary2)
        ) !important;
        color: #ffffff !important;
        font-size: 18px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, .18) !important;
    }

    .dm-agent-title {
        color: var(--dm-heading) !important;
        font-size: 1.75rem !important;
        line-height: 1 !important;
        font-weight: 800 !important;
        letter-spacing: -.03em !important;
    }

    .dm-agent-subtitle {
        margin-top: 7px !important;
        color: var(--dm-muted) !important;
        font-size: .9rem !important;
    }

    /* ---------- Conversation viewport ---------- */
    .st-key-chat_history_scroll_container,
    div[class*="st-key-agent_v2_scroll"] {
        height: calc(100vh - 18.8rem) !important;
        max-height: calc(100vh - 18.8rem) !important;
        min-height: 300px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        overscroll-behavior: contain !important;
        scrollbar-width: thin !important;
        padding: 2px 18px 12px 8px !important;
        margin: 0 auto !important;
        width: 100% !important;
    }

    .st-key-chat_history_scroll_container::-webkit-scrollbar,
    div[class*="st-key-agent_v2_scroll"]::-webkit-scrollbar {
        width: 7px !important;
    }

    .st-key-chat_history_scroll_container::-webkit-scrollbar-thumb,
    div[class*="st-key-agent_v2_scroll"]::-webkit-scrollbar-thumb {
        background: var(--dm-border) !important;
        border-radius: 999px !important;
    }

    .st-key-chat_history_scroll_container
    [data-testid="stVerticalBlock"],
    div[class*="st-key-agent_v2_scroll"]
    [data-testid="stVerticalBlock"] {
        height: auto !important;
        max-height: none !important;
        min-height: min-content !important;
        overflow: visible !important;
    }

    /* Streamlit chat turns look like the supplied reference:
       no giant cards, clean content rows with avatars. */
    .st-key-chat_history_scroll_container [data-testid="stChatMessage"],
    div[class*="st-key-agent_v2_scroll"]
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        padding: 10px 2px !important;
        margin: 0 !important;
    }

    .st-key-chat_history_scroll_container
    [data-testid="stChatMessageContent"],
    div[class*="st-key-agent_v2_scroll"]
    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        padding-top: 1px !important;
    }

    .st-key-chat_history_scroll_container
    [data-testid="stChatMessage"] p,
    div[class*="st-key-agent_v2_scroll"]
    [data-testid="stChatMessage"] p {
        color: var(--dm-text) !important;
        line-height: 1.55 !important;
    }

    .st-key-chat_history_scroll_container
    [data-testid="stChatMessageAvatarUser"],
    .st-key-chat_history_scroll_container
    [data-testid="stChatMessageAvatarAssistant"],
    div[class*="st-key-agent_v2_scroll"]
    [data-testid="stChatMessageAvatarUser"],
    div[class*="st-key-agent_v2_scroll"]
    [data-testid="stChatMessageAvatarAssistant"] {
        transform: scale(.92) !important;
    }

    .dm-turn-gap {
        height: 10px !important;
    }

    /* Source cards become compact, matching the reference density. */
    .st-key-chat_history_scroll_container
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--dm-border) !important;
        background: var(--dm-surface2) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    /* ---------- Empty state ---------- */
    .dm-agent-empty {
        min-height: 300px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        padding: 26px !important;
    }

    .dm-agent-empty-icon {
        width: 54px !important;
        height: 54px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 15px !important;
        background: var(--dm-soft) !important;
        border: 1px solid var(--dm-border) !important;
        font-size: 25px !important;
        margin-bottom: 16px !important;
    }

    .dm-agent-empty-title {
        color: var(--dm-heading) !important;
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        margin-bottom: 8px !important;
    }

    .dm-agent-empty-text {
        color: var(--dm-muted) !important;
        max-width: 570px !important;
        font-size: .9rem !important;
        line-height: 1.55 !important;
    }

    .dm-example-label {
        color: var(--dm-heading) !important;
        font-size: .95rem !important;
        font-weight: 750 !important;
        margin: 2px 0 8px !important;
    }

    .st-key-chat_history_scroll_container .stButton > button,
    div[class*="st-key-agent_v2_scroll"] .stButton > button {
        min-height: 42px !important;
        background: var(--dm-surface2) !important;
        color: var(--dm-text) !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 10px !important;
        white-space: normal !important;
    }

    .st-key-chat_history_scroll_container .stButton > button:hover,
    div[class*="st-key-agent_v2_scroll"] .stButton > button:hover {
        border-color: var(--dm-primary) !important;
        color: var(--dm-primary) !important;
    }

    /* ---------- RAG status strip ---------- */
    .dm-agent-pipeline {
        flex: 0 0 auto !important;
        min-height: 42px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-wrap: wrap !important;
        gap: 9px !important;
        padding: 8px 13px !important;
        margin: 4px 16px 3px !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 10px !important;
        background: var(--dm-surface2) !important;
        color: var(--dm-text) !important;
        font-size: .78rem !important;
    }

    .dm-pipeline-arrow {
        color: var(--dm-muted) !important;
    }

    .dm-agent-grounding {
        flex: 0 0 auto !important;
        color: var(--dm-muted) !important;
        font-size: .76rem !important;
        margin: 0 18px 5px !important;
    }

    /* ---------- Fixed composer ---------- */
    .st-key-docmind_chat_input,
    div[class*="st-key-docmind_chat_input"] {
        flex: 0 0 auto !important;
        position: relative !important;
        inset: auto !important;
        margin: 4px 16px 0 !important;
        padding: 0 !important;
        background: transparent !important;
        z-index: 20 !important;
    }

    .st-key-docmind_chat_input [data-testid="stChatInput"],
    div[class*="st-key-docmind_chat_input"] [data-testid="stChatInput"] {
        position: relative !important;
        inset: auto !important;
        min-height: 62px !important;
        margin: 0 !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 13px !important;
        background: var(--dm-input) !important;
        box-shadow: 0 8px 26px rgba(0,0,0,.10) !important;
    }

    .st-key-docmind_chat_input textarea,
    div[class*="st-key-docmind_chat_input"] textarea {
        min-height: 58px !important;
        padding: 17px 56px 14px 15px !important;
        background: var(--dm-input) !important;
        color: var(--dm-input-text) !important;
    }

    .st-key-docmind_chat_input button,
    div[class*="st-key-docmind_chat_input"] button {
        background: var(--dm-surface3) !important;
        border: 1px solid var(--dm-border) !important;
        color: var(--dm-text) !important;
        border-radius: 9px !important;
        margin-right: 6px !important;
    }

    .st-key-docmind_chat_input button:hover,
    div[class*="st-key-docmind_chat_input"] button:hover {
        background: var(--dm-primary) !important;
        border-color: var(--dm-primary) !important;
        color: #ffffff !important;
    }

    .dm-agent-disclaimer {
        flex: 0 0 auto !important;
        text-align: center !important;
        color: var(--dm-muted) !important;
        font-size: .67rem !important;
        padding: 7px 0 0 !important;
    }

    @media (max-width: 900px) {
        .st-key-ai_chat_workspace_box,
        div[class*="st-key-ai_chat_workspace_box"] {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        .st-key-chat_history_scroll_container,
        div[class*="st-key-agent_v2_scroll"] {
            height: 520px !important;
            max-height: 520px !important;
            min-height: 0 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# ISOLATED DOCMIND AGENT V2 — CENTER PANEL ONLY
# ============================================================

st.markdown(
    """
    <style>
    /* The center AI area uses normal flow.
       Older experimental chat CSS uses different keys/classes,
       so it cannot hide this version. */
    div[data-testid="column"]:has(.dm-agent-v2-column-marker) {
        height: calc(100vh - 5.25rem) !important;
        max-height: calc(100vh - 5.25rem) !important;
        min-height: 0 !important;
        overflow: hidden !important;
        padding: 0 14px 0 8px !important;
    }

    /* ---------- Header ---------- */
    .dm-agent-v2-header {
        text-align: center !important;
        padding: 4px 0 14px !important;
    }

    .dm-agent-v2-brand {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 10px !important;
    }

    .dm-agent-v2-logo {
        width: 38px !important;
        height: 38px !important;
        border-radius: 11px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: linear-gradient(
            135deg,
            var(--dm-primary),
            var(--dm-primary2)
        ) !important;
        color: #ffffff !important;
        font-size: 18px !important;
        box-shadow: 0 8px 22px rgba(0,0,0,.18) !important;
    }

    .dm-agent-v2-title {
        color: var(--dm-heading) !important;
        font-size: 1.72rem !important;
        line-height: 1 !important;
        font-weight: 800 !important;
        letter-spacing: -.025em !important;
    }

    .dm-agent-v2-subtitle {
        color: var(--dm-muted) !important;
        font-size: .88rem !important;
        margin-top: 7px !important;
    }

    /* ---------- Scrollable conversation ---------- */
    .st-key-agent_v2_scroll,
    div[class*="st-key-agent_v2_scroll"] {
        height: calc(100vh - 17.8rem) !important;
        max-height: calc(100vh - 17.8rem) !important;
        min-height: 330px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        overscroll-behavior: contain !important;
        scrollbar-width: thin !important;
        padding: 4px 14px 12px 6px !important;
        background: transparent !important;
    }

    .st-key-agent_v2_scroll::-webkit-scrollbar,
    div[class*="st-key-agent_v2_scroll"]::-webkit-scrollbar {
        width: 7px !important;
    }

    .st-key-agent_v2_scroll::-webkit-scrollbar-thumb,
    div[class*="st-key-agent_v2_scroll"]::-webkit-scrollbar-thumb {
        background: var(--dm-border) !important;
        border-radius: 999px !important;
    }

    /* Clean message rows like the reference. */
    .st-key-agent_v2_scroll [data-testid="stChatMessage"],
    div[class*="st-key-agent_v2_scroll"] [data-testid="stChatMessage"] {
        background: transparent !important;
        border: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        padding: 9px 2px !important;
        margin: 0 !important;
    }

    .st-key-agent_v2_scroll [data-testid="stChatMessageContent"],
    div[class*="st-key-agent_v2_scroll"]
    [data-testid="stChatMessageContent"] {
        background: transparent !important;
    }

    .st-key-agent_v2-scroll [data-testid="stChatMessage"] p,
    .st-key-agent_v2_scroll [data-testid="stChatMessage"] p,
    div[class*="st-key-agent_v2_scroll"] [data-testid="stChatMessage"] p {
        color: var(--dm-text) !important;
        line-height: 1.5 !important;
    }

    .dm-agent-v2-turn-gap {
        height: 8px !important;
    }

    /* Existing source cards stay readable but compact. */
    .st-key-agent_v2_scroll
    [data-testid="stVerticalBlockBorderWrapper"],
    div[class*="st-key-agent_v2_scroll"]
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--dm-border) !important;
        background: var(--dm-surface2) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    /* ---------- Welcome state ---------- */
    .dm-agent-v2-empty {
        min-height: 285px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        padding: 28px 22px !important;
    }

    .dm-agent-v2-empty-icon {
        width: 54px !important;
        height: 54px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 15px !important;
        background: var(--dm-soft) !important;
        border: 1px solid var(--dm-border) !important;
        font-size: 25px !important;
        margin-bottom: 15px !important;
    }

    .dm-agent-v2-empty-title {
        color: var(--dm-heading) !important;
        font-size: 1.12rem !important;
        font-weight: 800 !important;
        margin-bottom: 8px !important;
    }

    .dm-agent-v2-empty-text {
        color: var(--dm-muted) !important;
        max-width: 570px !important;
        line-height: 1.55 !important;
        font-size: .88rem !important;
    }

    .dm-agent-v2-example-title {
        color: var(--dm-heading) !important;
        font-size: .95rem !important;
        font-weight: 750 !important;
        margin: 4px 0 8px !important;
    }

    .st-key-agent_v2_scroll .stButton > button,
    div[class*="st-key-agent_v2_scroll"] .stButton > button {
        min-height: 42px !important;
        height: auto !important;
        white-space: normal !important;
        background: var(--dm-surface2) !important;
        color: var(--dm-text) !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 10px !important;
    }

    .st-key-agent_v2_scroll .stButton > button:hover,
    div[class*="st-key-agent_v2_scroll"] .stButton > button:hover {
        border-color: var(--dm-primary) !important;
        color: var(--dm-primary) !important;
    }

    /* ---------- RAG status strip ---------- */
    .dm-agent-v2-pipeline {
        min-height: 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        padding: 7px 12px !important;
        margin: 4px 10px 3px !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 10px !important;
        background: var(--dm-surface2) !important;
        color: var(--dm-text) !important;
        font-size: .76rem !important;
    }

    .dm-agent-v2-arrow {
        color: var(--dm-muted) !important;
    }

    .dm-agent-v2-grounding {
        color: var(--dm-muted) !important;
        font-size: .74rem !important;
        margin: 0 12px 5px !important;
    }

    /* ---------- Fixed composer ---------- */
    .st-key-agent_v2_input,
    div[class*="st-key-agent_v2_input"] {
        position: relative !important;
        inset: auto !important;
        margin: 5px 10px 0 !important;
        padding: 0 !important;
        background: transparent !important;
        z-index: 20 !important;
    }

    .st-key-agent_v2_input [data-testid="stChatInput"],
    div[class*="st-key-agent_v2_input"] [data-testid="stChatInput"] {
        min-height: 60px !important;
        position: relative !important;
        inset: auto !important;
        margin: 0 !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 13px !important;
        background: var(--dm-input) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,.10) !important;
    }

    .st-key-agent_v2_input textarea,
    div[class*="st-key-agent_v2_input"] textarea {
        min-height: 56px !important;
        padding: 16px 54px 13px 15px !important;
        background: var(--dm-input) !important;
        color: var(--dm-input-text) !important;
    }

    .st-key-agent_v2_input button,
    div[class*="st-key-agent_v2_input"] button {
        background: var(--dm-surface3) !important;
        border: 1px solid var(--dm-border) !important;
        color: var(--dm-text) !important;
        border-radius: 9px !important;
        margin-right: 6px !important;
    }

    .st-key-agent_v2_input button:hover,
    div[class*="st-key-agent_v2_input"] button:hover {
        background: var(--dm-primary) !important;
        border-color: var(--dm-primary) !important;
        color: #ffffff !important;
    }

    .dm-agent-v2-disclaimer {
        text-align: center !important;
        color: var(--dm-muted) !important;
        font-size: .66rem !important;
        padding: 6px 0 0 !important;
    }

    @media (max-width: 900px) {
        div[data-testid="column"]:has(.dm-agent-v2-column-marker) {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        .st-key-agent_v2_scroll,
        div[class*="st-key-agent_v2_scroll"] {
            height: 500px !important;
            max-height: 500px !important;
            min-height: 0 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINAL TOP-SPACE ALIGNMENT FIX
# ============================================================

st.markdown(
    """
    <style>
    /* Reduce the large unused area below Streamlit's top toolbar. */
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] .block-container,
    .main .block-container,
    .block-container {
        padding-top: .65rem !important;
        padding-bottom: .65rem !important;
    }

    /* Start the main center/right workspace immediately below the toolbar. */
    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {
        margin-top: 0 !important;
        padding-top: 0 !important;
        height: calc(100vh - 4.15rem) !important;
        max-height: calc(100vh - 4.15rem) !important;
        align-items: stretch !important;
    }

    /* Move only the center AI-agent column upward. */
    div[data-testid="column"]:has(.dm-agent-v2-column-marker) {
        margin-top: 0 !important;
        padding-top: 0 !important;
        top: 0 !important;
        height: calc(100vh - 4.15rem) !important;
        max-height: calc(100vh - 4.15rem) !important;
    }

    .dm-agent-v2-header {
        margin-top: 0 !important;
        padding-top: 2px !important;
    }

    /* Move the Documents sidebar upward as well.
       Keep all of its existing scrolling and document controls unchanged. */
    div[data-testid="column"]:has(.dm-right-panel-marker) {
        margin-top: 0 !important;
        padding-top: 0 !important;
        top: 0 !important;
        position: relative !important;
        height: calc(100vh - 4.15rem) !important;
        max-height: calc(100vh - 4.15rem) !important;
        min-height: 0 !important;
    }

    .dm-right-panel-title {
        margin-top: 0 !important;
        padding-top: 2px !important;
    }

    .st-key-right_documents_scroll_container,
    div[class*="st-key-right_documents_scroll_container"] {
        height: calc(100vh - 7rem) !important;
        max-height: calc(100vh - 7rem) !important;
    }

    /* Let the center conversation viewport use the newly gained height. */
    .st-key-agent_v2_scroll,
    div[class*="st-key-agent_v2_scroll"] {
        height: calc(100vh - 13.8rem) !important;
        max-height: calc(100vh - 13.8rem) !important;
    }

    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"],
        .block-container {
            padding-top: 1rem !important;
        }

        div[data-testid="column"]:has(.dm-agent-v2-column-marker),
        div[data-testid="column"]:has(.dm-right-panel-marker) {
            height: auto !important;
            max-height: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINAL TOP ALIGNMENT + FIXED CHAT COMPOSER
# ============================================================

st.markdown(
    """
    <style>
    /* --------------------------------------------------------
       MAIN WORKSPACE
       Start immediately below Streamlit's toolbar.
       -------------------------------------------------------- */
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] .block-container,
    .main .block-container,
    .block-container {
        padding-top: .8rem !important;
        padding-bottom: .65rem !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {
        height: calc(100vh - 4.3rem) !important;
        max-height: calc(100vh - 4.3rem) !important;
        min-height: 0 !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
        align-items: stretch !important;
        overflow: hidden !important;
    }

    /* --------------------------------------------------------
       CENTER AI COLUMN
       Vertically align its top with the DocMind brand in the
       left sidebar. The direct Streamlit vertical block becomes
       a flex column so the composer can stay at the bottom.
       -------------------------------------------------------- */
    div[data-testid="column"]:has(.dm-agent-v2-column-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
        overflow: hidden !important;
    }

    div[data-testid="column"]:has(.dm-agent-v2-column-marker)
    > div,
    div[data-testid="column"]:has(.dm-agent-v2-column-marker)
    > div > [data-testid="stVerticalBlock"] {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
    }

    div[data-testid="column"]:has(.dm-agent-v2-column-marker)
    > div > [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        align-items: stretch !important;
        overflow: hidden !important;
        gap: .35rem !important;
    }

    .dm-agent-v2-column-marker {
        display: none !important;
    }

    /* Header stays at the very top of the AI column. */
    .dm-agent-v2-header {
        flex: 0 0 auto !important;
        margin: 0 !important;
        padding: 2px 0 10px !important;
    }

    /* --------------------------------------------------------
       CHAT HISTORY
       This middle region receives all remaining height and is
       the only part of the AI section that scrolls.
       -------------------------------------------------------- */
    .st-key-agent_v2_scroll,
    div[class*="st-key-agent_v2_scroll"] {
        flex: 1 1 auto !important;
        height: auto !important;
        max-height: none !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        overscroll-behavior: contain !important;
        scrollbar-width: thin !important;
        margin: 0 !important;
        padding: 2px 14px 8px 6px !important;
    }

    .st-key-agent_v2_scroll > div,
    div[class*="st-key-agent_v2_scroll"] > div {
        min-height: 0 !important;
        max-height: none !important;
    }

    /* Welcome screen fills the middle region naturally. */
    .dm-agent-v2-empty {
        min-height: 250px !important;
        height: auto !important;
        max-height: none !important;
        padding: 24px 22px 18px !important;
    }

    /* Status strip remains outside the scrolling history. */
    .dm-agent-v2-pipeline,
    .dm-agent-v2-grounding {
        flex: 0 0 auto !important;
    }

    /* --------------------------------------------------------
       PROMPT COMPOSER
       Always visible at the bottom; never part of chat scrolling.
       -------------------------------------------------------- */
    .st-key-agent_v2_input,
    div[class*="st-key-agent_v2_input"] {
        flex: 0 0 auto !important;
        position: relative !important;
        inset: auto !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        margin: 5px 10px 0 !important;
        padding: 0 !important;
        background: transparent !important;
        z-index: 50 !important;
    }

    .st-key-agent_v2_input [data-testid="stChatInput"],
    div[class*="st-key-agent_v2_input"] [data-testid="stChatInput"] {
        position: relative !important;
        inset: auto !important;
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 100% !important;
        min-height: 60px !important;
        margin: 0 !important;
    }

    .dm-agent-v2-disclaimer {
        flex: 0 0 auto !important;
        padding: 5px 0 0 !important;
        margin: 0 !important;
    }

    /* --------------------------------------------------------
       RIGHT DOCUMENTS COLUMN
       Align its title with the top of the center AI section and
       let its content scroll independently underneath.
       -------------------------------------------------------- */
    div[data-testid="column"]:has(.dm-right-panel-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
        top: auto !important;
        position: relative !important;
        overflow: hidden !important;
    }

    div[data-testid="column"]:has(.dm-right-panel-marker)
    > div,
    div[data-testid="column"]:has(.dm-right-panel-marker)
    > div > [data-testid="stVerticalBlock"] {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
    }

    div[data-testid="column"]:has(.dm-right-panel-marker)
    > div > [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        overflow: hidden !important;
    }

    .dm-right-panel-title {
        flex: 0 0 auto !important;
        margin: 0 !important;
        padding: 2px 0 10px !important;
    }

    .st-key-right_documents_scroll_container,
    div[class*="st-key-right_documents_scroll_container"] {
        flex: 1 1 auto !important;
        height: auto !important;
        max-height: none !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
    }

    @media (max-width: 900px) {
        div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker),
        div[data-testid="column"]:has(.dm-agent-v2-column-marker),
        div[data-testid="column"]:has(.dm-right-panel-marker) {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        div[data-testid="column"]:has(.dm-agent-v2-column-marker)
        > div > [data-testid="stVerticalBlock"],
        div[data-testid="column"]:has(.dm-right-panel-marker)
        > div > [data-testid="stVerticalBlock"] {
            display: block !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        .st-key-agent_v2_scroll,
        div[class*="st-key-agent_v2_scroll"] {
            height: 500px !important;
            max-height: 500px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# CLEAN TOP ALIGNMENT OVERRIDE
# ============================================================

st.markdown(
    """
    <style>
    /* Main content begins directly below Streamlit's toolbar. */
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] .block-container,
    .main .block-container,
    .block-container {
        padding-top: .55rem !important;
        padding-bottom: .55rem !important;
    }

    /*
      The center and right columns share the same horizontal row.
      Keep that row flush to the top and use the available viewport.
    */
    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {
        margin-top: 0 !important;
        padding-top: 0 !important;
        align-items: flex-start !important;
        min-height: 0 !important;
        height: calc(100vh - 4.2rem) !important;
        max-height: calc(100vh - 4.2rem) !important;
        overflow: hidden !important;
    }

    /* Center AI / Analytics / Knowledge column starts at row top. */
    div[data-testid="column"]:has(.dm-agent-v2-column-marker),
    div[data-testid="column"]:has(.dm-analytics-scroll-marker),
    div[data-testid="column"]:has(.dm-kb-scroll-marker) {
        margin-top: 0 !important;
        padding-top: 0 !important;
        top: auto !important;
        min-height: 0 !important;
        height: 100% !important;
        max-height: 100% !important;
    }

    /* AI title is flush with the top of its center column. */
    .dm-agent-v2-header {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /*
      Right Documents column begins at exactly the same row top.
      Its existing internal scroller is preserved.
    */
    div[data-testid="column"]:has(.dm-right-panel-marker) {
        margin-top: 0 !important;
        padding-top: 0 !important;
        top: auto !important;
        position: relative !important;
        min-height: 0 !important;
        height: 100% !important;
        max-height: 100% !important;
        overflow: hidden !important;
    }

    .dm-right-panel-title {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Keep Documents content independently scrollable below its heading. */
    .st-key-right_documents_scroll_container,
    div[class*="st-key-right_documents_scroll_container"] {
        overflow-y: auto !important;
        overflow-x: hidden !important;
        min-height: 0 !important;
    }

    /*
      Keep the prompt visible. The center conversation takes the remaining
      space, while the composer stays outside that scrolling region.
    */
    .st-key-agent_v2_scroll,
    div[class*="st-key-agent_v2_scroll"] {
        min-height: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }

    .st-key-agent_v2_input,
    div[class*="st-key-agent_v2_input"] {
        visibility: visible !important;
        opacity: 1 !important;
        flex: 0 0 auto !important;
        position: relative !important;
        inset: auto !important;
        z-index: 30 !important;
    }

    /* Responsive layout returns to normal flow on smaller screens. */
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"],
        .block-container {
            padding-top: .85rem !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker),
        div[data-testid="column"]:has(.dm-agent-v2-column-marker),
        div[data-testid="column"]:has(.dm-right-panel-marker),
        div[data-testid="column"]:has(.dm-analytics-scroll-marker),
        div[data-testid="column"]:has(.dm-kb-scroll-marker) {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINAL SIDEBAR COLLAPSE + HEADER VISIBILITY FIX
# ============================================================

st.markdown(
    """
    <style>
    /*
      Do not set sidebar width, min-width, max-width, transform, left,
      display, visibility, or opacity here. Streamlit needs to control
      those properties for its native collapse/expand button.
    */

    /* Keep only the sidebar's internal scrolling behavior. */
    [data-testid="stSidebar"] {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {
        height: 100% !important;
        max-height: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
    }

    /* Move the workspace just below Streamlit's fixed toolbar. */
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] .block-container,
    .main .block-container,
    .block-container {
        padding-top: 1.35rem !important;
        padding-bottom: .55rem !important;
    }

    /* Center AI + right Documents row stays high, but no longer clips. */
    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {
        margin-top: 0 !important;
        padding-top: 0 !important;
        height: calc(100vh - 4.95rem) !important;
        max-height: calc(100vh - 4.95rem) !important;
        min-height: 0 !important;
        align-items: flex-start !important;
        overflow: hidden !important;
    }

    div[data-testid="column"]:has(.dm-agent-v2-column-marker),
    div[data-testid="column"]:has(.dm-right-panel-marker) {
        margin-top: 0 !important;
        padding-top: 0 !important;
        top: auto !important;
        min-height: 0 !important;
        height: 100% !important;
        max-height: 100% !important;
    }

    /* Ensure both headings are fully visible. */
    .dm-agent-v2-header,
    .dm-right-panel-title {
        margin-top: 0 !important;
        padding-top: .15rem !important;
    }

    /* Preserve the fixed composer and internal scrolling. */
    .st-key-agent_v2_scroll,
    div[class*="st-key-agent_v2_scroll"],
    .st-key-right_documents_scroll_container,
    div[class*="st-key-right_documents_scroll_container"] {
        min-height: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }

    .st-key-agent_v2_input,
    div[class*="st-key-agent_v2_input"] {
        visibility: visible !important;
        opacity: 1 !important;
        position: relative !important;
        inset: auto !important;
        z-index: 30 !important;
    }

    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"],
        .block-container {
            padding-top: .9rem !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker),
        div[data-testid="column"]:has(.dm-agent-v2-column-marker),
        div[data-testid="column"]:has(.dm-right-panel-marker) {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# RESPONSIVE AI AGENT + BOTTOM PROMPT ONLY
# ============================================================

st.markdown(
    """
    <style>
    /* --------------------------------------------------------
       AI AGENT COLUMN
       Responsive — no fixed pixel height for the whole agent.
       -------------------------------------------------------- */
    div[data-testid="column"]:has(.dm-agent-v2-column-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
    }

    div[data-testid="column"]:has(.dm-agent-v2-column-marker)
    > div,
    div[data-testid="column"]:has(.dm-agent-v2-column-marker)
    > div > [data-testid="stVerticalBlock"] {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
    }

    div[data-testid="column"]:has(.dm-agent-v2-column-marker)
    > div > [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
        gap: .4rem !important;
    }

    /* Header stays natural/responsive. */
    .dm-agent-v2-header {
        flex: 0 0 auto !important;
        margin: 0 !important;
        padding: .2rem 0 .6rem !important;
    }

    /* --------------------------------------------------------
       CHAT CONTENT
       Takes all remaining space and scrolls when needed.
       -------------------------------------------------------- */
    .st-key-agent_v2_scroll,
    div[class*="st-key-agent_v2_scroll"] {
        flex: 1 1 auto !important;
        height: auto !important;
        max-height: none !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        overscroll-behavior: contain !important;
        scrollbar-width: thin !important;
        margin: 0 !important;
        padding: .25rem .8rem .75rem .35rem !important;
    }

    .st-key-agent_v2_scroll > div,
    div[class*="st-key-agent_v2_scroll"] > div,
    .st-key-agent_v2_scroll [data-testid="stVerticalBlock"],
    div[class*="st-key-agent_v2_scroll"] [data-testid="stVerticalBlock"] {
        height: auto !important;
        max-height: none !important;
        min-height: 0 !important;
        overflow: visible !important;
    }

    /* Empty state is responsive instead of fixed-height. */
    .dm-agent-v2-empty {
        min-height: clamp(220px, 38vh, 340px) !important;
        height: auto !important;
        max-height: none !important;
        padding: clamp(18px, 3vh, 34px) 22px !important;
    }

    /* --------------------------------------------------------
       PROMPT INPUT
       This is the ONLY fixed/sticky part.
       -------------------------------------------------------- */
    .st-key-agent_v2_input,
    div[class*="st-key-agent_v2_input"] {
        flex: 0 0 auto !important;
        position: sticky !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 100 !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;

        margin: .35rem .65rem 0 !important;
        padding: .45rem 0 .1rem !important;

        background:
            linear-gradient(
                to bottom,
                rgba(0,0,0,0),
                var(--dm-surface) 34%
            ) !important;
    }

    .st-key-agent_v2_input [data-testid="stChatInput"],
    div[class*="st-key-agent_v2_input"] [data-testid="stChatInput"] {
        position: relative !important;
        inset: auto !important;
        width: 100% !important;
        min-height: 58px !important;
        margin: 0 !important;

        border: 1px solid var(--dm-border) !important;
        border-radius: 13px !important;
        background: var(--dm-input) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,.12) !important;
    }

    .st-key-agent_v2_input textarea,
    div[class*="st-key-agent_v2_input"] textarea {
        min-height: 54px !important;
        max-height: 140px !important;
        resize: none !important;
        padding: 15px 54px 13px 15px !important;

        background: var(--dm-input) !important;
        color: var(--dm-input-text) !important;
        overflow-y: auto !important;
    }

    .st-key-agent_v2_input button,
    div[class*="st-key-agent_v2_input"] button {
        flex: 0 0 auto !important;
    }

    .dm-agent-v2-disclaimer {
        flex: 0 0 auto !important;
        margin: 0 !important;
        padding: .3rem 0 0 !important;
    }

    /* --------------------------------------------------------
       RESPONSIVE BREAKPOINTS
       -------------------------------------------------------- */
    @media (max-width: 1200px) {
        .dm-agent-v2-title {
            font-size: 1.5rem !important;
        }

        .st-key-agent_v2_input,
        div[class*="st-key-agent_v2_input"] {
            margin-left: .35rem !important;
            margin-right: .35rem !important;
        }
    }

    @media (max-width: 900px) {
        div[data-testid="column"]:has(.dm-agent-v2-column-marker) {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
            display: block !important;
        }

        div[data-testid="column"]:has(.dm-agent-v2-column-marker)
        > div > [data-testid="stVerticalBlock"] {
            display: block !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        .st-key-agent_v2_scroll,
        div[class*="st-key-agent_v2_scroll"] {
            height: auto !important;
            max-height: none !important;
            min-height: 420px !important;
            overflow-y: visible !important;
        }

        .st-key-agent_v2_input,
        div[class*="st-key-agent_v2_input"] {
            position: sticky !important;
            bottom: .35rem !important;
            margin: .5rem 0 0 !important;
        }
    }

    @media (max-width: 600px) {
        .dm-agent-v2-header {
            padding-top: 0 !important;
        }

        .dm-agent-v2-brand {
            gap: 7px !important;
        }

        .dm-agent-v2-title {
            font-size: 1.32rem !important;
        }

        .dm-agent-v2-subtitle {
            font-size: .8rem !important;
        }

        .dm-agent-v2-empty {
            min-height: 210px !important;
            padding: 18px 12px !important;
        }

        .st-key-agent_v2_input textarea,
        div[class*="st-key-agent_v2_input"] textarea {
            min-height: 50px !important;
            max-height: 120px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AUTHORITATIVE PADDING COLLISION FIX
# ============================================================

st.markdown(
    """
    <style>
    /* Keep the centered title below Streamlit's fixed toolbar. */
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] .block-container,
    .main .block-container,
    .block-container {
        width: 100% !important;
        max-width: 1600px !important;
        padding-top: 4.65rem !important;
        padding-bottom: .75rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Fit the entire workspace inside the visible browser viewport. */
    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {
        height: calc(100dvh - 5.4rem) !important;
        max-height: calc(100dvh - 5.4rem) !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        align-items: stretch !important;
        overflow: hidden !important;
    }

    /* Final cascade layer: spacing only; colors and component styling stay unchanged. */
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] *,
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        box-sizing: border-box !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker),
    div[data-testid="column"]:has(.dm-agent-v2-column-marker),
    div[data-testid="column"]:has(.dm-right-panel-marker),
    div[data-testid="column"]:has(.dm-analytics-scroll-marker),
    div[data-testid="column"]:has(.dm-kb-scroll-marker) {
        min-width: 0 !important;
    }

    div[data-testid="column"]:has(.dm-agent-v2-column-marker),
    div[data-testid="column"]:has(.dm-right-panel-marker) {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        padding-bottom: 0 !important;
        overflow: hidden !important;
    }

    div[data-testid="column"]:has(.dm-agent-v2-column-marker) {
        display: flex !important;
        flex-direction: column !important;
        padding-top: 0 !important;
    }

    div[data-testid="column"]:has(.dm-agent-v2-column-marker) > div,
    div[data-testid="column"]:has(.dm-agent-v2-column-marker)
    > div > [data-testid="stVerticalBlock"] {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
    }

    div[data-testid="column"]:has(.dm-agent-v2-column-marker)
    > div > [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        gap: .3rem !important;
        overflow: hidden !important;
    }

    .dm-agent-v2-header {
        flex: 0 0 auto !important;
        margin: 0 !important;
        padding: .45rem .5rem .55rem !important;
    }

    .st-key-agent_v2_scroll,
    div[class*="st-key-agent_v2_scroll"] {
        flex: 1 1 0 !important;
        height: 0 !important;
        max-height: none !important;
        min-height: 0 !important;
        min-width: 0 !important;
        margin: 0 !important;
        padding-left: .35rem !important;
        padding-right: .75rem !important;
        padding-bottom: .8rem !important;
        scroll-padding-bottom: .8rem !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        overscroll-behavior: contain !important;
    }

    .st-key-agent_v2_scroll > div,
    div[class*="st-key-agent_v2_scroll"] > div,
    .st-key-agent_v2_scroll [data-testid="stVerticalBlock"],
    div[class*="st-key-agent_v2_scroll"] [data-testid="stVerticalBlock"] {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
    }

    .st-key-agent_v2_scroll p,
    .st-key-agent_v2_scroll li,
    div[class*="st-key-agent_v2_scroll"] p,
    div[class*="st-key-agent_v2_scroll"] li,
    .dm-doc-name,
    .dm-retrieved-title span {
        overflow-wrap: anywhere !important;
    }

    /* The flex layout keeps the input at the bottom without covering messages. */
    .st-key-agent_v2_input,
    div[class*="st-key-agent_v2_input"] {
        flex: 0 0 auto !important;
        position: relative !important;
        inset: auto !important;
        width: auto !important;
        max-width: calc(100% - 1.3rem) !important;
        margin: .2rem .65rem 0 !important;
        padding: .25rem 0 .05rem !important;
        z-index: 30 !important;
    }

    .st-key-agent_v2_input [data-testid="stChatInput"],
    div[class*="st-key-agent_v2_input"] [data-testid="stChatInput"],
    .st-key-agent_v2_input textarea,
    div[class*="st-key-agent_v2_input"] textarea {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    .st-key-agent_v2_input [data-testid="stChatInput"],
    div[class*="st-key-agent_v2_input"] [data-testid="stChatInput"] {
        min-height: 54px !important;
        margin: 0 !important;
    }

    .st-key-agent_v2_input textarea,
    div[class*="st-key-agent_v2_input"] textarea {
        min-height: 50px !important;
        max-height: 120px !important;
        padding: 13px 56px 11px 15px !important;
    }

    .dm-agent-v2-disclaimer {
        flex: 0 0 auto !important;
        margin: 0 !important;
        padding: .2rem .65rem 0 !important;
    }

    .dm-agent-v2-pipeline,
    .dm-agent-v2-grounding {
        flex: 0 0 auto !important;
    }

    .st-key-right_documents_scroll_container,
    div[class*="st-key-right_documents_scroll_container"] {
        min-width: 0 !important;
        width: 100% !important;
        padding-left: .1rem !important;
        padding-right: .45rem !important;
        padding-bottom: .75rem !important;
    }

    @media (max-width: 900px) {
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            overflow-y: auto !important;
        }

        [data-testid="stMainBlockContainer"],
        [data-testid="stMainBlockContainer"] .block-container,
        .main .block-container,
        .block-container {
            padding-top: 4.5rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        div[data-testid="column"]:has(.dm-agent-v2-column-marker),
        div[data-testid="column"]:has(.dm-right-panel-marker) {
            width: 100% !important;
            min-width: 0 !important;
            height: auto !important;
            max-height: none !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            overflow: visible !important;
        }

        div[data-testid="column"]:has(.dm-agent-v2-column-marker)
        > div > [data-testid="stVerticalBlock"] {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        .st-key-agent_v2_scroll,
        div[class*="st-key-agent_v2_scroll"] {
            flex: none !important;
            height: clamp(360px, 58dvh, 540px) !important;
            max-height: clamp(360px, 58dvh, 540px) !important;
            min-height: 0 !important;
            overflow-y: auto !important;
        }

        .st-key-agent_v2_input,
        div[class*="st-key-agent_v2_input"] {
            position: relative !important;
            bottom: auto !important;
            max-width: 100% !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
    }

    @media (max-width: 600px) {
        [data-testid="stMainBlockContainer"],
        [data-testid="stMainBlockContainer"] .block-container,
        .main .block-container,
        .block-container {
            padding-left: .7rem !important;
            padding-right: .7rem !important;
        }

        .dm-agent-v2-header {
            padding: .3rem .25rem .5rem !important;
        }

        .st-key-agent_v2_input textarea,
        div[class*="st-key-agent_v2_input"] textarea {
            padding-right: 52px !important;
        }
    }

    /* Compact vertical spacing on short laptop screens. */
    @media (min-width: 901px) and (max-height: 760px) {
        [data-testid="stMainBlockContainer"],
        [data-testid="stMainBlockContainer"] .block-container,
        .main .block-container,
        .block-container {
            padding-top: 4.25rem !important;
            padding-bottom: .45rem !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {
            height: calc(100dvh - 4.85rem) !important;
            max-height: calc(100dvh - 4.85rem) !important;
        }

        .dm-agent-v2-header {
            padding-top: .2rem !important;
            padding-bottom: .35rem !important;
        }

        .dm-agent-v2-logo {
            width: 34px !important;
            height: 34px !important;
        }

        .dm-agent-v2-subtitle {
            margin-top: 4px !important;
        }

        .dm-agent-v2-pipeline {
            min-height: 36px !important;
            padding-top: 5px !important;
            padding-bottom: 5px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Page-scrolling chat with a viewport-fixed composer. Scoped to the chat route.
st.markdown(
    """
    <style>
    [data-testid="stMain"]:has(.st-key-dm_center_chat) {
        height: 100dvh !important;
        max-height: 100dvh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scroll-padding-bottom: 180px;
    }
    [data-testid="stMainBlockContainer"]:has(.st-key-dm_center_chat),
    .block-container:has(.st-key-dm_center_chat) {
        height: auto !important;
        max-height: none !important;
        min-height: 100% !important;
        overflow: visible !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.st-key-dm_center_chat) {
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        align-items: flex-start !important;
    }
    /* Remove viewport locks only along the center column's wrapper chain. */
    :is([data-testid="column"], [data-testid="stColumn"]):has(.st-key-dm_center_chat):has(.st-key-dm_center_chat),
    :is([data-testid="column"], [data-testid="stColumn"]):has(.st-key-dm_center_chat):has(.st-key-dm_center_chat) > div,
    :is([data-testid="column"], [data-testid="stColumn"]):has(.st-key-dm_center_chat):has(.st-key-dm_center_chat) > div > [data-testid="stVerticalBlock"] {
        height: auto !important;
        max-height: none !important;
        min-height: 0 !important;
        overflow: visible !important;
        display: block !important;
        transform: none !important;
        contain: none !important;
    }
    .st-key-dm_center_chat.st-key-dm_center_chat.st-key-dm_center_chat {
        display: flex !important;
        flex-direction: column !important;
        height: auto !important;
        max-height: none !important;
        min-height: calc(100dvh - 6rem) !important;
        min-width: 0 !important;
        width: 100% !important;
        overflow: visible !important;
        gap: 1rem !important;
        padding: 0 0 var(--dm-composer-space, 180px) !important;
        box-sizing: border-box !important;
        transform: none !important;
        contain: none !important;
    }
    .st-key-dm_center_chat > div:has(.st-key-dm_center_header):has(.st-key-dm_center_composer),
    .st-key-dm_center_chat > div > [data-testid="stVerticalBlock"]:has(.st-key-dm_center_header):has(.st-key-dm_center_composer) {
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 1rem !important;
    }
    .st-key-dm_center_chat .st-key-dm_center_header,
    .st-key-dm_center_chat .st-key-dm_center_header > div,
    .st-key-dm_center_chat .st-key-dm_conversation_viewport,
    .st-key-dm_center_chat .st-key-dm_conversation_viewport > div,
    .st-key-dm_center_chat .st-key-dm_conversation_viewport [data-testid="stVerticalBlock"] {
        height: auto !important;
        max-height: none !important;
        min-height: 0 !important;
        overflow: visible !important;
        flex: 0 0 auto !important;
    }
    .st-key-dm_center_chat .st-key-dm_conversation_viewport {
        padding: .5rem .7rem 1rem !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 1rem !important;
        min-width: 0 !important;
    }
    .st-key-dm_center_chat .dm-agent-v2-empty {
        min-height: clamp(180px, 30dvh, 300px) !important;
        padding: clamp(16px, 3vw, 32px) 12px !important;
    }
    .st-key-dm_center_chat .dm-agent-v2-example-title {
        margin: .5rem 0 !important;
    }
    .st-key-dm_center_chat .stButton > button {
        white-space: normal !important;
        height: auto !important;
        min-height: 44px !important;
        padding: .65rem .8rem !important;
    }
    /* The composer remains in flow as a safe fallback before measurement. */
    .st-key-dm_center_chat .st-key-dm_center_composer {
        position: sticky !important;
        bottom: 0 !important;
        height: auto !important;
        max-height: none !important;
        min-height: 0 !important;
        padding: .65rem .6rem max(.6rem, env(safe-area-inset-bottom)) !important;
        background: var(--dm-bg) !important;
        box-sizing: border-box !important;
        z-index: 90 !important;
        gap: .35rem !important;
    }
    .st-key-dm_center_chat[data-composer-pinned="true"] .st-key-dm_center_composer {
        position: fixed !important;
        left: var(--dm-composer-left) !important;
        width: var(--dm-composer-width) !important;
        bottom: var(--dm-composer-bottom, 0px) !important;
        right: auto !important;
        top: auto !important;
        margin: 0 !important;
    }
    .st-key-dm_center_chat .st-key-agent_v2_input {
        position: relative !important;
        inset: auto !important;
        width: 100% !important;
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .st-key-dm_center_chat .st-key-dm_center_composer > div,
    .st-key-dm_center_chat .st-key-dm_center_composer [data-testid="stVerticalBlock"] {
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
        overflow: visible !important;
        gap: .35rem !important;
    }
    .st-key-dm_center_chat .dm-agent-v2-disclaimer {
        padding: .25rem 0 0 !important;
        margin: 0 !important;
        line-height: 1.4 !important;
    }
    .st-key-dm_center_chat [data-testid="stChatMessage"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        margin: 0 0 1rem !important;
        padding: .6rem .2rem !important;
    }
    .st-key-dm_center_chat [data-testid="stChatMessageContent"] {
        min-width: 0 !important;
        overflow-wrap: anywhere;
    }
    @media (max-width: 600px) {
        .st-key-dm_center_chat .dm-agent-v2-header {
            padding: .4rem .25rem !important;
        }
        .st-key-dm_center_chat .st-key-dm_conversation_viewport {
            padding-left: .15rem !important;
            padding-right: .15rem !important;
        }
        .st-key-dm_center_chat .st-key-dm_conversation_viewport [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        .st-key-dm_center_chat .st-key-dm_conversation_viewport :is([data-testid="column"], [data-testid="stColumn"]) {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    /* Reserve the existing right-hand column, but pin its CONTENT independently
       of the page and of Streamlit's sticky positioning wrappers. */
    @media (min-width: 901px) {
        [data-testid="stMain"]:has(.st-key-dm_center_chat)
        :is([data-testid="column"], [data-testid="stColumn"]):has(.st-key-dm_documents_dock) {
            position: relative !important;
            top: auto !important;
            align-self: flex-start !important;
            height: calc(100dvh - 5.4rem) !important;
            max-height: calc(100dvh - 5.4rem) !important;
            min-height: 0 !important;
            overflow: visible !important;
            transform: none !important;
            contain: none !important;
        }
        [data-testid="stMain"]:has(.st-key-dm_center_chat)
        :is([data-testid="column"], [data-testid="stColumn"]):has(.st-key-dm_documents_dock) > div,
        [data-testid="stMain"]:has(.st-key-dm_center_chat)
        :is([data-testid="column"], [data-testid="stColumn"]):has(.st-key-dm_documents_dock)
        > div > [data-testid="stVerticalBlock"] {
            height: auto !important;
            max-height: none !important;
            min-height: 0 !important;
            overflow: visible !important;
            transform: none !important;
            contain: none !important;
        }
        .st-key-dm_documents_dock.st-key-dm_documents_dock.st-key-dm_documents_dock.st-key-dm_documents_dock.st-key-dm_documents_dock[data-viewport-pinned="true"] {
            position: fixed !important;
            top: var(--dm-doc-top) !important;
            left: var(--dm-doc-left) !important;
            width: var(--dm-doc-width) !important;
            height: var(--dm-doc-height) !important;
            max-height: var(--dm-doc-height) !important;
            min-height: 0 !important;
            bottom: auto !important;
            right: auto !important;
            margin: 0 !important;
            padding: 0 .4rem 1rem 0 !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior-y: contain !important;
            scrollbar-width: thin !important;
            scrollbar-color: var(--dm-border) transparent !important;
            box-sizing: border-box !important;
            background: var(--dm-bg) !important;
            z-index: 80 !important;
            display: block !important;
        }
        .st-key-dm_documents_dock .st-key-right_documents_scroll_container,
        .st-key-dm_documents_dock .st-key-right_documents_scroll_container > div,
        .st-key-dm_documents_dock .st-key-right_documents_scroll_container [data-testid="stVerticalBlock"] {
            height: auto !important;
            max-height: none !important;
            min-height: 0 !important;
            overflow: visible !important;
            flex: 0 0 auto !important;
        }
        .st-key-dm_documents_dock[data-viewport-pinned="true"]::-webkit-scrollbar {
            width: 8px;
        }
        .st-key-dm_documents_dock[data-viewport-pinned="true"]::-webkit-scrollbar-thumb {
            background: var(--dm-border);
            border-radius: 8px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINAL MOBILE SIDEBAR COLLAPSE FIX
# ============================================================

st.markdown(
    """
    <style>
    @media (max-width: 900px) {

        /*
          Streamlit exposes sidebar state through aria-expanded.
          When collapsed, completely move the sidebar off-canvas and
          remove it from hit-testing so none of its contents remain visible.
        */
        section[data-testid="stSidebar"][aria-expanded="false"],
        [data-testid="stSidebar"][aria-expanded="false"] {
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;

            margin-left: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;

            transform: translateX(-110%) !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;

            border-right: 0 !important;
            overflow: hidden !important;
        }

        section[data-testid="stSidebar"][aria-expanded="false"] > div,
        [data-testid="stSidebar"][aria-expanded="false"] > div,
        section[data-testid="stSidebar"][aria-expanded="false"]
        [data-testid="stSidebarContent"],
        [data-testid="stSidebar"][aria-expanded="false"]
        [data-testid="stSidebarContent"] {
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            overflow: hidden !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }

        /*
          When expanded, restore a normal mobile drawer width.
          This is only applied while aria-expanded is explicitly true.
        */
        section[data-testid="stSidebar"][aria-expanded="true"],
        [data-testid="stSidebar"][aria-expanded="true"] {
            width: min(86vw, 320px) !important;
            min-width: min(86vw, 320px) !important;
            max-width: min(86vw, 320px) !important;

            transform: translateX(0) !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;

            margin-left: 0 !important;
            overflow: hidden !important;
        }

        section[data-testid="stSidebar"][aria-expanded="true"] > div,
        [data-testid="stSidebar"][aria-expanded="true"] > div,
        section[data-testid="stSidebar"][aria-expanded="true"]
        [data-testid="stSidebarContent"],
        [data-testid="stSidebar"][aria-expanded="true"]
        [data-testid="stSidebarContent"] {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }

        /*
          Keep Streamlit's collapsed/open control above the app so the
          user can always reopen the sidebar after it is hidden.
        */
        [data-testid="collapsedControl"] {
            z-index: 100000 !important;
            pointer-events: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINAL LEFT SIDEBAR FIXED BRAND/NEW CHAT + HEADER THEME TOGGLE
# ============================================================

st.markdown(
    f"""
    <style>
    /* The sidebar body remains its own scroll area. */
    [data-testid="stSidebar"] {{
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }}

    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {{
        height: 100% !important;
        max-height: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
        overscroll-behavior: contain !important;
    }}

    /* DocMind identity remains at the top while lower sidebar content scrolls. */
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.dm-sidebar-brand-sticky-marker) {{
        position: sticky !important;
        top: 0 !important;
        z-index: 50 !important;
        background: var(--dm-sidebar-bg) !important;
        margin: 0 !important;
        padding-top: .35rem !important;
        padding-bottom: .15rem !important;
    }}

    /* Marker row itself is invisible and consumes no height. */
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.dm-sidebar-newchat-sticky-marker) {{
        position: sticky !important;
        top: 4.65rem !important;
        z-index: 49 !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        background: var(--dm-sidebar-bg) !important;
    }}

    /* New Chat is the element immediately after its marker. */
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.dm-sidebar-newchat-sticky-marker)
    + [data-testid="stElementContainer"] {{
        position: sticky !important;
        top: 4.55rem !important;
        z-index: 50 !important;
        background: var(--dm-sidebar-bg) !important;
        padding-top: .55rem !important;
        padding-bottom: .8rem !important;
        margin-bottom: .2rem !important;
        border-bottom: 1px solid var(--dm-sidebar-border) !important;
    }}

    .dm-sidebar-brand-sticky-marker,
    .dm-sidebar-newchat-sticky-marker {{
        display: none !important;
    }}

    /* Documents heading theme toggle */
    div[data-testid="column"]:has(.dm-doc-theme-toggle-marker) {{
        display:flex !important;
        align-items:center !important;
        justify-content:flex-end !important;
        min-width:0 !important;
    }}

    .dm-doc-theme-toggle-marker {{
        display:none !important;
    }}

    div[data-testid="column"]:has(.dm-doc-theme-toggle-marker)
    [data-testid="stToggle"] {{
        margin:0 !important;
        width:auto !important;
    }}

    div[data-testid="column"]:has(.dm-doc-theme-toggle-marker)
    [data-testid="stToggle"] label {{
        padding:0 !important;
        margin:0 !important;
    }}

    div[data-testid="column"]:has(.dm-doc-theme-toggle-marker)
    [data-testid="stToggle"] p {{
        display:none !important;
    }}

    @media (max-width: 900px) {{
        div[data-testid="column"]:has(.dm-doc-theme-toggle-marker) {{
            justify-content:flex-end !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DOCUMENTS SIDEBAR — CLEAN FINAL LAYOUT
# ============================================================
# Desktop:
#   * whole Documents panel is fixed like the left sidebar
#   * heading/toggle/upload/submit stay fixed inside it
#   * only Uploaded Documents / Indexed KB / Retrieved Sources scroll
# Mobile:
#   * returns to normal Streamlit document flow
# ============================================================

st.markdown(
    """
    <style>
    @media (min-width: 901px) {

        /* Keep the right Streamlit column only as the layout placeholder. */
        div[data-testid="column"]:has(.dm-right-panel-marker),
        div[data-testid="stColumn"]:has(.dm-right-panel-marker) {
            position: relative !important;
            overflow: visible !important;
            min-height: 1px !important;
        }

        /* Entire Documents panel: fixed to viewport, never page-scrolls. */
        .st-key-dm_documents_dock,
        div[class*="st-key-dm_documents_dock"] {
            position: fixed !important;
            top: 4.75rem !important;
            right: 1.25rem !important;

            width: 340px !important;
            max-width: 340px !important;

            height: calc(100dvh - 5.15rem) !important;
            max-height: calc(100dvh - 5.15rem) !important;
            min-height: 0 !important;

            margin: 0 !important;
            padding: 0 .25rem 0 0 !important;
            box-sizing: border-box !important;

            background: var(--dm-bg) !important;
            z-index: 999 !important;

            overflow: hidden !important;
        }

        /* Streamlit wrapper immediately inside the keyed dock. */
        .st-key-dm_documents_dock > div,
        div[class*="st-key-dm_documents_dock"] > div {
            height: 100% !important;
            max-height: 100% !important;
            min-height: 0 !important;
            overflow: hidden !important;
        }

        /* The dock's main vertical block owns the flex layout. */
        .st-key-dm_documents_dock > div > [data-testid="stVerticalBlock"],
        div[class*="st-key-dm_documents_dock"] > div > [data-testid="stVerticalBlock"] {
            height: 100% !important;
            max-height: 100% !important;
            min-height: 0 !important;

            display: flex !important;
            flex-direction: column !important;
            flex-wrap: nowrap !important;
            gap: .35rem !important;

            overflow: hidden !important;
        }

        /* Fixed top block: Documents + toggle + Upload + Submit. */
        .st-key-dm_documents_fixed_header,
        div[class*="st-key-dm_documents_fixed_header"] {
            flex: 0 0 auto !important;
            width: 100% !important;
            height: auto !important;
            min-height: auto !important;
            max-height: none !important;

            overflow: visible !important;
            position: relative !important;
            z-index: 2 !important;

            background: var(--dm-bg) !important;
            padding-bottom: .55rem !important;
            border-bottom: 1px solid var(--dm-border) !important;
        }

        /* Lower block: the ONLY scrollable section. */
        .st-key-right_documents_scroll_container,
        div[class*="st-key-right_documents_scroll_container"] {
            flex: 1 1 auto !important;
            min-height: 0 !important;
            height: auto !important;
            max-height: none !important;

            overflow-y: auto !important;
            overflow-x: hidden !important;

            position: relative !important;
            padding-right: .45rem !important;
            padding-bottom: 1.25rem !important;
            box-sizing: border-box !important;

            scrollbar-gutter: stable !important;
            overscroll-behavior-y: contain !important;
        }

        /* Inner Streamlit wrappers must not create another scroll/clipping layer. */
        .st-key-right_documents_scroll_container > div,
        .st-key-right_documents_scroll_container > div > div,
        .st-key-right_documents_scroll_container [data-testid="stVerticalBlock"],
        div[class*="st-key-right_documents_scroll_container"] > div,
        div[class*="st-key-right_documents_scroll_container"] > div > div,
        div[class*="st-key-right_documents_scroll_container"] [data-testid="stVerticalBlock"] {
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
        }

        /* Ensure the final control/card is fully reachable. */
        .st-key-right_documents_scroll_container
        [data-testid="stVerticalBlock"] > :last-child {
            margin-bottom: 1rem !important;
        }

        /* Scrollbar only for lower document content. */
        .st-key-right_documents_scroll_container::-webkit-scrollbar,
        div[class*="st-key-right_documents_scroll_container"]::-webkit-scrollbar {
            width: 7px !important;
        }

        .st-key-right_documents_scroll_container::-webkit-scrollbar-track,
        div[class*="st-key-right_documents_scroll_container"]::-webkit-scrollbar-track {
            background: transparent !important;
        }

        .st-key-right_documents_scroll_container::-webkit-scrollbar-thumb,
        div[class*="st-key-right_documents_scroll_container"]::-webkit-scrollbar-thumb {
            background: var(--dm-border) !important;
            border-radius: 999px !important;
        }

        .st-key-right_documents_scroll_container::-webkit-scrollbar-thumb:hover,
        div[class*="st-key-right_documents_scroll_container"]::-webkit-scrollbar-thumb:hover {
            background: var(--dm-muted) !important;
        }
    }

    @media (max-width: 1200px) and (min-width: 901px) {
        .st-key-dm_documents_dock,
        div[class*="st-key-dm_documents_dock"] {
            width: 300px !important;
            max-width: 300px !important;
            right: 1rem !important;
        }
    }

    @media (max-width: 900px) {
        .st-key-dm_documents_dock,
        div[class*="st-key-dm_documents_dock"],
        .st-key-dm_documents_fixed_header,
        div[class*="st-key-dm_documents_fixed_header"],
        .st-key-right_documents_scroll_container,
        div[class*="st-key-right_documents_scroll_container"] {
            position: static !important;
            inset: auto !important;
            width: 100% !important;
            max-width: 100% !important;
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
            transform: none !important;
        }

        .st-key-dm_documents_dock > div,
        .st-key-dm_documents_dock > div > [data-testid="stVerticalBlock"],
        div[class*="st-key-dm_documents_dock"] > div,
        div[class*="st-key-dm_documents_dock"] > div > [data-testid="stVerticalBlock"] {
            height: auto !important;
            max-height: none !important;
            min-height: 0 !important;
            display: block !important;
            overflow: visible !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AUTHORITATIVE DOCUMENTS SCROLL FIX
# ============================================================
# The lower Documents area now uses Streamlit's native height container.
# This final CSS intentionally overrides all older experimental rules.
# ============================================================

st.markdown(
    """
    <style>
    @media (min-width: 901px) {

        /* Entire right sidebar remains fixed. */
        .st-key-dm_documents_dock,
        div[class*="st-key-dm_documents_dock"] {
            position: fixed !important;
            top: 4.75rem !important;
            right: 1.25rem !important;

            width: 340px !important;
            max-width: 340px !important;

            height: calc(100dvh - 5.15rem) !important;
            max-height: calc(100dvh - 5.15rem) !important;
            min-height: 0 !important;

            overflow: hidden !important;
            box-sizing: border-box !important;

            background: var(--dm-bg) !important;
            z-index: 999 !important;
        }

        /* Main dock layout = fixed header + flexible native scroll body. */
        .st-key-dm_documents_dock > div,
        div[class*="st-key-dm_documents_dock"] > div {
            height: 100% !important;
            max-height: 100% !important;
            min-height: 0 !important;
            overflow: hidden !important;
        }

        .st-key-dm_documents_dock > div > [data-testid="stVerticalBlock"],
        div[class*="st-key-dm_documents_dock"] > div > [data-testid="stVerticalBlock"] {
            height: 100% !important;
            max-height: 100% !important;
            min-height: 0 !important;

            display: flex !important;
            flex-direction: column !important;
            flex-wrap: nowrap !important;

            overflow: hidden !important;
            gap: .35rem !important;
        }

        /* Top controls never scroll. */
        .st-key-dm_documents_fixed_header,
        div[class*="st-key-dm_documents_fixed_header"] {
            flex: 0 0 auto !important;

            height: auto !important;
            max-height: none !important;

            overflow: visible !important;

            background: var(--dm-bg) !important;
            z-index: 10 !important;

            padding-bottom: .55rem !important;
            border-bottom: 1px solid var(--dm-border) !important;
        }

        /*
         * LOWER CONTENT:
         * Streamlit now creates this as a native height-limited container.
         * Let it own scrolling.
         */
        .st-key-right_documents_scroll_container,
        div[class*="st-key-right_documents_scroll_container"] {
            flex: 1 1 0 !important;

            /* Override the Python fallback height responsively. */
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;

            overflow-y: auto !important;
            overflow-x: hidden !important;

            position: relative !important;
            box-sizing: border-box !important;

            padding-right: .45rem !important;
            padding-bottom: 1rem !important;

            overscroll-behavior-y: contain !important;
            scrollbar-gutter: stable !important;
        }

        /*
         * DO NOT set overflow:visible on every nested wrapper here.
         * The native Streamlit height container needs its own internal
         * overflow machinery intact.
         */
        .st-key-right_documents_scroll_container
        [data-testid="stVerticalBlockBorderWrapper"],
        div[class*="st-key-right_documents_scroll_container"]
        [data-testid="stVerticalBlockBorderWrapper"] {
            flex: 1 1 auto !important;
            min-height: 0 !important;
            max-height: 100% !important;

            overflow-y: auto !important;
            overflow-x: hidden !important;
        }

        .st-key-right_documents_scroll_container
        [data-testid="stVerticalBlockBorderWrapper"] > div,
        div[class*="st-key-right_documents_scroll_container"]
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            min-height: min-content !important;
        }

        /* Make sure the final card/buttons can be reached. */
        .st-key-right_documents_scroll_container
        [data-testid="stVerticalBlock"] > :last-child {
            margin-bottom: 1.25rem !important;
        }

        /* Visible independent scrollbar. */
        .st-key-right_documents_scroll_container::-webkit-scrollbar,
        .st-key-right_documents_scroll_container
        [data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar,
        div[class*="st-key-right_documents_scroll_container"]::-webkit-scrollbar,
        div[class*="st-key-right_documents_scroll_container"]
        [data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar {
            width: 8px !important;
        }

        .st-key-right_documents_scroll_container::-webkit-scrollbar-thumb,
        .st-key-right_documents_scroll_container
        [data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb,
        div[class*="st-key-right_documents_scroll_container"]::-webkit-scrollbar-thumb,
        div[class*="st-key-right_documents_scroll_container"]
        [data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb {
            background: var(--dm-border) !important;
            border-radius: 999px !important;
        }
    }

    @media (max-width: 1200px) and (min-width: 901px) {
        .st-key-dm_documents_dock,
        div[class*="st-key-dm_documents_dock"] {
            width: 300px !important;
            max-width: 300px !important;
            right: 1rem !important;
        }
    }

    @media (max-width: 900px) {
        .st-key-dm_documents_dock,
        div[class*="st-key-dm_documents_dock"],
        .st-key-dm_documents_fixed_header,
        div[class*="st-key-dm_documents_fixed_header"],
        .st-key-right_documents_scroll_container,
        div[class*="st-key-right_documents_scroll_container"] {
            position: static !important;

            width: 100% !important;
            max-width: 100% !important;

            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;

            overflow: visible !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL AUTHORITATIVE RIGHT DOCUMENTS SIDEBAR
# ============================================================
# Desktop behavior:
#   1. Right Documents panel is fixed to viewport.
#   2. Right Streamlit column remains in layout as width placeholder.
#   3. Documents + toggle + Upload + Submit stay fixed.
#   4. Only Uploaded Documents / Indexed KB / Retrieved sources scroll.
#   5. No scroll listeners / no polling loops -> smooth scrolling.
# ============================================================

components.html(
    """
    <script>
    (() => {
        const win = window.parent;
        const doc = win.document;
        const KEY = "__docmindRightSidebarFinal";

        if (win[KEY]?.destroy) {
            try { win[KEY].destroy(); } catch (_) {}
        }

        let resizeObserver = null;
        let resizeRaf = 0;
        let startupTimers = [];

        const setImp = (el, prop, value) => {
            if (!el) return;
            el.style.setProperty(prop, value, "important");
        };

        const clearProps = (el, props) => {
            if (!el) return;
            props.forEach((p) => el.style.removeProperty(p));
        };

        const getParts = () => {
            const dock = doc.querySelector(".st-key-dm_documents_dock");
            if (!dock) return {};

            const column = dock.closest(
                '[data-testid="stColumn"], [data-testid="column"]'
            );

            const header = dock.querySelector(
                ".st-key-dm_documents_fixed_header"
            );

            const body = dock.querySelector(
                ".st-key-right_documents_scroll_container"
            );

            const dockVertical =
                dock.querySelector(':scope > div > [data-testid="stVerticalBlock"]') ||
                dock.querySelector(':scope > div [data-testid="stVerticalBlock"]');

            return { dock, column, header, body, dockVertical };
        };

        const resetMobile = () => {
            const { dock, column, header, body, dockVertical } = getParts();

            clearProps(dock, [
                "position", "top", "left", "right", "bottom",
                "width", "max-width", "height", "max-height",
                "min-height", "overflow", "overflow-y", "overflow-x",
                "display", "flex-direction", "box-sizing",
                "z-index", "margin", "padding-right", "transform",
                "background"
            ]);

            clearProps(column, [
                "position", "overflow", "align-self", "transform"
            ]);

            clearProps(dockVertical, [
                "height", "max-height", "min-height",
                "display", "flex-direction", "overflow", "gap"
            ]);

            clearProps(header, [
                "flex", "height", "max-height", "min-height",
                "position", "z-index", "overflow", "background"
            ]);

            clearProps(body, [
                "flex", "height", "max-height", "min-height",
                "overflow-y", "overflow-x", "position",
                "box-sizing", "padding-bottom", "padding-right"
            ]);
        };

        const layout = () => {
            const { dock, column, header, body, dockVertical } = getParts();

            if (!dock || !column || !header || !body) {
                return false;
            }

            if (win.innerWidth <= 900) {
                resetMobile();
                return true;
            }

            const rect = column.getBoundingClientRect();

            if (!rect.width || rect.width < 180) {
                return false;
            }

            const top = 76;
            const bottomGap = 8;
            const sideInset = 6;

            const left = Math.round(rect.left + sideInset);
            const width = Math.max(
                280,
                Math.round(rect.width - sideInset * 2)
            );

            const dockHeight = Math.max(
                360,
                win.innerHeight - top - bottomGap
            );

            // Keep the Streamlit column in layout as a width placeholder.
            setImp(column, "position", "relative");
            setImp(column, "overflow", "visible");
            setImp(column, "align-self", "flex-start");
            setImp(column, "transform", "none");

            // True fixed right sidebar.
            setImp(dock, "position", "fixed");
            setImp(dock, "top", `${top}px`);
            setImp(dock, "left", `${left}px`);
            setImp(dock, "right", "auto");
            setImp(dock, "bottom", "auto");
            setImp(dock, "width", `${width}px`);
            setImp(dock, "max-width", `${width}px`);
            setImp(dock, "height", `${dockHeight}px`);
            setImp(dock, "max-height", `${dockHeight}px`);
            setImp(dock, "min-height", "0");
            setImp(dock, "overflow", "hidden");
            setImp(dock, "box-sizing", "border-box");
            setImp(dock, "z-index", "999");
            setImp(dock, "margin", "0");
            setImp(dock, "padding-right", "0");
            setImp(dock, "transform", "none");
            setImp(dock, "background", "var(--dm-bg)");

            // Make the actual Streamlit vertical wrapper a proper column.
            if (dockVertical) {
                setImp(dockVertical, "height", "100%");
                setImp(dockVertical, "max-height", "100%");
                setImp(dockVertical, "min-height", "0");
                setImp(dockVertical, "display", "flex");
                setImp(dockVertical, "flex-direction", "column");
                setImp(dockVertical, "overflow", "hidden");
                setImp(dockVertical, "gap", "0.35rem");
            }

            // Fixed top controls.
            setImp(header, "flex", "0 0 auto");
            setImp(header, "height", "auto");
            setImp(header, "max-height", "none");
            setImp(header, "min-height", "0");
            setImp(header, "position", "relative");
            setImp(header, "z-index", "20");
            setImp(header, "overflow", "visible");
            setImp(header, "background", "var(--dm-bg)");

            // IMPORTANT: measure after header styles are applied.
            const headerHeight = Math.ceil(header.getBoundingClientRect().height);
            const gap = 8;

            const bodyHeight = Math.max(
                160,
                dockHeight - headerHeight - gap
            );

            // This is the one and only scroll surface.
            setImp(body, "flex", "0 0 auto");
            setImp(body, "height", `${bodyHeight}px`);
            setImp(body, "max-height", `${bodyHeight}px`);
            setImp(body, "min-height", `${bodyHeight}px`);
            setImp(body, "overflow-y", "auto");
            setImp(body, "overflow-x", "hidden");
            setImp(body, "position", "relative");
            setImp(body, "box-sizing", "border-box");
            setImp(body, "padding-right", "0.45rem");
            setImp(body, "padding-bottom", "1rem");

            // Do not let old CSS force the body wrappers into clipped heights.
            const inner = body.querySelectorAll(
                ':scope > div, :scope > div > div, [data-testid="stVerticalBlock"]'
            );

            inner.forEach((node) => {
                setImp(node, "height", "auto");
                setImp(node, "max-height", "none");
                setImp(node, "min-height", "0");
                setImp(node, "overflow", "visible");
            });

            return true;
        };

        const scheduleLayout = () => {
            win.cancelAnimationFrame(resizeRaf);
            resizeRaf = win.requestAnimationFrame(layout);
        };

        // Initial attempts only. No permanent interval.
        [0, 80, 200, 450, 900].forEach((delay) => {
            startupTimers.push(
                win.setTimeout(scheduleLayout, delay)
            );
        });

        // Recalculate only when real dimensions change.
        resizeObserver = new win.ResizeObserver(scheduleLayout);

        const observeWhenReady = () => {
            const { column, header } = getParts();

            if (column) resizeObserver.observe(column);
            if (header) resizeObserver.observe(header);
        };

        startupTimers.push(
            win.setTimeout(observeWhenReady, 250)
        );

        win.addEventListener(
            "resize",
            scheduleLayout,
            { passive: true }
        );

        win.visualViewport?.addEventListener(
            "resize",
            scheduleLayout,
            { passive: true }
        );

        win[KEY] = {
            destroy() {
                win.cancelAnimationFrame(resizeRaf);
                resizeObserver?.disconnect();

                startupTimers.forEach((timer) => {
                    win.clearTimeout(timer);
                });

                win.removeEventListener(
                    "resize",
                    scheduleLayout
                );

                win.visualViewport?.removeEventListener(
                    "resize",
                    scheduleLayout
                );
            }
        };
    })();
    </script>
    """,
    height=0,
    width=0,
)

st.markdown(
    """
    <style>
    @media (min-width: 901px) {

        /* Prevent the workspace from clipping the fixed right sidebar. */
        div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {
            overflow: visible !important;
        }

        /* Fixed top area. */
        .st-key-dm_documents_fixed_header,
        div[class*="st-key-dm_documents_fixed_header"] {
            flex: 0 0 auto !important;
            background: var(--dm-bg) !important;
            padding-bottom: .55rem !important;
            border-bottom: 1px solid var(--dm-border) !important;
        }

        /* The lower body is the sole scroll owner. */
        .st-key-right_documents_scroll_container,
        div[class*="st-key-right_documents_scroll_container"] {
            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior-y: contain !important;
            scrollbar-gutter: stable !important;
        }

        /* Ensure final item is reachable. */
        .st-key-right_documents_scroll_container
        [data-testid="stVerticalBlock"] > :last-child {
            margin-bottom: 1.25rem !important;
        }

        .st-key-right_documents_scroll_container::-webkit-scrollbar,
        div[class*="st-key-right_documents_scroll_container"]::-webkit-scrollbar {
            width: 8px !important;
        }

        .st-key-right_documents_scroll_container::-webkit-scrollbar-track,
        div[class*="st-key-right_documents_scroll_container"]::-webkit-scrollbar-track {
            background: transparent !important;
        }

        .st-key-right_documents_scroll_container::-webkit-scrollbar-thumb,
        div[class*="st-key-right_documents_scroll_container"]::-webkit-scrollbar-thumb {
            background: var(--dm-border) !important;
            border-radius: 999px !important;
        }

        .st-key-right_documents_scroll_container::-webkit-scrollbar-thumb:hover,
        div[class*="st-key-right_documents_scroll_container"]::-webkit-scrollbar-thumb:hover {
            background: var(--dm-muted) !important;
        }
    }

    @media (max-width: 900px) {
        .st-key-dm_documents_dock,
        div[class*="st-key-dm_documents_dock"],
        .st-key-dm_documents_fixed_header,
        div[class*="st-key-dm_documents_fixed_header"],
        .st-key-right_documents_scroll_container,
        div[class*="st-key-right_documents_scroll_container"] {
            position: static !important;
            width: 100% !important;
            max-width: 100% !important;
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
            transform: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL GAP FIX — DOCUMENTS HEADER WRAPPER
# ============================================================
# Removes the invisible stretched Streamlit wrapper between
# Upload/Submit and "Uploaded Documents".
# ============================================================

st.markdown(
    """
    <style>
    @media (min-width: 901px) {

        /*
         * The direct Streamlit element that CONTAINS the fixed header
         * must size itself only to the visible header content.
         */
        .st-key-dm_documents_dock
        [data-testid="stVerticalBlock"] > div:has(.st-key-dm_documents_fixed_header),
        div[class*="st-key-dm_documents_dock"]
        [data-testid="stVerticalBlock"] > div:has(.st-key-dm_documents_fixed_header) {
            flex: 0 0 auto !important;

            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;

            margin: 0 !important;
            padding: 0 !important;

            overflow: visible !important;
        }

        /*
         * Every wrapper INSIDE the fixed header must also remain content-sized.
         * Older rules in the file were forcing some of these to 100% height.
         */
        .st-key-dm_documents_fixed_header,
        .st-key-dm_documents_fixed_header > div,
        .st-key-dm_documents_fixed_header > div > div,
        .st-key-dm_documents_fixed_header [data-testid="stVerticalBlock"],
        div[class*="st-key-dm_documents_fixed_header"],
        div[class*="st-key-dm_documents_fixed_header"] > div,
        div[class*="st-key-dm_documents_fixed_header"] > div > div,
        div[class*="st-key-dm_documents_fixed_header"] [data-testid="stVerticalBlock"] {
            flex: 0 0 auto !important;

            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;

            margin-top: 0 !important;
            margin-bottom: 0 !important;

            overflow: visible !important;
        }

        /*
         * The Streamlit element that contains the scroll body must consume
         * the remaining sidebar space instead of being pushed downward.
         */
        .st-key-dm_documents_dock
        [data-testid="stVerticalBlock"] > div:has(.st-key-right_documents_scroll_container),
        div[class*="st-key-dm_documents_dock"]
        [data-testid="stVerticalBlock"] > div:has(.st-key-right_documents_scroll_container) {
            flex: 1 1 auto !important;

            min-height: 0 !important;
            height: auto !important;
            max-height: none !important;

            margin: 0 !important;
            padding: 0 !important;

            overflow: hidden !important;
        }

        /*
         * Scroll content begins directly under the fixed controls.
         */
        .st-key-right_documents_scroll_container,
        div[class*="st-key-right_documents_scroll_container"] {
            margin-top: 0 !important;
            padding-top: .35rem !important;
        }

        /*
         * Remove any accidental spacer immediately between the fixed header
         * and the document scroll body.
         */
        .st-key-dm_documents_dock
        [data-testid="stVerticalBlock"] > div:empty {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

components.html(
    """
    <script>
    (() => {
        const win = window.parent;
        const doc = win.document;

        const fixGap = () => {
            if (win.innerWidth <= 900) return;

            const dock = doc.querySelector(".st-key-dm_documents_dock");
            const header = doc.querySelector(".st-key-dm_documents_fixed_header");
            const body = doc.querySelector(".st-key-right_documents_scroll_container");

            if (!dock || !header || !body) return;

            // Normalize the Streamlit element wrappers that contain
            // the header and scroll body.
            const headerElement = header.closest('[data-testid="stElementContainer"]')
                || header.parentElement;

            const bodyElement = body.closest('[data-testid="stElementContainer"]')
                || body.parentElement;

            if (headerElement) {
                headerElement.style.setProperty("height", "auto", "important");
                headerElement.style.setProperty("min-height", "0", "important");
                headerElement.style.setProperty("max-height", "none", "important");
                headerElement.style.setProperty("flex", "0 0 auto", "important");
                headerElement.style.setProperty("margin", "0", "important");
            }

            if (bodyElement) {
                bodyElement.style.setProperty("flex", "1 1 auto", "important");
                bodyElement.style.setProperty("min-height", "0", "important");
                bodyElement.style.setProperty("margin", "0", "important");
            }

            // Recalculate lower scroll height from the ACTUAL visible header.
            const dockHeight = dock.clientHeight;
            const headerHeight = Math.ceil(header.getBoundingClientRect().height);

            const available = Math.max(
                160,
                dockHeight - headerHeight - 8
            );

            body.style.setProperty("height", `${available}px`, "important");
            body.style.setProperty("min-height", `${available}px`, "important");
            body.style.setProperty("max-height", `${available}px`, "important");
            body.style.setProperty("overflow-y", "auto", "important");
            body.style.setProperty("overflow-x", "hidden", "important");
        };

        [0, 100, 300, 700].forEach((delay) => {
            win.setTimeout(fixGap, delay);
        });

        win.addEventListener("resize", fixGap, { passive: true });
    })();
    </script>
    """,
    height=0,
    width=0,
)

# ============================================================
# FINAL HARD FIX — KEEP CHAT COMPOSER FIXED TO VIEWPORT
# ============================================================
# This override intentionally comes last so older experimental CSS cannot
# move the prompt box when the main page scrolls.
st.markdown(
    """
    <style>
    /* Desktop/tablet: the complete composer (input + disclaimer) is pinned
       to the browser viewport, not to the scrolling Streamlit page. */
    @media (min-width: 901px) {
        .st-key-dm_center_chat .st-key-dm_center_composer {
            position: fixed !important;
            left: var(--dm-composer-left, 0px) !important;
            width: var(--dm-composer-width, 100%) !important;
            right: auto !important;
            top: auto !important;
            bottom: var(--dm-composer-bottom, 0px) !important;
            margin: 0 !important;
            padding: .65rem .6rem max(.6rem, env(safe-area-inset-bottom)) !important;
            box-sizing: border-box !important;
            background: var(--dm-bg) !important;
            border-top: 1px solid color-mix(in srgb, var(--dm-border) 70%, transparent) !important;
            z-index: 9990 !important;
        }

        /* Leave enough room inside the scrollable conversation so the final
           answer/source card is never hidden behind the fixed composer. */
        .st-key-dm_center_chat .st-key-dm_conversation_viewport,
        .st-key-dm_center_chat .st-key-agent_v2_scroll {
            padding-bottom: var(--dm-composer-space, 8.5rem) !important;
            scroll-padding-bottom: var(--dm-composer-space, 8.5rem) !important;
        }

        .st-key-dm_center_chat .st-key-agent_v2_input,
        .st-key-dm_center_chat .st-key-agent_v2_input [data-testid="stChatInput"] {
            position: relative !important;
            inset: auto !important;
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
        }

        .st-key-dm_center_chat .dm-agent-v2-disclaimer {
            margin: 0 !important;
            padding: .35rem 0 0 !important;
        }
    }

    /* Mobile keeps the safer sticky behavior so the composer follows the
       mobile viewport without covering the entire narrow layout. */
    @media (max-width: 900px) {
        .st-key-dm_center_chat .st-key-dm_center_composer {
            position: sticky !important;
            left: auto !important;
            right: auto !important;
            bottom: 0 !important;
            width: 100% !important;
            z-index: 9990 !important;
            background: var(--dm-bg) !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Measure the visible center column and apply the coordinates directly to the
# composer. Direct inline !important styles make this resilient to Streamlit's
# generated wrapper classes and to earlier CSS blocks in this file.
components.html(
    """
    <script>
    (() => {
        const win = window.parent;
        const doc = win.document;
        const KEY = "__docmindFixedComposerV3";

        if (win[KEY]?.destroy) {
            try { win[KEY].destroy(); } catch (_) {}
        }

        let raf = 0;
        let resizeObserver = null;
        let mutationObserver = null;
        const timers = [];

        const setImp = (node, prop, value) => {
            if (!node) return;
            node.style.setProperty(prop, value, "important");
        };

        const layout = () => {
            const panel = doc.querySelector(".st-key-dm_center_chat");
            const composer = panel?.querySelector(".st-key-dm_center_composer");
            if (!panel || !composer) return false;

            // On narrow screens, completely leave desktop pinning mode.
            // Chrome responsive mode can keep the old desktop measurements
            // (--dm-composer-left / --dm-composer-width) unless they are
            // explicitly cleared here.
            if (win.innerWidth <= 900) {
                panel.removeAttribute("data-composer-pinned");

                panel.style.removeProperty("--dm-composer-left");
                panel.style.removeProperty("--dm-composer-width");
                panel.style.removeProperty("--dm-composer-bottom");
                panel.style.removeProperty("--dm-composer-space");

                composer.style.removeProperty("position");
                composer.style.removeProperty("left");
                composer.style.removeProperty("right");
                composer.style.removeProperty("top");
                composer.style.removeProperty("bottom");
                composer.style.removeProperty("width");
                composer.style.removeProperty("max-width");
                composer.style.removeProperty("margin");
                composer.style.removeProperty("transform");

                return true;
            }

            const rect = panel.getBoundingClientRect();
            const viewport = win.visualViewport;
            const vpLeft = viewport?.offsetLeft || 0;
            const vpTop = viewport?.offsetTop || 0;
            const vpWidth = viewport?.width || win.innerWidth;
            const vpHeight = viewport?.height || win.innerHeight;

            const left = Math.max(rect.left, vpLeft);
            const right = Math.min(rect.right, vpLeft + vpWidth);
            if (right <= left) return false;

            const keyboardBottom = Math.max(
                0,
                win.innerHeight - (vpTop + vpHeight)
            );

            setImp(panel, "--dm-composer-left", `${left}px`);
            setImp(panel, "--dm-composer-width", `${right - left}px`);
            setImp(panel, "--dm-composer-bottom", `${keyboardBottom}px`);

            setImp(composer, "position", "fixed");
            setImp(composer, "left", `${left}px`);
            setImp(composer, "width", `${right - left}px`);
            setImp(composer, "right", "auto");
            setImp(composer, "top", "auto");
            setImp(composer, "bottom", `${keyboardBottom}px`);
            setImp(composer, "margin", "0");
            setImp(composer, "z-index", "9990");

            const composerHeight = Math.ceil(
                composer.getBoundingClientRect().height
            );
            setImp(
                panel,
                "--dm-composer-space",
                `${composerHeight + 24}px`
            );

            panel.dataset.composerPinned = "true";
            return true;
        };

        const schedule = () => {
            win.cancelAnimationFrame(raf);
            raf = win.requestAnimationFrame(layout);
        };

        // Re-run when Streamlit rerenders widgets or the viewport changes.
        resizeObserver = new win.ResizeObserver(schedule);
        mutationObserver = new win.MutationObserver(schedule);

        const attachObservers = () => {
            const panel = doc.querySelector(".st-key-dm_center_chat");
            const composer = panel?.querySelector(".st-key-dm_center_composer");
            if (panel) resizeObserver.observe(panel);
            if (composer) resizeObserver.observe(composer);
            if (doc.body) {
                mutationObserver.observe(doc.body, {
                    childList: true,
                    subtree: true,
                });
            }
            schedule();
        };

        win.addEventListener("resize", schedule, { passive: true });
        doc.addEventListener("scroll", schedule, true);
        win.visualViewport?.addEventListener("resize", schedule, { passive: true });
        win.visualViewport?.addEventListener("scroll", schedule, { passive: true });

        [0, 80, 200, 500, 1000].forEach((delay) => {
            timers.push(win.setTimeout(attachObservers, delay));
        });

        win[KEY] = {
            destroy() {
                win.cancelAnimationFrame(raf);
                resizeObserver?.disconnect();
                mutationObserver?.disconnect();
                timers.forEach((timer) => win.clearTimeout(timer));
                win.removeEventListener("resize", schedule);
                doc.removeEventListener("scroll", schedule, true);
                win.visualViewport?.removeEventListener("resize", schedule);
                win.visualViewport?.removeEventListener("scroll", schedule);
            }
        };
    })();
    </script>
    """,
    height=0,
    width=0,
)


# ============================================================
# FINAL CHAT PRESENTATION OVERRIDES
# Clean ChatGPT-like answer presentation + compact thinking state
# ============================================================
st.markdown(
    """
    <style>
    /* Conversation turns: no cards / no boxed message backgrounds. */
    .st-key-dm_conversation_viewport [data-testid="stChatMessage"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        padding: 0.45rem 0.15rem 1.05rem 0.15rem !important;
        margin: 0 !important;
    }

    .st-key-dm_conversation_viewport [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
    .st-key-dm_conversation_viewport [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
        width: 2rem !important;
        height: 2rem !important;
        border-radius: 0.65rem !important;
        background: var(--dm-surface2) !important;
        border: 1px solid var(--dm-border) !important;
    }

    .st-key-dm_conversation_viewport [data-testid="stChatMessageContent"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding-top: 0.05rem !important;
        color: var(--dm-text) !important;
        max-width: 100% !important;
    }

    .st-key-dm_conversation_viewport [data-testid="stChatMessageContent"] p,
    .st-key-dm_conversation_viewport [data-testid="stChatMessageContent"] li {
        color: var(--dm-text) !important;
        font-size: 0.98rem !important;
        line-height: 1.7 !important;
    }

    .st-key-dm_conversation_viewport [data-testid="stChatMessageContent"] h1,
    .st-key-dm_conversation_viewport [data-testid="stChatMessageContent"] h2,
    .st-key-dm_conversation_viewport [data-testid="stChatMessageContent"] h3 {
        color: var(--dm-heading) !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.65rem !important;
    }

    /* Give each completed turn a subtle separator like the reference. */
    .st-key-dm_conversation_viewport [data-testid="stChatMessage"]:has(+ [data-testid="stChatMessage"]) {
        border-bottom: 1px solid var(--dm-border) !important;
    }

    /* Compact Streamlit spinner: no white status bar/card. */
    [data-testid="stSpinner"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0.55rem 0 !important;
        color: var(--dm-muted) !important;
    }

    [data-testid="stSpinner"] * {
        color: var(--dm-muted) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FINAL SIDEBAR CURRENT CHAT COLOR FIX
# ============================================================
# The current conversation uses the Streamlit key `current_chat_nav`.
# Older CSS targeted `history_nav_*`, so Streamlit's default focused/active
# button style could still turn the current chat white. Keep this block last
# so it wins the CSS cascade in both dark and light themes.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button,
    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button:hover,
    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button:focus,
    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button:focus-visible,
    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button:active {
        width: 100% !important;
        background: var(--dm-sidebar-hover) !important;
        background-color: var(--dm-sidebar-hover) !important;
        color: var(--dm-sidebar-text) !important;
        border: 1px solid var(--dm-sidebar-border) !important;
        border-radius: 9px !important;
        box-shadow: none !important;
        opacity: 1 !important;
        filter: none !important;
    }

    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button:hover,
    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button:focus,
    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button:focus-visible {
        border-color: var(--dm-primary) !important;
    }

    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button *,
    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button p,
    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button span,
    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button div {
        color: var(--dm-sidebar-text) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Prevent browser/Streamlit autofill-like focus paint from flashing white. */
    [data-testid="stSidebar"] [class*="st-key-current_chat_nav"] button:-webkit-focus-ring-color {
        outline-color: var(--dm-primary) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STABLE DOCMIND CHAT WORKSPACE
# ============================================================
# One layout owns the center area:
#   Header (fixed)
#   Conversation (only scrollable region)
#   Composer (fixed inside center column, NOT position:fixed)
#
# This deliberately overrides older experimental chat-layout CSS.
st.markdown(
    """
    <style>
    @media (min-width: 901px) {
        /* Prevent the browser/main Streamlit page from becoming the chat scroller. */
        html:has(.st-key-dm_center_chat),
        body:has(.st-key-dm_center_chat),
        [data-testid="stAppViewContainer"]:has(.st-key-dm_center_chat),
        [data-testid="stMain"]:has(.st-key-dm_center_chat) {
            height: 100dvh !important;
            max-height: 100dvh !important;
            overflow: hidden !important;
            overscroll-behavior: none !important;
        }

        [data-testid="stMain"]:has(.st-key-dm_center_chat)
        [data-testid="stMainBlockContainer"],
        [data-testid="stMain"]:has(.st-key-dm_center_chat)
        .block-container {
            height: 100dvh !important;
            max-height: 100dvh !important;
            min-height: 0 !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
            padding-bottom: .65rem !important;
        }

        /* Keep the three-column workspace inside the visible screen. */
        [data-testid="stMain"]:has(.st-key-dm_center_chat)
        div[data-testid="stHorizontalBlock"]:has(.dm-right-panel-marker) {
            height: calc(100dvh - 5.15rem) !important;
            max-height: calc(100dvh - 5.15rem) !important;
            min-height: 0 !important;
            overflow: hidden !important;
            align-items: stretch !important;
        }

        /* Center column wrapper must be allowed to shrink. */
        [data-testid="stMain"]:has(.st-key-dm_center_chat)
        :is([data-testid="column"], [data-testid="stColumn"]):has(.st-key-dm_center_chat),
        [data-testid="stMain"]:has(.st-key-dm_center_chat)
        :is([data-testid="column"], [data-testid="stColumn"]):has(.st-key-dm_center_chat) > div,
        [data-testid="stMain"]:has(.st-key-dm_center_chat)
        :is([data-testid="column"], [data-testid="stColumn"]):has(.st-key-dm_center_chat)
        > div > [data-testid="stVerticalBlock"] {
            height: 100% !important;
            max-height: 100% !important;
            min-height: 0 !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
        }

        /* Header + chat viewport + composer. */
        .st-key-dm_center_chat {
            display: flex !important;
            flex-direction: column !important;
            width: 100% !important;
            height: 100% !important;
            max-height: 100% !important;
            min-height: 0 !important;
            overflow: hidden !important;
            gap: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            --dm-composer-space: 0px !important;
        }

        .st-key-dm_center_chat
        > div:has(.st-key-dm_center_header):has(.st-key-dm_center_composer),
        .st-key-dm_center_chat
        > div
        > [data-testid="stVerticalBlock"]:has(.st-key-dm_center_header):has(.st-key-dm_center_composer) {
            display: flex !important;
            flex-direction: column !important;
            flex: 1 1 auto !important;
            width: 100% !important;
            height: 100% !important;
            max-height: 100% !important;
            min-height: 0 !important;
            overflow: hidden !important;
            gap: 0 !important;
        }

        /* DocMind title does not scroll. */
        .st-key-dm_center_chat .st-key-dm_center_header {
            flex: 0 0 auto !important;
            min-height: 0 !important;
            height: auto !important;
            overflow: visible !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* ONLY the conversation scrolls. */
        .st-key-dm_center_chat .st-key-dm_conversation_viewport {
            flex: 1 1 0 !important;
            min-height: 0 !important;
            height: 0 !important;
            max-height: none !important;
            width: 100% !important;

            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior-y: contain !important;
            touch-action: pan-y !important;
            scrollbar-gutter: stable !important;

            margin: 0 !important;
            padding: .65rem .75rem .55rem !important;
            box-sizing: border-box !important;
        }

        /* Let the message stack grow naturally so older turns remain reachable. */
        .st-key-dm_center_chat .st-key-dm_conversation_viewport > div,
        .st-key-dm_center_chat
        .st-key-dm_conversation_viewport > div > [data-testid="stVerticalBlock"] {
            width: 100% !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
            box-sizing: border-box !important;
        }

        /*
         * Short chats stay close to the composer.
         * Long chats exceed this minimum and therefore become scrollable.
         */
        .st-key-dm_center_chat
        .st-key-dm_conversation_viewport > div > [data-testid="stVerticalBlock"] {
            min-height: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
        }

        /* Empty/new-chat state must be centered and completely visible. */
        .st-key-dm_center_chat
        .st-key-dm_conversation_viewport:has(.dm-agent-v2-empty)
        > div > [data-testid="stVerticalBlock"] {
            justify-content: center !important;
        }

        .st-key-dm_center_chat .dm-agent-v2-empty {
            min-height: 0 !important;
            height: auto !important;
            padding: 1rem 1rem .8rem !important;
            margin: 0 auto !important;
            overflow: visible !important;
            visibility: visible !important;
            opacity: 1 !important;
        }

        .st-key-dm_center_chat .dm-agent-v2-empty-icon,
        .st-key-dm_center_chat .dm-agent-v2-empty-title,
        .st-key-dm_center_chat .dm-agent-v2-empty-text,
        .st-key-dm_center_chat .dm-agent-v2-example-title {
            visibility: visible !important;
            opacity: 1 !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        .st-key-dm_center_chat .dm-agent-v2-example-title {
            margin: .35rem 0 .55rem !important;
        }

        .st-key-dm_center_chat
        .st-key-dm_conversation_viewport:has(.dm-agent-v2-empty)
        [data-testid="stHorizontalBlock"] {
            flex: 0 0 auto !important;
            overflow: visible !important;
        }

        /* Remove oversized inter-turn spacing. */
        .st-key-dm_center_chat .dm-agent-v2-turn-gap {
            height: .45rem !important;
            min-height: .45rem !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Composer is the fixed bottom row of the center column. */
        .st-key-dm_center_chat .st-key-dm_center_composer,
        .st-key-dm_center_chat[data-composer-pinned="true"] .st-key-dm_center_composer {
            position: relative !important;
            inset: auto !important;
            left: auto !important;
            right: auto !important;
            top: auto !important;
            bottom: auto !important;

            flex: 0 0 auto !important;
            width: 100% !important;
            max-width: 100% !important;
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;

            margin: 0 !important;
            padding: .55rem .55rem max(.4rem, env(safe-area-inset-bottom)) !important;
            overflow: visible !important;
            box-sizing: border-box !important;

            background: var(--dm-bg) !important;
            border-top: 1px solid var(--dm-border) !important;
            z-index: 20 !important;
        }

        .st-key-dm_center_chat .st-key-dm_center_composer > div,
        .st-key-dm_center_chat
        .st-key-dm_center_composer [data-testid="stVerticalBlock"] {
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
            gap: .2rem !important;
        }

        .st-key-dm_center_chat .st-key-agent_v2_input,
        .st-key-dm_center_chat
        .st-key-agent_v2_input [data-testid="stChatInput"] {
            position: relative !important;
            inset: auto !important;
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
        }

        /* Latest answer/thinking row sits immediately above the composer. */
        .st-key-dm_center_chat
        .st-key-dm_conversation_viewport [data-testid="stChatMessage"]:last-of-type {
            margin-bottom: .2rem !important;
            padding-bottom: .2rem !important;
        }

        .st-key-dm_center_chat .dm-agent-v2-thinking {
            margin: .15rem 0 .35rem .1rem !important;
            padding: .2rem 0 !important;
            flex: 0 0 auto !important;
        }

        /* Visible conversation scrollbar. */
        .st-key-dm_center_chat
        .st-key-dm_conversation_viewport::-webkit-scrollbar {
            width: 8px !important;
        }

        .st-key-dm_center_chat
        .st-key-dm_conversation_viewport::-webkit-scrollbar-track {
            background: transparent !important;
        }

        .st-key-dm_center_chat
        .st-key-dm_conversation_viewport::-webkit-scrollbar-thumb {
            background: var(--dm-border) !important;
            border-radius: 999px !important;
            min-height: 34px !important;
        }
    }

    @media (max-width: 900px) {
        /* Mobile keeps normal document flow, but chat history can still scroll. */
        .st-key-dm_center_chat .st-key-dm_conversation_viewport {
            max-height: 58dvh !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }

        .st-key-dm_center_chat .st-key-dm_center_composer {
            position: sticky !important;
            bottom: 0 !important;
            background: var(--dm-bg) !important;
            z-index: 30 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ChatGPT-like scroll behavior:
# - Empty state stays at the top/center and is never auto-scrolled away.
# - After submitting a new question, the chat opens at the latest turn.
# - If the user scrolls upward to read history, we do not force them down.
components.html(
    """
    <script>
    (() => {
        const win = window.parent;
        const doc = win.document;
        const KEY = "__docmindStableScroller";

        if (win[KEY]?.destroy) {
            try { win[KEY].destroy(); } catch (_) {}
        }

        let viewport = null;
        let observer = null;
        let raf = 0;
        let userAwayFromBottom = false;
        let programmatic = false;
        const timers = [];

        const findViewport = () =>
            doc.querySelector(
                ".st-key-dm_center_chat .st-key-dm_conversation_viewport"
            );

        const emptyState = () =>
            !!viewport?.querySelector(".dm-agent-v2-empty");

        const nearBottom = (threshold = 100) => {
            if (!viewport) return true;
            return (
                viewport.scrollHeight -
                viewport.scrollTop -
                viewport.clientHeight
            ) <= threshold;
        };

        const goBottom = () => {
            if (!viewport || emptyState()) return;

            programmatic = true;
            win.cancelAnimationFrame(raf);
            raf = win.requestAnimationFrame(() => {
                viewport.scrollTop = viewport.scrollHeight;
                win.setTimeout(() => {
                    programmatic = false;
                    userAwayFromBottom = false;
                }, 30);
            });
        };

        const onScroll = () => {
            if (!viewport || programmatic || emptyState()) return;
            userAwayFromBottom = !nearBottom(80);
        };

        const attach = () => {
            const next = findViewport();
            if (!next) return;

            if (viewport && viewport !== next) {
                viewport.removeEventListener("scroll", onScroll);
            }

            viewport = next;
            viewport.addEventListener("scroll", onScroll, { passive: true });

            observer?.disconnect();
            observer = new win.MutationObserver(() => {
                if (emptyState()) {
                    viewport.scrollTop = 0;
                    return;
                }

                if (!userAwayFromBottom && nearBottom(160)) {
                    goBottom();
                }
            });

            observer.observe(viewport, {
                childList: true,
                subtree: true,
                characterData: true,
            });

            if (emptyState()) {
                viewport.scrollTop = 0;
            } else {
                goBottom();
            }
        };

        [0, 80, 180, 350, 700].forEach((delay) => {
            timers.push(win.setTimeout(attach, delay));
        });

        win[KEY] = {
            destroy() {
                win.cancelAnimationFrame(raf);
                observer?.disconnect();
                if (viewport) {
                    viewport.removeEventListener("scroll", onScroll);
                }
                timers.forEach((timer) => win.clearTimeout(timer));
            }
        };
    })();
    </script>
    """,
    height=0,
    width=0,
)


# ============================================================
# FINAL NATIVE CHAT SCROLL OVERRIDE
# ============================================================
st.markdown(
    """
    <style>
    @media (min-width: 901px) {
        /* Keep the outer Chat workspace fixed. */
        html:has(.st-key-dm_center_chat),
        body:has(.st-key-dm_center_chat),
        [data-testid="stAppViewContainer"]:has(.st-key-dm_center_chat),
        [data-testid="stMain"]:has(.st-key-dm_center_chat) {
            height: 100dvh !important;
            max-height: 100dvh !important;
            overflow: hidden !important;
        }

        /* Center column is a fixed-height vertical layout. */
        .st-key-dm_center_chat {
            height: calc(100dvh - 5.2rem) !important;
            max-height: calc(100dvh - 5.2rem) !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
            padding: 0 !important;
            margin: 0 !important;
            gap: 0 !important;
            --dm-composer-space: 0px !important;
        }

        .st-key-dm_center_chat
        > div:has(.st-key-dm_center_header):has(.st-key-dm_center_composer),
        .st-key-dm_center_chat
        > div
        > [data-testid="stVerticalBlock"]:has(.st-key-dm_center_header):has(.st-key-dm_center_composer) {
            height: 100% !important;
            max-height: 100% !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
            gap: 0 !important;
        }

        .st-key-dm_center_header {
            flex: 0 0 auto !important;
        }

        /*
         * Streamlit's native height=520 container remains the ONLY
         * scrollable chat/history surface. We resize it responsively.
         */
        .st-key-dm_conversation_viewport,
        .st-key-dm_empty_state_viewport {
            flex: 1 1 auto !important;
            height: auto !important;
            max-height: none !important;
            min-height: 0 !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior-y: contain !important;
            scrollbar-gutter: stable !important;
            padding: .6rem .7rem !important;
            box-sizing: border-box !important;
        }

        /*
         * IMPORTANT: do not force Streamlit's inner wrappers to 0px height.
         * They must grow with content so older messages are scrollable.
         */
        .st-key-dm_conversation_viewport > div,
        .st-key-dm_conversation_viewport [data-testid="stVerticalBlock"],
        .st-key-dm_empty_state_viewport > div,
        .st-key-dm_empty_state_viewport [data-testid="stVerticalBlock"] {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        /* Empty/new-chat screen is always visible. */
        .st-key-dm_empty_state_viewport {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
        }

        .st-key-dm_empty_state_viewport > div,
        .st-key-dm_empty_state_viewport [data-testid="stVerticalBlock"] {
            width: 100% !important;
        }

        .st-key-dm_empty_state_viewport .dm-agent-v2-empty,
        .st-key-dm_empty_state_viewport .dm-agent-v2-empty-icon,
        .st-key-dm_empty_state_viewport .dm-agent-v2-empty-title,
        .st-key-dm_empty_state_viewport .dm-agent-v2-empty-text,
        .st-key-dm_empty_state_viewport .dm-agent-v2-example-title {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: relative !important;
            inset: auto !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        .st-key-dm_empty_state_viewport .dm-agent-v2-empty {
            min-height: 220px !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
            padding: 1rem !important;
        }

        .st-key-dm_empty_state_viewport .dm-agent-v2-empty-icon {
            width: 54px !important;
            height: 54px !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .st-key-dm_empty_state_viewport .dm-agent-v2-empty-title,
        .st-key-dm_empty_state_viewport .dm-agent-v2-empty-text,
        .st-key-dm_empty_state_viewport .dm-agent-v2-example-title {
            display: block !important;
        }

        .st-key-dm_empty_state_viewport .dm-agent-v2-example-title {
            margin: .35rem 0 .55rem !important;
        }

        /* Composer is the bottom row, never an overlay. */
        .st-key-dm_center_composer,
        .st-key-dm_center_chat[data-composer-pinned="true"] .st-key-dm_center_composer {
            position: relative !important;
            inset: auto !important;
            left: auto !important;
            right: auto !important;
            top: auto !important;
            bottom: auto !important;
            flex: 0 0 auto !important;
            width: 100% !important;
            margin: 0 !important;
            padding: .55rem .55rem .4rem !important;
            background: var(--dm-bg) !important;
            border-top: 1px solid var(--dm-border) !important;
            z-index: 10 !important;
        }

        .st-key-dm_center_composer .st-key-agent_v2_input,
        .st-key-dm_center_composer [data-testid="stChatInput"] {
            position: relative !important;
            inset: auto !important;
            width: 100% !important;
            max-width: 100% !important;
        }

        .st-key-dm_conversation_viewport::-webkit-scrollbar {
            width: 8px !important;
        }

        .st-key-dm_conversation_viewport::-webkit-scrollbar-thumb {
            background: var(--dm-border) !important;
            border-radius: 999px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LATEST ANSWER WINDOW
# ------------------------------------------------------------
# Keep a compact visible conversation window above the composer.
# When a new answer is generated, the viewport automatically stays at
# the bottom so only the newest portion of a long answer is visible.
# The rest of the answer and older messages remain available by
# scrolling upward inside the chat area.
# ============================================================

st.markdown(
    """
    <style>
    @media (min-width: 901px) {

        /*
         * Reserve a comfortable ChatGPT-like reading window above the
         * composer instead of allowing the message area to consume the
         * entire center column.
         */
        .st-key-dm_conversation_viewport {
            flex: 0 1 auto !important;
            height: min(52dvh, 470px) !important;
            min-height: 300px !important;
            max-height: min(52dvh, 470px) !important;

            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior-y: contain !important;
            scrollbar-gutter: stable !important;

            margin-top: auto !important;
            padding: .6rem .75rem .55rem !important;
            box-sizing: border-box !important;
        }

        /*
         * Let the message stack grow beyond the visible window.
         * This is what makes the remaining part of long answers
         * available through internal scrolling.
         */
        .st-key-dm_conversation_viewport > div,
        .st-key-dm_conversation_viewport [data-testid="stVerticalBlock"] {
            height: auto !important;
            min-height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        /*
         * Keep the newest turn close to the composer.
         */
        .st-key-dm_conversation_viewport
        [data-testid="stChatMessage"]:last-of-type {
            margin-bottom: .15rem !important;
            padding-bottom: .15rem !important;
        }

        /*
         * Empty state should still use the full remaining center area.
         */
        .st-key-dm_empty_state_viewport {
            flex: 1 1 auto !important;
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            margin-top: 0 !important;
        }
    }

    @media (max-width: 900px) {
        .st-key-dm_conversation_viewport {
            height: 48dvh !important;
            min-height: 260px !important;
            max-height: 48dvh !important;
            overflow-y: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Make sure the newest generated answer opens at the bottom of the
# internal conversation window. The user can immediately scroll upward
# to see the rest of the answer or older turns.
components.html(
    """
    <script>
    (() => {
        const win = window.parent;
        const doc = win.document;
        const KEY = "__docmindLatestAnswerWindowV1";

        if (win[KEY]?.destroy) {
            try { win[KEY].destroy(); } catch (_) {}
        }

        let observer = null;
        let viewport = null;
        let userReadingHistory = false;
        let programmatic = false;
        const timers = [];

        const getViewport = () =>
            doc.querySelector(".st-key-dm_conversation_viewport");

        const nearBottom = (el, threshold = 100) =>
            el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;

        const goBottom = () => {
            if (!viewport) return;

            programmatic = true;
            viewport.scrollTop = viewport.scrollHeight;

            win.setTimeout(() => {
                programmatic = false;
                userReadingHistory = false;
            }, 40);
        };

        const onScroll = () => {
            if (!viewport || programmatic) return;
            userReadingHistory = !nearBottom(viewport, 80);
        };

        const attach = () => {
            const next = getViewport();
            if (!next) return;

            if (viewport && viewport !== next) {
                viewport.removeEventListener("scroll", onScroll);
            }

            viewport = next;
            viewport.addEventListener("scroll", onScroll, { passive: true });

            observer?.disconnect();
            observer = new win.MutationObserver(() => {
                /*
                 * Follow the answer while it is being generated only when
                 * the user has not intentionally scrolled upward.
                 */
                if (!userReadingHistory && nearBottom(viewport, 150)) {
                    goBottom();
                }
            });

            observer.observe(viewport, {
                childList: true,
                subtree: true,
                characterData: true,
            });

            /* Open the completed/newest answer at its latest portion. */
            goBottom();
        };

        [0, 80, 180, 350, 700].forEach((delay) => {
            timers.push(win.setTimeout(attach, delay));
        });

        win[KEY] = {
            destroy() {
                observer?.disconnect();
                if (viewport) {
                    viewport.removeEventListener("scroll", onScroll);
                }
                timers.forEach((timer) => win.clearTimeout(timer));
            }
        };
    })();
    </script>
    """,
    height=0,
    width=0,
)


# ============================================================
# LATEST ANSWER — SHOW FROM THE BEGINNING
# ------------------------------------------------------------
# When a new answer is ready, position the internal chat viewport at
# the START of the latest conversation turn instead of at its end.
# The answer begins directly in the visible area above the composer,
# and the user scrolls DOWN to read the remaining part.
# ============================================================

st.markdown(
    """
    <style>
    @media (min-width: 901px) {
        /* Keep a clear reading window immediately above the composer. */
        .st-key-dm_conversation_viewport {
            height: min(52dvh, 470px) !important;
            min-height: 300px !important;
            max-height: min(52dvh, 470px) !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior-y: contain !important;
            scrollbar-gutter: stable !important;
            margin-top: auto !important;
            scroll-padding-top: .5rem !important;
            scroll-padding-bottom: .5rem !important;
        }

        /* Do not bottom-align long completed answers. */
        .st-key-dm_conversation_viewport > div,
        .st-key-dm_conversation_viewport [data-testid="stVerticalBlock"] {
            justify-content: flex-start !important;
        }

        /* Empty/new-chat screen remains centered. */
        .st-key-dm_empty_state_viewport > div,
        .st-key-dm_empty_state_viewport [data-testid="stVerticalBlock"] {
            justify-content: center !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Replace bottom-follow behavior for completed answers:
# - locate the newest assistant chat message
# - scroll the INTERNAL chat viewport so that message begins at the top
# - never scroll the outer Streamlit page
# - after that, the user scrolls downward to read the rest
components.html(
    """
    <script>
    (() => {
        const win = window.parent;
        const doc = win.document;
        const KEY = "__docmindLatestAnswerStartV2";

        if (win[KEY]?.destroy) {
            try { win[KEY].destroy(); } catch (_) {}
        }

        const timers = [];
        let observer = null;
        let viewport = null;
        let lastAssistantSignature = "";

        const getViewport = () =>
            doc.querySelector(".st-key-dm_conversation_viewport");

        const getChatMessages = () => {
            if (!viewport) return [];
            return Array.from(
                viewport.querySelectorAll('[data-testid="stChatMessage"]')
            );
        };

        const getLatestAssistant = () => {
            const messages = getChatMessages();

            /*
             * In this app messages are rendered user, assistant, user,
             * assistant... so the last stChatMessage after generation is
             * normally the latest conversation turn.
             */
            return messages.length ? messages[messages.length - 1] : null;
        };

        const signatureFor = (node) => {
            if (!node) return "";
            return (node.innerText || "").trim().slice(0, 240);
        };

        const scrollLatestAnswerToStart = () => {
            if (!viewport) return;

            const latest = getLatestAssistant();
            if (!latest) return;

            const viewportRect = viewport.getBoundingClientRect();
            const latestRect = latest.getBoundingClientRect();

            /*
             * Convert the latest answer's current screen position into the
             * viewport's own scroll coordinate. A tiny offset keeps breathing
             * room above the message.
             */
            const target =
                viewport.scrollTop +
                (latestRect.top - viewportRect.top) -
                8;

            viewport.scrollTop = Math.max(0, target);
        };

        const attach = () => {
            viewport = getViewport();
            if (!viewport) return;

            observer?.disconnect();

            /*
             * On initial attach/rerun, show the newest completed answer
             * FROM ITS BEGINNING.
             */
            const latest = getLatestAssistant();
            if (latest) {
                lastAssistantSignature = signatureFor(latest);
                win.requestAnimationFrame(scrollLatestAnswerToStart);
            }

            observer = new win.MutationObserver(() => {
                const newest = getLatestAssistant();
                if (!newest) return;

                const signature = signatureFor(newest);

                /*
                 * Only reposition when a NEW assistant response appears.
                 * Do not keep forcing scroll position while the user is
                 * manually reading the answer.
                 */
                if (signature && signature !== lastAssistantSignature) {
                    lastAssistantSignature = signature;

                    win.setTimeout(() => {
                        scrollLatestAnswerToStart();
                    }, 60);

                    win.setTimeout(() => {
                        scrollLatestAnswerToStart();
                    }, 180);
                }
            });

            observer.observe(viewport, {
                childList: true,
                subtree: true,
                characterData: true,
            });
        };

        [0, 80, 180, 350, 700].forEach((delay) => {
            timers.push(win.setTimeout(attach, delay));
        });

        win[KEY] = {
            destroy() {
                observer?.disconnect();
                timers.forEach((timer) => win.clearTimeout(timer));
            }
        };
    })();
    </script>
    """,
    height=0,
    width=0,
)


# ============================================================
# AUTHORITATIVE NEWEST-ANSWER POSITION
# ============================================================
st.markdown(
    """
    <style>
    @media (min-width: 901px) {
        .st-key-dm_conversation_viewport {
            overflow-y: auto !important;
            overflow-x: hidden !important;
            scroll-behavior: auto !important;
            scroll-padding-top: .5rem !important;
        }

        /* Never bottom-align the completed conversation stack. */
        .st-key-dm_conversation_viewport > div,
        .st-key-dm_conversation_viewport > div > [data-testid="stVerticalBlock"],
        .st-key-dm_conversation_viewport [data-testid="stVerticalBlock"] {
            justify-content: flex-start !important;
        }

        #dm-latest-turn-start {
            display: block !important;
            height: 1px !important;
            min-height: 1px !important;
            margin: 0 !important;
            padding: 0 !important;
            visibility: visible !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

components.html(
    """
    <script>
    (() => {
        const win = window.parent;
        const doc = win.document;
        const KEY = "__docmindExactFirstLineV1";

        const stopOldAutoScrollers = () => {
            [
                "__docmindStableScroller",
                "__docmindLatestAnswerWindowV1",
                "__docmindLatestAnswerStartV2",
                "__docmindAnswerStartExactV1",
                "__docmindChatOnlyScrollerV1",
                "__docmindInternalChatScrollV2",
                "__docmindChatViewportFinalV1",
                "__docmindHistoryScrollFixV3",
                "__docmindUnifiedChatScrollerV4"
            ].forEach((key) => {
                if (win[key]?.destroy) {
                    try { win[key].destroy(); } catch (_) {}
                }
                try { delete win[key]; } catch (_) {}
            });
        };

        if (win[KEY]?.destroy) {
            try { win[KEY].destroy(); } catch (_) {}
        }

        let viewport = null;
        let observer = null;
        let userHasScrolled = false;
        let programmatic = false;
        const timers = [];

        const getViewport = () =>
            doc.querySelector(".st-key-dm_conversation_viewport");

        const getMarker = () =>
            doc.querySelector("#dm-latest-turn-start");

        const positionAtFirstLine = () => {
            viewport = getViewport();
            const marker = getMarker();

            if (!viewport || !marker) return;

            const viewportRect = viewport.getBoundingClientRect();
            const markerRect = marker.getBoundingClientRect();

            const target =
                viewport.scrollTop +
                (markerRect.top - viewportRect.top) -
                8;

            programmatic = true;
            viewport.scrollTop = Math.max(0, target);

            win.setTimeout(() => {
                programmatic = false;
            }, 50);
        };

        const onScroll = () => {
            if (!programmatic) {
                userHasScrolled = true;
            }
        };

        const attach = () => {
            stopOldAutoScrollers();

            const nextViewport = getViewport();
            if (!nextViewport) return;

            if (viewport && viewport !== nextViewport) {
                viewport.removeEventListener("scroll", onScroll);
            }

            viewport = nextViewport;
            viewport.addEventListener("scroll", onScroll, { passive: true });

            observer?.disconnect();

            /*
             * Initial completed-answer render:
             * show the exact beginning of the newest conversation turn.
             */
            userHasScrolled = false;
            positionAtFirstLine();

            observer = new win.MutationObserver(() => {
                /*
                 * During Streamlit's rerender, reposition only until the user
                 * intentionally starts scrolling. After that, never fight them.
                 */
                if (!userHasScrolled && getMarker()) {
                    positionAtFirstLine();
                }
            });

            observer.observe(viewport, {
                childList: true,
                subtree: true,
            });
        };

        [0, 60, 150, 300, 550, 900].forEach((delay) => {
            timers.push(
                win.setTimeout(() => {
                    stopOldAutoScrollers();
                    attach();
                }, delay)
            );
        });

        win[KEY] = {
            destroy() {
                observer?.disconnect();
                if (viewport) {
                    viewport.removeEventListener("scroll", onScroll);
                }
                timers.forEach((timer) => win.clearTimeout(timer));
            }
        };
    })();
    </script>
    """,
    height=0,
    width=0,
)


# ============================================================
# THINKING INDICATOR — DIRECTLY ABOVE COMPOSER
# ============================================================
st.markdown(
    """
    <style>
    /* The standalone thinking row between history and composer. */
    .st-key-dm_center_chat .dm-agent-v2-thinking {
        display: flex !important;
        align-items: center !important;
        gap: .55rem !important;

        width: 100% !important;
        min-height: 32px !important;
        height: auto !important;

        margin: 0 !important;
        padding: .35rem .75rem .4rem !important;
        box-sizing: border-box !important;

        color: var(--dm-muted) !important;
        background: var(--dm-bg) !important;

        position: relative !important;
        inset: auto !important;
        z-index: 25 !important;
    }

    .st-key-dm_center_chat .dm-agent-v2-thinking-dot {
        width: 8px !important;
        height: 8px !important;
        min-width: 8px !important;
        border-radius: 999px !important;
        background: var(--dm-primary) !important;
        display: inline-block !important;
        animation: dm-thinking-pulse 1.15s ease-in-out infinite !important;
    }

    .st-key-dm_center_chat .dm-agent-v2-thinking span:last-child {
        color: var(--dm-muted) !important;
        font-size: .88rem !important;
        line-height: 1.25 !important;
    }

    @keyframes dm-thinking-pulse {
        0%, 100% {
            opacity: .35;
            transform: scale(.85);
        }
        50% {
            opacity: 1;
            transform: scale(1.08);
        }
    }

    /*
     * Keep the composer immediately under the thinking row.
     * When the placeholder is empty it consumes essentially no space.
     */
    .st-key-dm_center_composer {
        margin-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL THINKING INDICATOR VISIBILITY
# ============================================================
st.markdown(
    """
    <style>
    .st-key-dm_center_composer {
        overflow: visible !important;
    }

    .st-key-dm_center_composer > div,
    .st-key-dm_center_composer [data-testid="stVerticalBlock"] {
        overflow: visible !important;
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
    }

    .st-key-dm_center_composer .dm-thinking-inline {
        display: flex !important;
        align-items: center !important;
        gap: .55rem !important;

        width: 100% !important;
        min-height: 34px !important;

        margin: 0 0 .35rem 0 !important;
        padding: .35rem .55rem !important;
        box-sizing: border-box !important;

        background: var(--dm-bg) !important;
        color: var(--dm-text) !important;

        font-size: .9rem !important;
        font-weight: 650 !important;
        line-height: 1.3 !important;

        visibility: visible !important;
        opacity: 1 !important;

        position: relative !important;
        z-index: 999 !important;
    }

    .st-key-dm_center_composer .dm-thinking-inline span:last-child {
        color: var(--dm-text) !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    .st-key-dm_center_composer .dm-thinking-pulse {
        display: inline-block !important;
        width: 9px !important;
        height: 9px !important;
        min-width: 9px !important;
        border-radius: 999px !important;
        background: var(--dm-primary) !important;
        animation: dmInlineThinkingPulse 1s ease-in-out infinite !important;
    }

    @keyframes dmInlineThinkingPulse {
        0%, 100% {
            opacity: .35;
            transform: scale(.82);
        }
        50% {
            opacity: 1;
            transform: scale(1.12);
        }
    }

    /* Input remains directly below the thinking row. */
    .st-key-dm_center_composer [data-testid="stChatInput"] {
        margin-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL — ALWAYS OPEN NEWEST ANSWER FROM ITS FIRST LINE
# ============================================================
# Streamlit's fixed-height container can place the actual scrollbar on an
# internal child element rather than on `.st-key-dm_conversation_viewport`
# itself. This script finds the REAL scrolling element and positions it at
# the marker immediately before the newest assistant answer.
st.markdown(
    """
    <style>
    #dm-latest-turn-start {
        display:block !important;
        height:1px !important;
        min-height:1px !important;
        margin:0 !important;
        padding:0 !important;
        scroll-margin-top:8px !important;
        visibility:visible !important;
    }

    /* Completed chat content must start normally; never bottom-align it. */
    .st-key-dm_conversation_viewport > div,
    .st-key-dm_conversation_viewport [data-testid="stVerticalBlock"] {
        justify-content:flex-start !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

components.html(
    """
    <script>
    (() => {
        const win = window.parent;
        const doc = win.document;
        const KEY = "__docmindTrueFirstLineFinal";

        if (win[KEY]?.destroy) {
            try { win[KEY].destroy(); } catch (_) {}
        }

        const timers = [];
        let observer = null;
        let userScrolled = false;
        let programmatic = false;
        let scroller = null;

        const root = () =>
            doc.querySelector(".st-key-dm_conversation_viewport");

        const marker = () =>
            doc.querySelector("#dm-latest-turn-start");

        const isScrollable = (el) => {
            if (!el) return false;
            const style = win.getComputedStyle(el);
            const oy = style.overflowY;
            return (
                (oy === "auto" || oy === "scroll") &&
                el.scrollHeight > el.clientHeight + 2
            );
        };

        const findRealScroller = () => {
            const container = root();
            const target = marker();

            if (!container || !target) return null;

            /*
             * First walk upward from the marker. This catches Streamlit's
             * internal fixed-height scrolling wrapper.
             */
            let node = target.parentElement;

            while (node && node !== doc.body) {
                if (isScrollable(node)) {
                    return node;
                }

                if (node === container) break;
                node = node.parentElement;
            }

            if (isScrollable(container)) {
                return container;
            }

            /*
             * Fallback: inspect descendants of the keyed Streamlit container.
             */
            const descendants = Array.from(
                container.querySelectorAll("*")
            );

            return (
                descendants.find((el) => isScrollable(el)) ||
                container
            );
        };

        const positionAtAnswerStart = () => {
            const target = marker();
            scroller = findRealScroller();

            if (!target || !scroller) return;

            programmatic = true;

            /*
             * Using bounding rectangles works even when Streamlit inserts
             * extra wrapper elements between the keyed container and content.
             */
            const scrollRect = scroller.getBoundingClientRect();
            const targetRect = target.getBoundingClientRect();

            const desired =
                scroller.scrollTop +
                (targetRect.top - scrollRect.top) -
                8;

            scroller.scrollTop = Math.max(0, desired);

            /*
             * scrollIntoView is a second safety net for browser/Streamlit
             * versions whose real scroll element changes after layout.
             */
            target.scrollIntoView({
                behavior: "auto",
                block: "start",
                inline: "nearest"
            });

            win.setTimeout(() => {
                /*
                 * Re-apply after scrollIntoView so the 8px breathing room is
                 * preserved without showing the middle/end of the answer.
                 */
                const activeScroller = findRealScroller();
                const activeTarget = marker();

                if (activeScroller && activeTarget) {
                    const sr = activeScroller.getBoundingClientRect();
                    const tr = activeTarget.getBoundingClientRect();

                    activeScroller.scrollTop = Math.max(
                        0,
                        activeScroller.scrollTop +
                        (tr.top - sr.top) -
                        8
                    );
                }

                programmatic = false;
            }, 30);
        };

        const onUserScroll = () => {
            if (!programmatic) {
                userScrolled = true;
            }
        };

        const attach = () => {
            const container = root();
            const target = marker();

            if (!container || !target) return;

            scroller = findRealScroller();

            if (scroller) {
                scroller.removeEventListener("scroll", onUserScroll);
                scroller.addEventListener(
                    "scroll",
                    onUserScroll,
                    { passive: true }
                );
            }

            userScrolled = false;
            positionAtAnswerStart();

            observer?.disconnect();

            /*
             * Streamlit may resize/rebuild wrappers for a short moment after
             * the rerun. Keep the answer at line one until the user manually
             * scrolls; after that we never fight their reading position.
             */
            observer = new win.MutationObserver(() => {
                if (!userScrolled && marker()) {
                    positionAtAnswerStart();
                }
            });

            observer.observe(container, {
                childList: true,
                subtree: true
            });
        };

        [0, 40, 100, 180, 300, 500, 800, 1200].forEach((delay) => {
            timers.push(
                win.setTimeout(() => {
                    if (!userScrolled) attach();
                }, delay)
            );
        });

        win[KEY] = {
            destroy() {
                observer?.disconnect();

                if (scroller) {
                    scroller.removeEventListener(
                        "scroll",
                        onUserScroll
                    );
                }

                timers.forEach((timer) => win.clearTimeout(timer));
            }
        };
    })();
    </script>
    """,
    height=0,
    width=0,
)


# ============================================================
# SIDEBAR CHAT DELETE ICON
# ============================================================



# ============================================================
# FINAL — SHOW NEWEST QUESTION BEFORE NEWEST ANSWER
# ============================================================
st.markdown(
    """
    <style>
    #dm-latest-turn-start {
        display:block !important;
        height:1px !important;
        min-height:1px !important;
        margin:0 !important;
        padding:0 !important;
        scroll-margin-top:8px !important;
        visibility:visible !important;
    }

    /* Keep the newest user question and answer in natural order. */
    .st-key-dm_conversation_viewport
    [data-testid="stChatMessage"] {
        position:relative !important;
        flex:0 0 auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CLEAR SESSION CONFIRMATION UI
# ============================================================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .dm-clear-confirm-box {
        margin: .45rem 0 .55rem !important;
        padding: .75rem .8rem !important;
        border: 1px solid rgba(239,115,115,.30) !important;
        border-radius: 10px !important;
        background: rgba(239,115,115,.06) !important;
    }

    [data-testid="stSidebar"] .dm-clear-confirm-title {
        color: var(--dm-sidebar-text) !important;
        font-size: .84rem !important;
        font-weight: 800 !important;
        margin-bottom: .25rem !important;
    }

    [data-testid="stSidebar"] .dm-clear-confirm-text {
        color: var(--dm-sidebar-muted) !important;
        font-size: .74rem !important;
        line-height: 1.45 !important;
        margin-bottom: .45rem !important;
    }

    [data-testid="stSidebar"] .dm-clear-confirm-warning {
        color: var(--dm-sidebar-muted) !important;
        font-size: .72rem !important;
        line-height: 1.45 !important;
    }

    [data-testid="stSidebar"] .dm-clear-confirm-warning b {
        color: var(--dm-danger) !important;
    }

    [data-testid="stSidebar"] .st-key-confirm_clear_everything button {
        border-color: rgba(239,115,115,.45) !important;
        color: var(--dm-danger) !important;
    }

    [data-testid="stSidebar"] .st-key-confirm_clear_everything button:hover {
        background: rgba(239,115,115,.10) !important;
        border-color: var(--dm-danger) !important;
        color: var(--dm-danger) !important;
    }

    [data-testid="stSidebar"] .st-key-confirm_clear_chats_only button:hover {
        border-color: var(--dm-primary) !important;
        color: var(--dm-primary) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL CLEAR-SESSION BUTTON THEME FIX
# ============================================================
st.markdown(
    """
    <style>
    /* Clear chats */
    [data-testid="stSidebar"] .st-key-confirm_clear_chats_only button {
        background: var(--dm-sidebar-hover) !important;
        color: var(--dm-sidebar-text) !important;
        border: 1px solid var(--dm-sidebar-border) !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .st-key-confirm_clear_chats_only button p,
    [data-testid="stSidebar"] .st-key-confirm_clear_chats_only button span {
        color: var(--dm-sidebar-text) !important;
    }

    [data-testid="stSidebar"] .st-key-confirm_clear_chats_only button:hover {
        background: var(--dm-soft) !important;
        color: var(--dm-primary) !important;
        border-color: var(--dm-primary) !important;
    }

    [data-testid="stSidebar"] .st-key-confirm_clear_chats_only button:hover p,
    [data-testid="stSidebar"] .st-key-confirm_clear_chats_only button:hover span {
        color: var(--dm-primary) !important;
    }

    /* Clear all data */
    [data-testid="stSidebar"] .st-key-confirm_clear_everything button {
        background: rgba(239, 115, 115, 0.08) !important;
        color: var(--dm-danger) !important;
        border: 1px solid rgba(239, 115, 115, 0.38) !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .st-key-confirm_clear_everything button p,
    [data-testid="stSidebar"] .st-key-confirm_clear_everything button span {
        color: var(--dm-danger) !important;
    }

    [data-testid="stSidebar"] .st-key-confirm_clear_everything button:hover {
        background: rgba(239, 115, 115, 0.15) !important;
        border-color: var(--dm-danger) !important;
        color: var(--dm-danger) !important;
    }

    [data-testid="stSidebar"] .st-key-confirm_clear_everything button:hover p,
    [data-testid="stSidebar"] .st-key-confirm_clear_everything button:hover span {
        color: var(--dm-danger) !important;
    }

    /* Cancel */
    [data-testid="stSidebar"] .st-key-cancel_clear_session button {
        background: transparent !important;
        color: var(--dm-sidebar-text) !important;
        border: 1px solid var(--dm-sidebar-border) !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .st-key-cancel_clear_session button p,
    [data-testid="stSidebar"] .st-key-cancel_clear_session button span {
        color: var(--dm-sidebar-text) !important;
    }

    [data-testid="stSidebar"] .st-key-cancel_clear_session button:hover {
        background: var(--dm-sidebar-hover) !important;
        border-color: var(--dm-primary) !important;
        color: var(--dm-primary) !important;
    }

    [data-testid="stSidebar"] .st-key-cancel_clear_session button:hover p,
    [data-testid="stSidebar"] .st-key-cancel_clear_session button:hover span {
        color: var(--dm-primary) !important;
    }

    /* Prevent focus/active states from flashing white */
    [data-testid="stSidebar"] .st-key-confirm_clear_chats_only button:focus,
    [data-testid="stSidebar"] .st-key-confirm_clear_chats_only button:active,
    [data-testid="stSidebar"] .st-key-confirm_clear_everything button:focus,
    [data-testid="stSidebar"] .st-key-confirm_clear_everything button:active,
    [data-testid="stSidebar"] .st-key-cancel_clear_session button:focus,
    [data-testid="stSidebar"] .st-key-cancel_clear_session button:active {
        box-shadow: none !important;
        outline: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MULTI-CHAT SIDEBAR ROWS
# ============================================================



# ============================================================
# PROFESSIONAL APPEARANCE / THEME CONTROL
# ============================================================
st.markdown(
    """
    <style>
    /* Outer appearance card */
    .st-key-documents_theme_control {
        margin: .55rem 0 .85rem !important;
        padding: .72rem .78rem .68rem !important;
        border: 1px solid var(--dm-border) !important;
        border-radius: 13px !important;
        background:
            linear-gradient(
                180deg,
                color-mix(in srgb, var(--dm-surface2) 92%, transparent),
                color-mix(in srgb, var(--dm-surface) 96%, transparent)
            ) !important;
        box-shadow: 0 5px 18px rgba(0,0,0,.06) !important;
        overflow: visible !important;
    }

    .st-key-documents_theme_control > div,
    .st-key-documents_theme_control [data-testid="stVerticalBlock"] {
        gap: .5rem !important;
    }

    /* Heading explains exactly what the control is for */
    .dm-theme-control-heading {
        display: flex !important;
        align-items: center !important;
        gap: .55rem !important;
        margin: 0 0 .08rem !important;
    }

    .dm-theme-control-icon {
        width: 30px !important;
        height: 30px !important;
        flex: 0 0 30px !important;

        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;

        border-radius: 9px !important;
        background: var(--dm-soft) !important;
        border: 1px solid var(--dm-border) !important;
        color: var(--dm-primary) !important;

        font-size: 17px !important;
        font-weight: 800 !important;
    }

    .dm-theme-control-title {
        color: var(--dm-heading) !important;
        font-size: .79rem !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
    }

    .dm-theme-control-subtitle {
        color: var(--dm-muted) !important;
        font-size: .64rem !important;
        font-weight: 550 !important;
        margin-top: .15rem !important;
        line-height: 1.1 !important;
    }

    /* Dark / Light labels */
    .dm-theme-choice {
        min-height: 34px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: .28rem !important;

        padding: .35rem .42rem !important;
        border-radius: 9px !important;
        border: 1px solid transparent !important;

        color: var(--dm-muted) !important;
        background: transparent !important;

        font-size: .68rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;

        transition:
            background .18s ease,
            color .18s ease,
            border-color .18s ease !important;
    }

    .dm-theme-choice.active {
        color: var(--dm-heading) !important;
        background: var(--dm-soft) !important;
        border-color: color-mix(
            in srgb,
            var(--dm-primary) 42%,
            var(--dm-border)
        ) !important;
        box-shadow: inset 0 0 0 1px rgba(53,201,138,.04) !important;
    }

    .dm-theme-choice-icon {
        font-size: .9rem !important;
        line-height: 1 !important;
        color: inherit !important;
    }

    /*
     * Make the actual Streamlit switch larger and more deliberate.
     * The surrounding labels make its purpose unambiguous even if
     * Streamlit changes its internal switch markup in a future release.
     */
    .st-key-documents_theme_control [data-testid="stToggle"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        min-height: 34px !important;
    }

    .st-key-documents_theme_control [data-testid="stToggle"] > label {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        cursor: pointer !important;
    }

    /* Best-effort enhancement of Streamlit's native switch track */
    .st-key-documents_theme_control
    [data-testid="stToggle"] [data-baseweb="checkbox"] > div {
        transform: scale(1.14) !important;
        transform-origin: center !important;
        box-shadow: 0 2px 8px rgba(0,0,0,.12) !important;
    }

    .st-key-documents_theme_control
    [data-testid="stToggle"] input:focus-visible + div {
        outline: 2px solid var(--dm-primary) !important;
        outline-offset: 3px !important;
    }

    /* Keep the row compact inside the right sidebar */
    .st-key-documents_theme_control
    div[data-testid="stHorizontalBlock"] {
        gap: .28rem !important;
        align-items: center !important;
    }

    @media (max-width: 1200px) and (min-width: 901px) {
        .dm-theme-choice {
            font-size: .63rem !important;
            padding-left: .3rem !important;
            padding-right: .3rem !important;
        }

        .dm-theme-control-title {
            font-size: .75rem !important;
        }
    }

    @media (max-width: 900px) {
        .st-key-documents_theme_control {
            margin-top: .45rem !important;
        }

        .dm-theme-choice {
            font-size: .7rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL THEME CONTROL — DARK MODE [TOGGLE] LIGHT MODE
# ============================================================
st.markdown(
    """
    <style>
    /* Remove the previous card/appearance styling completely. */
    .st-key-documents_theme_control {
        margin: .55rem 0 .8rem !important;
        padding: 0 !important;
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        overflow: visible !important;
    }

    .st-key-documents_theme_control > div,
    .st-key-documents_theme_control [data-testid="stVerticalBlock"] {
        gap: 0 !important;
        overflow: visible !important;
    }

    .st-key-documents_theme_control
    div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: center !important;
        gap: .35rem !important;
        width: 100% !important;
    }

    /* Labels on each side of the switch */
    .dm-theme-mode-label {
        min-height: 34px !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: .28rem !important;

        padding: .26rem .25rem !important;
        box-sizing: border-box !important;

        color: var(--dm-muted) !important;
        font-size: .68rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        white-space: nowrap !important;

        opacity: .72 !important;
        transition:
            color .18s ease,
            opacity .18s ease !important;
    }

    .dm-theme-mode-label.active {
        color: var(--dm-heading) !important;
        opacity: 1 !important;
    }

    .dm-theme-mode-icon {
        font-size: .92rem !important;
        line-height: 1 !important;
        color: inherit !important;
    }

    /*
     * Make the center Streamlit toggle feel more intentional without
     * replacing its reliable built-in functionality.
     */
    .st-key-documents_theme_control
    [data-testid="stToggle"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;

        min-width: 48px !important;
        min-height: 34px !important;

        margin: 0 !important;
        padding: 0 !important;
    }

    .st-key-documents_theme_control
    [data-testid="stToggle"] > label {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        cursor: pointer !important;
    }

    /* Slightly enlarge the native Streamlit switch. */
    .st-key-documents_theme_control
    [data-testid="stToggle"]
    [data-baseweb="checkbox"] > div {
        transform: scale(1.18) !important;
        transform-origin: center !important;
        box-shadow: 0 2px 8px rgba(0,0,0,.14) !important;
    }

    .st-key-documents_theme_control
    [data-testid="stToggle"] input:focus-visible + div {
        outline: 2px solid var(--dm-primary) !important;
        outline-offset: 3px !important;
    }

    /* Keep everything on one line in the Documents panel. */
    @media (min-width: 901px) {
        .st-key-documents_theme_control
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
        }
    }

    @media (max-width: 1200px) and (min-width: 901px) {
        .dm-theme-mode-label {
            font-size: .62rem !important;
            gap: .2rem !important;
        }

        .dm-theme-mode-icon {
            font-size: .84rem !important;
        }
    }

    @media (max-width: 900px) {
        .dm-theme-mode-label {
            font-size: .7rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL THEME TOGGLE — TEXT INSIDE THE PILL
# ============================================================
st.markdown(
    """
    <style>
    /*
     * One professional segmented toggle:
     *
     *  DARK MODE ACTIVE:
     *  ┌─────────────────────────────────┐
     *  │  🌙 Dark Mode   |   ☀ Light Mode │
     *  └─────────────────────────────────┘
     *
     *  LIGHT MODE ACTIVE:
     *  same control, right segment highlighted.
     */

    div[data-testid="stRadio"]:has(
        input[name*="documents_theme_segmented_toggle"]
    ) {
        width: 100% !important;
        margin: .55rem 0 .8rem !important;
    }

    div[data-testid="stRadio"]:has(
        input[name*="documents_theme_segmented_toggle"]
    ) > div {
        width: 100% !important;
    }

    div[data-testid="stRadio"]:has(
        input[name*="documents_theme_segmented_toggle"]
    ) [role="radiogroup"] {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 0 !important;

        width: 100% !important;
        min-height: 42px !important;

        padding: 3px !important;
        box-sizing: border-box !important;

        border-radius: 999px !important;
        border: 1px solid var(--dm-border) !important;

        background: var(--dm-surface2) !important;

        box-shadow:
            inset 0 1px 2px rgba(0,0,0,.08),
            0 3px 10px rgba(0,0,0,.06) !important;

        overflow: hidden !important;
    }

    div[data-testid="stRadio"]:has(
        input[name*="documents_theme_segmented_toggle"]
    ) [role="radio"] {
        position: relative !important;

        min-height: 36px !important;
        height: 36px !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;

        margin: 0 !important;
        padding: 0 .55rem !important;

        border: 0 !important;
        border-radius: 999px !important;

        background: transparent !important;
        color: var(--dm-muted) !important;

        cursor: pointer !important;
        transition:
            background .2s ease,
            color .2s ease,
            box-shadow .2s ease,
            transform .2s ease !important;
    }

    /* Hide Streamlit's circular radio bullet. */
    div[data-testid="stRadio"]:has(
        input[name*="documents_theme_segmented_toggle"]
    ) [role="radio"] > div:first-child {
        display: none !important;
    }

    div[data-testid="stRadio"]:has(
        input[name*="documents_theme_segmented_toggle"]
    ) [role="radio"] p {
        margin: 0 !important;
        color: inherit !important;

        font-size: .70rem !important;
        font-weight: 750 !important;
        line-height: 1 !important;
        white-space: nowrap !important;
    }

    /*
     * Add icons INSIDE each half of the toggle.
     */
    div[data-testid="stRadio"]:has(
        input[name*="documents_theme_segmented_toggle"]
    ) [role="radio"]:first-child p::before {
        content: "☾";
        display: inline-block !important;
        margin-right: .32rem !important;
        font-size: .9rem !important;
        vertical-align: -1px !important;
    }

    div[data-testid="stRadio"]:has(
        input[name*="documents_theme_segmented_toggle"]
    ) [role="radio"]:last-child p::before {
        content: "☀";
        display: inline-block !important;
        margin-right: .32rem !important;
        font-size: .9rem !important;
        vertical-align: -1px !important;
    }

    /*
     * Selected half becomes the knob/highlight.
     */
    div[data-testid="stRadio"]:has(
        input[name*="documents_theme_segmented_toggle"]
    ) [role="radio"][aria-checked="true"] {
        background: var(--dm-soft) !important;
        color: var(--dm-heading) !important;

        box-shadow:
            0 1px 5px rgba(0,0,0,.12),
            inset 0 0 0 1px
            color-mix(
                in srgb,
                var(--dm-primary) 35%,
                var(--dm-border)
            ) !important;
    }

    div[data-testid="stRadio"]:has(
        input[name*="documents_theme_segmented_toggle"]
    ) [role="radio"]:hover {
        color: var(--dm-heading) !important;
    }

    div[data-testid="stRadio"]:has(
        input[name*="documents_theme_segmented_toggle"]
    ) [role="radio"]:active {
        transform: scale(.985) !important;
    }

    /*
     * In light mode, make the selected Light segment visibly bright.
     */
    html:has(
        div[data-testid="stRadio"]
        input[name*="documents_theme_segmented_toggle"]:checked
    ) body {
        --dm-theme-segment-transition: .2s;
    }

    @media (max-width: 1200px) and (min-width: 901px) {
        div[data-testid="stRadio"]:has(
            input[name*="documents_theme_segmented_toggle"]
        ) [role="radio"] p {
            font-size: .64rem !important;
        }

        div[data-testid="stRadio"]:has(
            input[name*="documents_theme_segmented_toggle"]
        ) [role="radio"] {
            padding-left: .35rem !important;
            padding-right: .35rem !important;
        }
    }

    @media (max-width: 900px) {
        div[data-testid="stRadio"]:has(
            input[name*="documents_theme_segmented_toggle"]
        ) {
            max-width: 320px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHATGPT-LIKE SIDEBAR SPACING / BUTTON SIZES / VISUAL RHYTHM
# ============================================================
st.markdown(
    """
    <style>
    /* Sidebar spacing baseline */
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0.7rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
        padding-bottom: 1rem !important;
    }

    /* Brand area */
    .dm-brand {
        display: flex !important;
        align-items: center !important;
        gap: 0.78rem !important;
        padding: 0.15rem 0 0.85rem 0 !important;
        margin: 0 0 0.25rem 0 !important;
    }

    .dm-brand-icon {
        width: 44px !important;
        height: 44px !important;
        min-width: 44px !important;
        border-radius: 14px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 16px !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 22px rgba(53, 201, 138, 0.14) !important;
    }

    .dm-brand-title {
        font-size: 1.04rem !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
        margin: 0 !important;
    }

    .dm-brand-sub {
        font-size: 0.72rem !important;
        line-height: 1.2 !important;
        margin-top: 0.18rem !important;
        opacity: 0.9 !important;
    }

    /* Sidebar labels like Recents / Analytics / Knowledge / Retrieval */
    .dm-nav-label {
        margin: 1.05rem 0 0.5rem 0 !important;
        padding: 0 !important;
        font-size: 0.78rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.04em !important;
        text-transform: none !important;
        color: var(--dm-heading) !important;
        opacity: 0.92 !important;
    }

    /* New chat button - primary CTA */
    .stSidebar div[data-testid="stButton"] button {
        min-height: 46px !important;
        border-radius: 14px !important;
        padding: 0.8rem 0.95rem !important;
        font-size: 0.98rem !important;
        font-weight: 650 !important;
        line-height: 1 !important;
        transition:
            background 0.18s ease,
            border-color 0.18s ease,
            color 0.18s ease,
            transform 0.12s ease,
            box-shadow 0.18s ease !important;
        box-shadow: none !important;
    }

    .stSidebar div[data-testid="stButton"] button:hover {
        transform: translateY(-1px) !important;
    }

    /* Primary new chat button (first main CTA near top) */
    .stSidebar [data-testid="stSidebarContent"] > div > div > div[data-testid="stButton"]:first-of-type button,
    .stSidebar .st-key-sidebar_newchat_wrap div[data-testid="stButton"] button {
        min-height: 48px !important;
        border-radius: 14px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        background: linear-gradient(180deg, #38d694 0%, #2fc784 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        box-shadow:
            0 10px 24px rgba(53, 201, 138, 0.18),
            inset 0 1px 0 rgba(255,255,255,0.08) !important;
    }

    .stSidebar [data-testid="stSidebarContent"] > div > div > div[data-testid="stButton"]:first-of-type button:hover,
    .stSidebar .st-key-sidebar_newchat_wrap div[data-testid="stButton"] button:hover {
        background: linear-gradient(180deg, #43dca0 0%, #33ca89 100%) !important;
    }

    /* Chat item button / row buttons / nav buttons */
    .stSidebar .st-key-chat_list_container div[data-testid="stButton"] button,
    .stSidebar div[data-testid="stButton"] button[kind="secondary"],
    .stSidebar div[data-testid="stButton"] button {
        background: color-mix(in srgb, var(--dm-surface2) 95%, transparent) !important;
        border: 1px solid var(--dm-border) !important;
        color: var(--dm-heading) !important;
    }

    /* Ghost row hover like ChatGPT */
    .stSidebar div[data-testid="stButton"] button:hover {
        background: color-mix(in srgb, var(--dm-surface2) 88%, var(--dm-soft) 12%) !important;
        border-color: color-mix(in srgb, var(--dm-border) 72%, var(--dm-primary) 28%) !important;
        color: var(--dm-heading) !important;
    }

    /* Chat list feel */
    .stSidebar .st-key-chat_list_container,
    .stSidebar .st-key-sidebar-chat-list,
    .stSidebar .st-key-chat_scroll_container {
        margin-top: 0.15rem !important;
    }

    /* Try to make each chat row shorter and more like a recents item */
    .stSidebar div[data-testid="stButton"] button p {
        font-size: 0.96rem !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* Small utility buttons such as delete icon */
    .stSidebar button[title*="delete"],
    .stSidebar button[aria-label*="delete"],
    .stSidebar button[title*="Delete"],
    .stSidebar button[aria-label*="Delete"] {
        min-height: 40px !important;
        min-width: 40px !important;
        width: 40px !important;
        border-radius: 12px !important;
        padding: 0 !important;
    }

    /* Sliders / retrieval block spacing */
    .stSidebar [data-testid="stSlider"] {
        margin-top: 0.2rem !important;
        margin-bottom: 0.6rem !important;
    }

    .stSidebar [data-testid="stSlider"] p,
    .stSidebar .stSlider p {
        font-size: 0.95rem !important;
    }

    /* Clear session and secondary actions */
    .stSidebar button[key="clear_session_btn"],
    .stSidebar button[key="clear_session"],
    .stSidebar button {
        font-weight: 650 !important;
    }

    /* Divider rhythm */
    .stSidebar hr {
        margin: 0.8rem 0 0.9rem 0 !important;
        opacity: 0.45 !important;
    }

    /* More breathing room around section groups */
    .stSidebar [data-testid="stVerticalBlock"] > div {
        margin-bottom: 0.08rem !important;
    }

    /* ChatGPT-like compactness on narrow widths */
    @media (max-width: 900px) {
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }

        .dm-brand-title {
            font-size: 1rem !important;
        }

        .dm-brand-sub {
            font-size: 0.7rem !important;
        }

        .stSidebar div[data-testid="stButton"] button {
            min-height: 44px !important;
            font-size: 0.95rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — REMOVE SECTION-HEADING GAPS / COMPACT NAV SPACING
# ============================================================
st.markdown(
    """
    <style>
    /* Keep Recents as the only text heading, but tighten what follows. */
    [data-testid="stSidebar"] .dm-nav-label {
        margin-top: .9rem !important;
        margin-bottom: .4rem !important;
    }

    /* Analytics and Knowledge become compact navigation rows. */
    [data-testid="stSidebar"] .st-key-analytics_eval_nav,
    [data-testid="stSidebar"] .st-key-knowledge_nav {
        margin-top: .18rem !important;
        margin-bottom: .18rem !important;
    }

    [data-testid="stSidebar"] .st-key-analytics_eval_nav button,
    [data-testid="stSidebar"] .st-key-knowledge_nav button {
        min-height: 42px !important;
        height: 42px !important;
        padding: .55rem .72rem !important;
        border-radius: 11px !important;
        font-size: .92rem !important;
        justify-content: flex-start !important;
    }

    /* Add a small natural break before Retrieval controls, without a heading. */
    [data-testid="stSidebar"] .st-key-workspace_top_k {
        margin-top: .75rem !important;
        margin-bottom: .15rem !important;
        padding-top: .45rem !important;
        border-top: 1px solid var(--dm-sidebar-border) !important;
    }

    [data-testid="stSidebar"] .st-key-workspace_top_k label p {
        font-size: .9rem !important;
        font-weight: 650 !important;
        margin-bottom: .2rem !important;
    }

    /* Retrieval description sits closer to the slider. */
    [data-testid="stSidebar"] .st-key-workspace_top_k + div,
    [data-testid="stSidebar"] .st-key-workspace_top_k + [data-testid="stElementContainer"] {
        margin-top: .1rem !important;
    }

    /* Clear session gets breathing room but no oversized section gap. */
    [data-testid="stSidebar"] .st-key-clear_session_sidebar {
        margin-top: .65rem !important;
        margin-bottom: .15rem !important;
    }

    [data-testid="stSidebar"] .st-key-clear_session_sidebar button {
        min-height: 42px !important;
        height: 42px !important;
        border-radius: 11px !important;
        padding: .55rem .72rem !important;
        font-size: .92rem !important;
    }

    /* Reduce divider whitespace around the lower utility area. */
    [data-testid="stSidebar"] hr {
        margin: .65rem 0 !important;
    }

    /* Tighten the generic vertical rhythm in this sidebar. */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: .35rem !important;
    }

    /* Do not let Streamlit add large margins between adjacent sidebar items. */
    [data-testid="stSidebar"] [data-testid="stElementContainer"] {
        margin-bottom: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR CHAT DELETE — ICON ONLY, FULLY FUNCTIONAL
# ============================================================



# ============================================================
# FINAL CHAT ROW — DELETE ICON INSIDE THE CHAT BUTTON
# ============================================================



# ============================================================
# FINAL FIX — VISIBLE TRASH ICON INSIDE CHAT ROW
# ============================================================



# ============================================================
# FINAL DELETE ICON — MATERIAL ICON, NO EMPTY BOX
# ============================================================



# ============================================================
# DELETE ICON — NO BACKGROUND AT ALL
# ============================================================





# ============================================================
# FINAL PRECISE CHAT TRASH ICON
# - no independent background
# - no circle/pill
# - centered vertically
# - aligned cleanly at the far-right inside the chat row
# ============================================================



# ============================================================
# CHAT SIDEBAR — TRUE EMBEDDED DELETE ICON
# ============================================================
st.markdown(
    """
    <style>
    /* One row = one rounded chat button. */
    [data-testid="stSidebar"] [class*="st-key-chat_row_"] {
        position: relative !important;
        width: 100% !important;
        margin: 0 0 .28rem 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        overflow: visible !important;
    }

    [data-testid="stSidebar"] [class*="st-key-chat_row_"] > div,
    [data-testid="stSidebar"] [class*="st-key-chat_row_"] [data-testid="stVerticalBlock"] {
        position: relative !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        overflow: visible !important;
    }

    /* Main chat button fills the complete row. */
    [data-testid="stSidebar"] [class*="st-key-chat_nav_"] {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] [class*="st-key-chat_nav_"] button {
        position: relative !important;
        width: 100% !important;
        min-height: 46px !important;
        height: 46px !important;

        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;

        padding: .55rem 2.5rem .55rem .82rem !important;
        border-radius: 12px !important;
        text-align: left !important;
        box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] [class*="st-key-chat_nav_"] button p {
        width: 100% !important;
        margin: 0 !important;
        text-align: left !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /*
     * Hide the real Streamlit delete button COMPLETELY.
     * It stays in the DOM so the custom icon can click it programmatically.
     */
    [data-testid="stSidebar"] [class*="st-key-delete_chat_"] {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /*
     * This is the ONLY visible delete control.
     * It is a plain icon placed directly inside the chat row.
     */
    .dm-inline-chat-delete {
        position: absolute !important;
        top: 50% !important;
        right: .72rem !important;
        transform: translateY(-50%) !important;

        width: 18px !important;
        height: 18px !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;

        margin: 0 !important;
        padding: 0 !important;

        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;

        border: 0 !important;
        border-radius: 0 !important;
        outline: 0 !important;
        box-shadow: none !important;

        color: #8f9792 !important;
        opacity: .72 !important;
        cursor: pointer !important;
        z-index: 1000 !important;

        transition:
            color .15s ease,
            opacity .15s ease !important;
    }

    .dm-inline-chat-delete:hover,
    .dm-inline-chat-delete:focus,
    .dm-inline-chat-delete:active {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        border: 0 !important;
        border-radius: 0 !important;
        outline: 0 !important;
        box-shadow: none !important;

        color: var(--dm-danger) !important;
        opacity: 1 !important;
    }

    .dm-inline-chat-delete svg {
        width: 16px !important;
        height: 16px !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        fill: none !important;
        stroke: currentColor !important;
        stroke-width: 1.8 !important;
        stroke-linecap: round !important;
        stroke-linejoin: round !important;
        pointer-events: none !important;
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


components.html(
    """
    <script>
    (() => {
        const win = window.parent;
        const doc = win.document;
        const KEY = "__docmindInlineChatDeleteV1";

        if (win[KEY]?.destroy) {
            try { win[KEY].destroy(); } catch (_) {}
        }

        let observer = null;
        const timers = [];

        const mountIcons = () => {
            const rows = Array.from(
                doc.querySelectorAll(
                    '[data-testid="stSidebar"] [class*="st-key-chat_row_"]'
                )
            );

            rows.forEach((row) => {
                const chatWrapper = row.querySelector(
                    '[class*="st-key-chat_nav_"]'
                );
                const deleteWrapper = row.querySelector(
                    '[class*="st-key-delete_chat_"]'
                );

                const chatButton = chatWrapper?.querySelector("button");
                const realDeleteButton = deleteWrapper?.querySelector("button");

                if (!chatButton || !realDeleteButton) return;

                /* Keep only one injected icon per chat row. */
                let iconButton = chatButton.querySelector(
                    ".dm-inline-chat-delete"
                );

                if (!iconButton) {
                    /*
                     * Use a span instead of nesting a <button> inside the
                     * Streamlit chat <button>. This keeps the markup valid
                     * and allows exact vertical alignment to the chat text.
                     */
                    iconButton = doc.createElement("span");
                    iconButton.className = "dm-inline-chat-delete";
                    iconButton.setAttribute("role", "button");
                    iconButton.setAttribute("tabindex", "0");
                    iconButton.setAttribute(
                        "aria-label",
                        "Delete conversation"
                    );
                    iconButton.setAttribute(
                        "title",
                        "Delete conversation"
                    );

                    iconButton.innerHTML = `
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                            <path d="M3 6h18"></path>
                            <path d="M8 6V4h8v2"></path>
                            <path d="M19 6l-1 14H6L5 6"></path>
                            <path d="M10 11v5"></path>
                            <path d="M14 11v5"></path>
                        </svg>
                    `;

                    const triggerDelete = (event) => {
                        event.preventDefault();
                        event.stopPropagation();

                        const freshDeleteButton = row.querySelector(
                            '[class*="st-key-delete_chat_"] button'
                        );

                        if (freshDeleteButton) {
                            freshDeleteButton.click();
                        }
                    };

                    iconButton.addEventListener("click", triggerDelete);

                    iconButton.addEventListener("keydown", (event) => {
                        if (
                            event.key === "Enter" ||
                            event.key === " "
                        ) {
                            triggerDelete(event);
                        }
                    });

                    /*
                     * Append directly to the chat button. Because that button
                     * is position:relative, top:50% now means the true center
                     * of the "Policies" row/text.
                     */
                    chatButton.appendChild(iconButton);
                }
            });
        };

        const startObserver = () => {
            mountIcons();

            observer?.disconnect();
            observer = new win.MutationObserver(() => {
                mountIcons();
            });

            const sidebar = doc.querySelector(
                '[data-testid="stSidebar"]'
            );

            if (sidebar) {
                observer.observe(sidebar, {
                    childList: true,
                    subtree: true,
                });
            }
        };

        [0, 80, 180, 350, 700].forEach((delay) => {
            timers.push(win.setTimeout(startObserver, delay));
        });

        win[KEY] = {
            destroy() {
                observer?.disconnect();
                timers.forEach((timer) => win.clearTimeout(timer));
                doc.querySelectorAll(".dm-inline-chat-delete")
                    .forEach((node) => node.remove());
            }
        };
    })();
    </script>
    """,
    height=0,
    width=0,
)


# ============================================================
# FINAL ALIGNMENT — TRASH ICON CENTERED WITH CHAT TEXT
# ============================================================
st.markdown(
    """
    <style>
    /* The chat button is the positioning reference. */
    [data-testid="stSidebar"] [class*="st-key-chat_nav_"] button {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        min-height: 46px !important;
        height: 46px !important;
        padding-right: 2.45rem !important;
    }

    /*
     * Trash icon is INSIDE the same 46px button as "Policies".
     * This makes its vertical center exactly match the chat text.
     */
    [data-testid="stSidebar"]
    [class*="st-key-chat_nav_"]
    button .dm-inline-chat-delete {
        position: absolute !important;

        top: 50% !important;
        right: .78rem !important;
        transform: translateY(-50%) !important;

        width: 17px !important;
        height: 17px !important;
        min-width: 17px !important;
        min-height: 17px !important;

        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;

        margin: 0 !important;
        padding: 0 !important;

        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;

        border: 0 !important;
        border-radius: 0 !important;
        outline: 0 !important;
        box-shadow: none !important;

        color: #8f9792 !important;
        opacity: .75 !important;
        cursor: pointer !important;
        line-height: 1 !important;

        z-index: 20 !important;
    }

    [data-testid="stSidebar"]
    [class*="st-key-chat_nav_"]
    button .dm-inline-chat-delete:hover,
    [data-testid="stSidebar"]
    [class*="st-key-chat_nav_"]
    button .dm-inline-chat-delete:focus,
    [data-testid="stSidebar"]
    [class*="st-key-chat_nav_"]
    button .dm-inline-chat-delete:active {
        background: transparent !important;
        background-color: transparent !important;
        border: 0 !important;
        border-radius: 0 !important;
        outline: 0 !important;
        box-shadow: none !important;

        color: var(--dm-danger) !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"]
    [class*="st-key-chat_nav_"]
    button .dm-inline-chat-delete svg {
        display: block !important;
        width: 16px !important;
        height: 16px !important;

        margin: 0 !important;
        padding: 0 !important;

        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;

        fill: none !important;
        stroke: currentColor !important;
        stroke-width: 1.8 !important;
        stroke-linecap: round !important;
        stroke-linejoin: round !important;

        pointer-events: none !important;
    }

    /*
     * The hidden real Streamlit delete button remains fully hidden,
     * so it cannot contribute any extra height or visible background.
     */
    [data-testid="stSidebar"] [class*="st-key-delete_chat_"] {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        min-width: 1px !important;
        min-height: 1px !important;
        max-width: 1px !important;
        max-height: 1px !important;

        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;

        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RECENTS SEPARATOR
# ============================================================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .dm-recents-separator {
        width: 100% !important;
        height: 1px !important;
        margin: .72rem 0 .72rem 0 !important;
        padding: 0 !important;

        background: var(--dm-sidebar-border) !important;
        border: 0 !important;
        opacity: .85 !important;
    }

    /* Keep Analytics close to the separator like a clean navigation group. */
    [data-testid="stSidebar"] .dm-recents-separator
    + div {
        margin-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




# ============================================================
# SIDEBAR PROFESSIONAL SPACING — SAFE / NO OVERLAP
# ============================================================
st.markdown(
    """
    <style>
    /* --------------------------------------------------------
       SAFE SIDEBAR PADDING
       -------------------------------------------------------- */
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: .95rem 1rem 1rem !important;
        box-sizing: border-box !important;
    }

    /* IMPORTANT:
       Do not force all Streamlit vertical blocks or element containers
       to gap:0 / margin:0. Streamlit sliders, labels, and captions need
       their own internal spacing. */

    /* --------------------------------------------------------
       BRAND
       -------------------------------------------------------- */
    [data-testid="stSidebar"] .dm-brand {
        display: flex !important;
        align-items: center !important;
        gap: .72rem !important;
        min-height: 48px !important;
        margin: 0 0 .95rem 0 !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] .dm-brand-icon {
        width: 42px !important;
        height: 42px !important;
        min-width: 42px !important;
        flex: 0 0 42px !important;
        border-radius: 12px !important;
    }

    [data-testid="stSidebar"] .dm-brand-title {
        margin: 0 !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        line-height: 1.15 !important;
    }

    [data-testid="stSidebar"] .dm-brand-sub {
        margin: .15rem 0 0 !important;
        font-size: .68rem !important;
        line-height: 1.2 !important;
    }

    /* --------------------------------------------------------
       NEW CHAT
       -------------------------------------------------------- */
    [data-testid="stSidebar"] .st-key-new_chat_button {
        margin: 0 0 .9rem 0 !important;
    }

    [data-testid="stSidebar"] .st-key-new_chat_button button {
        width: 100% !important;
        min-height: 46px !important;
        height: 46px !important;
        padding: 0 .9rem !important;
        border-radius: 12px !important;
        font-size: .94rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
    }

    /* --------------------------------------------------------
       RECENTS
       -------------------------------------------------------- */
    [data-testid="stSidebar"] .dm-nav-label {
        display: block !important;
        margin: 0 0 .45rem 0 !important;
        padding: 0 !important;
        min-height: 16px !important;

        font-size: .76rem !important;
        font-weight: 800 !important;
        line-height: 1.25 !important;
        color: var(--dm-heading) !important;
        opacity: .9 !important;
    }

    [data-testid="stSidebar"] [class*="st-key-chat_row_"] {
        margin: 0 0 .48rem 0 !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] [class*="st-key-chat_nav_"] button {
        width: 100% !important;
        min-height: 44px !important;
        height: 44px !important;
        padding: 0 2.4rem 0 .82rem !important;
        border-radius: 11px !important;

        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;

        font-size: .91rem !important;
        font-weight: 600 !important;
        line-height: 1 !important;
        text-align: left !important;
    }

    [data-testid="stSidebar"] [class*="st-key-chat_nav_"] button p {
        margin: 0 !important;
        line-height: 1.1 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* Separator below recent chats */
    [data-testid="stSidebar"] .dm-recents-separator {
        width: 100% !important;
        height: 1px !important;
        margin: .72rem 0 .72rem !important;
        padding: 0 !important;
        background: var(--dm-sidebar-border) !important;
        opacity: .8 !important;
    }

    /* --------------------------------------------------------
       ANALYTICS + KNOWLEDGE
       -------------------------------------------------------- */
    [data-testid="stSidebar"] .st-key-analytics_eval_nav,
    [data-testid="stSidebar"] .st-key-knowledge_nav {
        margin: 0 0 .48rem 0 !important;
    }

    [data-testid="stSidebar"] .st-key-analytics_eval_nav button,
    [data-testid="stSidebar"] .st-key-knowledge_nav button {
        width: 100% !important;
        min-height: 44px !important;
        height: 44px !important;
        padding: 0 .82rem !important;
        border-radius: 11px !important;

        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;

        font-size: .91rem !important;
        font-weight: 600 !important;
        line-height: 1 !important;
        text-align: left !important;
    }

    [data-testid="stSidebar"] .st-key-analytics_eval_nav button p,
    [data-testid="stSidebar"] .st-key-knowledge_nav button p {
        margin: 0 !important;
        line-height: 1.1 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* --------------------------------------------------------
       RETRIEVAL
       Let Streamlit keep internal slider label/tick spacing.
       -------------------------------------------------------- */
    [data-testid="stSidebar"] .st-key-workspace_top_k {
        margin: .78rem 0 0 !important;
        padding: .72rem 0 0 !important;
        border-top: 1px solid var(--dm-sidebar-border) !important;
    }

    [data-testid="stSidebar"] .st-key-workspace_top_k label {
        margin-bottom: .22rem !important;
    }

    [data-testid="stSidebar"] .st-key-workspace_top_k label p {
        margin: 0 !important;
        font-size: .9rem !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
    }

    /* Restore breathing room for slider itself and min/max labels */
    [data-testid="stSidebar"] .st-key-workspace_top_k [data-testid="stSlider"] {
        margin-top: .2rem !important;
        margin-bottom: .55rem !important;
        padding-bottom: .15rem !important;
    }

    /* Do not collapse slider internals */
    [data-testid="stSidebar"] .st-key-workspace_top_k
    [data-testid="stSlider"] * {
        line-height: normal;
    }

    /* Retrieval caption */
    [data-testid="stSidebar"] .st-key-workspace_top_k + div {
        margin-top: .22rem !important;
        margin-bottom: 0 !important;
    }

    [data-testid="stSidebar"] .st-key-workspace_top_k + div p {
        margin: 0 !important;
        font-size: .77rem !important;
        line-height: 1.4 !important;
    }

    /* --------------------------------------------------------
       CLEAR SESSION
       -------------------------------------------------------- */
    [data-testid="stSidebar"] .st-key-clear_session_sidebar {
        margin: .88rem 0 0 !important;
        padding: .72rem 0 0 !important;
        border-top: 1px solid var(--dm-sidebar-border) !important;
    }

    [data-testid="stSidebar"] .st-key-clear_session_sidebar button {
        width: 100% !important;
        min-height: 44px !important;
        height: 44px !important;
        padding: 0 .82rem !important;
        border-radius: 11px !important;
        font-size: .91rem !important;
        font-weight: 600 !important;
        line-height: 1 !important;
    }

    /* Bottom caption gets its own space */
    [data-testid="stSidebar"] .st-key-clear_session_sidebar ~ div {
        margin-top: .55rem !important;
    }

    [data-testid="stSidebar"] .st-key-clear_session_sidebar ~ div p {
        margin: 0 !important;
        font-size: .74rem !important;
        line-height: 1.45 !important;
    }

    /* --------------------------------------------------------
       MOBILE / NARROW SIDEBAR
       -------------------------------------------------------- */
    @media (max-width: 900px) {
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-left: .85rem !important;
            padding-right: .85rem !important;
        }

        [data-testid="stSidebar"] .st-key-new_chat_button button,
        [data-testid="stSidebar"] [class*="st-key-chat_nav_"] button,
        [data-testid="stSidebar"] .st-key-analytics_eval_nav button,
        [data-testid="stSidebar"] .st-key-knowledge_nav button,
        [data-testid="stSidebar"] .st-key-clear_session_sidebar button {
            min-height: 42px !important;
            height: 42px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — UNIFORM VERTICAL GAP SYSTEM
# ============================================================
st.markdown(
    """
    <style>
    /*
     * One spacing unit for the sidebar.
     * Main controls/rows use 12px between them.
     * Section separators use the same 12px above/below.
     */

    :root {
        --dm-sidebar-gap: 12px;
        --dm-sidebar-small-gap: 6px;
    }

    /* New chat -> Recents */
    [data-testid="stSidebar"] .st-key-new_chat_button {
        margin-bottom: var(--dm-sidebar-gap) !important;
    }

    /* Recents heading -> first chat */
    [data-testid="stSidebar"] .dm-nav-label {
        margin-top: 0 !important;
        margin-bottom: var(--dm-sidebar-small-gap) !important;
    }

    /* Every recent chat row */
    [data-testid="stSidebar"] [class*="st-key-chat_row_"] {
        margin-top: 0 !important;
        margin-bottom: var(--dm-sidebar-gap) !important;
    }

    /* Avoid adding an extra-large gap after the final recent chat. */
    [data-testid="stSidebar"] [class*="st-key-chat_row_"]:has(
        + .dm-recents-separator
    ) {
        margin-bottom: 0 !important;
    }

    /* Separator after Recents */
    [data-testid="stSidebar"] .dm-recents-separator {
        margin-top: var(--dm-sidebar-gap) !important;
        margin-bottom: var(--dm-sidebar-gap) !important;
    }

    /* Analytics and Knowledge use the same 12px separation */
    [data-testid="stSidebar"] .st-key-analytics_eval_nav,
    [data-testid="stSidebar"] .st-key-knowledge_nav {
        margin-top: 0 !important;
        margin-bottom: var(--dm-sidebar-gap) !important;
    }

    /* Make both nav buttons visually equal */
    [data-testid="stSidebar"] .st-key-analytics_eval_nav button,
    [data-testid="stSidebar"] .st-key-knowledge_nav button {
        min-height: 44px !important;
        height: 44px !important;
    }

    /*
     * Retrieval block:
     * same 12px separation from Knowledge Base.
     * Keep slider internals untouched so labels/ticks don't overlap.
     */
    [data-testid="stSidebar"] .st-key-workspace_top_k {
        margin-top: 0 !important;
        padding-top: var(--dm-sidebar-gap) !important;
        margin-bottom: 0 !important;
        border-top: 1px solid var(--dm-sidebar-border) !important;
    }

    /* Top-K label -> slider */
    [data-testid="stSidebar"] .st-key-workspace_top_k label {
        margin-bottom: var(--dm-sidebar-small-gap) !important;
    }

    [data-testid="stSidebar"] .st-key-workspace_top_k [data-testid="stSlider"] {
        margin-top: 0 !important;
        margin-bottom: var(--dm-sidebar-small-gap) !important;
    }

    /* Retrieval caption spacing */
    [data-testid="stSidebar"] .st-key-workspace_top_k + div {
        margin-top: var(--dm-sidebar-small-gap) !important;
        margin-bottom: 0 !important;
    }

    /*
     * Clear session:
     * same 12px gap above its divider and below retrieval text.
     */
    [data-testid="stSidebar"] .st-key-clear_session_sidebar {
        margin-top: var(--dm-sidebar-gap) !important;
        padding-top: var(--dm-sidebar-gap) !important;
        margin-bottom: 0 !important;
        border-top: 1px solid var(--dm-sidebar-border) !important;
    }

    [data-testid="stSidebar"] .st-key-clear_session_sidebar button {
        min-height: 44px !important;
        height: 44px !important;
    }

    /* Bottom note gets the same consistent separation. */
    [data-testid="stSidebar"] .st-key-clear_session_sidebar ~ div {
        margin-top: var(--dm-sidebar-gap) !important;
    }

    /* Keep text baselines consistent across main sidebar buttons. */
    [data-testid="stSidebar"] .st-key-new_chat_button button,
    [data-testid="stSidebar"] [class*="st-key-chat_nav_"] button,
    [data-testid="stSidebar"] .st-key-analytics_eval_nav button,
    [data-testid="stSidebar"] .st-key-knowledge_nav button,
    [data-testid="stSidebar"] .st-key-clear_session_sidebar button {
        line-height: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL CHAT GAP MATCH — SAME AS ANALYTICS / KNOWLEDGE
# ============================================================
st.markdown(
    """
    <style>
    /*
     * Make spacing between chat rows exactly match the spacing
     * between Analytics & Evaluation and Knowledge Base.
     */

    /* Chat rows */
    [data-testid="stSidebar"] [class*="st-key-chat_row_"] {
        margin-top: 0 !important;
        margin-bottom: .48rem !important;
        padding: 0 !important;
    }

    /* Remove any extra Streamlit spacing around chat-row wrappers */
    [data-testid="stSidebar"] [class*="st-key-chat_row_"] > div,
    [data-testid="stSidebar"] [class*="st-key-chat_row_"] [data-testid="stVerticalBlock"],
    [data-testid="stSidebar"] [class*="st-key-chat_row_"] [data-testid="stElementContainer"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Ensure each chat button has the same height as nav buttons */
    [data-testid="stSidebar"] [class*="st-key-chat_nav_"] button {
        min-height: 44px !important;
        height: 44px !important;
    }

    /* Analytics / Knowledge keep the exact same spacing */
    [data-testid="stSidebar"] .st-key-analytics_eval_nav,
    [data-testid="stSidebar"] .st-key-knowledge_nav {
        margin-top: 0 !important;
        margin-bottom: .48rem !important;
    }

    [data-testid="stSidebar"] .st-key-analytics_eval_nav button,
    [data-testid="stSidebar"] .st-key-knowledge_nav button {
        min-height: 44px !important;
        height: 44px !important;
    }

    /*
     * Do not add another separator-sized gap after the last chat.
     * The dedicated Recents divider supplies the section break.
     */
    [data-testid="stSidebar"] [class*="st-key-chat_row_"]:last-of-type {
        margin-bottom: .48rem !important;
    }

    /* Recents heading spacing stays compact and consistent */
    [data-testid="stSidebar"] .dm-nav-label {
        margin-bottom: .48rem !important;
    }

    /* Recents separator: same vertical rhythm on both sides */
    [data-testid="stSidebar"] .dm-recents-separator {
        margin-top: .48rem !important;
        margin-bottom: .48rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL FIX — CHAT ROW GAPS EXACTLY MATCH NAV BUTTON GAPS
# ============================================================
st.markdown(
    """
    <style>
    /*
     * The larger gap between recent chats was coming from TWO sources:
     * 1) our chat-row margin, and
     * 2) Streamlit's internal vertical-block gap between the visible
     *    chat button and the hidden functional delete button.
     *
     * Collapse only that INTERNAL chat-row gap, then use the same
     * .48rem outer gap as Analytics / Knowledge Base.
     */

    /* Remove all internal spacing inside each chat row container. */
    [data-testid="stSidebar"] [class*="st-key-chat_row_"]
    [data-testid="stVerticalBlock"] {
        gap: 0 !important;
        row-gap: 0 !important;
    }

    [data-testid="stSidebar"] [class*="st-key-chat_row_"]
    [data-testid="stElementContainer"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* The hidden real delete button must consume zero layout height. */
    [data-testid="stSidebar"] [class*="st-key-delete_chat_"],
    [data-testid="stSidebar"] [class*="st-key-delete_chat_"] > div,
    [data-testid="stSidebar"] [class*="st-key-delete_chat_"]
    [data-testid="stElementContainer"],
    [data-testid="stSidebar"] [class*="st-key-delete_chat_"]
    [data-testid="stButton"] {
        position: absolute !important;
        width: 1px !important;
        min-width: 1px !important;
        max-width: 1px !important;
        height: 1px !important;
        min-height: 1px !important;
        max-height: 1px !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* One and only one vertical gap between recent chats. */
    [data-testid="stSidebar"] [class*="st-key-chat_row_"] {
        margin-top: 0 !important;
        margin-bottom: .48rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Match the same gap used by Analytics / Knowledge Base exactly. */
    [data-testid="stSidebar"] .st-key-analytics_eval_nav,
    [data-testid="stSidebar"] .st-key-knowledge_nav {
        margin-top: 0 !important;
        margin-bottom: .48rem !important;
    }

    /* Keep all three row types the same height too. */
    [data-testid="stSidebar"] [class*="st-key-chat_nav_"] button,
    [data-testid="stSidebar"] .st-key-analytics_eval_nav button,
    [data-testid="stSidebar"] .st-key-knowledge_nav button {
        height: 44px !important;
        min-height: 44px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL SIDEBAR RHYTHM
# Equal gaps + visible divider between Recents and navigation
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --dm-sidebar-item-gap-final: 10px;
        --dm-sidebar-divider-color-final: rgba(145, 158, 150, .24);
    }

    /* --------------------------------------------------------
       RECENT CHAT ROWS
       -------------------------------------------------------- */
    [data-testid="stSidebar"] [class*="st-key-chat_row_"] {
        margin: 0 0 var(--dm-sidebar-item-gap-final) 0 !important;
        padding: 0 !important;
    }

    /* Remove any hidden internal spacing from the chat-row container. */
    [data-testid="stSidebar"] [class*="st-key-chat_row_"]
    [data-testid="stVerticalBlock"],
    [data-testid="stSidebar"] [class*="st-key-chat_row_"]
    [data-testid="stElementContainer"] {
        gap: 0 !important;
        row-gap: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* All sidebar item buttons use the same height. */
    [data-testid="stSidebar"] [class*="st-key-chat_nav_"] button,
    [data-testid="stSidebar"] .st-key-analytics_eval_nav button,
    [data-testid="stSidebar"] .st-key-knowledge_nav button {
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        border-radius: 12px !important;
        box-sizing: border-box !important;
    }

    /* --------------------------------------------------------
       DIVIDER AFTER RECENT CHATS
       -------------------------------------------------------- */
    [data-testid="stSidebar"] .dm-recents-separator {
        display: block !important;
        visibility: visible !important;

        width: 100% !important;
        height: 1px !important;
        min-height: 1px !important;

        margin:
            0
            0
            var(--dm-sidebar-item-gap-final)
            0 !important;

        padding: 0 !important;

        background:
            var(--dm-sidebar-divider-color-final) !important;
        border: 0 !important;
        opacity: 1 !important;
    }

    /*
     * The final chat row already has the same 10px gap before the divider.
     * The divider then has the same 10px gap before Analytics.
     */

    /* --------------------------------------------------------
       ANALYTICS + KNOWLEDGE
       -------------------------------------------------------- */
    [data-testid="stSidebar"] .st-key-analytics_eval_nav,
    [data-testid="stSidebar"] .st-key-knowledge_nav {
        margin-top: 0 !important;
        margin-bottom: var(--dm-sidebar-item-gap-final) !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Prevent generic sidebar rules from introducing extra gaps. */
    [data-testid="stSidebar"] .st-key-analytics_eval_nav
    [data-testid="stElementContainer"],
    [data-testid="stSidebar"] .st-key-knowledge_nav
    [data-testid="stElementContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* --------------------------------------------------------
       RECENTS LABEL
       -------------------------------------------------------- */
    [data-testid="stSidebar"] .dm-nav-label {
        margin-top: 0 !important;
        margin-bottom: var(--dm-sidebar-item-gap-final) !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL SIDEBAR GAP FIX
# Recent chats now use the exact same spacing as Analytics / Knowledge
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --dm-sidebar-row-gap: 10px;
    }

    /*
     * IMPORTANT:
     * The hidden Streamlit delete button was still reserving layout space
     * inside every recent-chat container. Hide its ENTIRE element wrapper
     * from layout. The custom SVG trash icon still triggers the hidden
     * button programmatically, so delete functionality remains intact.
     */
    [data-testid="stSidebar"]
    [class*="st-key-chat_row_"]
    [data-testid="stElementContainer"]:has([class*="st-key-delete_chat_"]) {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        min-width: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    /* Remove internal extra spacing from each recent-chat container. */
    [data-testid="stSidebar"]
    [class*="st-key-chat_row_"]
    [data-testid="stVerticalBlock"] {
        gap: 0 !important;
        row-gap: 0 !important;
    }

    [data-testid="stSidebar"]
    [class*="st-key-chat_row_"] {
        margin-top: 0 !important;
        margin-bottom: var(--dm-sidebar-row-gap) !important;
        padding: 0 !important;
    }

    /* Analytics and Knowledge use the SAME exact outer gap. */
    [data-testid="stSidebar"] .st-key-analytics_eval_nav,
    [data-testid="stSidebar"] .st-key-knowledge_nav {
        margin-top: 0 !important;
        margin-bottom: var(--dm-sidebar-row-gap) !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* All four row types use exactly the same height. */
    [data-testid="stSidebar"] [class*="st-key-chat_nav_"] button,
    [data-testid="stSidebar"] .st-key-analytics_eval_nav button,
    [data-testid="stSidebar"] .st-key-knowledge_nav button {
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        box-sizing: border-box !important;
    }

    /*
     * Separator remains visible after the final recent chat.
     * Use the same spacing unit above and below it.
     */
    [data-testid="stSidebar"] .dm-recents-separator {
        display: block !important;
        width: 100% !important;
        height: 1px !important;
        min-height: 1px !important;

        margin-top: 0 !important;
        margin-bottom: var(--dm-sidebar-row-gap) !important;
        padding: 0 !important;

        background: rgba(145, 158, 150, .24) !important;
        opacity: 1 !important;
        border: 0 !important;
    }

    /* Avoid doubling the space immediately before the separator. */
    [data-testid="stSidebar"]
    [class*="st-key-chat_row_"]:has(
        + [data-testid="stElementContainer"] .dm-recents-separator
    ) {
        margin-bottom: var(--dm-sidebar-row-gap) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FINAL GLOBAL THEME POSITION — DESKTOP + MOBILE
# ============================================================
st.markdown(
    """
    <style>
    .st-key-dm_global_theme_control {
        position: fixed !important;
        top: 5.35rem !important;
        right: 2rem !important;
        left: auto !important;
        bottom: auto !important;
        width: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        overflow: visible !important;
        z-index: 100000 !important;
    }

    .st-key-dm_global_theme_control > div,
    .st-key-dm_global_theme_control [data-testid="stVerticalBlock"],
    .st-key-dm_global_theme_control [data-testid="stElementContainer"],
    .st-key-dm_global_theme_control [data-testid="stRadio"] {
        width: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }

    /* Main pill */
    .st-key-dm_global_theme_control [role="radiogroup"] {
        position: relative !important;
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        align-items: center !important;
        gap: 0 !important;

        width: 92px !important;
        height: 44px !important;
        margin: 0 !important;
        padding: 4px !important;

        border-radius: 999px !important;
        border: 1px solid rgba(255,255,255,.10) !important;
        background: #0d1320 !important;
        box-shadow: 0 8px 24px rgba(0,0,0,.18) !important;
        overflow: hidden !important;
    }

    /* Both icon options */
    .st-key-dm_global_theme_control [role="radio"] {
        position: relative !important;
        z-index: 2 !important;

        width: 42px !important;
        min-width: 42px !important;
        height: 36px !important;
        min-height: 36px !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;

        margin: 0 !important;
        padding: 0 !important;

        border: 0 !important;
        border-radius: 999px !important;
        background: transparent !important;
        color: #9aa5ba !important;
        box-shadow: none !important;
        cursor: pointer !important;

        transition: color .18s ease, transform .18s ease !important;
    }

    .st-key-dm_global_theme_control [role="radio"]:hover {
        color: #ffffff !important;
        transform: scale(1.03) !important;
    }

    /* Hide native radio circle entirely */
    .st-key-dm_global_theme_control [role="radio"] > div:first-child {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .st-key-dm_global_theme_control [role="radio"] p,
    .st-key-dm_global_theme_control [role="radio"] span {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 18px !important;
        line-height: 1 !important;
        color: inherit !important;
    }

    /* Selected side gets the raised circular/rounded knob effect */
    .st-key-dm_global_theme_control [role="radio"][aria-checked="true"] {
        background: #2f3b63 !important;
        color: #ffffff !important;
        box-shadow:
            inset 0 0 0 1px rgba(255,255,255,.04),
            0 4px 12px rgba(0,0,0,.28) !important;
    }

    /* Slightly mute unselected icon */
    .st-key-dm_global_theme_control [role="radio"][aria-checked="false"] {
        opacity: .78 !important;
    }

    .dm-right-panel-title {
        margin-top: 2.45rem !important;
    }

    @media (max-width: 900px) {
        .st-key-dm_global_theme_control {
            top: 4rem !important;
            right: .65rem !important;
            z-index: 2147483000 !important;
        }

        .st-key-dm_global_theme_control [role="radiogroup"] {
            width: 84px !important;
            height: 40px !important;
            padding: 4px !important;
        }

        .st-key-dm_global_theme_control [role="radio"] {
            width: 38px !important;
            min-width: 38px !important;
            height: 32px !important;
            min-height: 32px !important;
        }

        .st-key-dm_global_theme_control [role="radio"] p,
        .st-key-dm_global_theme_control [role="radio"] span {
            font-size: 16px !important;
        }

        .dm-right-panel-title {
            margin-top: 0 !important;
        }
    }

    @media (max-width: 430px) {
        .st-key-dm_global_theme_control {
            top: 3.8rem !important;
            right: .45rem !important;
        }

        .st-key-dm_global_theme_control [role="radiogroup"] {
            width: 78px !important;
            height: 38px !important;
            padding: 3px !important;
        }

        .st-key-dm_global_theme_control [role="radio"] {
            width: 36px !important;
            min-width: 36px !important;
            height: 32px !important;
            min-height: 32px !important;
        }

        .st-key-dm_global_theme_control [role="radio"] p,
        .st-key-dm_global_theme_control [role="radio"] span {
            font-size: 15px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# COMPACT PROFESSIONAL THEME TOGGLE
# ============================================================
# Disabled old toggle styles to avoid conflicts with the final moon/sun toggle above.



# ============================================================
# FINAL MOBILE CHAT COMPOSER CENTERING FIX
# ============================================================
# Important:
# Earlier desktop composer experiments use a high-specificity selector:
# .st-key-dm_center_chat[data-composer-pinned="true"] .st-key-dm_center_composer
# with fixed left/width values. In responsive/mobile mode those desktop
# coordinates can survive a DevTools viewport change. The rules below come
# LAST and explicitly neutralize that state for <= 900px.
st.markdown(
    """
    <style>
    @media (max-width: 900px) {

        /*
         * The center chat itself must use the full available mobile width.
         * Do not let old desktop width/left custom properties influence it.
         */
        .st-key-dm_center_chat,
        .st-key-dm_center_chat[data-composer-pinned="true"] {
            --dm-composer-left: 0px !important;
            --dm-composer-width: 100% !important;
            --dm-composer-bottom: 0px !important;
            --dm-composer-space: 0px !important;

            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;

            margin-left: auto !important;
            margin-right: auto !important;

            box-sizing: border-box !important;
        }

        /*
         * Override BOTH the normal composer selector and the old
         * data-composer-pinned desktop selector with equal/higher specificity.
         */
        .st-key-dm_center_chat .st-key-dm_center_composer,
        .st-key-dm_center_chat[data-composer-pinned="true"] .st-key-dm_center_composer,
        .st-key-dm_center_chat.st-key-dm_center_chat[data-composer-pinned="true"] .st-key-dm_center_composer {
            position: sticky !important;

            left: auto !important;
            right: auto !important;
            top: auto !important;
            bottom: 0 !important;

            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;

            margin-left: auto !important;
            margin-right: auto !important;

            transform: none !important;

            padding:
                .55rem
                .35rem
                max(.55rem, env(safe-area-inset-bottom))
                .35rem !important;

            box-sizing: border-box !important;
            background: var(--dm-bg) !important;
            z-index: 9990 !important;
        }

        /*
         * Streamlit wrappers around the chat input also need to stay fluid.
         * Otherwise a desktop wrapper width can make the visible input appear
         * offset even when the composer itself is 100% wide.
         */
        .st-key-dm_center_composer > div,
        .st-key-dm_center_composer [data-testid="stVerticalBlock"],
        .st-key-dm_center_composer [data-testid="stElementContainer"],
        .st-key-dm_center_composer .st-key-agent_v2_input,
        .st-key-dm_center_composer .st-key-agent_v2_input > div,
        .st-key-dm_center_composer .st-key-agent_v2_input [data-testid="stChatInput"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;

            left: auto !important;
            right: auto !important;

            margin-left: auto !important;
            margin-right: auto !important;

            transform: none !important;
            box-sizing: border-box !important;
        }

        /*
         * Give the actual input a small, symmetric phone-screen gutter.
         * This keeps it visually centered like ChatGPT on mobile.
         */
        .st-key-dm_center_composer .st-key-agent_v2_input {
            padding-left: .25rem !important;
            padding-right: .25rem !important;
        }

        .st-key-dm_center_composer [data-testid="stChatInput"] {
            margin: 0 auto !important;
        }

        /*
         * Prevent any horizontal overflow caused by old desktop measurements.
         */
        [data-testid="stMain"]:has(.st-key-dm_center_chat),
        [data-testid="stMainBlockContainer"]:has(.st-key-dm_center_chat),
        .block-container:has(.st-key-dm_center_chat) {
            overflow-x: hidden !important;
        }
    }

    @media (max-width: 480px) {
        .st-key-dm_center_chat .st-key-dm_center_composer,
        .st-key-dm_center_chat[data-composer-pinned="true"] .st-key-dm_center_composer,
        .st-key-dm_center_chat.st-key-dm_center_chat[data-composer-pinned="true"] .st-key-dm_center_composer {
            padding-left: .2rem !important;
            padding-right: .2rem !important;
        }

        .st-key-dm_center_composer .st-key-agent_v2_input {
            padding-left: .15rem !important;
            padding-right: .15rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

