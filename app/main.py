from pathlib import Path
import sys

import pandas as pd
import streamlit as st


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
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME STATE + CUSTOM FRONTEND CSS
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "light"


def toggle_theme():
    st.session_state.theme = (
        "dark" if st.session_state.theme == "light" else "light"
    )


if st.session_state.theme == "dark":
    THEME = {
        "bg": "#0b1220",
        "surface": "#121c2d",
        "surface2": "#17243a",
        "surface3": "#1c2a42",
        "text": "#e8eef8",
        "heading": "#f8fbff",
        "muted": "#9aaac2",
        "border": "#2a3a55",
        "primary": "#6f8fff",
        "primary2": "#4e72ef",
        "soft": "#162b50",
        "success": "#35c79a",
        "warning": "#f2bd68",
        "danger": "#f07474",
        "input": "#0f1929",
        "input_text": "#f4f7fd",
        "placeholder": "#8798b3",
        "sidebar": "#0c1730",
        "sidebar2": "#132443",
    }
else:
    THEME = {
        "bg": "#f5f8fc",
        "surface": "#ffffff",
        "surface2": "#f8faff",
        "surface3": "#eef4fb",
        "text": "#172b49",
        "heading": "#071d3d",
        "muted": "#4d6382",
        "border": "#c5d3e4",
        "primary": "#3568ee",
        "primary2": "#2856d7",
        "soft": "#edf4ff",
        "success": "#159b73",
        "warning": "#b87516",
        "danger": "#c94c4c",
        "input": "#ffffff",
        "input_text": "#172b49",
        "placeholder": "#7185a3",
        "sidebar": "#0d1930",
        "sidebar2": "#14274a",
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
        --dm-sidebar-text:{'#172b49' if st.session_state.theme == 'light' else '#eaf1ff'};
        --dm-sidebar-muted:{'#4d6382' if st.session_state.theme == 'light' else '#a9b9d5'};
        --dm-sidebar-value:{'#071d3d' if st.session_state.theme == 'light' else '#ffffff'};
        --dm-sidebar-border:{'#d6e0ec' if st.session_state.theme == 'light' else 'rgba(255,255,255,.08)'};
        --dm-sidebar-hover:{'#edf4ff' if st.session_state.theme == 'light' else '#132443'};

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
        max-width:1480px !important;
        width:100% !important;
        box-sizing:border-box !important;
        padding-top:6.5rem !important;
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
        background:linear-gradient(135deg,#3568ee,#7895ff);
        color:#fff !important;font-size:20px;
        box-shadow:0 8px 24px rgba(53,104,238,.28);
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
/* ---------- Fake global search ---------- */
.dm-fake-search {{
    height:42px;
    width:100%;
    box-sizing:border-box;
    display:flex;
    align-items:center;
    gap:10px;
    padding:0 15px;
    background:var(--dm-input) !important;
    border:1.5px solid var(--dm-border) !important;
    border-radius:12px;
    color:var(--dm-placeholder) !important;
    font-size:.86rem;
    box-shadow:0 4px 18px rgba(35,61,110,.06);
}}

.dm-search-icon {{
    font-size:20px;
    color:var(--dm-muted);
}}
    .dm-user-box {{color:var(--dm-muted) !important;text-align:right;font-size:.82rem;padding-top:6px;}}
    .dm-user-box b {{color:var(--dm-heading) !important;}}

    .dm-welcome {{color:var(--dm-heading) !important;font-size:1.72rem;font-weight:800;line-height:1.2;margin:4px 0;}}
    .dm-subtitle {{color:var(--dm-muted) !important;font-size:.9rem;line-height:1.55;}}

    .dm-page-title {{color:var(--dm-heading) !important;font-size:1.45rem;font-weight:800;margin:4px 0;}}
    .dm-page-subtitle {{color:var(--dm-muted) !important;font-size:.84rem;margin-bottom:16px;}}

    /* ---------- Cards ---------- */
    .dm-metric-card {{
        background:var(--dm-surface) !important;
        border:1px solid var(--dm-border) !important;
        border-radius:13px;padding:15px;min-height:98px;
        box-shadow:0 7px 24px rgba(35,61,110,.055);
    }}
    .dm-metric-label {{color:var(--dm-muted) !important;font-size:.73rem;font-weight:650;margin-bottom:6px;}}
    .dm-metric-value {{color:var(--dm-heading) !important;font-size:1.4rem;font-weight:800;}}
    .dm-metric-delta {{color:var(--dm-success) !important;font-size:.69rem;margin-top:5px;}}

    .dm-ask-card {{
        background:linear-gradient(135deg,var(--dm-soft),var(--dm-surface2)) !important;
        border:1px solid var(--dm-border) !important;border-radius:15px;
        padding:18px;margin:6px 0 18px;
    }}
    .dm-ask-title {{color:var(--dm-heading) !important;font-size:1rem;font-weight:800;}}
    .dm-ask-text {{color:var(--dm-muted) !important;font-size:.76rem;margin:4px 0 10px;}}
    .dm-chip {{
        display:inline-block;background:var(--dm-surface) !important;
        border:1px solid var(--dm-border) !important;color:var(--dm-muted) !important;
        border-radius:999px;padding:7px 10px;margin:3px 4px 0 0;font-size:.67rem;
    }}

    /* ---------- Typography ---------- */
    .stMarkdown, .stMarkdown p, .stCaption, .stText,
    .stSubheader, h1,h2,h3,h4,h5,h6,label,
    [data-testid="stWidgetLabel"] p {{color:var(--dm-text) !important;}}
    h1,h2,h3,h4,h5,h6 {{color:var(--dm-heading) !important;}}
    .stCaption, [data-testid="stCaptionContainer"] * {{color:var(--dm-muted) !important;}}

    /* ---------- Inputs ---------- */
    .stTextInput input,.stTextArea textarea,.stNumberInput input,
    input,textarea {{
        background:var(--dm-input) !important;
        color:var(--dm-input-text) !important;
        -webkit-text-fill-color:var(--dm-input-text) !important;
        border:1px solid var(--dm-border) !important;border-radius:11px !important;
        caret-color:var(--dm-primary) !important;
    }}
    .stTextInput input::placeholder,.stTextArea textarea::placeholder,
    input::placeholder,textarea::placeholder {{
        color:var(--dm-placeholder) !important;
        -webkit-text-fill-color:var(--dm-placeholder) !important;opacity:1 !important;
    }}

    /* ---------- Buttons ---------- */
    .stButton > button,.stDownloadButton > button {{
        min-height:40px !important;border-radius:10px !important;
        border:1px solid var(--dm-border) !important;background:var(--dm-surface) !important;
        color:var(--dm-text) !important;font-weight:700 !important;
        box-shadow:0 3px 12px rgba(35,61,110,.04) !important;opacity:1 !important;
    }}
    .stButton > button *, .stDownloadButton > button * {{
        color:inherit !important;fill:currentColor !important;opacity:1 !important;visibility:visible !important;
    }}
    .stButton > button:hover,.stDownloadButton > button:hover {{border-color:var(--dm-primary) !important;}}
    .stButton > button[kind="primary"],.stButton > button[data-testid="baseButton-primary"] {{
        background:linear-gradient(135deg,var(--dm-primary),var(--dm-primary2)) !important;
        border-color:var(--dm-primary) !important;color:#fff !important;
    }}
    .stButton > button[kind="primary"] *,.stButton > button[data-testid="baseButton-primary"] * {{color:#fff !important;fill:#fff !important;}}

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploader"] {{background:var(--dm-surface) !important;border-radius:13px !important;}}
    [data-testid="stFileUploaderDropzone"] {{
        background:var(--dm-surface2) !important;border:1px dashed var(--dm-border) !important;
        border-radius:12px !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{color:var(--dm-text) !important;opacity:1 !important;visibility:visible !important;}}
    [data-testid="stFileUploaderDropzone"] button {{background:var(--dm-primary) !important;border-color:var(--dm-primary) !important;color:#fff !important;}}
    [data-testid="stFileUploaderDropzone"] button * {{color:#fff !important;fill:#fff !important;}}

    /* ---------- Native Streamlit components ---------- */
    [data-testid="stMetric"] {{background:var(--dm-surface) !important;border:1px solid var(--dm-border) !important;border-radius:12px !important;}}
    [data-testid="stMetricLabel"] p,[data-testid="stMetricValue"],[data-testid="stMetricDelta"] {{color:var(--dm-heading) !important;}}
    [data-testid="stExpander"] {{background:var(--dm-surface) !important;border-color:var(--dm-border) !important;}}
    [data-testid="stExpander"] * {{color:var(--dm-text);}}
    [data-testid="stAlert"] p,[data-testid="stAlert"] span {{color:inherit !important;}}
    [data-baseweb="select"] {{background:var(--dm-input) !important;color:var(--dm-input-text) !important;}}
    [data-baseweb="select"] * {{color:var(--dm-input-text) !important;}}
    hr {{border-color:var(--dm-border) !important;}}
    [data-testid="stDataFrame"] {{border:1px solid var(--dm-border) !important;border-radius:10px;overflow:hidden;}}

    /* ---------- Visibility / layout failsafe ---------- */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    .block-container {{
        visibility:visible !important;
        opacity:1 !important;
    }}
    [data-testid="stMainBlockContainer"] {{
        width:100% !important;
        max-width:none !important;
        min-width:0 !important;
    }}
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="column"] {{
        visibility:visible !important;
        opacity:1 !important;
    }}
    [data-testid="stMarkdownContainer"] {{
        visibility:visible !important;
        opacity:1 !important;
    }}
    .dm-search-box, .dm-user-box, .dm-welcome, .dm-subtitle,
    .dm-page-title, .dm-page-subtitle, .dm-metric-card, .dm-ask-card,
    .dm-brand, .dm-theme-card, .dm-nav-label {{
        display:block !important;
        visibility:visible !important;
        opacity:1 !important;
    }}

    /* Chat */
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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="dm-brand">
            <div class="dm-brand-icon">◆</div>
            <div>
                <div class="dm-brand-title">DocMind</div>
                <div class="dm-brand-sub">RAG Knowledge Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="dm-nav-label">Main</div>', unsafe_allow_html=True)

    page = st.radio(
        "Main navigation",
        [
            "Dashboard",
            "AI Chat",
            "Knowledge Base",
            "Documents",
            "Search",
            "Analytics",
            "Evaluation",
            "Retrieval",
            "Conversations",
            "Settings",
        ],
        index=0,
        label_visibility="collapsed",
        key="docmind_page",
    )

    st.divider()

    st.markdown('<div class="dm-nav-label">Appearance</div>', unsafe_allow_html=True)
    current_dark = st.session_state.theme == "dark"
    theme_label = "☀️ Switch to Light" if current_dark else "🌙 Switch to Dark"
    if st.button(
        theme_label,
        key="sidebar_theme_button",
        width="stretch",
        help="Change the entire DocMind interface theme.",
    ):
        toggle_theme()
        st.rerun()

    st.markdown(
        f'<div class="dm-theme-note">Current theme: <b>{"Dark" if current_dark else "Light"}</b></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    try:
        vector_store = get_vector_store()
        indexed_chunks = vector_store.count()

        try:
            collection_data = vector_store.collection.get(include=["metadatas"])
            metadatas = collection_data.get("metadatas", [])
            indexed_documents = len({
                metadata.get("source")
                for metadata in metadatas
                if metadata and metadata.get("source")
            })
        except Exception:
            indexed_documents = 0

        st.markdown("#### 📚 Knowledge Base")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", indexed_documents)
        with col2:
            st.metric("Chunks", indexed_chunks)
    except Exception as exc:
        st.error(f"Knowledge base error: {exc}")

    st.divider()

    st.markdown("#### ⚙️ Retrieval")
    top_k = st.slider(
        "Top-K chunks",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of relevant document chunks retrieved for each question.",
    )
    st.caption(f"Retrieving the top **{top_k}** relevant chunks.")

    st.divider()

    st.markdown("#### 💬 Conversation")
    conversation_count = len(st.session_state.conversation_history)
    st.metric("Questions asked", conversation_count)
    if conversation_count > 0:
        if st.button("🗑️ Clear Conversation", width="stretch"):
            st.session_state.conversation_history = []
            st.session_state.last_result = None
            st.rerun()

    st.divider()

    st.markdown("#### 🟢 System Status")
    st.success("RAG Pipeline Ready")
    for item in [
        "✓ Local embeddings",
        "✓ ChromaDB vector store",
        "✓ Hybrid Dense + BM25 retrieval",
        "✓ Cross-encoder reranking",
        "✓ Grounded generation",
        "✓ Source citations",
        "✓ Conversation history",
        "✓ Evaluation & analytics",
    ]:
        st.caption(item)

    st.divider()
    st.info(
        "DocMind answers questions using retrieved document content rather than general model knowledge."
    )


# ============================================================
# HEADER
# ============================================================

# This real layout spacer is intentional: it pushes the first DocMind
# element below Streamlit's fixed header/Deploy toolbar.
search_col, status_col, user_col = st.columns([6.0, 1.0, 1.2])

with search_col:
    st.markdown(
        """
        <div class="dm-fake-search">
            <span class="dm-search-icon">⌕</span>
            <span>Search your documents, knowledge base, or ask a question...</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col:
    st.markdown(
        '<div class="dm-user-box">● <b>RAG Ready</b></div>',
        unsafe_allow_html=True,
    )

with user_col:
    st.markdown(
        '<div class="dm-user-box">🔔 &nbsp; 👤 <b>Admin</b></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

from datetime import datetime
_current_hour = datetime.now().hour
if 5 <= _current_hour < 12:
    _greeting = "Good morning"
elif 12 <= _current_hour < 18:
    _greeting = "Good afternoon"
else:
    _greeting = "Good evening"

st.markdown(
    f'<div class="dm-welcome">{_greeting}, ReXy 👋</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="dm-subtitle">Your knowledge at a glance. Search documents, ask grounded questions, and inspect retrieval evidence from one workspace.</div>',
    unsafe_allow_html=True,
)
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ============================================================
# TOP METRICS
# ============================================================

try:
    document_manager = get_document_manager()
    manager_stats = document_manager.get_statistics()

    managed_documents = manager_stats.get("document_count", 0)
    managed_chunks = manager_stats.get("indexed_chunks", 0)
    storage_bytes = manager_stats.get("total_size_bytes", 0)
    storage_mb = storage_bytes / (1024 * 1024)

except Exception:
    managed_documents = 0
    managed_chunks = 0
    storage_mb = 0

metric_cols = st.columns(4)

metric_data = [
    ("📄", "Total Documents", managed_documents, "Managed files"),
    ("🧩", "Indexed Chunks", managed_chunks, "Available to retrieve"),
    ("💾", "Storage", f"{storage_mb:.2f} MB", "Document storage"),
    ("🟢", "Ingestion Status", "Ready", "All systems operational"),
]

for col, (icon, label, value, delta) in zip(metric_cols, metric_data):
    with col:
        st.markdown(
            f"""
            <div class="dm-metric-card">
                <div class="dm-metric-label">{icon} &nbsp; {label}</div>
                <div class="dm-metric-value">{value}</div>
                <div class="dm-metric-delta">✓ {delta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# ============================================================
# APPLICATION NAVIGATION
# ============================================================

page_descriptions = {
    "Dashboard": "A high-level view of your knowledge base and the main RAG workspace.",
    "AI Chat": "Ask grounded questions and inspect answers, sources, and retrieval diagnostics.",
    "Knowledge Base": "Manage indexed knowledge and inspect document/index statistics.",
    "Documents": "Upload, index, re-index, inspect, and delete supported documents.",
    "Search": "Use the existing RAG question workflow as your semantic knowledge search.",
    "Analytics": "Explore the existing RAG evaluation and analytics datasets.",
    "Evaluation": "Review recorded evaluation experiments and answer-quality results.",
    "Retrieval": "Inspect retrieval settings and the latest retrieval diagnostics.",
    "Conversations": "Review the current browser-session conversation history.",
    "Settings": "Review appearance, retrieval controls, and current system status.",
}

st.markdown(
    f'<div class="dm-page-title">{page}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="dm-page-subtitle">{page_descriptions[page]}</div>',
    unsafe_allow_html=True,
)

if page in {"Dashboard", "AI Chat", "Search", "Conversations"}:

        st.markdown(
            """
            <div class="dm-ask-card">
                <div class="dm-ask-title">🤖 Ask your knowledge base</div>
                <div class="dm-ask-text">Get instant, accurate answers from your documents using RAG.</div>
                <span class="dm-chip">⌕ What is the main topic of the project?</span>
                <span class="dm-chip">⌕ Summarize the key findings</span>
                <span class="dm-chip">⌁ List the important requirements</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

                with st.chat_message(
                    "user"
                ):

                    st.markdown(
                        question_text
                    )

                with st.chat_message(
                    "assistant"
                ):

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

                    if sources:

                        with st.expander(
                            f"📚 Sources ({len(sources)})"
                        ):

                            for source in sources:

                                render_source_card(
                                    source
                                )

                    render_retrieval_details(
                        conversation,
                        expanded=False,
                    )

                if index < (
                    len(
                        st.session_state.conversation_history
                    ) - 1
                ):

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
                width="stretch",
            )

        with col2:

            clear_button = st.button(
                "🧹 Clear Input",
                width="stretch",
            )

        with col3:

            clear_history_button = st.button(
                "🗑️ Clear Conversation",
                width="stretch",
            )

        if clear_button:

            st.rerun()

        if clear_history_button:

            st.session_state.conversation_history = []
            st.session_state.last_result = None

            st.rerun()

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

        # ========================================================
        # CURRENT ANSWER
        # ========================================================

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

            sources = result.get(
                "sources",
                [],
            )

            st.subheader(
                f"📚 Sources ({len(sources)})"
            )

            if sources:

                for source in sources:

                    render_source_card(
                        source
                    )

            else:

                st.info(
                    "No source information was returned."
                )

            render_retrieval_details(
                result,
                expanded=False,
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
                width="stretch",
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
                width="stretch",
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

                The question is processed using hybrid retrieval:

                - Dense semantic retrieval
                - BM25 lexical retrieval
                - Weighted score fusion

                **6. Reranking**

                Candidate chunks are reranked using a cross-encoder
                when enabled.

                **7. Context building**

                Retrieved chunks are combined into grounded context.

                **8. Generation**

                Ollama Cloud generates an answer using the retrieved
                context.

                **9. Citations**

                Source filename, category, page and chunk information
                are displayed with the answer.

                **10. Observability**

                DocMind records retrieval, context-building and
                generation timings together with retrieval diagnostics.

                **11. Abstention**

                If the information cannot be supported by the available
                documents, DocMind responds:

                > I don't know based on the available documents.

                **12. Conversation history**

                Previous questions and answers are displayed for the
                current Streamlit session. Each new question still
                performs document retrieval independently.
                """
            )

if page in {"Analytics", "Evaluation"}:

        render_evaluation_dashboard()

if page in {"Knowledge Base", "Documents"}:

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
                width="stretch",
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
                width="stretch",
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
                width="stretch",
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
                    size_bytes
                    / 1024
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
                        width="stretch",
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
                        width="stretch",
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

if page == "Retrieval":
    st.markdown("### Retrieval configuration")
    st.info(f"The active retrieval setting is Top-K = {top_k}.")
    st.markdown("""
    **Existing retrieval pipeline**

    Dense semantic retrieval + BM25 lexical retrieval → weighted hybrid fusion → optional cross-encoder reranking → Top-K context → grounded generation.
    """)
    if st.session_state.last_result:
        render_retrieval_details(st.session_state.last_result, expanded=True)
    else:
        st.info("Ask a question from AI Chat first to populate live retrieval diagnostics.")

if page == "Settings":
    st.markdown("### Appearance")
    settings_col1, settings_col2 = st.columns(2)
    with settings_col1:
        st.metric("Current theme", "Dark" if st.session_state.theme == "dark" else "Light")
        if st.button("☀️ Switch to Light" if st.session_state.theme == "dark" else "🌙 Switch to Dark", key="settings_theme", width="stretch"):
            toggle_theme()
            st.rerun()
    with settings_col2:
        st.metric("RAG status", "Ready")
        st.caption("The interface theme is stored in Streamlit session state and reapplied on reruns.")

    st.markdown("### Retrieval")
    st.write(f"Active Top-K chunks: **{top_k}**")
    st.caption("The Top-K control remains available in the sidebar from every page.")

    st.markdown("### System")
    for item in [
        "Local embeddings",
        "Persistent ChromaDB",
        "Hybrid Dense + BM25 retrieval",
        "Optional cross-encoder reranking",
        "Grounded Ollama Cloud generation",
        "Application-controlled citations",
        "Conversation history",
        "Evaluation and analytics",
    ]:
        st.markdown(f"✓ {item}")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DocMind — RAG Knowledge Assistant | "
    "Hybrid Retrieval + Reranking + "
    "Grounded Generation + Source Attribution"
)
