# DocMind — RAG Knowledge Assistant

DocMind is an end-to-end Retrieval-Augmented Generation (RAG) knowledge assistant designed to answer questions from a controlled, non-confidential document knowledge base.

The system combines document ingestion, text cleaning, configurable chunking, local semantic embeddings, persistent ChromaDB storage, dense retrieval, BM25 lexical retrieval, hybrid search, optional cross-encoder reranking, grounded answer generation through Ollama Cloud, source citations, explicit abstention, evaluation experiments, and prompt-injection security testing.

The project demonstrates the complete lifecycle of a practical RAG application, from document ingestion and indexing to retrieval, grounded generation, evaluation, security testing, and interactive application delivery.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Key Features](#key-features)
* [System Architecture](#system-architecture)
* [RAG Pipeline](#rag-pipeline)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Knowledge Base](#knowledge-base)
* [Installation](#installation)
* [Environment Configuration](#environment-configuration)
* [Running the Application](#running-the-application)
* [Document Ingestion and Indexing](#document-ingestion-and-indexing)
* [Retrieval System](#retrieval-system)
* [Reranking](#reranking)
* [Grounded Answer Generation](#grounded-answer-generation)
* [Citations and Abstention](#citations-and-abstention)
* [Evaluation Framework](#evaluation-framework)
* [Normal LLM vs RAG Evaluation](#normal-llm-vs-rag-evaluation)
* [Security and Prompt Injection](#security-and-prompt-injection)
* [Testing](#testing)
* [Current Evaluation Results](#current-evaluation-results)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)
* [Internship Learning Outcomes](#internship-learning-outcomes)
* [Project Status](#project-status)
* [Author](#author)

---

## Project Overview

Large language models can generate fluent responses, but they may not have access to an organization's private, internal, or domain-specific information.

DocMind addresses this problem using Retrieval-Augmented Generation.

Instead of asking the language model to answer entirely from its pretrained knowledge, DocMind first retrieves relevant information from a controlled knowledge base and provides that information to the language model as context.

The high-level process is:

```text
User Question
     ↓
Retrieve Relevant Knowledge
     ↓
Build Grounded Context
     ↓
Generate Answer Using Context
     ↓
Return Answer + Sources
```

If sufficient evidence cannot be found in the knowledge base, the system can explicitly abstain instead of fabricating an answer.

---

## Key Features

### Document Processing

* PDF ingestion
* DOCX ingestion
* TXT ingestion
* Text cleaning
* Configurable character-based chunking
* Chunk overlap
* Page-level metadata for PDFs
* Document metadata preservation
* File validation
* Duplicate filename protection

### Knowledge Base Management

* Upload documents
* Upload and index documents
* Re-index individual documents
* Re-index all documents
* Delete documents
* Document listing
* Document statistics
* Persistent local ChromaDB storage

### Embeddings

* Local Sentence Transformer embeddings
* `sentence-transformers/all-MiniLM-L6-v2`
* 384-dimensional embeddings
* Normalized embeddings
* Batch embedding support

### Retrieval

* Dense semantic retrieval
* BM25 lexical retrieval
* Hybrid Dense + BM25 retrieval
* Weighted score fusion
* Top-K retrieval
* Retrieval diagnostics
* Retrieval distance analysis

### Reranking

* Optional cross-encoder reranking
* `cross-encoder/ms-marco-MiniLM-L-6-v2`
* Candidate generation followed by reranking
* Reranking experiments and evaluation

### Grounded Generation

* Ollama Cloud integration
* Configurable LLM model
* Context-only answering
* Explicit abstention
* Application-controlled source metadata
* Protection against instructions embedded in retrieved documents

### Evaluation

* Chunking experiments
* Retrieval evaluation
* Hybrid retrieval experiments
* Hybrid weight experiments
* Reranking experiments
* Retrieval distance analysis
* Answer-quality evaluation
* Normal LLM vs RAG comparison
* Context relevance analysis
* Failure analysis
* Latency analysis
* Knowledge-base analytics

### Security

* Prompt-injection testing
* Retrieved documents treated as untrusted data
* Protection against attempts to override system instructions
* Protection against prompt/secret disclosure attempts
* Security experiment documentation

### Interactive Application

* Streamlit interface
* RAG question answering
* Knowledge-base management
* Evaluation dashboard
* Retrieval diagnostics
* Experiment visualization

---

## System Architecture

```text
                         ┌──────────────────────┐
                         │      User Query      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Streamlit UI      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    AnswerService     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────┐
                  │        Retrieval Layer         │
                  │                                │
                  │  Dense Semantic Retrieval      │
                  │              +                 │
                  │       BM25 Retrieval           │
                  │              ↓                 │
                  │       Score Fusion             │
                  │              ↓                 │
                  │      Optional Reranking        │
                  └───────────────┬────────────────┘
                                  │
                                  ▼
                         ┌──────────────────────┐
                         │  Retrieved Context   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Ollama Cloud      │
                         │         LLM          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Grounded Answer +    │
                         │ Application Sources  │
                         └──────────────────────┘
```

### Document Indexing Architecture

```text
┌─────────────────┐
│ PDF / DOCX / TXT│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document Loader │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Cleaning   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Chunking        │
│ 500 / 100       │
└────────┬────────┘
         │
         ▼
┌────────────────────────────┐
│ Sentence Transformer       │
│ all-MiniLM-L6-v2           │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ ChromaDB                   │
│ Embeddings + Metadata      │
└────────────────────────────┘
```

---

## RAG Pipeline

DocMind follows this complete RAG pipeline:

```text
Documents
    ↓
Document Loader
    ↓
Text Cleaning
    ↓
Chunking
    ↓
Local Embeddings
    ↓
ChromaDB
    ↓
Dense Retrieval
    +
BM25 Retrieval
    ↓
Hybrid Score Fusion
    ↓
Optional Cross-Encoder Reranking
    ↓
Top-K Context
    ↓
Grounded Prompt
    ↓
Ollama Cloud
    ↓
Generated Answer
    +
Sources / Abstention
```

This architecture separates retrieval responsibilities from language generation responsibilities.

---

## Technology Stack

| Component            | Technology                               |
| -------------------- | ---------------------------------------- |
| Programming Language | Python                                   |
| User Interface       | Streamlit                                |
| Vector Database      | ChromaDB                                 |
| Embedding Framework  | Sentence Transformers                    |
| Embedding Model      | `sentence-transformers/all-MiniLM-L6-v2` |
| Lexical Retrieval    | BM25                                     |
| Reranking            | Cross-Encoder                            |
| LLM                  | Ollama Cloud                             |
| PDF Processing       | pypdf                                    |
| DOCX Processing      | python-docx                              |
| Data Processing      | pandas                                   |
| HTTP Client          | requests                                 |
| Configuration        | python-dotenv                            |
| Testing              | pytest                                   |

---

## Project Structure

```text
docmind-rag/
│
├── app/
│   ├── analytics/
│   │   ├── answer_quality.py
│   │   └── evaluation_data.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_service.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── answer_evaluation_dataset.py
│   │   ├── answer_evaluator.py
│   │   ├── chunking_experiment.py
│   │   ├── context_relevance.py
│   │   ├── dataset.py
│   │   ├── evaluator.py
│   │   ├── hybrid_reranker_evaluation.py
│   │   ├── hybrid_retrieval_experiment.py
│   │   ├── hybrid_weight_experiment.py
│   │   ├── normal_vs_rag.py
│   │   ├── reranker_experiment.py
│   │   ├── reranker_metrics_experiment.py
│   │   ├── reranker_question_comparison.py
│   │   └── retrieval_improvement_experiment.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── answer_service.py
│   │   ├── llm.py
│   │   └── prompts.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   └── loader.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py
│   │   └── retriever.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── create_knowledge_base.py
│   │   ├── document_manager.py
│   │   └── rebuild_knowledge_base.py
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── chroma_store.py
│   │
│   └── main.py
│
├── data/
│   ├── documents/
│   └── chroma/
│
├── docs/
│   ├── 01_rag_fundamentals.md
│   ├── 02_chunking_experiments.md
│   ├── 04_retrieval_evaluation.md
│   ├── 05_prompt_injection_security.md
│   ├── answer_evaluation_results.csv
│   ├── chunking_experiment_results.csv
│   ├── hybrid_retrieval_experiment.csv
│   ├── hybrid_weight_experiment.csv
│   ├── normal_vs_rag_results.csv
│   ├── reranker_metrics_experiment.csv
│   └── retrieval_distance_experiment.csv
│
├── knowledge_base/
│   ├── company/
│   ├── hr/
│   ├── products/
│   ├── security/
│   └── technical/
│
├── tests/
│   ├── test_chroma.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_generation.py
│   ├── test_ingestion.py
│   └── test_retrieval.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

> `data/documents/` and `data/chroma/` are local runtime directories and are excluded from Git through `.gitignore`.

---

## Knowledge Base

The repository contains a realistic, non-confidential demonstration knowledge base.

The knowledge base is organized into:

```text
knowledge_base/
├── company/
├── hr/
├── products/
├── security/
└── technical/
```

Example information includes:

* Company information
* Company policies
* HR and attendance policies
* Employee handbook information
* Product FAQs
* Product documentation
* Installation instructions
* Configuration documentation
* Troubleshooting information
* Security testing material

Supported document types:

```text
.pdf
.docx
.txt
```

Additional documents can be uploaded through the Streamlit interface.

---

## Installation

### 1. Get the Repository

Clone or download the repository and open the project directory in a terminal.

```bash
cd docmind-rag
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
```

Git Bash:

```bash
source .venv/Scripts/activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the project root.

The repository includes `.env.example` as a safe configuration template.

Example:

```env
OLLAMA_API_KEY=your_ollama_cloud_api_key
OLLAMA_MODEL=gpt-oss:120b-cloud
```

### Environment Variables

| Variable         | Required | Description                         |
| ---------------- | -------- | ----------------------------------- |
| `OLLAMA_API_KEY` | Yes      | API key used to access Ollama Cloud |
| `OLLAMA_MODEL`   | No       | Ollama Cloud model name             |

If `OLLAMA_MODEL` is not provided, the application uses:

```text
gpt-oss:120b-cloud
```

The API key is never hard-coded in the application.

The `.env` file is excluded from Git through `.gitignore`.

**Never commit real API keys or other secrets to the repository.**

---

## Running the Application

Activate the virtual environment and run:

```bash
streamlit run app/main.py
```

The Streamlit interface provides access to the main DocMind functionality.

### Main Application Areas

* RAG question answering
* Document management
* Upload and indexing
* Re-indexing
* Document deletion
* Retrieval configuration
* Evaluation dashboards
* Retrieval diagnostics
* Knowledge-base analytics
* Security analysis

---

## Document Ingestion and Indexing

DocMind supports:

```text
PDF
DOCX
TXT
```

The indexing process is:

```text
Upload
   ↓
Validate File
   ↓
Save Document
   ↓
Load Text
   ↓
Clean Text
   ↓
Create Chunks
   ↓
Generate Embeddings
   ↓
Store in ChromaDB
```

### Production Chunk Configuration

The current indexing configuration uses:

```text
Chunk size:     500 characters
Chunk overlap:  100 characters
```

The underlying chunking implementation remains configurable and is also evaluated through dedicated experiments.

### Metadata

Each indexed chunk can retain information such as:

* Source path
* Filename
* File type
* Document type
* Page
* Category
* Chunk number
* Chunk size
* Chunk overlap

---

## Retrieval System

DocMind uses multiple retrieval strategies.

### Dense Semantic Retrieval

The question is converted into a semantic embedding using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model generates a 384-dimensional normalized embedding.

The embedding is compared against document chunk embeddings stored in ChromaDB.

### BM25 Retrieval

DocMind also uses BM25 lexical retrieval.

BM25 is useful for matching important exact terms, product names, identifiers, and domain-specific terminology.

### Hybrid Retrieval

Dense and BM25 results are combined through weighted score fusion.

Conceptually:

```text
Dense Retrieval
       +
BM25 Retrieval
       ↓
Score Normalization
       ↓
Weighted Fusion
       ↓
Combined Candidate Ranking
```

The project includes experiments that study the effect of different Dense/BM25 weights.

The selected weights are treated as dataset-specific experimental configurations rather than universal values.

---

## Reranking

After candidate retrieval, DocMind can apply a cross-encoder reranker.

Current reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The retrieval process becomes:

```text
User Query
    ↓
Dense Retrieval
    +
BM25 Retrieval
    ↓
Hybrid Candidate Fusion
    ↓
Cross-Encoder Reranking
    ↓
Top-K Results
```

Reranking is optional and its behavior is evaluated through recorded experiments.

---

## Grounded Answer Generation

DocMind uses Ollama Cloud for final answer generation.

The current default model is:

```text
gpt-oss:120b-cloud
```

The language model receives:

* System instructions
* User question
* Retrieved document context

The system instructions require the model to:

* Answer using the retrieved context
* Avoid unsupported outside knowledge
* Avoid invented facts
* Avoid invented citations
* Treat retrieved documents as untrusted data
* Ignore instructions contained inside retrieved documents
* Abstain when the available evidence is insufficient

The application itself remains responsible for source metadata.

---

## Citations and Abstention

DocMind does not rely on the language model to invent source references.

Source information is constructed by the application from retrieval metadata.

Sources can contain:

* Filename
* Category
* Page
* Chunk
* Source path

### Abstention

When the system cannot find sufficient supporting information, it can return:

```text
I don't know based on the available documents.
```

This provides an explicit boundary between supported knowledge and unsupported questions.

---

## Evaluation Framework

DocMind includes a dedicated evaluation framework to measure retrieval and answer behavior.

### Chunking Experiments

The project evaluates different chunk sizes and overlaps.

Results:

```text
docs/chunking_experiment_results.csv
```

### Hybrid Retrieval Experiments

Results:

```text
docs/hybrid_retrieval_experiment.csv
```

### Hybrid Weight Experiments

Results:

```text
docs/hybrid_weight_experiment.csv
```

### Reranking Experiments

Results:

```text
docs/reranker_metrics_experiment.csv
```

### Retrieval Distance Experiments

Results:

```text
docs/retrieval_distance_experiment.csv
```

### Answer Evaluation

Results:

```text
docs/answer_evaluation_results.csv
```

### Normal LLM vs RAG

Results:

```text
docs/normal_vs_rag_results.csv
```

The Streamlit application includes dashboards for inspecting these recorded experiments.

---

## Normal LLM vs RAG Evaluation

The project includes a controlled comparison between normal LLM answering and RAG-based answering.

The same evaluation dataset is used for both approaches.

### Dataset

```text
30 questions
3 difficulty levels
4 categories
```

The recorded evaluation includes measurements such as:

* Normal LLM generation latency
* RAG retrieval latency
* RAG context processing latency
* RAG generation latency
* RAG end-to-end latency
* Evidence-support proxy
* RAG groundedness proxy
* Context relevance
* Citation availability
* Expected-source retrieval
* Abstention behavior
* Errors

### Recorded Results

The current experiment recorded:

| Metric                         | Recorded Value |
| ------------------------------ | -------------: |
| Evaluation questions           |             30 |
| Successful comparisons         |             30 |
| Expected source found by RAG   |           100% |
| RAG citation availability      |           100% |
| Normal citation availability   |             0% |
| Average normal generation time |         5.87 s |
| Average RAG retrieval time     |      435.89 ms |
| Average RAG generation time    |         1.72 s |
| Average RAG end-to-end time    |         2.15 s |
| Normal evidence-support proxy  |          0.035 |
| RAG groundedness proxy         |          0.851 |
| RAG context relevance          |          0.880 |

These results describe the recorded experiment, dataset, model, and implementation configuration.

The evidence-support and groundedness values are automated evaluation proxies rather than human-annotated truth scores. They should therefore not be interpreted as universal benchmarks.

---

## Security and Prompt Injection

DocMind includes a dedicated prompt-injection security experiment.

Retrieved documents are explicitly treated as:

```text
UNTRUSTED DATA
```

rather than instructions.

The system prompt instructs the LLM to ignore document content that attempts to:

* Override system instructions
* Reveal hidden prompts
* Reveal credentials
* Reveal API keys
* Reveal secrets
* Reveal internal instructions
* Change the assistant's intended behavior

The security experiment is documented in:

```text
docs/05_prompt_injection_security.md
```

A dedicated security test document is also included in:

```text
knowledge_base/security/
```

---

## Testing

The repository contains automated tests covering:

* ChromaDB
* Chunking
* Embeddings
* Generation
* Ingestion
* Retrieval

Run the test suite:

```bash
pytest -q
```

The current project verification has passed the available automated test suite.

---

## Current Evaluation Results

The Streamlit evaluation dashboard provides analysis of recorded project experiments.

The dashboard includes:

* Evaluation overview
* Retrieval method comparison
* Retrieval quality analysis
* Difficulty analysis
* Category analysis
* Reranker analysis
* Hybrid retrieval weight analysis
* Retrieval distance analysis
* Chunking experiment analysis
* Evaluation dataset coverage
* Knowledge-base analytics
* Failure analysis
* Latency analysis
* Normal LLM vs RAG comparison
* Security analysis

Recorded evaluation data is loaded from the project's CSV artifacts where applicable.

The dashboard does not need to rerun expensive experiments simply to display their recorded results.

---

## Limitations

### Character-Based Chunking

The current production configuration uses character-based chunking.

Semantic or structure-aware chunking could be explored in future work.

### Evaluation Proxies

Some answer-quality metrics are automated proxies.

Human evaluation would provide a stronger assessment of factual correctness and answer quality.

### Dataset Size

The current evaluation dataset contains 30 questions for the normal-vs-RAG comparison.

A larger and more diverse evaluation set would provide stronger evidence across additional domains and question types.

### Knowledge Base Scope

The current knowledge base is a controlled demonstration dataset.

Real-world performance can differ for:

* Very large document collections
* Noisy documents
* OCR-heavy documents
* Tables
* Images
* Multilingual documents
* Highly structured technical documents

### Vector Database

The project currently uses a local persistent ChromaDB instance.

A production multi-user deployment may require a server-based or managed vector database.

### LLM Dependency

Final answer generation depends on access to the configured Ollama Cloud API and a valid API key.

---

## Future Improvements

Possible future improvements include:

* Semantic or structure-aware chunking
* OCR support for scanned PDFs
* Table-aware document extraction
* Multilingual embeddings
* Larger human-annotated evaluation datasets
* Automated evaluation in CI/CD
* Authentication
* Multi-user access control
* Document versioning
* Permission-aware retrieval
* Production vector database deployment
* Streaming responses
* Additional retrieval strategies
* Larger-scale performance testing
* Improved observability and monitoring

---

## Internship Learning Outcomes

This project demonstrates practical experience in the following areas.

### RAG Engineering

* Retrieval-Augmented Generation
* Grounded question answering
* Knowledge-base construction
* Retrieval/generation separation

### Document Processing

* PDF extraction
* DOCX extraction
* TXT processing
* Text cleaning
* Chunking
* Metadata preservation

### Embeddings and Vector Databases

* Sentence Transformer embeddings
* Vector representations
* Normalized embeddings
* ChromaDB
* Semantic similarity search

### Information Retrieval

* Dense retrieval
* BM25
* Hybrid retrieval
* Score normalization
* Weighted retrieval
* Cross-encoder reranking

### LLM Engineering

* Prompt engineering
* Grounded context construction
* Abstention
* Citation architecture
* Ollama Cloud API integration

### Evaluation

* Chunking experiments
* Retrieval evaluation
* Hybrid retrieval evaluation
* Reranking evaluation
* Answer-quality evaluation
* Latency analysis
* Normal LLM vs RAG comparison

### AI Security

* Prompt-injection testing
* Untrusted retrieved content
* Instruction isolation
* Security experiment design

### Software Engineering

* Modular Python architecture
* Persistent storage
* Configuration management
* Error handling
* Automated testing
* Streamlit application development
* Reproducible evaluation artifacts

---

## Project Status

DocMind currently provides an end-to-end working RAG knowledge assistant with:

```text
PDF / DOCX / TXT ingestion
          ↓
Text cleaning
          ↓
Configurable chunking
          ↓
Local embeddings
          ↓
Persistent ChromaDB
          ↓
Dense retrieval
          +
BM25 retrieval
          ↓
Hybrid search
          ↓
Cross-encoder reranking
          ↓
Grounded Ollama Cloud generation
          ↓
Citations
          +
Abstention
          ↓
Evaluation
          ↓
Security testing
          ↓
Streamlit application
          ↓
Automated testing
```

The project is structured as an internship-ready demonstration of the complete RAG development lifecycle.

---

## Author

**Abdul Moiz**

**DocMind — RAG Knowledge Assistant**
