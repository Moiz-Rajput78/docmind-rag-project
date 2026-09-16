# DocMind — RAG Knowledge Assistant

## Final Technical Report

**Project:** DocMind — RAG Knowledge Assistant
**Project Type:** AI / Machine Learning / Retrieval-Augmented Generation
**Application:** Knowledge-base question answering system
**Primary Interface:** Streamlit
**Vector Database:** ChromaDB
**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
**LLM Provider:** Ollama Cloud
**Retrieval:** Dense + BM25 Hybrid Retrieval
**Reranking:** Cross-Encoder
**Evaluation Dataset:** 30-question Normal LLM vs RAG evaluation
**Status:** Completed

---

# 1. Executive Summary

DocMind is a Retrieval-Augmented Generation (RAG) knowledge assistant designed to answer questions using a controlled collection of organizational documents.

The system allows users to upload PDF, DOCX, and TXT documents, process them into searchable chunks, generate local semantic embeddings, store those embeddings in ChromaDB, retrieve relevant information using semantic and lexical retrieval, optionally rerank retrieved candidates, and generate grounded answers through an Ollama Cloud language model.

A central design principle of DocMind is that the language model should answer from retrieved evidence rather than relying on unsupported external knowledge or assumptions.

The system also provides source-aware citations, explicit abstention when relevant information cannot be found, prompt-injection protection, document management, retrieval experiments, evaluation dashboards, and automated tests.

The completed implementation covers the major requirements of a practical RAG system, including ingestion, chunking, embeddings, vector storage, retrieval, hybrid search, reranking, grounded generation, evaluation, security testing, and an interactive user interface.

---

# 2. Problem Statement

Traditional language models can generate fluent answers without having access to an organization's private or specialized documents. This creates several problems:

* The model may not know organization-specific information.
* The model may provide outdated information.
* The model can generate unsupported statements.
* Users cannot easily verify where an answer came from.
* Domain-specific terminology may not be represented correctly.
* The model may answer confidently even when the required information is unavailable.

DocMind addresses these problems by introducing a retrieval layer between the user's question and the language model.

Instead of asking the language model to answer entirely from its internal knowledge, DocMind first searches the organization's indexed knowledge base and provides relevant document sections as context.

The generation process can therefore be summarized as:

```text
User Question
      ↓
Query Embedding / Retrieval
      ↓
Relevant Knowledge Chunks
      ↓
Optional Reranking
      ↓
Grounded Context
      ↓
LLM
      ↓
Answer + Application-Owned Citations
```

---

# 3. Project Objectives

The main objectives of DocMind were:

1. Implement the complete RAG pipeline.
2. Support PDF, DOCX, and TXT documents.
3. Preserve useful document metadata.
4. Implement configurable text chunking.
5. Generate local semantic embeddings.
6. Store embeddings in a persistent vector database.
7. Implement semantic retrieval.
8. Implement lexical retrieval using BM25.
9. Combine dense and lexical retrieval using hybrid search.
10. Improve retrieval using cross-encoder reranking.
11. Generate grounded answers using retrieved context.
12. Provide source and page information with answers.
13. Abstain when the knowledge base does not contain sufficient evidence.
14. Protect the generation layer against prompt injection from documents.
15. Evaluate retrieval and answer quality.
16. Compare normal LLM answering with RAG-based answering.
17. Provide an interactive Streamlit interface.
18. Provide automated tests.
19. Document limitations and experimental results.
20. Produce a reproducible project suitable for technical evaluation.

---

# 4. RAG Fundamentals

Retrieval-Augmented Generation combines information retrieval with language generation.

A conventional language model follows approximately:

```text
Question
   ↓
Language Model
   ↓
Answer
```

A RAG system introduces a knowledge retrieval stage:

```text
Question
   ↓
Retriever
   ↓
Relevant Documents
   ↓
Context
   ↓
Language Model
   ↓
Grounded Answer
```

This architecture provides several advantages.

## 4.1 Knowledge Grounding

The model receives evidence from the project's knowledge base.

## 4.2 Domain Adaptation

The system can answer questions about documents that were not part of the model's original training data.

## 4.3 Source Traceability

The application can associate generated answers with the retrieved document chunks.

## 4.4 Reduced Unsupported Generation

The prompt explicitly instructs the model not to invent facts that are not supported by the supplied context.

## 4.5 Knowledge Base Updates

Documents can be uploaded, indexed, reindexed, and deleted without retraining the language model.

---

# 5. System Architecture

The overall DocMind architecture is:

```text
                         ┌──────────────────────┐
                         │      User / UI       │
                         │     Streamlit        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Main Application  │
                         │      app/main.py      │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
             ┌───────────────┐             ┌───────────────┐
             │ Document Mgmt │             │ Answer Service│
             └───────┬───────┘             └───────┬───────┘
                     │                             │
                     ▼                             ▼
             ┌───────────────┐             ┌───────────────┐
             │ PDF/DOCX/TXT  │             │   Retrieval   │
             │    Loader     │             └───────┬───────┘
             └───────┬───────┘                     │
                     ▼                             ▼
             ┌───────────────┐             ┌─────────────────┐
             │    Chunker    │             │ Hybrid Retriever│
             └───────┬───────┘             └────────┬────────┘
                     │                              │
                     ▼                    ┌─────────┴─────────┐
             ┌───────────────┐             │                   │
             │   Embeddings  │             ▼                   ▼
             │ SentenceTrans │          Dense                BM25
             └───────┬───────┘             │                   │
                     │                     └────────┬──────────┘
                     ▼                              │
             ┌───────────────┐                      ▼
             │   ChromaDB    │             ┌─────────────────┐
             │ Vector Store  │             │ Cross-Encoder   │
             └───────────────┘             │    Reranker     │
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │ Grounded Prompt │
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │  Ollama Cloud   │
                                            │       LLM       │
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │ Answer + Sources│
                                            └─────────────────┘
```

---

# 6. Technology Stack

| Component                 | Technology                    |
| ------------------------- | ----------------------------- |
| Programming Language      | Python                        |
| UI                        | Streamlit                     |
| Document Processing       | PyPDF / python-docx           |
| Embeddings                | Sentence Transformers         |
| Embedding Model           | `all-MiniLM-L6-v2`            |
| Vector Database           | ChromaDB                      |
| Lexical Retrieval         | BM25                          |
| Reranking                 | Cross-Encoder                 |
| LLM                       | Ollama Cloud                  |
| Environment Configuration | `.env`                        |
| Testing                   | pytest                        |
| Data Analysis             | pandas                        |
| Visualization             | matplotlib / Streamlit charts |
| Documentation             | Markdown                      |

---

# 7. Project Structure

The main repository is organized into separate modules for ingestion, embeddings, retrieval, generation, evaluation, analytics, and utilities.

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
│   ├── chroma/
│   └── documents/
│
├── docs/
│   ├── 01_rag_fundamentals.md
│   ├── 02_chunking_experiments.md
│   ├── 04_retrieval_evaluation.md
│   ├── 05_prompt_injection_security.md
│   ├── answer_evaluation_results.csv
│   ├── normal_vs_rag_results.csv
│   ├── chunking_experiment_results.csv
│   ├── hybrid_retrieval_experiment.csv
│   ├── hybrid_weight_experiment.csv
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

---

# 8. Knowledge Base

The project uses a realistic, non-confidential organizational knowledge base.

The knowledge base contains information covering areas such as:

* company information
* company policies
* HR policies
* employee handbook information
* products
* product FAQs
* technical configuration
* installation
* troubleshooting
* security testing

Example files include:

```text
knowledge_base/company/company_overview.pdf
knowledge_base/company/policies.pdf
knowledge_base/company/services.docx

knowledge_base/hr/attendance_policy.docx
knowledge_base/hr/employee_handbook.pdf
knowledge_base/hr/leave_policy.pdf

knowledge_base/products/faq.txt
knowledge_base/products/product_a.pdf
knowledge_base/products/product_b.txt

knowledge_base/security/prompt_injection_test.txt

knowledge_base/technical/configuration.docx
knowledge_base/technical/installation.pdf
knowledge_base/technical/troubleshooting.txt
```

This provides a multi-domain environment for testing retrieval and answer generation.

---

# 9. Document Ingestion

DocMind supports three primary document formats:

```text
PDF
DOCX
TXT
```

The ingestion system automatically determines the document type based on its file extension.

## 9.1 PDF Processing

PDF files are processed page by page.

The loader extracts:

* text
* filename
* source path
* file type
* document type
* page number

Page-level metadata is particularly useful for citation and source traceability.

## 9.2 DOCX Processing

DOCX documents are processed using document paragraphs.

The extracted information is associated with document metadata.

## 9.3 TXT Processing

TXT files are loaded using UTF-8 encoding.

## 9.4 Unsupported Formats

Unsupported file extensions are rejected instead of being silently processed.

This makes the ingestion behavior explicit and predictable.

---

# 10. Text Cleaning

Before chunking, extracted document text is normalized.

The cleaning process handles:

* inconsistent line endings
* trailing spaces
* excessive blank lines
* unnecessary repeated whitespace

This improves consistency before the text is divided into retrieval chunks.

---

# 11. Chunking Strategy

Large documents cannot normally be passed to retrieval and generation as one large block.

DocMind therefore divides documents into smaller chunks.

The production document manager currently uses:

```text
Chunk Size: 500 characters
Overlap: 100 characters
```

The underlying chunking function remains configurable.

Conceptually:

```text
Document
│
├── Chunk 1 ───────────────┐
│                          │
├── Chunk 2 ───────────────┤ overlap
│                          │
├── Chunk 3 ───────────────┤
│                          │
└── ...
```

The overlap helps preserve information that crosses chunk boundaries.

---

# 12. Chunking Experiments

DocMind includes a dedicated chunking experiment module.

The experiment framework allows different chunk configurations to be evaluated rather than assuming that one configuration is universally optimal.

The experiment results are stored in:

```text
docs/chunking_experiment_results.csv
```

The project also documents the experimental methodology in:

```text
docs/02_chunking_experiments.md
```

This provides an evidence-based approach to selecting retrieval configurations.

---

# 13. Metadata Preservation

Metadata is preserved throughout the ingestion and retrieval pipeline.

Important metadata fields include:

```text
source
filename
file_type
document_type
page
category
chunk_number
chunk_size
chunk_overlap
```

This metadata allows the application to associate retrieved text with its original document.

It also supports:

* citations
* filtering
* source analysis
* document management
* evaluation
* debugging
* retrieval diagnostics

---

# 14. Embedding System

DocMind uses a local Sentence Transformers model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding service converts text into numerical vectors.

The model produces:

```text
384-dimensional embeddings
```

Embeddings are normalized before being stored and queried.

Conceptually:

```text
Text
 ↓
Sentence Transformer
 ↓
384-dimensional vector
 ↓
ChromaDB
```

The same embedding service can be shared across retrieval components to avoid unnecessarily creating multiple model instances.

---

# 15. Vector Database

DocMind uses ChromaDB as its persistent vector database.

The vector store maintains a collection named:

```text
docmind_documents
```

The database stores:

* document chunks
* embeddings
* chunk IDs
* source metadata
* page information
* category information
* chunk configuration

The vector database is persisted locally under:

```text
data/chroma/
```

Local database files are excluded from Git using `.gitignore`.

---

# 16. Deterministic Chunk IDs

Chunks are assigned deterministic identifiers based on their source and position.

The ID structure follows the concept:

```text
source::page_{page}::chunk_{chunk_number}
```

Deterministic IDs help prevent unnecessary duplication when documents are indexed again.

They also support precise deletion and reindexing.

---

# 17. Semantic Retrieval

The basic retriever performs semantic search.

The process is:

```text
User Question
      ↓
Query Embedding
      ↓
ChromaDB Similarity Search
      ↓
Ranked Chunks
```

The retriever supports configurable `top_k` retrieval.

Each retrieved result contains information such as:

* chunk ID
* text
* distance
* metadata
* source
* filename
* file type
* document type
* page
* category
* chunk number

---

# 18. Hybrid Retrieval

Pure semantic search can miss exact terminology, identifiers, acronyms, and keyword-heavy queries.

DocMind therefore implements hybrid retrieval.

The hybrid retriever combines:

```text
Dense Semantic Retrieval
+
BM25 Lexical Retrieval
```

The general architecture is:

```text
                    Query
                      │
             ┌────────┴────────┐
             ▼                 ▼
        Dense Search        BM25 Search
             │                 │
             └────────┬────────┘
                      ▼
                 Score Fusion
                      │
                      ▼
               Candidate Ranking
```

Dense retrieval captures semantic similarity.

BM25 captures lexical similarity.

Combining the two provides complementary retrieval signals.

---

# 19. Hybrid Retrieval Weighting

The production hybrid retriever uses a dense/BM25 weighting configuration based on the project's experiments.

The implementation treats these weights as **dataset-specific experimental settings**, not as a universal optimal configuration.

This distinction is important because retrieval performance can change depending on:

* document collection
* question distribution
* vocabulary
* chunk size
* metadata
* retrieval model
* evaluation methodology

The project includes dedicated hybrid-weight experiments stored in:

```text
docs/hybrid_weight_experiment.csv
```

---

# 20. BM25 Retrieval

BM25 is used as the lexical retrieval component.

It is useful when a question contains exact terms that should be matched directly.

For example, keyword retrieval can help with:

* product names
* configuration keys
* policy terminology
* exact technical terms
* abbreviations
* specific identifiers

DocMind tokenizes indexed text and builds a BM25 retrieval representation.

---

# 21. Cross-Encoder Reranking

After hybrid retrieval produces candidates, DocMind can optionally apply a cross-encoder reranker.

The configured reranking model is:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The retrieval architecture becomes:

```text
Query
 ↓
Dense Retrieval ─┐
                  ├── Hybrid Fusion
BM25 Retrieval ───┘
                       ↓
                 Candidate Set
                       ↓
                Cross-Encoder
                   Reranking
                       ↓
                   Top-K
```

The reranker examines the query and candidate passage together and produces a more targeted relevance score.

This allows the system to distinguish between candidates that may have similar retrieval scores but different query relevance.

---

# 22. Retrieval Experiments

The project contains multiple retrieval evaluation components.

These include experiments for:

* hybrid retrieval
* hybrid weights
* reranking
* reranker metrics
* retrieval distance thresholds
* retrieval improvements
* question-level comparisons

The corresponding result artifacts are stored in the `docs/` directory.

This makes the retrieval system experimentally measurable instead of relying only on qualitative demonstrations.

---

# 23. Retrieval Distance Analysis

DocMind includes analysis of vector retrieval distance and threshold behavior.

The purpose is to understand how retrieval confidence changes as the similarity/distance threshold changes.

The results are stored in:

```text
docs/retrieval_distance_experiment.csv
```

This experiment helps investigate retrieval quality and the behavior of weakly related candidates.

---

# 24. Grounded Answer Generation

Once relevant chunks have been retrieved, the answer service builds a structured context for the language model.

The context includes information such as:

```text
Filename
Category
Page
Chunk
Retrieved Text
```

The model is then instructed to answer only from the supplied context.

The generation pipeline is:

```text
Question
   ↓
Retrieve
   ↓
Build Context
   ↓
Grounded Prompt
   ↓
Ollama Cloud
   ↓
Normalize Answer
   ↓
Attach Application-Owned Sources
```

---

# 25. LLM Integration

DocMind uses Ollama Cloud rather than requiring a local Ollama server.

The configuration is stored in environment variables.

The `.env.example` file contains the required configuration structure:

```text
OLLAMA_API_KEY=your_ollama_cloud_api_key
OLLAMA_MODEL=gpt-oss:120b-cloud
```

The actual API key is intentionally not included in the repository.

The application loads the API key from the environment and uses authenticated requests to the Ollama Cloud API.

---

# 26. Prompt Design

The generation prompt contains several important safeguards.

The model is instructed to:

* answer only from retrieved context
* avoid outside knowledge
* avoid unsupported assumptions
* avoid inventing facts
* avoid inventing names
* avoid inventing numbers
* avoid inventing dates
* combine relevant chunks when necessary
* provide partial answers when only part of the question is supported
* abstain when the context does not contain sufficient information

The application itself owns source citations.

The language model is not responsible for generating source numbers, page citations, or filenames.

This separation reduces the risk of fabricated citation references.

---

# 27. Abstention Behavior

A critical RAG feature is knowing when the system does not have enough evidence.

DocMind uses the explicit abstention message:

```text
I don't know based on the available documents.
```

If retrieval produces no usable evidence, the system can abstain instead of forcing an answer.

This is important because a useful knowledge assistant should distinguish between:

```text
"I found evidence."
```

and:

```text
"I do not have evidence."
```

---

# 28. Citation Architecture

DocMind builds source information from retrieval results rather than trusting the generated answer.

This provides a clear separation:

```text
Retriever
   ↓
Authoritative Source Metadata
   ↓
Application
   ↓
Citation Display
```

The LLM generates the answer.

The application determines the associated sources.

This design makes citations more reliable and auditable.

---

# 29. Document Management

The document manager provides lifecycle operations for knowledge-base documents.

Supported operations include:

```text
Upload
Index
List
View Metadata
Reindex
Reindex All
Delete
```

When a document is deleted, its corresponding vector entries are removed from ChromaDB and the physical document is removed from the managed document directory.

Reindexing removes the existing vectors associated with the source before indexing the document again.

---

# 30. Reindexing

Reindexing is useful when:

* a document has changed
* chunking configuration changes
* embeddings need to be regenerated
* the indexed content becomes inconsistent with the physical document

The reindex workflow is:

```text
Existing Document
       ↓
Remove Existing Vectors
       ↓
Load Document
       ↓
Clean Text
       ↓
Chunk
       ↓
Generate Embeddings
       ↓
Store New Vectors
```

The physical document is preserved if reindexing fails.

---

# 31. Security and Prompt-Injection Protection

Retrieved documents must be treated as untrusted data.

A malicious document could contain text such as instructions attempting to manipulate the model.

DocMind explicitly addresses this through its generation prompt.

The model is instructed that retrieved documents are data, not instructions.

The system is designed to ignore document content that attempts to:

* reveal system prompts
* reveal hidden instructions
* expose secrets
* expose credentials
* override system behavior
* change the application's instructions

The security test document is:

```text
knowledge_base/security/prompt_injection_test.txt
```

The corresponding security documentation is:

```text
docs/05_prompt_injection_security.md
```

---

# 32. Security Design Principle

The key security principle is:

```text
Retrieved Text ≠ Trusted Instructions
```

The RAG pipeline therefore distinguishes between:

```text
Application Instructions
        ↓
Trusted

Retrieved Documents
        ↓
Untrusted Data
```

This is especially important in systems where users can upload their own documents.

---

# 33. Evaluation Framework

DocMind includes a dedicated evaluation framework rather than relying only on manual testing.

Evaluation areas include:

* retrieval behavior
* chunking behavior
* hybrid retrieval
* retrieval weighting
* reranking
* distance thresholds
* answer quality
* context relevance
* normal LLM vs RAG behavior
* latency
* knowledge-base coverage
* security behavior

The evaluation artifacts are stored under:

```text
docs/
```

---

# 34. Answer Evaluation Dataset

The project includes a dedicated answer evaluation dataset containing:

```text
30 questions
```

The evaluation covers different:

* difficulty levels
* question categories
* expected source documents

The answer evaluation system records information related to answer behavior and retrieval support.

---

# 35. Normal LLM vs RAG Evaluation

One of the major experiments compares two approaches:

```text
Normal LLM
vs
RAG-Enhanced LLM
```

The comparison uses the same evaluation questions.

The evaluation dataset contains:

```text
30 questions
```

The recorded fields include:

* question
* difficulty
* category
* expected sources
* normal answer
* RAG answer
* normal generation latency
* RAG retrieval latency
* RAG context processing latency
* RAG generation latency
* RAG end-to-end latency
* evidence-support proxy
* groundedness proxy
* context relevance
* citation availability
* expected source found
* abstention behavior
* errors

The complete result file is:

```text
docs/normal_vs_rag_results.csv
```

---

# 36. Normal vs RAG Evaluation Results

The completed evaluation produced the following measurements.

| Metric                           | Recorded Result |
| -------------------------------- | --------------: |
| Evaluation Questions             |              30 |
| Successful Comparisons           |              30 |
| Expected Source Found by RAG     |            100% |
| RAG Citation Availability        |            100% |
| Normal LLM Citation Availability |              0% |
| Average Normal LLM Generation    |          5.87 s |
| Average RAG Retrieval            |       435.89 ms |
| Average RAG Generation           |          1.72 s |
| Average RAG End-to-End           |          2.15 s |
| Normal Evidence-Support Proxy    |           0.035 |
| RAG Groundedness Proxy           |           0.851 |
| RAG Context Relevance            |           0.880 |

These measurements describe the behavior of this particular implementation and evaluation dataset.

They should not be interpreted as universal benchmarks for all LLMs, RAG systems, or datasets.

---

# 37. Evaluation Metric Interpretation

## 37.1 Expected Source Found

This measures whether the expected source document was retrieved for an evaluation question.

The recorded result was:

```text
100%
```

for the evaluated questions.

## 37.2 Citation Availability

The application recorded:

```text
Normal LLM: 0%
RAG: 100%
```

This reflects the design difference between the two systems.

The normal LLM does not have access to the project's retrieval metadata, while the RAG application attaches sources from retrieved chunks.

## 37.3 Groundedness Proxy

The recorded RAG groundedness proxy was:

```text
0.851
```

This is an evaluation proxy rather than a human-annotated factual-truth score.

## 37.4 Context Relevance

The recorded RAG context relevance value was:

```text
0.880
```

This represents the measured relevance of retrieved context under the project's evaluation methodology.

## 37.5 Evidence-Support Proxy

The normal LLM evidence-support proxy was:

```text
0.035
```

This metric should not be interpreted as a direct factual-accuracy measurement.

---

# 38. Evaluation Limitations

The evaluation metrics have important limitations.

The project explicitly treats proxy metrics as measurements of system behavior rather than perfect measurements of truth.

In particular:

* automated proxies are not equivalent to expert human evaluation
* the dataset is limited in size
* the knowledge base is project-specific
* retrieval performance depends on document composition
* latency depends on hardware, network conditions, and model availability
* different LLMs can produce different results
* different chunk configurations can change retrieval behavior
* different evaluation datasets can produce different measurements

Future versions could add human-annotated evaluation and larger benchmark datasets.

---

# 39. Latency Analysis

DocMind records separate latency components.

The RAG pipeline can measure:

```text
Retrieval
Context Preparation
Generation
End-to-End
```

The recorded averages from the completed Normal-vs-RAG evaluation were:

```text
RAG Retrieval:       435.89 ms
RAG Generation:        1.72 s
RAG End-to-End:        2.15 s
```

Separating latency components helps determine where processing time is spent.

---

# 40. Knowledge Base Analytics

DocMind also provides read-only knowledge-base analytics.

The analytics can inspect:

* indexed chunk count
* physical document count
* indexed document count
* categories
* file types
* document indexing coverage
* chunks by document
* chunks by category
* page coverage
* chunk configuration
* Chroma metadata completeness
* source/path distribution
* vector-store health

These analytics operate on the existing knowledge base and do not automatically reindex or delete documents.

---

# 41. Streamlit Application

The complete system is integrated into a Streamlit interface.

The interface provides functionality for:

* asking questions
* viewing answers
* viewing sources
* uploading documents
* indexing documents
* reindexing documents
* deleting documents
* viewing knowledge-base information
* viewing evaluation results
* viewing retrieval analytics
* viewing security-related information

The application therefore provides both an end-user experience and a technical evaluation interface.

---

# 42. Application Workflow

A normal user workflow is:

```text
1. Open DocMind
       ↓
2. Upload a PDF/DOCX/TXT
       ↓
3. Document validation
       ↓
4. Save document
       ↓
5. Load and clean
       ↓
6. Chunk document
       ↓
7. Generate embeddings
       ↓
8. Store in ChromaDB
       ↓
9. Ask a question
       ↓
10. Retrieve relevant chunks
       ↓
11. Hybrid retrieval
       ↓
12. Optional reranking
       ↓
13. Build grounded context
       ↓
14. Generate answer
       ↓
15. Display answer + sources
```

---

# 43. Testing

Automated tests were implemented using pytest.

The test suite covers major system components:

```text
tests/test_chroma.py
tests/test_chunking.py
tests/test_embeddings.py
tests/test_generation.py
tests/test_ingestion.py
tests/test_retrieval.py
```

The completed test suite passed successfully.

This provides automated verification for important functionality including:

* document loading
* chunk creation
* embeddings
* vector storage
* retrieval
* answer generation

---

# 44. Test Philosophy

Testing focuses on validating the boundaries between components.

For example:

```text
Document
 ↓
Loader
 ↓
Chunker
 ↓
Embedding Service
 ↓
Vector Store
 ↓
Retriever
 ↓
Generation
```

Testing individual components makes failures easier to isolate.

The project also uses evaluation experiments for higher-level behavior that cannot be adequately tested with simple unit tests.

---

# 45. Configuration and Secret Management

Sensitive credentials are not committed to Git.

The project uses:

```text
.env
```

for local secrets.

The repository contains:

```text
.env.example
```

as a configuration template.

The `.gitignore` file excludes:

```text
.env
.env.*
```

while allowing:

```text
.env.example
```

This prevents API keys from accidentally being committed.

---

# 46. Git Repository Hygiene

The repository excludes local runtime artifacts such as:

* Python caches
* virtual environments
* environment secrets
* local ChromaDB files
* uploaded/generated local documents
* model weights
* logs
* test caches
* IDE-specific files

This keeps the source repository focused on application code, documentation, evaluation artifacts, and project configuration.

---

# 47. Error Handling

DocMind includes validation and error handling at multiple levels.

Examples include:

* invalid file formats
* empty documents
* invalid chunk configurations
* missing API configuration
* failed API requests
* invalid LLM responses
* empty retrieval results
* failed document indexing
* failed reindexing
* invalid document deletion targets

The system avoids silently treating failures as successful operations.

---

# 48. Failure Handling and Abstention

RAG systems should not always attempt to answer.

DocMind distinguishes between:

```text
Retrieved Evidence Available
```

and:

```text
No Sufficient Evidence
```

When useful context is unavailable, the application can return the explicit abstention response:

```text
I don't know based on the available documents.
```

This provides a safer behavior for questions outside the knowledge base.

---

# 49. Experimental Methodology

The project uses experiment-specific CSV files rather than embedding experimental results only inside application code.

Examples include:

```text
chunking_experiment_results.csv
hybrid_retrieval_experiment.csv
hybrid_weight_experiment.csv
reranker_metrics_experiment.csv
retrieval_distance_experiment.csv
answer_evaluation_results.csv
normal_vs_rag_results.csv
```

This makes the experiments:

* reproducible
* inspectable
* exportable
* suitable for visualization
* easier to compare

---

# 50. Retrieval Evaluation Design

Retrieval evaluation investigates how the system behaves under different retrieval configurations.

The experiments cover:

```text
Dense Retrieval
BM25
Hybrid Retrieval
Hybrid Weights
Reranking
Distance Thresholds
Chunking
```

This allows the system to be studied as a retrieval pipeline rather than treating the RAG system as a black box.

---

# 51. Why Hybrid Retrieval Is Useful

Dense and lexical retrieval solve different types of retrieval problems.

Dense retrieval is useful for semantic similarity.

For example:

```text
"What happens if an employee takes annual leave?"
```

may retrieve a passage discussing:

```text
employee vacation / paid time off
```

even when the wording differs.

BM25 is useful for exact terminology.

For example:

```text
"API_TIMEOUT configuration"
```

may benefit from exact lexical matching.

Combining the two gives the system access to both retrieval signals.

---

# 52. Why Reranking Is Useful

Initial retrieval is optimized for quickly finding candidates.

A reranker can then perform a more focused relevance assessment over the candidate set.

Therefore:

```text
Fast Candidate Retrieval
          ↓
More Expensive Relevance Assessment
```

can provide a practical architecture for improving the ordering of retrieved passages.

DocMind evaluates reranking separately rather than assuming that reranking always improves every query.

---

# 53. Reproducibility

The project contains the major components necessary to reproduce the application:

* source code
* requirements
* environment template
* knowledge-base examples
* evaluation datasets/results
* documentation
* tests
* README
* configuration instructions

Local runtime data such as ChromaDB is intentionally excluded from Git.

---

# 54. Current System Strengths

The completed system demonstrates the major components expected from a practical RAG application:

* multi-format document ingestion
* metadata preservation
* configurable chunking
* local semantic embeddings
* persistent vector storage
* semantic retrieval
* BM25 lexical retrieval
* hybrid retrieval
* cross-encoder reranking
* grounded answer generation
* application-owned citations
* explicit abstention
* document lifecycle management
* prompt-injection protection
* automated testing
* retrieval experiments
* answer evaluation
* Normal-vs-RAG comparison
* latency analysis
* Streamlit interface
* technical documentation

---

# 55. Current Limitations

Although the project is complete, several limitations remain.

## 55.1 Evaluation Dataset Size

The main Normal-vs-RAG evaluation contains 30 questions.

A larger benchmark would provide broader coverage.

## 55.2 Proxy Metrics

Some answer-quality measurements are automated proxies rather than human-annotated factual evaluations.

## 55.3 Local Embedding Performance

`all-MiniLM-L6-v2` provides an efficient embedding model, but larger embedding models may perform differently on specialized domains.

## 55.4 Document Parsing

PDF and DOCX extraction can struggle with complex layouts such as:

* tables
* multi-column documents
* scanned PDFs
* embedded images
* complex formatting

## 55.5 BM25 Tokenization

The current lexical retrieval implementation uses a relatively simple tokenizer and could be improved for specialized terminology.

## 55.6 Reranking Cost

Cross-encoder reranking is more computationally expensive than simple vector retrieval.

## 55.7 Cloud LLM Dependency

Answer generation depends on Ollama Cloud availability and network connectivity.

---

# 56. Future Improvements

Potential future work includes:

1. Human-annotated answer evaluation.
2. Larger evaluation datasets.
3. More sophisticated document parsing.
4. OCR support for scanned PDFs.
5. Table-aware document extraction.
6. Semantic or structure-aware chunking.
7. Additional embedding models.
8. Query rewriting.
9. Multi-query retrieval.
10. Metadata-aware retrieval filters.
11. Improved BM25 tokenization.
12. Additional reranking models.
13. Streaming answer generation.
14. Authentication and multi-user support.
15. Role-based document access.
16. Conversation memory with controlled retrieval.
17. Evaluation against additional LLM providers.
18. Production deployment with managed vector storage.
19. Monitoring and observability infrastructure.
20. Human feedback collection.

These are future enhancements rather than missing requirements for the completed implementation.

---

# 57. Internship Learning Outcomes

The project demonstrates practical understanding of several important AI engineering concepts.

## RAG Fundamentals

Understanding the relationship between retrieval and generation.

## Document Processing

Handling real-world document formats and metadata.

## Chunking

Understanding how document segmentation affects retrieval.

## Embeddings

Converting natural language into numerical representations for semantic search.

## Vector Databases

Using persistent vector storage for similarity retrieval.

## Information Retrieval

Understanding both semantic and lexical retrieval.

## Hybrid Search

Combining multiple retrieval signals.

## Reranking

Applying a second-stage relevance model.

## Prompt Engineering

Designing grounded and security-aware prompts.

## LLM Integration

Connecting an application to a cloud-hosted language model.

## Evaluation

Measuring retrieval and generation behavior using reproducible datasets.

## Security

Testing prompt-injection risks in retrieved documents.

## Software Engineering

Separating functionality into maintainable modules and adding automated tests.

## Application Development

Building an interactive Streamlit interface around the complete pipeline.

---

# 58. End-to-End Technical Summary

The complete DocMind pipeline can be summarized as:

```text
                 DOCUMENT INGESTION
                         │
                         ▼
              PDF / DOCX / TXT Loader
                         │
                         ▼
                    Text Cleaning
                         │
                         ▼
                 Configurable Chunking
                         │
                         ▼
                Sentence Transformer
                         │
                         ▼
                    384D Vectors
                         │
                         ▼
                      ChromaDB
                         │
                         │
                         ▼
USER QUESTION ──────► RETRIEVAL
                         │
                 ┌───────┴────────┐
                 │                │
                 ▼                ▼
              Dense             BM25
                 │                │
                 └───────┬────────┘
                         ▼
                   Hybrid Fusion
                         │
                         ▼
                  Candidate Chunks
                         │
                         ▼
                  Cross-Encoder
                   Reranking
                         │
                         ▼
                    Top Results
                         │
                         ▼
                 Grounded Context
                         │
                         ▼
                   Ollama Cloud
                         │
                         ▼
                  Generated Answer
                         │
                         ▼
             Application-Owned Sources
                         │
                         ▼
                   User Response
```

---

# 59. Final Evaluation Summary

The completed evaluation demonstrates that the application successfully executed the intended RAG workflow across the 30-question Normal-vs-RAG evaluation dataset.

The recorded measurements included:

```text
30 evaluation questions
30 successful comparisons

RAG expected-source retrieval: 100%
RAG citation availability:     100%
Normal citation availability:    0%

Average normal generation:       5.87 s
Average RAG retrieval:         435.89 ms
Average RAG generation:          1.72 s
Average RAG end-to-end:          2.15 s

Normal evidence-support proxy:  0.035
RAG groundedness proxy:          0.851
RAG context relevance:           0.880
```

These values describe the tested implementation and dataset. They are not universal performance guarantees.

---

# 60. Conclusion

DocMind is a complete Retrieval-Augmented Generation knowledge assistant that demonstrates the full lifecycle of a practical RAG application.

The system begins with document ingestion and metadata extraction, converts documents into configurable chunks, generates local semantic embeddings, stores them in ChromaDB, retrieves information through dense and BM25 search, optionally reranks candidates with a cross-encoder, and generates grounded responses through Ollama Cloud.

The application goes beyond basic question answering by implementing source-aware citations, explicit abstention, document lifecycle management, prompt-injection protection, retrieval experiments, answer evaluation, latency analysis, and automated testing.

The project also includes a Streamlit interface that exposes the functionality in an accessible application environment.

The evaluation framework provides reproducible experimental artifacts covering chunking, retrieval, hybrid weighting, reranking, distance behavior, answer quality, and Normal-vs-RAG comparison.

Overall, the project demonstrates the practical integration of:

```text
Document Processing
        +
Embeddings
        +
Vector Search
        +
Information Retrieval
        +
Hybrid Search
        +
Reranking
        +
LLM Generation
        +
Evaluation
        +
Security
        +
Application Engineering
```

The completed implementation provides a strong foundation for a production-oriented knowledge assistant while clearly documenting its current limitations and possible future improvements.

---

# 61. Project Status

**Implementation:** Complete
**RAG Pipeline:** Complete
**Document Ingestion:** Complete
**Chunking:** Complete
**Embeddings:** Complete
**Vector Storage:** Complete
**Semantic Retrieval:** Complete
**Hybrid Retrieval:** Complete
**Reranking:** Complete
**Grounded Generation:** Complete
**Citations:** Complete
**Abstention:** Complete
**Security Testing:** Complete
**Evaluation:** Complete
**Automated Tests:** Passed
**Streamlit Application:** Complete
**README Documentation:** Complete
**Technical Report:** Complete

---

# 62. Key Project Files

For quick reference:

```text
app/main.py
app/ingestion/loader.py
app/ingestion/chunker.py
app/embeddings/embedding_service.py
app/vectorstore/chroma_store.py
app/retrieval/retriever.py
app/retrieval/hybrid_retriever.py
app/generation/answer_service.py
app/generation/prompts.py
app/generation/llm.py
app/utils/document_manager.py

app/evaluation/
app/analytics/

docs/01_rag_fundamentals.md
docs/02_chunking_experiments.md
docs/04_retrieval_evaluation.md
docs/05_prompt_injection_security.md

docs/chunking_experiment_results.csv
docs/hybrid_retrieval_experiment.csv
docs/hybrid_weight_experiment.csv
docs/reranker_metrics_experiment.csv
docs/retrieval_distance_experiment.csv
docs/answer_evaluation_results.csv
docs/normal_vs_rag_results.csv

tests/

README.md
requirements.txt
.env.example
.gitignore
```

---

# 63. Final Statement

DocMind successfully implements an end-to-end Retrieval-Augmented Generation system with document ingestion, semantic and lexical retrieval, hybrid search, reranking, grounded generation, citations, abstention, security controls, evaluation, and an interactive application interface.

The project demonstrates both the theoretical principles of RAG and their practical implementation in a modular Python application.

**Project completed.**
