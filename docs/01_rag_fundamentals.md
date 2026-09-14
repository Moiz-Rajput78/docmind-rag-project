# DocMind — RAG Fundamentals

## 1. Introduction

DocMind is a Retrieval-Augmented Generation (RAG) based knowledge assistant.

The purpose of DocMind is to allow a user to ask questions about a private collection of documents while ensuring that the generated answers are grounded in information retrieved from those documents.

A traditional large language model (LLM) generates answers primarily from knowledge encoded in its model parameters. This can work well for general questions, but it creates problems when the user asks about private, specialized, or recently updated information.

RAG addresses this problem by combining two major capabilities:

1. Information retrieval
2. Natural-language generation

Instead of asking the language model to answer a question using only its internal knowledge, DocMind first searches a document collection for relevant information. The retrieved information is then provided to the language model as context.

The basic workflow is:

```text
User Question
      ↓
Question Processing
      ↓
Query Embedding
      ↓
Knowledge Base Search
      ↓
Relevant Document Chunks
      ↓
Context Construction
      ↓
Local LLM
      ↓
Grounded Answer
      ↓
Source Citations