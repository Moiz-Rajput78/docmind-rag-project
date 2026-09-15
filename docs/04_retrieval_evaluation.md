# Retrieval Evaluation

## 1. Objective

The purpose of this experiment is to evaluate the semantic retrieval component of DocMind RAG.

The evaluation measures whether the retrieval system can return the expected source documents for a set of questions from the synthetic NovaTech Solutions knowledge base.

The evaluation dataset contains:

- 30 total questions
- 10 easy questions
- 10 medium questions
- 10 difficult questions

The following retrieval metrics were measured:

- Hit Rate@K
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

The experiments were performed with:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Vector database: ChromaDB
- Stored chunks: 25
- Retrieval method: semantic vector similarity
- K values: 1, 3, 5, 10

---

## 2. Evaluation Dataset

The evaluation dataset contains questions covering:

- Company information
- Products
- Technical documentation
- Human resources policies
- Multi-document questions
- Troubleshooting scenarios

Questions are divided into three difficulty levels.

### Easy

Easy questions generally ask for a direct fact that should exist clearly in one document.

Examples:

- How many annual leave days do full-time employees receive?
- What is NovaDesk?
- What is NovaFlow?
- What departments does NovaTech have?

### Medium

Medium questions require retrieving more specific information or connecting information within the knowledge base.

Examples:

- How far in advance should planned annual leave normally be requested?
- What areas can be configured in NovaDesk?
- What should administrators verify after installing NovaDesk?

### Difficult

Difficult questions require combining information from multiple documents or handling troubleshooting scenarios.

Examples:

- NovaDesk is not starting after configuration changes. What troubleshooting steps should an administrator take?
- A new NovaDesk installation needs both browser access and administrative configuration. What prerequisites and post-installation checks are required?
- A NovaDesk user cannot sign in and the application also has configuration problems. Which documentation should be consulted and what areas should be checked?

---

## 3. Metrics

### Hit Rate@K

Hit Rate@K measures whether at least one expected source document appears in the first K retrieved results.

Formula:

```text
Hit Rate@K =
questions with at least one relevant result in Top-K
/
total questions