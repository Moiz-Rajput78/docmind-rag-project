# Chunking Experiments

## 1. Objective

The purpose of this experiment is to understand how chunk size and chunk overlap affect the document representation used by the DocMind RAG system.

Chunking is important because the retrieved chunks become the evidence supplied to the language model. Very small chunks may lose useful context, while very large chunks may contain unrelated information and reduce retrieval precision.

---

## 2. Experimental Setup

The synthetic NovaTech Solutions knowledge base contains 12 documents covering:

- Company information
- Company policies
- Products
- Technical documentation
- Human resources

The documents include PDF, DOCX, and TXT files.

Four chunk sizes were tested:

- 300 characters
- 500 characters
- 800 characters
- 1200 characters

Four overlap configurations were tested:

- 0 characters
- 50 characters
- 100 characters
- 200 characters

This produced 16 valid configurations.

---

## 3. Results

| Chunk Size | Overlap | Chunks | Average Length | Minimum | Maximum | Total Characters |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 0 | 37 | 240.35 | 22 | 300 | 8893 |
| 300 | 50 | 39 | 262.62 | 68 | 300 | 10242 |
| 300 | 100 | 45 | 270.87 | 102 | 300 | 12189 |
| 300 | 200 | 72 | 289.99 | 202 | 300 | 20879 |
| 500 | 0 | 24 | 370.54 | 123 | 500 | 8893 |
| 500 | 50 | 24 | 395.50 | 173 | 500 | 9492 |
| 500 | 100 | 25 | 407.64 | 121 | 500 | 10191 |
| 500 | 200 | 27 | 440.41 | 218 | 500 | 11891 |
| 800 | 0 | 15 | 592.93 | 17 | 800 | 8894 |
| 800 | 50 | 15 | 603.00 | 68 | 800 | 9045 |
| 800 | 100 | 15 | 613.07 | 118 | 800 | 9196 |
| 800 | 200 | 15 | 633.07 | 218 | 800 | 9496 |
| 1200 | 0 | 12 | 741.33 | 623 | 922 | 8896 |
| 1200 | 50 | 12 | 741.33 | 623 | 922 | 8896 |
| 1200 | 100 | 12 | 741.33 | 623 | 922 | 8896 |
| 1200 | 200 | 12 | 741.33 | 623 | 922 | 8896 |

---

## 4. Observations

### 4.1 Smaller chunks

The 300-character configuration produced between 37 and 72 chunks depending on overlap.

Smaller chunks provide more granular retrieval units. This can be useful when a user asks about a very specific fact.

However, very small chunks may not contain enough surrounding context to fully answer a question.

### 4.2 Medium-sized chunks

The 500-character configuration produced 24–27 chunks.

This provides substantially more context than 300-character chunks while maintaining relatively fine retrieval granularity.

The 500/100 configuration produced 25 chunks with an average length of approximately 408 characters.

### 4.3 Larger chunks

The 800-character configuration produced only 15 chunks.

The 1200-character configuration produced 12 chunks because the documents in this knowledge base are relatively short.

Larger chunks reduce the number of retrieval candidates but may combine several concepts into the same retrieval unit.

### 4.4 Effect of overlap

Increasing overlap increases the amount of duplicated text stored in the chunk collection.

For example, with 300-character chunks:

- 0 overlap → 37 chunks
- 100 overlap → 45 chunks
- 200 overlap → 72 chunks

A 200-character overlap therefore produces substantially more redundancy.

Overlap can nevertheless be useful when an important fact occurs near the boundary between two chunks.

---

## 5. Selected Configuration

For the initial DocMind RAG implementation, the selected configuration is:

**Chunk size: 500 characters**

**Chunk overlap: 100 characters**

The reason for this choice is that it provides a practical balance between:

1. Retrieval granularity
2. Context preservation
3. Number of stored chunks
4. Storage redundancy
5. Future context-window usage

This configuration will be treated as the initial baseline.

It will not be considered permanently optimal until retrieval and answer evaluation are performed.

---

## 6. Limitations

This experiment measures structural properties of chunking, such as chunk count and length.

It does not directly measure retrieval quality or answer correctness.

A configuration that produces fewer or more chunks is not necessarily better.

Therefore, the selected configuration should be evaluated again using the retrieval evaluation dataset and answer-quality metrics.

---

## 7. Conclusion

The experiment demonstrates that chunk size and overlap significantly affect the number and characteristics of retrieval units.

The initial baseline of 500-character chunks with 100-character overlap provides a reasonable balance for the current synthetic knowledge base.

Future experiments should compare this baseline against alternative configurations using actual retrieval metrics such as:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Context relevance
- Answer correctness
- Groundedness