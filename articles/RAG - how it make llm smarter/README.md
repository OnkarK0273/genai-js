# How RAG Makes LLMs Smarter

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a process that retrieves relevant information from external sources—such as documents, databases, or websites—and uses an LLM to generate answers based on that retrieved data.

## Problems RAG Solves

### 1. Hallucination

LLMs sometimes generate confident but incorrect answers because they are trained on static data and often try to "fill in the blanks" even when they lack specific knowledge.

- **How RAG helps:** It grounds the LLM's answers in actual facts by retrieving real, relevant information from external sources.

### 2. Outdated Knowledge

LLMs have a training cutoff (e.g., 2023) and cannot know about events or information published after that point.

- **How RAG helps:** It pulls the latest data from company knowledge bases, documentation, and recent articles.

### 3. Limited Context Window

LLMs can only process a limited amount of text at once (the context window). Cramming too much information results in data being cut off.

- **How RAG helps:** Instead of sending an entire database, RAG finds only the most relevant parts and sends them along with the query.

---

## The RAG Pipeline

### Indexing Phase

The indexing phase prepares external documents for retrieval.

1.  **Chunking:** Splitting long documents into smaller parts (chunks) to improve retrieval accuracy.

    | Chunking Method   | Best For                           |
    | :---------------- | :--------------------------------- |
    | Fixed-Length      | Quick start, short texts           |
    | Sliding Window    | Better context in chunks           |
    | Sentence-Based    | Articles, readable content         |
    | Paragraph-Based   | Manuals, essays                    |
    | Semantic Chunking | When high accuracy is needed       |
    | Header-Based      | Structured docs, technical content |
    | Tokenizer-Based   | LLM-ready chunks by default        |

2.  **Embedding:** Converting text into a vector (a list of numbers) that represents its meaning so a machine can understand it.
3.  **Vector DB:** A database that stores these embedded chunks.

### Retrieval Phase

When a user submits a query, the system finds the most similar chunks based on semantic meaning rather than just keywords.

- **Semantic Search:** The query is turned into a vector using an embedding model.
- **Vector Search:** The query vector is compared to stored document embeddings using **cosine similarity**.
- **Relevance:** A smaller distance or higher similarity score indicates a more relevant chunk.

### Generation Phase

The final stage where the LLM produces a response.

- The most relevant chunks are used as context for the user's question.
- **Formula:** User Question + Retrieved Chunks → Prompt → GPT.

---

## Advanced RAG Techniques

### Query Translation (Query Decomposition)

This advanced technique involves breaking down a complex problem into multiple sub-problems to improve document retrieval.

- **Parallel Decomposition:** Handling sub-queries simultaneously.
- **Iterative Decomposition:** Handling sub-queries step-by-step.

### Query Translation (Query Re-writing)

Techniques used to transform a query for better results:

- **Multi-Query:** Generating multiple versions of a query.
- **RAG Fusion:** Re-ranking search results.
- **HyDE (Hypothetical Document Embeddings):** Generating a hypothetical answer to improve retrieval.
