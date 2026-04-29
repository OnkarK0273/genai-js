# What is the RAG?

it retrieved relevant information from the external source such as doc, DB, Website and using LLM it generates the answer based on retrieved data.

# What RAG solves problems?

## 1\. Hallucination

LLMs sometimes generate **confident but incorrect answers** because:

- They're trained on static data
- They try to "fill in the blanks" even if they don't know

**🛠 How RAG helps**:

By retrieving real, relevant information from external sources, RAG grounds the LLM's answers in actual facts, reducing hallucinations.

## 2\. Outdated Knowledge

LLMs like GPT-3.5 or GPT-4 were trained on data that stops at a certain point (e.g., 2023). They **can’t know things published after that.**

**🛠 How RAG helps**:

RAG pulls the latest data from sources like:

- Company knowledge bases
- Documentation
- Recent articles
- Your personal notes

## 3\. Limited Context Window

LLMs can only “see” a limited amount of text at once (context window, like 8k–128k tokens). If you try to cram too much, important parts get cut off.

**🛠 How RAG helps**:

Instead of sending the whole database or document, RAG:

- Finds only the **most relevant parts**
- Sends them along with the query

# **How does basic Retrieval-Augmented Generation work?**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1745080917219/d4188f2f-6169-4fa2-bf45-e0e3d1a5282c.png)

## indexing phase

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1745080948004/3420f301-fb8c-4c02-ba2e-e4b11b7d7b2c.png)

### 1\. Chunking

- chunking means splitting long document into small parts (chunk)
- chunking is necessary it improves your retrieval accuracy.

common and useful ways to chunk documents in a RAG system:

| **Chunking Method** | **Best For**                       |
| ------------------- | ---------------------------------- |
| Fixed-Length        | Quick start, short texts           |
| Sliding Window      | Better context in chunks           |
| Sentence-Based      | Articles, readable content         |
| Paragraph-Based     | Manuals, essays                    |
| Semantic Chunking   | When high accuracy is needed       |
| Header-Based        | Structured docs, technical content |
| Tokenizer-Based     | LLM-ready chunks by default        |

### 2\. Embedding

- it is a vector (list of number) that represent meaning of word, sentence, paragraph in way that machine can understand.

  ```python
  "happy" → [0.21, -0.11, 0.56, ...]
  ```

### 3\. Vector DB

- it stores the vectors of embedded chunks
- Later, given a **query**, find the **most similar chunks** based on **semantic meaning**, not keywords

## Retrieval phase

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1745080968635/a0fc4bd7-dcdf-42fd-83e9-24030244ad82.png)

### 4\. Semantic Search

- That query is turned into a vector using an embedding model.
- **Vector Search**: The vector is compared to a set of document embeddings stored in a vector database and most similar documents (top-k) are retrieved based on cosine similarity.
- The smaller the distance or higher the similarity → the more relevant the chunk.

## Generation phase

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1745081002049/ddee6e0a-2370-462f-94c8-3f97174bd40c.png)

### 5\. Generation

- After retrieving the most relevant chunks using **semantic search**, the **generation phase** uses those chunks as **context** to answer the user’s question.

  ```python
  User Question + Retrieved Chunks → Prompt → GPT → Final Answer
  ```

---

Learned something? Hit the ❤️ to say “thanks!” and help others discover this article.

**Check out** [**my blog**](https://onkark.hashnode.dev/series/genai-devlopment) **for more things related GenAI**
