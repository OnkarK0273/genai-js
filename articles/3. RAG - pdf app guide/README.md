in this blog, I teach how to create basic pdf-based RAG app by using LangChain

# Why LangChain (in-short)

it provides a **modular, flexible framework** to build RAG based application

LangChain is provides you all necessary components that are required while building any RAG based application, so we don’t need to create from scratch.

# **Core Components of a Basic RAG Application**

1.  **Document Loading**
2.  **Text Splitting**
3.  **Embedding**
4.  **Vector Store**
5.  **Retrieval**
6.  **Generation**

> if you want to learn more about the core component of RAG checkout this blog - [link](https://onkark.hashnode.dev/how-rag-makes-llms-smarter)

# **Building Process: A Step-by-Step Guide**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1745148565395/2462f340-a6ee-4675-9793-7acffb4d5b24.png)

## Prerequisites

1.  Node.js Projects - `pnpm init`
2.  Docker installed
3.  Libraries required

    ```shell
    pnpm install @langchain/community @langchain/core @langchain/google-genai @langchain/groq @langchain/qdrant @langchain/textsplitters langchain pdf-parse
    ```

4.  Keys required `.env`

    ```yaml
    GROQ_API_KEY = [Your groq api keys for chat-models]
    GOOGLE_API_KEY = [Your google api keys for embed-models]
    QDRANT_URL = http://localhost:6333
    ```

5.  Qdrant running locally ( docker-compose[.](https://onkark.hashnode.dev/how-rag-makes-llms-smarter)db.yml )
    - create docker-compose.db.yml file and use the below code

    ```yaml
    services:
      qdrant:
        image: qdrant/qdrant
        ports:
          - 6333:6333
    ```

    - To run the QdrantDB locally use this code in terminal

    ```bash
    docker compose -f docker-compose.db.yml up
    ```

## File Structure

```plaintext
RAG - pdf app guide/
├── node_modules/             # Installed dependencies
├── src/
│   ├── pdf/                  # Folder for input PDF files
│   └── app.js                # Main application logic
├── .env                      # Local environment variables (Secrets)
├── docker-compose.db.yml     # Qdrant database configuration
├── package.json              # Project metadata and scripts
└── pnpm-lock.yaml            # PNPM dependency lockfile
```

## Step - 1: Document Loading and Splitting

- Loading a PDF with `PDFLoader`
- Chunking content using `RecursiveCharacterTextSplitter`
  - `chunkSize` - each chunk will contain up to 1000 characters.
  - `chunkOverlap` - each new chunk will share 200 characters with the previous one.

  ```typescript
  import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
  import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

  // step-1------------------------
  // load document
  const loader = new PDFLoader("src/pdf/genai-toolkit.pdf");
  const docs = await loader.load();

  // step 1 - splitting  the document

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 250,
    chunkOverlap: 50,
  });
  const split_doc = await splitter.splitDocuments(docs);
  ```

## Step -2: Generating Embeddings and Storing in Qdrant

- Creating embeddings with gemini's `gemini-embedding-001`
- Storing chunked documents in Qdrant vector store
  - it required collection name to store embedded chunks in QdrantDB

  ```typescript
  import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
  import { QdrantVectorStore } from "@langchain/qdrant";

  // step-2------------------------
  const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "gemini-embedding-001",
  });

  // create new collection in qdrant db
  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddings,
    {
      url: process.env.QDRANT_URL,
      collectionName: "genai-toolkit",
    },
  );

  // add document in db
  await vectorStore.addDocuments(split_doc);
  ```

## Step-3: Retrieving Relevant Chunks Based on a Query

- Loading existing vector store
- Performing semantic search using `similarity_search()`
  - `similarity_search()` - it compares the **query vector** with all stored **document vectors** in the Vector DB using **Cosine Similarity.**
  - based on cosine similarities it retrieves relevant chunk of document from VectorDB.

  ```typescript
  //step3 - Retrieving Relevant Chunks --------------------

  const query = "what is  AI Generalist";

  const similaritySearchResults = await vectorStore.similaritySearch(query);
  ```

## Step-4: Generating Final Response using LLM

- Formatting retrieved documents and query into a prompt
- Creating prompt using `ChatPromptTemplate`
- Sending the prompt to OpenAI's GPT model (e.g., `gpt-4o-mini`)

  ```typescript
  import { ChatGroq } from "@langchain/groq";

  let context = "";

  // combine all reterived doc into sinegle document and store in page_content

  for (let i = 0; i < similaritySearchResults.length; i++) {
    context += similaritySearchResults[i].pageContent;
  }

  // create chat message

  const messages = [
    {
      role: "system",
      content: `You are an assistant helping to answer user queries based on the following content. If the answer to the query is not found in the content, respond with: "This query is not present in the document." Here is the content:\\n\\n ${context}`,
    },
    { role: "user", content: `Explain in simple terms: what is ${query}` },
  ];

  // chat-groq model
  const model = new ChatGroq({
    model: "openai/gpt-oss-20b",
  });

  // send chat message into model

  const res = await model.invoke(messages);

  console.log(res.content);
  ```

## Sample Output

- **Input query**

  ```python
  query = "what is  AI Generalist"
  ```

- **Retrieved documen**t from vectorDB

  ```plaintext
  understanding of AI technologies and excel at applying the right tools to solve
  complex business challenges. They are the essential link between AI's technical
  potential and real-world value.
  Key Traits of an AI Generalist:understanding of AI technologies and excel at applying the right tools to solve
  complex business challenges. They are the essential link between AI's technical
  potential and real-world value.
  Key Traits of an AI Generalist:understanding of AI technologies and excel at applying the right tools to solve
  complex business challenges. They are the essential link between AI's technical
  potential and real-world value.
  Key Traits of an AI Generalist:Key Traits of an AI Generalist:
  Broad AI Model Knowledge: A deep familiarity with the landscape of AI
  models and their unique capabilities.
  Strategic Application: The ability to discern which AI tool is best suited for a
  ```

- input query + Retrieved document —> llm= **response**

  ```plaintext
  An **AI Generalist** is someone who knows a wide range of artificial‑intelligence tools and models, and can pick the right one to solve a particular problem.
  - They understand the technical details of many AI systems.
  - They can decide which tool is best for a given business challenge.
  - In short, they act as the bridge between AI’s technical possibilities and the practical value it can bring to real‑world projects.
  ```

## Resources

Source code - [link](https://github.com/OnkarK0273/genai-js/tree/main/articles/RAG%20-%20pdf%20app%20guide)

---

Learned something? Hit the ❤️ to say “thanks!” and help others discover this article.

**Check out** [**my blog**](https://onkark.hashnode.dev/series/genai-devlopment) **for more things related GenAI**
