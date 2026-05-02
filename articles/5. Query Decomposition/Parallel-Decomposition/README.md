This articles demonstrates two advanced Query Decomposition techniques—Parallel and Iterative decomposition—used to improve document retrieval in a RAG pipeline using LangChain and Google Generative AI.

if u don’t know about Query Decomposition visit our [**articles**](https://onkark.hashnode.dev/query-decomposition), here u learn what is Query Decomposition and their methods.

## **Setup and Initialization**

1.  Setup `.env` file

    ```shell
    GROQ_API_KEY = [Your groq api keys for chat-models]
    GOOGLE_API_KEY = [Your google api keys for embed-models]
    QDRANT_URL = http://localhost:6333
    ```

2.  Libraries required

    ```shell
    pnpm install @langchain/community @langchain/core @langchain/google-genai @langchain/groq @langchain/qdrant  langchain
    ```

3.  Setup `app.js` file  
    First, we set up the environment, load models, and configure the connection to the vector store (Qdrant) and the embedding model.

    ```typescript
    import { ChatGroq } from "@langchain/groq";
    import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
    import { QdrantVectorStore } from "@langchain/qdrant";
    import { ChatPromptTemplate } from "@langchain/core/prompts";
    import { StringOutputParser } from "@langchain/core/output_parsers";

    // --- LLM and Embedding setup ---

    // LLM for final answer synthesis
    const model = new ChatGroq({
      model: "openai/gpt-oss-20b",
    });

    // Embedding model for converting text to vector space
    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: "gemini-embedding-001",
    });

    // --- Vector store and retriever setup ---

    // Connect to the existing Qdrant collection
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: process.env.QDRANT_URL,
        collectionName: "genai-toolkit",
      },
    );

    // Create a basic retriever instance
    const retriever = vectorStore.asRetriever();
    ```

# Parallel Query Decomposition

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">GitHub code file - <a target="_self" rel="noopener noreferrer" class="text-primary underline underline-offset-2 hover:text-primary/80 cursor-pointer notion-link-token notion-focusable-token notion-enable-hover" href="https://github.com/OnkarK0273/articles/blob/main/genai/Query%20Translation/python/Parallel-Decomposition/main.py" style="pointer-events: none;">link</a></div>
</div>

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759836383448/4ad45a77-f62f-484a-9184-b195233b9d18.png)

> here is complete [**GitHub source code**](https://github.com/OnkarK0273/genai-js/tree/main/articles/5.%20Query%20Decomposition/Parallel-Decomposition)

## 1\. **Prompt**

- `templateDecomposition`\- a prompt instruction that tells llm to break down complex questions into 3 sub-questions
- `promptDecomposition` - `templateDecomposition` it wraps around `promptDecomposition` to use it in chain

```typescript
// --- Query-decomposition generation prompt ---
const templateDecomposition = `You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
note: generate only queries
Output (only 3 queries):`;

const promptDecomposition = ChatPromptTemplate.fromTemplate(
  templateDecomposition,
);
```

## 2\. **Structured Output**

- Here we using `querySchema` to describe what output should want from model
- We pass that `querySchema` to model using `withStructuredOutput` method.

  ```typescript
  const querySchema = {
    title: "queries",
    type: "object",
    properties: {
      Query1: { type: "string", description: "Query1 of user input" },
      Query2: { type: "string", description: "Query2 of user input" },
      Query3: { type: "string", description: "Query3 of user input" },
    },
    required: ["Query1", "Query2", "Query3"],
  };

  const modelWithStruture = model.withStructuredOutput(querySchema, {
    method: "jsonSchema",
  });
  ```

## 3\. **Query Generation Chain**

a chain do following things

1.  `promptDecomposition` - take input prompts.
2.  `modelWithStruture` - run it through to get strutured output.
3.  `(output) => Object.values(output)` - convert `modelWithStruture` output into array of values

    ```typescript
    // --- Query Generation Chain ---
    const generateQueriesChain = promptDecomposition
      .pipe(modelWithStruture)
      .pipe((output) => Object.values(output));
    ```

## 4\. **Prompt for Sub-question Answering**

- `reterivedTemplate` - Prompt instruction to answer sub-question using retrieved context.
- `reterivedPrompt` - it is used in chain

```typescript
// --- Retrieved sub-question answer generation prompt ---
const retrievedTemplate = `
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences.
Question: {question}
Context: {context}
Answer:
`;

const retrievedPrompt = ChatPromptTemplate.fromTemplate(retrievedTemplate);
```

## 5\.  **Sub-question Retrieval and Answer Generation**

### 1\. **Q&A Pair Formatting Utility.** `formatQAPairs`

Formats each `sub-question` and its answer into a readable string, ready to be used as context for the final answer.

### 2\. **Sub-question Retrieval and Answer Generation.** `retrieveAndRag`

`retrieveAndRag` this is main function that do followings things:

1.  `subQuestions` **Decompose** the main question into sub-questions using `queryGenration`.
2.  For each `subQuestion`:
    1.  **Retrieve** relevant context from the vector store.
    2.  **Run a chain** that feeds the sub-question and its context to the LLM to get an answer.
    3.  **Collect** all answers.

3.  **Format** all Q&A pairs into a single context string using `formatQAPairs`utility.

```typescript
// --- Utility: context merging sub-question with answers ---
function formatQAPairs(questions, answers) {
  /**
   * Format Q and A pairs
   */
  let formattedString = "";
  for (let i = 0; i < questions.length; i++) {
    formattedString += `Question ${i + 1}: ${questions[i]}\nAnswer ${i + 1}: ${answers[i]}\n\n`;
  }
  return formattedString.trim();
}

// --- Utility: Sub-questions and their answers ---
async function retrieveAndRag(question, queryGeneration, retrievedPrompt) {
  /**
   * RAG on each sub-question
   */
  const subQuestions = await queryGeneration.invoke({ question: question });

  const ragResults = [];

  for (const subQuestion of subQuestions) {
    // Create a chain for each sub-question
    const chain = retrievedPrompt.pipe(model).pipe(new StringOutputParser());

    // Retrieve context for the sub-question
    const context = await retriever.invoke(subQuestion);

    // Format context
    const contextString = context.map((doc) => doc.pageContent).join("\n");

    const ans = await chain.invoke({
      question: subQuestion,
      context: contextString,
    });

    ragResults.push(ans);
  }

  const context = formatQAPairs(subQuestions, ragResults);
  return context;
}
```

## 6\. **Final Synthesis Prompt and Chain**

- prompt template - prompt instruction for llm to answer the question by using all the Q&A pairs as context.
- `finalRagChain`\- run a prompt through llm and parse the output.
- **Execution -** Runs the final synthesis chain to get a concise, synthesized answer.

```typescript
// --- Main execution ---
async function main() {
  const question = "what AI generalist";
  const context = await retrieveAndRag(
    question,
    generateQueriesChain,
    retrievedPrompt,
  );

  // --- Prompt for final answer ---
  const template = `Here is a set of Q+A pairs:
    {context}
    Use these to synthesize an answer to the question: {question}
    `;

  const prompt = ChatPromptTemplate.fromTemplate(template);

  // --- Final RAG chain ---
  const finalRagChain = prompt.pipe(model).pipe(new StringOutputParser());

  // --- Run the chain ---
  const finalAns = await finalRagChain.invoke({
    context: context,
    question: question,
  });

  console.log("Final answer:", finalAns);
}

// Execute main function
main().catch(console.error);
```

# Resources

## **1.Github code**

- [Parallel Query Decomposition](https://github.com/OnkarK0273/genai-js/tree/main/articles/5.%20Query%20Decomposition/Parallel-Decomposition)
- [Iterative Query Decomposition](https://github.com/OnkarK0273/genai-js/tree/main/articles/5.%20Query%20Decomposition/Iterative%20Query-Decomposition)

## 2\. **Articles**

- [What is Query Decomposition](https://onkark.hashnode.dev/query-decomposition)

---

Learned something? Hit the ❤️ to say “thanks!” and help others discover this article.

**Check out** [**my blog**](https://onkark.hashnode.dev/series/genai-devlopment) **for more things related GenAI.**
