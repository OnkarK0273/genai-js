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

# Iterative Query Decomposition

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">GitHub code file - <a target="_self" rel="noopener noreferrer" class="text-primary underline underline-offset-2 hover:text-primary/80 cursor-pointer notion-link-token notion-focusable-token notion-enable-hover" href="https://github.com/OnkarK0273/articles/blob/main/genai/Query%20Translation/python/Query-Decomposition/main.py" style="pointer-events: none;">link</a></div>
</div>

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759836680777/04cd6be6-b782-4de1-ac0e-00636aa7d281.png)

> here is complete [**GitHub source code**](https://github.com/OnkarK0273/genai-js/tree/main/articles/5.%20Query%20Decomposition/Iterative%20Query-Decomposition)

## 1\. **Prompt**

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">this is same step as we discussed in <a target="_self" rel="noopener noreferrer nofollow" class="text-primary underline underline-offset-2 hover:text-primary/80 cursor-pointer" href="https://onkark.hashnode.dev/query-translation-query-decomposition-code-explanation#heading-1-prompt" style="pointer-events: none;">Parallel Query Decomposition</a></div>
</div>

- `templateDecomposition` - a prompt instruction that tells llm to break down complex questions into 3 sub-questions
- `promptDecomposition` - `templateDecomposition` it wraps around `promptDecomposition` to use it in chain

```typescript
// --- Query-decomposition generation prompt ---
const templateDecomposition = `You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
Generate multiple search queries related to: {question}
note: generate only queries
Output (only 3 queries):`;

const promptDecomposition = ChatPromptTemplate.fromTemplate(
  templateDecomposition,
);
```

## **2\. Structured Output**

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

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">this is also same step as we discussed in <a target="_self" rel="noopener noreferrer nofollow" class="text-primary underline underline-offset-2 hover:text-primary/80 cursor-pointer" href="https://onkark.hashnode.dev/query-translation-query-decomposition-code-explanation#heading-2-query-generation-chain" style="pointer-events: none;">Parallel Query Decomposition</a></div>
</div>

a chain do following things

1.  `promptDecomposition` - take input prompts.
2.  `modelWithStruture` - run it through to get strutured output.
3.  `(output) => Object.values(output)` - convert `modelWithStruture` output into array of values

```typescript
// --- Query Generation Chain ---
const queryGeneration = promptDecomposition
  .pipe(modelWithStruture)
  .pipe((output) => Object.values(output));
```

## 4\. **Prompt for Sub-question Answering**

A prompt template that asks the LLM to answer a sub-question using:

- The sub-question itself,
- Any background Q&A pairs,
- Additional retrieved context.

```typescript
// --- Retrieved sub-question answer generation prompt ---
const retrievedTemplate = `Here is the question you need to answer:

--- 
{question}
--- 

Here is any available background question + answer pairs:

--- 
{q_a_pairs}
--- 

Here is additional context relevant to the question:

--- 
{context}
--- 

Use the above context and any background question + answer pairs to answer the question:
{question}`;

const retrievedPrompt = ChatPromptTemplate.fromTemplate(retrievedTemplate);
```

## 5\. **Sub-question Retrieval and Answer Generation**

### 1\. **Q&A Pair Formatting Utility.** `formatQAPairs`

Formats a `subQuestion` and its answer into a readable string.

### 2\. **Sub-question Retrieval and Answer Generation.** `retrieveAndRag`

`retrieveAndRag` this is main function that do followings things:

1.  `subQuestions` **Decompose** the main question into sub-questions using `queryGenration`.
2.  For each `subQuestion`:
    1.  **Retrieve** relevant context from the vector store.
    2.  **Run a chain** that feeds the sub-question, any background Q&A pairs, and the retrieved context to the LLM to get an answer.
    3.  **Format** the Q&A pair and accumulate it.

3.  Returns all Q&A pairs as a single formatted string.

```typescript
// --- Utility: format Q and A pair ---
function formatQAPair(question, answer) {
  /**Format Q and A pair */
  return `Question: ${question}\nAnswer: ${answer}`;
}

// --- Each sub-question runs concurrently -> retrieve document -> generate response ---
async function retrieveAndRag(question, queryGeneration, retrievedPrompt) {
  /**RAG on each sub-question */
  const subQuestions = await queryGeneration.invoke({ question: question });

  let qAPairs = "";

  for (const subQuestion of subQuestions) {
    const chain = retrievedPrompt.pipe(model).pipe(new StringOutputParser());

    const context = await retriever.invoke(subQuestion);
    const contextString = context.map((doc) => doc.pageContent).join("\n\n");

    const ans = await chain.invoke({
      question: subQuestion,
      q_a_pairs: "",
      context: contextString,
    });

    const qAPair = formatQAPair(subQuestion, ans);
    qAPairs = qAPairs + "\n-----\n" + qAPair;
  }

  return qAPairs;
}
```

## 6\. **Final Synthesis Prompt and Chain**

- prompt template - prompt instruction for llm to answer the question by using all the Q&A pairs as context.
- `finalRagChain` - run a prompt through llm and parse the output.
- **Execution -** Runs the final synthesis chain to get a concise, synthesized answer.

```typescript
// --- Main execution ---
async function main() {
  const question = "what AI generalist";

  // Retrieve and generate context
  const context = await retrieveAndRag(
    question,
    queryGeneration,
    retrievedPrompt,
  );

  // --- Prompt for final answer ---
  const template = `Here is a set of Q+A pairs:
{context}
Use these to synthesize an answer to the question: {question}`;

  const prompt = ChatPromptTemplate.fromTemplate(template);

  // --- Final RAG chain ---
  const finalChain = prompt.pipe(model).pipe(new StringOutputParser());

  // --- Run the chain ---
  const finalAns = await finalChain.invoke({
    context: context,
    question: question,
  });

  console.log("Final answer:", finalAns);
}

// Run the main function
main().catch(console.error);
```

# Resources

## **1.Github code**

- [**Parallel** Query **Decomposition**](https://github.com/OnkarK0273/genai-js/tree/main/articles/5.%20Query%20Decomposition/Parallel-Decomposition)
- [Iterative Query Decomposition](https://github.com/OnkarK0273/genai-js/tree/main/articles/5.%20Query%20Decomposition/Iterative%20Query-Decomposition)

## 2\. **Articles**

- [What is Query Decomposition](https://onkark.hashnode.dev/query-decomposition)

---

Learned something? Hit the ❤️ to say “thanks!” and help others discover this article.

**Check out** [**my blog**](https://onkark.hashnode.dev/series/genai-devlopment) **for more things related GenAI.**
