import { ChatGroq } from "@langchain/groq";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

// --- LLM and Embedding setup ---
const model = new ChatGroq({
  model: "openai/gpt-oss-20b",
});

const embedder = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001",
});

// --- Vector store setup ---
const vectorStore = await QdrantVectorStore.fromExistingCollection(embedder, {
  url: process.env.QDRANT_URL,
  collectionName: "genai-toolkit",
});

const retriever = vectorStore.asRetriever();

// --- Query-decomposition generation prompt ---
const templateDecomposition = `You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
Generate multiple search queries related to: {question}
note: generate only queries
Output (only 3 queries):`;

const promptDecomposition = ChatPromptTemplate.fromTemplate(
  templateDecomposition,
);

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

// --- Query Generation Chain ---
const queryGeneration = promptDecomposition
  .pipe(modelWithStruture)
  .pipe((output) => Object.values(output));

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
