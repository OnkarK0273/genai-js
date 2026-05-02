import { ChatGroq } from "@langchain/groq";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

// --- LLM and Embedding setup ---
const model = new ChatGroq({
  model: "openai/gpt-oss-20b",
});

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001",
});

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

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "genai-toolkit",
});

const retriever = vectorStore.asRetriever();

// --- Query-decomposition generation prompt ---
const templateDecomposition = `You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
note: generate only queries
Output (only 3 queries):`;

const promptDecomposition = ChatPromptTemplate.fromTemplate(
  templateDecomposition,
);

// --- Query Generation Chain ---
const generateQueriesChain = promptDecomposition
  .pipe(modelWithStruture)
  .pipe((output) => Object.values(output));

// --- Retrieved sub-question answer generation prompt ---
const retrievedTemplate = `
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences.
Question: {question}
Context: {context}
Answer:
`;

const retrievedPrompt = ChatPromptTemplate.fromTemplate(retrievedTemplate);

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
