import { ChatGroq } from "@langchain/groq";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';

// --- Multi-query generation prompt ---
const multiQueryTemplate = `
You are a helpful assistant that generates multiple search queries based on a single input query.
Generate multiple search queries (3 queries) related to: {question}
Output (only generate queries):
query1
query2
query3
`;
const multiQueryPrompt = ChatPromptTemplate.fromTemplate(multiQueryTemplate);

// --- json schema for strutured output ---
const jsonSchema = {
  title: "queries",
  type: "object",
  properties: {
    Query1: { type: "string", description: "Query1 of user input" },
    Query2: { type: "string", description: "Query2 of user input" },
    Query3: { type: "string", description: "Query3 of user input" },
  },
  required: ["Query1", "Query2", "Query3"],
};

const model = new ChatGroq({
  model: "openai/gpt-oss-20b",
});

const modelWithStruture = model.withStructuredOutput(jsonSchema, {
  method: "jsonSchema",
});

// --- Query Generation Chain ---
const generateQueriesChain = multiQueryPrompt
  .pipe(modelWithStruture)
  .pipe((output) => Object.values(output));


const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001",
});

// ---- vectore store ----

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "genai-toolkit",
});

const retriever = vectorStore.asRetriever()

// --- Utility: Unique union of retrieved documents ---
function getUniqueUnion(documents) {
  const flattenedDocs = documents.flat();
  
  // Use a Map to track unique content. 
  // If a duplicate pageContent appears, the Map just overwrites the entry,
  // effectively keeping only one instance.
  const uniqueMap = new Map();
  
  flattenedDocs.forEach(doc => {
    // You can use doc.pageContent or a specific metadata field here
    uniqueMap.set(doc.pageContent, doc);
  });

  return Array.from(uniqueMap.values());
}


// --- Retrieval chain ---
const retrievalChain = generateQueriesChain
  .pipe(async (queries) => {
    const results = await Promise.all(
      queries.map((query) => retriever.invoke(query))
    );
    return results;
  })
  .pipe((documents) => getUniqueUnion(documents));

  // --- Prompt for final answer ---
const answerTemplate = `Answer the following question based on this context:
{context}
Question: {question}
`;
const answerPrompt = ChatPromptTemplate.fromTemplate(answerTemplate);

const finalRagChain = async (input) => {
  const context = await retrievalChain.invoke(input);
  const promptValue = await answerPrompt.invoke({
    context: JSON.stringify(context),
    question: input.question,
  });
  const answer = await model.invoke(promptValue);
  const output = await new StringOutputParser().invoke(answer);
  return output;
};

// --- Run the chain ---
(async () => {
  const userQuestion = 'what is AI generalist';
  const answer = await finalRagChain({ question: userQuestion });
  console.log('Final answer:', answer);
})();