import { ChatGroq } from "@langchain/groq";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

// --- Multi-query generation prompt ---
const multiQueryTemplate = `
Please write a scientific paper passage to answer the question
Question: {question}
Passage
`;

const prompt_hyde = ChatPromptTemplate.fromTemplate(multiQueryTemplate);

const large_parameter_model = new ChatGroq({
  model: "openai/gpt-oss-120b",
});

const model = new ChatGroq({
  model: "openai/gpt-oss-20b",
});

const generate_docs_for_retrieval = prompt_hyde
  .pipe(large_parameter_model)
  .pipe(async (answer) => await new StringOutputParser().invoke(answer));

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001",
});

// ---- vectore store ----
const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "genai-toolkit",
});

const retriever = vectorStore.asRetriever();

const retrieval_chain_hyde = generate_docs_for_retrieval.pipe(
  async (hyde_doc) => await retriever.invoke(hyde_doc),
);

// --- Prompt for final answer ---
const answerTemplate = `Answer the following question based on this context:
{context}
Question: {question}
`;
const answerPrompt = ChatPromptTemplate.fromTemplate(answerTemplate);

const finalRagChain = async (input) => {
  const context = await retrieval_chain_hyde.invoke(input);
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
  const userQuestion = "what is AI generalist";
  const answer = await finalRagChain({ question: userQuestion });
  console.log("Final answer:", answer);
})();
