import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatGroq } from "@langchain/groq";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { QdrantVectorStore } from "@langchain/qdrant";

// indexign phase-------------------------------
// load document
const loader = new PDFLoader("src/pdf/genai-toolkit.pdf");
const docs = await loader.load();

// step 1 - splitting  the document

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 250,
  chunkOverlap: 50,
});
const texts = await splitter.splitDocuments(docs);

// step 2 - embedding

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001",
});

// create new collection in qdrant db

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "genai-toolkit",
});

// add document in db
await vectorStore.addDocuments(texts);

// Retreival phase --------------------------------------------

// step 4 - semantic search reterived document

const query = "what is  AI Generalist";

const similaritySearchResults = await vectorStore.similaritySearch(query);

let context = "";
for (let i = 0; i < similaritySearchResults.length; i++) {
  context += similaritySearchResults[i].pageContent;
}

// genration phase ---------------------------

// step 5 - send query with Reterived doc. to GPT

const messages = [
  {
    role: "system",
    content: `You are an assistant helping to answer user queries based on the following content. If the answer to the query is not found in the content, respond with: "This query is not present in the document." Here is the content:\\n\\n ${context}`,
  },
  { role: "user", content: `Explain in simple terms: what is ${query}` },
];

const model = new ChatGroq({
  model: "openai/gpt-oss-20b",
});

// step 5 - invoke the model

const res = await model.invoke(messages);

console.log(res.content);
