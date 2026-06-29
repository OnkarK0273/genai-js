import Groq from "groq-sdk";
import { config } from "../../config/index.js";

const groq = new Groq({ apiKey: config.groq_api_key });

async function execution_pipeline() {
  const res = await groq.chat.completions.create({
    model: "llama-3.1-8b-instant",
    temperature: 0,
    stop: ["ga"],
    max_completion_tokens: 100,
    frequency_penalty: 0.5,
    presence_penalty: 0.0,
    messages: [
      {
        role: "user",
        content: "Hello LLM",
      },
    ],
  });

  console.log("Tuned Response:", res.choices[0]?.message.content);
}

execution_pipeline();
