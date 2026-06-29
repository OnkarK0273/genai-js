import Groq from "groq-sdk";
import { config } from "../../config/index.js";

const groq = new Groq({ apiKey: config.groq_api_key });

async function llm_calling() {
  try {
    const res = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [
        {
          role: "user",
          content: "Hello LLM",
        },
      ],
    });

    const ans = res.choices[0]?.message;
    console.log("Response:", ans?.content);
  } catch (error) {
    console.error("Inference Error:", error);
  }
}

llm_calling();
