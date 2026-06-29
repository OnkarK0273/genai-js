import { config } from "../../config/index.js";
import Groq from "groq-sdk";

const groq = new Groq({ apiKey: config.groq_api_key });

async function llm_calling() {
  try {
    const res = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [
        {
          role: "system",
          content:
            "You are an elite sentiment analysis engine. Classify reviews strictly into positive, negative, or neutral categories.",
        },
        {
          role: "user",
          content: "Review: Very good phone with an excellent camera.",
        },
      ],
    });

    const ans = res.choices[0]?.message;
    console.log("Sentiment Analysis Result:", ans?.content);
  } catch (error) {
    console.error("Inference Error:", error);
  }
}

llm_calling();
