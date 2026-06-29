import Groq from "groq-sdk";
import { config } from "../../config/index.js";

const groq = new Groq({ apiKey: config.groq_api_key });
async function llm_calling() {
  try {
    const response = await groq.chat.completions.create({
      model: "openai/gpt-oss-20b",
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "interview_evaluation",
          schema: {
            type: "object",
            properties: {
              confidence: {
                type: "number",
                minimum: 1,
                maximum: 10,
                description: "1-10 scale",
              },
              accuracy: {
                type: "number",
                minimum: 1,
                maximum: 10,
                description: "1-10 scale",
              },
              pass: {
                type: "boolean",
                description: "true or false",
              },
            },
            required: ["confidence", "accuracy", "pass"],
            additionalProperties: false,
          },
        },
      },
      messages: [
        {
          role: "system",
          content: `You are an interview grader assistant. Your task is to evaluate candidate responses to JavaScript interview questions.

        Provide scores based on:
        - confidence: How confident the candidate seems in their answer (1-10 scale)
        - accuracy: How technically correct the answer is (1-10 scale)
        - pass: Whether the candidate should pass this interview question (true/false)
        `,
        },
        {
          role: "user",
          content: `Q: What does === do in JavaScript?
            A: It checks strict equality-both value and type must match.

            Q: How do you create a promise that resolves after 1 second?
            A: const promise = new Promise((resolve) => setTimeout(resolve, 1000));

            Q: What is hoisting?
            A: JavaScript moves declarations (but not initialization) to the top of their scope before code runs.

            Q: Why use let instead of var?
            A: let is block scoped, avoiding the function scope quirks and re-declaration issues of var.
        `,
        },
      ],
    });

    const result = response.choices[0]?.message.content || "{}";
    const parsed = JSON.parse(result);
    console.log(JSON.stringify(parsed, null, 2));
  } catch (error) {
    console.log("error-", error);
  }
}

llm_calling();
