# Part 2: Production-Ready AI: Guaranteeing Structured JSON Outputs

## 1. Structured Output with Native JSON Object Mode

When building production-ready pipelines, applications often require the LLM's response to be delivered as a parsed data structure rather than plain text. Setting `response_format` to `{ type: "json_object" }` forces the model engine to return a valid JSON string.

To ensure stability in this mode, you must explicitly instruct the model inside the system prompt to output exclusively in JSON and outline the expected keys (`sentiment_analysis`, `sentiment`, `confidence_score`, `key_phrases`, `summary`). Once received, the response content can safely undergo standard `JSON.parse()` transformation.

```tsx
import { Groq } from "groq-sdk";

const groq = new Groq();

async function main() {
  const response = await groq.chat.completions.create({
    model: "openai/gpt-oss-20b",
    messages: [
      {
        role: "system",
        content: `You are a data analysis API that performs sentiment analysis on text.
                Respond only with JSON using this format:
                {
                    "sentiment_analysis": {
                    "sentiment": "positive|negative|neutral",
                    "confidence_score": 0.95,
                    "key_phrases": [
                        {
                        "phrase": "detected key phrase",
                        "sentiment": "positive|negative|neutral"
                        }
                    ],
                    "summary": "One sentence summary of the overall sentiment"
                    }
                }`,
      },
      {
        role: "user",
        content:
          "Analyze the sentiment of this customer review: 'I absolutely love this product! The quality exceeded my expectations, though shipping took longer than expected.'",
      },
    ],
    response_format: { type: "json_object" },
  });

  const result = JSON.parse(response.choices[0].message.content || "{}");
  console.log(result);
}

main();
```

## 2. Type-Safe Validation Using Zod Schemas

While JSON Object Mode guarantees that the string layout is valid JSON, it does not programmatically ensure that the keys, array items, and primitive data types strictly align with backend application requirements. To build runtime type-safety, a Zod schema (`supportTicketSchema`) can be defined to encapsulate strict data modeling properties (e.g., `category`, `priority`, `urgency_score`, nested objects like `customer_info`, components arrays, boolean flags, and string descriptions).

By passing the evaluated schema structure via `z.toJSONSchema(supportTicketSchema)` into the `json_schema` response format option under `response_format: { type: "json_schema", ... }`, the processing model (such as `moonshotai/kimi-k2-instruct-0905`) is constrained to generate an output matching this architecture. After parsing the text output into a raw JavaScript object, calling `supportTicketSchema.parse(rawResult)` provides static type inference (`z.infer<typeof supportTicketSchema>`) and runtime structural verification.

```tsx
import Groq from "groq-sdk";
import { z } from "zod";

const groq = new Groq();

const supportTicketSchema = z.object({
  category: z.enum([
    "api",
    "billing",
    "account",
    "bug",
    "feature_request",
    "integration",
    "security",
    "performance",
  ]),
  priority: z.enum(["low", "medium", "high", "critical"]),
  urgency_score: z.number(),
  customer_info: z.object({
    name: z.string(),
    company: z.string().optional(),
    tier: z.enum(["free", "paid", "enterprise", "trial"]),
  }),
  technical_details: z.array(
    z.object({
      component: z.string(),
      error_code: z.string().optional(),
      description: z.string(),
    }),
  ),
  keywords: z.array(z.string()),
  requires_escalation: z.boolean(),
  estimated_resolution_hours: z.number(),
  follow_up_date: z.string().datetime().optional(),
  summary: z.string(),
});

type SupportTicket = z.infer<typeof supportTicketSchema>;

const response = await groq.chat.completions.create({
  model: "moonshotai/kimi-k2-instruct-0905",
  messages: [
    {
      role: "system",
      content: `You are a customer support ticket classifier for SaaS companies.
                Analyze support tickets and categorize them for efficient routing and resolution.
                Output JSON only using the schema provided.`,
    },
    {
      role: "user",
      content: `Hello! I love your product and have been using it for 6 months.
                I was wondering if you could add a dark mode feature to the dashboard?
                Many of our team members work late hours and would really appreciate this.
                Also, it would be great to have keyboard shortcuts for common actions.
                Not urgent, but would be a nice enhancement!
                Best, Mike from StartupXYZ`,
    },
  ],
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "support_ticket_classification",
      schema: z.toJSONSchema(supportTicketSchema),
    },
  },
});

const rawResult = JSON.parse(response.choices[0].message.content || "{}");
const result = supportTicketSchema.parse(rawResult);
console.log(result);
```

## 3. Standard JSON-Schema Enforcement

For environments where external validation libraries like Zod are not required, standard native JSON-Schema configurations can be passed explicitly inside the `json_schema.schema` configuration object. This workflow directly sets properties (`confidence`, `accuracy`, `pass`), value parameter boundaries (`minimum: 1`, `maximum: 10`), explicit description strings (`"1-10 scale"`, `"true or false"`), data primitives (`number`, `boolean`), and an array listing the `required` properties.

By declaring `additionalProperties: false`, the integration explicitly instructs the LLM implementation to omit any additional data outside of the strictly provided schema fields. This enables clean execution tracking for analytical tasks such as evaluating coding questions across metrics like accuracy and passing state.

```tsx
import { Config } from "./config/index.js";

import Groq from "groq-sdk";

const groq = new Groq({ apiKey: Config.api_key });
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

    const result = response.choices[0].message.content || "{}";
    const parsed = JSON.parse(result);
    console.log(JSON.stringify(parsed, null, 2));
  } catch (error) {
    console.log("error-", error);
  }
}

llm_calling();
```

# Next articles in this series

1. Building a Smart Agent: Tool Calling & CLI Chatbots
