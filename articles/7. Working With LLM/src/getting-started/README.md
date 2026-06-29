# Part 1: Getting Started & Mastering LLM Control

## 1. Introduction to Groq API & Initial Setup

To master programmatic control over Large Language Models (LLMs), you must first initialize an instance of the `Groq` SDK client by securely configuring your application with an environment API key. This setup utilizes an insulated configuration import to eliminate explicit credential hardcoding in your repository.

The primary mechanism for generating completions is the `groq.chat.completions.create` runtime call. Here, you explicitly declare your processing model target—such as `llama-3.1-8b-instant`—and pass an array of conversational messages containing the user payload.

```tsx
// src/simple_completion.ts
import { Config } from "./config/index.js";
import Groq from "groq-sdk";

const groq = new Groq({ apiKey: Config.api_key });

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

    const ans = res.choices[0].message;
    console.log("Response:", ans.content);
  } catch (error) {
    console.error("Inference Error:", error);
  }
}

llm_calling();
```

## 2. Steering Behavior with System Prompts

System prompts act as fundamental behavioral guardrails for an LLM before the user payload is ever parsed. By appending an object with the role explicitly defined as "system", you redefine how the engine interprets subsequent runtime interactions.

In this implementation, the system prompt defines the context and persona boundaries: instructing the model that it specializes exclusively in categorical sentiment analysis. When the user role provides a raw review string, the model is already bound to those systemic behavioral rules.

```tsx
// src/system_prompt.ts
import { Config } from "./config/index.js";
import Groq from "groq-sdk";

const groq = new Groq({ apiKey: Config.api_key });

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

    const ans = res.choices[0].message;
    console.log("Sentiment Analysis Result:", ans.content);
  } catch (error) {
    console.error("Inference Error:", error);
  }
}

llm_calling();
```

## 3. Fine-Tuning Generations with LLM Settings

To achieve fine-grained control over model output predictability, length, and token penalties, core configuration parameters can be tuned directly inside the inference request payload.

| Parameter               | Type / Value       | Core Production Utility                                                                  |
| :---------------------- | :----------------- | :--------------------------------------------------------------------------------------- |
| `temperature`           | `0` to `2`         | Adjusts output predictability. Set strictly to `0` for absolute engineering determinism. |
| `stop`                  | `string` / `array` | Defines specific character sequences that instantly halt token generation.               |
| `max_completion_tokens` | `number`           | Bounds execution resource expenditure by capping total response token lengths.           |
| `frequency_penalty`     | `-2.0` to `2.0`    | Discourages the model from repeating the exact same words verbatim.                      |
| `presence_penalty`      | `-2.0` to `2.0`    | Encourages the model to introduce completely new topics into the pipeline.               |

Modifying these parameters alters the deterministic nature of your completion streams:

```typescript
// src/tuned_completion.ts
import { Config } from "./config/index.js";
import Groq from "groq-sdk";

const groq = new Groq({ apiKey: Config.api_key });

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

  console.log("Tuned Response:", res.choices[0].message.content);
}
```

# Next Articles in This Series

1. Production-Ready AI: Guaranteeing Structured JSON Outputs
