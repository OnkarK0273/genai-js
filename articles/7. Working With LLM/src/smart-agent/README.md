# Part 3: Building a Smart Agent: Tool Calling & CLI Chatbots

## 1. Implementing Functional Tool Calling with Tavily

To build an LLM agent capable of answering real-time questions, you can combine the Groq SDK with an external data tool like the Tavily API. This mechanism enables native **Tool Calling** (or function calling), where the model acts as a decision-making engine to determine when it needs outside information.

### The Agent Loop Mechanics

- **Tool Definition**: Inside the `tools` array parameter, the `serchQuery` function is explicitly described using a JSON schema format so the model understands its purpose and required arguments.
- **The Agentic While-Loop**: A `while (true)` execution cycle manages the decision flow.
- **Tool Execution Handling**: When `response.choices[0].message.tool_calls` is populated, the model yields execution, expecting the application to run the actual function (`serchQuery`).
- **State Management**: The application executes the Tavily search, extracts relevant webpage content chunks, maps them to a role type of `"tool"`, and pushes the result back into the `messgaeHistory` array. The model processes this newly appended context and generates the final answer.

```tsx
import { Config } from "./config/index.js";
import Groq from "groq-sdk";
import { tavily } from "@tavily/core";

const tvly = tavily({ apiKey: Config.tvly_api_key });
const model = new Groq({ apiKey: Config.api_key });

async function llm_calling() {
  const messgaeHistory = [
    {
      role: "system",
      content: `you are personal assitant who resolves user query
                available tools:
                serchQuery({query}:{query:string})
                description - this tool gives current information on internet by serch
            `,
    },
    {
      role: "user",
      content: "what is the current weather in kolhapur",
    },
  ];
  try {
    while (true) {
      const response = await model.chat.completions.create({
        model: "llama-3.1-8b-instant",
        messages: messgaeHistory,
        tools: [
          {
            type: "function",
            function: {
              name: "serchQuery",
              description:
                "this tool gives current information on internet by serch",
              parameters: {
                type: "object",
                properties: {
                  query: {
                    type: "string",
                    description: "serch query for to get info from internet",
                  },
                },
                required: ["query"],
              },
            },
          },
        ],
        tool_choice: "auto",
      });

      const toolCalling = response.choices[0].message.tool_calls;
      messgaeHistory.push(response.choices[0].message);

      if (!toolCalling) {
        console.log(response.choices[0].message.content);
        break;
      }

      for (const tool of toolCalling) {
        const funName = tool.function.name;
        const funArg = tool.function.arguments;

        if (funName == "serchQuery") {
          const toolRes = await serchQuery(JSON.parse(funArg));
          messgaeHistory.push({
            tool_call_id: tool.id,
            role: "tool",
            name: funName,
            content: toolRes,
          });
        }
      }
    }
  } catch (error) {
    console.log("error-", error);
  }
}

llm_calling();

async function serchQuery({ query }) {
  console.log("tool called enabled");
  const tvlyResponce = await tvly.search(query);

  const info = tvlyResponce.results.map((el) => el.content).join("\n");

  return info;
}
```

## 2. Creating an Interactive Terminal Chatbot with Conversational Memory

To upgrade the tool-calling workflow into a continuous chat experience, you can wrap the script in an interactive loop using Node.js's native `node:readline/promises` module. This architecture introduces conversational state preservation, allowing the user to follow up on previous answers.

### Interactive State Orchestration

- **Double Loop Pattern**:
  1. The **Outer Loop** continuously awaits fresh inputs from standard input via `rl.question("You: ")` and exits cleanly if the termination keyword `"bye"` is registered.
  2. The **Inner Loop** processes model generation and tool resolution cycles for that specific question, breaking out only when the assistant responds with plain text content instead of an additional tool request.
- **Contextual Anchor**: Providing a dynamic date variable (`new Date().toUTCString()`) inside the system prompt lets the model calculate relative temporal requests accurately (such as "yesterday" or "this week") across multiple turns of conversation.

```tsx
import { Config } from "./config/index.js";
import Groq from "groq-sdk";
import { tavily } from "@tavily/core";
import readline from "node:readline/promises";
const tvly = tavily({ apiKey: Config.tvly_api_key });
const model = new Groq({ apiKey: Config.api_key });

async function llm_calling() {
  try {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    const messgaeHistory = [
      {
        role: "system",
        content: `you are personal assitant who resolves user query
                    current data - ${new Date().toUTCString()}
                    available tools:
                    serchQuery({query}:{query:string})
                    description - this tool gives current information on internet by serch
                 `,
      },
    ];

    while (true) {
      const quetion = await rl.question("You: ");

      if (quetion === "bye") {
        break;
      }

      messgaeHistory.push({
        role: "user",
        content: quetion,
      });

      while (true) {
        const response = await model.chat.completions.create({
          model: "llama-3.1-8b-instant",
          messages: messgaeHistory,
          tools: [
            {
              type: "function",
              function: {
                name: "serchQuery",
                description:
                  "this tool gives current information on internet by serch",
                parameters: {
                  type: "object",
                  properties: {
                    query: {
                      type: "string",
                      description: "serch query for to get info from internet",
                    },
                  },
                  required: ["query"],
                },
              },
            },
          ],
          tool_choice: "auto",
        });

        const toolCalling = response.choices[0].message.tool_calls;
        messgaeHistory.push(response.choices[0].message);

        if (!toolCalling) {
          console.log("Assistant: ", response.choices[0].message.content);
          break;
        }

        for (const tool of toolCalling) {
          const funName = tool.function.name;
          const funArg = tool.function.arguments;

          if (funName == "serchQuery") {
            const toolRes = await serchQuery(JSON.parse(funArg));
            messgaeHistory.push({
              tool_call_id: tool.id,
              role: "tool",
              name: funName,
              content: toolRes,
            });
          }
        }
      }
    }

    rl.close();
  } catch (error) {
    console.log("error-", error);
  }
}

llm_calling();

async function serchQuery({ query }) {
  console.log("tool called enabled");
  const tvlyResponce = await tvly.search(query);

  const info = tvlyResponce.results.map((el) => el.content).join("\n");

  return info;
}
```
