// smart-agent/tool-calling.ts
import Groq from "groq-sdk";
import { tavily } from "@tavily/core";
import { config } from "../../config/index.js";

const tvly = tavily({ apiKey: config.tavily_api_key });
const model = new Groq({ apiKey: config.groq_api_key });

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
