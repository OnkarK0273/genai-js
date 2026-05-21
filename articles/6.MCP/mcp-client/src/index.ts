import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport";
import { Groq } from "groq-sdk/client.js";
import readline from "readline/promises";

const GROQ_API_KEY = process.env.GROQ_API_KEY;
if (!GROQ_API_KEY) {
  throw new Error("GROQ_API_KEY is not set");
}

class MCPClient {
  private mcp: Client;
  private groq: Groq;
  private transport:
    | StdioClientTransport
    | StreamableHTTPClientTransport
    | null = null;
  private tools: Groq.Chat.Completions.ChatCompletionTool[] = [];

  constructor() {
    this.groq = new Groq({
      apiKey: GROQ_API_KEY,
    });
    this.mcp = new Client({ name: "mcp-client-cli", version: "1.0.0" });
  }

  async connectToServer(
    serverScriptPath: string,
    serverType: "local" | "remote",
  ) {
    try {
      if (serverType === "local") {
        const isJs = serverScriptPath.endsWith(".js");
        const isPy = serverScriptPath.endsWith(".py");
        if (!isJs && !isPy) {
          throw new Error("Server script must be a .js or .py file");
        }
        const command = isPy
          ? process.platform === "win32"
            ? "python"
            : "python3"
          : process.execPath;

        this.transport = new StdioClientTransport({
          command,
          args: [serverScriptPath],
        });
      } else if (serverType === "remote") {
        const url = new URL(serverScriptPath);
        this.transport = new StreamableHTTPClientTransport(url);
      }

      await this.mcp.connect(this.transport as Transport);

      const toolsResult = await this.mcp.listTools();

      console.log("tool result -", toolsResult);
      this.tools = toolsResult.tools.map((tool) => {
        return {
          type: "function",
          function: {
            name: tool.name,
            description: tool.description as string,
            parameters: tool.inputSchema,
          },
        };
      });
      console.log(
        "Connected to server with tools:",
        this.tools.map((tool) => tool.function?.name),
      );
    } catch (e) {
      console.log("Failed to connect to MCP server: ", e);
      throw e;
    }
  }

  async processQuery(query: string) {
    const messages: Groq.Chat.ChatCompletionMessageParam[] = [
      {
        role: "user",
        content: query,
      },
    ];

    const response = await this.groq.chat.completions.create({
      model: "openai/gpt-oss-120b",
      messages,
      tools: this.tools,
    });

    const finalText = [];

    const choise = response.choices[0];

    const assistantMessage = choise?.message;

    if (assistantMessage?.content) {
      finalText.push(assistantMessage.content);
    }

    if (
      assistantMessage?.tool_calls &&
      assistantMessage.tool_calls.length > 0
    ) {
      messages.push(assistantMessage);

      for (const toolCalls of assistantMessage.tool_calls) {
        if (toolCalls.type !== "function") continue;

        const toolName = toolCalls.function.name;

        const toolArgs = JSON.parse(toolCalls.function.arguments);

        finalText.push(`[Calling tool ${toolName} with args ${toolArgs}]`);

        const result = await this.mcp.callTool({
          name: toolName,
          arguments: toolArgs,
        });

        messages.push({
          role: "tool",
          tool_call_id: toolCalls.id,
          content: JSON.stringify(result.content),
        });
      }

      const followupResponse = await this.groq.chat.completions.create({
        model: "openai/gpt-oss-120b",
        messages,
      });

      if (followupResponse.choices[0]?.message.content) {
        finalText.push(followupResponse.choices[0].message.content);
      }
    }
    return finalText.join("\n");
  }

  async chatLoop() {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    try {
      console.log("\nMCP Client Started!");
      console.log("Type your queries or 'quit' to exit.");

      while (true) {
        const message = await rl.question("\nQuery: ");
        if (message.toLowerCase() === "quit") {
          break;
        }
        const response = await this.processQuery(message);
        console.log("\n" + response);
      }
    } finally {
      rl.close();
    }
  }

  async cleanup() {
    await this.mcp.close();
  }
}

async function main() {
  if (process.argv.length < 3) {
    console.log("Usage: node index.ts <path_to_server_script>");
    return;
  }
  const mcpClient = new MCPClient();
  try {
    await mcpClient.connectToServer(process.argv[2] as string, "remote");
    await mcpClient.chatLoop();
  } catch (e) {
    console.error("Error:", e);
    await mcpClient.cleanup();
    process.exit(1);
  } finally {
    await mcpClient.cleanup();
    process.exit(0);
  }
}

main();
