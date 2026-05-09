import { Hono } from "hono";
import { server } from "./mcp"; // Your MCP server instance
import { serve } from "@hono/node-server";
import { StreamableHTTPTransport } from "@hono/mcp";

const app = new Hono();
const transport = new StreamableHTTPTransport();

app.get("/", (c) => c.json({ message: "MCP Server is running" }));

// 1. The SSE endpoint (The IDE connects here first)
app.all("/mcp", async (c) => {
  if (!server.isConnected()) {
    // Connect the mcp with the transport
    await server.connect(transport);
  }

  return transport.handleRequest(c);
});

serve({
  fetch: app.fetch,
  port: 8787,
});
