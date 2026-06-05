# Three core concepts while building MCP

1.  `Resources` - it is file like data that can read by clients, and provided to llm
2.  `Tools` - Function that perform task can be called by llm
3.  `Prompts` - pre-written template that user can re-use to perform a task

# How to build MCP server

## 1\. Local mcp server

![](https://cdn.hashnode.com/uploads/covers/662e9149ea7b8adaf16495b0/77fbf212-54fa-4828-a9ee-906fd6077d76.png)

### 1\. **Setting Up Your Environment**

- Create and set up our project:

  ```shell
  # Create project directory
  mkdir mcp-local
  cd mcp-local

  # Initialize npm project
  npm init -y

  # Install dependencies
  npm install zod @modelcontextprotocol/sdk

  # Install dev dependencies
  npm install -D @types/node typescript

  # Create src dir
  mkdir src
  cd src

  #create index file
  touch index.ts
  ```

- Update `package.json`

  ```typescript
  {
    "type": "module",
    "scripts": {
      "build": "tsc",
      "start": "node dist/index.js"
    }
  }
  ```

- Create a `tsconfig.json` in the root of your project

  ```typescript
  {
    "compilerOptions": {
      "target": "ES2022",
      "module": "Node16",
      "moduleResolution": "Node16",
      "outDir": "./dist",
      "rootDir": "./src",
      "strict": true,
      "esModuleInterop": true,
      "skipLibCheck": true,
      "forceConsistentCasingInFileNames": true
    },
    "include": ["src/**/*"],
    "exclude": ["node_modules"]
  }

  ```

### 2\. Creating local mcp server

create a `src/index.ts` file

1.  **McpServer instance:**

    Here we create instance of `McpServer` by providing name and version

    ```typescript
    //src/index.ts

    import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

    // Create server instance
    const server = new McpServer({
      name: "Student",
      version: "1.0.0",
    });
    ```

2.  **Resources: The "Read-Only" Data:**

    There is a method called `server.registerResource` it is acts like a file or a URL that the AI can "read" to get background information.

    The following things we provide to `server.registerResource` method
    1.  name - `"greeting-resource"`
    2.  uriOrTemplate - `"https://example.com/greetings/default"`
    3.  config - resource meta data `title` , `description` and `mimeType`
    4.  callback - it return object of `uri` and `text`

    ```typescript
    //src/index.ts

    import { ReadResourceResult } from "@modelcontextprotocol/sdk/types";

    server.registerResource(
      "greeting-resource",
      "https://example.com/greetings/default",
      {
        title: "Default Greeting", // Display name for UI
        description: "A simple greeting resource",
        mimeType: "text/plain",
      },
      async (): Promise<ReadResourceResult> => {
        return {
          contents: [
            {
              uri: "https://example.com/greetings/default",
              text: "Hello! I'm, your virtual assistant. I'm here to help you streamline your workflow and answer any questions you might have about. How can I assist you today?",
            },
          ],
        };
      },
    );
    ```

3.  **Prompts: The "Templates”**

    The `server.registerPrompt` section provides pre-defined templates that user can reuse to perform a task

    ```typescript
    //src/index.ts

    import { z } from "zod";
    server.registerPrompt(
      "student_list",
      {
        title: "student list",
        description: "A sample prompt to get list of student",
        argsSchema: {
          limit: z.string().describe("no between 1 to 9"),
        },
      },
      async ({ limit }): Promise<GetPromptResult> => {
        return {
          messages: [
            {
              role: "user",
              content: {
                type: "text",
                text: `give me list of student, give only ${limit} students`,
              },
            },
          ],
        };
      },
    );
    ```

4.  **Tools: The "Action" Functions**

    The `server.registerTool` section is the most powerful part. It allows the AI to **execute code** to get real-time data.

    ```typescript
    //src/index.ts

    import { z } from "zod";
    server.registerTool(
      "get_student",
      {
        description: "Get list of all student with thier enrolnment no.",
        inputSchema: {
          limit: z.number().describe("give it in 1 to 10 numbers"),
        },
      },
      async ({ limit }) => {
        const student = [
          {
            id: "STU-001",
            name: "Elena Rodriguez",
            age: 20,
            major: "Marine Biology",
            gpa: 3.85,
            email: "elena.r@university.edu",
            enrolled: true,
          },
          {
            id: "STU-002",
            name: "Marcus Chen",
            age: 22,
            major: "Cybersecurity",
            gpa: 3.92,
            email: "m.chen99@university.edu",
            enrolled: true,
          },
          {
            id: "STU-003",
            name: "Sarah Jenkins",
            age: 19,
            major: "Graphic Design",
            gpa: 3.4,
            email: "s.jenkins@university.edu",
            enrolled: true,
          },
          {
            id: "STU-004",
            name: "Amara Okafor",
            age: 21,
            major: "Mechanical Engineering",
            gpa: 3.78,
            email: "a.okafor@university.edu",
            enrolled: false,
          },
        ];

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(student.slice(0, limit)),
            },
          ],
        };
      },
    );
    ```

5.  **Main Execution and Communication Transport**

    We create `StdioServerTransport` and connect to `McpServer`instance to start listen through stdio transport.

    ```typescript
    //src/index.ts

    import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
    async function main() {
      const transport = new StdioServerTransport();
      await server.connect(transport);
      console.error("Student MCP Server running on stdio");
    }

    main().catch((error) => {
      console.error("Fatal error in main():", error);
      process.exit(1);
    });
    ```

<details data-node-type="hn-details-summary">
<summary>Complete Code Source</summary>
<pre class="not-prose"><code class="language-typescript">// src/index.ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  GetPromptResult,
  ReadResourceResult,
} from "@modelcontextprotocol/sdk/types";
import { z } from "zod";

// Create server instance
const server = new McpServer({
name: "Student",
version: "1.0.0",
});

server.registerResource(
"greeting-resource",
"https://example.com/greetings/default",
{
title: "Default Greeting", // Display name for UI
description: "A simple greeting resource",
mimeType: "text/plain",
},
async (): Promise&lt;ReadResourceResult&gt; =&gt; {
return {
contents: [
{
uri: "https://example.com/greetings/default",
text: "Hello! I'm, your virtual assistant. I'm here to help you streamline your workflow and answer any questions you might have about. How can I assist you today?",
},
],
};
},
);

server.registerPrompt(
"student_list",
{
title: "student list",
description: "A sample prompt to get list of student",
argsSchema: {
limit: z.string().describe("no between 1 to 9"),
},
},
async ({ limit }): Promise&lt;GetPromptResult&gt; =&gt; {
return {
messages: [
{
role: "user",
content: {
type: "text",
text: `give me list of student, give only ${limit} students`,
},
},
],
};
},
);

server.registerTool(
"get_student",
{
description: "Get list of all student with thier enrolnment no.",
inputSchema: {
limit: z.number().describe("give it in 1 to 10 numbers"),
},
},
async ({ limit }) =&gt; {
const student = [
{
id: "STU-001",
name: "Elena Rodriguez",
age: 20,
major: "Marine Biology",
gpa: 3.85,
email: "elena.r@university.edu",
enrolled: true,
},
{
id: "STU-002",
name: "Marcus Chen",
age: 22,
major: "Cybersecurity",
gpa: 3.92,
email: "m.chen99@university.edu",
enrolled: true,
},
{
id: "STU-003",
name: "Sarah Jenkins",
age: 19,
major: "Graphic Design",
gpa: 3.4,
email: "s.jenkins@university.edu",
enrolled: true,
},
{
id: "STU-004",
name: "Amara Okafor",
age: 21,
major: "Mechanical Engineering",
gpa: 3.78,
email: "a.okafor@university.edu",
enrolled: false,
},
{
id: "STU-005",
name: "Liam O'Connor",
age: 23,
major: "Philosophy",
gpa: 3.15,
email: "liam.oc@university.edu",
enrolled: true,
},
{
id: "STU-006",
name: "Sofia Rossi",
age: 20,
major: "Architecture",
gpa: 3.65,
email: "s.rossi@university.edu",
enrolled: true,
},
{
id: "STU-007",
name: "Julian Vance",
age: 21,
major: "Finance",
gpa: 3.5,
email: "vance.j@university.edu",
enrolled: true,
},
{
id: "STU-008",
name: "Chloe Kim",
age: 19,
major: "Psychology",
gpa: 4.0,
email: "chloe.kim@university.edu",
enrolled: true,
},
{
id: "STU-009",
name: "David Miller",
age: 24,
major: "History",
gpa: 2.95,
email: "d.miller@university.edu",
enrolled: false,
},
{
id: "STU-010",
name: "Ayesha Khan",
age: 22,
major: "Bio-Chemistry",
gpa: 3.72,
email: "a.khan@university.edu",
enrolled: true,
},
];

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(student.slice(0, limit)),
        },
      ],
    };

},
);

async function main() {
const transport = new StdioServerTransport();
await server.connect(transport);
console.error("Student MCP Server running on stdio");
}

main().catch((error) =&gt; {
console.error("Fatal error in main():", error);
// process.exit(1);
});

</code></pre>

</details>

## 2\. Remote MCP Server

![](https://cdn.hashnode.com/uploads/covers/662e9149ea7b8adaf16495b0/17ef19a0-5d71-4aed-84c6-51fa318b80f4.png)

### 1\. **Setting Up Your Environment**

- Create and set up our project:

  ```shell
  # Create project directory
  mkdir mcp-remote
  cd mcp-remote

  # Initialize npm project
  npm init -y

  # Install dependencies
  npm install zod @modelcontextprotocol/sdk hono @hono/mcp @hono/node-server

  # Install dev dependencies
  npm install -D @types/node typescript

  # Create src dir
  mkdir src
  cd src

  #create index file
  touch index.ts
  touch app.ts
  ```

- Update `package.json`

  ```typescript
  {
    "type": "module",
    "scripts": {
      "build": "tsc",
      "start": "node dist/index.js"
    }
  }
  ```

- Create a `tsconfig.json` in the root of your project

  ```typescript
  {
    "compilerOptions": {
      "target": "ES2022",
      "module": "Node16",
      "moduleResolution": "Node16",
      "outDir": "./dist",
      "rootDir": "./src",
      "strict": true,
      "esModuleInterop": true,
      "skipLibCheck": true,
      "forceConsistentCasingInFileNames": true
    },
    "include": ["src/**/*"],
    "exclude": ["node_modules"]
  }

  ```

### 2\. Creating mcp remote serve

In remote MCP Server the steps for creating Resources, Prompts and Tools are same as local mcp server, only difference in communication part.

### Communication Transport

create `app.ts` file

1.  **Setting up the Web Server**

    ```typescript
    //src/app.ts

    import { StreamableHTTPTransport } from "@hono/mcp";
    import { Hono } from "hono";

    const app = new Hono();
    const transport = new StreamableHTTPTransport();
    ```

    - `Hono` - we use hono for handling HTTP request, it is blazing fast framework
    - `StreamableHTTPTransport` - this is HTTP Transport Layer for remote mcp server and also it leverage SSE for streaming data.

2.  **MCP end point**

    In the `/mcp` endpoint where actual transport happens.

    ```typescript
    //src/app.ts

    import { server } from ".";

    app.all("/mcp", async (c) => {
      if (!server.isConnected()) {
        await server.connect(transport);
      }

      return transport.handleRequest(c);
    });
    ```

    - `app.all()` - This listen all HTTP methods `GET POST PATCH DELETE etc`
    - `server.connect` - we connect transport to server at 1st time of connection.
    - `transport.handleRequest(c)` - it takes requests from server

3.  **Listening the server**

    We listen the remote mcp server on port `8787`

    ```typescript
    //src/app.ts

    import { serve } from "@hono/node-server";
    serve({
      fetch: app.fetch,
      port: 8787,
    });
    ```

# How to Debug **MCP Inspector**

Now we build local/remote mcp servers to test or debug the mcp server there is official library provided by modelcontextprotocol i.e `MCP Inspector`

This provides interactive developer tool UI which is used for testing and debugging MCP servers very easily.

Here are steps of how to use this tool:

### Start the Interactive development UI server

```shell
npx @modelcontextprotocol/inspector
```

Interactive development UI server get started on port `http://localhost:6274` it is look like this:

![](https://cdn.hashnode.com/uploads/covers/662e9149ea7b8adaf16495b0/500bc734-d17a-4d5d-91f7-f7d3aa5b25a1.png)

### **Server connection panel**

Left side of UI is the server connection panel and following things are required to connect local/remote server

1.  MCP local server
    1.  Transport type - `STDIO`
    2.  Command - `node`
    3.  Arguments - `path/to/server/index.js`

2.  MCP remote server
    1.  Transport type - `Stremable HTTP`
    2.  URL - `http://localhost:8787/mcp`
    3.  Connection Type - `Via Proxy`

![](https://cdn.hashnode.com/uploads/covers/662e9149ea7b8adaf16495b0/426fbe78-4369-4e0c-924e-a89fe98d9b78.png)

### **Resources tab**

![](https://cdn.hashnode.com/uploads/covers/662e9149ea7b8adaf16495b0/c3190f39-5415-4ac0-a06d-607a336a0f38.png)

- Lists all available resources
- Shows resource metadata (MIME types, descriptions)
- Allows resource content inspection
- Supports subscription testing

### **Prompts tab**

![](https://cdn.hashnode.com/uploads/covers/662e9149ea7b8adaf16495b0/0f7528fa-8e8c-4b20-97fd-906ef950e588.png)

- Displays available prompt templates
- Shows prompt arguments and descriptions
- Enables prompt testing with custom arguments
- Previews generated messages

### **Tools tab**

![](https://cdn.hashnode.com/uploads/covers/662e9149ea7b8adaf16495b0/81bd8775-f190-426a-84fc-4f5f6aba4b49.png)

- Lists available tools
- Shows tool schemas and descriptions
- Enables tool testing with custom inputs
- Displays tool execution results

# Resources

1.  MCP Official Document - [Document](https://modelcontextprotocol.io/docs/develop/build-server)
2.  How To Build Local MCP Server Source Code - [GitHub](https://github.com/OnkarK0273/genai-js/tree/main/articles/6.MCP/mcp-local-server)
3.  How To Build Remote MCP Server Source Code - [GitHub](https://github.com/OnkarK0273/genai-js/tree/main/articles/6.MCP/mcp-remote-server)

# Read more

1.  Gen AI Using JS - [Github](https://github.com/OnkarK0273/genai-js)
