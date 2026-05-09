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

main().catch((error) => {
  console.error("Fatal error in main():", error);
  // process.exit(1);
});
