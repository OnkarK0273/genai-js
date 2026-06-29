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

const rawResult = JSON.parse(response.choices[0]?.message.content || "{}");
const result = supportTicketSchema.parse(rawResult);
console.log(result);
