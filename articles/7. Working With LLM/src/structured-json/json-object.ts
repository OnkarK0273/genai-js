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

  const result = JSON.parse(response.choices[0]?.message.content || "{}");
  console.log(result);
}

main();
