# Audio Processing Approach

## 1\. Tradition approach

![](https://cdn.hashnode.com/uploads/covers/662e9149ea7b8adaf16495b0/2b135c46-f1ca-49fc-9cfe-7cd8dc032de5.png)

- Building voice agents using legacy LLMs is notoriously complex. Because traditional models operate strictly within text-in, text-out boundaries, you are forced to stitch together independent subsystems
- To create audio agent workflow we require additional two models 1. STT (speech-to-text) and 2. TTS(text-to-speech) and entire flow of working as per diagram.
- This method works well for prototype based project to test the project but in production grade application. If you aim to deploy a production-grade AI voice agent, this legacy pipeline introduces significant conversational lag (latency) and suffers from compound translation errors across model boundaries.

## 2\. Modern approach

![](https://cdn.hashnode.com/uploads/covers/662e9149ea7b8adaf16495b0/bfc6f3d3-774c-4035-9aa6-af62c7291786.png)

- We can now architect native, real-time voice applications using modern Speech-to-Speech (STS) models, such as Gemini Live and OpenAI Realtime.
- We only require single model i.e STS (speech-to-speech) to create audio workflow for agent as shown in the diagram
- This models can take raw audio input and give raw output audio in real-time using websocket
- Because it using single model for audio processing their overall latency is very low like human type conversation.
- Because it take raw audio input so it can recognize emotion better also produce output better

## Traditional vs. Modern STS Pipelines

| **Dimension**         | **Legacy Pipeline (STT ➡️ LLM ➡️ TTS)**             | **Modern Pipeline (Native STS / Gemini Live)**       |
| --------------------- | --------------------------------------------------- | ---------------------------------------------------- |
| **Model Footprint**   | 3 separate models (STT model, LLM, TTS engine)      | 1 native end-to-end model                            |
| **Average Latency**   | High (2.5s – 5.0s due to sequential execution)      | Sub-second (human-like conversational speed)         |
| **Context Retention** | Low (loses vocal emotion, tone, and inflection)     | High (retains raw audio characteristics and emotion) |
| **Protocol**          | Typically standard REST requests or chunked uploads | Persistent Bi-directional WebSockets                 |

# Project architecture

This is audio agent project architecture using modern audio technology for that we are using `gemini-live` model because they provide generously free tier so it easy to create for everyone

To create this type of agent their are two ways for that

## 1\. Server to server

![](https://cdn.hashnode.com/uploads/covers/662e9149ea7b8adaf16495b0/0f1392c4-b172-4fd5-a554-efef17f93be1.png)

### Frontend to backend streaming

Rather than waiting for the user to finish speaking and sending a bloated, monolithic audio file, the frontend continuously streams raw audio chunks in real-time.

This continuous streaming data achieved through `web-socket` it establish two way persist communication channel between frontend and backend.

### Back-end Processing & Model Communication

Backend received that raw stream data and forward to `gemini-live` multi-model it can accept audio/text/video

### Gemini-Live Response Generation

It process the incoming stream data in real-time and immediately strem back the response in audio/text format.

### Delivering the Feedback Loop

Backend immediately stream `gemini-live` response back to frontend via `web-scoket` connection.

frontend render the text as well play back the audio seamlessly to the user

## 2\. Client to server

![](https://cdn.hashnode.com/uploads/covers/662e9149ea7b8adaf16495b0/76f6099c-34b1-4dce-8b78-357337c08377.png)

This architecture is highly secure and very fast, instead of sending heavy stream data through backend as proxy to `gemini-model`, this pattern uses **Direct Client-to-Service Streaming with Ephemeral Tokens** pattern.

### The Handshake & Authentication

Before streaming voice data, it secures the connection without exposing main api keys.

`Get-Ephemeral-token`: frontend makes the `HTTP` req to backend for temporal token.

`Ephemeral-token-Req`: backend contact `gemini-live` model and request short lived token

`Ephemeral-temp-token`: it issues the token and backend received and pass that token to frontend.

### Direct Web-Socket Establishment

Once the Front-end has that short-lived `Ephemeral-temp-token`, it bypasses the back-end for the heavy lifting

The **Front-end** uses that token to authenticate and open a **direct, bi-directional** `streaming-data (web-socket)` **connection** straight to **Gemini-Live**.

### Real-Time Interaction Loop

With direct two persist connection:

**Upstream (**`audio/text/video`**):** As the user speaks into the microphone, the Front-end streams the multimodal data directly into Gemini-Live.

**Downstream (**`audio/text`**):** Gemini-Live processes it instantly and streams the audio/text response right back to the Front-end for immediate playback.

### What are the benefits of this approach

1.  Highly scalable
2.  very fast
3.  Highly secure

# Reference

Gemini Official Live API - [Document](https://ai.google.dev/gemini-api/docs/live-api)
