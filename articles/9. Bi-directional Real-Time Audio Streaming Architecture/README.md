![](https://cdn.hashnode.com/uploads/covers/662e9149ea7b8adaf16495b0/ff6d7ca9-13c4-405c-8703-92c3599ef142.png)

If you are engineering an interactive AI audio agent that communicates directly with native device microphones and speakers, u can re-use this bi-directional, real-time audio streaming architecture provides the production blueprint.

---

### 🛠️ Tech Stack & Workspace Setup

- **Core API Framework:** Google GenAI SDK (`@google/genai` v1alpha Live API)
- **Runtime Ecosystem:** Node.js / Bun (TypeScript-first)
- **Frontend Layer:** Web Audio API, AudioWorklet API, Native WebSockets
- **Target Environment:** Modern Web Browsers (Chrome, Edge, Safari 14.1+)
- **Complete Workspace Blueprint:** [GitHub Repository](https://github.com/OnkarK0273/genai-js/tree/main/articles/9.%20Bi-directional%20Real-Time%20Audio%20Streaming%20Architecture)

---

# System Architecture Walkthrough

The platform splits processing logic cleanly down the middle into two distinct unidirectional processing pipelines: an upstream **Recording Pipeline** and a downstream **Playback Pipeline**.

## 1\. The Recording Pipeline

Our generative model requires an input stream formatted as raw, 16-bit Signed Linear PCM audio sampled at a stable 16kHz frequency.

### Input Source Node Setup

We begin by requesting hardware authorization to access the client's internal microphone array. Using customized performance constraint parameters, we map this continuous analog signal wave straight into the system's runtime source pipeline.

```typescript
// src/AudioService.ts
// Request access to the user's microphone with specific audio constraints
this.mediaStream = await navigator.mediaDevices.getUserMedia({
  audio: {
    // Sets the audio sampling frequency to 16kHz
    sampleRate: INPUT_SAMPLE_RATE,
    // Requests a single audio channel (mono) to save network bandwidth
    channelCount: 1,
    // Enables echo cancellation to prevent speaker feedback loops
    echoCancellation: true,
    // Reduces background steady-state noise (like fans or room hums)
    noiseSuppression: true,
    // Automatically adjusts microphone volume to keep digital levels consistent
    autoGainControl: true,
  },
});

// Bridge the live microphone stream directly into the Web Audio API context
this.inputSource = this.inputAudioContext.createMediaStreamSource(
  this.mediaStream,
);
```

### Offloading to the Audio Worklet Node

Modern browsers leverage their single main execution thread to orchestrate critical UI rendering phases, listen for user interactions, and handle layout scrolling. Processing continuous, high-frequency raw audio on this main thread inevitably introduces audio stuttering and frame dropouts.

To bypass this bottleneck, we employ an `AudioWorkletNode`. This offloads raw sample extractions to a completely isolated, high-priority background audio thread. The worker processes data blocks smoothly at an execution pace of roughly 125 cycles per second (one evaluation every 8ms), firing arrays back to the application context via its dedicated communications port.

```typescript
// src/worklet/mic-processor.js
class MicProcessor extends AudioWorkletProcessor {
  // Runs ~125 times/sec (every 8ms) at a 16kHz sample rate / 128 buffer size
  process(inputs) {
    // Safety check: ensure input data stream exists
    if (!inputs.length) return true;

    const input = inputs[0]; // Isolate the primary input source
    if (!input.length) return true; // Ensure the target source channel contains data

    const channelData = input[0]; // Extract raw PCM data from the mono track

    // Clone the buffer data array so the main execution thread can process it without mutation
    const pcm = new Float32Array(channelData.length);
    pcm.set(channelData);

    // Dispatch the localized audio data block back to the main application context
    this.port.postMessage(pcm);

    // Keep the audio processor module active
    return true;
  }
}

// Register the custom processor class for global main-thread operations
registerProcessor("mic-processor", MicProcessor);
```

### Compacting the Data Payload: PCM Mapping & Base64 Packing

The `AudioWorkletNode` emits compressed raw audio chunks every 8ms. Attempting to transmit these high-frequency arrays directly over raw network sockets will rapidly saturate the connection pipeline.

To make this data readable for the model, we translate the complex 32-bit decimal strings into standard 16-bit integers and safely map them inside a clamped volume range.

```typescript
// src/audioUtils.ts
export function createPCMBlob(data: Float32Array) {
  // Convert 32-bit float audio [-1.0, 1.0] to 16-bit signed integers [-32768, 32767]
  const int16 = new Int16Array(data.length);

  for (let i = 0; i < data.length; i++) {
    // Clamp the raw input value to ensure it stays strictly within the [-1.0, 1.0] safety boundary
    const element = Math.max(-1, Math.min(1, data[i]));

    // Map float down to Int16 boundaries (negative scales to -32768, positive scales to 32767)
    int16[i] = element < 0 ? element * 32768 : element * 32767;
  }

  return {
    data: arrayBufferToBase64(int16),
    mimeType: "audio/pcm; rate=16000",
  };
}
```

Because passing raw 16-bit binary streams straight through network interfaces risks cross-platform structural character corruption, we serialize the resulting buffer arrays into a secure, URL-safe Base64 alphanumeric text string.

```typescript
// src/audioUtils.ts
function arrayBufferToBase64(data: Int16Array): string {
  // Cast the 16-bit integer buffer down as raw 8-bit bytes
  const bytes = new Uint8Array(data.buffer);

  // Parse raw byte values into a clean binary string format
  let str = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    str += String.fromCharCode(bytes[i]);
  }

  // Safe string translation deployment
  return btoa(str);
}
```

### Establishing the Network Socket Link

Once mapped to a Base64 text string, the runtime continuously pipes these chunks through a persistent, low-latency WebSocket connection established directly with the upstream AI infrastructure.

```typescript
// src/AudioService.ts
import { createPCMBlob } from "@/utils/audioUtils";
import { INPUT_SAMPLE_RATE, MODEL } from "@/config/constants";
import { ConnectConfig } from "@/types";
import { GoogleGenAI, Modality, Session } from "@google/genai";

export class AudioService {
  private ai: GoogleGenAI;
  private activeSession: Session | null = null;
  private inputAudioContext: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;

  constructor(token: string) {
    this.ai = new GoogleGenAI({
      apiKey: token,
      apiVersion: "v1alpha",
    });
  }

  async startSession(connectConfig: ConnectConfig) {
    try {
      const config = {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: {
              voiceName: connectConfig.selected_assistant_voice,
            },
          },
        },
        systemInstruction: "You are a helpful real-time AI audio agent.",
      };

      this.activeSession = await this.ai.live.connect({
        model: MODEL,
        callbacks: {
          onopen: () => console.debug("Live WebSocket Session Established"),
          onmessage: (message) =>
            console.debug("Inbound Engine Payload:", message),
          onerror: (e) =>
            console.debug("WebSocket Session Exception:", e.message),
          onclose: (e) =>
            console.debug("WebSocket Connection Terminated:", e.reason),
        },
        config: config,
      });

      this.inputAudioContext = new AudioContext({
        sampleRate: INPUT_SAMPLE_RATE, // Targeted at 16000Hz
      });

      this.workletNode = new AudioWorkletNode(
        this.inputAudioContext,
        "mic-processor",
      );

      this.workletNode.port.onmessage = (event) => {
        const pcmBlob = createPCMBlob(event.data as Float32Array);
        this.activeSession?.sendRealtimeInput({ audio: pcmBlob });
      };
    } catch (error) {
      console.error("Critical Runtime Failure during Initialization:", error);
    }
  }
}
```

## 2\. The Playback Pipeline

The AI engine transmits an outbound response layer of Base64-encoded, 16-bit PCM audio sampled at a dense 24kHz. To transform these string payloads back into sharp physical audio waves, the playback stack performs an exact reverse processing configuration.

| Step   | Processor / Method      | Data Transformation Type                        | Target Frequency / Standard             |
| ------ | ----------------------- | ----------------------------------------------- | --------------------------------------- |
| **01** | `atob()` Decoder        | Base64 Text String ➡️ Binary `Uint8Array`       | Web Standard Character Matrix           |
| **02** | `decodeAudioData()`     | `Uint8Array` Bytes ➡️ Normalised `Float32Array` | Clamped Math Boundaries $\[-1.0, 1.0\]$ |
| **03** | `AudioBufferSourceNode` | Memory Array Loaded ➡️ Hardware Buffer Clip     | Native Runtime Output Context           |
| **04** | `GainNode` ➡️ Output    | Volume Attenuation ➡️ Physical Audio Speakers   | 24kHz High-Fidelity Audio Stream        |

### Reverting Text Strings into Binary Byte Streams

```typescript
// src/audioUtils.ts
export function base64ToUint8Array(base64: string): Uint8Array {
  // Decode the incoming Base64 ASCII text string into a native binary layout
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);

  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  return bytes;
}
```

### Transforming Raw Bytes into Playback Float Frameworks

We ingest our structural `Uint8Array` view matrices directly inside an `Int16Array` wrapper. Because standard hardware sound boards exclusively handle decimal variations positioned strictly between $-1.0$ and $1.0$, we balance the collection by dividing each array component by $32768.0$. This produces a smooth `Float32Array` matrix ready for immediate device output rendering.

```typescript
// src/audioUtils.ts
export async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }

  return buffer;
}
```

### Triggering Audio Node Output Environments

```typescript
// src/AudioService.ts
const uintData = base64ToUint8Array(audioData);

this.outputAudioContext = new AudioContext({
  sampleRate: OUTPUT_SAMPLE_RATE, // Rendered at 24000Hz
});

if (!this.outputAudioContext) return;

const audioBuffer = await decodeAudioData(
  uintData,
  this.outputAudioContext,
  OUTPUT_SAMPLE_RATE,
  1, // Mono Channel Layout configuration
);

// Bind the calculated decimals directly into a new runtime Audio Buffer Source
const source = this.outputAudioContext.createBufferSource();
source.buffer = audioBuffer;

// Pipe the source through an active GainNode to prevent audio clipping, then route directly to hardware speakers
source.connect(this.outputAudioContext.destination);
source.start(0);
```

# Resources

Sorce code - [GitHub](https://github.com/OnkarK0273/genai-js/tree/main/articles/9.%20Bi-directional%20Real-Time%20Audio%20Streaming%20Architecture)
