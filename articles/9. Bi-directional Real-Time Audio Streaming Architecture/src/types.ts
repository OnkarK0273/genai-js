export enum ConnectionState {
  DISCONNECTED = "DISCONNECTED",
  CONNECTING = "CONNECTING",
  CONNECTED = "CONNECTED",
  ERROR = "ERROR",
}

export enum agentState {
  THINKING = "thinking",
  LISTENING = "listening",
  TALKING = "talking",
}

export interface LiveManagerCallbacks {
  onConectionStateChange: (state: ConnectionState) => void;
  onAgentStateChange: (state: agentState) => void;
  onTranscript: (
    sender: "user" | "model",
    text: string,
    isPartial: boolean,
  ) => void;
  onAudioLevel: (level: number, type: "input" | "output") => void;
  onError: (error: string) => void;
}

export interface ConnectConfig {
  selected_topic: string;
  description: string;

  selected_launguage_name: string;
  selected_launguage_code: string;
  selected_launguage_region: string;

  selected_proefficent_level: string;
  selected_assistant_voice: string;
}
