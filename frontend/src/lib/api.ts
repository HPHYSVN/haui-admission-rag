export type Source = {
  source_id: string;
  chunk_id: string;
  title: string;
  url: string;
  year: number | null;
  category: string | null;
  score: number;
};

export type ChatResponse = {
  answer: string;
  sources: Source[];
  request_id: string;
};

export type HealthResponse = {
  status: "ready" | "degraded";
  rag_loaded: boolean;
  model_configured: boolean;
};

const API_URL = import.meta.env.VITE_API_URL ?? "";

export async function getHealth(signal: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/api/v1/health`, { signal });
  if (!response.ok) throw new Error("Health check failed");
  const body: unknown = await response.json();
  if (!isHealthResponse(body)) throw new Error("Invalid health response");
  return body;
}

export async function askHaUI(
  question: string,
  history: Array<{ role: "user" | "assistant"; content: string }>,
  signal: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history: history.slice(-8) }),
    signal,
  });
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? String(body.detail)
        : "Không thể kết nối với trợ lý lúc này.";
    throw new Error(detail);
  }
  if (!isChatResponse(body)) {
    throw new Error("Phản hồi từ máy chủ không đúng định dạng.");
  }
  return body;
}

function isHealthResponse(body: unknown): body is HealthResponse {
  if (typeof body !== "object" || body === null) return false;
  const candidate = body as Record<string, unknown>;
  return (
    (candidate.status === "ready" || candidate.status === "degraded") &&
    typeof candidate.rag_loaded === "boolean" &&
    typeof candidate.model_configured === "boolean"
  );
}

function isChatResponse(body: unknown): body is ChatResponse {
  if (typeof body !== "object" || body === null) return false;
  const candidate = body as Record<string, unknown>;
  return (
    typeof candidate.answer === "string" &&
    typeof candidate.request_id === "string" &&
    Array.isArray(candidate.sources)
  );
}
