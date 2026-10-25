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

export type ChatStreamMeta = {
  request_id: string;
  provider?: string;
  model?: string;
  year?: number | null;
};

export type ChatStreamHandlers = {
  onMeta?: (meta: ChatStreamMeta) => void;
  onToken?: (token: string) => void;
  onSources?: (sources: Source[]) => void;
};

export type HealthResponse = {
  status: "ready" | "degraded";
  rag_loaded: boolean;
  model_configured: boolean;
};

/** Error carrying the backend's stable machine-readable `code`, when present. */
export class ApiError extends Error {
  code?: string;

  constructor(message: string, code?: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
  }
}

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
    throw apiErrorFromBody(body);
  }
  if (!isChatResponse(body)) {
    throw new Error("Phản hồi từ máy chủ không đúng định dạng.");
  }
  return body;
}

/**
 * Stream one answer over SSE and return the completed response.
 *
 * Tokens are delivered to `onToken` as they arrive; the resolved promise is the
 * same shape as `askHaUI` so the UI can use one rendering path. Errors emitted by
 * the server after the stream starts are converted back into `ApiError`.
 */
export async function streamHaUI(
  question: string,
  history: Array<{ role: "user" | "assistant"; content: string }>,
  signal: AbortSignal,
  handlers: ChatStreamHandlers = {},
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/api/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, history: history.slice(-8) }),
    signal,
  });
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    throw apiErrorFromBody(body);
  }
  if (!response.body) {
    throw new ApiError("Trình duyệt không hỗ trợ đọc luồng phản hồi.", "stream_unsupported");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  let sources: Source[] = [];
  let requestId = "";
  let done = false;

  const dispatch = (block: string) => {
    const frame = parseSseFrame(block);
    if (!frame) return;
    if (frame.event === "meta") {
      const payload = parseJsonObject(frame.data, "meta");
      if (typeof payload.request_id === "string") requestId = payload.request_id;
      handlers.onMeta?.({
        request_id: requestId,
        provider: typeof payload.provider === "string" ? payload.provider : undefined,
        model: typeof payload.model === "string" ? payload.model : undefined,
        year: typeof payload.year === "number" ? payload.year : null,
      });
      return;
    }
    if (frame.event === "token") {
      const payload = parseJsonObject(frame.data, "token");
      if (typeof payload.text !== "string") {
        throw new ApiError("Luồng phản hồi không đúng định dạng.", "stream_bad_event");
      }
      answer += payload.text;
      handlers.onToken?.(payload.text);
      return;
    }
    if (frame.event === "sources") {
      const payload = parseJsonObject(frame.data, "sources");
      if (!Array.isArray(payload.sources) || !payload.sources.every(isSource)) {
        throw new ApiError("Danh sách nguồn không đúng định dạng.", "stream_bad_event");
      }
      sources = payload.sources;
      handlers.onSources?.(sources);
      return;
    }
    if (frame.event === "done") {
      const payload = parseJsonObject(frame.data, "done");
      if (typeof payload.request_id === "string" && payload.request_id) {
        requestId = payload.request_id;
      }
      done = true;
      return;
    }
    if (frame.event === "error") {
      const payload = parseJsonObject(frame.data, "error");
      throw new ApiError(
        typeof payload.message === "string" ? payload.message : "Không thể tạo câu trả lời lúc này.",
        typeof payload.code === "string" ? payload.code : "stream_error",
      );
    }
  };

  try {
    while (true) {
      const { value, done: readerDone } = await reader.read();
      buffer = (buffer + decoder.decode(value, { stream: !readerDone })).replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        dispatch(block);
        boundary = buffer.indexOf("\n\n");
      }
      if (readerDone) break;
    }
    if (buffer.trim()) dispatch(buffer);
  } catch (error) {
    await reader.cancel().catch(() => undefined);
    throw error;
  } finally {
    reader.releaseLock();
  }

  if (!done) {
    throw new ApiError(
      "Kết nối bị gián đoạn trước khi câu trả lời hoàn tất.",
      "stream_incomplete",
    );
  }
  if (!requestId) {
    throw new ApiError("Phản hồi từ máy chủ thiếu mã yêu cầu.", "stream_bad_event");
  }
  return { answer, sources, request_id: requestId };
}

function apiErrorFromBody(body: unknown): ApiError {
  const detail =
    isRecord(body) && "detail" in body
      ? String(body.detail)
      : "Không thể kết nối với trợ lý lúc này.";
  const code =
    isRecord(body) && "code" in body ? String(body.code) : undefined;
  return new ApiError(detail, code);
}

type SseFrame = { event: string; data: string };

function parseSseFrame(block: string): SseFrame | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return null;
  return { event, data: data.join("\n") };
}

function parseJsonObject(data: string, event: string): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(data);
  } catch {
    throw new ApiError(`Sự kiện ${event} không phải JSON hợp lệ.`, "stream_bad_event");
  }
  if (!isRecord(value)) {
    throw new ApiError(`Sự kiện ${event} không đúng định dạng.`, "stream_bad_event");
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isHealthResponse(body: unknown): body is HealthResponse {
  if (!isRecord(body)) return false;
  return (
    (body.status === "ready" || body.status === "degraded") &&
    typeof body.rag_loaded === "boolean" &&
    typeof body.model_configured === "boolean"
  );
}

function isSource(value: unknown): value is Source {
  if (!isRecord(value)) return false;
  return (
    typeof value.source_id === "string" &&
    typeof value.chunk_id === "string" &&
    typeof value.title === "string" &&
    typeof value.url === "string" &&
    (value.year === null || typeof value.year === "number") &&
    (value.category === null || typeof value.category === "string") &&
    typeof value.score === "number" &&
    Number.isFinite(value.score)
  );
}

function isChatResponse(body: unknown): body is ChatResponse {
  if (!isRecord(body)) return false;
  return (
    typeof body.answer === "string" &&
    typeof body.request_id === "string" &&
    Array.isArray(body.sources) &&
    body.sources.every(isSource)
  );
}
