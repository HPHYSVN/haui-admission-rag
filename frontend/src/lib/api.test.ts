import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, askHaUI, getHealth, streamHaUI } from "./api";

const signal = new AbortController().signal;

const validSource = {
  source_id: "doc-1",
  chunk_id: "chunk-1",
  title: "Tài liệu",
  url: "https://example.org",
  year: 2026,
  category: "01_admission",
  score: 0.91,
};

function mockFetch(response: {
  ok: boolean;
  status?: number;
  body: unknown;
}) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: response.ok,
      status: response.status ?? (response.ok ? 200 : 400),
      json: () => Promise.resolve(response.body),
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("getHealth", () => {
  it("accepts the additive health payload", async () => {
    mockFetch({
      ok: true,
      body: {
        status: "ready",
        rag_loaded: true,
        model_configured: true,
        index_ready: true,
        chunk_count: 1224,
        provider: "ollama",
        model: "qwen3.5:4b",
        app_version: "1.1.0",
      },
    });

    await expect(getHealth(signal)).resolves.toMatchObject({ status: "ready" });
  });

  it("rejects a malformed health payload", async () => {
    mockFetch({ ok: true, body: { status: "maybe" } });
    await expect(getHealth(signal)).rejects.toThrow("Invalid health response");
  });
});

describe("askHaUI", () => {
  it("surfaces backend detail and stable code", async () => {
    mockFetch({
      ok: false,
      status: 429,
      body: {
        detail: "Bạn đã gửi quá nhiều yêu cầu.",
        code: "rate_limited",
        request_id: "r1",
      },
    });

    await expect(askHaUI("Học phí?", [], signal)).rejects.toMatchObject({
      name: "ApiError",
      message: "Bạn đã gửi quá nhiều yêu cầu.",
      code: "rate_limited",
    });
  });

  it("rejects a payload whose sources are malformed", async () => {
    mockFetch({
      ok: true,
      body: {
        answer: "Trả lời",
        request_id: "r1",
        sources: [{ source_id: "s1" }],
      },
    });

    await expect(askHaUI("Học phí?", [], signal)).rejects.toThrow(
      "Phản hồi từ máy chủ không đúng định dạng.",
    );
  });

  it("returns a valid response unchanged", async () => {
    const body = {
      answer: "Trả lời",
      request_id: "r1",
      sources: [
        {
          source_id: "doc-1",
          chunk_id: "chunk-1",
          title: "Tài liệu",
          url: "https://example.org",
          year: 2026,
          category: "01_admission",
          score: 0.91,
        },
      ],
    };
    mockFetch({ ok: true, body });

    await expect(askHaUI("Học phí?", [], signal)).resolves.toEqual(body);
  });

  it("throws ApiError with no code when the body has none", async () => {
    mockFetch({ ok: false, status: 500, body: { detail: "Lỗi" } });

    const error = await askHaUI("Học phí?", [], signal).catch((caught) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBeUndefined();
  });
});

function mockSse(chunks: string[]) {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({ ok: true, status: 200, body }),
  );
}

describe("streamHaUI", () => {
  it("delivers tokens and sources, then resolves the full answer", async () => {
    const tokens: string[] = [];
    const sources: unknown[] = [];
    mockSse([
      'event: meta\ndata: {"request_id":"r1","provider":"vllm","model":"m","year":2026}\n\n',
      'event: token\ndata: {"text":"Xin "}\n\n',
      'event: token\ndata: {"text":"chào"}\n\n',
      `event: sources\ndata: ${JSON.stringify({ sources: [validSource] })}\n\n`,
      'event: done\ndata: {"request_id":"r1"}\n\n',
    ]);

    const result = await streamHaUI("Học phí?", [], signal, {
      onToken: (token) => tokens.push(token),
      onSources: (value) => sources.push(value),
    });

    expect(tokens).toEqual(["Xin ", "chào"]);
    expect(sources).toHaveLength(1);
    expect(result.answer).toBe("Xin chào");
    expect(result.request_id).toBe("r1");
    expect(result.sources).toEqual([validSource]);
  });

  it("reassembles events split across transport chunks", async () => {
    mockSse([
      'event: tok',
      'en\ndata: {"text":"A"}',
      '\n\nevent: done\ndata: {"request_id":"r2"}\n\n',
    ]);

    await expect(streamHaUI("Học phí?", [], signal)).resolves.toMatchObject({
      answer: "A",
      request_id: "r2",
    });
  });

  it("converts a sanitized error event into ApiError", async () => {
    mockSse([
      'event: meta\ndata: {"request_id":"r3"}\n\n',
      'event: error\ndata: {"code":"provider_error","message":"Không thể tạo câu trả lời lúc này."}\n\n',
    ]);

    await expect(streamHaUI("Học phí?", [], signal)).rejects.toMatchObject({
      name: "ApiError",
      code: "provider_error",
      message: "Không thể tạo câu trả lời lúc này.",
    });
  });

  it("rejects a stream that ends without a done event", async () => {
    mockSse([
      'event: meta\ndata: {"request_id":"r4"}\n\n',
      'event: token\ndata: {"text":"dở dang"}\n\n',
    ]);

    await expect(streamHaUI("Học phí?", [], signal)).rejects.toMatchObject({
      code: "stream_incomplete",
    });
  });

  it("rejects malformed token payloads", async () => {
    mockSse([
      'event: token\ndata: {"nope":1}\n\n',
    ]);

    await expect(streamHaUI("Học phí?", [], signal)).rejects.toMatchObject({
      code: "stream_bad_event",
    });
  });
});
