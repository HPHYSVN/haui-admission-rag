import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  BookOpen,
  Check,
  Clock3,
  Copy,
  ExternalLink,
  Menu,
  Plus,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import { getHealth, Source, streamHaUI } from "./lib/api";
import { MessageBody } from "./components/MessageBody";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  failed?: boolean;
  streaming?: boolean;
};

type ChatSession = {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
};

const SESSIONS_KEY = "haui-chat-sessions";
const CURRENT_SESSION_KEY = "haui-chat-current-session";

const suggestions = [
  ["Phương thức xét tuyển", "HaUI có những phương thức xét tuyển nào năm 2026?"],
  ["Điểm chuẩn", "Điểm chuẩn ngành kỹ thuật phần mềm năm 2026 là bao nhiêu?"],
  ["Học phí & học bổng", "Học phí và học bổng năm 2026 của HaUI như thế nào?"],
  ["Thủ tục nhập học", "Tôi cần chuẩn bị gì để làm thủ tục nhập học?"],
];

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState("");
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [copiedId, setCopiedId] = useState("");
  const [serviceStatus, setServiceStatus] = useState<"checking" | "ready" | "degraded">("checking");
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    void getHealth(controller.signal)
      .then((health) => setServiceStatus(health.status))
      .catch(() => setServiceStatus("degraded"));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const savedDraft = localStorage.getItem("haui-chat-draft");
    if (savedDraft) setDraft(savedDraft);
    const savedSessions = localStorage.getItem(SESSIONS_KEY);
    const savedCurrentId = localStorage.getItem(CURRENT_SESSION_KEY) ?? "";
    if (savedSessions) {
      try {
        const parsed: unknown = JSON.parse(savedSessions);
        if (Array.isArray(parsed) && parsed.every(isStoredSession)) {
          setSessions(parsed);
          const current = parsed.find((session) => session.id === savedCurrentId) ?? parsed[0];
          if (current) {
            setCurrentSessionId(current.id);
            setMessages(current.messages);
          }
        }
      } catch {
        localStorage.removeItem(SESSIONS_KEY);
      }
    }
  }, []);

  useEffect(() => {
    if (draft) localStorage.setItem("haui-chat-draft", draft);
    else localStorage.removeItem("haui-chat-draft");
  }, [draft]);

  useEffect(() => {
    if (!currentSessionId || !messages.length) return;
    const persistable = messages.filter(
      (message) => message.role === "user" || message.content || (message.sources?.length ?? 0) > 0,
    );
    setSessions((current) => {
      const updated = current.map((session) =>
        session.id === currentSessionId ? { ...session, messages: persistable, updatedAt: Date.now() } : session,
      );
      localStorage.setItem(SESSIONS_KEY, JSON.stringify(updated));
      return updated;
    });
    localStorage.setItem(CURRENT_SESSION_KEY, currentSessionId);
  }, [messages, currentSessionId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function submit(question = draft) {
    const trimmed = question.trim();
    if (!trimmed || loading) return;
    setDraft("");
    setError("");
    setLoading(true);
    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content: trimmed };
    let sessionId = currentSessionId;
    if (!sessionId) {
      sessionId = crypto.randomUUID();
      const newSession: ChatSession = {
        id: sessionId,
        title: trimmed.slice(0, 58),
        messages: [userMessage],
        updatedAt: Date.now(),
      };
      setCurrentSessionId(sessionId);
      setSessions((current) => {
        const updated = [newSession, ...current];
        localStorage.setItem(SESSIONS_KEY, JSON.stringify(updated));
        return updated;
      });
      localStorage.setItem(CURRENT_SESSION_KEY, sessionId);
    }
    const assistantId = crypto.randomUUID();
    setMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);
    const controller = new AbortController();
    abortRef.current = controller;
    // Only the newest request may mutate the transcript. Aborting or starting a
    // new chat detaches this controller, so late tokens cannot leak into it.
    const isCurrent = () => abortRef.current === controller;
    try {
      const history = messages.map(({ role, content }) => ({ role, content }));
      const result = await streamHaUI(trimmed, history, controller.signal, {
        onToken: (token) => {
          if (!isCurrent()) return;
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + token }
                : message,
            ),
          );
        },
        onSources: (sources) => {
          if (!isCurrent()) return;
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId ? { ...message, sources } : message,
            ),
          );
        },
      });
      if (!isCurrent()) return;
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? { ...message, content: result.answer, sources: result.sources, streaming: false }
            : message,
        ),
      );
    } catch (requestError) {
      if (!isCurrent()) return;
      if ((requestError as Error).name === "AbortError") {
        // Keep whatever arrived before the user cancelled, but stop the caret.
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId ? { ...message, streaming: false } : message,
          ),
        );
      } else {
        setMessages((current) => {
          const target = current.find((message) => message.id === assistantId);
          if (target && !target.content) {
            return current.filter((message) => message.id !== assistantId);
          }
          return current.map((message) =>
            message.id === assistantId
              ? { ...message, streaming: false, failed: true }
              : message,
          );
        });
        setError(requestError instanceof Error ? requestError.message : "Đã xảy ra lỗi.");
      }
    } finally {
      if (isCurrent()) {
        abortRef.current = null;
        setLoading(false);
      }
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    void submit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  }

  function resetChat() {
    abortRef.current?.abort();
    setMessages([]);
    setCurrentSessionId("");
    setError("");
    setLoading(false);
    setDraft("");
    setSidebarOpen(false);
  }

  function openSession(session: ChatSession) {
    abortRef.current?.abort();
    setMessages(session.messages);
    setCurrentSessionId(session.id);
    setError("");
    setLoading(false);
    setDraft("");
    localStorage.setItem(CURRENT_SESSION_KEY, session.id);
    setSidebarOpen(false);
  }

  function deleteSession(sessionId: string) {
    const updated = sessions.filter((session) => session.id !== sessionId);
    setSessions(updated);
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(updated));
    if (sessionId === currentSessionId) {
      setMessages([]);
      setCurrentSessionId("");
      localStorage.removeItem(CURRENT_SESSION_KEY);
    }
  }

  async function copyAnswer(message: Message) {
    await navigator.clipboard.writeText(message.content);
    setCopiedId(message.id);
    window.setTimeout(() => setCopiedId(""), 1600);
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "is-open" : ""}`}>
        <div className="sidebar-top">
          <div className="brand-mark" aria-hidden="true">H</div>
          <div>
            <p className="eyebrow">HAUI / ADMISSIONS</p>
            <p className="brand-name">Admission Desk</p>
          </div>
          <button className="icon-button mobile-close" onClick={() => setSidebarOpen(false)} aria-label="Đóng menu">
            <X size={18} />
          </button>
        </div>
        <button className="new-chat" onClick={resetChat}>
          <Plus size={18} /> Cuộc trò chuyện mới
        </button>
        <div className="history-heading"><span>Lịch sử gần đây</span><Clock3 size={14} /></div>
        {sessions.length === 0 ? (
          <div className="history-empty">
            <span className="history-dot"><Check size={13} /></span>
            <p>Cuộc trò chuyện của bạn<br />sẽ xuất hiện tại đây.</p>
          </div>
        ) : (
          <div className="session-list" aria-label="Các cuộc trò chuyện đã lưu">
            {sessions
              .slice()
              .sort((a, b) => b.updatedAt - a.updatedAt)
              .map((session) => (
                <div className={`session-item ${session.id === currentSessionId ? "is-active" : ""}`} key={session.id}>
                  <button className="session-button" onClick={() => openSession(session)}>
                    <span>{session.title}</span>
                    <small>{new Date(session.updatedAt).toLocaleDateString("vi-VN")}</small>
                  </button>
                  <button className="session-delete" onClick={() => deleteSession(session.id)} aria-label={`Xóa ${session.title}`}>
                    <X size={14} />
                  </button>
                </div>
              ))}
          </div>
        )}
        <div className="sidebar-footer">
          <div className={`status-indicator ${serviceStatus}`}>
            <span />
            {serviceStatus === "checking" ? "Đang kiểm tra kết nối" : serviceStatus === "ready" ? "Hệ thống sẵn sàng" : "Hệ thống cần kiểm tra"}
          </div>
          <p>Trả lời dựa trên nguồn tuyển sinh chính thức. Không lưu nội dung hội thoại trên máy chủ.</p>
        </div>
      </aside>

      {sidebarOpen && <button className="scrim" aria-label="Đóng menu" onClick={() => setSidebarOpen(false)} />}
      <main className="main-panel">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setSidebarOpen(true)} aria-label="Mở menu">
            <Menu size={20} />
          </button>
          <div className="topbar-title">
            <span className="live-dot" /> Trợ lý tuyển sinh HaUI
          </div>
          <div className="trust-badge"><ShieldCheck size={15} /> Có căn cứ</div>
        </header>

        <section className="conversation" aria-live="polite" aria-busy={loading}>
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-kicker"><span /> Thông tin tuyển sinh 2026</div>
              <h1>Chọn đúng hướng,<br /><em>bắt đầu tự tin.</em></h1>
              <p className="welcome-copy">
                Hỏi về ngành học, phương thức xét tuyển, điểm chuẩn hay thủ tục nhập học.
                Tôi sẽ tìm câu trả lời từ kho tài liệu chính thức của HaUI.
              </p>
              <div className="suggestions">
                {suggestions.map(([label, question]) => (
                  <button key={label} className="suggestion" onClick={() => void submit(question)}>
                    <span>{label}</span><ArrowUp size={15} className="suggestion-arrow" />
                  </button>
                ))}
              </div>
              <div className="welcome-note"><Sparkles size={14} /> Trợ lý AI · Nguồn được kiểm chứng</div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((message) => (
                <article className={`message ${message.role}${message.streaming ? " is-streaming" : ""}${message.failed ? " is-failed" : ""}`} key={message.id}>
                  <div className="message-label">{message.role === "user" ? "Bạn" : "HaUI Desk"}</div>
                  <MessageBody content={message.content} />
                  {message.role === "assistant" && message.sources && message.sources.length > 0 && (
                    <SourceList sources={message.sources} />
                  )}
                  {message.role === "assistant" && (
                    <div className="message-actions">
                      <button onClick={() => void copyAnswer(message)}><Copy size={13} /> {copiedId === message.id ? "Đã sao chép" : "Sao chép"}</button>
                    </div>
                  )}
                </article>
              ))}
              {loading && !messages.some((message) => message.streaming && message.content) && (
                <div className="processing" role="status">
                  <span /><span /><span /> Đang tra cứu tài liệu
                  <button className="cancel-button" type="button" onClick={() => abortRef.current?.abort()}>
                    Hủy
                  </button>
                </div>
              )}
              <div ref={endRef} />
            </div>
          )}
        </section>

        <div className="composer-wrap">
          {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => void submit([...messages].reverse().find((message) => message.role === "user")?.content ?? "")}><RotateCcw size={14} /> Thử lại</button></div>}
          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Bạn muốn biết điều gì về tuyển sinh HaUI?"
              rows={1}
              maxLength={2000}
              aria-label="Câu hỏi tuyển sinh"
              disabled={loading}
            />
            <div className="composer-meta">
              <span>Enter để gửi · Shift + Enter để xuống dòng</span>
              <button className="send-button" type="submit" disabled={!draft.trim() || loading} aria-label="Gửi câu hỏi">
                <ArrowUp size={19} />
              </button>
            </div>
          </form>
          <p className="disclaimer" role="note">Lưu ý: AI có thể mắc lỗi. Hãy kiểm tra nguồn chính thức trước quyết định quan trọng.</p>
        </div>
      </main>
    </div>
  );
}

function isStoredMessage(value: unknown): value is Message {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.id === "string" &&
    (candidate.role === "user" || candidate.role === "assistant") &&
    typeof candidate.content === "string" &&
    (candidate.sources === undefined || Array.isArray(candidate.sources))
  );
}

function isStoredSession(value: unknown): value is ChatSession {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.id === "string" &&
    typeof candidate.title === "string" &&
    typeof candidate.updatedAt === "number" &&
    Array.isArray(candidate.messages) &&
    candidate.messages.every(isStoredMessage)
  );
}

function SourceList({ sources }: { sources: Source[] }) {
  return (
    <div className="sources">
      <div className="sources-heading"><BookOpen size={14} /> Nguồn tham khảo</div>
      {sources.map((source) => (
        <a className="source" href={source.url} target="_blank" rel="noreferrer" key={source.chunk_id}>
          <span className="source-index">{String(sources.indexOf(source) + 1).padStart(2, "0")}</span>
          <span className="source-info"><strong>{source.title}</strong><small>{source.year ?? "Tài liệu chính thức"} · Mở nguồn gốc</small></span>
          <ExternalLink size={14} />
        </a>
      ))}
    </div>
  );
}

export default App;
