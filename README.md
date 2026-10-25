# Chatbot tư vấn tuyển sinh HaUI bằng LLM và RAG

Đây là hệ thống chatbot tư vấn tuyển sinh cho Trường Đại học Công nghiệp Hà
Nội (HaUI). Hệ thống kết hợp kho tài liệu chính thức, truy xuất lai BM25 +
embedding đa ngôn ngữ và mô hình ngôn ngữ lớn chạy qua endpoint tương thích
OpenAI (vLLM, Ollama hoặc API bên ngoài).

Mục tiêu của dự án là trả lời ngắn gọn, có căn cứ và có nguồn tham khảo; không
suy đoán khi kho dữ liệu không đủ bằng chứng.

## Tính năng

- Thu thập và chuẩn hóa tài liệu tuyển sinh từ HaUI, Bộ GD&ĐT và các cổng
  thông tin nhà nước.
- Kiểm tra tài liệu theo JSON Schema và loại bỏ tài liệu trùng hoặc không hợp
  lệ.
- Review tài liệu và chunk thủ công trước khi đưa vào truy xuất.
- Truy xuất kết hợp BM25 và embedding multilingual.
- Rerank các ứng viên bằng cross-encoder multilingual-compatible trước khi
  đưa context vào LLM.
- Lọc nghiêm ngặt theo năm và danh mục khi câu hỏi có yêu cầu cụ thể.
- Chính sách eligibility theo chu kỳ/hiệu lực: giữ tài liệu lịch sử, quarantine
  bản nháp, và từ chối trả lời khi thiếu bằng chứng đúng năm.
- Sinh câu trả lời qua bất kỳ endpoint tương thích OpenAI nào (vLLM, Ollama,
  API ngoài) mà không cần sửa code.
- Chat streaming qua SSE (`POST /api/v1/chat/stream`) với token được chuyển tiếp
  ngay, có hủy request và dọn dẹp kết nối provider.
- Nhập tài liệu an toàn bằng CLI từ file hoặc URL, có kiểm tra SSRF, chống path
  traversal, chống trùng và giữ bước review thủ công.
- Hiển thị nguồn được lấy trực tiếp từ các chunk đã truy xuất.
- Giao diện React/Vite tiếng Việt, responsive, có trạng thái lỗi và trạng thái
  kết nối.
- Lịch sử cuộc trò chuyện được lưu cục bộ trong trình duyệt bằng
  `localStorage`; không lưu nội dung hội thoại trên backend.

## Kiến trúc

```text
Chế độ A: CLI / evaluation          Chế độ B: HTTP API
  python -m rag.cli ...               Trình duyệt -> React/Vite :5173
  evaluation/run_ragas.py                        -> FastAPI :8001
            |                                      -> app/services/rag_service.py
            +------------------+-------------------+
                               v
                       rag/factory.py  (một đường khởi tạo duy nhất)
                               v
       rag/pipeline.py + rag/prompts/  (retrieval, prompt, sources)
                               v
       rag/llm/  -> mọi endpoint tương thích OpenAI
                    (vLLM :8000, Ollama :11434, API ngoài, ...)
```

Chi tiết: [`docs/architecture.md`](docs/architecture.md).

Luồng xử lý câu hỏi:

1. Người dùng gửi câu hỏi từ giao diện.
2. FastAPI kiểm tra độ dài, lịch sử và định dạng request.
3. Bộ lọc metadata và chính sách eligibility xác định tài liệu/năm được dùng.
4. BM25 và FAISS/embedding xếp hạng các chunk liên quan.
5. Nếu thiếu bằng chứng đúng năm, pipeline trả lời từ chối an toàn thay vì mượn
   dữ liệu năm khác.
6. Pipeline đóng gói context trong ngân sách ký tự đã giới hạn.
7. Provider đã cấu hình sinh câu trả lời dựa trên context (JSON hoặc stream SSE).
8. Backend tạo danh sách nguồn từ chunk truy xuất, không để model tự tạo URL.
9. Frontend hiển thị câu trả lời, nguồn, trạng thái và lỗi phù hợp.

## Cấu trúc chính

```text
app/                    FastAPI, schema và service API
crawler/                Crawler và bộ làm sạch dữ liệu
data/documents/         Tài liệu chuẩn hóa JSONL
data/chunks/final/      Chunk đã review và phê duyệt
data/indexes/           Chỉ mục vector cục bộ, không commit
frontend/               Ứng dụng React + TypeScript + Vite
rag/                    BM25, embedding, eligibility, ingestion, pipeline và CLI
schemas/                JSON Schema cho corpus
scripts/                Công cụ chunk, validate, merge và review
tests/                  Test RAG và test API
```

## Yêu cầu

- Linux, macOS hoặc WSL.
- Python 3.11 trở lên.
- `uv`.
- Node.js và npm.
- Một endpoint tương thích OpenAI đang chạy (vLLM, Ollama hoặc API ngoài).
- Đủ dung lượng để tải model embedding ở lần truy vấn semantic đầu tiên.

## Cài đặt

### Backend và RAG

```bash
uv sync --locked
cp .env.example .env
```

### Frontend

```bash
cd frontend
npm install
cd ..
```

## Cấu hình môi trường

Sửa file `.env` ở thư mục gốc:

```dotenv
# Sinh câu trả lời: mọi endpoint tương thích OpenAI dùng chung một adapter.
LLM_PROVIDER=openai_compatible   # openai_compatible | vllm | ollama | openai
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4
LLM_API_KEY=
LLM_TIMEOUT=120
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=800

# Tên cũ VLLM_BASE_URL / VLLM_MODEL / VLLM_API_KEY vẫn được hỗ trợ.

# Profile prompt: haui_admission (mặc định) hoặc generic.
RAG_PROMPT_PROFILE=haui_admission

# Truy xuất và index cục bộ, độc lập với model sinh câu trả lời.
RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-large
RAG_EMBED_BATCH_SIZE=32
RAG_ALPHA=0.5
RAG_SCORE_THRESHOLD=0.08
RAG_MIN_TERM_COVERAGE=0.3
RAG_RERANKER_ENABLED=true
RAG_RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
RAG_RERANKER_CANDIDATES=12
RAG_RERANKER_RETRIEVAL_WEIGHT=4.0
RAG_TOP_K=4

# Cấu hình FastAPI.
APP_ALLOWED_ORIGINS=http://localhost:5173
APP_MAX_QUESTION_LENGTH=2000
APP_MAX_HISTORY_MESSAGES=8
APP_MAX_REQUEST_BYTES=65536
APP_RATE_LIMIT_PER_MINUTE=0
APP_API_KEY=
```

Danh sách đầy đủ biến `LLM_*`, bảng alias `VLLM_*`, ví dụ vLLM/Ollama/API ngoài:
[`docs/llm-providers.md`](docs/llm-providers.md).

Không commit `.env`, API key, token hoặc thông tin xác thực thật.

Reranker được tải lazy ở lần truy vấn đầu tiên. Để chạy baseline không có
reranker, đặt `RAG_RERANKER_ENABLED=false`; model mặc định là cross-encoder
`jinaai/jina-reranker-v2-base-multilingual`.

## Chạy hệ thống

### 1. Chạy endpoint LLM

Giữ vLLM ở cổng 8000 và trỏ `LLM_BASE_URL` (hoặc alias `VLLM_BASE_URL`) tới
đó. Với Ollama, dùng `LLM_PROVIDER=ollama`, `LLM_BASE_URL=http://localhost:11434/v1`.
Kiểm tra:

```bash
curl http://localhost:8000/v1/models
```

Nếu lệnh trả về model đang phục vụ thì không khởi chạy FastAPI trên cổng 8000.

### 2. Chạy FastAPI

Mở terminal thứ hai:

```bash
uv run uvicorn app.main:app --reload --port 8001
```

Kiểm tra trạng thái:

```bash
curl http://localhost:8001/api/v1/health
```

Kết quả sẵn sàng có dạng (các trường mới chỉ mang tính chẩn đoán, không chứa
path/URL/secret):

```json
{
  "status": "ready",
  "rag_loaded": true,
  "model_configured": true,
  "index_ready": true,
  "chunk_count": 1224,
  "provider": "openai_compatible",
  "model": "JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4",
  "app_version": "1.1.0"
}
```

`/api/v1/health` luôn trả HTTP 200 (kể cả `degraded`) và không tải model. Kiểm tra
sâu cho Docker/orchestrator:

```bash
curl -i http://localhost:8001/api/v1/ready
```

`/ready` trả 200 chỉ khi đã nạp chunk, index tồn tại trên đĩa và endpoint LLM trả
lời; ngược lại trả 503.

Tài liệu OpenAPI có tại <http://localhost:8001/docs>. Hợp đồng API đầy đủ:
[`docs/api.md`](docs/api.md).

### 3. Chạy frontend

Mở terminal thứ ba:

```bash
cd frontend
npm run dev
```

Mở <http://localhost:5173>.

Vite tự chuyển tiếp `/api` đến `http://localhost:8001`. Nếu triển khai frontend
ở origin khác, đặt biến `VITE_API_URL` trước khi build.

### 4. Chạy bằng Docker (tuỳ chọn)

```bash
docker compose build
docker compose up -d
docker compose run --rm backend python -m rag.cli index --all
```

Frontend ở <http://localhost:8080>, backend ở <http://localhost:8001>. Index và
cache model nằm trong named volume nên không mất khi container được tạo lại.
Chi tiết và cách xử lý sự cố: [`docs/docker.md`](docs/docker.md).

### Đổi nhà cung cấp LLM

Không cần sửa code. Ví dụ với Ollama:

```dotenv
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3.5:4b
```

Đổi model sinh câu trả lời không cần rebuild index. Đổi model embedding thì cần
rebuild index. Xem [`docs/llm-providers.md`](docs/llm-providers.md) và
[`docs/corpus-and-indexes.md`](docs/corpus-and-indexes.md).

## API

### `GET /api/v1/health`

Trả về trạng thái sẵn sàng tối giản, không tiết lộ path, URL hoặc secret. Luôn
HTTP 200; trường `status` là `ready` hoặc `degraded`.

### `GET /api/v1/ready`

Kiểm tra sâu (chunk, index trên đĩa, provider). HTTP 200 khi mọi thứ sẵn sàng,
503 khi chưa.

### `POST /api/v1/chat`

Request:

```json
{
  "question": "HaUI có những phương thức xét tuyển nào năm 2026?",
  "history": [
    {
      "role": "user",
      "content": "Tôi muốn hỏi về tuyển sinh HaUI."
    }
  ]
}
```

Response:

```json
{
  "answer": "Câu trả lời dựa trên tài liệu đã truy xuất.",
  "sources": [
    {
      "source_id": "01_admission_001",
      "chunk_id": "01_admission_001_chunk_001",
      "title": "Thông tin tuyển sinh",
      "url": "https://tuyensinh.haui.edu.vn/...",
      "year": 2026,
      "category": "01_admission",
      "score": 0.91
    }
  ],
  "request_id": "..."
}
```

Backend giới hạn câu hỏi 2.000 ký tự và history 8 message. Câu trả lời được
trả về dạng plain text; frontend không thực thi HTML, script hoặc nội dung động
do model sinh ra.

Mọi lỗi có cùng dạng `{detail, code, request_id}` (riêng lỗi sinh câu trả lời giữ
thêm `error_type` để tương thích). Các mã ổn định: `invalid_request`,
`question_too_long`, `history_too_long`, `payload_too_large`, `unauthorized`,
`rate_limited`, `provider_*`, `service_unavailable`, `internal_error`.
Bảng đầy đủ: [`docs/api.md`](docs/api.md).

### `POST /api/v1/chat/stream`

Cùng request body, giới hạn, xác thực và chính sách truy xuất như `/chat`, nhưng
trả về `text/event-stream`. Thứ tự event cố định:

1. `meta` — `request_id`, `provider`, `model`, `year`.
2. `token` — 0 hoặc nhiều event, mỗi event có `text`.
3. `sources` — mảng nguồn do backend tạo.
4. `done` — `finish_reason` là `stop` hoặc `abstain`.

```bash
curl -N http://localhost:8001/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"Học phí năm 2026 là bao nhiêu?","history":[]}'
```

Token được chuyển tiếp ngay, không buffer toàn bộ câu trả lời. Nếu lỗi xảy ra sau
khi stream đã mở, backend gửi event `error` đã được làm sạch thay vì đổi HTTP
status. Frontend dùng `fetch` + `ReadableStream` (không dùng `EventSource`) và
hủy request khi người dùng chuyển cuộc trò chuyện. Nginx đi kèm tắt buffering cho
đúng path này. Chi tiết event và lỗi: [`docs/api.md`](docs/api.md).

## Lịch sử local trên trình duyệt

Mỗi lần người dùng bắt đầu một cuộc trò chuyện mới, frontend tạo một session
riêng và lưu vào `localStorage`:

- `haui-chat-sessions`: danh sách session, tiêu đề, thời gian cập nhật và
  message.
- `haui-chat-current-session`: session đang mở.
- `haui-chat-draft`: bản nháp câu hỏi hiện tại.

Sidebar hiển thị các session theo thời gian cập nhật. Người dùng có thể mở lại
session, tạo cuộc trò chuyện mới hoặc xóa từng session. Dữ liệu chỉ nằm trong
trình duyệt hiện tại và không được đồng bộ giữa thiết bị.

Để xóa toàn bộ dữ liệu local, mở DevTools và chạy:

```js
localStorage.removeItem("haui-chat-sessions");
localStorage.removeItem("haui-chat-current-session");
localStorage.removeItem("haui-chat-draft");
```

## Pipeline dữ liệu và review

Nhập tài liệu từ file hoặc URL (mặc định vào trạng thái `review_pending`):

```bash
uv run python -m rag.cli ingest --help
uv run python -m rag.cli ingest file path/to/document.pdf --category 01_admission
uv run python -m rag.cli ingest url https://tuyensinh.haui.edu.vn/... --category 01_admission --dry-run
```

URL chỉ chấp nhận `http`/`https`, chặn địa chỉ nội bộ/private, kiểm tra lại
redirect và giới hạn dung lượng. Trùng tên tài liệu sẽ báo lỗi trừ khi dùng
`--update`. Không có endpoint HTTP để nhập tài liệu.

Review tương tác:

```bash
uv run python -m rag.cli review
```

Kiểm tra corpus:

```bash
uv run python scripts/validate.py
```

Quy trình dữ liệu:

1. Crawler hoặc `rag ingest` lưu HTML/PDF gốc vào `data/raw/`.
2. Cleaner chuẩn hóa văn bản và metadata.
3. Tài liệu chuẩn được lưu trong `data/documents/*.jsonl`.
4. Review tài liệu xác nhận nội dung.
5. Chunk proposal được tạo và review.
6. Chunk cuối được lưu trong `data/chunks/final/`.
7. Chính sách `data/workflow/document_eligibility.json` gắn phạm vi năm/hiệu lực
   và quarantine khi load chunk.
8. Chỉ mục BM25 và vector được tạo khi truy vấn.

Không commit model embedding, FAISS index hoặc dữ liệu tạm vào repository.

## Kiểm thử và build

Chạy toàn bộ test backend:

```bash
uv run python -m unittest discover -s tests -v
```

Kiểm tra schema corpus:

```bash
uv run python scripts/validate.py
```

Test frontend (render an toàn + validator API):

```bash
cd frontend
npm test
```

Build frontend và kiểm tra dependency:

```bash
cd frontend
npm run build
npm audit
```

## Bảo mật và giới hạn triển khai

- Input người dùng được kiểm tra bằng Pydantic và giới hạn kích thước.
- Body vượt `APP_MAX_REQUEST_BYTES` bị trả 413 trước khi parse JSON.
- CORS chỉ cho phép origin được khai báo trong `APP_ALLOWED_ORIGINS`.
- API không expose endpoint LLM trực tiếp cho browser.
- Lỗi nội bộ không được hiển thị nguyên văn cho người dùng; mọi lỗi có `code`
  ổn định và `request_id`.
- Prompt coi user message và retrieved document là dữ liệu không tin cậy.
- Nguồn được tạo từ retrieval result, không từ output tự do của model.
- Frontend hiển thị plain text và không dùng `dangerouslySetInnerHTML`.
- Có thể bật `APP_API_KEY` (header `X-API-Key`) và
  `APP_RATE_LIMIT_PER_MINUTE`; cả hai mặc định tắt để giữ tương thích.
- Rate limit là in-process, theo từng worker; production nên đặt giới hạn và
  xác thực thật ở reverse proxy/gateway.
- API chưa lưu hội thoại server-side; lịch sử chỉ nằm trong `localStorage`.

Không nên public API trực tiếp ra Internet. Khi triển khai thật cần bổ sung
reverse proxy, HTTPS, rate limiting, authentication/authorization, logging có
kiểm soát và chính sách retention.

## Giới hạn đã biết

- Streaming chỉ hỗ trợ qua SSE; không có WebSocket hay resume sau khi mất kết
  nối. Client phải coi stream kết thúc mà thiếu `done`/`error` là chưa hoàn tất.
- Chưa có nền tảng phân quyền đầy đủ; `APP_API_KEY` chỉ là một cổng đơn giản.
- Rate limit là in-process, không chia sẻ giữa nhiều worker.
- Ingestion/review vẫn chỉ chạy bằng CLI, không có endpoint quản trị.
- URL ingestion vẫn còn rủi ro DNS rebinding giữa lúc kiểm tra và lúc gọi; triển
  khai thật nên đặt sau network boundary/proxy kiểm soát.
- Prompt injection không thể loại bỏ hoàn toàn; đã giảm thiểu bằng quy tắc
  "tài liệu không phải chỉ thị", phân tách ngữ cảnh, nguồn do server tạo và
  render dạng text. Rủi ro còn lại được ghi nhận, không phủ nhận.
- Lần truy vấn semantic đầu tiên cần tải model embedding/reranker; chạy trước
  `index --all` hoặc dùng cache volume trong Docker.

## Khắc phục sự cố

### Cổng 8000 đã được sử dụng

Thông thường đó là vLLM đang chạy trên host. Kiểm tra:

```bash
ss -ltnp 'sport = :8000'
curl http://localhost:8000/v1/models
```

Giữ vLLM ở 8000 và chạy FastAPI ở 8001.

### FastAPI báo `degraded`

Kiểm tra:

- `data/chunks/final/` có file JSONL.
- Workflow state đã đánh dấu chunk là `final` hoặc `indexed`.
- `data/indexes/vector/` đã có index; nếu chưa, chạy
  `uv run python -m rag.cli index --all`.
- `.env` có `LLM_MODEL` (hoặc `VLLM_MODEL`).
- Endpoint LLM trả lời ở `/v1/models`; kiểm tra bằng `/api/v1/ready`.

### Frontend không gọi được API

Kiểm tra backend ở cổng 8001, origin trong `APP_ALLOWED_ORIGINS` và Vite đang
chạy ở cổng 5173. Sau khi sửa `.env`, khởi động lại backend.

## Nguyên tắc dữ liệu

- Chỉ sử dụng nguồn chính thức của HaUI, Bộ GD&ĐT và cơ quan nhà nước.
- Không dùng diễn đàn, mạng xã hội hoặc nguồn chưa xác minh.
- Không đưa menu, banner, footer, quảng cáo và OCR không đáng tin vào corpus.
- Giữ nguyên schema JSONL và provenance của từng tài liệu/chunk.
- Mỗi document phải có `id`, `title`, `content`, `source_url`, `source_name`,
  `source_type`, `category`, `crawled_date`, `language` và `status`.
