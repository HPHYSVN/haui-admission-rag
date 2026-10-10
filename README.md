# Chatbot tư vấn tuyển sinh HaUI bằng LLM và RAG

Đây là hệ thống chatbot tư vấn tuyển sinh cho Trường Đại học Công nghiệp Hà
Nội (HaUI). Hệ thống kết hợp kho tài liệu chính thức, truy xuất lai BM25 +
embedding đa ngôn ngữ và mô hình ngôn ngữ lớn chạy qua API tương thích vLLM.

Mục tiêu của dự án là trả lời ngắn gọn, có căn cứ và có nguồn tham khảo; không
suy đoán khi kho dữ liệu không đủ bằng chứng.

## Tính năng

- Thu thập và chuẩn hóa tài liệu tuyển sinh từ HaUI, Bộ GD&ĐT và các cổng
  thông tin nhà nước.
- Kiểm tra tài liệu theo JSON Schema và loại bỏ tài liệu trùng hoặc không hợp
  lệ.
- Review tài liệu và chunk thủ công trước khi đưa vào truy xuất.
- Truy xuất kết hợp BM25 và embedding multilingual.
- Lọc nghiêm ngặt theo năm và danh mục khi câu hỏi có yêu cầu cụ thể.
- Sinh câu trả lời qua máy chủ vLLM tương thích OpenAI.
- Hiển thị nguồn được lấy trực tiếp từ các chunk đã truy xuất.
- Giao diện React/Vite tiếng Việt, responsive, có trạng thái lỗi và trạng thái
  kết nối.
- Lịch sử cuộc trò chuyện được lưu cục bộ trong trình duyệt bằng
  `localStorage`; không lưu nội dung hội thoại trên backend.

## Kiến trúc

```text
Trình duyệt
  -> React/Vite :5173
  -> FastAPI :8001
  -> RAG pipeline hiện có
  -> vLLM :8000/v1
```

Luồng xử lý câu hỏi:

1. Người dùng gửi câu hỏi từ giao diện.
2. FastAPI kiểm tra độ dài, lịch sử và định dạng request.
3. Bộ lọc metadata xác định các tài liệu có thể dùng.
4. BM25 và FAISS/embedding xếp hạng các chunk liên quan.
5. Pipeline đóng gói context trong ngân sách ký tự đã giới hạn.
6. vLLM sinh câu trả lời dựa trên context.
7. Backend tạo danh sách nguồn từ chunk truy xuất, không để model tự tạo URL.
8. Frontend hiển thị câu trả lời, nguồn, trạng thái và lỗi phù hợp.

## Cấu trúc chính

```text
app/                    FastAPI, schema và service API
crawler/                Crawler và bộ làm sạch dữ liệu
data/documents/         Tài liệu chuẩn hóa JSONL
data/chunks/final/      Chunk đã review và phê duyệt
data/indexes/           Chỉ mục vector cục bộ, không commit
frontend/               Ứng dụng React + TypeScript + Vite
rag/                    BM25, embedding, pipeline và CLI
schemas/                JSON Schema cho corpus
scripts/                Công cụ chunk, validate, merge và review
tests/                  Test RAG và test API
```

## Yêu cầu

- Linux, macOS hoặc WSL.
- Python 3.11 trở lên.
- `uv`.
- Node.js và npm.
- Máy chủ vLLM đang chạy model tương thích OpenAI.
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
# API tương thích OpenAI của vLLM; cổng 8000 dành cho vLLM.
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4
VLLM_API_KEY=

# Model embedding đa ngôn ngữ, tải vào cache cục bộ khi dùng lần đầu.
RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RAG_ALPHA=0.5
RAG_SCORE_THRESHOLD=0.08
RAG_MIN_TERM_COVERAGE=0.3

# Cấu hình FastAPI.
APP_ALLOWED_ORIGINS=http://localhost:5173
APP_MAX_QUESTION_LENGTH=2000
APP_MAX_HISTORY_MESSAGES=8
RAG_TOP_K=4
```

Không commit `.env`, API key, token hoặc thông tin xác thực thật.

## Chạy hệ thống

### 1. Chạy vLLM

Giữ vLLM ở cổng 8000 theo `VLLM_BASE_URL`. Kiểm tra:

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

Kết quả sẵn sàng có dạng:

```json
{
  "status": "ready",
  "rag_loaded": true,
  "model_configured": true
}
```

Tài liệu OpenAPI có tại <http://localhost:8001/docs>.

### 3. Chạy frontend

Mở terminal thứ ba:

```bash
cd frontend
npm run dev
```

Mở <http://localhost:5173>.

Vite tự chuyển tiếp `/api` đến `http://localhost:8001`. Nếu triển khai frontend
ở origin khác, đặt biến `VITE_API_URL` trước khi build.

## API

### `GET /api/v1/health`

Trả về trạng thái sẵn sàng tối giản, không tiết lộ chẩn đoán nội bộ hoặc secret.

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

Review tương tác:

```bash
uv run python -m rag.cli review
```

Kiểm tra corpus:

```bash
uv run python scripts/validate.py
```

Quy trình dữ liệu:

1. Crawler lưu HTML/PDF gốc vào `data/raw/`.
2. Cleaner chuẩn hóa văn bản và metadata.
3. Tài liệu chuẩn được lưu trong `data/documents/*.jsonl`.
4. Review tài liệu xác nhận nội dung.
5. Chunk proposal được tạo và review.
6. Chunk cuối được lưu trong `data/chunks/final/`.
7. Chỉ mục BM25 và vector được tạo khi truy vấn.

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

Build frontend và kiểm tra dependency:

```bash
cd frontend
npm run build
npm audit
```

## Bảo mật và giới hạn triển khai

- Input người dùng được kiểm tra bằng Pydantic và giới hạn kích thước.
- CORS chỉ cho phép origin được khai báo trong `APP_ALLOWED_ORIGINS`.
- API không expose vLLM trực tiếp cho browser.
- Lỗi nội bộ không được hiển thị nguyên văn cho người dùng.
- Prompt coi user message và retrieved document là dữ liệu không tin cậy.
- Nguồn được tạo từ retrieval result, không từ output tự do của model.
- Frontend hiển thị plain text và không dùng `dangerouslySetInnerHTML`.
- API hiện chưa có xác thực, rate limit hoặc lưu hội thoại server-side.

Không nên public API trực tiếp ra Internet. Khi triển khai thật cần bổ sung
reverse proxy, HTTPS, rate limiting, authentication/authorization, logging có
kiểm soát và chính sách retention.

## Khắc phục sự cố

### Cổng 8000 đã được sử dụng

Thông thường đó là vLLM. Kiểm tra:

```bash
ss -ltnp 'sport = :8000'
curl http://localhost:8000/v1/models
```

Giữ vLLM ở 8000 và chạy FastAPI ở 8001.

### FastAPI báo `degraded`

Kiểm tra:

- `data/chunks/final/` có file JSONL.
- Workflow state đã đánh dấu chunk là `final` hoặc `indexed`.
- `.env` có `VLLM_MODEL`.
- vLLM trả lời ở `/v1/models`.

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
