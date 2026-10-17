# Ba sơ đồ RAG - bản tiếng Việt

Các sơ đồ dưới đây là phiên bản giản lược của ba sơ đồ trong [README.md](./README.md). Sơ đồ 1 mô tả chuẩn bị và đánh giá dữ liệu, sơ đồ 2 mô tả luồng đang chạy trong CLI, còn sơ đồ 3 là kiến trúc đề xuất cho tương lai, chưa phải chức năng hiện có.

## 1. Chuẩn bị dữ liệu và cải thiện kết quả

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"|, all subgraphs closed by end, ids unique -->
```mermaid
flowchart TD
    subgraph PrepareVi["Chuẩn bị dữ liệu"]
        viSources["Trang web và tệp dữ liệu"] --> viCollect["Thu thập dữ liệu"]
        viCollect --> viExtract["Trích xuất nội dung"]
        viExtract --> viNormalize["Chuẩn hóa và loại trùng"]
        viNormalize --> viReview["Rà soát tài liệu"]
        viReview --> viPropose["Tạo đề xuất chunk"]
        viPropose --> viApprove["Duyệt và chỉnh sửa chunk"]
        viApprove --> viFinal["Chunk đã duyệt"]
        viFinal --> viBuild["Tạo chỉ mục BM25 và FAISS"]
    end
    subgraph EvaluateVi["Đánh giá và cải thiện"]
        viQuestions["Câu hỏi và nguồn mong đợi"] --> viRun["Chạy retrieval evaluation"]
        viBuild --> viRun
        viRun --> viMetrics["Xem Recall at K và lỗi"]
        viMetrics --> viTuneChunk["Điều chỉnh chunk"]
        viMetrics --> viTuneSearch["Điều chỉnh retrieval"]
        viTuneChunk --> viPropose
        viTuneSearch --> viRun
    end
```

Tài liệu gốc và câu hỏi đánh giá được lưu riêng. Kết quả đánh giá giúp xác định nên sửa cách chia chunk hay cấu hình retrieval.

## 2. Luồng hỏi đáp hiện tại trong CLI

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"|, all subgraphs closed by end, ids unique -->
```mermaid
flowchart TD
    vi2Question["Câu hỏi"] --> vi2Filters["Lọc theo năm và danh mục"]
    vi2Filters --> vi2Eligible["Danh sách chunk phù hợp"]
    vi2Question --> vi2Bm25["Tìm kiếm BM25"]
    vi2Question --> vi2Faiss["Tìm kiếm vector FAISS"]
    vi2Chunks[("Chunk đã duyệt")]
    vi2Vectors[("Vectors và FAISS index")]
    vi2Model["Embedding model"]
    vi2Chunks --> vi2Bm25
    vi2Vectors --> vi2Faiss
    vi2Model --> vi2Faiss
    vi2Bm25 --> vi2Bm25Gate["Giữ kết quả trong danh sách phù hợp"]
    vi2Eligible --> vi2Bm25Gate
    vi2Faiss --> vi2FaissGate["Giữ kết quả trong danh sách phù hợp"]
    vi2Eligible --> vi2FaissGate
    vi2Bm25Gate --> vi2Fuse["Kết hợp điểm"]
    vi2FaissGate --> vi2Fuse
    vi2Fuse --> vi2Check["Kiểm tra độ liên quan"]
    vi2Check --> vi2Top["Chọn Top K chunk"]
    vi2Top --> vi2Prompt["Tạo context và prompt"]
    vi2Prompt --> vi2Vllm["vLLM"]
    vi2Vllm --> vi2Answer["Câu trả lời kèm nguồn"]
```

CLI lọc kết quả BM25 và FAISS theo danh sách chunk phù hợp, sau đó kết hợp điểm và kiểm tra độ liên quan. Nguồn trong câu trả lời được dựng từ các chunk tìm thấy.

## 3. Kiến trúc Backend và Frontend đề xuất

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"|, all subgraphs closed by end, ids unique -->
```mermaid
flowchart LR
    subgraph FutureUiVi["Giao diện"]
        vi3User["Sinh viên hoặc cán bộ"] --> vi3Chat["Chat UI"]
        vi3Chat --> vi3Api["Backend API"]
        vi3ReviewUi["Giao diện rà soát dữ liệu"] --> vi3AdminApi["Admin API"]
    end
    subgraph FutureBackendVi["Backend RAG"]
        vi3Api --> vi3Guard["Kiểm tra đầu vào và phiên"]
        vi3Guard --> vi3Query["Xử lý và phân loại câu hỏi"]
        vi3Query --> vi3Bm25["Tìm kiếm BM25"]
        vi3Query --> vi3Dense["Tìm kiếm vector FAISS"]
        vi3Bm25 --> vi3Fusion["Kết hợp thứ hạng"]
        vi3Dense --> vi3Fusion
        vi3Fusion --> vi3Rerank["Reranker"]
        vi3Rerank --> vi3Gate["Kiểm tra độ tin cậy"]
        vi3Gate --> vi3Context["Đóng gói context và nguồn"]
        vi3Context --> vi3Generate["LLM tạo câu trả lời"]
        vi3Generate --> vi3Verify["Kiểm tra trích dẫn"]
        vi3Verify --> vi3Response["Trả lời và nguồn"]
        vi3Response --> vi3Chat
        vi3AdminApi --> vi3Ingest["Nạp và rà soát dữ liệu"]
        vi3Ingest --> vi3Corpus[("Kho dữ liệu phiên bản hóa")]
        vi3Corpus --> vi3Index["Tạo chỉ mục"]
        vi3Index --> vi3Bm25
        vi3Index --> vi3Dense
        vi3ReviewUi --> vi3Eval["Đánh giá và theo dõi chất lượng"]
        vi3Eval --> vi3Corpus
    end
    subgraph FutureModelsVi["Dịch vụ mô hình"]
        vi3EmbedModel["Embedding model"]
        vi3RerankModel["Reranker model"]
        vi3LlmModel["LLM service"]
    end
    vi3Dense --> vi3EmbedModel
    vi3Rerank --> vi3RerankModel
    vi3Generate --> vi3LlmModel
```

Đây là hướng phát triển có thể bổ sung Frontend, API, reranking, kiểm tra trích dẫn và công cụ đánh giá; cần đo chất lượng và độ trễ trước khi triển khai.
