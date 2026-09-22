# Chatbot hỗ trợ tư vấn tuyển sinh đại học HaUI bằng LLM và RAG

Dự án này nhằm xây dựng một hệ thống chatbot hỗ trợ tư vấn tuyển sinh đại học của Trường Đại học Công nghiệp Hà Nội (HaUI), sử dụng các công nghệ hiện đại trong xử lý ngôn ngữ tự nhiên: mô hình ngôn ngữ lớn (LLM) và kỹ thuật truy xuất thông tin theo kiến thức (RAG - Retrieval Augmented Generation).

Mục tiêu của dự án không chỉ là trả lời câu hỏi đơn lẻ, mà là xây dựng một hệ thống tư vấn đáng tin cậy, dựa trên dữ liệu tuyển sinh chính thức, cập nhật và có nguồn gốc rõ ràng từ trường, Bộ GD&ĐT và các cổng thông tin nhà nước.

## Tổng quan dự án

Trường HaUI có lượng thông tin tuyển sinh rất đa dạng: đề án tuyển sinh, thông báo, phương thức xét tuyển, điểm chuẩn, quy định của Bộ GD&ĐT, học phí, chính sách hỗ trợ sinh viên, quy chế học tập, ký túc xá, đời sống sinh viên, v.v. Thông tin này thường nằm rải rác trên nhiều trang web, định dạng khác nhau (HTML, PDF, ảnh scan), và dễ gây khó khăn cho người học khi tìm kiếm thông tin.

Chatbot RAG được thiết kế để giải quyết vấn đề này bằng cách:

- thu thập dữ liệu tuyển sinh từ nguồn chính thức
- chuẩn hóa dữ liệu thành schema thống nhất
- loại bỏ dữ liệu trùng lặp và không hợp lệ
- lưu trữ dưới dạng tài liệu có cấu trúc
- truy xuất các đoạn liên quan dựa trên câu hỏi của người dùng
- kết hợp với LLM để tạo câu trả lời ngắn gọn, chính xác và dễ hiểu

## Vấn đề dự án muốn giải quyết

Hiện nay, thông tin tuyển sinh thường phân tán và thay đổi theo từng năm, trong khi người học thường gặp các câu hỏi như:

- Ngành nào phù hợp với điểm thi?
- Học phí và các chính sách hỗ trợ là gì?
- Có những phương thức tuyển sinh nào?
- Điểm chuẩn của năm nay/ngày trước là bao nhiêu?
- Cách đăng ký xét tuyển, hồ sơ cần chuẩn bị như thế nào?
- Chế độ học, ký túc xá, học bổng, hỗ trợ sinh viên có như thế nào?
- Quy định của Bộ GD&ĐT về tuyển sinh có thay đổi hay không?

Chatbot cần phải trả lời dựa trên thông tin chính thức, không suy đoán, không dùng nguồn không đáng tin cậy và không làm sai lệch mục tiêu tuyển sinh.

## Kiến trúc tổng thể

Dự án được xây dựng theo mô hình RAG chuẩn:

1. Thu thập dữ liệu
   - Crawl dữ liệu từ website chính thức của HaUI, Bộ GD&ĐT và văn bản nhà nước
   - Lưu HTML/PDF gốc vào `data/raw/`

2. Xử lý dữ liệu
   - Làm sạch văn bản
   - Trích xuất metadata: tiêu đề, năm, số văn bản, ngày công bố, danh mục, nguồn
   - Kiểm tra schema và dữ liệu đầu vào
   - Loại bỏ trùng lặp

3. Lưu trữ tri thức
   - Chuyển đổi dữ liệu sang JSONL theo schema chuẩn
   - Lưu tại `data/documents/*.jsonl`

4. Truy xuất
   - Tạo vector index cho tài liệu (sau giai đoạn corpus sạch)
   - Tìm kiếm các document phù hợp với câu hỏi của người dùng

5. Tạo câu trả lời
   - Kết hợp ngữ cảnh truy xuất với LLM
   - Trả lời theo phong cách tư vấn tuyển sinh, rõ ràng, chính xác, có căn cứ



## Nguyên tắc dữ liệu

Dự án tuân thủ các nguyên tắc sau để đảm bảo tri thức chất lượng, đáng tin cậy và an toàn cho chatbot:

- Chỉ dùng nguồn chính thức: HaUI, Bộ GD&ĐT, cơ quan nhà nước
- Không dùng diễn đàn, mạng xã hội, fanpage chưa xác minh
- Dữ liệu phải là document văn bản / thông báo, không phải tin tức tổng hợp
- Tránh lưu các phần nhiễu như menu, banner, footer, quảng cáo
- Với các trang điểm chuẩn có dạng ảnh scan, không OCR sai và không đưa nội dung nhận diện không chắc chắn vào corpus; thay vào đó ghi rõ thông tin ảnh gốc và URL ảnh chính thức
- Giữ nguyên định dạng JSONL và schema chuẩn
- Mỗi document phải có `id`, `title`, `content`, `source_url`, `source_name`, `source_type`, `category`, `crawled_date`, `language`, `status`
- Không thêm trường tùy ý ngoài schema chuẩn


