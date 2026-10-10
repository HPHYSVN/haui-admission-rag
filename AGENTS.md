# Repository Agent Guidance

## Design Context

### Users
Thí sinh THPT và phụ huynh đang tìm hiểu tuyển sinh Đại học Công nghiệp Hà Nội
(HaUI), trên điện thoại hoặc laptop, cần câu trả lời ngắn gọn, có căn cứ và
nguồn chính thức để tiếp tục kiểm tra.

### Brand Personality
Tin cậy, bình tĩnh và chỉn chu. Dùng tiếng Việt mặc định, rõ ràng, tôn trọng,
không cường điệu hoặc tạo cảm giác quảng cáo.

### Aesthetic Direction
Liquid Glass × Luxury Minimalism: nền graphite, chữ ivory ấm, accent
champagne-gold tiết chế, typography biên tập và khoảng thở rộng. Chỉ dùng glass
cho chrome phụ; nội dung hội thoại và nguồn phải đủ đục để đọc rõ. Responsive,
reduced-motion friendly, có fallback khi blur không khả dụng. Tránh neon, glow
nặng, gradient dày, card bo tròn quá mức và motion trang trí.

### Implementation Defaults
Giữ pipeline RAG hiện có và corpus/schema; phần frontend mới ưu tiên React +
TypeScript strict + Vite, backend FastAPI + Pydantic + Uvicorn. Chưa có
component/form/table/virtualization library hoặc shadcn/Radix/Base UI baseline.
Precedence: existing stack, explicit user choices, then framework defaults.

### Design Principles
- Establish hierarchy with spacing, weight, and grouping before adding effects.
- Show one clear next action: ask, inspect a source, or retry.
- Use warm neutrals and restrained gold to signal trust, never decoration.
- Keep answers and provenance readable and opaque; glass belongs to secondary chrome.
- Be honest about evidence, uncertainty, errors, and unavailable services.

See [.better-web-ui.md](./.better-web-ui.md) for the canonical design context.
