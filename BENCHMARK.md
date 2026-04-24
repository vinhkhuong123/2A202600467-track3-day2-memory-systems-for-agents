# Benchmark Lab 17: Multi-Memory Agent

**Học viên:** (Tên của bạn)
**Cấu hình Agent:** LangGraph với 4 loại memory backends (Short-term, Long-term FakeRedis, Episodic JSON, Semantic ChromaDB). Context Manager giới hạn budget token: 400.

---

## 1. Kết quả Benchmark 10 Scenarios (No-memory vs With-memory)

*Dữ liệu được trích xuất trực tiếp từ kết quả thực thi trong `benchmark_results.json`.*

| # | Scenario | No-memory result | With-memory result | Pass? |
|---|----------|------------------|---------------------|-------|
| 1 | Profile Recall (Tên) | "Xin lỗi, nhưng tôi không biết tên của bạn. Bạn có thể cho tôi biết tên của bạn không?" | "Tên của bạn là Linh." | Pass |
| 2 | Allergy conflict update | Trả lời chung chung các nguyên liệu cần tránh (Bột mì, đường, trứng...). Không biết người dùng bị dị ứng. | "Nếu tôi làm bánh cho bạn, tôi sẽ cần tránh các nguyên liệu có chứa đậu nành..." (Nhớ được fact mới nhất). | Pass |
| 3 | Episodic Recall (Debug) | Trả lời chung chung các bước debug (kiểm tra IP, cổng, tường lửa...). | "Bạn đã đề cập rằng để debug lỗi connection refused, bạn cần sử dụng tên dịch vụ Docker." | Pass |
| 4 | Semantic Retrieval (FAQ) | Hallucination: "LangGraph là... liên quan đến việc sử dụng các cấu trúc đồ thị..." | Lấy chính xác từ ChromaDB: "LangGraph là một thư viện được thiết kế để xây dựng các ứng dụng stateful và multi-actor với các mô hình ngôn ngữ lớn (LLMs)." | Pass |
| 5 | Profile Update (Nghề nghiệp) | "Tôi không có thông tin cụ thể về bạn, vì vậy tôi không thể biết nghề nghiệp hiện tại của bạn là gì." | "Nghề nghiệp hiện tại của bạn là lập trình viên." (Đã cập nhật từ giáo viên). | Pass |
| 6 | Episodic + Profile (Món ăn) | Gợi ý ngẫu nhiên: Cơm tấm, Phở, Bún chả, Mì Quảng... | "Nếu bạn thích ăn phở, có thể thử một tô phở bò hoặc phở gà..." (Nhớ sở thích). | Pass |
| 7 | Semantic Retrieval 2 | Giải thích chung về ChromaDB. | Lấy chính xác từ ChromaDB: "ChromaDB là một cơ sở dữ liệu vector mã nguồn mở, được thiết kế dành cho các ứng dụng AI." | Pass |
| 8 | Short-term Memory (Trim) | "Xin lỗi, nhưng tôi không có thông tin về những gì bạn đã nói trước đó." | "Bạn vừa kể tên quả táo màu đỏ đầu tiên." (Hoạt động tốt trong sliding window ngắn hạn). | Pass |
| 9 | Profile fact - Pets | "Xin lỗi, nhưng tôi không biết tên thú cưng của bạn." | "Thú cưng của bạn tên là Tom." (Lấy từ FakeRedis Long-term). | Pass |
| 10 | Context merging | Trả lời chung chung về việc học LangGraph, không biết tên user hay sở thích. | Merge mượt mà: "LangGraph là một thư viện... Nếu bạn thích lập trình AI... học LangGraph có thể là một lựa chọn tốt..." | Pass |

---

## 2. Reflection: Privacy & Limitations

### Privacy / Rủi ro
1. **Memory nào giúp agent nhất?** 
   - Long-term profile giúp cá nhân hóa cực tốt (nhớ tên, nghề nghiệp, sở thích).
2. **Memory nào rủi ro nhất nếu retrieve sai?**
   - **Long-term profile / Dị ứng:** Cực kỳ rủi ro (Scenario 2). Nếu hệ thống retrieve sai conflict hoặc ghi nhận sai dị ứng đậu nành thành dị ứng sữa bò, agent có thể đưa ra gợi ý gây hại trực tiếp đến sức khỏe người dùng (Medical/PII Data).
3. **Nếu user yêu cầu xóa memory, xóa ở backend nào?**
   - Phải thực hiện xóa toàn diện trên cả **4 backends**: Xóa key trong FakeRedis (profile), tìm và clear file `episodic.jsonl` (episodic), reset buffer (short-term) và xóa dữ liệu vector trong ChromaDB (semantic). Nếu chỉ xóa 1 nơi, dữ liệu nhạy cảm vẫn còn tồn đọng.
   - **Consent & TTL:** Hệ thống đã được tích hợp cơ chế TTL 7 ngày (604800 giây) cho các dữ liệu cá nhân lưu trong Long-term memory (FakeRedis). Cơ chế này giúp tự động dọn dẹp các thông tin nhạy cảm (PII) theo thời gian, giảm thiểu rủi ro bảo mật do lưu trữ vô thời hạn. Tuy nhiên, vẫn cần phải xin explicit consent từ người dùng trước khi lưu trữ.

### Limitation kỹ thuật của solution hiện tại
- **Priority-based Trim yếu:** Hiện tại Context Manager chỉ cắt gọt cứng nhắc dựa trên giới hạn token budget (400 tokens) bằng cách bỏ đi các episode cũ. Nếu Profile của user quá lớn, nó có thể chiếm hết sạch token khiến cho hệ thống không thể nạp thêm Semantic Knowledge vào prompt, dẫn tới giảm độ chính xác của RAG.
- **Save Memory bằng LLM Extraction:** Việc parse facts trong node `save_memory` hiện đang bắt chuỗi JSON thủ công. LLM đôi khi trả về format không chuẩn khiến cho `json.loads` thất bại (chỉ có thể pass bỏ qua). Trong hệ thống production, cần sử dụng *Structured Output (Function Calling)* hoặc thư viện Pydantic Output Parser để có cơ chế retry tự động.
