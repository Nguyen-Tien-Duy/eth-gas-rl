# Reward Function Audit: Report vs. Code

Tài liệu này đối chứng sự khác biệt giữa mô tả trong Báo cáo (Thesis/Report) và thực tế triển khai trong mã nguồn hiện tại.

## 1. So sánh Công thức Tổng quát

| Thành phần | Báo cáo (Section 3.8) | Code (Builder/JAX) | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Logic Urgency** | **Tăng dần** theo thời gian ($e^{\alpha(1-\tau)}$) | **Giảm dần** ($e^{\alpha(1-t/H)}$) | 🔴 **Lỗi Logic nghiêm trọng** |
| **Gas Cost** | Có tính Marginal Cost ($C_{mar} \cdot n_t$) | Chỉ tính chênh lệch Gas | 🟡 Thiếu chi tiết |
| **Hệ số Chặn** | Có Overhead ($C_{base} \cdot \mathbf{1}[n>0]$) | Có Overhead | ✅ Khớp |
| **Reward Scale** | $\sigma = 10^9$ (Gwei to ETH) | $Scale = 100$ | 🟡 Khác biệt về độ lớn |

## 2. Chi tiết các sai lệch "Chết người"

### A. Lỗi đảo ngược tính cấp bách (Urgency Inversion)
*   **Trong Báo cáo:** Khi sắp đến Deadline, hình phạt phải tăng vọt để ép Agent xả hàng.
*   **Trong Code Builder:** Công thức `np.exp(alpha * (1.0 - time_ratio))` lại khiến hình phạt **lớn nhất ở đầu Episode** và **nhỏ nhất ở cuối**. 
*   **Hậu quả:** Oracle (bộ giải tối ưu) bị dạy rằng "càng về cuối càng không sợ bị phạt", dẫn đến dữ liệu mẫu bị sai lệch hoàn toàn so với mục tiêu tối ưu.

### B. Sự khác biệt về Tham số (Hyperparameters)

| Tham số | Giá trị trong Báo cáo | Giá trị trong Code Builder | Chênh lệch |
| :--- | :--- | :--- | :--- |
| **$\alpha$ (Exponent)** | 3.0 | 2.0 | 1.5x |
| **$\beta$ (Weight)** | 100.0 | 0.1 | **1000x** |
| **$\lambda$ (Deadline)** | 5,000,000,000 | 100.0 | **50,000,000x** |

> [!CAUTION]
> **Nhận xét khoa học:** Sự chênh lệch $10^7$ lần về `lambda` (phạt deadline) là lý do tại sao Agent trong code hiện tại "không hề sợ deadline" và sẵn sàng để Backlog lên tới hàng nghìn giao dịch. Trong báo cáo, hình phạt này cực nặng để đảm bảo an toàn tuyệt đối.

## 3. Đề xuất chỉnh sửa để đồng bộ (Action Plan)

Để Agent JAX có thể đạt được kết quả 9.5% như trong báo cáo, chúng ta cần:
1.  **Sửa Builder:** Đổi thành `np.exp(alpha * time_ratio)` và nâng các hệ số $\beta, \lambda$ lên đúng tầm vóc trong báo cáo.
2.  **Thêm Marginal Cost:** Đưa $C_{mar} = 15,000$ vào hàm tính Reward để Agent cân nhắc chi phí gas thực tế trên mỗi unit giao dịch.
3.  **Đồng bộ Scale:** Sử dụng chung một hệ số scale $\sigma$ để các chỉ số Loss khi training dễ quan sát hơn.

---
**Kết luận:** Hiện tại Code và Báo cáo đang nói hai ngôn ngữ khác nhau. Việc đồng bộ hóa là **BẮT BUỘC** để kết quả thực nghiệm có giá trị khoa học.
