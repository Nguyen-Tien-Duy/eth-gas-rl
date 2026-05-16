# Data Generation & Reward Specification (Updated)

Tài liệu này hệ thống hóa phương pháp tạo dữ liệu Offline RL và cấu trúc phần thưởng kinh tế cho dự án tối ưu hóa phí gas Ethereum. Hệ thống hiện đã được đồng bộ hoàn toàn với báo cáo Final Project.

## 1. Chiến lược thu thập dữ liệu (Behavior Policy Mixing)

Dữ liệu huấn luyện (Training Set) được trộn theo tỷ lệ:
*   **40% Expert Policy:** `solve_episode_milp` - Tối ưu toàn cục.
*   **30% Noisy Expert:** `Expert + Gaussian(0, 5.0)` - Tăng tính Robustness.
*   **20% Heuristic Policy:** **Cheap Gas** ($P_t < P_{ref}$) hoặc **Panic Mode** ($t > 0.9H$).
*   **10% Random Policy:** `t % (min(Q, C) + 1)` - Phủ không gian trạng thái.

## 2. Cấu trúc Reward 3 tầng (Certified Version)

Hàm Reward sử dụng hệ số Scale $\sigma = 10^9$ để chuyển đổi đơn vị từ Gwei sang ETH, đảm bảo tính ổn định khi huấn luyện.

### Tầng 1: Hiệu quả kinh tế (Efficiency Tier)
$$R_{eff} = \frac{(n \times Gas_{ref}) - (C_{base} \cdot \mathbf{1}[n>0] + C_{mar} \cdot n) \times Gas_{curr}}{\sigma}$$
*   **C_base:** 21,000 (Gas fixed overhead)
*   **C_mar:** 15,000 (Gas marginal cost per transaction)
*   **Mục tiêu:** Tối đa hóa lợi nhuận ròng sau khi trừ chi phí thực thi.

### Tầng 2: Tính cấp bách (Urgency Tier)
$$R_{urg} = \frac{\beta}{\sigma} \times Queue \times e^{\alpha \cdot \frac{t}{H}}$$
*   **Hằng số:** $\beta = 100.0, \alpha = 3.0$
*   **Ý nghĩa:** Hình phạt **tăng mạnh theo hàm mũ** khi thời gian trôi về cuối Episode ($t \to H$).

### Tầng 3: Thảm họa (Catastrophe Tier)
$$R_{cat} = \frac{\lambda_d}{\sigma} \times \max(0, Queue_{final})$$
*   **Hằng số:** $\lambda_d = 5,000,000,000$ (5 tỷ)
*   **Mục tiêu:** Ép buộc hàng đợi phải được giải tỏa hoàn toàn tại block cuối cùng.

---
**Tổng hợp Reward:** $Total = R_{eff} - R_{urg} - R_{cat}$

> [!IMPORTANT]
> **TÍNH ĐỒNG BỘ:** Toàn bộ công thức trên đã được triển khai nhất quán trong cả bộ tạo dữ liệu (`builder.py`) và môi trường huấn luyện JAX (`physics_jax.py`).
