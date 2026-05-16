# Offline RL Dataset Quality Assessment Framework

Tài liệu này liệt kê các tiêu chí đánh giá và biểu đồ khoa học chuẩn mực để kiểm tra chất lượng bộ dữ liệu Offline RL trước khi huấn luyện Agent.

## 1. Các Tiêu chí Định lượng (Quantitative Metrics)

| # | Tiêu chí | Ý nghĩa | Công thức / Cách tính |
|---|----------|---------|----------------------|
| 1 | **Trajectory Quality (TQ)** | Đo mức độ "giỏi" trung bình của dữ liệu | $TQ = \frac{\bar{R}_{dataset} - R_{random}}{R_{expert} - R_{random}}$ |
| 2 | **State-Action Coverage (SACo)** | Đo mức độ đa dạng của không gian trạng thái | Số lượng cặp (s,a) duy nhất / Tổng số cặp |
| 3 | **MIP Gap** | Đo chất lượng lời giải Oracle | Đã có: Mean=0.05%, P95=0.22% ✅ |
| 4 | **Reward Range** | Phạm vi phần thưởng, phát hiện outlier | $[R_{min}, R_{max}]$ per policy |
| 5 | **Queue Clearance Rate** | Tỷ lệ Episode xả hết mempool | $\%\{Q_{final} = 0\}$ per policy |

## 2. Các Biểu đồ Khoa học (Scientific Visualizations)

### A. Reward Distribution by Policy (Violin Plot / Box Plot)
**Mục đích:** So sánh phân phối reward giữa 4 policy (Expert, Noisy, Heuristic, Random).
**Kỳ vọng:** Expert > Noisy > Heuristic > Random. Nếu Heuristic đôi khi vượt Expert, hàm reward có vấn đề.
```
Biểu đồ: seaborn.violinplot(x='policy_type', y='episode_return', data=df)
```

### B. State Coverage Heatmap (t-SNE / PCA Projection)
**Mục đích:** Chiếu 14 chiều state xuống 2D để xem các policy phủ không gian trạng thái như thế nào.
**Kỳ vọng:** Expert tập trung ở vùng "tối ưu", Random phủ rộng nhất, Heuristic ở giữa.
```
Biểu đồ: sklearn.manifold.TSNE → scatter plot, color = policy_type
```

### C. Action Distribution Histogram (Per Policy)
**Mục đích:** Xem Agent Expert đang "ưa thích" hành động nào (xả bao nhiêu % queue).
**Kỳ vọng:** Expert có phân phối bimodal (hoặc 0% hoặc 100%), Random có phân phối đều.
```
Biểu đồ: plt.hist(actions, bins=50, alpha=0.5) cho mỗi policy
```

### D. Episode Return vs. Queue Clearance (Scatter Plot)
**Mục đích:** Kiểm tra mối quan hệ giữa tổng reward và khả năng xả hết mempool.
**Kỳ vọng:** Tương quan dương mạnh — Episode có reward cao phải có $Q_{final}$ thấp.
```
Biểu đồ: plt.scatter(episode_return, final_queue, c=policy_type)
```

### E. Gas Price vs. Action Heatmap (Decision Boundary)
**Mục đích:** Xem Agent Expert ra quyết định như thế nào dựa trên giá gas.
**Kỳ vọng:** Khi gas thấp (< Gas_ref), Expert xả mạnh. Khi gas cao, Expert chờ đợi.
```
Biểu đồ: plt.scatter(gas_price, action, c=queue_size, cmap='coolwarm')
```

### F. Temporal Reward Profile (Line Plot)
**Mục đích:** Xem reward thay đổi như thế nào theo thời gian trong 1 Episode.
**Kỳ vọng:** Reward giảm dần về cuối do Urgency tăng, trừ khi Agent xả kịp.
```
Biểu đồ: plt.plot(time_step, reward) cho 1 episode mẫu, mỗi policy 1 đường
```

### G. Feature Correlation Matrix (Heatmap)
**Mục đích:** Kiểm tra tính độc lập của 14 đặc trưng.
**Kỳ vọng:** Không có cặp nào tương quan > 0.95 (redundancy). Nếu có, cần loại bỏ.
```
Biểu đồ: seaborn.heatmap(df[feature_cols].corr(), annot=True)
```

## 3. Thứ tự ưu tiên đề xuất

1. **[BẮT BUỘC]** Reward Distribution by Policy (A) — Kiểm tra xem Oracle có thực sự "giỏi hơn" không.
2. **[BẮT BUỘC]** Action Distribution (C) — Kiểm tra hành vi Expert có hợp lý không.
3. **[QUAN TRỌNG]** Episode Return vs Queue (D) — Xác nhận Reward-Safety alignment.
4. **[QUAN TRỌNG]** Feature Correlation (G) — Loại bỏ feature thừa trước khi train.
5. **[NÊN CÓ]** State Coverage t-SNE (B) — Trực quan hóa cho báo cáo.
6. **[NÊN CÓ]** Gas vs Action Heatmap (E) — Cho phần trình bày.
7. **[TÙY CHỌN]** Temporal Profile (F) — Debug nếu Agent học sai.
