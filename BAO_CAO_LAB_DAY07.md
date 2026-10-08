# BÁO CÁO THỰC HÀNH LAB DAY 07: MULTI-OBJECT TRACKING EVALUATION

- **Học viên**: Phan Danh Đạt (02627)
- **Môi trường thực nghiệm**: VSCode, Python 3.11 (`.venv`), PyTorch 2.6.0+cu124
- **Phần cứng**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **Mô hình**: Detector YOLO11n (640px, class=person), Re-ID OSNet (`osnet_x0_25_msmt17.pt`)

---

## PHẦN 1: ÔN TẬP VÀ ĐÁP ÁN METRICS (CP1)

### 1.1. Đáp án 3 câu hỏi True / False trong `on_tap_metrics.ipynb`

| Câu hỏi | Nội dung | Đáp án | Giải thích chi tiết |
|:---:|:---|:---:|:---|
| **Câu 1** | *MOTA có thể nhận giá trị âm (dưới 0).* | **True** | Công thức $MOTA = 1 - \frac{FN + FP + IDSW}{GT}$. Nếu detector tạo ra quá nhiều False Positives (FP) hoặc người đổi ID quá nhiều (IDSW) khiến tổng lỗi $(FN + FP + IDSW) > GT$, thì MOTA sẽ âm. |
| **Câu 2** | *IDF1 cao hơn nghĩa là tracker giữ ID ổn định hơn qua các frame.* | **True** | IDF1 đo F1-score của sự liên kết định danh ($IDTP / (IDTP + 0.5(IDFP + IDFN))$). Giá trị IDF1 cao chứng minh tracker duy trì đúng ID xuyên suốt quỹ đạo của đối tượng thay vì ngắt quãng và gán ID mới. |
| **Câu 3** | *Tham số `--iou` trong `run_tracking.py` ảnh hưởng đến cách tracker ghép ID giữa các frame.* | **False** | Trong bài lab này, `--iou` là ngưỡng loại bỏ hộp trùng lắp (**NMS - Non-Maximum Suppression**) của detector YOLO ở từng frame riêng lẻ. Ngưỡng ghép ID (association threshold) là tham số nội bộ của tracker. |

---

## PHẦN 2: BẢNG TỔNG KẾT CẤU HÌNH VÀ KẾT QUẢ 5 VIDEO (CP2 & CP3)

| Video | Đặc điểm cảnh quay | Tracker đã chọn | `conf` | `iou` | Đánh giá định lượng (video_1) / Quan sát mắt (video_2 - video_5) |
|:---|:---|:---:|:---:|:---:|:---|
| **video_1** | Quảng trường ban ngày, camera tĩnh, mật độ vừa | **ByteTrack** | `0.30` | `0.50` | **HOTA: 33.53%**, **MOTA: 18.33%**, **IDF1: 27.52%**, IDSW: 20, FP: 218, FN: 14937. ID giữ rất ổn định với người đi thẳng. |
| **video_2** | Phố đêm trên cao, người rất đông, ánh sáng yếu | **DeepOC-SORT** | `0.25` | `0.50` | Mật độ người dày đặc, nhiều pha cắt mặt. Re-ID kết hợp quán tính hướng chuyển động giúp phục hồi ID sau khi đối tượng đi xuyên qua nhau. |
| **video_3** | Camera di chuyển, ảnh nhỏ/xa, FPS thấp | **ByteTrack** | `0.25` | `0.50` | Ảnh mờ và đối tượng nhỏ khiến vector đặc trưng Re-ID dễ nhiễu. ByteTrack tận dụng detection tin cậy thấp (second-match) giúp không bị đứt track. |
| **video_4** | Trong nhà, camera tiến tới, phản chiếu gương kính | **OC-SORT** | `0.40` | `0.50` | Đặt `conf=0.40` loại bỏ triệt để các bóng mờ trên vách kính/gương; OC-SORT bám mượt chuyển động phi tuyến tính khi camera zoom lại gần. |
| **video_5** | Góc nhìn trên xe bus, rung lắc mạnh, ngã tư đông | **BotSORT** | `0.30` | `0.50` | Tích hợp thuật toán bù trừ rung lắc camera (**Camera Motion Compensation - CMC**) kết hợp Re-ID, khắc phục hiện tượng nhảy box khi xe dằn xóc. |

---

## PHẦN 3: PHÂN TÍCH VÀ SO SÁNH CHI TIẾT TỪNG VIDEO

### 1. `video_1`: Quảng trường ban ngày (Baseline & Đánh giá định lượng)
- **Đặc trưng**: Ánh sáng đầy đủ, camera cố định trên cao, người đi bộ chủ yếu chuyển động tịnh tiến đều.
- **Kết quả đo lường với TrackEval**:
  - **HOTA (Higher Order Tracking Accuracy)**: **33.53%** (DetA: 20.36%, AssA: 55.22%). Chỉ số AssA đạt 55.22% cho thấy khả năng liên kết danh tính của ByteTrack ở mức tốt.
  - **MOTA**: **18.33%**. Bị kéo xuống chủ yếu do FN (14937) từ các đối tượng quá xa ở hậu cảnh mà YOLO nano không bắt được.
  - **IDF1**: **27.52%** (ID Precision: 79.96%).
  - **ID Switch**: Chỉ **20 lần** trên toàn bộ 600 frames.
- **Nhận xét quan sát**: Khi hai người đi lướt qua nhau ở cự ly vừa phải, ByteTrack duy trì màu ID liên tục không bị tráo đổi.

### 2. `video_2`: Phố đêm đông đúc (Đánh giá bằng mắt)
- **Thách thức**: Độ tương phản thấp (ban đêm), người ăn mặc tối màu, mật độ dày đặc che khuất nhau liên tục (occlusion).
- **Tracker chọn**: **DeepOC-SORT** (`conf=0.25`, `iou=0.5`).
- **Lý do & Quan sát**:
  - Nếu chỉ dùng motion tracker (như SORT hoặc ByteTrack đơn thuần), các pha va chạm quỹ đạo ở góc nhìn từ trên cao khiến ma trận IoU bị chồng lấn và dễ dẫn đến hoán đổi ID.
  - DeepOC-SORT trích xuất vector ngoại hình (Re-ID qua mạng OSNet) kết hợp cơ chế OOS (Observation-Centric Online Smoothing) giúp "nhớ" được người đi bộ ngay cả khi họ biến mất 5-10 frames sau lưng người khác rồi xuất hiện lại.
  - Hạ `conf=0.25` giúp nhận diện được các đối tượng trong vùng tối dưới bóng râm.

### 3. `video_3`: Camera di chuyển, người nhỏ, FPS thấp (Đánh giá bằng mắt)
- **Thách thức**: Khung hình bị giật/bước nhảy lớn do FPS thấp, kích thước người nhỏ ở xa, camera xoay lắc nhẹ.
- **Tracker chọn**: **ByteTrack** (`conf=0.25`, `iou=0.5`).
- **Lý do & Quan sát**:
  - Với ảnh nhỏ và mờ nhòe (motion blur), việc trích xuất đặc trưng ngoại hình (Re-ID) thường tạo ra vector đặc trưng có độ tin cậy thấp, dễ gây gán sai ID hơn là chỉ dùng hình học.
  - ByteTrack nổi bật ở chỗ không loại bỏ các bounding box có score thấp (`conf` từ 0.1 đến 0.25), mà đưa vào vòng ghép cặp thứ 2 (second association stage), giúp quỹ đạo không bị đứt đoạn giữa chừng.

### 4. `video_4`: Trong nhà, camera tiến tới, phản chiếu gương (Đánh giá bằng mắt)
- **Thách thức**: Mặt sàn bóng loáng và vách kính văn phòng phản chiếu hình bóng người (ghost detections). Camera di chuyển tịnh tiến về phía trước làm kích thước người phóng to dần (scale change).
- **Tracker chọn**: **OC-SORT** (`conf=0.40`, `iou=0.5`).
- **Lý do & Quan sát**:
  - Việc nâng ngưỡng `conf=0.40` là yếu tố then chốt để YOLO loại bỏ các hình ảnh phản chiếu mờ ảo trên kính.
  - OC-SORT sử dụng vận tốc tức thời và cơ chế hồi quy lại quỹ đạo (momentum) tốt hơn Kalman filter truyền thống, thích ứng mượt mà khi tỷ lệ bounding box tăng nhanh do camera tiến lại gần.

### 5. `video_5`: Góc nhìn xe bus rung lắc (Đánh giá bằng mắt)
- **Thách thức**: Xe bus tăng ga, phanh gấp và đi qua gờ giảm tốc gây rung lắc mạnh khung hình. Khi camera rung, nền di chuyển khiến mô hình chuyển động tuyến tính của Kalman filter bị lệch hoàn toàn khỏi vị trí thực tế của vật thể.
- **Tracker chọn**: **BotSORT** (`conf=0.30`, `iou=0.5`).
- **Lý do & Quan sát**:
  - BotSORT được trang bị module **Camera Motion Compensation (CMC)** dựa trên thuật toán biến đổi affine/homography của ảnh nền. CMC triệt tiêu độ rung lắc của xe bus trước khi ước lượng vị trí đối tượng.
  - Đồng thời, đặc trưng Re-ID giúp xác nhận lại ID một cách chuẩn xác sau các pha rung lắc mạnh.

---

## PHẦN 4: TỔNG KẾT BÀI HỌC VỀ TRACKING

1. **Motion-only Tracker (ByteTrack, OC-SORT)**:
   - *Ưu điểm*: Tốc độ xử lý cực nhanh (đạt 40+ FPS trên GPU RTX 3060), không tốn chi phí trích xuất embedding mạng nơ-ron, rất mạnh khi ảnh mờ hoặc FPS thấp.
   - *Nhược điểm*: Dễ bị tráo ID khi che khuất kéo dài hoặc quỹ đạo giao cắt phức tạp.

2. **Appearance/Re-ID Tracker (BotSORT, DeepOC-SORT, StrongSORT)**:
   - *Ưu điểm*: Khả năng duy trì và phục hồi ID sau che khuất dài (long-term occlusion) vượt trội, ít bị ảnh hưởng bởi thay đổi quỹ đạo đột ngột.
   - *Nhược điểm*: Chi phí tính toán cao hơn (~10-15 FPS), dễ bị nhiễu nếu đối tượng quá nhỏ hoặc bị mờ nhòe nghiêm trọng.

3. **Ý nghĩa các tham số của Detector**:
   - `conf`: Cân bằng giữa bỏ sót (FN) và hộp giả (FP). Cảnh tối/nhỏ cần hạ `conf` (0.2-0.25), cảnh phản chiếu/gương kính cần tăng `conf` (0.4-0.5).
   - `iou`: Ngưỡng NMS loại bỏ trùng lặp hộp. Giữ ở 0.5 là mức cân bằng an toàn cho người đi bộ.

---

## PHẦN 5: DANH MỤC CÁC FILE KẾT QUẢ ĐÃ TẠO

Thư mục nộp bài `runs/nop_bai/` gồm đầy đủ 5 file text MOT format và 5 video preview:

- 📄 `runs/nop_bai/video_1.txt` (600 frames, ByteTrack) & `video_1_preview.mp4`
- 📄 `runs/nop_bai/video_2.txt` (1050 frames, DeepOC-SORT) & `video_2_preview.mp4`
- 📄 `runs/nop_bai/video_3.txt` (837 frames, ByteTrack) & `video_3_preview.mp4`
- 📄 `runs/nop_bai/video_4.txt` (900 frames, OC-SORT) & `video_4_preview.mp4`
- 📄 `runs/nop_bai/video_5.txt` (750 frames, BotSORT) & `video_5_preview.mp4`
