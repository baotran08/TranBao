# 📈 Ứng Dụng Tối Ưu Hóa Danh Mục Đầu Tư (EMA + RSI + PSO)

Ứng dụng web được xây dựng trên nền tảng **Streamlit** giúp nhà đầu tư phân tích dữ liệu lịch sử giá cổ phiếu, tự động tìm kiếm tham số tối ưu thông qua thuật toán tối ưu hóa bầy đàn (PSO), xếp hạng chọn lọc danh mục cổ phiếu và thực hiện kiểm định lịch sử (Backtest) với nhiều cơ chế quản lý vốn.

---

## ✨ Các Tính Năng Chính

1. **📊 Quản lý & Trực quan hóa Dữ liệu:** Hỗ trợ đọc tệp dữ liệu mặc định `HOSE_2020_2023.csv` hoặc tải lên tệp dữ liệu CSV tùy chỉnh. Vẽ biểu đồ so sánh tương tác đa mã.
2. **🧬 Tối ưu hóa tham số (PSO):** Tự động tìm kiếm bộ tham số (EMA Fast/Slow, RSI Wilder Window, RSI Buy/Sell limits, Stop Loss) tối ưu hóa chỉ số Sharpe trên tập dữ liệu học In-Sample.
3. **🏅 Tuyển chọn cổ phiếu:** Xếp hạng toàn bộ cổ phiếu theo chỉ số Sharpe / Lợi nhuận và chọn ra Top N cổ phiếu xuất sắc nhất để đưa vào danh mục.
4. **📈 Backtest Out-of-Sample (OOS):** Mô phỏng chi tiết chiến lược giao dịch trong giai đoạn 2021–2023, tự động tái cân bằng danh mục (Rebalancing) theo định kỳ (tháng, quý, năm) với các cơ chế tỷ trọng đều (Equal) hoặc tỷ trọng theo hiệu quả (Performance).
5. **🔍 Phân tích chi tiết từng cổ phiếu:** Hiển thị điểm mua (Buy), điểm bán (Sell) và điểm cắt lỗ thực tế (Stop Loss) trên biểu đồ giá của từng mã cổ phiếu cụ thể.
6. **🧪 Kiểm định ý nghĩa thống kê:** Chạy các kiểm định t-test và Wilcoxon signed-rank test để xác minh tính vượt trội của chiến lược so với VN-Index và Buy & Hold.

---

## 📁 Cấu Trúc Thư Mục Dự Án

Để deploy lên GitHub và Streamlit Cloud, thư mục của bạn nên chứa các tệp cơ bản sau:
```
Final/
├── app.py                 # Mã nguồn giao diện và logic chính của ứng dụng Streamlit
├── requirements.txt       # Danh sách các thư viện Python cần thiết
├── README.md              # Tài liệu hướng dẫn sử dụng và cài đặt này
└── HOSE_2020_2023.csv     # Tệp dữ liệu lịch sử cổ phiếu (giai đoạn 2020-2023)
```

---

## 💻 Hướng Dẫn Chạy Cục Bộ (Local)

### Cách 1: Sử dụng bộ quản lý môi trường siêu tốc `uv` (Khuyên dùng)
Nếu máy của bạn đã cài đặt `uv` (công cụ quản lý Python cực nhanh của Astral):
```bash
# Di chuyển vào thư mục dự án
cd path/to/Final

# Chạy ứng dụng trực tiếp, uv sẽ tự cài Python và thư viện
uv run streamlit run app.py
```

### Cách 2: Sử dụng `pip` truyền thống
1. **Chuẩn bị môi trường ảo Python (Virtual Environment):**
   ```bash
   python -m venv venv
   # Kích hoạt trên Windows:
   venv\Scripts\activate
   # Kích hoạt trên macOS/Linux:
   source venv/bin/activate
   ```
2. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Chạy ứng dụng:**
   ```bash
   streamlit run app.py
   ```
Sau khi khởi chạy thành công, trình duyệt sẽ tự động mở trang web ở địa chỉ `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Deploy Lên Streamlit Cloud (Miễn phí)

Streamlit Cloud cho phép bạn deploy và chia sẻ ứng dụng web của mình trực tiếp từ kho lưu trữ GitHub hoàn toàn miễn phí. Các bước thực hiện như sau:

### Bước 1: Đưa mã nguồn lên GitHub
1. Tạo một kho lưu trữ (Repository) mới trên tài khoản GitHub cá nhân của bạn (ví dụ đặt tên là `portfolio-optimization-hose`).
2. Khởi tạo Git tại thư mục dự án của bạn và đẩy (push) toàn bộ các file (`app.py`, `requirements.txt`, `README.md` và `HOSE_2020_2023.csv`) lên GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for portfolio optimization app"
   git branch -M main
   git remote add origin https://github.com/TÊN_USER_CỦA_BẠN/TÊN_REPO.git
   git push -u origin main
   ```

### Bước 2: Đăng nhập vào Streamlit Cloud
1. Truy cập vào trang web [share.streamlit.io](https://share.streamlit.io/).
2. Chọn **Connect with GitHub** để đăng nhập bằng tài khoản GitHub của bạn.

### Bước 3: Deploy ứng dụng
1. Sau khi đăng nhập thành công, nhấp vào nút **Create app** (hoặc **New app**).
2. Điền các thông tin cấu hình ứng dụng:
   - **Repository:** Chọn kho lưu trữ chứa mã nguồn vừa tải lên GitHub.
   - **Branch:** Chọn nhánh chạy chính (thường là `main`).
   - **Main file path:** Nhập tên file chạy chính: `app.py`.
3. Nhấp vào nút **Deploy!** ở góc dưới.

Streamlit Cloud sẽ tiến hành tạo máy chủ ảo, cài đặt tất cả các thư viện được khai báo trong tệp `requirements.txt` và khởi động ứng dụng web. Quá trình này thường mất khoảng 1-2 phút. Khi hoàn tất, bạn sẽ có một đường dẫn public dạng `https://your-app-name.streamlit.app` để chia sẻ cho mọi người cùng sử dụng!
